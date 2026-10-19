//! Staged-local selection verification. No authored Raster execution is changed.
use anyhow::{ensure, Context, Result};
use raster::BytesPage;
use raster_core::{
    cfs::{ControlFlowSchema, SequenceDef},
    input::{
        encode_list_metadata_payload, parse_bytes_page_payload, verify_bytes_page_geometry,
        verify_selection_proof, verify_selection_witness, ExternalEncoding, SelectionPayloadKind,
        SelectorPath, SelectorSegment, StorageRef,
    },
    trace::{ProgramStartEvent, TraceEvent},
};
use raster_runtime::{EntryArgumentsBinding, TraceRecorder};
use serde::{de::DeserializeOwned, Serialize};

/// Uses the public recorder solely as a selection-witness provider. It records
/// one real entry binding in an in-memory root context, never a stage trace or
/// a replacement for the authored CFS. It works even when RASTER_AUTH=0.
pub(crate) struct VerifiedSelections {
    reference: StorageRef,
    recorder: TraceRecorder,
}

impl VerifiedSelections {
    /// Consume the page directly from its proven payload. This avoids a second
    /// index-driven page deserialization and keeps a streaming source independent
    /// of whichever invocation currently occupies Raster's thread-local store.
    pub(crate) fn page(&self, path: SelectorPath) -> Result<BytesPage> {
        let witness = self.recorder.storage_selection_witness(
            &self.reference,
            &path,
            SelectionPayloadKind::Raw,
        )?;
        ensure!(
            !witness.is_reference_only()
                && witness.proof.path == path
                && witness.proof.root_hash.as_slice() == self.reference.commitment
                && verify_selection_proof(&witness.bytes, &witness.proof)
                && verify_bytes_page_geometry(&witness),
            "weight page failed commitment or geometry verification: {path:?}"
        );
        // Raster's canonical parser pins the Raw BytesPage payload kind and
        // validates its length. The consumed values come from these exact bytes,
        // rather than from separately trusted index fields or a typed decoder.
        let (index, offset, _, bytes) = parse_bytes_page_payload(&witness.bytes)
            .context("verified weight selection is not a raw bytes page")?;
        Ok(BytesPage::__from_parts(index, offset, bytes.to_vec()))
    }

    pub(crate) fn new(
        binding: EntryArgumentsBinding,
        input: Option<&str>,
        manifest: Option<&str>,
    ) -> Result<Self> {
        let mut cfs = ControlFlowSchema::new("staged-weight-selections");
        let mut main = SequenceDef::new("main");
        main.entry_arguments = binding.arguments.iter().map(|a| a.name.clone()).collect();
        cfs.sequences.push(main);
        let mut recorder = TraceRecorder::new(cfs);
        recorder.set_external_input(input, manifest)?;
        recorder.record(TraceEvent::ProgramStart(ProgramStartEvent {
            arguments: binding.arguments,
        }));
        Ok(Self {
            reference: binding.reference,
            recorder,
        })
    }

    pub(crate) fn select<T: DeserializeOwned + Serialize>(&self, path: SelectorPath) -> Result<T> {
        let selected = raster_runtime::select_stored_value::<T>(&self.reference, &path)
            .with_context(|| format!("select weight {path:?}"))?;
        let witness = self.recorder.storage_selection_witness(
            &self.reference,
            &path,
            SelectionPayloadKind::Raw,
        )?;
        ensure!(
            selected.selection.path == path
                && selected.selection.source_root_hash.as_slice() == self.reference.commitment
                && selected.selection.payload_kind == SelectionPayloadKind::Raw
                && verify_selection_witness(&selected.selection, &witness)
                && verify_bytes_page_geometry(&witness),
            "weight selection failed commitment or page geometry verification: {path:?}"
        );
        // A valid proof of raw bytes alone does not authenticate deserialization
        // through an index. Bind the actual typed value to that proven payload.
        let (typed_payload, _, _) = raster::encode_raster_value(&selected.value)?;
        ensure!(
            typed_payload == witness.bytes,
            "weight typed selection differs from verified payload: {path:?}"
        );
        Ok(selected.value)
    }
}

pub(crate) fn fields(names: &[&str]) -> SelectorPath {
    SelectorPath::new(
        names
            .iter()
            .map(|name| SelectorSegment::Field((*name).into()))
            .collect(),
    )
}

impl VerifiedSelections {
    /// A list header is a different authenticated payload kind from a raw list.
    /// Check both its proof and the actual fields used to control the sweep.
    pub(crate) fn list_len(&self, path: SelectorPath) -> Result<u64> {
        let metadata = raster_runtime::stored_list_metadata(&self.reference, &path)?;
        let witness = self.recorder.storage_selection_witness(
            &self.reference,
            &path,
            SelectionPayloadKind::List,
        )?;
        let selection = &metadata.selected.commitment;
        ensure!(
            selection.path == path
                && selection.source_root_hash.as_slice() == self.reference.commitment
                && selection.payload_kind == SelectionPayloadKind::List
                && verify_selection_witness(selection, &witness)
                && metadata.selected.bytes == witness.bytes
                && encode_list_metadata_payload(metadata.len, metadata.elements_root)
                    == witness.bytes,
            "weight list metadata failed commitment verification: {path:?}"
        );
        Ok(metadata.len)
    }
}

pub(crate) fn raster_inputs(binding: &EntryArgumentsBinding) -> bool {
    binding
        .arguments
        .iter()
        .all(|arg| arg.encoding == ExternalEncoding::Raster)
}

pub(crate) fn input_documents_from_args() -> (Option<String>, Option<String>) {
    let args: Vec<_> = std::env::args().collect();
    let arg = |name| {
        args.iter()
            .position(|s| s == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    (arg("--input"), arg("--input-manifest"))
}

pub(crate) const PAGE_SIZE: u64 = 196_608;

#[derive(Clone, Copy)]
pub(crate) struct PageRegion {
    pub byte_len: u64,
    pub page_size: u64,
    pub page_count: u64,
}

impl PageRegion {
    pub(crate) fn load(source: &VerifiedSelections, prefix: &[&str]) -> Result<Self> {
        let path = |field| {
            let mut names = prefix.to_vec();
            names.push(field);
            fields(&names)
        };
        let byte_len: u64 = source.select(path("byte_len"))?;
        let page_size: u64 = source.select(path("page_size"))?;
        let page_count = source.list_len(path("pages"))?;
        ensure!(page_size == PAGE_SIZE, "weight page size mismatch");
        ensure!(
            page_count == byte_len.div_ceil(page_size),
            "weight page count mismatch"
        );
        Ok(Self {
            byte_len,
            page_size,
            page_count,
        })
    }

    pub(crate) fn page(
        &self,
        source: &VerifiedSelections,
        prefix: &[&str],
        index: u64,
    ) -> Result<BytesPage> {
        ensure!(index < self.page_count, "weight page index out of range");
        let mut names = prefix.to_vec();
        names.push("pages");
        let mut path = fields(&names);
        path.segments.push(SelectorSegment::Index(index));
        let page = source.page(path)?;
        let offset = index
            .checked_mul(self.page_size)
            .context("weight page offset overflow")?;
        ensure!(
            page.index() == index
                && page.offset() == offset
                && page.len() as u64 == self.page_size.min(self.byte_len - offset),
            "weight page geometry mismatch"
        );
        Ok(page)
    }
}
