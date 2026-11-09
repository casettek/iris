//! Multi-page fixtures exercise the production loaders as well as the eager oracle.
use crate::{
    artifact_io::with_stage_sequence_scope,
    cache::{materialization_key_from_stage_files, CachedInputs, MaterializationCache},
    embedding::{self, SelectedEmbedding, StaticEmbedding},
    routines,
    verified_weights::{Q8_HEAD_PAGE_SIZE as HEAD_PAGE_SIZE, Q8_PAGE_SIZE as PAGE_SIZE},
};
use anyhow::Result;
use inference_kernels::weight_budget::WeightBudget;
use input_embedding::input::{EmbeddingTable, PromptTokenization};
use raster::{Bytes, BytesPage, List};
use serde::Serialize;
use std::{
    fs,
    path::PathBuf,
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const ONE: i32 = 1 << 16;
struct Fixture {
    dir: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!(
            "iris-weight-pages-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("input.json"), b"{}").unwrap();
        fs::write(dir.join("input_manifest.json"), b"{}").unwrap();
        Self { dir }
    }
    fn release(&self) {
        raster_runtime::install_file_source_resolver(
            &self.dir.join("input.json"),
            &self.dir.join("input_manifest.json"),
        )
        .unwrap();
    }
    fn write<T: Serialize>(&self, name: &str, value: &T) {
        self.release();
        let (data, index, commitment) = raster::encode_raster_value(value).unwrap();
        fs::write(self.dir.join(format!("{name}.bin")), data).unwrap();
        fs::write(self.dir.join(format!("{name}.rindex")), index).unwrap();
        self.update(
            "input.json",
            name,
            serde_json::json!({"path":format!("{name}.bin"),
            "index_path":format!("{name}.rindex"),"load_preference":"mmap"}),
        );
        self.update(
            "input_manifest.json",
            name,
            serde_json::json!({"type":"sha256",
            "encoding":"raster","commitment":commitment}),
        );
    }
    fn postcard<T: Serialize + raster::Selectable>(&self, name: &str, value: &T) {
        self.release();
        fs::write(
            self.dir.join(format!("{name}.bin")),
            raster_core::postcard::to_allocvec(value).unwrap(),
        )
        .unwrap();
        self.update("input_manifest.json", name, serde_json::json!({"type":"sha256",
            "encoding":"postcard","commitment":raster_runtime::postcard_structural_commitment(value).unwrap()}));
    }
    fn update(&self, file: &str, name: &str, value: serde_json::Value) {
        let path = self.dir.join(file);
        let mut doc: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        doc[name] = value;
        fs::write(path, serde_json::to_vec(&doc).unwrap()).unwrap();
    }
    fn selected(
        &self,
        tokens: &[u32],
        cache: Option<&MaterializationCache>,
    ) -> Result<SelectedEmbedding> {
        let input = self.dir.join("input.json");
        let manifest = self.dir.join("input_manifest.json");
        let key = materialization_key_from_stage_files::<StaticEmbedding>(
            &input,
            &manifest,
            "embedding",
        )?;
        with_stage_sequence_scope(&input, &manifest, || {
            let binding = raster::start_program(&[
                raster::entry_argument_spec::<PromptTokenization>("prompt"),
                raster::entry_argument_spec::<EmbeddingTable>("embedding"),
            ])?;
            embedding::load(
                binding,
                input.to_str(),
                manifest.to_str(),
                tokens.iter().copied(),
                cache,
                key,
            )
        })
    }
    fn prompt(
        &self,
        cache: Option<&MaterializationCache>,
    ) -> Result<input_embedding::input::ActivationSequence> {
        let inputs = routines::input_embedding::load_inputs_from_paths_with_cache(
            &self.dir.join("input.json"),
            &self.dir.join("input_manifest.json"),
            &CachedInputs::new(),
            cache,
        )?;
        routines::input_embedding::run_direct(&inputs)
    }
    fn compare_raster<T: Serialize>(&self, binary: &str, output: &T, label: &str) {
        let bin = PathBuf::from(
            std::env::var_os("STAGED_INFER_REPLAY_BIN_DIR").expect("build Raster binaries first"),
        );
        let dest = self.dir.join(label);
        fs::create_dir_all(&dest).unwrap();
        let result = std::process::Command::new(
            bin.join(format!("{binary}{}", std::env::consts::EXE_SUFFIX)),
        )
        .arg("--input")
        .arg(self.dir.join("input.json"))
        .arg("--input-manifest")
        .arg(self.dir.join("input_manifest.json"))
        .env("RASTER_AUTH", "1")
        .env("RASTER_OUTPUT_DIR", &dest)
        .env("RASTER_TRACE_PATH", dest.join("trace.json"))
        .env("RASTER_TRACE_FORMAT", "json")
        .output()
        .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let (data, index, commitment) = raster::encode_raster_value(output).unwrap();
        assert_eq!(data, fs::read(dest.join("output.bin")).unwrap());
        assert_eq!(index, fs::read(dest.join("output.rindex")).unwrap());
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(dest.join("output_manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest["output"]["commitment"], commitment);
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.release();
        if !std::thread::panicking() {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }
}
/// Bytes of one `Q8_0` row `width` values wide.
fn row_bytes(width: u32) -> u64 {
    u64::from(width) / 32 * 34
}
/// Varied `Q8_0` rows filling `pages - 1` pages of `page_size` and three
/// more rows: finite scales across a wide range, signed values.
fn q8_rows(pages: usize, width: u32, page_size: u64) -> Vec<u8> {
    let rows = (pages as u64 - 1) * page_size / row_bytes(width) + 3;
    (0..rows * u64::from(width) / 32)
        .flat_map(|block| {
            let mut bytes = (0x2000 + (block * 31 % 977) as u16).to_le_bytes().to_vec();
            bytes.extend((0..32).map(|i| (((block * 32 + i) % 193) as i32 - 96) as i8 as u8));
            bytes
        })
        .collect()
}
fn table(pages: usize, width: u32) -> EmbeddingTable {
    let bytes = q8_rows(pages, width, PAGE_SIZE);
    EmbeddingTable {
        hidden_size: width,
        embedding_scale: -ONE / 2,
        values: Bytes::paged(bytes).unwrap(),
    }
}
fn prompt_fixture(pages: usize, width: u32) -> (Fixture, EmbeddingTable, PromptTokenization) {
    let f = Fixture::new();
    let table = table(pages, width);
    let per_page = (PAGE_SIZE / row_bytes(width)) as u32;
    let prompt = PromptTokenization {
        token_ids: List::from(vec![
            0,
            per_page - 1,
            per_page,
            per_page,
            (table.values.byte_len() / row_bytes(width) - 1) as u32,
        ]),
        ..Default::default()
    };
    f.write("embedding", &table);
    f.write(
        "witness",
        &model_identity::sparse::witness(table.values.pages.as_slice()).unwrap(),
    );
    f.write("prompt", &prompt);
    f.write("claimed", &input_embedding::input::ClaimedOutput::default());
    f.write(
        "previous",
        &input_embedding::input::ActivationSequence {
            model: Default::default(),
            rows: List::new(),
            errors: List::new(),
            kv: List::new(),
            start_position: 0,
        },
    );
    (f, table, prompt)
}
fn eager_prompt(
    table: &EmbeddingTable,
    prompt: &PromptTokenization,
) -> input_embedding::input::ActivationSequence {
    inference_kernels::kernels::input_embedding::run_input_embedding_direct(
        inference_kernels::kernels::input_embedding::InputEmbeddingDirectInputs {
            prompt,
            embedding: table,
        },
    )
    .unwrap()
}

#[test]
fn embedding_gathers_distinct_pages_and_caches_only_metadata() {
    let (f, table, prompt) = prompt_fixture(8, 32);
    let expected = eager_prompt(&table, &prompt);
    for bytes in [0, 32, 1 << 20] {
        let budget = Arc::new(WeightBudget::new(bytes));
        let cache = MaterializationCache::with_budget(budget.clone());
        let a = f
            .selected(prompt.token_ids.as_slice(), Some(&cache))
            .unwrap();
        let b = f
            .selected(prompt.token_ids.as_slice(), Some(&cache))
            .unwrap();
        assert_eq!(a.pages.len(), 3);
        assert_eq!(b.pages.len(), 3);
        assert_eq!(f.prompt(Some(&cache)).unwrap(), expected);
        assert!(budget.stats().retained_bytes <= bytes);
        if bytes > 10000 {
            assert!(Arc::ptr_eq(&a.metadata, &b.metadata));
            assert!(budget.stats().retained_bytes < 10000);
        }
    }
}

#[test]
fn embedding_fixed_point_boundaries_and_empty_prompt() {
    let (f, mut table, mut prompt) = prompt_fixture(4, 32);
    // Extreme half-precision scales (largest, smallest subnormal, zero, both
    // signs) against extreme bytes: dequantized rows saturate both ways.
    let scales = [0x7bff_u16, 0xfbff, 0x0001, 0x8001, 0x0000, 0x3c00];
    let values = [-128_i8, 127, -1, 0, 1];
    table.values = Bytes::paged(
        (0..table.values.byte_len() as usize / 34)
            .flat_map(|block| {
                let mut bytes = scales[block % scales.len()].to_le_bytes().to_vec();
                bytes.extend((0..32).map(|i| values[(block + i) % values.len()] as u8));
                bytes
            })
            .collect::<Vec<_>>(),
    )
    .unwrap();
    for scale in [i32::MIN, i32::MAX, 0, ONE, -ONE] {
        table.embedding_scale = scale;
        f.write("embedding", &table);
        f.write(
            "witness",
            &model_identity::sparse::witness(table.values.pages.as_slice()).unwrap(),
        );
        assert_eq!(f.prompt(None).unwrap(), eager_prompt(&table, &prompt));
    }
    prompt.token_ids = List::new();
    f.write("prompt", &prompt);
    assert!(f.prompt(None).unwrap().rows.is_empty());
}

#[test]
fn an_unknown_token_reads_the_first_row_and_keeps_its_id() {
    let (f, table, mut prompt) = prompt_fixture(4, 32);
    let past_the_end = (table.values.byte_len() / row_bytes(32)) as u32;
    prompt.token_ids = List::from(vec![0, past_the_end, u32::MAX]);
    f.write("prompt", &prompt);
    let expected = eager_prompt(&table, &prompt);
    assert_eq!(f.prompt(None).unwrap(), expected);
    assert_eq!(expected.rows[1].token_id, past_the_end);
    assert_eq!(expected.rows[2].token_id, u32::MAX);
    assert_eq!(expected.rows[1].values, expected.rows[0].values);
    assert_eq!(expected.rows[2].values, expected.rows[0].values);
}

#[test]
fn embedding_rejects_cross_page_rows_and_bad_geometry() {
    // 238-byte rows do not divide the page, so this row spans two.
    let (f, mut table, mut prompt) = prompt_fixture(4, 224);
    prompt.token_ids = List::from(vec![(PAGE_SIZE / row_bytes(224)) as u32]);
    f.write("prompt", &prompt);
    assert!(f.prompt(None).is_err());
    prompt.token_ids = List::from(vec![0]);
    f.write("prompt", &prompt);
    table.values.pages[0] =
        BytesPage::__from_parts(0, 4, table.values.pages[0].as_slice().to_vec());
    f.write("embedding", &table);
    f.write(
        "witness",
        &model_identity::sparse::witness(table.values.pages.as_slice()).unwrap(),
    );
    assert!(f.prompt(None).is_err());
}

#[test]
fn embedding_rejects_corruption_and_separates_commitments_and_paths() {
    let (f, mut table, prompt) = prompt_fixture(4, 32);
    let cache = MaterializationCache::with_budget(Arc::new(WeightBudget::new(1 << 20)));
    let first = f
        .selected(prompt.token_ids.as_slice(), Some(&cache))
        .unwrap();
    table.embedding_scale += 1;
    f.write("embedding", &table);
    f.write(
        "witness",
        &model_identity::sparse::witness(table.values.pages.as_slice()).unwrap(),
    );
    let second = f
        .selected(prompt.token_ids.as_slice(), Some(&cache))
        .unwrap();
    assert!(!Arc::ptr_eq(&first.metadata, &second.metadata));
    let (other, _, other_prompt) = prompt_fixture(4, 32);
    let other_weights = other
        .selected(other_prompt.token_ids.as_slice(), Some(&cache))
        .unwrap();
    assert!(!Arc::ptr_eq(&first.metadata, &other_weights.metadata));
    f.release();
    let path = f.dir.join("embedding.bin");
    let mut data = fs::read(&path).unwrap();
    *data.last_mut().unwrap() ^= 0x80;
    fs::write(&path, data).unwrap();
    assert!(f.prompt(Some(&cache)).is_err());
    f.write("embedding", &table);
    f.write(
        "witness",
        &model_identity::sparse::witness(table.values.pages.as_slice()).unwrap(),
    );
    let mut other_table = table.clone();
    other_table.embedding_scale += 1;
    let (_, index, _) = raster::encode_raster_value(&other_table).unwrap();
    fs::write(f.dir.join("embedding.rindex"), index).unwrap();
    assert!(f.prompt(Some(&cache)).is_err());
    f.write("embedding", &table);
    f.write(
        "witness",
        &model_identity::sparse::witness(table.values.pages.as_slice()).unwrap(),
    );
    f.update(
        "input_manifest.json",
        "embedding",
        serde_json::json!({"type":"sha256","encoding":"raster","commitment":"00".repeat(32)}),
    );
    assert!(f.prompt(Some(&cache)).is_err());
}

#[test]
fn embedding_postcard_and_mixed_inputs_keep_compatibility() {
    let (f, table, prompt) = prompt_fixture(4, 32);
    let expected = eager_prompt(&table, &prompt);
    f.postcard("prompt", &prompt);
    assert_eq!(f.prompt(None).unwrap(), expected);
    f.postcard("embedding", &table);
    assert_eq!(f.prompt(None).unwrap(), expected);
}

#[test]
#[ignore = "requires unchanged Raster stage binaries"]
fn embedding_multi_page_authenticated_raster_parity() {
    let (f, mut table, _) = prompt_fixture(4, 32);
    f.compare_raster("input-embedding", &f.prompt(None).unwrap(), "prompt-raster");
    for scale in [i32::MIN, i32::MAX] {
        table.embedding_scale = scale;
        f.write("embedding", &table);
        f.write(
            "witness",
            &model_identity::sparse::witness(table.values.pages.as_slice()).unwrap(),
        );
        f.compare_raster(
            "input-embedding",
            &f.prompt(None).unwrap(),
            &format!("prompt-scale-{scale}"),
        );
    }
    // Both execution paths reject rows without a whole Q8_0 block.
    let mut zero = table;
    zero.hidden_size = 0;
    f.write("embedding", &zero);
    f.write(
        "witness",
        &model_identity::sparse::witness(zero.values.pages.as_slice()).unwrap(),
    );
    assert!(f.prompt(None).is_err());
    let bin = PathBuf::from(std::env::var_os("STAGED_INFER_REPLAY_BIN_DIR").unwrap());
    let dest = f.dir.join("zero-width-raster");
    fs::create_dir(&dest).unwrap();
    let result = std::process::Command::new(bin.join("input-embedding"))
        .arg("--input")
        .arg(f.dir.join("input.json"))
        .arg("--input-manifest")
        .arg(f.dir.join("input_manifest.json"))
        .env("RASTER_AUTH", "1")
        .env("RASTER_OUTPUT_DIR", &dest)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(!dest.join("output.bin").exists());
}

#[test]
#[ignore = "release benchmark; informational timings"]
fn embedding_loading_benchmark() {
    let (f, table, prompt) = prompt_fixture(128, 32);
    let mut eager = 0;
    let mut selective = 0;
    for _ in 0..7 {
        let start = Instant::now();
        with_stage_sequence_scope(
            &f.dir.join("input.json"),
            &f.dir.join("input_manifest.json"),
            || {
                let binding = raster::start_program(&[
                    raster::entry_argument_spec::<PromptTokenization>("prompt"),
                    raster::entry_argument_spec::<EmbeddingTable>("embedding"),
                ])?;
                let weights = raster::materialize_auth_return(raster::entry_argument_auth_ref::<
                    EmbeddingTable,
                >(
                    binding.reference, "embedding"
                ));
                std::hint::black_box(eager_prompt(&weights, &prompt));
                Ok(())
            },
        )
        .unwrap();
        eager += start.elapsed().as_nanos();
        let start = Instant::now();
        std::hint::black_box(f.prompt(None).unwrap());
        selective += start.elapsed().as_nanos();
    }
    println!(
        "embedding table_bytes={} selected_pages=3 eager_mean_ms={:.3} selective_mean_ms={:.3}",
        table.values.byte_len(),
        eager as f64 / 7e6,
        selective as f64 / 7e6
    );
}

use crate::head::{self, PagedHead, StaticHead, BATCH_PAGES};
use sequence_check::input::{
    pack_i32s, ActivationRow, ActivationSequence, ClaimedOutput, DecodeEdge, FinalHead,
    FinalHeadParams,
};
use sequence_check::prompt::{PromptContext, PromptState};

const HEAD_WIDTH: u32 = 256;

/// The row before the claimed range, which predicts its first token.
fn head_previous(width: u32) -> ActivationSequence {
    ActivationSequence {
        model: Default::default(),
        rows: List::from(vec![ActivationRow {
            token_id: 0,
            values: pack_i32s(&vec![ONE; width as usize]),
        }]),
        errors: List::new(),
        kv: List::new(),
        start_position: 29,
    }
}
fn head_claimed() -> ClaimedOutput {
    ClaimedOutput {
        token_ids: List::from(vec![1, 0]),
    }
}
fn head_context() -> PromptContext {
    PromptContext {
        range_size: 4,
        prompt_tokens: 30,
        ..Default::default()
    }
}
fn head_fixture(pages: usize) -> (Fixture, FinalHead, ActivationSequence) {
    let f = Fixture::new();
    let width = HEAD_WIDTH;
    let head = FinalHead {
        params: FinalHeadParams {
            hidden_size: width,
            norm_eps: 1,
            softcap: 0,
            norm_weights: pack_i32s(&vec![ONE; width as usize]),
        },
        projection: Bytes::paged(q8_rows(pages, width, HEAD_PAGE_SIZE)).unwrap(),
    };
    let row = |token_id, values: &[i32]| ActivationRow {
        token_id,
        values: pack_i32s(values),
    };
    let activations = ActivationSequence {
        model: Default::default(),
        rows: List::from(vec![
            row(
                1,
                &(0..width)
                    .map(|i| (i as i32 - 128) * 731)
                    .collect::<Vec<_>>(),
            ),
            row(0, &vec![ONE; width as usize]),
        ]),
        errors: List::new(),
        kv: List::new(),
        start_position: 30,
    };
    f.write("head", &head);
    f.write("activations", &activations);
    f.write("previous", &head_previous(width));
    f.write("claimed", &head_claimed());
    f.write("prior", &DecodeEdge::default());
    f.write(
        "prompt",
        &PromptState {
            context: head_context(),
            ..Default::default()
        },
    );
    (f, head, activations)
}
impl Fixture {
    fn head(&self, cache: Option<&MaterializationCache>) -> Result<PagedHead> {
        let input = self.dir.join("input.json");
        let manifest = self.dir.join("input_manifest.json");
        let key = materialization_key_from_stage_files::<StaticHead>(&input, &manifest, "head")?;
        with_stage_sequence_scope(&input, &manifest, || {
            let binding = raster::start_program(&[
                raster::entry_argument_spec::<ActivationSequence>("activations"),
                raster::entry_argument_spec::<FinalHead>("head"),
            ])?;
            head::load(binding, input.to_str(), manifest.to_str(), cache, key)
        })
    }
    /// The whole stage over the paged head.
    fn check(&self, cache: Option<&MaterializationCache>) -> Result<DecodeEdge> {
        let inputs = routines::sequence_check::load_inputs_from_paths_with_cache(
            &self.dir.join("input.json"),
            &self.dir.join("input_manifest.json"),
            &CachedInputs::new(),
            cache,
        )?;
        routines::sequence_check::run_direct(&inputs)
    }
}
fn eager_check(head: &FinalHead, activations: &ActivationSequence) -> DecodeEdge {
    inference_kernels::kernels::sequence_check::run_sequence_check_direct(
        inference_kernels::kernels::sequence_check::SequenceCheckDirectInputs {
            activations,
            previous: &head_previous(head.params.hidden_size),
            prior: &DecodeEdge::default(),
            head,
            context: &head_context(),
        },
    )
    .unwrap()
}

#[test]
fn head_batches_are_bounded_and_the_check_matches_eager() {
    for pages in [2, 33, 66] {
        let (f, head, activations) = head_fixture(pages);
        let rows = activations
            .rows
            .iter()
            .map(|row| sequence_check::input::unpack_i32s(&row.values).unwrap())
            .collect::<Vec<_>>();
        let positions = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let (selections, stats) =
            head::select_positions_measured(&positions, &f.head(None).unwrap()).unwrap();
        assert_eq!(selections.len(), 2);
        assert_eq!(stats.pages, pages as u64);
        assert_eq!(stats.peak_batch_pages, pages.min(BATCH_PAGES));
        assert!(stats.peak_batch_payload_bytes <= BATCH_PAGES * HEAD_PAGE_SIZE as usize);
        let actual = f.check(None).unwrap();
        assert_eq!(actual, eager_check(&head, &activations));
        assert_eq!(actual.generated_token_ids.len(), 2);
        assert_eq!(actual.check_fault, 0);
    }
}

#[test]
fn head_saturation_normalization_softcap_and_metadata_cache() {
    let (f, mut head, mut activations) = head_fixture(4);
    // Extreme scales and bytes, positive then negative within each row,
    // drive the exact sums far past i32 before the one saturating rounding.
    head.params.norm_weights = pack_i32s(&vec![i32::MAX; 256]);
    head.projection = Bytes::paged(
        (0..head.projection.byte_len() as usize / 272)
            .flat_map(|_| {
                (0..8).flat_map(|block| {
                    let mut bytes = 0x7bff_u16.to_le_bytes().to_vec();
                    bytes.extend([if block < 4 { 127_i8 } else { -128 } as u8; 32]);
                    bytes
                })
            })
            .collect::<Vec<_>>(),
    )
    .unwrap();
    activations.rows[1].values = pack_i32s(&vec![i32::MAX; 256]);
    f.write("activations", &activations);
    for bytes in [0, 32, 1 << 20] {
        let budget = Arc::new(WeightBudget::new(bytes));
        let cache = MaterializationCache::with_budget(budget.clone());
        for softcap in [0, ONE, 30 * ONE] {
            head.params.softcap = softcap;
            f.write("head", &head);
            let a = f.head(Some(&cache)).unwrap();
            let b = f.head(Some(&cache)).unwrap();
            if bytes > 10000 {
                assert!(Arc::ptr_eq(&a.metadata, &b.metadata));
            }
            assert_eq!(
                f.check(Some(&cache)).unwrap(),
                eager_check(&head, &activations)
            );
        }
        assert!(budget.stats().retained_bytes <= bytes);
    }
}

#[test]
fn head_empty_projection_and_invalid_parameters_are_refused() {
    let (f, mut head, _) = head_fixture(2);
    head.projection = Bytes::paged(Vec::new()).unwrap();
    f.write("head", &head);
    assert!(f.check(None).is_err());
    for variant in 0..4 {
        let mut bad = head.clone();
        match variant {
            0 => bad.params.hidden_size = 0,
            1 => bad.params.norm_eps = -1,
            2 => bad.params.softcap = -1,
            _ => bad.params.norm_weights = pack_i32s(&[ONE]),
        };
        f.write("head", &bad);
        assert!(f.head(None).is_err());
    }
}

#[test]
fn a_row_of_the_wrong_width_is_published_as_a_fault() {
    let (f, head, mut activations) = head_fixture(2);
    activations.rows[0].values = pack_i32s(&[ONE]);
    f.write("activations", &activations);
    let edge = f.check(None).unwrap();
    assert_eq!(edge, eager_check(&head, &activations));
    assert_eq!(
        edge.check_fault,
        sequence_check::input::CHECK_FAULT_UNSCORED
    );
}

#[test]
fn head_rejects_missing_extra_misaligned_and_partial_rows() {
    let (f, head, _) = head_fixture(4);
    for variant in 0..4 {
        let mut bad = head.clone();
        let mut pages = bad.projection.pages.iter().cloned().collect::<Vec<_>>();
        match variant {
            0 => {
                pages.pop();
            }
            1 => pages.push(pages.last().unwrap().clone()),
            2 => {
                pages[1] =
                    BytesPage::__from_parts(1, HEAD_PAGE_SIZE + 4, pages[1].as_slice().to_vec())
            }
            _ => pages[0] = BytesPage::__from_parts(0, 0, pages[0].as_slice()[..100].to_vec()),
        }
        bad.projection.pages = List::from(pages);
        f.write("head", &bad);
        assert!(f.check(None).is_err());
    }
    // 238-byte rows: 220 of them end 136 bytes into the second page.
    let mut split = head;
    split.params.hidden_size = 224;
    split.params.norm_weights = pack_i32s(&[ONE; 224]);
    split.projection = Bytes::paged(vec![0; 220 * 238]).unwrap();
    let activations = ActivationSequence {
        model: Default::default(),
        rows: List::from(vec![
            ActivationRow {
                token_id: 1,
                values: pack_i32s(&[ONE; 224]),
            },
            ActivationRow {
                token_id: 0,
                values: pack_i32s(&[ONE; 224]),
            },
        ]),
        errors: List::new(),
        kv: List::new(),
        start_position: 30,
    };
    f.write("head", &split);
    f.write("activations", &activations);
    f.write("previous", &head_previous(224));
    // Raster rejects the rows that cross a page, and so does the paged path.
    assert!(f.check(None).is_err());
}

#[test]
fn head_last_page_corruption_and_cache_identity_are_checked() {
    let (f, mut head, activations) = head_fixture(34);
    let cache = MaterializationCache::with_budget(Arc::new(WeightBudget::new(1 << 20)));
    let a = f.head(Some(&cache)).unwrap().metadata;
    head.params.softcap = ONE;
    f.write("head", &head);
    let b = f.head(Some(&cache)).unwrap().metadata;
    assert!(!Arc::ptr_eq(&a, &b));
    f.release();
    let path = f.dir.join("head.bin");
    let mut data = fs::read(&path).unwrap();
    *data.last_mut().unwrap() ^= 0x80;
    fs::write(&path, data).unwrap();
    assert!(f.check(Some(&cache)).is_err());
    f.write("head", &head);
    assert_eq!(
        f.check(Some(&cache)).unwrap(),
        eager_check(&head, &activations)
    );
    f.update(
        "input_manifest.json",
        "head",
        serde_json::json!({"type":"sha256","encoding":"raster","commitment":"00".repeat(32)}),
    );
    assert!(f.check(Some(&cache)).is_err());
}

#[test]
fn head_postcard_and_mixed_inputs_keep_compatibility() {
    let (f, head, activations) = head_fixture(4);
    let expected = eager_check(&head, &activations);
    f.postcard("activations", &activations);
    assert_eq!(f.check(None).unwrap(), expected);
    f.postcard("head", &head);
    assert_eq!(f.check(None).unwrap(), expected);
}

#[test]
fn paged_head_keeps_its_bound_source_when_another_invocation_starts() {
    let (first, _, activations) = head_fixture(4);
    let loaded = first.head(None).unwrap();
    let row = sequence_check::input::unpack_i32s(&activations.rows[0].values).unwrap();
    let before = head::select_positions(&[row.as_slice()], &loaded).unwrap();
    let (other, mut other_head, _) = head_fixture(2);
    other_head.params.softcap = ONE;
    other.write("head", &other_head);
    let _other_loaded = other.head(None).unwrap();
    assert_eq!(
        head::select_positions(&[row.as_slice()], &loaded).unwrap(),
        before
    );
}

#[test]
#[ignore = "requires unchanged Raster stage binaries"]
fn head_multi_page_authenticated_raster_parity() {
    let (f, mut head, mut activations) = head_fixture(4);
    for boundary in [false, true] {
        if boundary {
            head.params.softcap = ONE;
            head.params.norm_weights = pack_i32s(&vec![i32::MAX; 256]);
            head.projection = Bytes::paged(
                (0..head.projection.byte_len() as usize / 272)
                    .flat_map(|_| {
                        (0..8).flat_map(|block| {
                            let mut bytes = 0x7bff_u16.to_le_bytes().to_vec();
                            bytes.extend([if block < 4 { 127_i8 } else { -128 } as u8; 32]);
                            bytes
                        })
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            activations.rows[1].values = pack_i32s(&vec![i32::MAX; 256]);
            f.write("head", &head);
            f.write("activations", &activations);
        }
        f.compare_raster(
            "sequence-check",
            &f.check(None).unwrap(),
            &format!("head-raster-{boundary}"),
        );
    }
    head.params.hidden_size = 224;
    head.params.norm_weights = pack_i32s(&[ONE; 224]);
    head.projection = Bytes::paged(vec![0; 220 * 238]).unwrap();
    f.write("head", &head);
    f.write("previous", &head_previous(224));
    activations.rows = List::from(vec![
        ActivationRow {
            token_id: 1,
            values: pack_i32s(&[ONE; 224]),
        },
        ActivationRow {
            token_id: 0,
            values: pack_i32s(&[ONE; 224]),
        },
    ]);
    f.write("activations", &activations);
    assert!(f.check(None).is_err());
    let bin = PathBuf::from(std::env::var_os("STAGED_INFER_REPLAY_BIN_DIR").unwrap());
    let dest = f.dir.join("raster-rejects-split-rows");
    fs::create_dir(&dest).unwrap();
    let rejected = std::process::Command::new(
        bin.join(format!("sequence-check{}", std::env::consts::EXE_SUFFIX)),
    )
    .arg("--input")
    .arg(f.dir.join("input.json"))
    .arg("--input-manifest")
    .arg(f.dir.join("input_manifest.json"))
    .env("RASTER_AUTH", "1")
    .env("RASTER_OUTPUT_DIR", &dest)
    .env("RASTER_TRACE_PATH", dest.join("trace.json"))
    .env("RASTER_TRACE_FORMAT", "json")
    .output()
    .unwrap();
    assert!(
        rejected.status.success(),
        "{}",
        String::from_utf8_lossy(&rejected.stderr)
    );
    // Authored head execution publishes a semantic error instead of aborting.
    let value = raster_runtime::read_raster_value(
        &dest.join("output.bin"),
        &dest.join("output.rindex"),
        &raster_runtime::ReadLimits::default(),
    )
    .unwrap();
    let raster_runtime::RasterValue::Struct { fields, .. } = value else {
        panic!("expected decode edge");
    };
    assert!(fields.iter().any(|(name, value)| name == "check_fault"
        && matches!(value, raster_runtime::RasterValue::Int { value, .. } if *value != 0)));
}

#[test]
#[ignore = "release benchmark; informational timings"]
fn head_loading_benchmark() {
    let (f, head, activations) = head_fixture(128);
    let mut eager = 0;
    let mut paged = 0;
    for _ in 0..7 {
        let start = Instant::now();
        std::hint::black_box(eager_check(&head, &activations));
        eager += start.elapsed().as_nanos();
        let start = Instant::now();
        std::hint::black_box(f.check(None).unwrap());
        paged += start.elapsed().as_nanos();
    }
    println!(
        "head projection_bytes={} max_batch_pages={BATCH_PAGES} eager_mean_ms={:.3} paged_mean_ms={:.3}",
        head.projection.byte_len(),
        eager as f64 / 7e6,
        paged as f64 / 7e6
    );
}

// Test-only wire mirror for adversarial rindex04 edits. Production never parses
// index internals and always authenticates the list metadata through public APIs.
mod index_fixture {
    use serde::{Deserialize, Serialize};
    #[derive(Serialize, Deserialize)]
    pub struct Index {
        version: u32,
        root_node: u64,
        root_commitment: [u8; 32],
        pub nodes: Vec<Node>,
    }
    #[derive(Serialize, Deserialize)]
    pub struct Node {
        offset: u64,
        len: u64,
        root_hash: [u8; 32],
        pub kind: Kind,
    }
    #[derive(Serialize, Deserialize)]
    pub enum Kind {
        Unit,
        Leaf {
            type_name: String,
        },
        Struct {
            fields: Vec<Field>,
        },
        List {
            len: u64,
            elements: Vec<u64>,
            merkle_levels: Vec<Level>,
        },
    }
    #[derive(Serialize, Deserialize)]
    pub struct Field {
        name: String,
        child: u64,
    }
    #[derive(Serialize, Deserialize)]
    pub struct Level {
        hashes: Vec<[u8; 32]>,
    }
}

#[test]
fn forged_index_list_lengths_cannot_skip_committed_pages() {
    let (f, head, _) = head_fixture(4);
    for forged in [0, 1, 5] {
        f.write("head", &head);
        let path = f.dir.join("head.rindex");
        let raw = fs::read(&path).unwrap();
        assert_eq!(&raw[..8], b"rindex04");
        let mut index: index_fixture::Index = raster_core::postcard::from_bytes(&raw[8..]).unwrap();
        let mut found = false;
        for node in &mut index.nodes {
            if let index_fixture::Kind::List { len, elements, .. } = &mut node.kind {
                if elements.len() == 4 {
                    *len = forged;
                    found = true;
                }
            }
        }
        assert!(found);
        let mut bytes = b"rindex04".to_vec();
        bytes.extend(raster_core::postcard::to_allocvec(&index).unwrap());
        fs::write(path, bytes).unwrap();
        assert!(f.check(None).is_err());
    }
}

impl Fixture {
    fn standalone(&self, stage: &str, label: &str) -> (std::process::Output, PathBuf) {
        let bin = PathBuf::from(
            std::env::var_os("STAGED_INFER_REPLAY_BIN_DIR").expect("build staged binary first"),
        );
        let dest = self
            .dir
            .join("target/staged-infer/runs")
            .join(label)
            .join(stage);
        fs::create_dir_all(&dest).unwrap();
        let mut inputs: serde_json::Value =
            serde_json::from_slice(&fs::read(self.dir.join("input.json")).unwrap()).unwrap();
        for entry in inputs.as_object_mut().unwrap().values_mut() {
            for key in ["path", "index_path"] {
                if let Some(relative) = entry.get(key).and_then(serde_json::Value::as_str) {
                    entry[key] = serde_json::json!(self.dir.join(relative));
                }
            }
        }
        fs::write(
            dest.join("input.json"),
            serde_json::to_vec(&inputs).unwrap(),
        )
        .unwrap();
        fs::copy(
            self.dir.join("input_manifest.json"),
            dest.join("input_manifest.json"),
        )
        .unwrap();
        let output = std::process::Command::new(
            bin.join(format!("staged-infer{}", std::env::consts::EXE_SUFFIX)),
        )
        .current_dir(&self.dir)
        .arg("--run-stage")
        .arg(&dest)
        .arg("--input")
        .arg(dest.join("input.json"))
        .arg("--input-manifest")
        .arg(dest.join("input_manifest.json"))
        .env("RASTER_AUTH", "0")
        .env("RASTER_OUTPUT_DIR", &dest)
        .output()
        .unwrap();
        (output, dest)
    }
}

#[test]
#[ignore = "requires built staged-infer binary"]
fn standalone_publishes_exact_outputs_and_rejects_corruption_before_publication() {
    let (f, _, _) = prompt_fixture(4, 4);
    let expected = f.prompt(None).unwrap();
    let (result, dest) = f.standalone("embed_p0", "valid");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let (data, index, _) = raster::encode_raster_value(&expected).unwrap();
    assert_eq!(data, fs::read(dest.join("output.bin")).unwrap());
    assert_eq!(index, fs::read(dest.join("output.rindex")).unwrap());
    f.release();
    let path = f.dir.join("embedding.bin");
    let mut bytes = fs::read(&path).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    fs::write(path, bytes).unwrap();
    let (result, dest) = f.standalone("embed_p0", "invalid");
    assert!(!result.status.success());
    for file in ["output.bin", "output.rindex", "output_manifest.json"] {
        assert!(!dest.join(file).exists());
    }
    let (f, _, _) = head_fixture(34);
    let expected = f.check(None).unwrap();
    let (result, dest) = f.standalone("check_c0", "valid");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let (data, index, _) = raster::encode_raster_value(&expected).unwrap();
    assert_eq!(data, fs::read(dest.join("output.bin")).unwrap());
    assert_eq!(index, fs::read(dest.join("output.rindex")).unwrap());
    f.release();
    let path = f.dir.join("head.bin");
    let mut bytes = fs::read(&path).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    fs::write(path, bytes).unwrap();
    let (result, dest) = f.standalone("check_c7", "invalid");
    assert!(!result.status.success());
    for file in ["output.bin", "output.rindex", "output_manifest.json"] {
        assert!(!dest.join(file).exists());
    }
}

#[test]
fn sparse_embedding_checks_current_proofs_with_warm_cache() {
    let (f, table, mut prompt) = prompt_fixture(4, 32);
    let root = raster::core::tree::value_root(&table).unwrap();
    let descriptor = model_identity::ModelDescriptor {
        version: 1,
        tokenizer: [1; 32],
        decoder: [2; 32],
        embedding: root,
        head: [3; 32],
        auxiliary: vec![[4; 32]],
        transformer: vec![[5; 32]],
    };
    let id = descriptor
        .model_id()
        .unwrap()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    prompt.context.model = model_identity::ModelContext::bind(&descriptor.encode().unwrap(), &id);
    prompt.context.model_id = id;
    prompt.token_ids = vec![0].into();
    prompt
        .context
        .model
        .bind_statement(&"11".repeat(32), false, 8, 4);
    prompt
        .context
        .model
        .bind_schedule(model_identity::pipeline_core::StageSchedule {
            prompt_ranges: 1,
            merge_batches: 1,
            claimed_ranges: 1,
        });
    prompt.context.model.begin_prompt_range();
    prompt.context.model.continue_prompt_range();
    prompt.context.model.append_consumed_token(0);
    prompt.context.model.finish_prompt_range(true, 1, 0);
    f.write("prompt", &prompt);
    let witness = model_identity::sparse::witness(table.values.pages.as_slice()).unwrap();
    let cache = MaterializationCache::with_budget(Arc::new(WeightBudget::new(1 << 20)));
    let run = || f.prompt(Some(&cache)).unwrap();
    let check = |label: &str, clean: bool| {
        let output = run();
        assert_eq!(output.model.fault == 0, clean, "{label}");
        if std::env::var_os("STAGED_INFER_REPLAY_BIN_DIR").is_some() {
            f.compare_raster("input-embedding", &output, label);
        }
    };
    check("sparse-honest", true);
    let mut invalid = witness.clone();
    let mut paths = invalid.paths.as_slice().to_vec();
    paths[0] = paths[1].clone();
    invalid.paths = paths.into();
    f.write("witness", &invalid);
    check("sparse-warm-invalid-proof", false);
    assert_ne!(
        run().model.fault,
        0,
        "cached metadata must not cache the old witness verdict"
    );
    f.write("witness", &witness);
    check("sparse-restored", true);

    let mutate = |index: usize| {
        let mut changed = table.clone();
        let mut pages = changed.values.pages.as_slice().to_vec();
        let page = &pages[index];
        let mut bytes = page.as_slice().to_vec();
        bytes[2] ^= 1;
        pages[index] = BytesPage::__from_parts(page.index(), page.offset(), bytes);
        changed.values.pages = pages.into();
        changed
    };
    f.write("embedding", &mutate(2));
    check("sparse-unread-page", true);
    assert_eq!(
        run().model.fault,
        0,
        "unchanged consumed page authenticates without reading other pages"
    );
    f.write("embedding", &mutate(0));
    check("sparse-consumed-page", false);
    assert_ne!(
        run().model.fault,
        0,
        "consumed bytes must match the original root"
    );
    let replacement = model_identity::sparse::witness(mutate(0).values.pages.as_slice()).unwrap();
    f.write("witness", &replacement);
    check("sparse-replaced-witness", false);
    assert_ne!(
        run().model.fault,
        0,
        "new self-consistent witness cannot replace the statement root"
    );
}

#[test]
fn sparse_merge_fault_survives_later_matches_and_cached_tokenizer() {
    use prompt_prepare::input::*;
    let f = Fixture::new();
    let mut buckets = vec![MergeBucket::default(); 16];
    let index = merge_bucket_of("a", "b", 16) as usize;
    buckets[index].rules = vec![BpeMerge {
        left: "a".into(),
        right: "b".into(),
        merged: "ab".into(),
        rank: 0,
    }]
    .into();
    let tokenizer = PromptTokenizer {
        vocab_bucket_count: 1,
        merge_bucket_count: 16,
        boundary_bucket_count: 1,
        vocab_buckets: vec![VocabBucket::default()].into(),
        merge_buckets: buckets.into(),
        boundary_buckets: vec![BoundaryBucket::default()].into(),
        ..Default::default()
    };
    let witness = tokenizer_witness(&tokenizer).unwrap();
    let root = raster::core::tree::value_root(&tokenizer).unwrap();
    let descriptor = model_identity::ModelDescriptor {
        version: 1,
        tokenizer: root,
        decoder: [1; 32],
        embedding: [2; 32],
        head: [3; 32],
        auxiliary: vec![[4; 32]],
        transformer: vec![[5; 32]],
    };
    let id = descriptor
        .model_id()
        .unwrap()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    let initial = PromptState {
        pieces: vec![
            BpePiece {
                text: "a".into(),
                segment: 0,
            },
            BpePiece {
                text: "b".into(),
                segment: 0,
            },
        ]
        .into(),
        context: PromptContext {
            // A seeded range: the state a merge batch follows.
            model: {
                let mut model =
                    model_identity::ModelContext::bind(&descriptor.encode().unwrap(), &id);
                model.bind_statement(&"11".repeat(32), false, 8, 4);
                model.bind_schedule(model_identity::pipeline_core::StageSchedule {
                    prompt_ranges: 1,
                    merge_batches: 1,
                    claimed_ranges: 1,
                });
                model.begin_prompt_range();
                model
            },
            model_id: id,
            ..Default::default()
        },
        ..Default::default()
    };
    f.write("tokenizer", &tokenizer);
    f.write("witness", &witness);
    f.write("initial_pieces", &initial);
    let cache = MaterializationCache::with_budget(Arc::new(WeightBudget::new(1 << 20)));
    let run = || {
        let inputs = routines::prompt_merge::load_inputs_from_paths_with_cache(
            &f.dir.join("input.json"),
            &f.dir.join("input_manifest.json"),
            &CachedInputs::new(),
            Some(&cache),
        )
        .unwrap();
        routines::prompt_merge::run_direct(&inputs).unwrap()
    };
    assert_eq!(run().context.model.fault, 0);
    let mut bad = witness.clone();
    let mut paths = bad.merge_paths.as_slice().to_vec();
    paths[merge_bucket_of("", "a", 16) as usize].siblings =
        BytesPage::__from_parts(0, 0, vec![0; 31]);
    bad.merge_paths = paths.into();
    f.write("witness", &bad);
    let output = run();
    assert_eq!(
        output.pieces.as_slice(),
        &[BpePiece {
            text: "ab".into(),
            segment: 0
        }]
    );
    assert_ne!(output.context.model.fault, 0);
    if std::env::var_os("STAGED_INFER_REPLAY_BIN_DIR").is_some() {
        f.compare_raster("prompt-merge", &output, "merge-sticky-fault");
    }
    f.write("witness", &witness);
    assert_eq!(run().context.model.fault, 0);
}
