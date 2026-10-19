//! Multi-page fixtures exercise the production loaders as well as the eager oracle.
use crate::{
    artifact_io::with_stage_sequence_scope,
    cache::{materialization_key_from_stage_files, CachedInputs, MaterializationCache},
    embedding::{self, SelectedEmbedding, StaticEmbedding},
    routines,
    verified_weights::PAGE_SIZE,
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
    fn decode(&self) -> Result<decode_embed::input::ActivationSequence> {
        let inputs = routines::decode_embed::load_inputs_from_paths(
            &self.dir.join("input.json"),
            &self.dir.join("input_manifest.json"),
            &CachedInputs::new(),
        )?;
        routines::decode_embed::run_direct(&inputs)
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
fn table(pages: usize, width: u32) -> EmbeddingTable {
    let count = (pages - 1) * PAGE_SIZE as usize / 4 + width as usize * 3;
    let bytes = (0..count)
        .flat_map(|i| (((i % 193) as i32 - 96) * 19871).to_le_bytes())
        .collect::<Vec<_>>();
    EmbeddingTable {
        hidden_size: width,
        embedding_scale: -ONE / 2,
        values: Bytes::paged(bytes).unwrap(),
    }
}
fn prompt_fixture(pages: usize, width: u32) -> (Fixture, EmbeddingTable, PromptTokenization) {
    let f = Fixture::new();
    let table = table(pages, width);
    let per_page = PAGE_SIZE as u32 / (width * 4);
    let prompt = PromptTokenization {
        token_ids: List::from(vec![
            0,
            per_page - 1,
            per_page,
            per_page,
            (table.values.byte_len() / (u64::from(width) * 4) - 1) as u32,
        ]),
    };
    f.write("embedding", &table);
    f.write("prompt", &prompt);
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
fn edge(token_id: u32) -> decode_embed::input::DecodeEdge {
    decode_embed::input::DecodeEdge {
        has_selected: true,
        decode_position: 137,
        token_id,
        value: 0,
        generated_token_ids: List::from(vec![token_id]),
    }
}

#[test]
fn embedding_gathers_distinct_pages_and_caches_only_metadata() {
    let (f, table, prompt) = prompt_fixture(8, 4);
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
fn embedding_fixed_point_boundaries_empty_prompt_and_decode_position() {
    let (f, mut table, mut prompt) = prompt_fixture(4, 4);
    let boundary = [i32::MIN, i32::MAX, -1, 0, 1, -ONE, ONE];
    table.values = Bytes::paged(
        (0..table.values.byte_len() as usize / 4)
            .flat_map(|i| boundary[i % boundary.len()].to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    for scale in [i32::MIN, i32::MAX, 0, ONE, -ONE] {
        table.embedding_scale = scale;
        f.write("embedding", &table);
        assert_eq!(f.prompt(None).unwrap(), eager_prompt(&table, &prompt));
        let selected = edge(prompt.token_ids[4]);
        f.write("selected", &selected);
        let actual = f.decode().unwrap();
        assert_eq!(actual.start_position, 137);
        let eager_table = decode_embed::input::EmbeddingTable {
            hidden_size: table.hidden_size,
            embedding_scale: scale,
            values: table.values.clone(),
        };
        let expected = inference_kernels::kernels::decode_embed::run_decode_embed_direct(
            inference_kernels::kernels::decode_embed::DecodeEmbedDirectInputs {
                selected: &selected,
                embedding: &eager_table,
            },
        )
        .unwrap();
        assert_eq!(actual, expected);
    }
    prompt.token_ids = List::new();
    f.write("prompt", &prompt);
    assert!(f.prompt(None).unwrap().rows.is_empty());
    let mut selected = edge(0);
    selected.has_selected = false;
    f.write("selected", &selected);
    assert!(f.decode().is_err());
}

#[test]
fn embedding_rejects_missing_rows_cross_page_rows_and_bad_geometry() {
    let (f, mut table, mut prompt) = prompt_fixture(4, 5);
    prompt.token_ids = List::from(vec![(PAGE_SIZE / 20) as u32]);
    f.write("prompt", &prompt);
    assert!(f.prompt(None).is_err());
    prompt.token_ids = List::from(vec![u32::MAX]);
    f.write("prompt", &prompt);
    assert!(f.prompt(None).is_err());
    prompt.token_ids = List::from(vec![0]);
    f.write("prompt", &prompt);
    table.values.pages[0] =
        BytesPage::__from_parts(0, 4, table.values.pages[0].as_slice().to_vec());
    f.write("embedding", &table);
    assert!(f.prompt(None).is_err());
}

#[test]
fn embedding_rejects_corruption_and_separates_commitments_and_paths() {
    let (f, mut table, prompt) = prompt_fixture(4, 4);
    let cache = MaterializationCache::with_budget(Arc::new(WeightBudget::new(1 << 20)));
    let first = f
        .selected(prompt.token_ids.as_slice(), Some(&cache))
        .unwrap();
    table.embedding_scale += 1;
    f.write("embedding", &table);
    let second = f
        .selected(prompt.token_ids.as_slice(), Some(&cache))
        .unwrap();
    assert!(!Arc::ptr_eq(&first.metadata, &second.metadata));
    let (other, _, other_prompt) = prompt_fixture(4, 4);
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
    let mut other_table = table.clone();
    other_table.embedding_scale += 1;
    let (_, index, _) = raster::encode_raster_value(&other_table).unwrap();
    fs::write(f.dir.join("embedding.rindex"), index).unwrap();
    assert!(f.prompt(Some(&cache)).is_err());
    f.write("embedding", &table);
    f.update(
        "input_manifest.json",
        "embedding",
        serde_json::json!({"type":"sha256","encoding":"raster","commitment":"00".repeat(32)}),
    );
    assert!(f.prompt(Some(&cache)).is_err());
}

#[test]
fn embedding_postcard_and_mixed_inputs_keep_compatibility() {
    let (f, table, prompt) = prompt_fixture(4, 4);
    let expected = eager_prompt(&table, &prompt);
    f.postcard("prompt", &prompt);
    assert_eq!(f.prompt(None).unwrap(), expected);
    f.postcard("embedding", &table);
    assert_eq!(f.prompt(None).unwrap(), expected);
}

#[test]
#[ignore = "requires unchanged Raster stage binaries"]
fn embedding_multi_page_authenticated_raster_parity() {
    let (f, mut table, prompt) = prompt_fixture(4, 4);
    f.compare_raster("input-embedding", &f.prompt(None).unwrap(), "prompt-raster");
    let selected = edge(prompt.token_ids[4]);
    f.write("selected", &selected);
    f.compare_raster("decode-embed", &f.decode().unwrap(), "decode-raster");
    for scale in [i32::MIN, i32::MAX] {
        table.embedding_scale = scale;
        f.write("embedding", &table);
        f.compare_raster(
            "input-embedding",
            &f.prompt(None).unwrap(),
            &format!("prompt-scale-{scale}"),
        );
        f.compare_raster(
            "decode-embed",
            &f.decode().unwrap(),
            &format!("decode-scale-{scale}"),
        );
    }
    // Raster permits zero-width embedding rows; the eager prompt kernel rejects them.
    let mut zero = table;
    zero.hidden_size = 0;
    f.write("embedding", &zero);
    f.compare_raster(
        "input-embedding",
        &f.prompt(None).unwrap(),
        "zero-width-raster",
    );
}

#[test]
#[ignore = "release benchmark; informational timings"]
fn embedding_loading_benchmark() {
    let (f, table, prompt) = prompt_fixture(128, 4);
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
use prefill_finalize::input::{
    pack_i32s, ActivationRow, ActivationSequence, FinalHead, FinalHeadParams,
};

fn head_fixture(pages: usize) -> (Fixture, FinalHead, ActivationSequence) {
    let f = Fixture::new();
    let width = 256;
    let head = FinalHead {
        params: FinalHeadParams {
            hidden_size: width,
            norm_eps: 1,
            softcap: 0,
            norm_weights: pack_i32s(&vec![ONE; width as usize]),
        },
        projection: table(pages, width).values,
    };
    let activations = ActivationSequence {
        rows: List::from(vec![
            ActivationRow {
                token_id: 0,
                values: pack_i32s(&vec![ONE; width as usize]),
            },
            ActivationRow {
                token_id: 1,
                values: pack_i32s(
                    &(0..width)
                        .map(|i| (i as i32 - 128) * 731)
                        .collect::<Vec<_>>(),
                ),
            },
        ]),
        errors: List::new(),
        kv: List::new(),
        start_position: 29,
    };
    f.write("head", &head);
    f.write("activations", &activations);
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
    fn finalize(
        &self,
        cache: Option<&MaterializationCache>,
    ) -> Result<prefill_finalize::input::PrefillLogits> {
        let inputs = routines::prefill_finalize::load_inputs_from_paths_with_cache(
            &self.dir.join("input.json"),
            &self.dir.join("input_manifest.json"),
            &CachedInputs::new(),
            cache,
        )?;
        routines::prefill_finalize::run_direct(&inputs)
    }
}
fn eager_head(
    head: &FinalHead,
    activations: &ActivationSequence,
) -> prefill_finalize::input::PrefillLogits {
    inference_kernels::kernels::prefill_finalize::run_prefill_finalize_direct(
        inference_kernels::kernels::prefill_finalize::PrefillFinalizeDirectInputs {
            activations,
            head,
        },
    )
    .unwrap()
}

#[test]
fn head_batches_are_bounded_and_all_logits_match_eager() {
    for pages in [2, 33, 66] {
        let (f, head, activations) = head_fixture(pages);
        let (actual, stats) = head::run_measured(&activations, &f.head(None).unwrap()).unwrap();
        assert_eq!(actual, eager_head(&head, &activations));
        assert_eq!(actual.decode_position, 31);
        assert_eq!(stats.pages, pages as u64);
        assert_eq!(stats.peak_batch_pages, pages.min(BATCH_PAGES));
        assert!(stats.peak_batch_payload_bytes <= BATCH_PAGES * PAGE_SIZE as usize);
        assert_eq!(
            actual.logits.len() as u64,
            head.projection.byte_len() / (256 * 4)
        );
        for (index, logit) in actual.logits.iter().enumerate() {
            assert_eq!(logit.token_id, index as u32);
        }
    }
}

#[test]
fn head_saturation_normalization_softcap_and_metadata_cache() {
    let (f, mut head, mut activations) = head_fixture(4);
    // Long runs of extreme products force saturating i64 MACs, then change sign.
    head.params.norm_weights = pack_i32s(&vec![i32::MAX; 256]);
    head.projection = Bytes::paged(
        (0..head.projection.byte_len() as usize / 4)
            .flat_map(|i| (if i % 256 < 128 { i32::MAX } else { i32::MIN }).to_le_bytes())
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
                f.finalize(Some(&cache)).unwrap(),
                eager_head(&head, &activations)
            );
        }
        assert!(budget.stats().retained_bytes <= bytes);
    }
}

#[test]
fn head_empty_projection_and_invalid_activations_or_parameters() {
    let (f, mut head, mut activations) = head_fixture(2);
    head.projection = Bytes::paged(Vec::new()).unwrap();
    f.write("head", &head);
    assert_eq!(f.finalize(None).unwrap(), eager_head(&head, &activations));
    activations.rows = List::new();
    f.write("activations", &activations);
    assert!(f.finalize(None).is_err());
    activations.rows = List::from(vec![ActivationRow {
        token_id: 0,
        values: pack_i32s(&[ONE]),
    }]);
    f.write("activations", &activations);
    assert!(f.finalize(None).is_err());
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
            2 => pages[1] = BytesPage::__from_parts(1, PAGE_SIZE + 4, pages[1].as_slice().to_vec()),
            _ => pages[0] = BytesPage::__from_parts(0, 0, pages[0].as_slice()[..100].to_vec()),
        }
        bad.projection.pages = List::from(pages);
        f.write("head", &bad);
        assert!(f.finalize(None).is_err());
    }
    let mut split = head;
    split.params.hidden_size = 5;
    split.params.norm_weights = pack_i32s(&[ONE; 5]);
    split.projection = Bytes::paged(vec![0; PAGE_SIZE as usize + 12]).unwrap();
    let activations = ActivationSequence {
        rows: List::from(vec![ActivationRow {
            token_id: 0,
            values: pack_i32s(&[ONE; 5]),
        }]),
        errors: List::new(),
        kv: List::new(),
        start_position: 0,
    };
    f.write("head", &split);
    f.write("activations", &activations);
    // Eager concatenation accepts this shape, but Raster rejects the split rows.
    assert!(!eager_head(&split, &activations).logits.is_empty());
    assert!(f.finalize(None).is_err());
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
    assert!(f.finalize(Some(&cache)).is_err());
    f.write("head", &head);
    assert_eq!(
        f.finalize(Some(&cache)).unwrap(),
        eager_head(&head, &activations)
    );
    f.update(
        "input_manifest.json",
        "head",
        serde_json::json!({"type":"sha256","encoding":"raster","commitment":"00".repeat(32)}),
    );
    assert!(f.finalize(Some(&cache)).is_err());
}

#[test]
fn head_postcard_and_mixed_inputs_keep_compatibility() {
    let (f, head, activations) = head_fixture(4);
    let expected = eager_head(&head, &activations);
    f.postcard("activations", &activations);
    assert_eq!(f.finalize(None).unwrap(), expected);
    f.postcard("head", &head);
    assert_eq!(f.finalize(None).unwrap(), expected);
}

#[test]
fn paged_head_keeps_its_bound_source_when_another_invocation_starts() {
    let (first, head, activations) = head_fixture(4);
    let loaded = first.head(None).unwrap();
    let (other, mut other_head, _) = head_fixture(2);
    other_head.params.softcap = ONE;
    other.write("head", &other_head);
    let _other_loaded = other.head(None).unwrap();
    assert_eq!(
        head::run(&activations, &loaded).unwrap(),
        eager_head(&head, &activations)
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
                (0..head.projection.byte_len() as usize / 4)
                    .flat_map(|i| (if i % 256 < 128 { i32::MAX } else { i32::MIN }).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            activations.rows[1].values = pack_i32s(&vec![i32::MAX; 256]);
            f.write("head", &head);
            f.write("activations", &activations);
        }
        f.compare_raster(
            "prefill-finalize",
            &f.finalize(None).unwrap(),
            &format!("head-raster-{boundary}"),
        );
    }
    head.params.hidden_size = 5;
    head.params.norm_weights = pack_i32s(&[ONE; 5]);
    head.projection = Bytes::paged(vec![0; PAGE_SIZE as usize + 12]).unwrap();
    activations.rows = List::from(vec![ActivationRow {
        token_id: 0,
        values: pack_i32s(&[ONE; 5]),
    }]);
    f.write("head", &head);
    f.write("activations", &activations);
    assert!(f.finalize(None).is_err());
    let bin = PathBuf::from(std::env::var_os("STAGED_INFER_REPLAY_BIN_DIR").unwrap());
    let dest = f.dir.join("raster-rejects-split-rows");
    fs::create_dir(&dest).unwrap();
    let rejected = std::process::Command::new(
        bin.join(format!("prefill-finalize{}", std::env::consts::EXE_SUFFIX)),
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
    assert!(!rejected.status.success());
    assert!(!dest.join("output.bin").exists());
}

#[test]
#[ignore = "release benchmark; informational timings"]
fn head_loading_benchmark() {
    let (f, head, activations) = head_fixture(128);
    let mut eager = 0;
    let mut paged = 0;
    let mut verify = 0;
    let mut arithmetic = 0;
    for _ in 0..7 {
        let start = Instant::now();
        with_stage_sequence_scope(
            &f.dir.join("input.json"),
            &f.dir.join("input_manifest.json"),
            || {
                let binding = raster::start_program(&[
                    raster::entry_argument_spec::<ActivationSequence>("activations"),
                    raster::entry_argument_spec::<FinalHead>("head"),
                ])?;
                let weights = raster::materialize_auth_return(raster::entry_argument_auth_ref::<
                    FinalHead,
                >(
                    binding.reference, "head"
                ));
                std::hint::black_box(eager_head(&weights, &activations));
                Ok(())
            },
        )
        .unwrap();
        eager += start.elapsed().as_nanos();
        let start = Instant::now();
        let loaded = f.head(None).unwrap();
        let (output, stats) = head::run_measured(&activations, &loaded).unwrap();
        std::hint::black_box(output);
        paged += start.elapsed().as_nanos();
        verify += stats.load_verify.as_nanos();
        arithmetic += stats.arithmetic.as_nanos();
    }
    println!("head projection_bytes={} max_batch_pages={} eager_mean_ms={:.3} paged_mean_ms={:.3} page_verify_mean_ms={:.3} arithmetic_mean_ms={:.3}",
        head.projection.byte_len(),BATCH_PAGES,eager as f64/7e6,paged as f64/7e6,verify as f64/7e6,arithmetic as f64/7e6);
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
        assert!(f.finalize(None).is_err());
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
    let (result, dest) = f.standalone("input_embedding", "valid");
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
    let (result, dest) = f.standalone("input_embedding", "invalid");
    assert!(!result.status.success());
    for file in ["output.bin", "output.rindex", "output_manifest.json"] {
        assert!(!dest.join(file).exists());
    }
    let (f, _, _) = head_fixture(34);
    let expected = f.finalize(None).unwrap();
    let (result, dest) = f.standalone("prefill_finalize", "valid");
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
    let (result, dest) = f.standalone("decode_finalize_t7", "invalid");
    assert!(!result.status.success());
    for file in ["output.bin", "output.rindex", "output_manifest.json"] {
        assert!(!dest.join(file).exists());
    }
}
