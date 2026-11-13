use super::*;
use crate::{artifact_io::with_stage_sequence_scope, cache::materialization_key_from_stage_files};
use inference_kernels::{
    kernels::prefill_prepare_aux::{run_prefill_prepare_aux_direct, PrefillPrepareAuxDirectInputs},
    weight_budget::WeightBudget,
};
use prefill_prepare_aux::input::{ActivationRow, PleLayer};
use serde::Serialize;
use std::{
    fs,
    path::PathBuf,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const ONE: i32 = 1 << 16;

/// Bytes of one `Q8_0` PLE row `width` values wide.
fn row_bytes(width: usize) -> usize {
    width / 32 * 34
}

struct Fixture {
    dir: PathBuf,
    layer: PleLayer,
    embedded: ActivationSequence,
}

impl Fixture {
    fn new(pages: usize, width: usize) -> Self {
        let rows = (pages - 1) * Q8_PAGE_SIZE as usize / row_bytes(width) + 3;
        // Varied finite scales and signed bytes, one `Q8_0` block at a time.
        let embeddings: Vec<u8> = (0..rows * width / 32)
            .flat_map(|block| {
                let mut bytes = (0x2400 + (block * 37 % 701) as u16).to_le_bytes().to_vec();
                bytes.extend((0..32).map(|i| (((block * 32 + i) % 97) as i32 - 48) as i8 as u8));
                bytes
            })
            .collect();
        let layer = PleLayer {
            params: PleLayerParams {
                layer_idx: 3,
                hidden_size: 2,
                ple_width: width as u32,
                embedding_scale: ONE / 2,
                projection_scalar: -ONE / 2,
                input_scale: ONE,
                norm_eps: 1,
                norm_weights: pack_i32s(&vec![ONE; width]),
            },
            embeddings: Bytes::paged(embeddings).unwrap(),
            projection: Bytes::paged(
                (0..width * 2)
                    .flat_map(|i| ((i as i32 - 3) * 17001).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        };
        let rows_per_page = (Q8_PAGE_SIZE as usize / row_bytes(width)) as u32;
        let tokens = [0, rows_per_page, rows_per_page, (rows - 1) as u32];
        let embedded = ActivationSequence {
            model: Default::default(),
            rows: List::from(
                tokens
                    .iter()
                    .map(|&token_id| ActivationRow {
                        token_id,
                        values: pack_i32s(&[ONE, -ONE / 3]),
                    })
                    .collect::<Vec<_>>(),
            ),
            errors: List::new(),
            kv: List::new(),
            start_position: 0,
        };
        let dir = std::env::temp_dir().join(format!(
            "iris-selective-ple-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let fixture = Self {
            dir,
            layer,
            embedded,
        };
        fixture.write(false);
        fixture
    }

    fn write(&self, postcard: bool) {
        // End the previous invocation before replacing test artifacts. Windows
        // cannot truncate a file while Raster's runtime still owns its mmap.
        self.release_mappings();
        let mut input = serde_json::Map::new();
        let mut manifest = serde_json::Map::new();
        for (name, value) in [
            (
                "witness",
                raster::encode_raster_value(
                    &model_identity::sparse::witness(self.layer.embeddings.pages.as_slice())
                        .unwrap(),
                )
                .unwrap(),
            ),
            ("layer", raster::encode_raster_value(&self.layer).unwrap()),
            (
                "embedded",
                raster::encode_raster_value(&self.embedded).unwrap(),
            ),
        ] {
            let (data, index, commitment) = value;
            fs::write(self.dir.join(format!("{name}.bin")), data).unwrap();
            fs::write(self.dir.join(format!("{name}.rindex")), index).unwrap();
            input.insert(name.into(), serde_json::json!({"path":format!("{name}.bin"),"index_path":format!("{name}.rindex"),"load_preference":"mmap"}));
            manifest.insert(
                name.into(),
                serde_json::json!({"type":"sha256","encoding":"raster","commitment":commitment}),
            );
        }
        if postcard {
            let bytes = raster_core::postcard::to_allocvec(&self.layer).unwrap();
            fs::write(self.dir.join("layer.bin"), bytes).unwrap();
            manifest["layer"] = serde_json::json!({"type":"sha256","encoding":"postcard", "commitment":raster_runtime::postcard_structural_commitment(&self.layer).unwrap()});
        }
        fs::write(
            self.dir.join("input.json"),
            serde_json::to_vec(&input).unwrap(),
        )
        .unwrap();
        fs::write(
            self.dir.join("input_manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
    }

    fn load(&self, cache: Option<&MaterializationCache>) -> Result<SelectedWeights> {
        let input = self.dir.join("input.json");
        let manifest = self.dir.join("input_manifest.json");
        let key =
            materialization_key_from_stage_files::<StaticWeights>(&input, &manifest, "layer")?;
        with_stage_sequence_scope(&input, &manifest, || {
            let binding = raster::start_program(&[
                raster::entry_argument_spec::<ActivationSequence>("embedded"),
                raster::entry_argument_spec::<PleLayer>("layer"),
            ])?;
            load(
                binding,
                input.to_str(),
                manifest.to_str(),
                &self.embedded,
                cache,
                key,
            )
        })
    }

    fn assert_parity(&self, selected: &SelectedWeights) {
        let eager = run_prefill_prepare_aux_direct(PrefillPrepareAuxDirectInputs {
            embedded: &self.embedded,
            layer: &self.layer,
        })
        .unwrap();
        assert_eq!(run(&self.embedded, selected).unwrap(), eager);
    }

    fn release_mappings(&self) {
        let input = self.dir.join("input.json");
        let manifest = self.dir.join("input_manifest.json");
        if input.exists() && manifest.exists() {
            // A fresh resolver parses the documents without opening the model.
            raster_runtime::install_file_source_resolver(&input, &manifest).unwrap();
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.release_mappings();
        if !std::thread::panicking() {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }
}

#[test]
fn reads_only_distinct_requested_pages_and_reuses_verified_static_weights() {
    let f = Fixture::new(4, 32);
    let budget = Arc::new(WeightBudget::new(1 << 20));
    let cache = MaterializationCache::with_budget(budget.clone());
    let first = f.load(Some(&cache)).unwrap();
    assert_eq!(first.pages.keys().copied().collect::<Vec<_>>(), [0, 1, 3]);
    assert_eq!(
        first.pages.values().map(BytesPage::len).sum::<usize>(),
        2 * Q8_PAGE_SIZE as usize + 3 * row_bytes(32)
    );
    f.assert_parity(&first);
    let second = f.load(Some(&cache)).unwrap();
    assert!(Arc::ptr_eq(&first.weights, &second.weights));
    assert_eq!(budget.stats().hits, 1);
    assert!(budget.stats().retained_bytes < PAGE_SIZE as usize);
    f.assert_parity(&second);
}

#[test]
fn cache_budget_bypass_and_changed_commitments_preserve_values() {
    let mut f = Fixture::new(4, 32);
    for bytes in [0, 1, 1 << 20] {
        let budget = Arc::new(WeightBudget::new(bytes));
        let cache = MaterializationCache::with_budget(budget.clone());
        let old = f.load(Some(&cache)).unwrap();
        f.layer.params.input_scale /= 2;
        f.write(false);
        let new = f.load(Some(&cache)).unwrap();
        assert!(!Arc::ptr_eq(&old.weights, &new.weights));
        f.assert_parity(&new);
        assert!(budget.stats().retained_bytes <= bytes);
    }
}

#[test]
fn saturation_and_negative_fixed_point_values_match_eager() {
    let mut f = Fixture::new(4, 32);
    f.layer.params.embedding_scale = i32::MAX;
    f.layer.params.input_scale = i32::MIN;
    f.layer.params.projection_scalar = i32::MIN;
    f.embedded.rows = List::from(vec![ActivationRow {
        token_id: 0,
        values: pack_i32s(&[i32::MAX, i32::MIN]),
    }]);
    f.layer.projection = Bytes::paged(
        [i32::MAX, i32::MIN, -1, 0, 1, -ONE, ONE, i32::MAX]
            .iter()
            .cycle()
            .take(f.layer.projection.byte_len() as usize / 4)
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    f.write(false);
    f.assert_parity(&f.load(None).unwrap());
}

#[test]
fn an_unknown_token_reads_the_first_row_and_matches_eager() {
    let mut f = Fixture::new(4, 32);
    let row = |token_id| ActivationRow {
        token_id,
        values: pack_i32s(&[ONE, ONE]),
    };
    let past_the_end = (f.layer.embeddings.byte_len() / row_bytes(32) as u64) as u32;
    f.embedded.rows = List::from(vec![row(0), row(past_the_end), row(u32::MAX)]);
    f.write(false);
    let selected = f.load(None).unwrap();
    f.assert_parity(&selected);
    let output = run(&f.embedded, &selected).unwrap();
    assert_eq!(output.rows[1], output.rows[0]);
    assert_eq!(output.rows[2], output.rows[0]);
}

#[test]
fn cross_page_rows_and_projection_geometry_fail() {
    // 238-byte rows do not divide a page: this row starts in page 0 and ends in 1.
    let mut f = Fixture::new(4, 224);
    f.embedded.rows = List::from(vec![ActivationRow {
        token_id: (Q8_PAGE_SIZE as usize / row_bytes(224)) as u32,
        values: pack_i32s(&[ONE, ONE]),
    }]);
    f.write(false);
    assert!(f.load(None).is_err());
    assert!(
        run_prefill_prepare_aux_direct(PrefillPrepareAuxDirectInputs {
            embedded: &f.embedded,
            layer: &f.layer
        })
        .is_err()
    );
    let mut f = Fixture::new(4, 32);
    let page = &f.layer.projection.pages()[0];
    f.layer.projection.pages = List::from(vec![BytesPage::__from_parts(
        0,
        4,
        page.as_slice().to_vec(),
    )]);
    f.write(false);
    assert!(f.load(None).is_err());
}

#[test]
fn committed_malformed_embedding_geometry_is_rejected() {
    let mut f = Fixture::new(4, 32);
    f.load(None).unwrap();
    let mut pages = f
        .layer
        .embeddings
        .pages()
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    pages[0] = BytesPage::__from_parts(0, 4, pages[0].as_slice().to_vec());
    f.layer.embeddings.pages = List::from(pages);
    f.write(false);
    assert!(f.load(None).is_err());
}

#[test]
fn corrupt_selected_bytes_and_index_proofs_never_enter_cache() {
    let f = Fixture::new(4, 32);
    let budget = Arc::new(WeightBudget::new(1 << 20));
    let cache = MaterializationCache::with_budget(budget.clone());
    let mut data = fs::read(f.dir.join("layer.bin")).unwrap();
    *data.last_mut().unwrap() ^= 0x40;
    fs::write(f.dir.join("layer.bin"), data).unwrap();
    assert!(f.load(Some(&cache)).is_err());
    assert_eq!(budget.stats().retained_bytes, 0);
    f.write(false);
    // Corrupt a selected page without changing the index or declared root.
    let mut data = fs::read(f.dir.join("layer.bin")).unwrap();
    let needle = f.layer.embeddings.pages()[0].as_slice();
    let start = data
        .windows(needle.len())
        .position(|w| w == needle)
        .unwrap();
    data[start + 10] ^= 0x80;
    fs::write(f.dir.join("layer.bin"), data).unwrap();
    assert!(f.load(Some(&cache)).is_err());
    f.write(false);
    f.assert_parity(&f.load(Some(&cache)).unwrap());
    // Replace the index with a well-formed index for a different parameter root.
    let mut other = f.layer.clone();
    other.params.input_scale += 1;
    let (_, index, _) = raster::encode_raster_value(&other).unwrap();
    f.release_mappings();
    fs::write(f.dir.join("layer.rindex"), index).unwrap();
    assert!(f.load(None).is_err());
}

#[test]
fn changed_declared_commitment_cannot_hit_a_verified_cache_entry() {
    let f = Fixture::new(4, 32);
    let cache = MaterializationCache::with_budget(Arc::new(WeightBudget::new(1 << 20)));
    f.load(Some(&cache)).unwrap();
    let path = f.dir.join("input_manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    manifest["layer"]["commitment"] = serde_json::Value::String("00".repeat(32));
    fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(f.load(Some(&cache)).is_err());
}

#[test]
fn typed_values_must_match_proven_payload() {
    // Simulate a decoder/index returning different typed values even though
    // the independently read raw payload and its proof are valid.
    #[derive(Serialize)]
    #[serde(transparent)]
    struct Altered(PleLayerParams);
    impl<'de> serde::Deserialize<'de> for Altered {
        fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
            let mut p = <PleLayerParams as serde::Deserialize>::deserialize(d)?;
            p.input_scale += 1;
            Ok(Self(p))
        }
    }
    let f = Fixture::new(4, 32);
    let input = f.dir.join("input.json");
    let manifest = f.dir.join("input_manifest.json");
    with_stage_sequence_scope(&input, &manifest, || {
        let binding = raster::start_program(&[
            raster::entry_argument_spec::<ActivationSequence>("embedded"),
            raster::entry_argument_spec::<PleLayer>("layer"),
        ])?;
        let source = VerifiedSelections::new(binding, input.to_str(), manifest.to_str())?;
        let result = source.select::<Altered>(fields(&["layer", "params"]));
        assert!(result
            .err()
            .unwrap()
            .to_string()
            .contains("typed selection differs"));
        Ok(())
    })
    .unwrap();
}

#[test]
fn postcard_inputs_keep_eager_compatibility() {
    let f = Fixture::new(4, 32);
    f.write(true);
    let inputs = crate::routines::prefill_prepare_aux::load_inputs_from_paths(
        &f.dir.join("input.json"),
        &f.dir.join("input_manifest.json"),
        &crate::cache::CachedInputs::new(),
    )
    .unwrap();
    let actual = crate::routines::prefill_prepare_aux::run_direct(&inputs).unwrap();
    let eager = run_prefill_prepare_aux_direct(PrefillPrepareAuxDirectInputs {
        embedded: &f.embedded,
        layer: &f.layer,
    })
    .unwrap();
    assert_eq!(actual, eager);
}

#[test]
fn loading_mode_rejects_unknown_values() {
    assert_eq!(Loading::parse("selective").unwrap(), Loading::Selective);
    assert_eq!(Loading::parse("eager").unwrap(), Loading::Eager);
    for value in ["", "auto", "SELECTIVE"] {
        assert!(Loading::parse(value).is_err());
    }
}

#[test]
#[ignore = "requires actual Raster stage binaries in STAGED_INFER_REPLAY_BIN_DIR"]
fn multi_page_matches_authenticated_raster_stage() {
    let bin = PathBuf::from(
        std::env::var_os("STAGED_INFER_REPLAY_BIN_DIR").expect("build Raster stage binaries first"),
    );
    let mut f = Fixture::new(4, 32);
    for boundary_values in [false, true] {
        if boundary_values {
            f.layer.params.embedding_scale = i32::MAX;
            f.layer.params.projection_scalar = i32::MIN;
            f.layer.params.input_scale = i32::MIN;
            f.layer.projection = Bytes::paged(
                [i32::MAX, i32::MIN, -1, 0, 1, -ONE, ONE, i32::MAX]
                    .iter()
                    .cycle()
                    .take(f.layer.projection.byte_len() as usize / 4)
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            for row in f.embedded.rows.iter_mut() {
                row.values = pack_i32s(&[i32::MAX, i32::MIN]);
            }
            f.write(false);
        }
        let output = run(&f.embedded, &f.load(None).unwrap()).unwrap();
        let (data, index, commitment) = raster::encode_raster_value(&output).unwrap();
        let dest = f.dir.join(format!("authenticated-{boundary_values}"));
        fs::create_dir(&dest).unwrap();
        let result = std::process::Command::new(bin.join(format!(
            "prefill-prepare-aux{}",
            std::env::consts::EXE_SUFFIX
        )))
        .arg("--input")
        .arg(f.dir.join("input.json"))
        .arg("--input-manifest")
        .arg(f.dir.join("input_manifest.json"))
        .env(raster_runtime::auth::AUTH_ENV, "1")
        .env(raster_runtime::OUTPUT_DIR_ENV, &dest)
        .env(raster_runtime::TRACE_PATH_ENV, dest.join("trace.json"))
        .env(raster_runtime::TRACE_FORMAT_ENV, "json")
        .output()
        .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(data, fs::read(dest.join("output.bin")).unwrap());
        assert_eq!(index, fs::read(dest.join("output.rindex")).unwrap());
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(dest.join("output_manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest["output"]["commitment"], commitment);
    }
}

#[test]
#[ignore = "release loading benchmark; no wall-clock threshold"]
fn selective_loading_benchmark() {
    let f = Fixture::new(128, 32);
    let input = f.dir.join("input.json");
    let manifest = f.dir.join("input_manifest.json");
    let mut eager_ns = 0;
    let mut selective_ns = 0;
    for _ in 0..5 {
        let start = Instant::now();
        with_stage_sequence_scope(&input, &manifest, || {
            let binding = raster::start_program(&[
                raster::entry_argument_spec::<ActivationSequence>("embedded"),
                raster::entry_argument_spec::<PleLayer>("layer"),
            ])?;
            let layer = raster::materialize_auth_return(
                raster::entry_argument_auth_ref::<PleLayer>(binding.reference, "layer"),
            );
            let output = run_prefill_prepare_aux_direct(PrefillPrepareAuxDirectInputs {
                embedded: &f.embedded,
                layer: &layer,
            })?;
            std::hint::black_box(output);
            Ok(())
        })
        .unwrap();
        eager_ns += start.elapsed().as_nanos();
        let start = Instant::now();
        let selected = f.load(None).unwrap();
        std::hint::black_box(run(&f.embedded, &selected).unwrap());
        selective_ns += start.elapsed().as_nanos();
        assert_eq!(selected.pages.len(), 3);
        f.assert_parity(&selected);
    }
    println!("PLE benchmark: table_bytes={} selected_embedding_bytes={} eager_mean_ms={:.3} selective_mean_ms={:.3}",
        f.layer.embeddings.byte_len(), 2 * Q8_PAGE_SIZE as usize + 3 * row_bytes(32), eager_ns as f64 / 5e6, selective_ns as f64 / 5e6);
}

#[test]
fn auxiliary_sparse_witness_is_rechecked_after_cache_hit() {
    let mut f = Fixture::new(4, 32);
    let root = raster::core::tree::value_root(&f.layer).unwrap();
    let d = model_identity::ModelDescriptor {
        version: 2,
        tokenizer: [1; 32],
        decoder: [2; 32],
        embedding: [3; 32],
        head: [4; 32],
        auxiliary: vec![root; 4],
        transformer: vec![[5; 32]; 4],
    };
    let id = d
        .model_id()
        .unwrap()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    let mut prompt = model_identity::ModelContext::bind(&d.encode().unwrap(), &id);
    prompt.bind_statement(&"11".repeat(32), false, 8, 128);
    prompt.bind_schedule(model_identity::pipeline_core::StageSchedule {
        prompt_ranges: 1,
        claimed_ranges: 1,
    });
    prompt.begin_prompt_range();
    for row in f.embedded.rows.iter() {
        prompt.append_consumed_token(row.token_id);
    }
    prompt.finish_prompt_range(true, f.embedded.rows.len() as u32, 0);
    f.embedded.model = model_identity::ModelContext::embedding(&prompt, &Default::default());
    for row in f.embedded.rows.iter() {
        f.embedded.model.append_consumed_token(row.token_id);
    }
    f.embedded
        .model
        .finish_embedding_execution(0, f.embedded.rows.len() as u64, 0);
    f.write(false);
    let cache = MaterializationCache::with_budget(Arc::new(WeightBudget::new(1 << 20)));
    let run = || {
        let inputs = crate::routines::prefill_prepare_aux::load_inputs_from_paths_with_cache(
            &f.dir.join("input.json"),
            &f.dir.join("input_manifest.json"),
            &crate::cache::CachedInputs::new(),
            Some(&cache),
        )
        .unwrap();
        crate::routines::prefill_prepare_aux::run_direct(&inputs).unwrap()
    };
    assert_eq!(run().model.fault, 0);
    let mut witness = model_identity::sparse::witness(f.layer.embeddings.pages.as_slice()).unwrap();
    let mut paths = witness.paths.as_slice().to_vec();
    paths[0].siblings = BytesPage::__from_parts(0, 0, vec![0; 31]);
    witness.paths = paths.into();
    f.release_mappings();
    let (data, index, commitment) = raster::encode_raster_value(&witness).unwrap();
    fs::write(f.dir.join("witness.bin"), data).unwrap();
    fs::write(f.dir.join("witness.rindex"), index).unwrap();
    let manifest_path = f.dir.join("input_manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    manifest["witness"]["commitment"] = commitment.into();
    fs::write(manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert_ne!(run().model.fault, 0);
    f.write(false);
    assert_eq!(run().model.fault, 0);
}
