use super::*;
use serde::Serialize;
use crate::{artifact_io::with_stage_sequence_scope, cache::materialization_key_from_stage_files};
use inference_kernels::{
    kernels::prefill_prepare_aux::{run_prefill_prepare_aux_direct, PrefillPrepareAuxDirectInputs},
    weight_budget::WeightBudget,
};
use prefill_prepare_aux::input::{ActivationRow, PleLayer};
use std::{
    fs,
    path::PathBuf,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const ONE: i32 = 1 << 16;

struct Fixture {
    dir: PathBuf,
    layer: PleLayer,
    embedded: ActivationSequence,
}

impl Fixture {
    fn new(pages: usize, width: usize) -> Self {
        let count = (pages - 1) * PAGE_SIZE as usize / 4 + width * 3;
        let values: Vec<_> = (0..count).map(|i| ((i % 97) as i32 - 48) * 2311).collect();
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
            embeddings: Bytes::paged(
                values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            projection: Bytes::paged(
                (0..width * 2)
                    .flat_map(|i| ((i as i32 - 3) * 17001).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        };
        let rows_per_page = PAGE_SIZE as u32 / (width as u32 * 4);
        let tokens = [0, rows_per_page, rows_per_page, (count / width - 1) as u32];
        let embedded = ActivationSequence {
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
    let f = Fixture::new(4, 4);
    let budget = Arc::new(WeightBudget::new(1 << 20));
    let cache = MaterializationCache::with_budget(budget.clone());
    let first = f.load(Some(&cache)).unwrap();
    assert_eq!(first.pages.keys().copied().collect::<Vec<_>>(), [0, 1, 3]);
    assert_eq!(
        first.pages.values().map(BytesPage::len).sum::<usize>(),
        2 * PAGE_SIZE as usize + 48
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
    let mut f = Fixture::new(4, 4);
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
    let mut f = Fixture::new(4, 4);
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
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    f.write(false);
    f.assert_parity(&f.load(None).unwrap());
}

#[test]
fn bad_tokens_cross_page_rows_and_projection_geometry_fail() {
    let mut f = Fixture::new(4, 4);
    f.load(None).unwrap();
    f.embedded.rows = List::from(vec![ActivationRow {
        token_id: u32::MAX,
        values: pack_i32s(&[ONE, ONE]),
    }]);
    f.write(false);
    assert!(f.load(None).is_err());
    // Width 5 does not divide a page: this row starts in page 0 and ends in 1.
    let mut f = Fixture::new(4, 5);
    f.embedded.rows = List::from(vec![ActivationRow {
        token_id: (PAGE_SIZE / 20) as u32,
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
    let mut f = Fixture::new(4, 4);
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
    let mut f = Fixture::new(4, 4);
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
    let f = Fixture::new(4, 4);
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
    let f = Fixture::new(4, 4);
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
    let f = Fixture::new(4, 4);
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
    let f = Fixture::new(4, 4);
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
    let mut f = Fixture::new(4, 4);
    for boundary_values in [false, true] {
        if boundary_values {
            f.layer.params.embedding_scale = i32::MAX;
            f.layer.params.projection_scalar = i32::MIN;
            f.layer.params.input_scale = i32::MIN;
            f.layer.projection = Bytes::paged(
                [i32::MAX, i32::MIN, -1, 0, 1, -ONE, ONE, i32::MAX]
                    .iter()
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
    let f = Fixture::new(128, 4);
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
        f.layer.embeddings.byte_len(), 2 * PAGE_SIZE + 48, eager_ns as f64 / 5e6, selective_ns as f64 / 5e6);
}
