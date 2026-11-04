use decode_select_token::input::{DecodeEdge, LogitEntry, PrefillLogits};
use input_embedding::input::{EmbeddingTable, PromptTokenization};
use output_finalize::input::{DecoderTable, DecoderToken};
use prefill_finalize::input::{FinalHead, FinalHeadParams};
use prefill_prepare_aux::input::{PleLayer, PleLayerParams};
use prompt_prepare::input::{BpePieces, MergeBucket, PromptTokenizer, TokenEntry, VocabBucket};
use raster::{Bytes, List};
use serde::Serialize;
use staged_infer::{kernels, routines};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

const ONE: i32 = 1 << 16;

fn bytes_of_i32s(values: &[i32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Hidden width of the fixtures: one `Q8_0` block, the narrowest int8 row.
const W: usize = 32;

/// Rows `W` wide, each holding its listed values then zeros.
fn wide_rows(rows: &[&[i32]]) -> Vec<i32> {
    rows.iter()
        .flat_map(|head| {
            let mut row = vec![0; W];
            row[..head.len()].copy_from_slice(head);
            row
        })
        .collect()
}

fn identity() -> Vec<i32> {
    (0..W * W)
        .map(|i| if i % (W + 1) == 0 { ONE } else { 0 })
        .collect()
}

/// Q16.16 `values` as exact `Q8_0` blocks: each block takes the largest
/// power-of-two scale that represents all its values exactly.
fn q8_paged<const P: u64>(values: &[i32]) -> Bytes<P> {
    let mut out = Vec::new();
    for block in values.chunks(32) {
        assert_eq!(block.len(), 32, "Q8_0 rows are whole blocks");
        let k = (0..=14)
            .find(|k| {
                let step = ONE >> k;
                block
                    .iter()
                    .all(|v| v % step == 0 && (v / step).abs() <= 127)
            })
            .expect("values are not exactly representable as Q8_0");
        out.extend_from_slice(&(((15 - k) as u16) << 10).to_le_bytes());
        out.extend(block.iter().map(|v| (v / (ONE >> k)) as i8 as u8));
    }
    Bytes::<P>::paged(out).unwrap()
}

fn paged(values: &[i32]) -> Bytes<196_608> {
    Bytes::<196_608>::paged(bytes_of_i32s(values)).unwrap()
}

fn assert_same_json<T: Serialize>(left: &T, right: &T) {
    assert_eq!(
        serde_json::to_value(left).unwrap(),
        serde_json::to_value(right).unwrap()
    );
}

#[test]
fn decode_select_token_keeps_first_token_on_tie() {
    let inputs = routines::decode_select_token::Inputs {
        logits: PrefillLogits {
            decode_position: 9,
            logits: List::from(vec![
                LogitEntry {
                    token_id: 5,
                    value: 10,
                },
                LogitEntry {
                    token_id: 3,
                    value: 10,
                },
            ]),
            errors: List::new(),
        },
        prior: DecodeEdge {
            has_selected: true,
            decode_position: 8,
            token_id: 11,
            value: 1,
            claim_mismatches: 0,
            first_claim_mismatch_position: 0,
            generated_token_ids: List::from(vec![11, 12]),
        },
    };
    let output = routines::decode_select_token::run_direct(&inputs).unwrap();
    let kernel_output = kernels::decode_select_token::run_decode_select_token_direct(
        kernels::decode_select_token::DecodeSelectTokenDirectInputs {
            logits: &inputs.logits,
            prior: &inputs.prior,
        },
    )
    .unwrap();

    assert_same_json(&output, &kernel_output);
    assert!(output.has_selected);
    assert_eq!(output.decode_position, 9);
    assert_eq!(output.token_id, 5);
    assert_eq!(output.value, 10);
    assert_eq!(output.generated_token_ids.as_slice(), &[11, 12, 5]);
}

#[test]
fn decode_init_outputs_empty_edge() {
    let output = routines::decode_init::run_direct(&routines::decode_init::Inputs).unwrap();
    let kernel_output =
        kernels::decode_init::run_decode_init_direct(kernels::decode_init::DecodeInitDirectInputs)
            .unwrap();

    assert_same_json(&output, &kernel_output);
    assert!(!output.has_selected);
    assert_eq!(output.decode_position, 0);
    assert_eq!(output.token_id, 0);
    assert_eq!(output.value, 0);
    assert!(output.generated_token_ids.is_empty());
}

#[test]
fn cached_runner_executes_without_writing_output_artifact() {
    let base = temp_dir("cached-runner");
    fs::create_dir_all(&base).unwrap();
    let input = base.join("input.json");
    let input_manifest = base.join("input_manifest.json");
    fs::write(&input, "{}").unwrap();
    fs::write(&input_manifest, "{}").unwrap();

    let output = routines::run_cached_from_paths(
        &routines::StageKind::DecodeInit,
        &input,
        &input_manifest,
        &Default::default(),
    )
    .unwrap();

    assert!(matches!(
        output.output,
        staged_infer::cache::CachedStageValue::DecodeEdge(_)
    ));
    assert!(!base.join("output.bin").exists());
    fs::remove_dir_all(base).unwrap();
}

#[test]
fn decode_embed_uses_selected_decode_position() {
    let selected = decode_embed::input::DecodeEdge {
        has_selected: true,
        decode_position: 9,
        token_id: 1,
        value: 10,
        claim_mismatches: 0,
        first_claim_mismatch_position: 0,
        generated_token_ids: List::from(vec![1]),
    };
    let embedding: std::sync::Arc<decode_embed::input::EmbeddingTable> =
        decode_embed::input::EmbeddingTable {
            hidden_size: W as u32,
            embedding_scale: ONE,
            values: q8_paged(&wide_rows(&[&[], &[ONE, 2 * ONE]])),
        }
        .into();
    let inputs = routines::decode_embed::Inputs::eager(selected, embedding.clone());
    let output = routines::decode_embed::run_direct(&inputs).unwrap();
    let kernel_output = kernels::decode_embed::run_decode_embed_direct(
        kernels::decode_embed::DecodeEmbedDirectInputs {
            selected: &inputs.selected,
            embedding: &embedding,
        },
    )
    .unwrap();

    assert_same_json(&output, &kernel_output);
    assert_eq!(output.rows.len(), 1);
    assert_eq!(output.rows[0].token_id, 1);
    assert_eq!(output.start_position, 9);
    assert!(output.errors.is_empty());
}

fn temp_dir(label: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "staged-infer-routines-{label}-{}-{nanos}",
        std::process::id()
    ))
}

#[test]
fn output_finalize_decodes_generated_ids() {
    let inputs = routines::output_finalize::Inputs {
        edge: output_finalize::input::DecodeEdge {
            has_selected: true,
            decode_position: 2,
            token_id: 1,
            value: 10,
            claim_mismatches: 2,
            first_claim_mismatch_position: 1,
            generated_token_ids: List::from(vec![0, 1]),
        },
        decoder: DecoderTable {
            tokens: List::from(vec![
                DecoderToken {
                    token: "▁Hi".to_string(),
                    special: false,
                    terminal: false,
                },
                DecoderToken {
                    token: "!".to_string(),
                    special: false,
                    terminal: false,
                },
            ]),
        },
    };
    let output = routines::output_finalize::run_direct(&inputs).unwrap();
    let kernel_output = kernels::output_finalize::run_output_finalize_direct(
        kernels::output_finalize::OutputFinalizeDirectInputs {
            edge: &inputs.edge,
            decoder: &inputs.decoder,
        },
    )
    .unwrap();

    assert_same_json(&output, &kernel_output);
    // The claim verdict is published as the edge carried it.
    assert_eq!(output.claim_mismatches, 2);
    assert_eq!(output.first_claim_mismatch_position, 1);
    assert_eq!(output.generated_token_count, 2);
    assert_eq!(output.generated_token_ids.as_slice(), &[0, 1]);
    assert_eq!(output.generated_text, " Hi!");
    assert_eq!(output.stop_reason, "max_new_tokens");
}

#[test]
fn prompt_prepare_resolves_vocab_bucket() {
    let inputs = routines::prompt_prepare::Inputs {
        tokenizer: PromptTokenizer {
            vocab_bucket_count: 1,
            merge_bucket_count: 1,
            vocab_buckets: List::from(vec![VocabBucket {
                entries: List::from(vec![TokenEntry {
                    token: "hello".to_string(),
                    id: 42,
                }]),
            }]),
            merge_buckets: List::from(vec![MergeBucket { rules: List::new() }]),
        }
        .into(),
        initial_pieces: BpePieces {
            pieces: List::from(vec![prompt_prepare::input::BpePiece {
                text: "hello".to_string(),
                segment: 0,
            }]),
        },
    };
    let output = routines::prompt_prepare::run_direct(&inputs).unwrap();
    let kernel_output = kernels::prompt_prepare::run_prompt_prepare_direct(
        kernels::prompt_prepare::PromptPrepareDirectInputs {
            tokenizer: &inputs.tokenizer,
            initial_pieces: &inputs.initial_pieces,
        },
    )
    .unwrap();

    assert_same_json(&output, &kernel_output);
    assert_eq!(output.token_ids.as_slice(), &[42]);
}

#[test]
fn input_embedding_gathers_and_scales_rows() {
    let prompt = PromptTokenization {
        token_ids: List::from(vec![1]),
    };
    let embedding: std::sync::Arc<input_embedding::input::EmbeddingTable> = EmbeddingTable {
        hidden_size: W as u32,
        embedding_scale: ONE,
        values: q8_paged(&wide_rows(&[&[], &[ONE, 2 * ONE]])),
    }
    .into();
    let inputs = routines::input_embedding::Inputs::eager(prompt, embedding.clone());
    let output = routines::input_embedding::run_direct(&inputs).unwrap();
    let kernel_output = kernels::input_embedding::run_input_embedding_direct(
        kernels::input_embedding::InputEmbeddingDirectInputs {
            prompt: &inputs.prompt,
            embedding: &embedding,
        },
    )
    .unwrap();

    assert_same_json(&output, &kernel_output);
    assert_eq!(output.rows.len(), 1);
    assert_eq!(output.rows[0].token_id, 1);
    assert!(output.errors.is_empty());
    assert!(output.kv.is_empty());
}

#[test]
fn prefill_prepare_aux_publishes_one_row() {
    let norm = prefill_prepare_aux::input::pack_i32s(&[ONE; W]);
    let activation = prefill_prepare_aux::input::ActivationSequence {
        rows: List::from(vec![prefill_prepare_aux::input::ActivationRow {
            token_id: 0,
            values: prefill_prepare_aux::input::pack_i32s(&wide_rows(&[&[ONE]])),
        }]),
        errors: List::new(),
        kv: List::new(),
        start_position: 0,
    };

    let layer = std::sync::Arc::new(PleLayer {
        params: PleLayerParams {
            layer_idx: 7,
            hidden_size: W as u32,
            ple_width: W as u32,
            embedding_scale: ONE,
            projection_scalar: ONE,
            input_scale: ONE,
            norm_eps: 0,
            norm_weights: norm.clone(),
        },
        embeddings: q8_paged(&wide_rows(&[&[ONE]])),
        projection: paged(&identity()),
    });
    let inputs = routines::prefill_prepare_aux::Inputs::eager(activation, layer.clone());
    let output = routines::prefill_prepare_aux::run_direct(&inputs).unwrap();
    let kernel_output = kernels::prefill_prepare_aux::run_prefill_prepare_aux_direct(
        kernels::prefill_prepare_aux::PrefillPrepareAuxDirectInputs {
            embedded: &inputs.embedded,
            layer: &layer,
        },
    )
    .unwrap();

    assert_same_json(&output, &kernel_output);
    assert_eq!(output.layer_idx, 7);
    assert_eq!(output.rows.len(), 1);
    assert!(output.errors.is_empty());
}

#[test]
fn prefill_prepare_aux_preserves_prompt_row_order() {
    let norm = prefill_prepare_aux::input::pack_i32s(&[ONE; W]);
    let row = |token_id, values: &[i32]| prefill_prepare_aux::input::ActivationRow {
        token_id,
        values: prefill_prepare_aux::input::pack_i32s(&wide_rows(&[values])),
    };
    let layer = || PleLayer {
        params: PleLayerParams {
            layer_idx: 7,
            hidden_size: W as u32,
            ple_width: W as u32,
            embedding_scale: ONE,
            projection_scalar: ONE,
            input_scale: ONE,
            norm_eps: 0,
            norm_weights: norm.clone(),
        },
        embeddings: q8_paged(&wide_rows(&[&[], &[ONE], &[0, ONE]])),
        projection: paged(&identity()),
    };
    let run = |rows| {
        routines::prefill_prepare_aux::run_direct(&routines::prefill_prepare_aux::Inputs::eager(
            prefill_prepare_aux::input::ActivationSequence {
                rows: List::from(rows),
                errors: List::new(),
                kv: List::new(),
                start_position: 0,
            },
            layer().into(),
        ))
        .unwrap()
    };

    let output = run(vec![
        row(2, &[ONE, 0]),
        row(0, &[0, ONE]),
        row(1, &[ONE, ONE]),
    ]);
    let expected = [
        run(vec![row(2, &[ONE, 0])]).rows[0].values.clone(),
        run(vec![row(0, &[0, ONE])]).rows[0].values.clone(),
        run(vec![row(1, &[ONE, ONE])]).rows[0].values.clone(),
    ];

    assert_eq!(output.rows.len(), 3);
    assert!(output.errors.is_empty());
    assert_eq!(output.rows[0].values, expected[0]);
    assert_eq!(output.rows[1].values, expected[1]);
    assert_eq!(output.rows[2].values, expected[2]);
}

#[test]
fn sequence_check_matches_finalize_then_select_at_every_position() {
    use sequence_check::input as check;
    let hidden = W;
    let vocab = 16;
    // Fractional, signed, and with repeated projection rows so ties occur.
    let value = |seed: usize| ((seed * 7919 % 131) as i32 - 65) * (ONE / 37);
    // One Q8_0 block per row at scale 2^-7; rows repeat with period 5.
    let projection_bytes = (0..vocab)
        .flat_map(|row| {
            let mut block = 0x2000_u16.to_le_bytes().to_vec();
            block.extend(
                (0..hidden).map(|i| ((((row % 5) * 13 + i * 7) % 131) as i32 - 65) as i8 as u8),
            );
            block
        })
        .collect::<Vec<_>>();
    let projection = || Bytes::<52_224>::paged(projection_bytes.clone()).unwrap();
    let norm = (0..hidden).map(|i| ONE + value(i) / 3).collect::<Vec<_>>();
    let row_values = |row: usize| {
        (0..hidden)
            .map(|i| value(row * 13 + i * 5 + 1))
            .collect::<Vec<_>>()
    };
    let prompt_len = 3;
    let rows = 6;

    // The autoregressive path: finalize the single predicting row, select.
    let finalize_head: std::sync::Arc<FinalHead> = FinalHead {
        params: FinalHeadParams {
            hidden_size: hidden as u32,
            norm_eps: 1 << 10,
            softcap: 30 * ONE,
            norm_weights: prefill_finalize::input::pack_i32s(&norm),
        },
        projection: projection(),
    }
    .into();
    let mut prior = DecodeEdge::default();
    for row in prompt_len - 1..rows - 1 {
        let logits =
            routines::prefill_finalize::run_direct(&routines::prefill_finalize::Inputs::eager(
                prefill_finalize::input::ActivationSequence {
                    rows: List::from(vec![prefill_finalize::input::ActivationRow {
                        token_id: 0,
                        values: prefill_finalize::input::pack_i32s(&row_values(row)),
                    }]),
                    errors: List::new(),
                    kv: List::new(),
                    start_position: row as u32,
                },
                finalize_head.clone(),
            ))
            .unwrap();
        prior = routines::decode_select_token::run_direct(&routines::decode_select_token::Inputs {
            logits: decode_select_token::input::PrefillLogits {
                decode_position: logits.decode_position,
                logits: List::from(
                    logits
                        .logits
                        .iter()
                        .map(|entry| LogitEntry {
                            token_id: entry.token_id,
                            value: entry.value,
                        })
                        .collect::<Vec<_>>(),
                ),
                errors: List::new(),
            },
            prior,
        })
        .unwrap();
    }

    // Claim the greedy ids except one, which the check must not echo.
    let mut claimed = prior
        .generated_token_ids
        .iter()
        .copied()
        .collect::<Vec<_>>();
    claimed[1] = (claimed[1] + 1) % vocab as u32;
    let check_head: std::sync::Arc<check::FinalHead> = check::FinalHead {
        params: check::FinalHeadParams {
            hidden_size: hidden as u32,
            norm_eps: 1 << 10,
            softcap: 30 * ONE,
            norm_weights: check::pack_i32s(&norm),
        },
        projection: projection(),
    }
    .into();
    // Prompt range rows 0..3, then claimed ranges rows 3..5 and row 5: the
    // second range's first token is predicted by the first range's last row,
    // and its transcript continues the first's.
    let sequence = |range: std::ops::Range<usize>| check::ActivationSequence {
        rows: List::from(
            range
                .clone()
                .map(|row| check::ActivationRow {
                    token_id: 0,
                    values: check::pack_i32s(&row_values(row)),
                })
                .collect::<Vec<_>>(),
        ),
        errors: List::new(),
        kv: List::new(),
        start_position: range.start as u32,
    };
    let claimed_range = |ids: &[u32]| check::ClaimedOutput {
        token_ids: List::from(ids.to_vec()),
    };
    let run_check = |claimed: &[u32]| {
        let first =
            routines::sequence_check::run_direct(&routines::sequence_check::Inputs::eager(
                sequence(3..5),
                sequence(0..3),
                claimed_range(&claimed[..2]),
                check::DecodeEdge::default(),
                check_head.clone(),
            ))
            .unwrap();
        let edge = routines::sequence_check::run_direct(&routines::sequence_check::Inputs::eager(
            sequence(5..6),
            sequence(3..5),
            claimed_range(&claimed[2..]),
            first.clone(),
            check_head.clone(),
        ))
        .unwrap();
        (first, edge)
    };
    let (first, edge) = run_check(&claimed);

    // The verdict is committed, and the second range carries the first's.
    assert_eq!(prior.claim_mismatches, 0);
    assert_eq!(first.claim_mismatches, 1);
    assert_eq!(first.first_claim_mismatch_position, prompt_len as u32 + 1);
    assert_eq!(edge.claim_mismatches, 1);
    assert_eq!(edge.first_claim_mismatch_position, prompt_len as u32 + 1);

    // A disagreement only in the second range is counted there.
    let mut late = prior
        .generated_token_ids
        .iter()
        .copied()
        .collect::<Vec<_>>();
    late[2] = (late[2] + 1) % vocab as u32;
    let (late_first, late_edge) = run_check(&late);
    assert_eq!(late_first.claim_mismatches, 0);
    assert_eq!(late_first.first_claim_mismatch_position, 0);
    assert_eq!(late_edge.claim_mismatches, 1);
    assert_eq!(late_edge.first_claim_mismatch_position, prompt_len as u32 + 2);

    // An honest claim publishes a clean verdict and the autoregressive edge.
    let honest = prior
        .generated_token_ids
        .iter()
        .copied()
        .collect::<Vec<_>>();
    let (_, honest_edge) = run_check(&honest);
    assert_eq!(honest_edge.claim_mismatches, 0);
    assert_eq!(honest_edge.first_claim_mismatch_position, 0);
    assert_eq!(
        serde_json::to_value(&honest_edge).unwrap(),
        serde_json::to_value(&prior).unwrap()
    );

    assert_eq!(edge.has_selected, prior.has_selected);
    assert_eq!(edge.decode_position, prior.decode_position);
    assert_eq!(edge.token_id, prior.token_id);
    assert_eq!(edge.value, prior.value);
    assert_eq!(
        edge.generated_token_ids.iter().collect::<Vec<_>>(),
        prior.generated_token_ids.iter().collect::<Vec<_>>()
    );
}

#[test]
fn prefill_finalize_scores_projection_rows() {
    let norm = prefill_finalize::input::pack_i32s(&[ONE; W]);
    let activations = prefill_finalize::input::ActivationSequence {
        rows: List::from(vec![prefill_finalize::input::ActivationRow {
            token_id: 7,
            values: prefill_finalize::input::pack_i32s(&wide_rows(&[&[ONE]])),
        }]),
        errors: List::new(),
        kv: List::new(),
        start_position: 4,
    };

    let head: std::sync::Arc<FinalHead> = FinalHead {
        params: FinalHeadParams {
            hidden_size: W as u32,
            norm_eps: 0,
            softcap: 0,
            norm_weights: norm,
        },
        projection: q8_paged(&wide_rows(&[&[ONE], &[0, ONE]])),
    }
    .into();
    let inputs = routines::prefill_finalize::Inputs::eager(activations, head.clone());
    let output = routines::prefill_finalize::run_direct(&inputs).unwrap();
    let kernel_output = kernels::prefill_finalize::run_prefill_finalize_direct(
        kernels::prefill_finalize::PrefillFinalizeDirectInputs {
            activations: &inputs.activations,
            head: &head,
        },
    )
    .unwrap();

    assert_same_json(&output, &kernel_output);
    assert_eq!(output.decode_position, 5);
    assert_eq!(output.logits.len(), 2);
    assert!(output.errors.is_empty());
}
