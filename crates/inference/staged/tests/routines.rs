use input_embedding::input::{EmbeddingTable, PromptTokenization};
use output_finalize::input::{DecoderTable, DecoderToken};
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
        descriptor: raster::BytesPage::__from_parts(0, 0, vec![]),
        witness: output_finalize::input::DecoderWitness {
            token_count: 0,
            paths: List::new(),
        },
        edge: output_finalize::input::DecodeEdge {
            model: Default::default(),
            has_selected: true,
            decode_position: 2,
            token_id: 1,
            value: 10,
            claim_mismatches: 2,
            first_claim_mismatch_position: 1,
            check_fault: 0,
            check_fault_position: 0,
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
        context: output_finalize::prompt::PromptContext {
            max_new_tokens: 2,
            range_size: 4,
            ..Default::default()
        },
    };
    let inputs = authenticate_decoder_inputs(inputs);
    let output = routines::output_finalize::run_direct(&inputs).unwrap();
    let kernel_output = kernels::output_finalize::run_output_finalize_direct(
        kernels::output_finalize::OutputFinalizeDirectInputs {
            edge: &inputs.edge,
            decoder: &inputs.decoder,
            context: &inputs.context,
            descriptor: &inputs.descriptor,
            witness: &inputs.witness,
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
    assert_eq!(output.output_fault, 0);
}

/// A decoder with text, an accent split into byte fallback, a special token and
/// a terminal one, and a run over `ids` with this budget.
fn finalize(ids: &[u32], max_new_tokens: u32) -> output_finalize::input::GeneratedOutput {
    let token = |token: &str, special, terminal| DecoderToken {
        token: token.to_string(),
        special,
        terminal,
    };
    let inputs = routines::output_finalize::Inputs {
        descriptor: raster::BytesPage::__from_parts(0, 0, vec![]),
        witness: output_finalize::input::DecoderWitness {
            token_count: 0,
            paths: List::new(),
        },
        edge: output_finalize::input::DecodeEdge {
            has_selected: !ids.is_empty(),
            decode_position: 9,
            token_id: ids.last().copied().unwrap_or(0),
            generated_token_ids: List::from(ids.to_vec()),
            ..Default::default()
        },
        decoder: DecoderTable {
            tokens: List::from(vec![
                token("▁caf", false, false),
                token("<0xC3>", false, false),
                token("<0xA9>", false, false),
                token("<ctl>", true, false),
                token("<eos>", true, true),
            ]),
        },
        context: output_finalize::prompt::PromptContext {
            input_text_sha256: String::from("input-hash"),
            model_id: String::from("model-id"),
            raw_prompt: true,
            max_new_tokens,
            range_size: 4,
            prompt_tokens: 9,
            complete: true,
            fault: 2,
            fault_position: 7,
            ..Default::default()
        },
    };
    let inputs = authenticate_decoder_inputs(inputs);
    let output = routines::output_finalize::run_direct(&inputs).unwrap();
    assert_same_json(
        &output,
        &kernels::output_finalize::run_output_finalize_direct(
            kernels::output_finalize::OutputFinalizeDirectInputs {
                edge: &inputs.edge,
                decoder: &inputs.decoder,
                context: &inputs.context,
                descriptor: &inputs.descriptor,
                witness: &inputs.witness,
            },
        )
        .unwrap(),
    );
    output
}

#[test]
fn output_finalize_commits_the_exact_text_it_emits() {
    use sha2::{Digest, Sha256};
    let output = finalize(&[0, 1, 2, 3, 0, 4], 6);
    // Byte fallback joins into UTF-8, the marker becomes a space, and the
    // special token contributes nothing.
    assert_eq!(output.generated_text, " café caf");
    assert_eq!(output.stop_reason, "eos");
    assert_eq!(
        output.output_text_sha256,
        format!("{:x}", Sha256::digest(" café caf".as_bytes()))
    );
    assert_ne!(output.output_text_sha256, output.generated_token_ids_sha256);
    assert_eq!(
        output.generated_token_ids_sha256,
        format!("{:x}", Sha256::digest(b"[0,1,2,3,0,4]"))
    );

    // Text after a terminal token is not the answer, so it is not committed.
    let past = finalize(&[0, 4, 0], 6);
    assert_eq!(past.generated_text, " caf");
    assert_eq!(
        past.output_text_sha256,
        format!("{:x}", Sha256::digest(" caf".as_bytes()))
    );
}

#[test]
fn output_finalize_reports_what_the_chain_consumed() {
    let output = finalize(&[0], 1);
    assert_eq!(output.input_text_sha256, "11".repeat(32));
    assert_eq!(output.model_id.len(), 64);
    assert_ne!(output.model_fault, 0); // The deliberately faulty prompt cannot authorize a claim.
    assert!(output.raw_prompt);
    assert_eq!(output.max_new_tokens, 1);
    assert_eq!(output.range_size, 4);
    assert_eq!(output.prompt_tokens, 9);
    assert!(output.prompt_complete);
    assert_eq!(output.prompt_fault, 2);
    assert_eq!(output.prompt_fault_position, 7);
}

#[test]
fn output_finalize_publishes_budget_and_eos_faults_instead_of_failing() {
    use output_finalize::input::*;
    for (ids, budget, expected) in [
        (vec![0, 0], 2, OUTPUT_FAULT_NONE),
        (vec![0, 4], 2, OUTPUT_FAULT_NONE),
        (vec![0, 4], 8, OUTPUT_FAULT_NONE),
        (vec![], 2, OUTPUT_FAULT_NO_TOKENS),
        (vec![0, 0, 0], 2, OUTPUT_FAULT_OVER_BUDGET),
        (vec![0, 0], 8, OUTPUT_FAULT_SHORT_WITHOUT_EOS),
        (vec![4, 0], 2, OUTPUT_FAULT_PAST_EOS),
        // A run that names no budget has none to spare.
        (vec![0], 0, OUTPUT_FAULT_OVER_BUDGET),
    ] {
        assert_eq!(
            finalize(&ids, budget).output_fault,
            expected,
            "{ids:?} of {budget}"
        );
    }

    // The edge's flag and its transcript disagree: published, not raised.
    let inputs = routines::output_finalize::Inputs {
        descriptor: raster::BytesPage::__from_parts(0, 0, vec![]),
        witness: output_finalize::input::DecoderWitness {
            token_count: 0,
            paths: List::new(),
        },
        edge: output_finalize::input::DecodeEdge {
            has_selected: true,
            ..Default::default()
        },
        decoder: DecoderTable {
            tokens: List::new(),
        },
        context: Default::default(),
    };
    let inputs = authenticate_decoder_inputs(inputs);
    let output = routines::output_finalize::run_direct(&inputs).unwrap();
    assert_eq!(output.output_fault, OUTPUT_FAULT_EDGE_DISAGREES);
}

#[test]
fn prompt_prepare_resolves_vocab_bucket() {
    let tokenizer = PromptTokenizer {
        vocab_bucket_count: 1,
        merge_bucket_count: 1,
        vocab_buckets: List::from(vec![VocabBucket {
            entries: List::from(vec![TokenEntry {
                token: "hello".to_string(),
                id: 42,
            }]),
        }]),
        merge_buckets: List::from(vec![MergeBucket { rules: List::new() }]),
        ..Default::default()
    };
    let mut inputs = routines::prompt_prepare::Inputs {
        witness: prompt_prepare::input::tokenizer_witness(&tokenizer)
            .unwrap()
            .into(),
        tokenizer: tokenizer.into(),
        initial_pieces: BpePieces {
            pieces: List::from(vec![prompt_prepare::input::BpePiece {
                text: "hello".to_string(),
                segment: 0,
            }]),
            ..Default::default()
        },
    };
    let root = raster::core::tree::value_root(inputs.tokenizer.as_ref()).unwrap();
    let d = model_identity::ModelDescriptor {
        version: 1,
        tokenizer: root,
        decoder: [2; 32],
        embedding: [3; 32],
        head: [4; 32],
        auxiliary: vec![[5; 32]],
        transformer: vec![[6; 32]],
    };
    inputs.initial_pieces.context.model_id = hex::encode(d.model_id().unwrap());
    inputs.initial_pieces.context.model = model_identity::ModelContext::bind(
        &d.encode().unwrap(),
        &inputs.initial_pieces.context.model_id,
    );
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
        ..Default::default()
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
        model: Default::default(),
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
                model: Default::default(),
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
fn sequence_check_matches_a_greedy_oracle_at_every_position() {
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

    // What direct generation does: normalise the predicting row, score the whole
    // vocabulary, keep the first maximum.
    let mut prior = check::DecodeEdge::default();
    let mut greedy = Vec::new();
    let stride = det_num::q8::row_bytes(hidden).unwrap();
    for row in prompt_len - 1..rows - 1 {
        let mut normalized = row_values(row);
        check::rms_norm(&mut normalized, &norm, 1 << 10).unwrap();
        let (mut best_id, mut best) = (0, i32::MIN);
        for (id, bytes) in projection_bytes.chunks_exact(stride).enumerate() {
            let value = check::softcap(det_num::q8::dot_row(&normalized, bytes).unwrap(), 30 * ONE);
            if id == 0 || value > best {
                (best_id, best) = (id as u32, value);
            }
        }
        greedy.push(best_id);
        prior = check::DecodeEdge {
            has_selected: true,
            decode_position: row as u32 + 1,
            token_id: best_id,
            value: best,
            generated_token_ids: List::from(greedy.clone()),
            ..Default::default()
        };
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
        model: Default::default(),
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
    let context = sequence_check::prompt::PromptContext {
        range_size: 4,
        prompt_tokens: 3,
        ..Default::default()
    };
    let check_range =
        |mut activations: check::ActivationSequence, previous, claimed: &[u32], prior| {
            activations.rows = List::from(
                activations
                    .rows
                    .iter()
                    .zip(claimed)
                    .map(|(row, id)| check::ActivationRow {
                        token_id: *id,
                        values: row.values.clone(),
                    })
                    .collect::<Vec<_>>(),
            );
            routines::sequence_check::run_direct(&routines::sequence_check::Inputs::eager(
                activations,
                previous,
                prior,
                check_head.clone(),
                context.clone(),
            ))
            .unwrap()
        };
    let run_check = |claimed: &[u32]| {
        let first = check_range(
            sequence(3..5),
            sequence(0..3),
            &claimed[..2],
            check::DecodeEdge::default(),
        );
        let edge = check_range(sequence(5..6), sequence(3..5), &claimed[2..], first.clone());
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
    assert_eq!(
        late_edge.first_claim_mismatch_position,
        prompt_len as u32 + 2
    );

    // An honest claim publishes a clean verdict and direct generation's edge.
    let honest = prior
        .generated_token_ids
        .iter()
        .copied()
        .collect::<Vec<_>>();
    // The check derives its claim from the actual consumed tokens. There is
    // no separate argument through which a caller can substitute greedy IDs.
    let mut consumed = sequence(3..5);
    consumed.rows = List::from(
        consumed
            .rows
            .iter()
            .zip(&claimed[..2])
            .map(|(row, id)| check::ActivationRow {
                token_id: *id,
                values: row.values.clone(),
            })
            .collect::<Vec<_>>(),
    );
    let substituted =
        routines::sequence_check::run_direct(&routines::sequence_check::Inputs::eager(
            consumed,
            sequence(0..3),
            check::DecodeEdge::default(),
            check_head.clone(),
            context.clone(),
        ))
        .unwrap();
    assert_eq!(substituted.claim_mismatches, 1);

    let (_, honest_edge) = run_check(&honest);
    assert_eq!(honest_edge.claim_mismatches, 0);
    assert_eq!(honest_edge.first_claim_mismatch_position, 0);
    // This numerical oracle deliberately has no model descriptor.
    assert_ne!(honest_edge.model.fault, 0);
    let mut actual = serde_json::to_value(&honest_edge).unwrap();
    let mut expected = serde_json::to_value(&prior).unwrap();
    actual.as_object_mut().unwrap().remove("model");
    expected.as_object_mut().unwrap().remove("model");
    assert_eq!(actual, expected);

    assert_eq!(edge.has_selected, prior.has_selected);
    assert_eq!(edge.decode_position, prior.decode_position);
    assert_eq!(edge.token_id, prior.token_id);
    assert_eq!(edge.value, prior.value);
    assert_eq!(
        edge.generated_token_ids.iter().collect::<Vec<_>>(),
        prior.generated_token_ids.iter().collect::<Vec<_>>()
    );

    // A check that cannot stand behind its edge finishes and says why.
    let start = check::DecodeEdge::default();
    let empty = |start_position| check::ActivationSequence {
        model: Default::default(),
        rows: List::new(),
        errors: List::new(),
        kv: List::new(),
        start_position,
    };
    let fault = |edge: &check::DecodeEdge| (edge.check_fault, edge.check_fault_position);

    // No rows before the range: the first token has nothing to predict it, and
    // the rest of the range is still scored.
    let unscored = check_range(sequence(3..5), empty(3), &honest[..2], start.clone());
    assert_eq!(fault(&unscored), (check::CHECK_FAULT_UNSCORED, 3));
    assert_eq!(unscored.generated_token_ids.len(), 1);

    let nothing = check_range(empty(3), sequence(0..3), &[], start.clone());
    assert_eq!(fault(&nothing), (check::CHECK_FAULT_NO_CLAIMED_TOKENS, 3));
    assert!(!nothing.has_selected);

    let too_large = check_range(sequence(3..8), sequence(0..3), &[0; 5], start.clone());
    assert_eq!(fault(&too_large), (check::CHECK_FAULT_RANGE_TOO_LARGE, 3));

    let late_start = check_range(sequence(4..6), sequence(0..3), &honest[..2], start.clone());
    assert_eq!(fault(&late_start), (check::CHECK_FAULT_DISCONTINUOUS, 4));

    let (first, _) = run_check(&honest);
    let skipped = check_range(sequence(6..7), sequence(3..5), &honest[2..], first);
    assert_eq!(fault(&skipped), (check::CHECK_FAULT_DISCONTINUOUS, 6));

    // The earliest fault stands, and a verdict is still a verdict: a fault is
    // not a disagreement with the claim.
    let carried = check_range(sequence(5..6), sequence(3..5), &honest[2..], unscored);
    assert_eq!(fault(&carried), (check::CHECK_FAULT_UNSCORED, 3));
    let (_, wrong) = run_check(&claimed);
    assert_eq!(fault(&wrong), (check::CHECK_FAULT_NONE, 0));
    assert_eq!(wrong.claim_mismatches, 1);
}

fn authenticate_decoder_inputs(
    mut inputs: routines::output_finalize::Inputs,
) -> routines::output_finalize::Inputs {
    let d = model_identity::ModelDescriptor {
        version: 1,
        tokenizer: [1; 32],
        decoder: raster::core::tree::value_root(&inputs.decoder).unwrap(),
        embedding: [2; 32],
        head: [3; 32],
        auxiliary: vec![[4; 32]],
        transformer: vec![[5; 32]],
    };
    inputs.context.model_id = hex::encode(d.model_id().unwrap());
    inputs.descriptor = raster::BytesPage::__from_parts(0, 0, d.encode().unwrap());
    inputs.context.model =
        model_identity::ModelContext::bind(inputs.descriptor.as_slice(), &inputs.context.model_id);
    inputs.context.input_text_sha256 = "11".repeat(32);
    inputs.context.range_size = inputs.context.range_size.max(4);
    inputs.context.prompt_tokens = inputs.context.prompt_tokens.max(1);
    inputs.context.model.bind_statement(
        &inputs.context.input_text_sha256,
        inputs.context.raw_prompt,
        inputs.context.max_new_tokens,
        inputs.context.range_size,
    );
    // One merge batch per prompt range, and the claimed ids in whole ranges.
    let schedule = model_identity::pipeline_core::StageSchedule {
        prompt_ranges: inputs
            .context
            .prompt_tokens
            .div_ceil(inputs.context.range_size),
        merge_batches: 1,
        claimed_ranges: (inputs.edge.generated_token_ids.len() as u32)
            .div_ceil(inputs.context.range_size)
            .max(1),
    };
    inputs.context.model.bind_schedule(schedule);
    for start in (0..inputs.context.prompt_tokens).step_by(inputs.context.range_size as usize) {
        inputs.context.model.begin_prompt_range();
        inputs.context.model.continue_prompt_range();
        let end = (start + inputs.context.range_size).min(inputs.context.prompt_tokens);
        for _ in start..end {
            inputs.context.model.append_consumed_token(0);
        }
        inputs
            .context
            .model
            .finish_prompt_range(end == inputs.context.prompt_tokens, end, 0);
    }
    inputs.edge.model = inputs.context.model.clone();
    let mut e = inputs.edge.model.execution_context().unwrap();
    e.before = e.end;
    for &id in inputs.edge.generated_token_ids.iter() {
        e.end.append(id).unwrap();
    }
    e.claimed = true;
    e.phase = model_identity::pipeline_core::CHECKED;
    e.step = model_identity::pipeline_core::Step::check(schedule.claimed_ranges - 1);
    inputs.edge.decode_position = e.end.position.saturating_sub(1);
    inputs.edge.token_id = inputs.edge.generated_token_ids.last().copied().unwrap_or(0);
    inputs.edge.model.set_execution(&e);
    inputs.edge.model.producer = model_identity::context::DECODE;
    inputs.edge.model.completed_layers = inputs.edge.model.layers;
    inputs.witness = output_finalize::decoder_witness(&inputs.decoder).unwrap();
    inputs
}

#[test]
fn decoder_substitution_publishes_sticky_fault_and_preserves_statement_identity() {
    let token = DecoderToken {
        token: "hello".into(),
        special: false,
        terminal: false,
    };
    let mut inputs = authenticate_decoder_inputs(routines::output_finalize::Inputs {
        edge: output_finalize::input::DecodeEdge {
            has_selected: true,
            generated_token_ids: vec![0, 0].into(),
            ..Default::default()
        },
        decoder: DecoderTable {
            tokens: vec![token].into(),
        },
        context: output_finalize::prompt::PromptContext {
            max_new_tokens: 2,
            complete: true,
            ..Default::default()
        },
        descriptor: raster::BytesPage::__from_parts(0, 0, vec![]),
        witness: output_finalize::input::DecoderWitness {
            token_count: 0,
            paths: List::new(),
        },
    });
    let id = inputs.context.model_id.clone();
    assert_eq!(
        routines::output_finalize::run_direct(&inputs)
            .unwrap()
            .model_fault,
        0
    );
    inputs.decoder = DecoderTable {
        tokens: vec![DecoderToken {
            token: "substituted".into(),
            special: false,
            terminal: true,
        }]
        .into(),
    };
    let result = routines::output_finalize::run_direct(&inputs).unwrap();
    assert_eq!(result.model_fault, 1);
    assert_eq!(result.model_id, id);
    assert!(result.generated_text.is_empty());
    assert_eq!(result.generated_token_ids.as_slice(), &[0, 0]);
    // An honestly rebuilt witness for the wrong data still fails the old model.
    inputs.context.model =
        model_identity::ModelContext::bind(inputs.descriptor.as_slice(), &inputs.context.model_id);
    inputs.edge.model = inputs.context.model.clone();
    inputs.edge.model.producer = model_identity::context::DECODE;
    inputs.edge.model.completed_layers = inputs.edge.model.layers;
    inputs.witness = output_finalize::decoder_witness(&inputs.decoder).unwrap();
    assert_eq!(
        routines::output_finalize::run_direct(&inputs)
            .unwrap()
            .model_fault,
        1
    );
}
