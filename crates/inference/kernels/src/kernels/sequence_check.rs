use anyhow::{ensure, Result};
use raster::List;
use sequence_check::input::{
    empty_row, rms_norm, softcap, unpack_i32s, ActivationRow, ActivationSequence, CheckFrame,
    ClaimedOutput, DecodeEdge, ErrorSummary, FinalHead, PositionSelection, SelectionSummary,
};
use sequence_check::prompt::PromptContext;
use sequence_check::{check_fault_of, observe_selection};

use crate::tensor::{MatrixSource, Q8Matrix};

/// The final-norm and softcap parameters every check position shares.
pub struct CheckHeadParams<'a> {
    pub hidden_size: u32,
    pub norm_eps: i64,
    pub softcap: i32,
    pub norm_weights: &'a [i32],
}

/// One generated position: the normalised row that predicts it, or `None`
/// when there is no row to normalise, which the check publishes as a fault.
pub struct CheckPosition {
    pub decode_position: u32,
    pub claimed_token_id: u32,
    pub normalized: Option<Vec<i32>>,
}

/// This stage's copy of the prompt context: the stages share its source, not a
/// Rust type.
pub fn prompt_context(source: &prompt_prepare::input::PromptContext) -> PromptContext {
    PromptContext {
        input_text_sha256: source.input_text_sha256.clone(),
        model_id: source.model_id.clone(),
        raw_prompt: source.raw_prompt,
        max_new_tokens: source.max_new_tokens,
        range_size: source.range_size,
        next_piece: source.next_piece,
        prompt_tokens: source.prompt_tokens,
        complete: source.complete,
        fault: source.fault,
        fault_position: source.fault_position,
    }
}

/// The greedy selection at one generated position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Selection {
    pub token_id: u32,
    pub value: i32,
}

/// Running first-maximum argmax: only a strictly greater score replaces the
/// incumbent, so visiting token ids in order keeps the first maximum.
#[derive(Debug, Clone, Copy, Default)]
pub struct Argmax(Option<Selection>);

impl Argmax {
    pub fn observe(&mut self, token_id: u32, value: i32) {
        match self.0 {
            Some(best) if value <= best.value => {}
            _ => self.0 = Some(Selection { token_id, value }),
        }
    }

    pub fn finish(self) -> Option<Selection> {
        self.0
    }
}

/// Scores one row of the projection exactly as the head page tiles do.
/// One vocabulary entry's softcapped score against a `Q8_0` head row.
pub fn score_row(normalized: &[i32], row: &[u8], cap: i32) -> Result<i32> {
    Ok(softcap(
        det_num::q8::dot_row(normalized, row).map_err(anyhow::Error::msg)?,
        cap,
    ))
}

/// Locates and normalises the row predicting each claimed token of a range.
///
/// The predicting rows are the previous range's last row followed by this
/// range's rows: claimed token `k` sits at `start_position + k` and is
/// predicted by the row before it. Causal attention means that row has seen
/// exactly the tokens a decode step of direct generation saw.
///
/// A previous range with no rows leaves the stage's empty row at the boundary.
pub fn check_positions(
    activations: &ActivationSequence,
    previous: &ActivationSequence,
    claimed: &ClaimedOutput,
    params: &CheckHeadParams<'_>,
) -> Result<Vec<CheckPosition>> {
    let empty = empty_row();
    let boundary = previous.rows.iter().last().unwrap_or(&empty);
    let predicting = std::iter::once(boundary)
        .chain(activations.rows.iter())
        .collect::<Vec<_>>();
    claimed
        .token_ids
        .iter()
        .enumerate()
        .map(|(k, claimed_token_id)| {
            let row = predicting.get(k).ok_or_else(|| {
                anyhow::anyhow!(
                    "claimed token {k} has no predicting row: the range has {} rows",
                    activations.rows.len()
                )
            })?;
            Ok(CheckPosition {
                decode_position: activations.start_position + k as u32,
                claimed_token_id: *claimed_token_id,
                normalized: normalize_row(row, params),
            })
        })
        .collect()
}

/// `normalize_position`, which records its failure where this returns `None`.
fn normalize_row(row: &ActivationRow, params: &CheckHeadParams<'_>) -> Option<Vec<i32>> {
    let mut values = unpack_i32s(&row.values).ok()?;
    if values.len() != params.hidden_size as usize {
        return None;
    }
    rms_norm(&mut values, params.norm_weights, params.norm_eps).ok()?;
    Some(values)
}

/// What this range's check establishes its fault from.
pub fn check_frame(
    activations: &ActivationSequence,
    prior: &DecodeEdge,
    context: &PromptContext,
) -> CheckFrame {
    CheckFrame {
        start_position: activations.start_position,
        range_size: context.range_size,
        prompt_tokens: context.prompt_tokens,
        prior_has_selected: prior.has_selected,
        prior_decode_position: prior.decode_position,
        prior_fault: prior.check_fault,
        prior_fault_position: prior.check_fault_position,
    }
}

/// Publishes the selections in the `DecodeEdge` shape `output-finalize`
/// consumes, after the earlier ranges' transcript. `selections` holds one entry
/// per position that has a normalised row.
///
/// Neither a disagreement with the claim nor a fault is raised. The verdict is
/// the earlier ranges' count plus this range's, keeping the earliest position,
/// and the fault is the earliest range's.
pub fn decode_edge(
    positions: &[CheckPosition],
    selections: &[Selection],
    prior: &DecodeEdge,
    frame: &CheckFrame,
) -> Result<DecodeEdge> {
    let mut remaining = selections.iter();
    let mut summary = SelectionSummary::default();
    let mut errors = ErrorSummary::default();
    let mut generated_token_ids = prior
        .generated_token_ids
        .iter()
        .copied()
        .collect::<Vec<_>>();
    for position in positions {
        if position.normalized.is_none() {
            if errors.count == 0 {
                errors.first_position = position.decode_position;
            }
            errors.count += 1;
            continue;
        }
        let selection = remaining
            .next()
            .ok_or_else(|| anyhow::anyhow!("a scored check position has no selection"))?;
        observe_selection(
            &mut summary,
            &PositionSelection {
                decode_position: position.decode_position,
                token_id: selection.token_id,
                value: selection.value,
                claimed_token_id: position.claimed_token_id,
            },
        );
        generated_token_ids.push(selection.token_id);
    }
    ensure!(
        remaining.next().is_none(),
        "more selections than scored check positions"
    );
    let fault = check_fault_of(&summary, &errors, frame);
    Ok(DecodeEdge {
        has_selected: summary.has_selected,
        decode_position: summary.decode_position,
        token_id: summary.token_id,
        value: summary.value,
        claim_mismatches: prior.claim_mismatches.saturating_add(summary.mismatches),
        first_claim_mismatch_position: if prior.claim_mismatches > 0 {
            prior.first_claim_mismatch_position
        } else {
            summary.first_mismatch_position
        },
        check_fault: fault.fault,
        check_fault_position: fault.position,
        generated_token_ids: List::from(generated_token_ids),
    })
}

/// The positions that have a normalised row, in order.
pub fn scorable(positions: &[CheckPosition]) -> Vec<&[i32]> {
    positions
        .iter()
        .filter_map(|position| position.normalized.as_deref())
        .collect()
}

pub struct SequenceCheckDirectInputs<'a> {
    pub activations: &'a ActivationSequence,
    pub previous: &'a ActivationSequence,
    pub claimed: &'a ClaimedOutput,
    pub prior: &'a DecodeEdge,
    pub head: &'a FinalHead,
    pub context: &'a PromptContext,
}

/// Already-materialized comparison path over an owned head.
pub fn run_sequence_check_direct(inputs: SequenceCheckDirectInputs<'_>) -> Result<DecodeEdge> {
    let head = &inputs.head.params;
    ensure!(
        head.hidden_size > 0,
        "output head requires a non-zero hidden_size"
    );
    ensure!(head.norm_eps >= 0, "rms-norm epsilon must be non-negative");
    ensure!(
        head.softcap >= 0,
        "final logit softcap must be non-negative (0 disables it)"
    );
    let norm_weights = unpack_i32s(&head.norm_weights).map_err(anyhow::Error::msg)?;
    ensure!(
        norm_weights.len() == head.hidden_size as usize,
        "final norm weights width mismatch"
    );
    let params = CheckHeadParams {
        hidden_size: head.hidden_size,
        norm_eps: head.norm_eps,
        softcap: head.softcap,
        norm_weights: &norm_weights,
    };
    let positions = check_positions(inputs.activations, inputs.previous, inputs.claimed, &params)?;

    let stride = det_num::q8::row_bytes(head.hidden_size as usize).unwrap_or(0) as u64;
    let byte_len = inputs.head.projection.byte_len();
    ensure!(
        stride != 0 && byte_len % stride == 0,
        "projection has {byte_len} bytes, expected a whole number of hidden-width Q8_0 rows"
    );
    let projection = Q8Matrix::from_region(
        "projection",
        &inputs.head.projection,
        (byte_len / stride) as usize,
        head.hidden_size as usize,
    )?;
    let normalized = scorable(&positions);
    let mut argmax = vec![Argmax::default(); normalized.len()];
    for token_id in 0..projection.rows() {
        let row = projection.row(token_id);
        for (position, best) in normalized.iter().zip(argmax.iter_mut()) {
            best.observe(token_id as u32, score_row(position, row, head.softcap)?);
        }
    }
    let selections = argmax
        .into_iter()
        .map(|best| {
            best.finish()
                .ok_or_else(|| anyhow::anyhow!("the head scored no token"))
        })
        .collect::<Result<Vec<_>>>()?;
    decode_edge(
        &positions,
        &selections,
        inputs.prior,
        &check_frame(inputs.activations, inputs.prior, inputs.context),
    )
}
