use anyhow::{bail, ensure, Result};
use raster::List;
use sequence_check::input::{
    dot, rms_norm, softcap, unpack_i32s, ActivationSequence, ClaimedOutput, DecodeEdge, FinalHead,
    PromptTokenization,
};

use crate::tensor::Matrix;

/// The final-norm and softcap parameters every check position shares.
pub struct CheckHeadParams<'a> {
    pub hidden_size: u32,
    pub norm_eps: i64,
    pub softcap: i32,
    pub norm_weights: &'a [i32],
}

/// One generated position: the normalised row that predicts it.
pub struct CheckPosition {
    pub decode_position: u32,
    pub claimed_token_id: u32,
    pub normalized: Vec<i32>,
}

/// The greedy selection at one generated position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Selection {
    pub token_id: u32,
    pub value: i32,
}

/// Running first-maximum argmax: only a strictly greater score replaces the
/// incumbent, so visiting token ids in order reproduces `decode-select-token`.
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
pub fn score_row(normalized: &[i32], row: &[i32], cap: i32) -> Result<i32> {
    Ok(softcap(
        dot(normalized, row).map_err(anyhow::Error::msg)?,
        cap,
    ))
}

/// Locates and normalises the row predicting each claimed token.
///
/// Generated token `k` is predicted by row `prompt_len - 1 + k`: causal
/// attention means that row has seen exactly the prompt and the first `k`
/// claimed tokens, the same inputs the autoregressive decode step saw.
pub fn check_positions(
    activations: &ActivationSequence,
    prompt: &PromptTokenization,
    claimed: &ClaimedOutput,
    params: &CheckHeadParams<'_>,
) -> Result<Vec<CheckPosition>> {
    let prompt_len = prompt.token_ids.len();
    if prompt_len == 0 {
        bail!("teacher-forced check requires a non-empty prompt to predict the first token");
    }
    if claimed.token_ids.is_empty() {
        bail!("teacher-forced check requires at least one claimed output token");
    }
    let rows = activations.rows.iter().collect::<Vec<_>>();
    claimed
        .token_ids
        .iter()
        .enumerate()
        .map(|(k, claimed_token_id)| {
            let row_index = prompt_len - 1 + k;
            let row = rows.get(row_index).ok_or_else(|| {
                anyhow::anyhow!(
                    "claimed token {k} is predicted by row {row_index}, but the pass has {} rows",
                    rows.len()
                )
            })?;
            let mut normalized = unpack_i32s(&row.values).map_err(anyhow::Error::msg)?;
            ensure!(
                normalized.len() == params.hidden_size as usize,
                "predicting row has {} values, expected hidden_size {}",
                normalized.len(),
                params.hidden_size
            );
            rms_norm(&mut normalized, params.norm_weights, params.norm_eps)
                .map_err(anyhow::Error::msg)?;
            Ok(CheckPosition {
                decode_position: activations.start_position + row_index as u32 + 1,
                claimed_token_id: *claimed_token_id,
                normalized,
            })
        })
        .collect()
}

/// Publishes the selections in the `DecodeEdge` shape `output-finalize`
/// consumes. Disagreement with the claim is published, not raised.
pub fn decode_edge(positions: &[CheckPosition], selections: &[Selection]) -> Result<DecodeEdge> {
    ensure!(
        positions.len() == selections.len(),
        "{} check positions produced {} selections",
        positions.len(),
        selections.len()
    );
    let (last_position, last) = positions.last().zip(selections.last()).ok_or_else(|| {
        anyhow::anyhow!("teacher-forced check requires at least one claimed output token")
    })?;
    Ok(DecodeEdge {
        has_selected: true,
        decode_position: last_position.decode_position,
        token_id: last.token_id,
        value: last.value,
        generated_token_ids: List::from(
            selections
                .iter()
                .map(|selection| selection.token_id)
                .collect::<Vec<_>>(),
        ),
    })
}

/// The first generated position whose selection is not the claimed token.
pub fn first_mismatch(positions: &[CheckPosition], selections: &[Selection]) -> Option<u32> {
    positions
        .iter()
        .zip(selections)
        .find(|(position, selection)| position.claimed_token_id != selection.token_id)
        .map(|(position, _)| position.decode_position)
}

pub struct SequenceCheckDirectInputs<'a> {
    pub activations: &'a ActivationSequence,
    pub prompt: &'a PromptTokenization,
    pub claimed: &'a ClaimedOutput,
    pub head: &'a FinalHead,
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
    let positions = check_positions(inputs.activations, inputs.prompt, inputs.claimed, &params)?;

    let stride = u64::from(head.hidden_size) * 4;
    let byte_len = inputs.head.projection.byte_len();
    ensure!(
        byte_len % stride == 0,
        "projection has {byte_len} bytes, expected a whole number of hidden-width rows"
    );
    let projection = Matrix::from_region(
        "projection",
        &inputs.head.projection,
        (byte_len / stride) as usize,
        head.hidden_size as usize,
    )?;
    let mut argmax = vec![Argmax::default(); positions.len()];
    for token_id in 0..projection.rows() {
        let row = projection.row(token_id);
        for (position, best) in positions.iter().zip(argmax.iter_mut()) {
            best.observe(
                token_id as u32,
                score_row(&position.normalized, row, head.softcap)?,
            );
        }
    }
    let selections = argmax
        .into_iter()
        .map(|best| {
            best.finish()
                .ok_or_else(|| anyhow::anyhow!("the head scored no token"))
        })
        .collect::<Result<Vec<_>>>()?;
    decode_edge(&positions, &selections)
}
