use anyhow::{bail, Result};
use decode_select_token::input::{DecodeEdge, PrefillLogits};
use raster::List;

pub struct DecodeSelectTokenDirectInputs<'a> {
    pub logits: &'a PrefillLogits,
    pub prior: &'a DecodeEdge,
}

pub fn run_decode_select_token_direct(
    inputs: DecodeSelectTokenDirectInputs<'_>,
) -> Result<DecodeEdge> {
    let mut best_token = 0;
    let mut best_value = 0;
    let mut has_value = false;

    for entry in inputs.logits.logits.iter() {
        if !has_value || entry.value > best_value {
            has_value = true;
            best_token = entry.token_id;
            best_value = entry.value;
        }
    }

    if !has_value {
        bail!("output decode requires at least one logit to select the next token");
    }

    let mut generated_token_ids = inputs
        .prior
        .generated_token_ids
        .iter()
        .copied()
        .collect::<Vec<_>>();
    generated_token_ids.push(best_token);

    Ok(DecodeEdge {
        has_selected: true,
        decode_position: inputs.logits.decode_position,
        token_id: best_token,
        value: best_value,
        generated_token_ids: List::from(generated_token_ids),
    })
}
