use anyhow::{bail, Result};
use output_finalize::input::{DecodeEdge, DecoderTable, GeneratedOutput};
use raster::{BytesPage, List};
use sha2::{Digest, Sha256};

pub struct OutputFinalizeDirectInputs<'a> {
    pub edge: &'a DecodeEdge,
    pub decoder: &'a DecoderTable,
}

pub fn run_output_finalize_direct(
    inputs: OutputFinalizeDirectInputs<'_>,
) -> Result<GeneratedOutput> {
    let mut state = output_finalize::initial_finalize_state();
    for token_id in inputs.edge.generated_token_ids.iter().copied() {
        let token = inputs
            .decoder
            .tokens
            .get(token_id as usize)
            .ok_or_else(|| anyhow::anyhow!("decoder table has no token id {token_id}"))?;
        state = output_finalize::advance_finalize_state(state, token_id, token.clone());
    }
    if !state.pending_bytes.as_slice().is_empty() {
        state
            .text
            .push_str(&String::from_utf8_lossy(state.pending_bytes.as_slice()));
        state.pending_bytes = BytesPage::__from_parts(0, 0, Vec::new());
    }
    state.json.push(']');

    if inputs.edge.has_selected != (state.count > 0) {
        bail!(
            "decode edge selection flag {} disagrees with generated token count {}",
            inputs.edge.has_selected,
            state.count
        );
    }

    let digest = Sha256::digest(state.json.as_bytes());
    Ok(GeneratedOutput {
        generated_token_count: state.count,
        generated_token_ids: List::from(
            inputs
                .edge
                .generated_token_ids
                .iter()
                .copied()
                .collect::<Vec<_>>(),
        ),
        generated_token_ids_sha256: format!("{digest:x}"),
        generated_text: state.text,
        stop_reason: if state.stopped {
            String::from("eos")
        } else {
            String::from("max_new_tokens")
        },
    })
}
