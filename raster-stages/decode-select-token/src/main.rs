//! Phase 6 — `decode_select_token`: sequences.
//!
//! ```text
//!   logits ──recur (chunk = 256)──▶ running argmax
//!   prior.generated_token_ids ──recur──▶ copy ──▶ [selected] ──recur──▶ append ──▶ DecodeEdge
//! ```
//!
//! `chunk` is the only tuning knob here. The vocabulary is 262,144 entries, so
//! one replay unit per entry would be 262,144 of them to pick a maximum; a
//! `Block` of 256 makes it 1,024 units of ~2 KB each. The bound is a literal
//! pinned in the CFS, so coarsening this way costs nothing in what a verifier
//! can check.

use raster::prelude::*;

use decode_select_token::input::*;
use decode_select_token::*;

/// One decode iteration's selection entrypoint.
///
/// `logits` comes from prefill on iteration zero and from the prior transition
/// afterwards. `prior` comes from `decode-init` on iteration zero and from the
/// prior selection afterwards.
#[sequence]
fn main(logits: PrefillLogits, prior: DecodeEdge) -> Result<DecodeEdge> {
    let decode_position = select!(u32, logits.clone().decode_position);
    let scores = select!(List<LogitEntry>, logits.logits);

    let best_seed = call!(begin_argmax);
    let best = call_recur!(
        tile = scan_logit_chunk,
        input = scores,
        chunk = 256,
        state = best_seed,
        args = ()
    );

    let selected = call!(finish_selection, best, decode_position)?;
    let token_id = select!(u32, selected.clone().token_id);
    let base = call!(begin_decode_edge, selected);

    // The edge is built across two derived sites, each push-only: the prior
    // transcript first, then this stage's token. The first decode step's prior
    // (from `decode-init`) has an empty transcript, which the copy site skips.
    let prior_ids = select!(List<u32>, prior.generated_token_ids);
    let copied = call_recur!(
        tile = copy_generated_token_ids,
        input = prior_ids,
        chunk = 64,
        output = base,
        args = ()
    );
    let selected_ids = call!(selected_token_ids, token_id);
    let new_ids = select!(List<u32>, selected_ids.token_ids);
    let edge = call_recur!(
        tile = append_selected_token,
        input = new_ids,
        output = copied,
        args = ()
    );
    Ok(edge)
}
