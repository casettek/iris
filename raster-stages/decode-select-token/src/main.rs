//! Phase 6 — `decode_select_token`: sequences.
//!
//! ```text
//!   logits ──recur (chunk = 256)──▶ running argmax
//!   prior.generated_token_ids ──recur──▶ copy · append selected ──▶ DecodeEdge
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

    let best = call_recur!(
        tile = scan_logit_chunk,
        input = scores,
        chunk = 256,
        state = ArgmaxState {
            has_value: false,
            token_id: 0,
            value: 0
        },
        args = ()
    );

    let selected = call!(finish_selection, best, decode_position)?;
    let draft = call!(
        begin_decode_edge,
        new!(DecodeEdge),
        clone!(selected)
    );
    let prior_ids = select!(List<u32>, prior.generated_token_ids);
    let draft = call_recur!(
        tile = copy_generated_token_ids,
        input = prior_ids,
        chunk = 64,
        output = draft,
        finalize = false,
        args = ()
    );
    let draft = call!(append_selected_token, draft, selected);
    let edge = finalize(draft);
    Ok(edge)
}
