//! Phase 5 — `prefill-finalize`: sequences.
//!
//! ```text
//!   rows       ──recur──▶ track final position ──▶ normalise
//!   projection.pages ──recur──▶ score page            ──▶ PrefillLogits
//!   errors     ──recur──▶ summarise ──▶ assert none
//! ```

use raster::prelude::*;

use prefill_finalize::input::*;
use prefill_finalize::*;

/// Phase 5 entrypoint.
///
/// `activations` is bound by the chain to the last transformer layer's
/// authorized output; `head` is the committed output head.
#[sequence]
fn main(activations: ActivationSequence, head: FinalHead) -> Result<PrefillLogits> {
    let params = select!(FinalHeadParams, head.clone().params);
    let params = call!(validate_final_head, params)?;

    let start_position = select!(u32, activations.clone().start_position);
    let rows = select!(List<ActivationRow>, activations.rows);
    let empty = call!(empty_final_position);
    let position = call_recur!(
        tile = track_final_position,
        input = rows,
        state = empty,
        args = ()
    );

    let normalized = call!(normalize_final_position, position.clone(), params.clone())?;

    let token_count = select!(u32, position.token_count);
    let draft = call!(begin_logits, token_count, start_position);

    let pages = select!(List<BytesPage>, head.projection.pages);
    let logits = call_recur!(
        tile = project_logit_page,
        input = pages,
        output = draft,
        args = (normalized, params)
    );
    raster::println!("prefill finalize → {:?}", logits);

    let errors = select!(List<String>, logits.clone().errors);
    let summary_seed = call!(begin_error_summary);
    let summary = call_recur!(
        tile = summarise_errors,
        input = errors,
        state = summary_seed,
        args = ()
    );
    let error_count = select!(u32, summary.clone().count);
    let first_error = select!(String, summary.first);
    call!(assert_logits_complete, error_count, first_error)?;

    Ok(logits)
}
