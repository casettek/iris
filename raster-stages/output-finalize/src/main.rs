//! Generated output finalization sequences.

use raster::prelude::*;

use output_finalize::input::*;
use output_finalize::*;

/// Resolves one generated token by its authorized ID and advances the bounded
/// detokenization/commitment state.
#[sequence(kind = recur)]
fn decode_generated_token(
    input: RecurSequenceInput<u32>,
    state: RecurSequenceState<FinalizeState>,
    tokens: List<DecoderToken>,
) -> RecurSequenceState<FinalizeState> {
    let token_id = into_ref!(input);
    let token = select!(DecoderToken, tokens[token_id]);
    call!(consume_decoder_token, state, token_id, token)
}

#[sequence]
fn main(edge: DecodeEdge, decoder: DecoderTable) -> Result<GeneratedOutput> {
    let has_selected = select!(bool, edge.clone().has_selected);
    let ids = select!(List<u32>, edge.generated_token_ids);
    let ids_for_decode = clone!(ids);
    let tokens = select!(List<DecoderToken>, decoder.tokens);

    let state = call!(begin_finalize_state);
    let state = call_recur_seq!(
        sequence = decode_generated_token,
        input = ids_for_decode,
        state = state,
        args = (tokens,)
    );
    let state = call!(finish_finalize_state, state);
    let count = select!(u32, state.clone().count);
    call!(validate_decode_edge, has_selected, count)?;

    let draft = call!(begin_generated_output, new!(GeneratedOutput), state);
    let output = call_recur!(
        tile = copy_output_token_ids,
        input = ids,
        chunk = 64,
        output = draft,
        args = ()
    );
    raster::println!("generated output → {:?}", output);
    Ok(output)
}
