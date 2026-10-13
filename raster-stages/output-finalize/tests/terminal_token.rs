//! The terminal-token path.
//!
//! The chain's decode loop is a static unroll, so this branch is the only thing
//! that distinguishes "the model finished its turn" from "the repeat count ran
//! out" — and a chain run long enough to make a real model emit `<turn|>` costs
//! hours of stage execution.

use output_finalize::input::DecoderToken;
use output_finalize::{advance_finalize_state, initial_finalize_state};

fn word(text: &str) -> DecoderToken {
    DecoderToken {
        token: text.to_string(),
        special: false,
        terminal: false,
    }
}

fn end_of_turn() -> DecoderToken {
    DecoderToken {
        token: "<turn|>".to_string(),
        special: true,
        terminal: true,
    }
}

#[test]
fn text_ends_at_the_terminal_token_and_the_transcript_does_not() {
    let mut state = initial_finalize_state();
    state = advance_finalize_state(state, 229361, word("ZK"));
    state = advance_finalize_state(state, 106, end_of_turn());
    state = advance_finalize_state(state, 105, word("more"));

    assert!(state.stopped);
    // The answer stops where the model stopped …
    assert_eq!(state.text, "ZK");
    // … while every generated id stays in the count and the committed stream.
    assert_eq!(state.count, 3);
    assert_eq!(state.json, "[229361,106,105");
}

#[test]
fn a_run_with_no_terminal_token_stays_open() {
    let mut state = initial_finalize_state();
    state = advance_finalize_state(state, 229361, word("ZK"));

    assert!(!state.stopped);
    assert_eq!(state.text, "ZK");
}
