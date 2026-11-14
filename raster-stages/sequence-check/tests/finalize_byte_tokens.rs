//! Decoding a byte-level tokenizer's output.
//!
//! Each token is bytes, and one character of the text can be split between
//! tokens. The text must be what decoding all the bytes at once gives.

use sequence_check::input::DecoderToken;
use sequence_check::prompt::tekken::byte_char;
use sequence_check::{advance_finalize_state, initial_finalize_state};

fn bytes(bytes: &[u8]) -> DecoderToken {
    DecoderToken {
        token: bytes.iter().map(|byte| byte_char(*byte)).collect(),
        special: false,
        terminal: false,
        bytes: true,
    }
}

fn end_of_text() -> DecoderToken {
    DecoderToken {
        token: "</s>".to_string(),
        special: true,
        terminal: true,
        bytes: false,
    }
}

#[test]
fn a_character_split_between_tokens_is_one_character() {
    let mut state = initial_finalize_state();
    state = advance_finalize_state(state, 1, bytes(b"caf\xC3"));
    assert_eq!(state.text, "caf");
    assert_eq!(state.pending_bytes.as_slice(), b"\xC3");
    state = advance_finalize_state(state, 2, bytes(b"\xA9 "));
    assert_eq!(state.text, "café ");
    assert!(state.pending_bytes.as_slice().is_empty());
}

#[test]
fn the_text_is_the_lossy_decoding_of_all_the_bytes() {
    let outputs: [&[u8]; 7] = [
        b"h\xC3\xA9 \xF0\x9F\x99\x82!",
        b"\xE4\xB8",
        b"\xE4\xB8a",
        b"\xC3\x28",
        b"a\xF0\x9F\x99",
        b"\x80\x80a\xFF",
        b"\xF0\x9F\x28\xED\xA0\x80",
    ];
    for output in outputs {
        let expected = String::from_utf8_lossy(output);
        for cut in 0..=output.len() {
            for second in cut..=output.len() {
                let mut state = initial_finalize_state();
                state = advance_finalize_state(state, 1, bytes(&output[..cut]));
                state = advance_finalize_state(state, 2, bytes(&output[cut..second]));
                state = advance_finalize_state(state, 3, bytes(&output[second..]));
                state = advance_finalize_state(state, 4, end_of_text());
                assert_eq!(state.text, expected, "{output:?} cut at {cut} and {second}");
                assert!(state.pending_bytes.as_slice().is_empty());
            }
        }
    }
}

#[test]
fn a_token_of_another_kind_is_read_as_before() {
    let mut state = initial_finalize_state();
    let word = DecoderToken {
        token: "\u{2581}ZK".to_string(),
        special: false,
        terminal: false,
        bytes: false,
    };
    state = advance_finalize_state(state, 1, word);
    assert_eq!(state.text, " ZK");
}
