//! The word scanner: the split of Mistral's "tekken" tokenizer, taken one
//! chunk of text at a time.
//!
//! ```text
//!   text ─ special tokens out ─▶ plain text ─ split ─▶ words ─ bytes ─▶ atoms
//! ```
//!
//! Plain text is cut into words by one pattern over Unicode categories:
//!
//! ```text
//!   lead? capitals* lower+      a word, with at most one symbol or blank before it
//!   lead? capitals+ lower*
//!   one digit
//!   space? symbols+ (break | /)*
//!   blanks* breaks+
//!   blanks+ not before text     so the last blank before text is that word's lead
//!   blanks+
//! ```
//!
//! The first alternative that matches at a place is the word. A letter with no
//! case and a combining mark count as both a capital and a lower-case letter.
//!
//! A stage cannot hold a whole text, so the pattern is not run as a pattern.
//! [`feed`] takes one character and keeps a place in the list above, the bytes
//! of the word so far, and at most [`HOLD_BYTES`] it cannot place yet. Run
//! preparation has the pattern as a backtracking matcher, and the two are
//! compared on every short string of character kinds.
//!
//! A word is one atom, written as the tokenizer writes bytes: one character a
//! byte. A word longer than the longest token is put out in parts.

use alloc::string::{String, ToString};

use super::{
    specials_of, BpePiece, PromptAtom, ScanState, SplitState, TokenizerProfile, ATOM_FIRST_PART,
    ATOM_LATER_PART, ATOM_WHOLE,
};

#[path = "unicode_classes.rs"]
pub mod unicode_classes;

use unicode_classes::{LOWER, MARK, MODIFIER, NUMBER, OTHER, OTHER_LETTER, RANGES, TITLE, UPPER};

/// The most bytes the scanner keeps while it cannot tell which word they are
/// in. Two things can run that long: blanks after a line break, which belong
/// to the break's word only if another break follows, and capitals after a
/// letter with no case, which belong to its word only if such a letter or a
/// lower-case one follows. Past this the scanner places them as if the text
/// had ended, and says so in [`ScanState::overflowed`].
pub const HOLD_BYTES: u32 = 1024;

/// [`ScanState::mode`]: between words.
pub const AT_START: u32 = 0;
/// In a word's lower-case letters.
pub const IN_LOWER: u32 = 1;
/// In a word that is capitals so far.
pub const IN_CAPITALS: u32 = 2;
/// In a word's capitals after a letter with no case. Capitals since the last
/// such letter are held.
pub const IN_MIXED: u32 = 3;
/// One symbol is held: it leads a word, or starts a run of symbols.
pub const AT_LEAD: u32 = 4;
/// In a run of symbols.
pub const IN_SYMBOLS: u32 = 5;
/// In the breaks and slashes that end a run of symbols.
pub const IN_TAIL: u32 = 6;
/// In blanks with no line break yet. The last one is held.
pub const IN_BLANKS: u32 = 7;
/// In blanks after a line break. Those since the last break are held.
pub const AFTER_BREAK: u32 = 8;

/// The character a byte is written as: itself where that prints, and one of
/// `U+0100..` in order otherwise.
pub fn byte_char(byte: u8) -> char {
    let code = match byte {
        0x00..=0x20 => 0x100 + u32::from(byte),
        0x7F..=0xA0 => 0x121 + u32::from(byte - 0x7F),
        0xAD => 0x143,
        _ => u32::from(byte),
    };
    char::from_u32(code).expect("a code point below U+0200")
}

/// The byte a character is written for, when it is one of [`byte_char`]'s.
pub fn char_byte(ch: char) -> Option<u8> {
    let code = ch as u32;
    let byte = match code {
        0x100..=0x120 => code - 0x100,
        0x121..=0x142 => code - 0x121 + 0x7F,
        0x143 => 0xAD,
        0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF => code,
        _ => return None,
    };
    Some(byte as u8)
}

/// The character's general category, as one of the table's classes.
pub fn class_of(ch: char) -> u8 {
    let code = ch as u32;
    if code < 0x80 {
        return match ch {
            '0'..='9' => NUMBER,
            'A'..='Z' => UPPER,
            'a'..='z' => LOWER,
            _ => OTHER,
        };
    }
    let (mut low, mut high) = (0, RANGES.len());
    while low < high {
        let middle = (low + high) / 2;
        let (first, last, class) = RANGES[middle];
        if code < first {
            high = middle;
        } else if code > last {
            low = middle + 1;
        } else {
            return class;
        }
    }
    OTHER
}

/// The characters with the White_Space property.
pub fn is_space(ch: char) -> bool {
    matches!(
        ch,
        '\u{9}'..='\u{D}'
            | ' '
            | '\u{85}'
            | '\u{A0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200A}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202F}'
            | '\u{205F}'
            | '\u{3000}'
    )
}

/// What the pattern can tell apart in a character.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// An upper-case or title-case letter.
    Upper,
    Lower,
    /// A letter with no case: it continues capitals and lower case alike.
    Both,
    /// A combining mark: as `Both` in a word, and a symbol among symbols.
    Mark,
    Digit,
    /// `\r` or `\n`.
    Break,
    Space,
    /// Whitespace that is neither a break nor the space.
    Blank,
    Slash,
    Symbol,
}

fn kind_of(ch: char) -> Kind {
    match ch {
        '\r' | '\n' => return Kind::Break,
        ' ' => return Kind::Space,
        '/' => return Kind::Slash,
        _ => {}
    }
    if is_space(ch) {
        return Kind::Blank;
    }
    match class_of(ch) {
        UPPER | TITLE => Kind::Upper,
        LOWER => Kind::Lower,
        MODIFIER | OTHER_LETTER => Kind::Both,
        MARK => Kind::Mark,
        NUMBER => Kind::Digit,
        _ => Kind::Symbol,
    }
}

fn word_atom(raw: &str, part: u32) -> PromptAtom {
    PromptAtom {
        text: raw.bytes().map(byte_char).collect(),
        special: false,
        part,
    }
}

/// Puts out the start of a word that has outgrown one atom, a part at a time.
fn spill(state: &mut ScanState, limit: usize, emit: &mut impl FnMut(PromptAtom)) {
    while state.word as usize > limit {
        let mut cut = limit;
        while !state.pending.is_char_boundary(cut) {
            cut -= 1;
        }
        let part = if state.long {
            ATOM_LATER_PART
        } else {
            ATOM_FIRST_PART
        };
        emit(word_atom(&state.pending[..cut], part));
        state.pending.drain(..cut);
        state.word -= cut as u32;
        state.long = true;
    }
}

/// Adds what is held, and `len` more bytes, to the word.
fn take(state: &mut ScanState, len: usize, limit: usize, emit: &mut impl FnMut(PromptAtom)) {
    state.word += state.held + len as u32;
    state.held = 0;
    spill(state, limit, emit);
}

/// Ends the word. What is held stays, at the start of the pending text.
fn end_word(state: &mut ScanState, emit: &mut impl FnMut(PromptAtom)) {
    let end = state.word as usize;
    if end > 0 {
        let part = if state.long {
            ATOM_LATER_PART
        } else {
            ATOM_WHOLE
        };
        emit(word_atom(&state.pending[..end], part));
        state.pending.drain(..end);
    }
    state.word = 0;
    state.long = false;
}

/// Ends the word and makes what is held a word of its own.
fn end_both(state: &mut ScanState, limit: usize, emit: &mut impl FnMut(PromptAtom)) {
    end_word(state, emit);
    take(state, 0, limit, emit);
    end_word(state, emit);
}

/// Ends the text the scanner has been fed: the whole text, or the plain text
/// before a special token.
pub fn finish_words(state: &mut ScanState, limit: usize, emit: &mut impl FnMut(PromptAtom)) {
    match state.mode {
        // Held capitals, and held blanks after a break, are their own word.
        IN_MIXED | AFTER_BREAK => end_both(state, limit, emit),
        // A held symbol is a word, and a last blank ends its run.
        _ => {
            take(state, 0, limit, emit);
            end_word(state, emit);
        }
    }
    state.mode = AT_START;
}

/// Holds `len` more bytes. More than [`HOLD_BYTES`] cannot be kept.
fn hold(state: &mut ScanState, len: usize, limit: usize, emit: &mut impl FnMut(PromptAtom)) {
    state.held += len as u32;
    if state.held > HOLD_BYTES {
        state.overflowed = true;
        finish_words(state, limit, emit);
    }
}

/// Moves the scanner past one character of plain text.
pub fn feed(state: &mut ScanState, ch: char, limit: usize, emit: &mut impl FnMut(PromptAtom)) {
    let kind = kind_of(ch);
    let len = ch.len_utf8();
    loop {
        match state.mode {
            AT_START => match kind {
                Kind::Upper => {
                    take(state, len, limit, emit);
                    state.mode = IN_CAPITALS;
                }
                Kind::Lower => {
                    take(state, len, limit, emit);
                    state.mode = IN_LOWER;
                }
                Kind::Both | Kind::Mark => {
                    take(state, len, limit, emit);
                    state.mode = IN_MIXED;
                }
                Kind::Digit => {
                    take(state, len, limit, emit);
                    end_word(state, emit);
                }
                Kind::Break => {
                    take(state, len, limit, emit);
                    state.mode = AFTER_BREAK;
                }
                Kind::Space | Kind::Blank => {
                    state.mode = IN_BLANKS;
                    hold(state, len, limit, emit);
                }
                Kind::Slash | Kind::Symbol => {
                    state.mode = AT_LEAD;
                    hold(state, len, limit, emit);
                }
            },
            IN_LOWER => match kind {
                Kind::Lower | Kind::Both | Kind::Mark => take(state, len, limit, emit),
                _ => {
                    end_word(state, emit);
                    state.mode = AT_START;
                    continue;
                }
            },
            IN_CAPITALS => match kind {
                Kind::Upper => take(state, len, limit, emit),
                Kind::Lower => {
                    take(state, len, limit, emit);
                    state.mode = IN_LOWER;
                }
                Kind::Both | Kind::Mark => {
                    take(state, len, limit, emit);
                    state.mode = IN_MIXED;
                }
                _ => {
                    end_word(state, emit);
                    state.mode = AT_START;
                    continue;
                }
            },
            IN_MIXED => match kind {
                // In the word only if a letter that can end it follows.
                Kind::Upper => hold(state, len, limit, emit),
                Kind::Lower => {
                    take(state, len, limit, emit);
                    state.mode = IN_LOWER;
                }
                Kind::Both | Kind::Mark => take(state, len, limit, emit),
                _ => {
                    end_both(state, limit, emit);
                    state.mode = AT_START;
                    continue;
                }
            },
            AT_LEAD => match kind {
                Kind::Upper => {
                    take(state, len, limit, emit);
                    state.mode = IN_CAPITALS;
                }
                Kind::Lower => {
                    take(state, len, limit, emit);
                    state.mode = IN_LOWER;
                }
                Kind::Both | Kind::Mark => {
                    take(state, len, limit, emit);
                    state.mode = IN_MIXED;
                }
                _ => {
                    take(state, 0, limit, emit);
                    state.mode = IN_SYMBOLS;
                    continue;
                }
            },
            IN_SYMBOLS => match kind {
                Kind::Symbol | Kind::Slash | Kind::Mark => take(state, len, limit, emit),
                Kind::Break => {
                    take(state, len, limit, emit);
                    state.mode = IN_TAIL;
                }
                _ => {
                    end_word(state, emit);
                    state.mode = AT_START;
                    continue;
                }
            },
            IN_TAIL => match kind {
                Kind::Break | Kind::Slash => take(state, len, limit, emit),
                _ => {
                    end_word(state, emit);
                    state.mode = AT_START;
                    continue;
                }
            },
            IN_BLANKS => match kind {
                Kind::Space | Kind::Blank => {
                    take(state, 0, limit, emit);
                    hold(state, len, limit, emit);
                }
                Kind::Break => {
                    take(state, len, limit, emit);
                    state.mode = AFTER_BREAK;
                }
                // The blanks before the last are a word, and the last leads
                // the word this letter starts.
                Kind::Upper => {
                    end_word(state, emit);
                    take(state, len, limit, emit);
                    state.mode = IN_CAPITALS;
                }
                Kind::Lower => {
                    end_word(state, emit);
                    take(state, len, limit, emit);
                    state.mode = IN_LOWER;
                }
                Kind::Both | Kind::Mark => {
                    end_word(state, emit);
                    take(state, len, limit, emit);
                    state.mode = IN_MIXED;
                }
                // Only the space joins a run of symbols.
                Kind::Slash | Kind::Symbol => {
                    end_word(state, emit);
                    if &state.pending[..state.held as usize] == " " {
                        take(state, len, limit, emit);
                        state.mode = IN_SYMBOLS;
                    } else {
                        take(state, 0, limit, emit);
                        end_word(state, emit);
                        state.mode = AT_START;
                        continue;
                    }
                }
                Kind::Digit => {
                    end_both(state, limit, emit);
                    state.mode = AT_START;
                    continue;
                }
            },
            // AFTER_BREAK, the one place left.
            _ => match kind {
                Kind::Break => take(state, len, limit, emit),
                Kind::Space | Kind::Blank => hold(state, len, limit, emit),
                _ => {
                    end_word(state, emit);
                    // The blanks since the last break are a run with no break,
                    // which this character ends.
                    let held = state.held as usize;
                    match state.pending[..held].chars().next_back() {
                        Some(last) => {
                            state.held = last.len_utf8() as u32;
                            state.word = (held - last.len_utf8()) as u32;
                            spill(state, limit, emit);
                            state.mode = IN_BLANKS;
                        }
                        None => state.mode = AT_START,
                    }
                    continue;
                }
            },
        }
        return;
    }
}

/// Turns pending text into atoms: special tokens, and the words of the plain
/// text between them. Until the text ends, text that could still become a
/// special token waits.
pub fn drain_words(
    state: &mut ScanState,
    last: bool,
    profile: &TokenizerProfile,
    emit: &mut impl FnMut(PromptAtom),
) {
    let limit = (profile.longest_token as usize).max(4);
    let mut leads = [false; 256];
    let mut lookahead = 0;
    for special in specials_of(profile) {
        leads[usize::from(special.as_bytes()[0])] = true;
        lookahead = lookahead.max(special.len());
    }
    loop {
        let rest = &state.pending[(state.word + state.held) as usize..];
        let Some(ch) = rest.chars().next() else {
            break;
        };
        if leads[usize::from(rest.as_bytes()[0])] {
            if !last && rest.len() < lookahead {
                break;
            }
            let special = specials_of(profile).find(|text| rest.starts_with(text));
            if let Some(special) = special {
                // A special token ends the plain text before it.
                finish_words(state, limit, emit);
                emit(PromptAtom {
                    text: special.to_string(),
                    special: true,
                    part: ATOM_WHOLE,
                });
                state.pending.drain(..special.len());
                continue;
            }
        }
        feed(state, ch, limit, emit);
    }
    if last {
        finish_words(state, limit, emit);
    }
}

/// Applies one word atom: the pieces it becomes, and the split's next
/// position. Every word is a segment of its own.
///
/// A whole word that is a token is that token, which is the tokenizer's own
/// rule. Any other word is its bytes, for the merge to work on.
pub fn apply_word(
    state: &SplitState,
    atom: &PromptAtom,
    in_vocabulary: bool,
    emit: &mut impl FnMut(BpePiece),
) -> SplitState {
    let mut next = state.clone();
    next.previous_unknown = false;
    if atom.part != ATOM_LATER_PART {
        next.segment = next.segment.saturating_add(1);
    }
    if atom.part == ATOM_WHOLE && in_vocabulary {
        emit(BpePiece {
            text: atom.text.clone(),
            segment: next.segment,
        });
    } else {
        for ch in atom.text.chars() {
            emit(BpePiece {
                text: String::from(ch),
                segment: next.segment,
            });
        }
    }
    next
}
