// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/// What a Raster fault proof says, read from its journal: the journal of
/// Raster's transition program (`raster_core::transition::TransitionJournal`).
struct FaultVerdict {
    /// The window's last item diverged from the trace commitment it names.
    /// The fields below are read only then.
    bool finished;
    /// The transition program the window's earlier steps were checked by.
    bytes32 transitionImageId;
    /// The program the window's input manifest was checked by.
    bytes32 authorizationImageId;
    /// The inputs the window ran on.
    bytes32 inputManifest;
    /// The program the window ran.
    bytes32 program;
    /// The digest of the trace commitment header the window refutes.
    bytes32 refutedTrace;
}

/// Reads a `TransitionJournal` as RISC Zero wrote it. Its encoding is a run
/// of little-endian 32-bit words, in field order:
///
/// ```text
/// u64, usize        two words, low first
/// bytes             the length in one word, then the bytes, padded to a word
/// a list            the count in one word, then each item
/// an enum           the variant in one word, then what it holds
/// ```
///
/// The journal begins with the window's opening state, whose size depends on
/// where the window opens, so the fields a contract wants are found by
/// walking it:
///
/// ```text
/// init_state
///   init_frontier            u64 position, bytes leaf, list of bytes ommers
///   init_storage_frontier    the same
///   init_storage_root        bytes
///   init_storage_index_root  bytes
///   fingerprint              usize bits per item, list of u64, usize length
/// current_state              2 is `Finished`, and holds nothing
/// transition_image_id        bytes
/// authorization_image_id     bytes
/// input_manifest_commitment  bytes
/// program_commitment         bytes
/// refuted_trace_commitment   bytes
/// ```
///
/// What follows is not read. The layout is the transition program's, so it
/// changes only with that program's id, which a contract pins.
library RasterJournal {
    error MalformedJournal();

    uint256 private constant FINISHED = 2;

    function read(bytes memory journal) internal pure returns (FaultVerdict memory verdict) {
        uint256 at = _frontier(journal, 0);
        at = _frontier(journal, at);
        (, at) = _bytes(journal, at);
        (, at) = _bytes(journal, at);
        uint256 count;
        (count, at) = _word(journal, at + 8);
        uint256 state;
        (state, at) = _word(journal, at + 8 * count + 8);
        if (state != FINISHED) return verdict;
        verdict.finished = true;
        (verdict.transitionImageId, at) = _hash(journal, at);
        (verdict.authorizationImageId, at) = _hash(journal, at);
        (verdict.inputManifest, at) = _hash(journal, at);
        (verdict.program, at) = _hash(journal, at);
        (verdict.refutedTrace, at) = _hash(journal, at);
    }

    function _frontier(bytes memory journal, uint256 at) private pure returns (uint256) {
        (, at) = _bytes(journal, at + 8);
        uint256 count;
        (count, at) = _word(journal, at);
        for (uint256 i = 0; i < count; i++) {
            (, at) = _bytes(journal, at);
        }
        return at;
    }

    /// One word, and where the next begins.
    function _word(bytes memory journal, uint256 at) private pure returns (uint256 value, uint256 next) {
        next = at + 4;
        if (next > journal.length) revert MalformedJournal();
        value = uint256(uint8(journal[at])) | (uint256(uint8(journal[at + 1])) << 8)
            | (uint256(uint8(journal[at + 2])) << 16) | (uint256(uint8(journal[at + 3])) << 24);
    }

    /// Where a byte string's bytes begin and how many there are, and where
    /// the field after it begins.
    function _bytes(bytes memory journal, uint256 at) private pure returns (uint256 length, uint256 next) {
        (length, at) = _word(journal, at);
        next = at + ((length + 3) / 4) * 4;
        if (next > journal.length) revert MalformedJournal();
    }

    /// A byte string that must be one 32-byte hash.
    function _hash(bytes memory journal, uint256 at) private pure returns (bytes32 value, uint256 next) {
        uint256 length;
        (length, next) = _bytes(journal, at);
        if (length != 32) revert MalformedJournal();
        assembly ("memory-safe") {
            value := mload(add(add(journal, 0x20), add(at, 4)))
        }
    }
}
