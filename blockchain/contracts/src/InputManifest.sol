// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/// A stage's input list, `input_manifest.json`, read one way only.
///
/// A checkpoint commits to the list by the SHA-256 of its bytes. For those
/// bytes to say one thing to every reader they must be exactly the canonical
/// encoding of what they hold, which is what `read_input_manifest` in
/// `inference_artifacts::checks` requires and what this reads:
///
/// ```text
/// {
///   "name": {
///     "commitment": "...",
///     "encoding": "...",
///     "type": "..."
///   },
///   "next": {
///     ...
///   }
/// }
/// ```
///
/// Two spaces of indent, a line feed after each line but the last, the
/// entries in ascending order of their names' bytes with none repeated, the
/// three fields in that order, and nothing after the closing brace. An empty
/// list is `{}`. Every name and field is plain text: printable ASCII with no
/// quote and no backslash, so nothing is escaped. Anything else is not a
/// list, and a claim whose stage commits to such bytes is void.
///
/// Deployed by itself and linked into the claims contract, which is near the
/// size a contract may be.
library InputManifest {
    /// The commitment the list holds for `input`. `readable` is false for
    /// bytes that are not a list in the form above; `found` is false when the
    /// list has no such input.
    function commitmentOf(bytes memory manifest, bytes memory input)
        public
        pure
        returns (bool readable, bool found, bytes memory commitment)
    {
        if (manifest.length == 2) {
            return (manifest[0] == "{" && manifest[1] == "}", false, "");
        }
        bool ok;
        uint256 at;
        (ok, at) = _literal(manifest, 0, "{\n");
        if (!ok) return (false, false, "");
        bytes memory previous;
        bool first = true;
        while (true) {
            bytes memory name;
            bytes memory value;
            (ok, at) = _literal(manifest, at, '  "');
            if (!ok) return (false, false, "");
            (ok, name, at) = _plain(manifest, at);
            if (!ok || (!first && !_less(previous, name))) return (false, false, "");
            (ok, at) = _literal(manifest, at, ': {\n    "commitment": "');
            if (!ok) return (false, false, "");
            (ok, value, at) = _plain(manifest, at);
            if (!ok) return (false, false, "");
            if (keccak256(name) == keccak256(input)) {
                found = true;
                commitment = value;
            }
            (ok, at) = _literal(manifest, at, ',\n    "encoding": "');
            if (!ok) return (false, false, "");
            (ok,, at) = _plain(manifest, at);
            if (!ok) return (false, false, "");
            (ok, at) = _literal(manifest, at, ',\n    "type": "');
            if (!ok) return (false, false, "");
            (ok,, at) = _plain(manifest, at);
            if (!ok) return (false, false, "");
            (ok, at) = _literal(manifest, at, "\n  }");
            if (!ok) return (false, false, "");
            if (at + 2 == manifest.length && manifest[at] == "\n" && manifest[at + 1] == "}") {
                return (true, found, commitment);
            }
            (ok, at) = _literal(manifest, at, ",\n");
            if (!ok) return (false, false, "");
            previous = name;
            first = false;
        }
    }

    /// The commitment the list holds for `input`, as the 32 bytes its text
    /// writes. `is32` is false when the text is not a commitment's.
    function valueOf(bytes memory manifest, bytes memory input)
        public
        pure
        returns (bool readable, bool found, bool is32, bytes32 value)
    {
        bytes memory text;
        (readable, found, text) = commitmentOf(manifest, input);
        if (readable && found) (is32, value) = asCommitment(text);
    }

    /// The 32 bytes a commitment's text writes: 64 lowercase hex digits and
    /// nothing else. Any other text is no commitment, and `ok` is false.
    function asCommitment(bytes memory text) public pure returns (bool ok, bytes32 value) {
        if (text.length != 64) return (false, 0);
        uint256 read;
        for (uint256 i = 0; i < 64; i++) {
            uint8 char = uint8(text[i]);
            uint8 digit;
            if (char >= 0x30 && char <= 0x39) digit = char - 0x30;
            else if (char >= 0x61 && char <= 0x66) digit = char - 0x57;
            else return (false, 0);
            read = (read << 4) | digit;
        }
        return (true, bytes32(read));
    }

    /// Whether `expected` is at `at`, and where what follows it begins.
    function _literal(bytes memory manifest, uint256 at, bytes memory expected)
        private
        pure
        returns (bool, uint256)
    {
        if (at + expected.length > manifest.length) return (false, at);
        for (uint256 i = 0; i < expected.length; i++) {
            if (manifest[at + i] != expected[i]) return (false, at);
        }
        return (true, at + expected.length);
    }

    /// The plain text from `at` up to its closing quote, and where what
    /// follows the quote begins.
    function _plain(bytes memory manifest, uint256 at) private pure returns (bool, bytes memory value, uint256) {
        uint256 end = at;
        while (end < manifest.length && manifest[end] != '"') {
            uint8 char = uint8(manifest[end]);
            if (char < 0x20 || char > 0x7e || char == 0x5c) return (false, value, at);
            end++;
        }
        if (end == manifest.length) return (false, value, at);
        value = new bytes(end - at);
        for (uint256 i = 0; i < value.length; i++) {
            value[i] = manifest[at + i];
        }
        return (true, value, end + 1);
    }

    /// Whether `a` sorts before `b` by its bytes.
    function _less(bytes memory a, bytes memory b) private pure returns (bool) {
        uint256 shared = a.length < b.length ? a.length : b.length;
        for (uint256 i = 0; i < shared; i++) {
            if (a[i] != b[i]) return a[i] < b[i];
        }
        return a.length < b.length;
    }
}
