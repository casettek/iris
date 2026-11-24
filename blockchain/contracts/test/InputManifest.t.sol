// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import "../src/InputManifest.sol";
import "../src/IrisEncoding.sol";

/// A stage's input list is read one way only: the canonical form, and
/// nothing else. `read_input_manifest` in `inference_artifacts::checks`
/// holds the same cases in Rust.
contract InputManifestTest {
    // Two inputs, as `serde_json` writes them: sorted, two spaces of indent.
    string private constant LIST = '{\n  "activations": {\n    "commitment": "aa11",\n    "encoding": "raster",\n'
        '    "type": "sha256"\n  },\n  "ple": {\n    "commitment": "bb22",\n    "encoding": "raster",\n'
        '    "type": "sha256"\n  }\n}';

    function _read(string memory list, string memory input) private pure returns (bool, bool, string memory) {
        (bool readable, bool found, bytes memory commitment) = InputManifest.commitmentOf(bytes(list), bytes(input));
        return (readable, found, string(commitment));
    }

    function _unreadable(string memory list) private pure {
        (bool readable,,) = _read(list, "activations");
        require(!readable, list);
    }

    function _replace(string memory text, string memory from, string memory to) private pure returns (string memory) {
        bytes memory source = bytes(text);
        bytes memory old = bytes(from);
        for (uint256 at = 0; at + old.length <= source.length; at++) {
            bool here = true;
            for (uint256 i = 0; i < old.length && here; i++) {
                here = source[at + i] == old[i];
            }
            if (!here) continue;
            bytes memory head = new bytes(at);
            for (uint256 i = 0; i < at; i++) {
                head[i] = source[i];
            }
            bytes memory tail = new bytes(source.length - at - old.length);
            for (uint256 i = 0; i < tail.length; i++) {
                tail[i] = source[at + old.length + i];
            }
            return string(abi.encodePacked(head, to, tail));
        }
        revert("nothing to replace");
    }

    function test_a_canonical_list_gives_each_input_its_commitment() public pure {
        (bool readable, bool found, string memory commitment) = _read(LIST, "activations");
        require(readable && found && IrisEncoding.sameText(commitment, "aa11"), "activations");
        (readable, found, commitment) = _read(LIST, "ple");
        require(readable && found && IrisEncoding.sameText(commitment, "bb22"), "ple");
        (readable, found,) = _read(LIST, "prior_kv");
        require(readable && !found, "an input the list does not hold");
        // A name that another begins with is another name.
        (readable, found,) = _read(LIST, "pl");
        require(readable && !found, "a shorter name");
    }

    function test_an_empty_list_is_two_braces() public pure {
        (bool readable, bool found,) = _read("{}", "activations");
        require(readable && !found, "{}");
        _unreadable("{\n}");
        _unreadable("{ }");
        _unreadable("");
        _unreadable("[]");
    }

    function test_other_spacing_is_not_a_list() public pure {
        _unreadable(string(abi.encodePacked(LIST, "\n")));
        _unreadable(_replace(LIST, '"ple": {', '"ple":  {'));
        _unreadable(_replace(LIST, '  "ple"', '"ple"'));
        _unreadable(_replace(LIST, '"commitment": "aa11"', '"commitment":"aa11"'));
        _unreadable(_replace(LIST, "},\n", "}, \n"));
        _unreadable(_replace(LIST, "\n  }\n}", "\n  }}"));
    }

    function test_names_are_in_order_and_not_repeated() public pure {
        // The same entries the other way round.
        _unreadable(_replace(_replace(LIST, '"ple"', '"zzz"'), '"activations"', '"zzzz"'));
        // One name twice: a reader that took the first and a reader that
        // took the last would load different inputs.
        _unreadable(_replace(LIST, '"ple"', '"activations"'));
    }

    function test_fields_are_the_three_in_their_order() public pure {
        _unreadable(_replace(LIST, '"encoding": "raster",\n    "type": "sha256"\n  },', '"type": "sha256",\n    "encoding": "raster"\n  },'));
        _unreadable(_replace(LIST, '"type": "sha256"\n  },', '"type": "sha256",\n    "extra": "1"\n  },'));
        _unreadable(_replace(LIST, '    "encoding": "raster",\n    "type": "sha256"\n  },', '    "type": "sha256"\n  },'));
    }

    function test_text_is_plain() public pure {
        // An escape, a control character and a byte outside ASCII: each is
        // text some reader would have to interpret.
        _unreadable(_replace(LIST, "aa11", 'aa\\"11'));
        _unreadable(_replace(LIST, "aa11", "aa\\\\11"));
        _unreadable(_replace(LIST, "aa11", "aa\t11"));
        _unreadable(_replace(LIST, "ple", unicode"plé"));
    }

    /// A commitment is read as the 32 bytes its text writes, and text that
    /// is not 64 lowercase hex digits is no commitment: it equals none.
    function test_a_commitment_is_64_lowercase_hex_digits() public pure {
        bytes memory text = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";
        (bool ok, bytes32 value) = InputManifest.asCommitment(text);
        assert(ok);
        assert(value == 0x00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff);

        (ok,) = InputManifest.asCommitment("00112233445566778899AABBCCDDEEFF00112233445566778899aabbccddeeff");
        assert(!ok);
        (ok,) = InputManifest.asCommitment("0x112233445566778899aabbccddeeff00112233445566778899aabbccddeeff");
        assert(!ok);
        (ok,) = InputManifest.asCommitment("00112233445566778899aabbccddeeff00112233445566778899aabbccddeef");
        assert(!ok);
        (ok,) = InputManifest.asCommitment("00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff0");
        assert(!ok);
        (ok,) = InputManifest.asCommitment("");
        assert(!ok);
    }

    function test_an_input_is_read_as_its_commitment() public pure {
        bytes memory list = bytes(
            string.concat(
                "{\n",
                '  "layer": {\n',
                '    "commitment": "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff",\n',
                '    "encoding": "raster",\n',
                '    "type": "sha256"\n',
                "  },\n",
                '  "marks": {\n',
                '    "commitment": "not a commitment",\n',
                '    "encoding": "raster",\n',
                '    "type": "sha256"\n',
                "  }\n",
                "}"
            )
        );
        (bool readable, bool found, bool is32, bytes32 value) = InputManifest.valueOf(list, "layer");
        assert(readable && found && is32);
        assert(value == 0x00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff);
        (readable, found, is32,) = InputManifest.valueOf(list, "marks");
        assert(readable && found && !is32);
        (readable, found, is32,) = InputManifest.valueOf(list, "absent");
        assert(readable && !found && !is32);
        (readable,,,) = InputManifest.valueOf("{", "layer");
        assert(!readable);
    }
}

/// The shape the protocol fixes for a trace: `protocol_window_size` and
/// `FraudProofConfig::from_window_size` in Raster.
contract TraceShapeTest {
    function test_the_window_is_128_or_the_largest_power_of_two_below_the_length() public pure {
        require(IrisEncoding.protocolWindow(0) == 0 && IrisEncoding.protocolWindow(2) == 0, "too short");
        require(IrisEncoding.protocolWindow(3) == 2 && IrisEncoding.protocolWindow(4) == 2, "3 and 4");
        require(IrisEncoding.protocolWindow(5) == 4 && IrisEncoding.protocolWindow(9) == 8, "5 and 9");
        require(IrisEncoding.protocolWindow(128) == 64 && IrisEncoding.protocolWindow(129) == 128, "128 and 129");
        require(IrisEncoding.protocolWindow(1 << 40) == 128, "long");
    }

    function test_a_window_reveals_128_fingerprint_bits() public pure {
        require(IrisEncoding.protocolBitsPerItem(128) == 1 && IrisEncoding.protocolBitsPerItem(64) == 2, "128 and 64");
        require(IrisEncoding.protocolBitsPerItem(4) == 32 && IrisEncoding.protocolBitsPerItem(2) == 64, "4 and 2");
    }
}
