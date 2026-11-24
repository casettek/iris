// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import "./IrisTypes.sol";

/// The byte encodings iris hashes, as `procedure_definition::encoding` and the
/// stages' own digests write them: a tag ending in a zero byte, fields in
/// order, numbers little-endian, text and lists after a 32-bit count, a hash
/// as a byte 1 and its 32 bytes. Everything is SHA-256.
library IrisEncoding {
    function le32(uint32 value) internal pure returns (bytes4) {
        return bytes4(
            ((value & 0x000000FF) << 24) | ((value & 0x0000FF00) << 8) | ((value & 0x00FF0000) >> 8)
                | ((value & 0xFF000000) >> 24)
        );
    }

    function le64(uint64 value) internal pure returns (bytes8) {
        return bytes8((uint64(uint32(le32(uint32(value)))) << 32) | uint64(uint32(le32(uint32(value >> 32)))));
    }

    function count(uint256 items) internal pure returns (bytes4) {
        return le32(uint32(items));
    }

    function text(string memory value) internal pure returns (bytes memory) {
        return abi.encodePacked(le32(uint32(bytes(value).length)), value);
    }

    function hash(bytes32 value) internal pure returns (bytes memory) {
        return abi.encodePacked(bytes1(0x01), value);
    }

    function flag(bool value) internal pure returns (bytes1) {
        return value ? bytes1(0x01) : bytes1(0x00);
    }

    function hashes(bytes32[] memory values) internal pure returns (bytes memory encoded) {
        encoded = abi.encodePacked(count(values.length));
        for (uint256 i = 0; i < values.length; i++) {
            encoded = abi.encodePacked(encoded, hash(values[i]));
        }
    }

    function sameText(string memory a, string memory b) internal pure returns (bool) {
        return keccak256(bytes(a)) == keccak256(bytes(b));
    }

    /// 32 bytes as 64 lowercase hex digits.
    function hexText(bytes32 value) internal pure returns (string memory) {
        bytes memory digits = "0123456789abcdef";
        bytes memory out = new bytes(64);
        for (uint256 i = 0; i < 32; i++) {
            out[2 * i] = digits[uint8(value[i]) >> 4];
            out[2 * i + 1] = digits[uint8(value[i]) & 0x0f];
        }
        return string(out);
    }

    // --- a statement: `Statement::digest` ---

    function statement(
        bytes32 inputText,
        bytes32 source,
        bytes32 modelId,
        bool rawPrompt,
        uint32 maxNewTokens,
        uint32 rangeSize
    ) internal pure returns (bytes32) {
        return sha256(
            abi.encodePacked(
                "iris/statement/v2\x00", inputText, source, modelId, flag(rawPrompt), le32(maxNewTokens), le32(rangeSize)
            )
        );
    }

    // --- a source record: `source_id` ---

    function sourceId(SourceRecord memory record) internal pure returns (bytes32) {
        if (!record.present) {
            return bytes32(0);
        }
        return sha256(
            abi.encodePacked(
                "iris/source/v1\x00",
                text(record.textSha256),
                text(record.recipeSha256),
                text(record.documentsSha256),
                text(record.signer),
                le64(record.textTotal),
                le32(record.documents),
                le32(record.fault),
                le64(record.faultPosition)
            )
        );
    }

    // --- the forms: `run_forms` ---

    /// The largest form the result step compares.
    uint256 internal constant MAX_FORM = 1024;

    /// One step of the result stage: the hash chained over the forms so far
    /// with this one.
    function chainForm(bytes32 chained, Form memory form) internal pure returns (bytes32) {
        return sha256(abi.encodePacked(chained, le32(form.code), le32(uint32(bytes(form.text).length)), form.text));
    }

    /// The code of the first form the output is, without the whitespace
    /// around it, or zero; and the hash over the forms.
    function runForms(Form[] memory forms, string memory output)
        internal
        pure
        returns (uint32 code, bytes32 chained)
    {
        bytes32 trimmed = keccak256(trim(bytes(output)));
        for (uint256 i = 0; i < forms.length; i++) {
            if (code == 0 && bytes(forms[i].text).length <= MAX_FORM && keccak256(bytes(forms[i].text)) == trimmed) {
                code = forms[i].code;
            }
            chained = chainForm(chained, forms[i]);
        }
    }

    // --- whitespace: Rust's `str::trim` ---

    /// The length of the whitespace character that starts at `at`, or zero.
    /// The characters are Unicode's White_Space set, which is what Rust's
    /// `str::trim` removes.
    function whitespaceAt(bytes memory value, uint256 at) internal pure returns (uint256) {
        uint8 first = uint8(value[at]);
        if ((first >= 0x09 && first <= 0x0d) || first == 0x20) {
            return 1;
        }
        if (first == 0xc2 && at + 1 < value.length) {
            uint8 second = uint8(value[at + 1]);
            return (second == 0x85 || second == 0xa0) ? 2 : 0;
        }
        if (at + 2 < value.length) {
            uint8 second = uint8(value[at + 1]);
            uint8 third = uint8(value[at + 2]);
            if (first == 0xe1) {
                return (second == 0x9a && third == 0x80) ? 3 : 0;
            }
            if (first == 0xe2) {
                if (second == 0x80) {
                    return ((third >= 0x80 && third <= 0x8a) || third == 0xa8 || third == 0xa9 || third == 0xaf)
                        ? 3
                        : 0;
                }
                return (second == 0x81 && third == 0x9f) ? 3 : 0;
            }
            if (first == 0xe3) {
                return (second == 0x80 && third == 0x80) ? 3 : 0;
            }
        }
        return 0;
    }

    /// The length of the whitespace character that ends just before `end`,
    /// or zero.
    function whitespaceBefore(bytes memory value, uint256 start, uint256 end) internal pure returns (uint256) {
        for (uint256 length = 1; length <= 3; length++) {
            if (end < start + length) {
                return 0;
            }
            if (whitespaceAt(value, end - length) == length) {
                return length;
            }
        }
        return 0;
    }

    function trim(bytes memory value) internal pure returns (bytes memory trimmed) {
        uint256 start = 0;
        uint256 end = value.length;
        while (start < end) {
            uint256 length = whitespaceAt(value, start);
            if (length == 0 || start + length > end) {
                break;
            }
            start += length;
        }
        while (end > start) {
            uint256 length = whitespaceBefore(value, start, end);
            if (length == 0) {
                break;
            }
            end -= length;
        }
        trimmed = new bytes(end - start);
        for (uint256 i = 0; i < trimmed.length; i++) {
            trimmed[i] = value[start + i];
        }
    }

    // --- a checkpoint and its list: `checkpoint_hash`, `checkpoints_root` ---

    uint256 internal constant TREE_DEPTH = 20;

    function checkpointHash(Checkpoint memory checkpoint) internal pure returns (bytes32) {
        bytes memory sources = abi.encodePacked(count(checkpoint.inputSources.length));
        for (uint256 i = 0; i < checkpoint.inputSources.length; i++) {
            sources = abi.encodePacked(
                sources, text(checkpoint.inputSources[i].name), le32(checkpoint.inputSources[i].position)
            );
        }
        return sha256(
            abi.encodePacked(
                "iris/checkpoint/v1\x00",
                text(checkpoint.stage),
                hash(checkpoint.program),
                hash(checkpoint.inputCommitment),
                sources,
                hash(checkpoint.outputCommitment),
                hash(checkpoint.outputSha256)
            )
        );
    }

    /// The list commitment a checkpoint hash at `index` and its path give,
    /// for a list of `total` checkpoints.
    function listRoot(bytes32 checkpoint, uint32 index, uint32 total, bytes32[] memory path)
        internal
        pure
        returns (bytes32)
    {
        require(path.length == TREE_DEPTH && index < total, "not an opening");
        bytes32 node = sha256(abi.encodePacked(bytes1(0x00), checkpoint));
        for (uint256 height = 0; height < TREE_DEPTH; height++) {
            node = (index >> height) & 1 == 0
                ? sha256(abi.encodePacked(bytes1(0x01), node, path[height]))
                : sha256(abi.encodePacked(bytes1(0x01), path[height], node));
        }
        return sha256(abi.encodePacked("iris/checkpoints/v1\x00", le32(total), hash(node)));
    }

    // --- a Raster trace commitment header ---

    /// A number as postcard writes it: seven bits a byte, low first.
    function varint(uint64 value) internal pure returns (bytes memory out) {
        while (value >= 0x80) {
            out = abi.encodePacked(out, bytes1(uint8(value & 0x7f) | 0x80));
            value >>= 7;
        }
        out = abi.encodePacked(out, bytes1(uint8(value)));
    }

    /// `TraceCommitmentHeader::digest`: what a fault proof's journal names as
    /// the trace it refutes, and what a posted trace body is found by.
    function traceHeaderDigest(TraceHeader memory header) internal pure returns (bytes32) {
        return sha256(
            abi.encodePacked(
                "raster/trace-commitment/v2",
                varint(header.bitsPerItem),
                varint(header.fingerprintLen),
                bytes1(0x20),
                header.fingerprintRoot,
                bytes1(0x20),
                header.revealedItems,
                varint(header.windowSize),
                bytes1(0x20),
                header.revealedTailRoots
            )
        );
    }

    /// The window size a trace of `traceLength` items must be committed at:
    /// `raster_prover::challenge::protocol_window_size`. 128, or for a
    /// shorter trace the largest power of two below its length. Zero for a
    /// trace too short for any window.
    function protocolWindow(uint64 traceLength) internal pure returns (uint64) {
        if (traceLength <= 2) return 0;
        uint64 below = 1;
        for (uint64 rest = traceLength - 1; rest > 1; rest >>= 1) {
            below <<= 1;
        }
        return below < 128 ? below : 128;
    }

    /// The fingerprint bits an item has at a window size:
    /// `FraudProofConfig::from_window_size`. A window reveals 128 bits.
    function protocolBitsPerItem(uint64 window) internal pure returns (uint64) {
        return (128 + window - 1) / window;
    }

    // --- a posted blob: `attestation_digest` ---

    function record(Record memory value) internal pure returns (bytes memory) {
        return abi.encodePacked(hash(value.statement), hash(value.outputTextSha256), flag(value.valid));
    }

    function statedRecords(StatedRecords memory stated) internal pure returns (bytes memory encoded) {
        encoded =
            abi.encodePacked(hash(stated.checkpointsRoot), hash(stated.resultProgram), count(stated.resultInputs.length));
        for (uint256 i = 0; i < stated.resultInputs.length; i++) {
            encoded =
                abi.encodePacked(encoded, text(stated.resultInputs[i].name), le32(stated.resultInputs[i].position));
        }
        encoded = abi.encodePacked(encoded, count(stated.results.length));
        for (uint256 i = 0; i < stated.results.length; i++) {
            encoded = abi.encodePacked(
                encoded,
                record(stated.results[i].record),
                le32(stated.results[i].result.code),
                hash(stated.results[i].result.formsSha256)
            );
        }
        encoded = abi.encodePacked(encoded, flag(stated.hasConsensus));
        if (stated.hasConsensus) {
            encoded = abi.encodePacked(
                encoded, flag(stated.consensus.sameInput), flag(stated.consensus.agree), le32(stated.consensus.code)
            );
        }
        encoded = abi.encodePacked(encoded, count(stated.checks.length));
        for (uint256 i = 0; i < stated.checks.length; i++) {
            encoded = abi.encodePacked(
                encoded, le32(stated.checks[i].checkpoint), hash(stated.checks[i].program), record(stated.checks[i].record)
            );
        }
    }

    /// The digest a blob's contents list a stated-records entry by.
    function statedRecordsDigest(StatedRecords memory stated) internal pure returns (bytes32) {
        return sha256(abi.encodePacked("iris/stated-records/v1\x00", statedRecords(stated)));
    }

    function contents(BlobContents memory held) internal pure returns (bytes memory) {
        return abi.encodePacked(
            held.submitter,
            hashes(held.runInputs),
            hashes(held.checkpointLists),
            hashes(held.traceBodies),
            hashes(held.records)
        );
    }

    /// What the attester signs for a blob: where it is, the root of its
    /// block, and what it holds.
    function attestationDigest(PostedBlob memory blob) internal pure returns (bytes32) {
        return sha256(
            abi.encodePacked(
                "iris/blob-attestation/v2\x00",
                le64(blob.height),
                blob.namespace,
                hash(blob.commitment),
                le64(blob.length),
                hash(blob.dataRoot),
                contents(blob.contents)
            )
        );
    }

    // --- a procedure: `Procedure::id` ---

    // --- reading the encoding back: a cursor over bytes ---

    function readCount(bytes memory encoded, uint256 at) internal pure returns (uint256 value, uint256 next) {
        require(at + 4 <= encoded.length, "encoding cut short");
        value = uint256(uint8(encoded[at])) | (uint256(uint8(encoded[at + 1])) << 8)
            | (uint256(uint8(encoded[at + 2])) << 16) | (uint256(uint8(encoded[at + 3])) << 24);
        next = at + 4;
    }

    function readText(bytes memory encoded, uint256 at) internal pure returns (bytes memory value, uint256 next) {
        uint256 length;
        (length, next) = readCount(encoded, at);
        require(next + length <= encoded.length, "encoding cut short");
        value = new bytes(length);
        for (uint256 i = 0; i < length; i++) {
            value[i] = encoded[next + i];
        }
        next += length;
    }

    function readHash(bytes memory encoded, uint256 at) internal pure returns (bytes32 value, uint256 next) {
        require(at + 33 <= encoded.length && encoded[at] == 0x01, "not a hash");
        assembly {
            value := mload(add(add(encoded, 0x21), at))
        }
        next = at + 33;
    }

    /// Whether bytes are a model's commitments as the id encodes them: a
    /// count, then each role and commitment, and nothing after.
    function wellFormedCommitments(bytes memory encoded) internal pure returns (bool) {
        (uint256 items, uint256 at) = readCount(encoded, 0);
        for (uint256 i = 0; i < items; i++) {
            (, at) = readText(encoded, at);
            (, at) = readHash(encoded, at);
        }
        return at == encoded.length;
    }

    function procedureId(Procedure memory procedure) internal pure returns (bytes32) {
        bytes memory encoded = abi.encodePacked(
            "iris/procedure/v3\x00",
            le32(procedure.version),
            text(procedure.procedureKind),
            text(procedure.family),
            procedure.programs,
            count(procedure.models.length)
        );
        for (uint256 i = 0; i < procedure.models.length; i++) {
            encoded = abi.encodePacked(
                encoded,
                hash(procedure.models[i].modelId),
                procedure.models[i].commitments,
                le32(procedure.models[i].rangeSize)
            );
        }
        encoded = abi.encodePacked(encoded, flag(procedure.rawPrompt), le32(procedure.maxNewTokens));
        if (procedure.hasSource) {
            encoded = abi.encodePacked(encoded, bytes1(0x01), text(procedure.signer));
            encoded = procedure.hasRecipe
                ? abi.encodePacked(encoded, bytes1(0x01), hash(procedure.recipe))
                : abi.encodePacked(encoded, bytes1(0x00));
        } else {
            encoded = abi.encodePacked(encoded, bytes1(0x00));
        }
        encoded = abi.encodePacked(encoded, count(procedure.forms.length));
        for (uint256 i = 0; i < procedure.forms.length; i++) {
            encoded = abi.encodePacked(encoded, text(procedure.forms[i].text), le32(procedure.forms[i].code));
        }
        return sha256(encoded);
    }
}
