// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/// What this pipeline knows of Celestia's blocks: for a height, the root of the
/// block's data and the block's time.
///
/// A trusted relayer posts them. That stands in for a light client of
/// Celestia on this pipeline (Blobstream was one): a contract that takes the same
/// facts with a proof replaces this one behind `blockOf`, and nothing that
/// reads it changes.
contract DataRootOracle {
    struct Block {
        bytes32 dataRoot;
        uint64 time;
    }

    address public immutable relayer;
    mapping(uint64 => Block) private blocks;

    event BlockPosted(uint64 indexed height, bytes32 dataRoot, uint64 time);

    error NotTheRelayer();
    error AnotherRootPosted(uint64 height);

    constructor(address relayer_) {
        relayer = relayer_;
    }

    /// Post a block's data root and time. Posting the same block again
    /// changes nothing, and a block's root is never replaced.
    function post(uint64 height, bytes32 dataRoot, uint64 time) external {
        if (msg.sender != relayer) revert NotTheRelayer();
        Block memory known = blocks[height];
        if (known.time != 0) {
            if (known.dataRoot != dataRoot || known.time != time) revert AnotherRootPosted(height);
            return;
        }
        blocks[height] = Block(dataRoot, time);
        emit BlockPosted(height, dataRoot, time);
    }

    /// The data root and time of the block at `height`. The time is zero for
    /// a block nobody has posted.
    function blockOf(uint64 height) external view returns (bytes32 dataRoot, uint64 time) {
        Block memory known = blocks[height];
        return (known.dataRoot, known.time);
    }
}
