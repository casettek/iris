// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import "./IrisTypes.sol";
import "./IrisEncoding.sol";

/// The configurations claims are made of, each registered before any claim
/// arrives and named by its id: `inference_artifacts::registry`.
///
/// A configuration's id is computed here from the configuration itself, in
/// the encoding `Configuration::id` uses, so the id a claim names stands for
/// exactly one set of programs, models, parameters and requirements.
contract IrisRegistry {
    address public immutable owner;
    mapping(bytes32 => bytes) private configurations;
    /// For a configuration, the stage project each approved program is of, as
    /// the hash of its name. Zero for a program the configuration does not
    /// approve.
    mapping(bytes32 => mapping(bytes32 => bytes32)) public stageOf;

    event Registered(bytes32 indexed id);

    error NotTheOwner();
    error NotRegistered(bytes32 id);
    error MalformedConfiguration();

    constructor(address owner_) {
        owner = owner_;
    }

    /// Register a configuration, ABI-encoded, and return its id. Registering
    /// one twice changes nothing.
    function register(bytes calldata encoded) external returns (bytes32 id) {
        if (msg.sender != owner) revert NotTheOwner();
        Configuration memory configuration = abi.decode(encoded, (Configuration));
        id = IrisEncoding.configurationId(configuration);
        if (configurations[id].length != 0) {
            return id;
        }
        configurations[id] = encoded;
        for (uint256 i = 0; i < configuration.models.length; i++) {
            if (!IrisEncoding.wellFormedCommitments(configuration.models[i].commitments)) {
                revert MalformedConfiguration();
            }
        }
        _approve(id, configuration.programs);
        emit Registered(id);
    }

    /// Walk the programs as the id encodes them, and record each program's
    /// stage. The bytes must be exactly a list of programs: the id is a hash
    /// over them, so what is read here is what the id stands for.
    function _approve(bytes32 id, bytes memory programs) private {
        (uint256 items, uint256 at) = IrisEncoding.readCount(programs, 0);
        for (uint256 i = 0; i < items; i++) {
            bytes memory stage;
            bytes32 program;
            (stage, at) = IrisEncoding.readText(programs, at);
            (program, at) = IrisEncoding.readHash(programs, at);
            at = _skipChainedInputs(programs, at);
            uint256 runInputs;
            (runInputs, at) = IrisEncoding.readCount(programs, at);
            for (uint256 j = 0; j < runInputs; j++) {
                (, at) = IrisEncoding.readText(programs, at);
            }
            stageOf[id][program] = keccak256(stage);
        }
        if (at != programs.length) revert MalformedConfiguration();
    }

    function _skipChainedInputs(bytes memory programs, uint256 at) private pure returns (uint256) {
        uint256 inputs;
        (inputs, at) = IrisEncoding.readCount(programs, at);
        for (uint256 i = 0; i < inputs; i++) {
            (, at) = IrisEncoding.readText(programs, at);
            uint256 producers;
            (producers, at) = IrisEncoding.readCount(programs, at);
            for (uint256 j = 0; j < producers; j++) {
                (, at) = IrisEncoding.readText(programs, at);
            }
            uint256 emptyValues;
            (emptyValues, at) = IrisEncoding.readCount(programs, at);
            for (uint256 j = 0; j < emptyValues; j++) {
                (, at) = IrisEncoding.readHash(programs, at);
            }
        }
        return at;
    }

    /// What a configuration says of one input of one of its programs:
    /// whether the program takes it from an earlier stage, and if so the
    /// stages that may produce it (each as the hash of its name, as `stageOf`
    /// gives a program's) and the empty values it may hold in place of one.
    function chainedInput(bytes32 id, bytes32 program, string calldata input)
        external
        view
        returns (bool chained, bytes32[] memory producers, bytes32[] memory emptyValues)
    {
        bytes memory encoded = configurations[id];
        if (encoded.length == 0) revert NotRegistered(id);
        bytes memory programs = abi.decode(encoded, (Configuration)).programs;
        bytes32 wanted = keccak256(bytes(input));
        (uint256 items, uint256 at) = IrisEncoding.readCount(programs, 0);
        for (uint256 i = 0; i < items; i++) {
            bytes32 current;
            (, at) = IrisEncoding.readText(programs, at);
            (current, at) = IrisEncoding.readHash(programs, at);
            uint256 inputs;
            (inputs, at) = IrisEncoding.readCount(programs, at);
            for (uint256 j = 0; j < inputs; j++) {
                bytes memory name;
                (name, at) = IrisEncoding.readText(programs, at);
                uint256 listed;
                (listed, at) = IrisEncoding.readCount(programs, at);
                producers = new bytes32[](listed);
                for (uint256 k = 0; k < listed; k++) {
                    bytes memory producer;
                    (producer, at) = IrisEncoding.readText(programs, at);
                    producers[k] = keccak256(producer);
                }
                (listed, at) = IrisEncoding.readCount(programs, at);
                emptyValues = new bytes32[](listed);
                for (uint256 k = 0; k < listed; k++) {
                    (emptyValues[k], at) = IrisEncoding.readHash(programs, at);
                }
                if (current == program && keccak256(name) == wanted) {
                    return (true, producers, emptyValues);
                }
            }
            uint256 runInputs;
            (runInputs, at) = IrisEncoding.readCount(programs, at);
            for (uint256 j = 0; j < runInputs; j++) {
                (, at) = IrisEncoding.readText(programs, at);
            }
        }
        return (false, new bytes32[](0), new bytes32[](0));
    }

    function isRegistered(bytes32 id) external view returns (bool) {
        return configurations[id].length != 0;
    }

    function configuration(bytes32 id) external view returns (Configuration memory) {
        bytes memory encoded = configurations[id];
        if (encoded.length == 0) revert NotRegistered(id);
        return abi.decode(encoded, (Configuration));
    }

    /// The id of a configuration, registered or not.
    function idOf(bytes calldata encoded) external pure returns (bytes32) {
        return IrisEncoding.configurationId(abi.decode(encoded, (Configuration)));
    }
}
