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
    /// For a configuration, what each approved program's entry says of its
    /// inputs, as the id encodes it: read one program at a time, since the
    /// whole configuration is large.
    mapping(bytes32 => mapping(bytes32 => bytes)) private inputsOf;

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
    /// stage and what it says of its inputs. The bytes must be exactly a list
    /// of programs: the id is a hash over them, so what is read here is what
    /// the id stands for.
    function _approve(bytes32 id, bytes memory programs) private {
        (uint256 items, uint256 at) = IrisEncoding.readCount(programs, 0);
        for (uint256 i = 0; i < items; i++) {
            bytes memory stage;
            bytes32 program;
            (stage, at) = IrisEncoding.readText(programs, at);
            (program, at) = IrisEncoding.readHash(programs, at);
            uint256 inputs = at;
            at = _skipInputs(programs, at);
            stageOf[id][program] = keccak256(stage);
            bytes memory entry = new bytes(at - inputs);
            for (uint256 j = 0; j < entry.length; j++) {
                entry[j] = programs[inputs + j];
            }
            inputsOf[id][program] = entry;
        }
        if (at != programs.length) revert MalformedConfiguration();
    }

    /// A program's inputs as its entry lists them: the chained ones, each
    /// with its producers and empty values; the run's; and the pinned ones,
    /// each with its values.
    function _skipInputs(bytes memory programs, uint256 at) private pure returns (uint256) {
        uint256 inputs;
        uint256 listed;
        (inputs, at) = IrisEncoding.readCount(programs, at);
        for (uint256 i = 0; i < inputs; i++) {
            (, at) = IrisEncoding.readText(programs, at);
            (listed, at) = IrisEncoding.readCount(programs, at);
            for (uint256 j = 0; j < listed; j++) {
                (, at) = IrisEncoding.readText(programs, at);
            }
            (listed, at) = IrisEncoding.readCount(programs, at);
            for (uint256 j = 0; j < listed; j++) {
                (, at) = IrisEncoding.readHash(programs, at);
            }
        }
        (inputs, at) = IrisEncoding.readCount(programs, at);
        for (uint256 i = 0; i < inputs; i++) {
            (, at) = IrisEncoding.readText(programs, at);
        }
        (inputs, at) = IrisEncoding.readCount(programs, at);
        for (uint256 i = 0; i < inputs; i++) {
            (, at) = IrisEncoding.readText(programs, at);
            (listed, at) = IrisEncoding.readCount(programs, at);
            for (uint256 j = 0; j < listed; j++) {
                (, at) = IrisEncoding.readHash(programs, at);
            }
        }
        return at;
    }

    /// What a configuration says of one input of one of its approved
    /// programs: where the input may come from. Nothing is set for a name the
    /// program's entry does not list.
    function inputOf(bytes32 id, bytes32 program, string calldata input)
        external
        view
        returns (InputRule memory rule)
    {
        bytes memory entry = inputsOf[id][program];
        if (entry.length == 0) revert NotRegistered(id);
        bytes32 wanted = keccak256(bytes(input));
        uint256 inputs;
        uint256 listed;
        uint256 at;
        bytes memory name;
        (inputs, at) = IrisEncoding.readCount(entry, 0);
        for (uint256 i = 0; i < inputs; i++) {
            (name, at) = IrisEncoding.readText(entry, at);
            bool here = keccak256(name) == wanted;
            (listed, at) = IrisEncoding.readCount(entry, at);
            bytes32[] memory producers = new bytes32[](listed);
            for (uint256 j = 0; j < listed; j++) {
                bytes memory producer;
                (producer, at) = IrisEncoding.readText(entry, at);
                producers[j] = keccak256(producer);
            }
            (listed, at) = IrisEncoding.readCount(entry, at);
            bytes32[] memory emptyValues = new bytes32[](listed);
            for (uint256 j = 0; j < listed; j++) {
                (emptyValues[j], at) = IrisEncoding.readHash(entry, at);
            }
            if (here) {
                rule.chained = true;
                rule.producers = producers;
                rule.emptyValues = emptyValues;
            }
        }
        (inputs, at) = IrisEncoding.readCount(entry, at);
        for (uint256 i = 0; i < inputs; i++) {
            (name, at) = IrisEncoding.readText(entry, at);
            if (keccak256(name) == wanted) rule.run = true;
        }
        (inputs, at) = IrisEncoding.readCount(entry, at);
        for (uint256 i = 0; i < inputs; i++) {
            (name, at) = IrisEncoding.readText(entry, at);
            (listed, at) = IrisEncoding.readCount(entry, at);
            bytes32[] memory values = new bytes32[](listed);
            for (uint256 j = 0; j < listed; j++) {
                (values[j], at) = IrisEncoding.readHash(entry, at);
            }
            if (keccak256(name) == wanted) {
                rule.pinned = true;
                rule.pinnedValues = values;
            }
        }
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
