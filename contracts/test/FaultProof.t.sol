// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import "../src/IrisClaims.sol";
import "../src/RasterVerifier.sol";

/// The part of Foundry's test host this file uses.
interface Vm {
    function readFile(string calldata path) external view returns (string memory);
    function parseJsonBytes(string calldata json, string calldata key) external pure returns (bytes memory);
    function parseJsonBytes32(string calldata json, string calldata key) external pure returns (bytes32);
    function expectRevert(bytes4 selector) external;
    function expectRevert(bytes calldata data) external;
}

/// The claims contract with a way to hold a challenge that never arrived, so
/// that a fault proof can be judged against it without a claim behind it.
contract HeldChallenges is IrisClaims {
    constructor(IRiscZeroVerifier verifier_, bytes32 transition, bytes32 authorization)
        IrisClaims(
            IrisRegistry(address(1)), DataRootOracle(address(1)), address(1), 0, 0, 0, verifier_, transition, authorization
        )
    {}

    function hold(bytes32 id, bytes32 trace, bytes32 program, bytes32 inputs) external {
        challenges[id] = Admitted({
            claimId: bytes32(uint256(1)),
            challenger: address(2),
            stage: 0,
            traceHeaderDigest: trace,
            claimedOutput: bytes32(0),
            challengedAt: 1,
            program: program,
            inputCommitment: inputs,
            dismissedAt: 0
        });
    }
}

/// `submitFaultProof` on a real proof: `test/fault_proof.json` is a Raster
/// fault proof of a small stage, as `iris evm fault-proof --receipt` prints
/// it. The proof names its own programs, so it stays a proof whatever
/// Raster's current ones are; what would break it is a change of RISC Zero's
/// circuits, which is a change of the vendored verifier.
contract FaultProofTest {
    Vm private constant vm = Vm(address(uint160(uint256(keccak256("hevm cheat code")))));
    bytes32 private constant ID = bytes32(uint256(0xc4a11e46e));

    RasterVerifier private verifier;
    bytes private journal;
    bytes private seal;
    bytes32 private transition;
    bytes32 private authorization;
    bytes32 private trace;
    bytes32 private program;
    bytes32 private inputs;

    function setUp() public {
        verifier = new RasterVerifier();
        string memory proof = vm.readFile("test/fault_proof.json");
        journal = vm.parseJsonBytes(proof, ".journal");
        seal = vm.parseJsonBytes(proof, ".seal");
        transition = vm.parseJsonBytes32(proof, ".verdict.transition_image_id");
        authorization = vm.parseJsonBytes32(proof, ".verdict.authorization_image_id");
        trace = vm.parseJsonBytes32(proof, ".verdict.refuted_trace");
        program = vm.parseJsonBytes32(proof, ".verdict.program");
        inputs = vm.parseJsonBytes32(proof, ".verdict.input_manifest");
    }

    function _held(bytes32 trace_, bytes32 program_, bytes32 inputs_) private returns (HeldChallenges claims) {
        claims = new HeldChallenges(verifier, transition, authorization);
        claims.hold(ID, trace_, program_, inputs_);
    }

    function _proof(bytes memory journal_, bytes memory seal_) private pure returns (bytes memory) {
        return abi.encode(FaultProof({challengeId: ID, journal: journal_, seal: seal_}));
    }

    function _dismissedAt(HeldChallenges claims) private view returns (uint64 dismissedAt) {
        (,,,,,,,, dismissedAt) = claims.challenges(ID);
    }

    function test_the_verifier_takes_the_proof() public view {
        verifier.verify(seal, transition, sha256(journal));
    }

    function test_the_journal_is_read_as_iris_reads_it() public view {
        FaultVerdict memory verdict = RasterJournal.read(journal);
        require(verdict.finished, "not finished");
        require(verdict.transitionImageId == transition, "transition");
        require(verdict.authorizationImageId == authorization, "authorization");
        require(verdict.refutedTrace == trace, "trace");
        require(verdict.program == program, "program");
        require(verdict.inputManifest == inputs, "inputs");
    }

    function test_a_proof_of_the_challenge_dismisses_it_once() public {
        HeldChallenges claims = _held(trace, program, inputs);
        require(_dismissedAt(claims) == 0, "dismissed before the proof");
        claims.submitFaultProof(_proof(journal, seal));
        require(_dismissedAt(claims) == uint64(block.timestamp), "not dismissed");
        vm.expectRevert(abi.encodeWithSelector(IrisClaims.AlreadyDismissed.selector, ID));
        claims.submitFaultProof(_proof(journal, seal));
    }

    function test_a_challenge_never_admitted_is_unknown() public {
        HeldChallenges claims = new HeldChallenges(verifier, transition, authorization);
        vm.expectRevert(abi.encodeWithSelector(IrisClaims.UnknownChallenge.selector, ID));
        claims.submitFaultProof(_proof(journal, seal));
    }

    // The cheat: a real proof that some other trace is false.
    function test_a_proof_of_another_trace_is_refused() public {
        HeldChallenges claims = _held(~trace, program, inputs);
        vm.expectRevert(IrisClaims.AnotherTraceCommitment.selector);
        claims.submitFaultProof(_proof(journal, seal));
        require(_dismissedAt(claims) == 0, "dismissed");
    }

    // And a real proof that the trace is not a run of another program, or on
    // other inputs, which is true of every trace.
    function test_a_proof_over_another_program_is_refused() public {
        HeldChallenges claims = _held(trace, ~program, inputs);
        vm.expectRevert(IrisClaims.AnotherProgram.selector);
        claims.submitFaultProof(_proof(journal, seal));
    }

    function test_a_proof_over_other_inputs_is_refused() public {
        HeldChallenges claims = _held(trace, program, ~inputs);
        vm.expectRevert(IrisClaims.AnotherInputManifest.selector);
        claims.submitFaultProof(_proof(journal, seal));
    }

    function test_a_proof_checked_by_programs_not_pinned_is_refused() public {
        HeldChallenges claims = new HeldChallenges(verifier, ~transition, authorization);
        claims.hold(ID, trace, program, inputs);
        vm.expectRevert(IrisClaims.TransitionImageIdMismatch.selector);
        claims.submitFaultProof(_proof(journal, seal));

        claims = new HeldChallenges(verifier, transition, ~authorization);
        claims.hold(ID, trace, program, inputs);
        vm.expectRevert(IrisClaims.AuthorizationImageIdMismatch.selector);
        claims.submitFaultProof(_proof(journal, seal));
    }

    // Everything the contract reads from the journal can be right and the
    // proof still not be one: the seal is over the whole journal.
    function test_a_journal_or_a_seal_changed_is_not_verified() public {
        HeldChallenges claims = _held(trace, program, inputs);

        bytes memory changed = journal;
        changed[changed.length - 1] ^= 0x01;
        vm.expectRevert(IrisClaims.ProofNotVerified.selector);
        claims.submitFaultProof(_proof(changed, seal));

        changed = seal;
        changed[changed.length - 1] ^= 0x01;
        vm.expectRevert(IrisClaims.ProofNotVerified.selector);
        claims.submitFaultProof(_proof(journal, changed));

        // Another verifier's selector, and no seal at all.
        changed = seal;
        changed[0] ^= 0x01;
        vm.expectRevert(IrisClaims.ProofNotVerified.selector);
        claims.submitFaultProof(_proof(journal, changed));
        vm.expectRevert(IrisClaims.ProofNotVerified.selector);
        claims.submitFaultProof(_proof(journal, ""));

        require(_dismissedAt(claims) == 0, "dismissed");
    }

    function test_a_journal_cut_short_is_malformed() public {
        HeldChallenges claims = _held(trace, program, inputs);
        bytes memory cut = new bytes(64);
        for (uint256 i = 0; i < cut.length; i++) {
            cut[i] = journal[i];
        }
        vm.expectRevert(RasterJournal.MalformedJournal.selector);
        claims.submitFaultProof(_proof(cut, seal));
        vm.expectRevert(RasterJournal.MalformedJournal.selector);
        claims.submitFaultProof(_proof("", seal));
    }
}
