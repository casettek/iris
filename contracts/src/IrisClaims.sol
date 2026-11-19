// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import "./IrisTypes.sol";
import "./IrisEncoding.sol";
import "./IrisRegistry.sol";
import "./DataRootOracle.sol";

/// Where a claim and a challenge arrive, and the checks made of each on
/// arrival: `inference_artifacts::contract` and `::availability`, in the
/// order `iris claim check` makes them. Every error is the Rust rejection of
/// the same name.
///
/// Two things stand in for proofs here. A blob's availability proof is the
/// signature of a trusted attester over where the blob is, the root of its
/// block and what it holds. And Celestia's block roots come from a trusted
/// relayer through `DataRootOracle`.
///
/// What happens after arrival is not here: the defence of a challenge, the
/// clocks that close a claim, bonds, and what a contract does with a claim
/// nobody successfully challenged.
contract IrisClaims {
    IrisRegistry public immutable registry;
    DataRootOracle public immutable oracle;
    /// Whose signature stands for a blob's availability proof.
    address public immutable attester;
    /// How long Celestia keeps a blob retrievable after its block, in seconds.
    uint64 public immutable retention;
    /// How long a claim stays open to challenge.
    uint64 public immutable challengePeriod;
    /// How long a defender has to answer a challenge.
    uint64 public immutable faultProofWindow;

    struct Accepted {
        bytes32 configuration;
        address claimer;
        uint32 checkpointCount;
        bytes32 checkpointsRoot;
        uint64 submittedAt;
    }

    struct Admitted {
        bytes32 claimId;
        address challenger;
        uint32 stage;
        bytes32 traceHeaderDigest;
        bytes32 claimedOutput;
        uint64 challengedAt;
    }

    mapping(bytes32 => Accepted) public claims;
    mapping(bytes32 => Admitted) public challenges;

    event ClaimAccepted(
        bytes32 indexed claimId, bytes32 indexed configuration, address indexed claimer, bytes32 checkpointsRoot
    );
    event ChallengeAdmitted(
        bytes32 indexed challengeId, bytes32 indexed claimId, address indexed challenger, uint32 stage
    );

    // The claim is of no registered configuration, or does not hold what a
    // claim of its pipeline does.
    error UnregisteredConfiguration(bytes32 id);
    error AnotherShape();
    error AlreadySubmitted(bytes32 id);
    // An inference's statement, output or verdict.
    error StatementMismatch(uint256 inference);
    error OutputMismatch(uint256 inference);
    error NotValid(uint256 inference);
    // The source, the forms, the result and the agreement.
    error AnotherSourceRecord();
    error UnexpectedSource();
    error MissingSource();
    error SourceFault(uint32 fault);
    error AnotherSigner();
    error AnotherRecipe();
    error AnotherForms(uint256 inference);
    error AnotherResult(uint256 inference);
    error AnotherConsensus();
    // What was posted.
    error NothingPosted();
    error NotIncluded(uint256 blob);
    error ExpiresTooSoon(uint256 blob);
    error NotProven(uint256 blob);
    error AnotherSubmitter(uint256 blob);
    error SourceNotPosted();
    error PromptTextNotPosted();
    error CheckpointsNotPosted();
    error TraceBodyNotPosted();
    // The records the claim's proof states.
    error RecordsNotStated();
    error NotTheResultStage();
    error AnotherStatedResult();
    error NoCheck(uint256 inference);
    error NotACheck(uint256 inference);
    error AnotherStatedRecord(uint256 inference);
    // A challenge.
    error UnknownClaim(bytes32 id);
    error ChallengePeriodOver();
    error NotTheCheckpoint();
    error UnapprovedProgram(bytes32 program);
    error OutputNotDisputed();

    constructor(
        IrisRegistry registry_,
        DataRootOracle oracle_,
        address attester_,
        uint64 retention_,
        uint64 challengePeriod_,
        uint64 faultProofWindow_
    ) {
        registry = registry_;
        oracle = oracle_;
        require(attester_ != address(0), "no attester");
        attester = attester_;
        retention = retention_;
        challengePeriod = challengePeriod_;
        faultProofWindow = faultProofWindow_;
    }

    // ----------------------------------------------------------------- claim

    /// Take a claim, ABI-encoded, and accept it or revert with why not.
    function submitClaim(bytes calldata encoded) external returns (bytes32 claimId) {
        Claim memory claim = abi.decode(encoded, (Claim));
        // The id is of the claim as decoded, encoded again: the same claim has
        // one id however the bytes it arrived in were laid out.
        claimId = sha256(abi.encode(claim));
        if (claims[claimId].submittedAt != 0) revert AlreadySubmitted(claimId);
        if (!registry.isRegistered(claim.configuration)) revert UnregisteredConfiguration(claim.configuration);
        Configuration memory configuration = registry.configuration(claim.configuration);

        bool consensus = IrisEncoding.sameText(configuration.pipeline, "consensus");
        uint256 inferences = consensus ? 2 : 1;
        if (
            claim.inferences.length != inferences || configuration.models.length != inferences
                || claim.hasConsensus != consensus
        ) revert AnotherShape();

        _checkInferences(configuration, claim);
        _checkOrigin(configuration, claim);
        _checkPosted(claim.availability, claim.claimer, uint64(block.timestamp) + challengePeriod);
        _checkHolds(claim);
        _checkStatedRecords(consensus, claim);

        claims[claimId] = Accepted({
            configuration: claim.configuration,
            claimer: claim.claimer,
            checkpointCount: claim.checkpointCount,
            checkpointsRoot: claim.checkpointsRoot,
            submittedAt: uint64(block.timestamp)
        });
        emit ClaimAccepted(claimId, claim.configuration, claim.claimer, claim.checkpointsRoot);
    }

    /// `check_submission`: each inference's statement is the configuration's
    /// model and parameters over the claimed input and source, its text is
    /// the one its record commits to, and its record is valid.
    function _checkInferences(Configuration memory configuration, Claim memory claim) private pure {
        for (uint256 i = 0; i < claim.inferences.length; i++) {
            Inference memory inference = claim.inferences[i];
            bytes32 statement = IrisEncoding.statement(
                claim.inputTextSha256,
                claim.sourceSha256,
                configuration.models[i].modelId,
                configuration.rawPrompt,
                configuration.maxNewTokens,
                configuration.models[i].rangeSize
            );
            if (inference.record.statement != statement) revert StatementMismatch(i);
            if (sha256(bytes(inference.outputText)) != inference.record.outputTextSha256) revert OutputMismatch(i);
            if (!inference.record.valid) revert NotValid(i);
        }
    }

    /// `check_origin`: the source record is the one the statement names and
    /// meets what the configuration requires, each result is what the
    /// configuration's forms give the submitted text, and so is the agreement.
    function _checkOrigin(Configuration memory configuration, Claim memory claim) private pure {
        if (claim.hasSource && IrisEncoding.sourceId(claim.source) != claim.sourceSha256) {
            revert AnotherSourceRecord();
        }
        if (!configuration.hasSource) {
            if (claim.sourceSha256 != bytes32(0)) revert UnexpectedSource();
        } else {
            if (!claim.hasSource) revert MissingSource();
            if (!claim.source.present || claim.source.fault != 0) revert SourceFault(claim.source.fault);
            if (!IrisEncoding.sameText(claim.source.signer, configuration.signer)) revert AnotherSigner();
            if (
                configuration.hasRecipe
                    && !IrisEncoding.sameText(claim.source.recipeSha256, IrisEncoding.hexText(configuration.recipe))
            ) revert AnotherRecipe();
        }

        uint32[2] memory codes;
        for (uint256 i = 0; i < claim.inferences.length; i++) {
            (uint32 code, bytes32 forms) = IrisEncoding.runForms(configuration.forms, claim.inferences[i].outputText);
            if (claim.inferences[i].result.formsSha256 != forms) revert AnotherForms(i);
            if (claim.inferences[i].result.code != code) revert AnotherResult(i);
            codes[i] = code;
        }
        if (claim.hasConsensus) {
            // `consensus_of`: with no forms, the same text once trimmed; with
            // forms, the same form. Both records are valid and of one input,
            // which the checks above held.
            bool agree = configuration.forms.length == 0
                ? keccak256(IrisEncoding.trim(bytes(claim.inferences[0].outputText)))
                    == keccak256(IrisEncoding.trim(bytes(claim.inferences[1].outputText)))
                : codes[0] != 0 && codes[0] == codes[1];
            uint32 code = agree ? codes[0] : 0;
            if (!claim.consensus.sameInput || claim.consensus.agree != agree || claim.consensus.code != code) {
                revert AnotherConsensus();
            }
        }
    }

    // ---------------------------------------------------------------- posted

    /// `check_posted`: every blob is in a block whose root the oracle has, is
    /// retrievable until `neededUntil`, is attested, and was posted for
    /// `submitter`.
    function _checkPosted(PostedBlob[] memory blobs, address submitter, uint64 neededUntil) private view {
        if (blobs.length == 0) revert NothingPosted();
        for (uint256 i = 0; i < blobs.length; i++) {
            (bytes32 dataRoot, uint64 time) = oracle.blockOf(blobs[i].height);
            if (time == 0 || dataRoot != blobs[i].dataRoot) revert NotIncluded(i);
            if (time + retention < neededUntil) revert ExpiresTooSoon(i);
            if (_signer(IrisEncoding.attestationDigest(blobs[i]), blobs[i].signature) != attester) {
                revert NotProven(i);
            }
            if (blobs[i].contents.submitter != submitter) revert AnotherSubmitter(i);
        }
    }

    function _signer(bytes32 digest, bytes memory signature) private pure returns (address) {
        if (signature.length != 65) {
            return address(0);
        }
        bytes32 r;
        bytes32 s;
        uint8 v;
        assembly {
            r := mload(add(signature, 0x20))
            s := mload(add(signature, 0x40))
            v := byte(0, mload(add(signature, 0x60)))
        }
        return ecrecover(digest, v, r, s);
    }

    function _has(bytes32[] memory values, bytes32 wanted) private pure returns (bool) {
        for (uint256 i = 0; i < values.length; i++) {
            if (values[i] == wanted) return true;
        }
        return false;
    }

    /// `check_holds`: the blobs hold the new information the statement names
    /// and the claim's checkpoint list.
    function _checkHolds(Claim memory claim) private pure {
        bool information;
        bool checkpoints;
        for (uint256 i = 0; i < claim.availability.length; i++) {
            BlobContents memory held = claim.availability[i].contents;
            information = information
                || (
                    claim.sourceSha256 == bytes32(0)
                        ? _has(held.promptTexts, claim.inputTextSha256)
                        : _has(held.sources, claim.sourceSha256)
                );
            checkpoints = checkpoints || _has(held.checkpointLists, claim.checkpointsRoot);
        }
        if (!information) {
            if (claim.sourceSha256 == bytes32(0)) revert PromptTextNotPosted();
            revert SourceNotPosted();
        }
        if (!checkpoints) revert CheckpointsNotPosted();
    }

    // --------------------------------------------------------------- records

    function _sameRecord(Record memory a, Record memory b) private pure returns (bool) {
        return a.statement == b.statement && a.outputTextSha256 == b.outputTextSha256 && a.valid == b.valid;
    }

    /// `check_stated_records`: the claim ends in its pipeline's result stage,
    /// each record is the one in the check that stage read, and the results
    /// are what that stage states. The records were read out of the stages'
    /// outputs by the claim's proof; nothing is decoded here.
    function _checkStatedRecords(bool consensus, Claim memory claim) private view {
        // The entry the claim gives in full is one a posted blob lists, and
        // is of the claim's own checkpoint list.
        StatedRecords memory stated = claim.stated;
        bytes32 digest = IrisEncoding.statedRecordsDigest(stated);
        bool listed;
        for (uint256 i = 0; i < claim.availability.length; i++) {
            listed = listed || _has(claim.availability[i].contents.records, digest);
        }
        if (!listed || stated.checkpointsRoot != claim.checkpointsRoot) revert RecordsNotStated();

        bytes32 resultStage = keccak256(consensus ? bytes("consensus-result") : bytes("claim-result"));
        if (registry.stageOf(claim.configuration, stated.resultProgram) != resultStage) revert NotTheResultStage();

        if (stated.results.length != claim.inferences.length || stated.hasConsensus != claim.hasConsensus) {
            revert AnotherStatedResult();
        }
        for (uint256 i = 0; i < claim.inferences.length; i++) {
            if (
                !_sameRecord(stated.results[i].record, claim.inferences[i].record)
                    || stated.results[i].result.code != claim.inferences[i].result.code
                    || stated.results[i].result.formsSha256 != claim.inferences[i].result.formsSha256
            ) revert AnotherStatedResult();
        }
        if (
            claim.hasConsensus
                && (
                    stated.consensus.sameInput != claim.consensus.sameInput
                        || stated.consensus.agree != claim.consensus.agree
                        || stated.consensus.code != claim.consensus.code
                )
        ) revert AnotherStatedResult();

        bytes32 checkStage = keccak256("sequence-check");
        for (uint256 i = 0; i < claim.inferences.length; i++) {
            string memory input = consensus ? (i == 0 ? "first" : "second") : "check";
            (bool named, StatedCheck memory check) = _checkRead(stated, input);
            if (!named) revert NoCheck(i);
            if (registry.stageOf(claim.configuration, check.program) != checkStage) revert NotACheck(i);
            if (!_sameRecord(check.record, claim.inferences[i].record)) revert AnotherStatedRecord(i);
        }
    }

    /// The check the last stage names as `input`, when the proof states its
    /// record.
    function _checkRead(StatedRecords memory stated, string memory input)
        private
        pure
        returns (bool named, StatedCheck memory check)
    {
        for (uint256 i = 0; i < stated.resultInputs.length; i++) {
            if (!IrisEncoding.sameText(stated.resultInputs[i].name, input)) continue;
            for (uint256 j = 0; j < stated.checks.length; j++) {
                if (stated.checks[j].checkpoint == stated.resultInputs[i].position) {
                    return (true, stated.checks[j]);
                }
            }
            return (false, check);
        }
    }

    // ------------------------------------------------------------- challenge

    /// Take a challenge, ABI-encoded, and admit it or revert with why not. It
    /// disputes the output of one stage of an accepted claim: the challenger
    /// opens the claim's checkpoint at that stage, states the output its own
    /// run of the stage ends in, and has posted the body of its trace.
    function submitChallenge(bytes calldata encoded) external returns (bytes32 challengeId) {
        Challenge memory challenge = abi.decode(encoded, (Challenge));
        Accepted memory claim = claims[challenge.claimId];
        if (claim.submittedAt == 0) revert UnknownClaim(challenge.claimId);
        if (block.timestamp > claim.submittedAt + challengePeriod) revert ChallengePeriodOver();

        if (challenge.path.length != IrisEncoding.TREE_DEPTH || challenge.stage >= claim.checkpointCount) {
            revert NotTheCheckpoint();
        }
        bytes32 root = IrisEncoding.listRoot(
            IrisEncoding.checkpointHash(challenge.checkpoint), challenge.stage, claim.checkpointCount, challenge.path
        );
        if (root != claim.checkpointsRoot) revert NotTheCheckpoint();
        // A stage that names a program outside the approved set is not a
        // matter for a replay: it voids the claim by itself, which is the
        // lifecycle's to act on.
        if (registry.stageOf(claim.configuration, challenge.checkpoint.program) == bytes32(0)) {
            revert UnapprovedProgram(challenge.checkpoint.program);
        }
        if (challenge.claimedOutput == challenge.checkpoint.outputSha256) revert OutputNotDisputed();

        _checkPosted(challenge.availability, challenge.challenger, uint64(block.timestamp) + faultProofWindow);
        bool body;
        for (uint256 i = 0; i < challenge.availability.length; i++) {
            body = body || _has(challenge.availability[i].contents.traceBodies, challenge.traceHeaderDigest);
        }
        if (!body) revert TraceBodyNotPosted();

        challengeId = sha256(abi.encode(challenge));
        if (challenges[challengeId].challengedAt != 0) revert AlreadySubmitted(challengeId);
        challenges[challengeId] = Admitted({
            claimId: challenge.claimId,
            challenger: challenge.challenger,
            stage: challenge.stage,
            traceHeaderDigest: challenge.traceHeaderDigest,
            claimedOutput: challenge.claimedOutput,
            challengedAt: uint64(block.timestamp)
        });
        emit ChallengeAdmitted(challengeId, challenge.claimId, challenge.challenger, challenge.stage);
    }
}
