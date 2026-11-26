// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import "./IrisTypes.sol";
import "./IrisEncoding.sol";
import "./IrisRegistry.sol";
import "./DataRootOracle.sol";
import "./RasterJournal.sol";
import "./InputManifest.sol";
import {IRiscZeroVerifier} from "risc0/IRiscZeroVerifier.sol";

/// Where a claim and a challenge arrive, and the checks made of each on
/// arrival: `blockchain_rules::checks` and `::availability`, in the
/// order `iris claim check` makes them. Every error is the Rust rejection of
/// the same name.
///
/// Two things stand in for proofs here. A blob's availability proof is the
/// signature of a trusted attester over where the blob is, the root of its
/// block and what it holds. And Celestia's block roots come from a trusted
/// relayer through `DataRootOracle`.
///
/// A challenge is answered with a Raster fault proof of the challenger's
/// trace (`submitFaultProof`), which this contract verifies for real: the
/// Groth16 proof through RISC Zero's verifier, and what its journal says
/// against the challenge it holds.
///
/// What follows from any of it is not here: the clocks that close a claim or
/// a challenge, bonds, how several challenges of one claim are settled, and
/// what a contract does with a claim nobody successfully challenged.
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
    uint64 public immutable faultProofPeriod;
    /// RISC Zero's verifier of a Groth16 proof.
    IRiscZeroVerifier public immutable verifier;
    /// Raster's transition program: the one a fault proof is a proof of.
    bytes32 public immutable transitionImageId;
    /// Raster's authorization program: the one a fault proof's inputs were
    /// checked by.
    bytes32 public immutable authorizationImageId;

    struct Accepted {
        bytes32 procedure;
        address claimer;
        uint32 checkpointCount;
        bytes32 checkpointsRoot;
        uint64 submittedAt;
        /// The SHA-256 of the commitments of the run inputs the claim
        /// posted, in the order its blobs state them.
        bytes32 postedInputs;
    }

    struct Admitted {
        bytes32 claimId;
        address challenger;
        uint32 stage;
        bytes32 traceHeaderDigest;
        bytes32 claimedOutput;
        uint64 challengedAt;
        /// The disputed checkpoint's program and inputs: what a fault proof
        /// of the challenger's trace has to be about.
        bytes32 program;
        bytes32 inputCommitment;
        /// When a fault proof showed the challenger's trace false, or zero.
        uint64 dismissedAt;
    }

    /// What a challenger showed wrong with a claim on the spot, with no
    /// replay: `blockchain_rules::checks::StageAdmission` and
    /// `LinkFault`. The first stands for no fault.
    enum ClaimFault {
        None,
        // The stage's program is outside the procedure's approved set.
        UnapprovedProgram,
        // The stage's input list is not in its canonical form, or lacks an
        // input its program takes.
        UnreadableInputs,
        // An input the program takes from an earlier stage is not an earlier
        // stage's output: no producer is named, it is not earlier, it is not
        // of a kind that makes this input, or it made something else.
        NotAnEarlierOutput,
        // An input the program is given from fixed data is no commitment
        // the procedure pins for it.
        NotAPinnedValue,
        // An input the run gives is no value the claim posted.
        NotAPostedValue
    }

    /// A claim a challenger showed void, the stage it pointed at and when.
    struct Voided {
        ClaimFault fault;
        uint32 stage;
        uint64 voidedAt;
    }

    mapping(bytes32 => Accepted) public claims;
    mapping(bytes32 => Admitted) public challenges;
    mapping(bytes32 => Voided) public voided;

    event ClaimAccepted(
        bytes32 indexed claimId, bytes32 indexed procedure, address indexed claimer, bytes32 checkpointsRoot
    );
    event ChallengeAdmitted(
        bytes32 indexed challengeId, bytes32 indexed claimId, address indexed challenger, uint32 stage
    );
    event ChallengeDismissed(bytes32 indexed challengeId, bytes32 indexed claimId);
    event ClaimVoided(bytes32 indexed claimId, uint32 stage, ClaimFault fault);

    // The claim is of no registered procedure, or does not hold what a
    // claim of its procedure kind does.
    error UnregisteredProcedure(bytes32 id);
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
    error AnotherRelayed(uint256 inference);
    error AnotherForms(uint256 inference);
    error AnotherResult(uint256 inference);
    error AnotherConsensus();
    // What was posted.
    error NothingPosted();
    error NotIncluded(uint256 blob);
    error ExpiresTooSoon(uint256 blob);
    error NotProven(uint256 blob);
    error AnotherSubmitter(uint256 blob);
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
    // The shape of a challenger's trace: `raster_prover::challenge`'s
    // admission of it.
    error TraceTooShort();
    error WindowSizeNotProtocol();
    error NonCanonicalBitsPerItem();
    // A fault pointed at with no replay. The first two say there is none.
    error NoFault();
    error AlreadyVoided(bytes32 id);
    error NotTheInputManifest();
    error UnapprovedStage();
    error NotAnInput();
    error NotThePostedInputs();
    error ProducerNotPresented();
    // A fault proof.
    error UnknownChallenge(bytes32 id);
    error AlreadyDismissed(bytes32 id);
    error NotAFaultVerdict();
    error TransitionImageIdMismatch();
    error AuthorizationImageIdMismatch();
    error AnotherTraceCommitment();
    error AnotherProgram();
    error AnotherInputManifest();
    error ProofNotVerified();

    constructor(
        IrisRegistry registry_,
        DataRootOracle oracle_,
        address attester_,
        uint64 retention_,
        uint64 challengePeriod_,
        uint64 faultProofPeriod_,
        IRiscZeroVerifier verifier_,
        bytes32 transitionImageId_,
        bytes32 authorizationImageId_
    ) {
        registry = registry_;
        oracle = oracle_;
        require(attester_ != address(0), "no attester");
        attester = attester_;
        retention = retention_;
        challengePeriod = challengePeriod_;
        faultProofPeriod = faultProofPeriod_;
        require(address(verifier_) != address(0), "no verifier");
        verifier = verifier_;
        transitionImageId = transitionImageId_;
        authorizationImageId = authorizationImageId_;
    }

    // ----------------------------------------------------------------- claim

    /// Take a claim, ABI-encoded, and accept it or revert with why not.
    function submitClaim(bytes calldata encoded) external returns (bytes32 claimId) {
        Claim memory claim = abi.decode(encoded, (Claim));
        // The id is of the claim as decoded, encoded again: the same claim has
        // one id however the bytes it arrived in were laid out.
        claimId = sha256(abi.encode(claim));
        if (claims[claimId].submittedAt != 0) revert AlreadySubmitted(claimId);
        if (!registry.isRegistered(claim.procedure)) revert UnregisteredProcedure(claim.procedure);
        Procedure memory procedure = registry.procedure(claim.procedure);

        // One inference per model the procedure pins, and an agreement
        // exactly when its result states one.
        if (claim.inferences.length != procedure.models.length || claim.hasConsensus != procedure.agreement) {
            revert AnotherShape();
        }

        _checkInferences(procedure, claim);
        _checkOrigin(procedure, claim);
        _checkPosted(claim.availability, claim.claimer, uint64(block.timestamp) + challengePeriod);
        bytes32 postedInputs = _checkHolds(claim);
        _checkStatedRecords(procedure, claim);

        claims[claimId] = Accepted({
            procedure: claim.procedure,
            claimer: claim.claimer,
            checkpointCount: claim.checkpointCount,
            checkpointsRoot: claim.checkpointsRoot,
            submittedAt: uint64(block.timestamp),
            postedInputs: postedInputs
        });
        emit ClaimAccepted(claimId, claim.procedure, claim.claimer, claim.checkpointsRoot);
    }

    /// `check_submission`: each inference's statement is the procedure's
    /// model and parameters over the claimed input and source, its text is
    /// the one its record commits to, and its record is valid. An inference
    /// a relay gives its prompt is over that relay's text and record, and
    /// the claim holds one such record for each (`check_procedure_kind`).
    function _checkInferences(Procedure memory procedure, Claim memory claim) private pure {
        uint256 relay;
        for (uint256 i = 0; i < claim.inferences.length; i++) {
            Inference memory inference = claim.inferences[i];
            bytes32 input = claim.inputTextSha256;
            bytes32 source = claim.sourceSha256;
            if (procedure.relayedFrom[i] != 0) {
                if (relay == claim.relays.length) revert AnotherShape();
                input = claim.relays[relay].inputTextSha256;
                source = IrisEncoding.sourceId(claim.relays[relay].record);
                relay++;
            }
            bytes32 statement = IrisEncoding.statement(
                input,
                source,
                procedure.models[i].modelId,
                procedure.rawPrompt,
                procedure.maxNewTokens,
                procedure.models[i].rangeSize
            );
            if (inference.record.statement != statement) revert StatementMismatch(i);
            if (sha256(bytes(inference.outputText)) != inference.record.outputTextSha256) revert OutputMismatch(i);
            if (!inference.record.valid) revert NotValid(i);
        }
        if (relay != claim.relays.length) revert AnotherShape();
    }

    /// `check_origin`: the source record is the one the statement names and
    /// meets what the procedure requires, each result is what the
    /// procedure's forms give the submitted text, and so is the agreement.
    function _checkOrigin(Procedure memory procedure, Claim memory claim) private pure {
        if (claim.hasSource && IrisEncoding.sourceId(claim.source) != claim.sourceSha256) {
            revert AnotherSourceRecord();
        }
        if (!procedure.hasSource) {
            if (claim.sourceSha256 != bytes32(0)) revert UnexpectedSource();
        } else {
            if (!claim.hasSource) revert MissingSource();
            if (!claim.source.present || claim.source.fault != 0) revert SourceFault(claim.source.fault);
            if (!IrisEncoding.sameText(claim.source.signer, procedure.signer)) revert AnotherSigner();
            if (
                procedure.hasRecipe
                    && !IrisEncoding.sameText(claim.source.recipeSha256, IrisEncoding.hexText(procedure.recipe))
            ) revert AnotherRecipe();
        }

        // A relay's record has no fault, is of the text its inference's
        // statement is over, names the claim record of the inference it
        // relays, and is built with the recipe the procedure pins.
        uint256 relay;
        for (uint256 i = 0; i < claim.inferences.length; i++) {
            uint32 earlier = procedure.relayedFrom[i];
            if (earlier == 0) continue;
            Relay memory relayed = claim.relays[relay++];
            if (!relayed.record.present || relayed.record.fault != 0) revert SourceFault(relayed.record.fault);
            if (
                relayed.record.documents != 1
                    || !IrisEncoding.sameText(relayed.record.textSha256, IrisEncoding.hexText(relayed.inputTextSha256))
                    || !IrisEncoding.sameText(
                        relayed.record.documentsSha256,
                        IrisEncoding.hexText(IrisEncoding.relayed(claim.inferences[earlier - 1].record))
                    )
            ) revert AnotherRelayed(i);
            if (
                procedure.hasRelayRecipe
                    && !IrisEncoding.sameText(relayed.record.recipeSha256, IrisEncoding.hexText(procedure.relayRecipe))
            ) revert AnotherRecipe();
        }

        // An inference the result stage does not read has no result.
        uint32[] memory codes = new uint32[](claim.inferences.length);
        for (uint256 i = 0; i < claim.inferences.length; i++) {
            uint32 code;
            bytes32 forms;
            if (bytes(procedure.resultInputs[i]).length != 0) {
                (code, forms) = IrisEncoding.runForms(procedure.forms, claim.inferences[i].outputText);
            }
            if (claim.inferences[i].result.formsSha256 != forms) revert AnotherForms(i);
            if (claim.inferences[i].result.code != code) revert AnotherResult(i);
            codes[i] = code;
        }
        if (claim.hasConsensus) {
            // `consensus_of`: with no forms, the same text once trimmed; with
            // forms, the same form. Both records are valid and of one input,
            // which the checks above held.
            bool agree = procedure.forms.length == 0
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

    /// `check_holds`: the blobs hold the claim's checkpoint list. Returns
    /// `posted_inputs_digest`: the digest of the run inputs they hold. Which
    /// inputs a claim must have posted is not judged here: a stage input
    /// that was not posted is a fault of the checkpoint that names it
    /// (`voidClaim`).
    function _checkHolds(Claim memory claim) private pure returns (bytes32) {
        bool checkpoints;
        bytes memory posted;
        for (uint256 i = 0; i < claim.availability.length; i++) {
            BlobContents memory held = claim.availability[i].contents;
            checkpoints = checkpoints || _has(held.checkpointLists, claim.checkpointsRoot);
            posted = abi.encodePacked(posted, held.runInputs);
        }
        if (!checkpoints) revert CheckpointsNotPosted();
        return sha256(posted);
    }

    // --------------------------------------------------------------- records

    function _sameRecord(Record memory a, Record memory b) private pure returns (bool) {
        return a.statement == b.statement && a.outputTextSha256 == b.outputTextSha256 && a.valid == b.valid;
    }

    /// `check_stated_records`: the claim ends in its procedure's result stage,
    /// each record that stage read is the one in the check it read it from,
    /// and the results are what that stage states. The records were read out
    /// of the stages' outputs by the claim's proof; nothing is decoded here.
    /// An inference the result stage does not read is held by the relay that
    /// reads it (`_checkOrigin`).
    function _checkStatedRecords(Procedure memory procedure, Claim memory claim) private view {
        // The entry the claim gives in full is one a posted blob lists, and
        // is of the claim's own checkpoint list.
        StatedRecords memory stated = claim.stated;
        bytes32 digest = IrisEncoding.statedRecordsDigest(stated);
        bool listed;
        for (uint256 i = 0; i < claim.availability.length; i++) {
            listed = listed || _has(claim.availability[i].contents.records, digest);
        }
        if (!listed || stated.checkpointsRoot != claim.checkpointsRoot) revert RecordsNotStated();

        if (registry.stageOf(claim.procedure, stated.resultProgram) != keccak256(bytes(procedure.resultProgram))) {
            revert NotTheResultStage();
        }

        // The stage states the inferences it read, in order.
        if (stated.hasConsensus != claim.hasConsensus) revert AnotherStatedResult();
        uint256 read;
        for (uint256 i = 0; i < claim.inferences.length; i++) {
            if (bytes(procedure.resultInputs[i]).length == 0) continue;
            if (read == stated.results.length) revert AnotherStatedResult();
            StatedInference memory result = stated.results[read++];
            if (
                !_sameRecord(result.record, claim.inferences[i].record)
                    || result.result.code != claim.inferences[i].result.code
                    || result.result.formsSha256 != claim.inferences[i].result.formsSha256
            ) revert AnotherStatedResult();
        }
        if (read != stated.results.length) revert AnotherStatedResult();
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
            if (bytes(procedure.resultInputs[i]).length == 0) continue;
            (bool named, StatedCheck memory check) = _checkRead(stated, procedure.resultInputs[i]);
            if (!named) revert NoCheck(i);
            if (registry.stageOf(claim.procedure, check.program) != checkStage) revert NotACheck(i);
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

        _open(claim, challenge.stage, challenge.checkpoint, challenge.path);
        // A stage that names a program outside the approved set is not a
        // matter for a replay: it voids the claim by itself, and
        // `voidClaim` is how a challenger says so.
        if (registry.stageOf(claim.procedure, challenge.checkpoint.program) == bytes32(0)) {
            revert UnapprovedProgram(challenge.checkpoint.program);
        }
        if (challenge.claimedOutput == challenge.checkpoint.outputSha256) revert OutputNotDisputed();
        // The trace is in the one shape the protocol fixes for its length. A
        // window the challenger chose, or another packing of the fingerprint,
        // is a trace no defender is held to answer.
        uint64 window = IrisEncoding.protocolWindow(challenge.header.fingerprintLen);
        if (window == 0) revert TraceTooShort();
        if (challenge.header.windowSize != window) revert WindowSizeNotProtocol();
        if (challenge.header.bitsPerItem != IrisEncoding.protocolBitsPerItem(window)) {
            revert NonCanonicalBitsPerItem();
        }
        bytes32 traceHeaderDigest = IrisEncoding.traceHeaderDigest(challenge.header);

        _checkPosted(challenge.availability, challenge.challenger, uint64(block.timestamp) + faultProofPeriod);
        bool body;
        for (uint256 i = 0; i < challenge.availability.length; i++) {
            body = body || _has(challenge.availability[i].contents.traceBodies, traceHeaderDigest);
        }
        if (!body) revert TraceBodyNotPosted();

        challengeId = sha256(abi.encode(challenge));
        if (challenges[challengeId].challengedAt != 0) revert AlreadySubmitted(challengeId);
        challenges[challengeId] = Admitted({
            claimId: challenge.claimId,
            challenger: challenge.challenger,
            stage: challenge.stage,
            traceHeaderDigest: traceHeaderDigest,
            claimedOutput: challenge.claimedOutput,
            challengedAt: uint64(block.timestamp),
            program: challenge.checkpoint.program,
            inputCommitment: challenge.checkpoint.inputCommitment,
            dismissedAt: 0
        });
        emit ChallengeAdmitted(challengeId, challenge.claimId, challenge.challenger, challenge.stage);
    }

    /// Hold a presented checkpoint to its place in a claim's list.
    function _open(Accepted memory claim, uint32 stage, Checkpoint memory checkpoint, bytes32[] memory path)
        private
        pure
    {
        if (path.length != IrisEncoding.TREE_DEPTH || stage >= claim.checkpointCount) revert NotTheCheckpoint();
        bytes32 root =
            IrisEncoding.listRoot(IrisEncoding.checkpointHash(checkpoint), stage, claim.checkpointCount, path);
        if (root != claim.checkpointsRoot) revert NotTheCheckpoint();
    }

    // -----------------------------------------------------------------------
    // A fault pointed at, with no replay
    // -----------------------------------------------------------------------
    //
    // A replay runs the program a checkpoint names on the inputs it names, so
    // it says nothing of whether that is the program that should run or
    // whether those inputs came from where the claim says. A challenger
    // points at either, the contract reads the claim's own checkpoints, and
    // the claim is void in the same transaction: there is nothing to answer.
    // Anyone may point. What a void claim costs its claimer is not here.

    function _void(bytes32 claimId, uint32 stage, ClaimFault fault) private {
        voided[claimId] = Voided({fault: fault, stage: stage, voidedAt: uint64(block.timestamp)});
        emit ClaimVoided(claimId, stage, fault);
    }

    function _held(bytes32 claimId) private view returns (Accepted memory claim) {
        claim = claims[claimId];
        if (claim.submittedAt == 0) revert UnknownClaim(claimId);
        if (voided[claimId].voidedAt != 0) revert AlreadyVoided(claimId);
    }

    /// Show that a checkpoint of a claim is not one the procedure
    /// allows: `encoded` is a `CheckpointFault`, the claim's own checkpoint at
    /// a stage, opened, and what is asked of it.
    ///
    /// With no input named, the question is the stage's program: it must be
    /// one the procedure approves. With an input named, the question is
    /// where that input comes from. An input the program takes from an
    /// earlier stage must be that stage's output; an input bound to fixed
    /// data, such as the model's, must be a commitment the procedure
    /// pins; an input the run gives must be one the claim posted. The checks
    /// and their order are `CheckpointHashes::check_link`'s. Reverts with
    /// `NoFault` when the checkpoint is as it should be.
    ///
    /// So a stage's inputs are always ones a challenger also holds, and two
    /// parties that disagree about a stage ran it on the same inputs. What a
    /// posted input says is not judged, nor whether a held value is the one
    /// the pipeline gives that stage: the stage checks both as it runs, and a
    /// replay on the same inputs shows a stage that said otherwise.
    ///
    /// Nothing here is the challenger's word. The stage's checkpoint says
    /// which checkpoint produced the input, the input list it commits to says
    /// what the stage loaded, and the procedure says where the input may
    /// come from.
    function voidClaim(bytes calldata encoded) external {
        CheckpointFault memory fault = abi.decode(encoded, (CheckpointFault));
        Accepted memory claim = _held(fault.claimId);
        _open(claim, fault.stage, fault.checkpoint, fault.path);
        bool approved = registry.stageOf(claim.procedure, fault.checkpoint.program) != bytes32(0);
        ClaimFault found;
        if (bytes(fault.input).length == 0) {
            found = approved ? ClaimFault.None : ClaimFault.UnapprovedProgram;
        } else {
            if (sha256(fault.manifest) != fault.checkpoint.inputCommitment) revert NotTheInputManifest();
            // What an unapproved program takes is not defined: ask of its
            // program instead.
            if (!approved) revert UnapprovedStage();
            found = _inputFault(claim, fault);
        }
        if (found == ClaimFault.None) revert NoFault();
        _void(fault.claimId, fault.stage, found);
    }

    /// What is wrong with where a stage's input comes from, or `None`.
    function _inputFault(Accepted memory claim, CheckpointFault memory fault) private view returns (ClaimFault) {
        InputRule memory rule = registry.inputOf(claim.procedure, fault.checkpoint.program, fault.input);
        // A name the program's entry does not list is not an input of it.
        if (!rule.piped && !rule.pinned && !rule.run) revert NotAnInput();

        // A commitment is 32 bytes, written in the list as 64 lowercase hex
        // digits. Text that is not that is no commitment, and equals none.
        (bool readable, bool found, bool is32, bytes32 loaded) =
            InputManifest.valueOf(fault.manifest, bytes(fault.input));
        if (!readable || !found) return ClaimFault.UnreadableInputs;
        if (!rule.piped) {
            // Bound to fixed data: the model's, or another pinned identity.
            // The stage checks what it loads against that identity as it
            // runs, and that holds only for an input a challenger also has,
            // so the stage may name nothing but a pinned commitment.
            if (is32 && _has(rule.pinnedValues, loaded)) return ClaimFault.None;
            if (!rule.run) return ClaimFault.NotAPinnedValue;
            // Given by the run: the claimer posted it, or a challenger has
            // nothing to run the stage on.
            if (sha256(abi.encodePacked(fault.posted)) != claim.postedInputs) revert NotThePostedInputs();
            return is32 && _has(fault.posted, loaded) ? ClaimFault.None : ClaimFault.NotAPostedValue;
        }
        if (is32 && _has(rule.emptyValues, loaded)) return ClaimFault.None;
        (bool named, uint32 producer) = _producerOf(fault.checkpoint, fault.input);
        if (!named || producer >= fault.stage) return ClaimFault.NotAnEarlierOutput;
        if (!fault.hasProducer) revert ProducerNotPresented();
        _open(claim, producer, fault.producer, fault.producerPath);
        bytes32 kind = registry.stageOf(claim.procedure, fault.producer.program);
        if (kind == bytes32(0) || !_has(rule.producers, kind)) return ClaimFault.NotAnEarlierOutput;
        return is32 && fault.producer.outputCommitment == loaded ? ClaimFault.None : ClaimFault.NotAnEarlierOutput;
    }

    /// The checkpoint a stage's own checkpoint names as the producer of one
    /// of its inputs.
    function _producerOf(Checkpoint memory checkpoint, string memory input) private pure returns (bool, uint32) {
        for (uint256 i = 0; i < checkpoint.inputSources.length; i++) {
            if (IrisEncoding.sameText(checkpoint.inputSources[i].name, input)) {
                return (true, checkpoint.inputSources[i].position);
            }
        }
        return (false, 0);
    }

    // -----------------------------------------------------------------------
    // A fault proof
    // -----------------------------------------------------------------------

    /// Show a challenger's trace false: `encoded` is a `FaultProof`, a Raster
    /// fault proof as `iris fault prove` writes it.
    ///
    /// A challenge states that honest execution of the disputed checkpoint's
    /// program on its inputs gives another output, and posts the trace of
    /// that execution. A fault proof is a window of that trace re-executed in
    /// RISC Zero, ending at a step the trace gets wrong. So the challenge is
    /// dismissed when the proof is one of Raster's transition program, has
    /// reached that verdict, and is about this challenge: the trace it posted,
    /// the checkpoint's program and the checkpoint's inputs. These are the
    /// checks of `raster_core::pipeline::defend` for a refutation, made here.
    ///
    /// Anyone may send it: the proof says the same whoever holds it. No clock
    /// is read, and nothing follows from a dismissal yet.
    function submitFaultProof(bytes calldata encoded) external {
        FaultProof memory proof = abi.decode(encoded, (FaultProof));
        Admitted storage challenge = challenges[proof.challengeId];
        if (challenge.challengedAt == 0) revert UnknownChallenge(proof.challengeId);
        if (challenge.dismissedAt != 0) revert AlreadyDismissed(proof.challengeId);

        FaultVerdict memory verdict = RasterJournal.read(proof.journal);
        if (!verdict.finished) revert NotAFaultVerdict();
        // The journal names the programs its own earlier steps and its inputs
        // were checked by. A proof that names others was checked by nothing
        // this contract knows.
        if (verdict.transitionImageId != transitionImageId) revert TransitionImageIdMismatch();
        if (verdict.authorizationImageId != authorizationImageId) revert AuthorizationImageIdMismatch();
        if (verdict.refutedTrace != challenge.traceHeaderDigest) revert AnotherTraceCommitment();
        if (verdict.program != challenge.program) revert AnotherProgram();
        if (verdict.inputManifest != challenge.inputCommitment) revert AnotherInputManifest();

        // The verifier reverts on a proof it does not take, with its own
        // errors or with none.
        try verifier.verify(proof.seal, transitionImageId, sha256(proof.journal)) {}
        catch {
            revert ProofNotVerified();
        }

        challenge.dismissedAt = uint64(block.timestamp);
        emit ChallengeDismissed(proof.challengeId, challenge.claimId);
    }
}
