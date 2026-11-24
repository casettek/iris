// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

// The values the contracts take, as `blockchain_rules` states them in Rust.
// A 32-byte hash is `bytes32` here and 64 hex digits in the Rust files. The
// one digest the stages compute over hex text, a source record's, keeps its
// text here, so that it is hashed as the stages hash it.

// --- a procedure: `blockchain_rules::registry::Procedure` ---

// A model the procedure pins. `commitments` is the part of the id's
// encoding that lists what the model id hashes: a count, then each role and
// commitment.
struct Model {
    bytes32 modelId;
    bytes commitments;
    uint32 rangeSize;
}

struct Form {
    string text;
    uint32 code;
}

struct Procedure {
    uint32 version;
    string procedureKind;
    string family;
    // The part of the id's encoding that lists the approved programs: a
    // count, then each program's stage, id, chained inputs, run inputs and
    // pinned inputs. The registry walks it for each program's stage.
    bytes programs;
    Model[] models;
    bool rawPrompt;
    uint32 maxNewTokens;
    bool hasSource;
    string signer;
    bool hasRecipe;
    bytes32 recipe;
    Form[] forms;
}

// What a procedure says of one input of one of its programs: where the
// input may come from. An input can be of more than one kind.
struct InputRule {
    // Taken from an earlier stage: the stages that may produce it, each as
    // the hash of its name, and the empty values it may hold in place of one.
    bool chained;
    bytes32[] producers;
    bytes32[] emptyValues;
    // Bound to fixed data, such as the model's: the commitments it may hold.
    bool pinned;
    bytes32[] pinnedValues;
    // Given by the run itself.
    bool run;
}

// --- a claim: `blockchain_rules::checks::ClaimSubmission` ---

struct Record {
    bytes32 statement;
    bytes32 outputTextSha256;
    bool valid;
}

struct Result {
    uint32 code;
    bytes32 formsSha256;
}

struct Inference {
    string outputText;
    Record record;
    Result result;
}

struct Consensus {
    bool sameInput;
    bool agree;
    uint32 code;
}

// A source record, with its hashes as the hex text the stages hash.
struct SourceRecord {
    bool present;
    string textSha256;
    uint64 textTotal;
    string recipeSha256;
    uint32 documents;
    string documentsSha256;
    string signer;
    uint32 fault;
    uint64 faultPosition;
}

struct NamedPosition {
    string name;
    uint32 position;
}

struct StatedInference {
    Record record;
    Result result;
}

struct StatedCheck {
    uint32 checkpoint;
    bytes32 program;
    Record record;
}

// `blockchain_rules::availability::StatedRecords`
struct StatedRecords {
    bytes32 checkpointsRoot;
    bytes32 resultProgram;
    NamedPosition[] resultInputs;
    StatedInference[] results;
    bool hasConsensus;
    Consensus consensus;
    StatedCheck[] checks;
}

// `blockchain_rules::availability::BlobContents`. Each stated-records
// entry is listed by its digest; a claim gives the one it relies on in full.
struct BlobContents {
    address submitter;
    // The commitment of each run input in the blob: what a stage's input
    // list names it by.
    bytes32[] runInputs;
    bytes32[] checkpointLists;
    bytes32[] traceBodies;
    bytes32[] records;
}

// A posted blob: where it is on Celestia, the root of its block, what it
// holds, and the signature of the attester over all of it.
struct PostedBlob {
    uint64 height;
    bytes29 namespace;
    bytes32 commitment;
    uint64 length;
    bytes32 dataRoot;
    BlobContents contents;
    bytes signature;
}

struct Claim {
    bytes32 procedure;
    address claimer;
    bytes32 inputTextSha256;
    bytes32 sourceSha256;
    bool hasSource;
    SourceRecord source;
    Inference[] inferences;
    bool hasConsensus;
    Consensus consensus;
    uint32 checkpointCount;
    bytes32 checkpointsRoot;
    // What the claim's proof states of its records, which one of the posted
    // blobs must list by digest.
    StatedRecords stated;
    PostedBlob[] availability;
}

// --- a challenge: `blockchain_rules::checks::ChallengeSubmission` ---

// `procedure_definition::checkpoint::Checkpoint`
struct Checkpoint {
    string stage;
    bytes32 program;
    bytes32 inputCommitment;
    NamedPosition[] inputSources;
    bytes32 outputCommitment;
    bytes32 outputSha256;
}

// The header of a Raster trace commitment:
// `raster_core::transition::TraceCommitmentHeader`. A challenge gives it
// whole, so that a contract can hold its shape to the protocol's and make the
// digest the posted trace body is found by.
struct TraceHeader {
    uint64 bitsPerItem;
    uint64 fingerprintLen;
    bytes32 fingerprintRoot;
    bytes32 revealedItems;
    uint64 windowSize;
    bytes32 revealedTailRoots;
}

struct Challenge {
    bytes32 claimId;
    address challenger;
    uint32 stage;
    Checkpoint checkpoint;
    bytes32[] path;
    TraceHeader header;
    bytes32 claimedOutput;
    PostedBlob[] availability;
}

// --- a fault a challenger points at, with no replay ---

// A checkpoint that is not one the procedure allows: the claim's own
// checkpoint at a stage, opened, and what is asked of it.
//
// With `input` empty the question is the stage's program, and nothing after
// `path` is read. With an input named the question is where that input comes
// from: `manifest` is the bytes of the input list the checkpoint commits to,
// and `producer` the checkpoint its own checkpoint names as the input's
// producer, when it names one. For an input the run gives, `posted` is the
// commitment of every run input the claim posted, in the order its blobs
// state them: the contract holds their digest.
struct CheckpointFault {
    bytes32 claimId;
    uint32 stage;
    Checkpoint checkpoint;
    bytes32[] path;
    string input;
    bytes manifest;
    bool hasProducer;
    Checkpoint producer;
    bytes32[] producerPath;
    bytes32[] posted;
}

/// A Raster fault proof of a challenger's trace: the journal of Raster's
/// transition program as RISC Zero wrote it, and the Groth16 seal over it as
/// RISC Zero's verifier takes it (the verifier's selector, then the proof).
struct FaultProof {
    bytes32 challengeId;
    bytes journal;
    bytes seal;
}
