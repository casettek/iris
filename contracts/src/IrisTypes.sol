// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

// The values the contracts take, as `inference_artifacts` states them in Rust.
// A 32-byte hash is `bytes32` here and 64 hex digits in the Rust files. The
// one digest the stages compute over hex text, a source record's, keeps its
// text here, so that it is hashed as the stages hash it.

// --- a configuration: `inference_artifacts::registry::Configuration` ---

// A model the configuration pins. `commitments` is the part of the id's
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

struct Configuration {
    uint32 version;
    string pipeline;
    string family;
    // The part of the id's encoding that lists the approved programs: a
    // count, then each program's stage, id, chained inputs and run inputs.
    // The registry walks it for each program's stage.
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

// --- a claim: `inference_artifacts::contract::ClaimSubmission` ---

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

// `inference_artifacts::availability::StatedRecords`
struct StatedRecords {
    bytes32 checkpointsRoot;
    bytes32 resultProgram;
    NamedPosition[] resultInputs;
    StatedInference[] results;
    bool hasConsensus;
    Consensus consensus;
    StatedCheck[] checks;
}

// `inference_artifacts::availability::BlobContents`. Each stated-records
// entry is listed by its digest; a claim gives the one it relies on in full.
struct BlobContents {
    address submitter;
    bytes32[] sources;
    bytes32[] promptTexts;
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
    bytes32 configuration;
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

// --- a challenge: `inference_artifacts::contract::ChallengeSubmission` ---

// `inference_artifacts::checkpoint::Checkpoint`
struct Checkpoint {
    string stage;
    bytes32 program;
    bytes32 inputCommitment;
    NamedPosition[] inputSources;
    bytes32 outputCommitment;
    bytes32 outputSha256;
}

struct Challenge {
    bytes32 claimId;
    address challenger;
    uint32 stage;
    Checkpoint checkpoint;
    bytes32[] path;
    bytes32 traceHeaderDigest;
    bytes32 claimedOutput;
    PostedBlob[] availability;
}
