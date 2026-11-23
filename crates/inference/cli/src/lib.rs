use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, ExitCode, ExitStatus};

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};

mod arrangement;
mod availability;
mod celestia;
mod challenge;
mod challenge_profile;
mod claim;
mod contract;
mod convert;
mod devstack;
mod evm;
mod fault;
mod infer;
mod parity;
mod posted;
#[cfg(feature = "replay-profile")]
mod replay_profile;
mod run_inputs;
mod source;
mod stage_bindings;
#[cfg(test)]
mod test_support;

pub use challenge::{
    build_challenge, ChallengeBuildOptions, ChallengeBuildOutcome, ChallengeBuildResult,
    ChallengeInput,
};
pub use claim::{
    build_claim, corrupt_checkpoint_hashes, ClaimBuildOptions, ClaimBuildResult,
    CorruptCheckpointHashesOptions, CorruptCheckpointHashesResult, CorruptCheckpointSelection,
};
pub use contract::{check_claim, CheckLine, ClaimCheckOptions, ClaimCheckReport};
pub use infer::run_infer;
pub use inference_artifacts::{
    build_checkpoint_trace, write_claim_artifacts, ChallengeBundle, ChallengeTrace, Checkpoint,
    CheckpointTrace, ClaimBundle, ClaimEndpoint, Divergence, DivergenceReason, ReplayPackage,
    CHALLENGE_BUNDLE_JSON, CHALLENGE_TRACE_JSON, CHECKPOINT_HASHES_TXT, CHECKPOINT_TRACE_JSON,
    CLAIM_BUNDLE_JSON, DIVERGENCE_JSON, REPLAY_PACKAGE_JSON,
};
#[cfg(feature = "replay-profile")]
pub use replay_profile::{run_replay_profile, ReplayProfileOptions};

#[derive(Debug, Parser)]
#[command(name = "iris")]
struct Cli {
    /// Persistent auxiliary worker; value is its retained-weight byte budget.
    #[arg(long, hide = true)]
    aux_worker: Option<usize>,

    #[command(subcommand)]
    command: Option<Commands>,

    /// Run one staged-infer stage and publish its canonical output artifact.
    #[arg(long = "run-stage", alias = "run-prefill-range-stage", hide = true)]
    run_stage: Option<PathBuf>,

    /// Compare one already-completed no-auth stage without rerunning the chain.
    #[arg(long = "compare-stage", alias = "compare-prefill-range-stage")]
    compare_stage: Option<PathBuf>,

    #[arg(long, hide = true)]
    input: Option<PathBuf>,

    #[arg(long = "input-manifest", hide = true)]
    input_manifest: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Model preparation workflows.
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    /// Proposer claim workflows.
    Claim {
        #[command(subcommand)]
        command: ClaimCommand,
    },
    /// Fast unconstrained deterministic inference.
    Infer(InferArgs),
    /// The host signer: a trusted stand-in for a notary.
    Attest {
        #[command(subcommand)]
        command: source::AttestCommand,
    },
    /// A run's source, without running the model.
    Source {
        #[command(subcommand)]
        command: source::SourceCommand,
    },
    /// Measure the heaviest tile replays in one exact stage instance.
    #[cfg(feature = "replay-profile")]
    ReplayProfile(ReplayProfileArgs),
    /// Run the pinned three-path end-to-end parity suite.
    TestParity {
        /// Leave out the challenge leg, whose authenticated replays take about an
        /// hour. The report records that it was skipped.
        #[arg(long)]
        skip_challenges: bool,
        /// The bundle under `model-bundles` to run: one family's small model.
        #[arg(long, default_value = "parity-gemma")]
        bundle: String,
    },
    /// Compare a chain run with direct generation of the same prompt and
    /// output, position by position.
    #[command(hide = true)]
    CompareTeacherForced {
        #[arg(long = "teacher-forced")]
        teacher_forced: PathBuf,
        #[arg(long)]
        direct: PathBuf,
    },
    #[command(hide = true)]
    ParityWorker {
        #[arg(long)]
        run: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        direct: bool,
    },
    /// Challenge workflows reserved for the verifier path.
    Challenge {
        #[command(subcommand)]
        command: ChallengeCommand,
    },
    /// The registry of configurations claims are made of.
    Registry {
        #[command(subcommand)]
        command: RegistryCommand,
    },
    /// The data-availability layer: what a relayer and an attester need.
    Da {
        #[command(subcommand)]
        command: DaCommand,
    },
    /// The calldata of the calls iris's contracts take. Nothing is sent.
    Evm {
        #[command(subcommand)]
        command: EvmCommand,
    },
    /// Drive the local chains (devstack/README.md): send iris's calls to the
    /// contracts deployed there, and hold the contracts to iris. For the
    /// local stack and its tests only: transactions go out through an
    /// account the development node keeps unlocked.
    Devstack {
        #[command(subcommand)]
        command: DevstackCommand,
    },
    /// Fault proof workflows reserved for the proof path.
    Fault {
        #[command(subcommand)]
        command: FaultCommand,
    },
}

#[derive(Debug, Subcommand)]
enum ModelCommand {
    /// Import a model bundle into committed Raster inference externals.
    Import(ModelImportArgs),
    /// Convert a GGUF model (Q8_0 weights) into a model bundle.
    Convert(ModelConvertArgs),
}

#[derive(Debug, Args)]
struct ModelConvertArgs {
    /// The GGUF file.
    #[arg(long)]
    gguf: PathBuf,
    /// Directory holding the model's config.json and tokenizer.json, copied
    /// into the bundle.
    #[arg(long)]
    metadata: PathBuf,
    /// Bundle directory to write.
    #[arg(long)]
    out: PathBuf,
    /// Write block tensors as dequantized Q16.16 values instead of Q8_0.
    #[arg(long, hide = true)]
    dequantize: bool,
}

#[derive(Debug, Args)]
struct ModelImportArgs {
    #[arg(long = "model")]
    model_dir: PathBuf,
    /// Also save a copy of the model-specific Raster template at this path.
    #[arg(long)]
    manifest: Option<PathBuf>,
    #[arg(long = "artifact-root", default_value = inference_artifacts::MODEL_ARTIFACTS_DIR)]
    artifact_root: PathBuf,
    #[arg(long = "model-id")]
    model_id: Option<String>,
    #[arg(long = "only-tokenizer")]
    only_tokenizer: bool,
    #[arg(long = "only-layers")]
    only_layers: bool,
    #[arg(long = "only-embedding")]
    only_embedding: bool,
    #[arg(long = "only-ple")]
    only_ple: bool,
    #[arg(long = "only-direct")]
    only_direct: bool,
    /// Accepted and ignored: import no longer writes fixtures into stage directories.
    #[arg(long)]
    no_stage_fixtures: bool,
    /// Largest range, in prompt tokens or claimed ids.
    #[arg(long = "range-size", default_value_t = model_import::DEFAULT_RANGE_SIZE)]
    range_size: u32,
}

#[derive(Debug, Args)]
struct InferArgs {
    /// Run spec to execute.
    #[arg(long, default_value = inference_artifacts::INFERENCE_RUN_SPEC_TOML)]
    run: PathBuf,
}

#[cfg(feature = "replay-profile")]
#[derive(Debug, Args)]
struct ReplayProfileArgs {
    #[arg(long, default_value = inference_artifacts::INFERENCE_RUN_SPEC_TOML)]
    run: PathBuf,
    /// Exact expanded stage name, e.g. range_p0_l13 or prompt_check_p0.
    #[arg(long)]
    stage: String,
    /// Number of distinct tiles selected by their slowest native invocation.
    #[arg(long, default_value = "3")]
    top_n: std::num::NonZeroUsize,
}

#[derive(Debug, Subcommand)]
enum ClaimCommand {
    /// Build a staged-infer checkpoint claim.
    Build(ClaimBuildArgs),
    /// Copy and corrupt one claimed checkpoint hash for manual challenge
    /// tests. With `--submission`, write the claim a lying claimer would
    /// submit: another output stated for one checkpoint, with its own list.
    Corrupt(ClaimCorruptHashesArgs),
    /// Read a submitted claim's data back from the data-availability layer
    /// into a new directory, with a run spec over what it posted: what a
    /// challenger starts from.
    Fetch(ClaimFetchArgs),
    /// Ask of a fetched claim what no replay shows: whether every checkpoint
    /// names an approved program, and whether every input a stage takes from
    /// an earlier one is that stage's output. Nothing is run.
    Audit(ClaimAuditArgs),
    /// Post the data `claim build` prepared, and write where it is into the
    /// claim's submission.
    Post(ClaimPostArgs),
    /// Make the checks a contract would make on a claim's submission.
    Check(ClaimCheckArgs),
    /// Write what a contract of a run's pipeline over its models would pin.
    Pins(ClaimPinsArgs),
}

#[derive(Debug, Args)]
struct ClaimPostArgs {
    /// The `claim_submission.json` that `claim build` wrote. The blobs to
    /// post are beside it (`posting-blob-<n>.bin`).
    #[arg(long)]
    submission: PathBuf,

    /// The stand-in data-availability layer's directory. Default:
    /// `IRIS_DA_DIR`, or `~/.cache/iris/stub-celestia`.
    #[arg(long = "da-dir")]
    da_dir: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct ClaimAuditArgs {
    /// The claim's `claim_submission.json`, in the directory `iris claim
    /// fetch` wrote: with its checkpoint list, checkpoints and input lists.
    #[arg(long)]
    submission: PathBuf,

    /// A run spec of the configuration the claim names.
    #[arg(long, default_value = inference_artifacts::INFERENCE_RUN_SPEC_TOML)]
    run: PathBuf,

    /// The pins the contract holds. As for `iris claim check`.
    #[arg(long)]
    pins: Option<PathBuf>,

    /// Write every call a contract can be asked of the claim with no replay,
    /// as a JSON list: a `voidClaim` for each checkpoint's program and for
    /// each input its program takes, each with what the contract is
    /// expected to answer. The calls that point at a fault are the ones to
    /// send.
    #[arg(long)]
    calls: Option<PathBuf>,
}

/// What a lying claim says falsely of its checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ClaimFaultArg {
    /// Another output than the stage's.
    Output,
    /// A program outside the approved set.
    Program,
    /// An input that is not what its producer made.
    Input,
    /// An input the claim may name, in a place the chain does not give it:
    /// the value an earlier stage of the same program is given.
    Arrangement,
    /// The check's record stated valid where the check does not find it so,
    /// with the result stage run on that record. Of a claim whose claimed
    /// output is not the model's.
    Verdict,
}

#[derive(Debug, Args)]
struct ClaimFetchArgs {
    /// The claim's `claim_submission.json` as it was submitted, with where
    /// its data is.
    #[arg(long)]
    submission: PathBuf,

    /// Your own run spec of the configuration the claim names: its
    /// pipeline, models and parameters, and for a claim with a source the
    /// recipe and the signer's key. Its prompt and its documents are
    /// replaced by the ones the claim posted.
    #[arg(long, default_value = inference_artifacts::INFERENCE_RUN_SPEC_TOML)]
    run: PathBuf,

    /// The pins the contract holds. As for `iris claim check`.
    #[arg(long)]
    pins: Option<PathBuf>,

    /// The stand-in data-availability layer's directory. As for `iris claim
    /// check`.
    #[arg(long = "da-dir")]
    da_dir: Option<PathBuf>,

    /// A new directory to write the claim's files and `inference.toml` to.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Debug, Args)]
struct ClaimPinsArgs {
    /// Run spec naming the pipeline and its models.
    #[arg(long, default_value = inference_artifacts::INFERENCE_RUN_SPEC_TOML)]
    run: PathBuf,

    /// Where to write the pins. They are made from the models' imports and
    /// the stage projects' locks as they are now, so run this from the iris
    /// root.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Debug, Args)]
struct ClaimCheckArgs {
    /// Run spec naming the model and the parameters the contract requires.
    #[arg(long, default_value = inference_artifacts::INFERENCE_RUN_SPEC_TOML)]
    run: PathBuf,

    /// The `claim_submission.json` that `claim build` wrote.
    #[arg(long)]
    submission: PathBuf,

    /// A Raster chain run directory of the same claim. With it, every
    /// checkpoint of the claim is held to the boundary that run committed:
    /// the same program, input manifest and output at every stage.
    #[arg(long = "raster-run")]
    raster_run: Option<PathBuf>,

    /// The pins the contract holds, as `iris claim pins` wrote them. Without
    /// it a `single` contract's are the model import's, and those of a
    /// pipeline over several models are made from the imports now.
    #[arg(long)]
    pins: Option<PathBuf>,

    /// The registry of configurations, as `iris registry add` writes it. The
    /// claim is judged by the configuration it names there. Without it, by
    /// the one the run spec and the pins describe.
    #[arg(long)]
    registry: Option<PathBuf>,

    /// The stand-in data-availability layer's directory. Default:
    /// `IRIS_DA_DIR`, or `~/.cache/iris/stub-celestia`.
    #[arg(long = "da-dir")]
    da_dir: Option<PathBuf>,

    /// When the claim is taken to be submitted, in Unix seconds. Default: now.
    #[arg(long = "submitted-at")]
    submitted_at: Option<u64>,
}

#[derive(Debug, Subcommand)]
enum DaCommand {
    /// Print a block's height, data root and time, as a relayer posts them
    /// to a contract's oracle.
    BlockRoot {
        #[arg(long)]
        height: u64,
        /// The stand-in layer's directory. Default: `IRIS_CELESTIA_RPC`'s
        /// node, or the default stand-in.
        #[arg(long = "da-dir")]
        da_dir: Option<PathBuf>,
    },
    /// As the trusted host, check each blob a submission posted and sign
    /// where it is, its block's root and what it holds, with the key in
    /// `IRIS_ATTESTER_KEY`. The signatures are written into the submission.
    Attest {
        /// A claim's or a challenge's submission.
        #[arg(long)]
        submission: PathBuf,
        #[arg(long = "da-dir")]
        da_dir: Option<PathBuf>,
    },
    /// What a submission put on the layer, measured there: every blob it
    /// names is read back and its size split among what it holds.
    Contents {
        /// A claim's or a challenge's submission, posted.
        #[arg(long)]
        submission: PathBuf,
        #[arg(long = "da-dir")]
        da_dir: Option<PathBuf>,
    },
}

#[derive(Debug, Subcommand)]
enum EvmCommand {
    /// The call that registers a run spec's configuration.
    Configuration {
        #[arg(long, default_value = inference_artifacts::INFERENCE_RUN_SPEC_TOML)]
        run: PathBuf,
        #[arg(long)]
        pins: Option<PathBuf>,
    },
    /// The call that submits a claim.
    Claim {
        /// The claim's `claim_submission.json`, posted and attested.
        #[arg(long)]
        submission: PathBuf,
    },
    /// The call that submits a challenge.
    Challenge {
        /// The `challenge_submission.json` that `challenge post` wrote,
        /// attested.
        #[arg(long)]
        challenge: PathBuf,
        /// The disputed claim's `claim_submission.json`, in its directory.
        #[arg(long)]
        submission: PathBuf,
    },
    /// The call that answers a challenge with a Raster fault proof of its
    /// trace. Without a challenge: the proof's journal and seal as a contract
    /// takes them, and what a contract reads from the journal.
    FaultProof {
        /// The `fault.receipt` that `fault prove` wrote.
        #[arg(long)]
        receipt: PathBuf,
        /// The answered challenge's `challenge_submission.json`.
        #[arg(long, requires = "submission")]
        challenge: Option<PathBuf>,
        /// The disputed claim's `claim_submission.json`, in its directory.
        #[arg(long, requires = "challenge")]
        submission: Option<PathBuf>,
    },
    /// The ids of Raster's proof programs that a contract pins.
    ProofPrograms,
}

/// What a `devstack submit` sends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SubmitWhat {
    Claim,
    Challenge,
    FaultProof,
    Fault,
}

/// What a `devstack replay` holds the contracts to iris on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ReplayWhat {
    Claim,
    Challenge,
    Faults,
}

#[derive(Debug, Subcommand)]
enum DevstackCommand {
    /// Write the deployment file `devstack/deploy.sh` leaves: where the
    /// contracts are, what they were deployed with, and their errors by
    /// selector.
    Deployment {
        /// The file to write.
        #[arg(long)]
        out: PathBuf,
        /// A field of the file, as `name=value`. A value that is a whole
        /// number is written as one.
        #[arg(long = "set", value_name = "NAME=VALUE")]
        fields: Vec<String>,
        /// What `forge inspect <contract> errors --json` printed for each
        /// contract, one after another.
        #[arg(long)]
        errors: PathBuf,
    },
    /// Send a claim, a challenge, a fault proof or a fault to the contracts,
    /// after attesting its blobs, relaying their block roots and
    /// registering the claim's configuration. Prints what the contract said
    /// and exits nonzero when it refused.
    Submit {
        what: SubmitWhat,
        /// The run spec of the claim's configuration, to register it.
        #[arg(long)]
        run: Option<PathBuf>,
        /// The claim's `claim_submission.json`.
        #[arg(long)]
        submission: Option<PathBuf>,
        /// The `challenge_submission.json` that `challenge post` wrote.
        #[arg(long)]
        challenge: Option<PathBuf>,
        /// The `fault.receipt` that `fault prove` wrote.
        #[arg(long)]
        receipt: Option<PathBuf>,
        /// The calls `iris claim audit --calls` wrote.
        #[arg(long)]
        calls: Option<PathBuf>,
        /// Which of those calls to send.
        #[arg(long)]
        call: Option<usize>,
        #[arg(long)]
        no_attest: bool,
        #[arg(long)]
        no_relay: bool,
        #[arg(long)]
        no_register: bool,
    },
    /// Hold the contracts to iris's own verdict, case by case: on a posted
    /// claim and changed copies of it, on a posted challenge and changed
    /// copies of it, or on every on-the-spot question of a fetched claim.
    /// The claim or challenge itself is then submitted. Exits nonzero when
    /// the two differ on any case.
    Replay {
        what: ReplayWhat,
        /// The run spec of the configuration.
        #[arg(long)]
        run: PathBuf,
        /// The claim's `claim_submission.json`.
        #[arg(long)]
        submission: PathBuf,
        /// The `challenge_submission.json` that `challenge post` wrote.
        #[arg(long)]
        challenge: Option<PathBuf>,
    },
}

#[derive(Debug, Subcommand)]
enum RegistryCommand {
    /// Register the configuration a run spec describes: its pipeline's
    /// programs, its models, its parameters and what it requires of the
    /// source and the result. Prints its id.
    Add(RegistryAddArgs),
}

#[derive(Debug, Args)]
struct RegistryAddArgs {
    /// Run spec naming the pipeline, its models and the parameters.
    #[arg(long, default_value = inference_artifacts::INFERENCE_RUN_SPEC_TOML)]
    run: PathBuf,

    /// The pins of the pipeline's programs and models, as `iris claim pins`
    /// wrote them. Without it, as `iris claim check` finds them.
    #[arg(long)]
    pins: Option<PathBuf>,

    /// The registry file to add to. Made when it does not exist.
    #[arg(long)]
    registry: PathBuf,
}

#[derive(Debug, Args)]
struct ChallengePostArgs {
    /// The `challenge_bundle.json` that `challenge build` wrote.
    #[arg(long)]
    challenge: PathBuf,

    /// The claim's `claim_submission.json`. The claim's checkpoints are read
    /// from the directory it is in.
    #[arg(long)]
    submission: PathBuf,

    /// Run spec naming the pipeline and models of the contract.
    #[arg(long, default_value = inference_artifacts::INFERENCE_RUN_SPEC_TOML)]
    run: PathBuf,

    /// The pins the contract holds. As for `iris claim check`.
    #[arg(long)]
    pins: Option<PathBuf>,

    /// The address the challenge is attributed to, as `claim build
    /// --claimer`. Default: `IRIS_CHALLENGER`, or the zero address.
    #[arg(long)]
    challenger: Option<String>,

    /// The stand-in data-availability layer's directory. As for `iris claim
    /// check`.
    #[arg(long = "da-dir")]
    da_dir: Option<PathBuf>,

    /// When the challenge is taken to be submitted, in Unix seconds. Default:
    /// now.
    #[arg(long = "submitted-at")]
    submitted_at: Option<u64>,

    /// Post this trace commitment in place of the bundle's own: the copy
    /// `iris challenge corrupt` wrote, for a challenge a defender can refute.
    #[arg(long)]
    commitment: Option<PathBuf>,

    /// Where to write the challenge submission. Default:
    /// `challenge_submission.json` beside the bundle.
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct ChallengeFetchArgs {
    /// The `challenge_submission.json` as it was submitted, with where its
    /// trace body is.
    #[arg(long)]
    challenge: PathBuf,

    /// The stand-in data-availability layer's directory. As for `iris claim
    /// check`.
    #[arg(long = "da-dir")]
    da_dir: Option<PathBuf>,

    /// The file to write the trace commitment to (a `commit.bin`).
    #[arg(long)]
    out: PathBuf,
}

#[derive(Debug, Args)]
struct ChallengeCheckArgs {
    /// The `challenge_submission.json` that `challenge post` wrote.
    #[arg(long)]
    challenge: PathBuf,

    /// The disputed claim's `claim_submission.json`, in its directory.
    #[arg(long)]
    submission: PathBuf,

    /// Run spec naming the pipeline and models of the contract.
    #[arg(long, default_value = inference_artifacts::INFERENCE_RUN_SPEC_TOML)]
    run: PathBuf,

    /// The pins the contract holds. As for `iris claim check`.
    #[arg(long)]
    pins: Option<PathBuf>,

    /// The stand-in data-availability layer's directory. As for `iris claim
    /// check`.
    #[arg(long = "da-dir")]
    da_dir: Option<PathBuf>,

    /// When the challenge is taken to be submitted, in Unix seconds. Default:
    /// now.
    #[arg(long = "submitted-at")]
    submitted_at: Option<u64>,
}

#[derive(Debug, Args)]
struct ClaimBuildArgs {
    /// Run spec to execute.
    #[arg(long, default_value = inference_artifacts::INFERENCE_RUN_SPEC_TOML)]
    run: PathBuf,

    /// Debug: claim these comma-separated output ids
    /// instead of discovering them, to build claims that should be refuted.
    #[arg(long = "claimed-ids", value_delimiter = ',', hide = true)]
    claimed_ids: Option<Vec<u32>>,

    /// The comma-separated output ids `iris infer` printed for this same run
    /// spec. The claim is built from them without running direct inference
    /// again; the chain still re-derives every token and rejects any other ids.
    #[arg(
        long = "direct-output-ids",
        value_delimiter = ',',
        conflicts_with = "claimed_ids"
    )]
    direct_output_ids: Option<Vec<u32>>,

    /// Use the previous per-stage child-process staged-infer runner.
    #[arg(long = "staged-subprocess", alias = "direct-subprocess", hide = true)]
    staged_subprocess: bool,

    /// The address the claim is attributed to: `0x` and 40 lowercase hex
    /// digits. Every blob the claim posts names it. Default: `IRIS_CLAIMER`,
    /// or the zero address.
    #[arg(long)]
    claimer: Option<String>,
}

#[derive(Debug, Args)]
struct ClaimCorruptHashesArgs {
    /// Checkpoints hash list to copy and corrupt.
    #[arg(long = "checkpoints", required_unless_present = "submission")]
    checkpoints: Option<PathBuf>,

    /// An honest claim's `claim_submission.json`, in its directory. The
    /// lying claim is written to `--output`, a new directory, ready for
    /// `iris claim post`.
    #[arg(long, conflicts_with_all = ["checkpoints", "trace"], requires = "output")]
    submission: Option<PathBuf>,

    /// With `--submission`: the address the lying claim is attributed to.
    /// Default: the honest claim's.
    #[arg(long, requires = "submission")]
    claimer: Option<String>,

    /// With `--submission`: what the lying claim says falsely of the
    /// checkpoint. An output is a lie a replay shows. A program or an input
    /// is one a challenger points at (`iris claim audit`).
    #[arg(long, value_enum, default_value = "output", requires = "submission")]
    fault: ClaimFaultArg,

    /// With `--fault input`: the input to lie about. Default: the first the
    /// checkpoint names a producer for.
    #[arg(long, requires = "submission")]
    input: Option<String>,

    /// Optional output path for the corrupted hash list. With
    /// `--submission`, the lying claim's directory.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Optional checkpoint trace used only to resolve --stage to an index.
    #[arg(long)]
    trace: Option<PathBuf>,

    /// Corrupt this zero-based checkpoint index.
    #[arg(long)]
    index: Option<usize>,

    /// Corrupt the checkpoint with this stage name.
    #[arg(long)]
    stage: Option<String>,

    /// Pick a checkpoint to corrupt. This is the default when no selector is supplied.
    #[arg(long)]
    random: bool,

    /// Seed for reproducible random selection.
    #[arg(long)]
    seed: Option<String>,
}

#[derive(Debug, Subcommand)]
enum ChallengeCommand {
    /// Locate the first divergent checkpoint in a claim.
    Locate,
    /// Build a challenge bundle for a divergent claim.
    Build(ChallengeBuildArgs),
    /// Flip an interior fingerprint bit in a challenge copy.
    Corrupt(fault::CorruptArgs),
    /// Measure guest cycles and select a heavy fault window from a completed challenge.
    Profile(challenge_profile::ProfileArgs),
    /// Post a challenge's trace body to the data-availability layer, write
    /// the challenge submission, and make the checks a contract would make
    /// on it.
    Post(ChallengePostArgs),
    /// Read the trace a submitted challenge names back from the
    /// data-availability layer: what a defender audits its own run against.
    Fetch(ChallengeFetchArgs),
    /// Make the checks a contract would make on a challenge's submission as
    /// it is. Nothing is posted.
    Check(ChallengeCheckArgs),
}

#[derive(Debug, Subcommand)]
enum FaultCommand {
    /// Check that this binary embeds real proving guests.
    Check {
        /// Verify program identities used by automatic backend challenges.
        #[arg(long)]
        raster_manifest: Option<PathBuf>,
    },
    /// Prove a bad Raster challenge.
    Prove(fault::ProveArgs),
}

pub fn run_from_env() -> Result<ExitCode> {
    execute(Cli::parse())
}

pub fn execute_from<I, T>(args: I) -> Result<ExitCode>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    execute(Cli::try_parse_from(args)?)
}

pub fn parse_for_test<I, T>(args: I) -> Result<()>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    Cli::try_parse_from(args)?;
    Ok(())
}

fn execute(cli: Cli) -> Result<ExitCode> {
    if let Some(bytes) = cli.aux_worker {
        staged_infer::worker::run_worker(bytes)?;
        return Ok(ExitCode::SUCCESS);
    }
    if let Some(stage_dir) = cli.run_stage.as_ref() {
        require_input_args(&cli)?;
        staged_infer::parity::run_hidden_stage_direct(stage_dir)?;
        return Ok(ExitCode::SUCCESS);
    }

    if let Some(stage_dir) = cli.compare_stage {
        if cli.input.is_none() || cli.input_manifest.is_none() {
            let status = run_hidden_compare(&stage_dir)?;
            return finish_parity_run(status, &stage_dir);
        }
        let report = staged_infer::parity::run_hidden_stage_compare(&stage_dir)?;
        return Ok(if report.matched {
            ExitCode::SUCCESS
        } else {
            ExitCode::from(1)
        });
    }

    match cli.command {
        #[cfg(feature = "replay-profile")]
        Some(Commands::ReplayProfile(args)) => {
            warn_if_debug_build("replay-profile");
            run_replay_profile(ReplayProfileOptions {
                base_dir: std::env::current_dir()?,
                run_spec_path: args.run,
                current_exe: std::env::current_exe()?,
                stage: args.stage,
                top_n: args.top_n.get(),
            })?;
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::TestParity {
            skip_challenges,
            bundle,
        }) => {
            parity::run(skip_challenges, &bundle)?;
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::CompareTeacherForced {
            teacher_forced,
            direct,
        }) => {
            let comparison = parity::compare_teacher_forced(&teacher_forced, &direct)?;
            println!("{}", serde_json::to_string_pretty(&comparison)?);
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::ParityWorker {
            run,
            output,
            direct,
        }) => {
            parity::worker(&run, &output, direct)?;
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Model {
            command: ModelCommand::Convert(args),
        }) => {
            convert::convert_bundle(&args.gguf, &args.metadata, &args.out, args.dequantize)?;
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Model {
            command: ModelCommand::Import(args),
        }) => {
            model_import::import_model(args.into_import_config())
                .map_err(|error| anyhow::anyhow!("{error}"))?;
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Claim {
            command: ClaimCommand::Build(args),
        }) => {
            warn_if_debug_build("claim build");
            let run_spec_path = args.run.clone();
            let result = build_claim(args.into_options()?)?;
            println!("pipeline: {}", result.pipeline);
            match (result.final_result.as_ref(), result.consensus.as_ref()) {
                (Some(first), Some(consensus)) => {
                    print_consensus(first, consensus);
                    for (inference, result) in [first, &consensus.second].into_iter().enumerate() {
                        println!("inference: {inference}");
                        print_infer_result(result, Some(&run_spec_path), inference);
                    }
                }
                (Some(final_result), None) => {
                    print_infer_result(final_result, Some(&run_spec_path), 0)
                }
                (None, _) => {}
            }
            println!(
                "checkpoint trace: {}",
                result.checkpoint_trace_path.display()
            );
            println!("checkpoints: {}", result.checkpoint_hashes_path.display());
            println!("claim bundle: {}", result.claim_bundle_path.display());
            if let Some(path) = result.claim_submission_path.as_ref() {
                println!("claim submission: {}", path.display());
                if path
                    .with_file_name(availability::POSTING_OUTPUTS_JSON)
                    .exists()
                {
                    println!("posting: prepared, not posted (`iris claim post`)");
                }
            }
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Claim {
            command: ClaimCommand::Post(args),
        }) => {
            let base_dir =
                std::env::current_dir().context("failed to resolve current directory")?;
            let submission = absolute(&base_dir, &args.submission);
            let da_dir = args.da_dir.map(|dir| absolute(&base_dir, &dir));
            let posted = contract::post_claim_submission(&submission, da_dir.as_deref(), None)?;
            for blob in &posted.blobs {
                println!(
                    "blob: height {} commitment {} ({} bytes)",
                    blob.pointer.height, blob.pointer.commitment, blob.length
                );
            }
            println!("claim submission: {}", submission.display());
            Ok(match posted.rejection {
                None => {
                    println!("ok        data posted, proven and retrievable through the challenge period");
                    ExitCode::SUCCESS
                }
                Some(reason) => {
                    println!("REJECTED  data posted: {reason}");
                    ExitCode::FAILURE
                }
            })
        }
        Some(Commands::Claim {
            command: ClaimCommand::Check(args),
        }) => {
            let report = check_claim(ClaimCheckOptions {
                base_dir: std::env::current_dir().context("failed to resolve current directory")?,
                run_spec_path: args.run,
                submission_path: args.submission,
                raster_run: args.raster_run,
                pins_path: args.pins,
                registry_path: args.registry,
                da_dir: args.da_dir,
                submitted_at: args.submitted_at,
            })?;
            for line in &report.lines {
                match &line.rejection {
                    None => println!("ok        {}", line.check),
                    Some(reason) => println!("REJECTED  {}: {reason}", line.check),
                }
            }
            Ok(if report.accepted() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            })
        }
        Some(Commands::Claim {
            command: ClaimCommand::Pins(args),
        }) => {
            let base_dir =
                std::env::current_dir().context("failed to resolve current directory")?;
            let pins = contract::run_pins(&base_dir, &args.run)?;
            inference_artifacts::write_json(&args.out, &pins)?;
            println!("pipeline: {}", pins.pipeline);
            for model in &pins.models {
                println!("model id: {}", model.model_id);
            }
            println!("stage programs: {}", pins.programs.stage_programs.len());
            println!("contract pins: {}", args.out.display());
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Claim {
            command: ClaimCommand::Fetch(args),
        }) => {
            let base_dir =
                std::env::current_dir().context("failed to resolve current directory")?;
            let fetched = posted::fetch_claim(posted::FetchClaimOptions {
                submission_path: absolute(&base_dir, &args.submission),
                run_spec_path: absolute(&base_dir, &args.run),
                pins_path: args.pins.map(|path| absolute(&base_dir, &path)),
                da_dir: args.da_dir.map(|path| absolute(&base_dir, &path)),
                out_dir: absolute(&base_dir, &args.out),
            })?;
            println!("blobs read: {} ({} bytes)", fetched.blobs, fetched.bytes);
            println!("checkpoints: {}", fetched.checkpoints);
            println!("input manifests: {}", fetched.input_manifests);
            println!("claim submission: {}", fetched.submission_path.display());
            println!("run spec: {}", fetched.run_spec_path.display());
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Claim {
            command: ClaimCommand::Audit(args),
        }) => {
            let base_dir =
                std::env::current_dir().context("failed to resolve current directory")?;
            let audit = posted::audit_claim(posted::AuditClaimOptions {
                submission_path: absolute(&base_dir, &args.submission),
                run_spec_path: absolute(&base_dir, &args.run),
                pins_path: args.pins.map(|path| absolute(&base_dir, &path)),
            })?;
            println!("checkpoints: {}", audit.checkpoints);
            println!("programs approved: {}", audit.programs_approved);
            println!("inputs held to their origin: {}", audit.inputs_linked);
            println!("faults: {}", audit.faults.len());
            for fault in &audit.faults {
                let input = fault
                    .input
                    .as_ref()
                    .map(|input| format!(", input `{input}`"))
                    .unwrap_or_default();
                println!(
                    "fault: checkpoint {} (`{}`){input}: {} (call {})",
                    fault.checkpoint, fault.stage, fault.fault, fault.call
                );
            }
            if let Some(calls) = args.calls {
                let calls = absolute(&base_dir, &calls);
                inference_artifacts::write_json(&calls, &audit.calls)?;
                println!("calls: {}", calls.display());
            }
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Claim {
            command: ClaimCommand::Corrupt(args),
        }) if args.submission.is_some() => {
            let base_dir =
                std::env::current_dir().context("failed to resolve current directory")?;
            let selection = args.selection()?;
            anyhow::ensure!(
                args.input.is_none()
                    || matches!(
                        args.fault,
                        ClaimFaultArg::Input | ClaimFaultArg::Arrangement
                    ),
                "--input names the input of a `--fault input` or `--fault arrangement` lie"
            );
            let lying = posted::corrupt_claim(posted::CorruptClaimOptions {
                submission_path: absolute(&base_dir, &args.submission.expect("checked above")),
                selection,
                fault: match args.fault {
                    ClaimFaultArg::Output => posted::CorruptFault::Output,
                    ClaimFaultArg::Program => posted::CorruptFault::Program,
                    ClaimFaultArg::Input => posted::CorruptFault::Input(args.input),
                    ClaimFaultArg::Arrangement => posted::CorruptFault::Arrangement(args.input),
                    ClaimFaultArg::Verdict => posted::CorruptFault::Verdict,
                },
                claimer: args.claimer,
                out_dir: absolute(&base_dir, &args.output.expect("clap requires it")),
            })?;
            println!("corrupted checkpoints: {}", lying.hashes_path.display());
            println!("corruption manifest: {}", lying.manifest_path.display());
            println!("corrupted checkpoint index: {}", lying.checkpoint_index);
            println!(
                "corrupted checkpoint number: {}",
                lying.checkpoint_index + 1
            );
            println!("corrupted stage: {}", lying.stage);
            println!("original hash: {}", lying.original_hash);
            println!("corrupted hash: {}", lying.corrupted_hash);
            println!("claim submission: {}", lying.submission_path.display());
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Claim {
            command: ClaimCommand::Corrupt(args),
        }) => {
            let result = corrupt_checkpoint_hashes(args.into_options()?)?;
            println!(
                "corrupted checkpoints: {}",
                result.corrupted_hashes_path.display()
            );
            println!(
                "corruption manifest: {}",
                result.corruption_manifest_path.display()
            );
            println!("corrupted checkpoint index: {}", result.checkpoint_index);
            println!("corrupted checkpoint number: {}", result.checkpoint_number);
            if let Some(stage) = result.stage.as_ref() {
                println!("corrupted stage: {stage}");
            }
            println!("original hash: {}", result.original_hash);
            println!("corrupted hash: {}", result.corrupted_hash);
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Infer(args)) => {
            warn_if_debug_build("infer");
            let run_spec_path = args.run.clone();
            let run = run_infer(args.run)?;
            println!("pipeline: {}", run.pipeline);
            let source = source::run_source_of(&run_spec_path)?;
            match (run.reports.as_slice(), run.consensus.as_ref()) {
                ([report], None) => {
                    if let Some(count) = report.input_token_count {
                        println!("input token count: {count}");
                    }
                    // Before the result: its last lines are the generated text.
                    if let Some(source) = &source {
                        source::print_source_lines(source);
                    }
                    print_infer_result(&report.result, Some(&run_spec_path), 0);
                    print_infer_timing_summary(&report.timings);
                }
                ([first, second], Some(consensus)) => {
                    if let Some(source) = &source {
                        source::print_source_lines(source);
                    }
                    print_consensus(&first.result, consensus);
                    for (inference, report) in [first, second].into_iter().enumerate() {
                        println!("inference: {inference}");
                        if let Some(count) = report.input_token_count {
                            println!("input token count: {count}");
                        }
                        print_infer_result(&report.result, Some(&run_spec_path), inference);
                        print_infer_timing_summary(&report.timings);
                    }
                }
                _ => anyhow::bail!("the `{}` pipeline gave no result to print", run.pipeline),
            }
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Attest { command }) => {
            match command {
                source::AttestCommand::Keygen { out } => source::keygen(&out)?,
                source::AttestCommand::Sign(args) => source::sign(args)?,
            }
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Source {
            command: source::SourceCommand::Inputs { run, out },
        }) => {
            source::stage_inputs(&run, &out)?;
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Source {
            command: source::SourceCommand::Render { run, text_out },
        }) => {
            let record = source::render(&run, text_out.as_deref())?;
            Ok(if record.fault == 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            })
        }
        Some(Commands::Challenge { command }) => match command {
            ChallengeCommand::Profile(args) => {
                challenge_profile::run(args)?;
                Ok(ExitCode::SUCCESS)
            }
            ChallengeCommand::Corrupt(args) => {
                fault::corrupt(args)?;
                Ok(ExitCode::SUCCESS)
            }
            ChallengeCommand::Locate => {
                bail!("iris challenge locate is not implemented yet")
            }
            ChallengeCommand::Build(args) => {
                warn_if_debug_build("challenge build");
                let result = build_challenge(args.into_options()?)?;
                print_challenge_result(&result);
                Ok(ExitCode::SUCCESS)
            }
            ChallengeCommand::Post(args) => {
                let base_dir =
                    std::env::current_dir().context("failed to resolve current directory")?;
                let posted = contract::post_challenge(contract::ChallengePostOptions {
                    run_spec_path: absolute(&base_dir, &args.run),
                    pins_path: args.pins.map(|path| absolute(&base_dir, &path)),
                    challenge_bundle_path: absolute(&base_dir, &args.challenge),
                    submission_path: absolute(&base_dir, &args.submission),
                    challenger: args
                        .challenger
                        .or_else(|| std::env::var("IRIS_CHALLENGER").ok()),
                    da_dir: args.da_dir.map(|path| absolute(&base_dir, &path)),
                    submitted_at: args.submitted_at,
                    commitment_path: args.commitment.map(|path| absolute(&base_dir, &path)),
                    out_path: args.out.map(|path| absolute(&base_dir, &path)),
                })?;
                for line in &posted.report.lines {
                    match &line.rejection {
                        None => println!("ok        {}", line.check),
                        Some(reason) => println!("REJECTED  {}: {reason}", line.check),
                    }
                }
                println!("challenge submission: {}", posted.submission_path.display());
                Ok(if posted.report.accepted() {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::FAILURE
                })
            }
            ChallengeCommand::Fetch(args) => {
                let base_dir =
                    std::env::current_dir().context("failed to resolve current directory")?;
                let fetched = posted::fetch_challenge(posted::FetchChallengeOptions {
                    challenge_path: absolute(&base_dir, &args.challenge),
                    da_dir: args.da_dir.map(|path| absolute(&base_dir, &path)),
                    out_path: absolute(&base_dir, &args.out),
                })?;
                println!("trace header: {}", fetched.header_digest);
                println!(
                    "trace commitment: {} ({} bytes)",
                    fetched.commitment_path.display(),
                    fetched.bytes
                );
                Ok(ExitCode::SUCCESS)
            }
            ChallengeCommand::Check(args) => {
                let base_dir =
                    std::env::current_dir().context("failed to resolve current directory")?;
                let report =
                    contract::check_challenge_submission(contract::ChallengeCheckOptions {
                        run_spec_path: absolute(&base_dir, &args.run),
                        pins_path: args.pins.map(|path| absolute(&base_dir, &path)),
                        challenge_path: absolute(&base_dir, &args.challenge),
                        submission_path: absolute(&base_dir, &args.submission),
                        da_dir: args.da_dir.map(|path| absolute(&base_dir, &path)),
                        submitted_at: args.submitted_at,
                    })?;
                for line in &report.lines {
                    match &line.rejection {
                        None => println!("ok        {}", line.check),
                        Some(reason) => println!("REJECTED  {}: {reason}", line.check),
                    }
                }
                Ok(if report.accepted() {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::FAILURE
                })
            }
        },
        Some(Commands::Da { command }) => {
            let base_dir =
                std::env::current_dir().context("failed to resolve current directory")?;
            match command {
                DaCommand::BlockRoot { height, da_dir } => {
                    let da_dir = da_dir.map(|dir| absolute(&base_dir, &dir));
                    let root = availability::open_layer(da_dir.as_deref())?.block_root(height)?;
                    println!("{}", serde_json::to_string_pretty(&root)?);
                }
                DaCommand::Attest { submission, da_dir } => {
                    let da_dir = da_dir.map(|dir| absolute(&base_dir, &dir));
                    let layer = availability::open_layer(da_dir.as_deref())?;
                    let path = absolute(&base_dir, &submission);
                    let attested = evm::attest_submission(&path, layer.as_ref())?;
                    println!("attested blobs: {attested}");
                    println!("submission: {}", path.display());
                }
                DaCommand::Contents { submission, da_dir } => {
                    let da_dir = da_dir.map(|dir| absolute(&base_dir, &dir));
                    let measured = posted::measure_posting(
                        &absolute(&base_dir, &submission),
                        da_dir.as_deref(),
                    )?;
                    println!("{}", serde_json::to_string_pretty(&measured)?);
                }
            }
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Devstack { command }) => {
            let base_dir =
                std::env::current_dir().context("failed to resolve current directory")?;
            let at = |path: &PathBuf| absolute(&base_dir, path);
            match command {
                DevstackCommand::Deployment {
                    out,
                    fields,
                    errors,
                } => {
                    let mut written = serde_json::Map::new();
                    for field in &fields {
                        let (name, value) = field
                            .split_once('=')
                            .with_context(|| format!("`{field}` is not name=value"))?;
                        let value = match value.parse::<u64>() {
                            Ok(number) => serde_json::json!(number),
                            Err(_) => serde_json::json!(value),
                        };
                        written.insert(name.to_string(), value);
                    }
                    let errors_text = fs::read_to_string(at(&errors))
                        .with_context(|| format!("failed to read {}", errors.display()))?;
                    let count = devstack::write_deployment(
                        &at(&out),
                        serde_json::Value::Object(written.clone()),
                        &errors_text,
                    )?;
                    println!("{}", serde_json::to_string(&written)?);
                    println!("{count} errors");
                    Ok(ExitCode::SUCCESS)
                }
                DevstackCommand::Submit {
                    what,
                    run,
                    submission,
                    challenge,
                    receipt,
                    calls,
                    call,
                    no_attest,
                    no_relay,
                    no_register,
                } => {
                    let node = devstack::Node::open()?;
                    let layer = availability::open_layer(None)?;
                    let before = devstack::submit::Before {
                        attest: !no_attest,
                        relay: !no_relay,
                        register: !no_register,
                    };
                    let need = |path: &Option<PathBuf>, flag: &str| {
                        path.as_ref()
                            .map(&at)
                            .with_context(|| format!("this submission needs {flag}"))
                    };
                    let outcome = match what {
                        SubmitWhat::Claim => devstack::submit::claim(
                            &node,
                            layer.as_ref(),
                            run.as_ref().map(&at).as_deref(),
                            &need(&submission, "--submission")?,
                            before,
                        )?,
                        SubmitWhat::Challenge => devstack::submit::challenge(
                            &node,
                            layer.as_ref(),
                            &need(&submission, "--submission")?,
                            &need(&challenge, "--challenge")?,
                            before,
                        )?,
                        SubmitWhat::FaultProof => devstack::submit::fault_proof(
                            &node,
                            &need(&submission, "--submission")?,
                            &need(&challenge, "--challenge")?,
                            &need(&receipt, "--receipt")?,
                        )?,
                        SubmitWhat::Fault => devstack::submit::fault(
                            &node,
                            &need(&calls, "--calls")?,
                            call.context("a fault needs --call")?,
                        )?,
                    };
                    println!("{}", serde_json::to_string_pretty(&outcome)?);
                    Ok(if outcome["accepted"] == true {
                        ExitCode::SUCCESS
                    } else {
                        ExitCode::FAILURE
                    })
                }
                DevstackCommand::Replay {
                    what,
                    run,
                    submission,
                    challenge,
                } => {
                    let node = devstack::Node::open()?;
                    let layer = availability::open_layer(None)?;
                    let (run, submission) = (at(&run), at(&submission));
                    match what {
                        ReplayWhat::Claim => {
                            devstack::replay::claim(&node, layer.as_ref(), &run, &submission)?
                        }
                        ReplayWhat::Challenge => devstack::replay::challenge(
                            &node,
                            layer.as_ref(),
                            &run,
                            &submission,
                            &at(&challenge.context("a challenge needs --challenge")?),
                        )?,
                        ReplayWhat::Faults => {
                            devstack::replay::faults(&node, layer.as_ref(), &run, &submission)?
                        }
                    }
                    Ok(ExitCode::SUCCESS)
                }
            }
        }
        Some(Commands::Evm { command }) => {
            let base_dir =
                std::env::current_dir().context("failed to resolve current directory")?;
            let call = match command {
                EvmCommand::Configuration { run, pins } => {
                    evm::configuration_call(&contract::spec_configuration(
                        &absolute(&base_dir, &run),
                        pins.map(|path| absolute(&base_dir, &path)).as_deref(),
                    )?)?
                }
                EvmCommand::Claim { submission } => {
                    evm::claim_call(&absolute(&base_dir, &submission))?
                }
                EvmCommand::Challenge {
                    challenge,
                    submission,
                } => evm::challenge_call(
                    &absolute(&base_dir, &challenge),
                    &absolute(&base_dir, &submission),
                )?,
                EvmCommand::FaultProof {
                    receipt,
                    challenge: Some(challenge),
                    submission: Some(submission),
                } => evm::fault_proof_call(
                    &absolute(&base_dir, &challenge),
                    &absolute(&base_dir, &submission),
                    &absolute(&base_dir, &receipt),
                )?,
                EvmCommand::FaultProof { receipt, .. } => {
                    evm::fault_proof_parts(&absolute(&base_dir, &receipt))?
                }
                EvmCommand::ProofPrograms => evm::proof_programs(),
            };
            println!("{}", serde_json::to_string_pretty(&call)?);
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Registry {
            command: RegistryCommand::Add(args),
        }) => {
            let base_dir =
                std::env::current_dir().context("failed to resolve current directory")?;
            let configuration = contract::spec_configuration(
                &absolute(&base_dir, &args.run),
                args.pins.map(|path| absolute(&base_dir, &path)).as_deref(),
            )?;
            let registry_path = absolute(&base_dir, &args.registry);
            let mut registry = inference_artifacts::registry::Registry::read(&registry_path)?;
            let id = registry.register(configuration);
            registry.write(&registry_path)?;
            println!("configuration: {id}");
            println!("registry: {}", registry_path.display());
            Ok(ExitCode::SUCCESS)
        }
        Some(Commands::Fault { command }) => match command {
            FaultCommand::Check { raster_manifest } => {
                fault::check_runtime()?;
                if let Some(manifest) = raster_manifest {
                    fault::check_challenge_programs(&manifest)?;
                }
                println!("Fault proving runtime available");
                println!(
                    "Proving build: {}",
                    if cfg!(feature = "cuda") {
                        "CUDA"
                    } else {
                        "CPU"
                    }
                );
                Ok(ExitCode::SUCCESS)
            }
            FaultCommand::Prove(args) => {
                fault::prove(args)?;
                Ok(ExitCode::SUCCESS)
            }
        },
        None => {
            bail!("iris requires a workflow command; try `model import`, `claim build`, or `infer`")
        }
    }
}

#[cfg(debug_assertions)]
fn warn_if_debug_build(workflow: &str) {
    eprintln!(
        "warning: iris {workflow} is running from a debug build; \
         use `cargo run --release -p iris-cli -- {workflow}` \
         for the optimized staged-infer path"
    );
}

#[cfg(not(debug_assertions))]
fn warn_if_debug_build(_workflow: &str) {}

#[derive(Debug, Args)]
struct ChallengeBuildArgs {
    /// Run spec selecting the model to verify against the claim.
    #[arg(long, default_value = inference_artifacts::INFERENCE_RUN_SPEC_TOML)]
    run: PathBuf,

    /// Claim bundle to challenge. Preferred over --trace because it carries frozen run metadata.
    #[arg(long)]
    claim: Option<PathBuf>,

    /// Existing checkpoint trace to challenge.
    #[arg(long)]
    trace: Option<PathBuf>,

    /// Public claimed checkpoints hash list to challenge.
    #[arg(long = "checkpoints")]
    checkpoint_hashes: Option<PathBuf>,

    /// Claim bundle used only as frozen run context for --checkpoints.
    #[arg(long = "claim-context")]
    claim_context: Option<PathBuf>,

    /// A submitted claim as `iris claim fetch` wrote it: its
    /// `claim_submission.json`, with its checkpoint list beside it. Give
    /// `--run` the spec that command wrote there. Nothing of the claimer's
    /// own run is read.
    #[arg(long, conflicts_with_all = ["claim", "trace", "checkpoint_hashes", "claim_context"])]
    submission: Option<PathBuf>,

    /// With `--submission`: the comma-separated output ids your own `iris
    /// infer` printed for the fetched run spec, taken in place of running
    /// direct inference again.
    #[arg(
        long = "direct-output-ids",
        value_delimiter = ',',
        requires = "submission"
    )]
    direct_output_ids: Option<Vec<u32>>,

    /// Use the previous per-stage child-process staged-infer runner.
    #[arg(long = "staged-subprocess", alias = "direct-subprocess", hide = true)]
    staged_subprocess: bool,
}

fn require_input_args(cli: &Cli) -> Result<()> {
    if cli.input.is_none() || cli.input_manifest.is_none() {
        bail!("hidden stage mode requires --input and --input-manifest");
    }
    Ok(())
}

fn print_challenge_result(result: &ChallengeBuildResult) {
    match &result.outcome {
        ChallengeBuildOutcome::NoDivergence {
            claimed_reference_path,
            recomputed_trace_path,
            recomputed_chain_dir,
        } => {
            println!("no divergence found");
            println!("claimed reference: {}", claimed_reference_path.display());
            println!("recomputed trace: {}", recomputed_trace_path.display());
            println!("recomputed chain: {}", recomputed_chain_dir.display());
        }
        ChallengeBuildOutcome::ClaimedOutputRejected {
            reason,
            claimed_reference_path,
            recomputed_trace_path,
            recomputed_chain_dir,
        } => {
            println!("no divergence found; claimed output rejected");
            println!("reason: {reason}");
            println!("claimed reference: {}", claimed_reference_path.display());
            println!("recomputed trace: {}", recomputed_trace_path.display());
            println!("recomputed chain: {}", recomputed_chain_dir.display());
        }
        ChallengeBuildOutcome::DivergenceBuilt {
            divergence,
            replay_package,
            challenge_bundle_path,
            ..
        } => {
            println!("divergence found: {}", divergence.stage);
            println!("checkpoint index: {}", divergence.checkpoint_index);
            println!("checkpoint number: {}", divergence.checkpoint_index + 1);
            println!("reason: {:?}", divergence.reason);
            if let Some(claimed_hash) = divergence.claimed_hash.as_ref() {
                println!("claimed checkpoint hash: {claimed_hash}");
            }
            if let Some(recomputed_hash) = divergence.recomputed_hash.as_ref() {
                println!("recomputed checkpoint hash: {recomputed_hash}");
            }
            if let Some(claimed) = divergence.claimed.as_ref() {
                println!("claimed commitment: {}", claimed.output_commitment);
                println!("claimed sha256: {}", claimed.output_sha256);
            }
            if let Some(recomputed) = divergence.recomputed.as_ref() {
                println!("recomputed commitment: {}", recomputed.output_commitment);
                println!("recomputed sha256: {}", recomputed.output_sha256);
            }
            println!(
                "raster fingerprint: {}",
                replay_package.commit_path.display()
            );
            println!("challenge bundle: {}", challenge_bundle_path.display());
        }
    }
}

/// What a consensus states of its two inferences. Printed before either
/// inference's result, whose last lines are its generated text.
fn print_consensus(
    first: &staged_infer::InferenceResult,
    consensus: &inference_artifacts::ConsensusOutcome,
) {
    println!("consensus: {}", consensus.description(first));
    println!("consensus code: {}", consensus.code);
    println!("same input: {}", consensus.same_input);
}

/// One inference's result. `inference` is which of the run's models made it,
/// whose tokenizer the ids decode by.
fn print_infer_result(
    result: &staged_infer::InferenceResult,
    run_spec_path: Option<&Path>,
    inference: usize,
) {
    println!("generated token count: {}", result.generated_token_count);
    println!("generated token ids: {:?}", result.generated_token_ids);
    println!(
        "generated token ids sha256: {}",
        result.generated_token_ids_sha256
    );
    if let Some(run_spec_path) = run_spec_path {
        match render_token_trace_for_run(result, run_spec_path, inference) {
            Ok(trace) => print!("{trace}"),
            Err(error) => eprintln!("generated token trace unavailable: {error:#}"),
        }
    }
    println!("stop reason: {}", result.stop_reason);
    println!("output text sha256: {}", result.output_text_sha256);
    println!("input text sha256: {}", result.input_text_sha256);
    println!("source sha256: {}", result.source_sha256);
    println!("model id: {}", result.model_id);
    println!(
        "faults: prompt {}, check {}, output {}, model {}; claim mismatches: {}",
        result.prompt_fault,
        result.check_fault,
        result.output_fault,
        result.model_fault,
        result.claim_mismatches
    );
    // The claim record a contract reads, beside the fields it is stated from.
    println!("statement sha256: {}", result.statement_sha256);
    println!("valid: {}", result.valid);
    // What the result stage made of the output.
    println!("result code: {}", result.result_code);
    println!("result forms sha256: {}", result.result_forms_sha256);
    println!("generated text:");
    println!("{}", result.generated_text);
}

fn render_token_trace_for_run(
    result: &staged_infer::InferenceResult,
    run_spec_path: &Path,
    inference: usize,
) -> Result<String> {
    let base_dir = std::env::current_dir().context("failed to read current directory")?;
    let run_spec_path = absolute(&base_dir, run_spec_path);
    let run_spec = inference_artifacts::read_run_spec(&run_spec_path)?;
    let run_spec_dir = run_spec_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("run spec has no parent directory"))?;
    // The ids are of this one of the run's models, and decode by its tokenizer.
    let model_manifest_path = absolute(run_spec_dir, run_spec.model_manifest_of(inference)?);
    let model_manifest: inference_artifacts::ModelManifest =
        inference_artifacts::read_json(&model_manifest_path)?;
    let model_manifest_dir = model_manifest_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("model manifest has no parent directory"))?;
    let tokenizer_path = absolute(model_manifest_dir, &model_manifest.bundle.tokenizer_path);
    let tokenizer: serde_json::Value = inference_artifacts::read_json(&tokenizer_path)?;
    let decoder = TokenDecoder::from_tokenizer(&tokenizer, &model_manifest.eos_token_ids)?;
    Ok(render_token_trace(result, &decoder))
}

fn render_token_trace(result: &staged_infer::InferenceResult, decoder: &TokenDecoder) -> String {
    let mut out = String::from("generated tokens:\n");
    let mut stopped = false;
    for (idx, token_id) in result.generated_token_ids.iter().copied().enumerate() {
        let display = decoder.display(token_id, stopped);
        if display.kind == TokenKind::Terminal {
            stopped = true;
        }
        out.push_str(&format!(
            "  {idx}: id={token_id} token={} kind={} rendered={}\n",
            display.token,
            display.kind.as_str(),
            display.rendered
        ));
    }
    out
}

#[derive(Debug)]
struct TokenDecoder {
    tokens: BTreeMap<u32, String>,
    special_ids: BTreeSet<u32>,
    terminal_ids: BTreeSet<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenKind {
    Text,
    Special,
    Terminal,
    Byte,
    Unknown,
    AfterStop,
}

struct TokenDisplay {
    token: String,
    kind: TokenKind,
    rendered: String,
}

impl TokenKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Special => "special",
            Self::Terminal => "terminal",
            Self::Byte => "byte",
            Self::Unknown => "unknown",
            Self::AfterStop => "after_stop",
        }
    }
}

impl TokenDecoder {
    fn from_tokenizer(tokenizer: &serde_json::Value, eos_ids: &[u32]) -> Result<Self> {
        let mut tokens = tokenizer
            .get("model")
            .and_then(|model| model.get("vocab"))
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| anyhow::anyhow!("tokenizer.json has no vocab"))?
            .iter()
            .map(|(token, id)| {
                Ok((
                    id.as_u64()
                        .ok_or_else(|| anyhow::anyhow!("token '{token}' has non-integer id"))?
                        as u32,
                    token.clone(),
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let mut special_ids = BTreeSet::new();
        if let Some(added_tokens) = tokenizer
            .get("added_tokens")
            .and_then(serde_json::Value::as_array)
        {
            for entry in added_tokens {
                let Some(id) = entry.get("id").and_then(serde_json::Value::as_u64) else {
                    continue;
                };
                if let Some(content) = entry.get("content").and_then(serde_json::Value::as_str) {
                    tokens
                        .entry(id as u32)
                        .or_insert_with(|| content.to_string());
                }
                if entry
                    .get("special")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
                {
                    special_ids.insert(id as u32);
                }
            }
        }
        Ok(Self {
            tokens,
            special_ids,
            terminal_ids: eos_ids.iter().copied().collect(),
        })
    }

    fn display(&self, token_id: u32, stopped: bool) -> TokenDisplay {
        let Some(token) = self.tokens.get(&token_id) else {
            return TokenDisplay {
                token: String::from("<unknown>"),
                kind: TokenKind::Unknown,
                rendered: String::from("<unknown>"),
            };
        };
        if stopped {
            return TokenDisplay {
                token: quote_token(token),
                kind: TokenKind::AfterStop,
                rendered: String::from("<skipped>"),
            };
        }
        if self.terminal_ids.contains(&token_id) {
            return TokenDisplay {
                token: quote_token(token),
                kind: TokenKind::Terminal,
                rendered: String::from("<stops text>"),
            };
        }
        if self.special_ids.contains(&token_id) {
            return TokenDisplay {
                token: quote_token(token),
                kind: TokenKind::Special,
                rendered: String::from("<skipped>"),
            };
        }
        if let Some(byte) = byte_fallback(token) {
            return TokenDisplay {
                token: quote_token(token),
                kind: TokenKind::Byte,
                rendered: format!("<byte 0x{byte:02X}>"),
            };
        }
        TokenDisplay {
            token: quote_token(token),
            kind: TokenKind::Text,
            rendered: quote_token(&token.replace('\u{2581}', " ")),
        }
    }
}

fn byte_fallback(token: &str) -> Option<u8> {
    let bytes = token.as_bytes();
    if bytes.len() != 6 || &bytes[..3] != b"<0x" || bytes[5] != b'>' {
        return None;
    }
    Some(hex_nibble(bytes[3])? * 16 + hex_nibble(bytes[4])?)
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn quote_token(token: &str) -> String {
    format!("{token:?}")
}

fn absolute(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn print_infer_timing_summary(timings: &staged_infer::InferenceTimings) {
    eprint!("{}", render_infer_timing_summary(timings));
}

fn render_infer_timing_summary(timings: &staged_infer::InferenceTimings) -> String {
    if is_direct_infer_timing(timings) {
        return render_direct_infer_timing_summary(timings);
    }

    let mut out = String::new();
    out.push_str("infer timing:\n");
    out.push_str(&format!(
        "  total: {}\n",
        format_duration(timings.total_duration)
    ));
    if !timings.aux_waves.is_empty() {
        for wave in &timings.aux_waves {
            out.push_str(&format!(
                "  aux wave {}: {} stages, parallelism {}, wall {}, stage-sum {}",
                wave.name,
                wave.stage_count,
                wave.parallelism,
                format_duration(wave.wall_duration),
                format_duration(wave.stage_duration_sum)
            ));
            out.push('\n');
        }
    }

    let synthesis = timings
        .stages
        .iter()
        .map(|stage| stage.input_synthesis_duration)
        .sum();
    let input_load = timings
        .stages
        .iter()
        .map(|stage| stage.input_load_duration)
        .sum();
    let kernel = timings
        .stages
        .iter()
        .map(|stage| stage.kernel_duration)
        .sum();
    let encode = timings
        .stages
        .iter()
        .map(|stage| stage.encode_duration)
        .sum();
    out.push_str(&format!("  stages: {}\n", timings.stages.len()));
    out.push_str(&format!(
        "  input synthesis: {}\n",
        format_duration(synthesis)
    ));
    out.push_str(&format!("  input load: {}\n", format_duration(input_load)));
    out.push_str(&format!("  kernels: {}\n", format_duration(kernel)));
    out.push_str(&format!("  encode: {}\n", format_duration(encode)));
    out
}

fn is_direct_infer_timing(timings: &staged_infer::InferenceTimings) -> bool {
    !timings.stages.is_empty()
        && timings
            .stages
            .iter()
            .all(|stage| stage.routine == "direct-infer")
}

fn render_direct_infer_timing_summary(timings: &staged_infer::InferenceTimings) -> String {
    let named = |name: &str| {
        timings
            .stages
            .iter()
            .find(|stage| stage.stage == name)
            .map(|stage| stage.total_duration)
            .unwrap_or_default()
    };
    let accounted = named("load_direct_model")
        + named("prompt_prepare")
        + named("input_embedding")
        + named("prefill")
        + named("decode")
        + named("output_finalize");
    let kernel_records = timings
        .stages
        .iter()
        .filter(|stage| !matches!(stage.stage.as_str(), "prefill" | "decode"))
        .count();

    let mut out = String::new();
    out.push_str("direct-infer timing:\n");
    out.push_str(&format!(
        "  total: {}\n",
        format_duration(timings.total_duration)
    ));
    out.push_str(&format!("  phase records: {kernel_records}\n"));
    out.push_str(&format!(
        "  model load: {}\n",
        format_duration(named("load_direct_model"))
    ));
    out.push_str(&format!(
        "  prompt prepare: {}\n",
        format_duration(named("prompt_prepare"))
    ));
    out.push_str(&format!(
        "  input embedding: {}\n",
        format_duration(named("input_embedding"))
    ));
    out.push_str(&format!(
        "  prefill: {}\n",
        format_duration(named("prefill"))
    ));
    out.push_str(&format!("  decode: {}\n", format_duration(named("decode"))));
    out.push_str(&format!(
        "  output finalize: {}\n",
        format_duration(named("output_finalize"))
    ));
    out.push_str(&format!(
        "  accounted phase time: {}\n",
        format_duration(accounted)
    ));
    out
}

fn format_duration(duration: std::time::Duration) -> String {
    if duration.as_secs() >= 1 {
        format!("{:.2}s", duration.as_secs_f64())
    } else if duration.as_millis() >= 1 {
        format!("{}ms", duration.as_millis())
    } else if duration.as_micros() >= 1 {
        format!("{}us", duration.as_micros())
    } else {
        format!("{}ns", duration.as_nanos())
    }
}

fn finish_parity_run(status: ExitStatus, stage_dir: &Path) -> Result<ExitCode> {
    match staged_infer::parity::read_parity_report(stage_dir) {
        Ok(report) => {
            let chain_dir = stage_dir
                .parent()
                .ok_or_else(|| anyhow::anyhow!("selected stage has no chain run directory"))?;
            print!("{}", render_timing_summary_or_warning(chain_dir, &report));
        }
        Err(error) if status.success() => return Err(error),
        Err(_) => {}
    }
    Ok(exit_code_from_status(status))
}

fn render_timing_summary_or_warning(
    chain_dir: &Path,
    report: &staged_infer::parity::ParityReport,
) -> String {
    staged_infer::parity::read_chain_execution_times(chain_dir)
        .and_then(|timings| staged_infer::parity::render_timing_summary(&timings, report))
        .unwrap_or_else(|error| format!("\ntiming summary unavailable: {error:#}\n"))
}

fn run_hidden_compare(stage_dir: &Path) -> Result<ExitStatus> {
    let input = stage_dir.join("input.json");
    let input_manifest = stage_dir.join("input_manifest.json");
    let prior_report = staged_infer::parity::parity_dir(stage_dir).join("report.json");
    if let Err(error) = fs::remove_file(&prior_report) {
        if error.kind() != std::io::ErrorKind::NotFound {
            return Err(error)
                .with_context(|| format!("failed to remove {}", prior_report.display()));
        }
    }
    let exe = std::env::current_exe().context("failed to locate current executable")?;
    let status = comparison_command(&exe, stage_dir, &input, &input_manifest)
        .status()
        .context("failed to start staged-infer comparison child")?;
    Ok(status)
}

fn comparison_command(
    exe: &Path,
    stage_dir: &Path,
    input: &Path,
    input_manifest: &Path,
) -> ProcessCommand {
    let mut command = ProcessCommand::new(exe);
    command
        .arg("--compare-stage")
        .arg(stage_dir)
        .arg("--input")
        .arg(input)
        .arg("--input-manifest")
        .arg(input_manifest)
        .env(raster_runtime::auth::AUTH_ENV, "0")
        .env_remove(raster_runtime::TRACE_PATH_ENV)
        .env_remove(raster_runtime::tracing::COMMIT_PATH_ENV)
        .env_remove(raster_runtime::tracing::COMMIT_PROJECT_ENV)
        .env_remove(raster_runtime::tracing::COMMIT_WINDOW_ENV)
        .env_remove(raster_runtime::TRACE_FORMAT_ENV)
        .env_remove(raster_runtime::OUTPUT_DIR_ENV)
        .env_remove(raster_runtime::PROFILE_PATH_ENV)
        .env_remove(raster_runtime::PROFILE_STREAM_PATH_ENV)
        .env_remove(raster_runtime::PROFILE_RUN_ID_ENV);
    command
}

fn exit_code_from_status(status: ExitStatus) -> ExitCode {
    match status.code() {
        Some(code) => ExitCode::from(code.clamp(0, u8::MAX as i32) as u8),
        None => ExitCode::from(1),
    }
}

impl ModelImportArgs {
    fn into_import_config(self) -> model_import::ImportConfig {
        model_import::ImportConfig {
            model_dir: self.model_dir,
            manifest: self.manifest,
            artifact_root: self.artifact_root,
            model_id: self.model_id,
            only_tokenizer: self.only_tokenizer,
            only_layers: self.only_layers,
            only_embedding: self.only_embedding,
            only_ple: self.only_ple,
            only_direct: self.only_direct,
            no_stage_fixtures: self.no_stage_fixtures,
            range_size: self.range_size,
        }
    }
}

impl ClaimBuildArgs {
    fn into_options(self) -> Result<ClaimBuildOptions> {
        let current_exe = std::env::current_exe().context("failed to locate current executable")?;
        let staged_backend = if self.staged_subprocess {
            staged_infer::chain_runner::StagedExecutionBackend::Subprocess
        } else {
            staged_infer::chain_runner::StagedExecutionBackend::InProcess
        };
        let mut options =
            ClaimBuildOptions::from_current_dir(current_exe, staged_backend, self.run)?;
        options.claimed_token_ids = self.claimed_ids;
        options.direct_output_ids = self.direct_output_ids;
        options.claimer = self.claimer.or_else(|| std::env::var("IRIS_CLAIMER").ok());
        Ok(options)
    }
}

impl ClaimCorruptHashesArgs {
    fn selection(&self) -> Result<CorruptCheckpointSelection> {
        let selector_count = usize::from(self.index.is_some())
            + usize::from(self.stage.is_some())
            + usize::from(self.random);
        if selector_count > 1 {
            anyhow::bail!("claim corrupt accepts only one of --index, --stage, or --random");
        }
        Ok(if let Some(index) = self.index {
            CorruptCheckpointSelection::Index(index)
        } else if let Some(stage) = self.stage.clone() {
            CorruptCheckpointSelection::Stage(stage)
        } else {
            CorruptCheckpointSelection::Random {
                seed: self.seed.clone(),
            }
        })
    }

    fn into_options(self) -> Result<CorruptCheckpointHashesOptions> {
        let selection = self.selection()?;
        Ok(CorruptCheckpointHashesOptions {
            hashes_path: self
                .checkpoints
                .context("claim corrupt requires --checkpoints or --submission")?,
            trace_path: self.trace,
            output_path: self.output,
            selection,
        })
    }
}

impl ChallengeBuildArgs {
    fn into_options(self) -> Result<ChallengeBuildOptions> {
        let posted = self
            .submission
            .map(|submission_path| -> Result<ChallengeInput> {
                Ok(ChallengeInput::Posted {
                    submission_path: absolute(
                        &std::env::current_dir().context("failed to resolve current directory")?,
                        &submission_path,
                    ),
                })
            });
        let input = match (
            self.claim,
            self.trace,
            self.checkpoint_hashes,
            self.claim_context,
        ) {
            (None, None, None, None) if posted.is_some() => posted.expect("checked")?,
            (Some(claim), None, None, None) => ChallengeInput::Claim(claim),
            (None, Some(trace), None, None) => ChallengeInput::Trace(trace),
            (None, None, Some(checkpoint_hashes_path), Some(claim_context_path)) => {
                ChallengeInput::CheckpointHashes {
                    claim_context_path,
                    checkpoint_hashes_path,
                }
            }
            (None, None, Some(_), None) => {
                anyhow::bail!("challenge build --checkpoints requires --claim-context")
            }
            (None, None, None, Some(_)) => {
                anyhow::bail!("challenge build --claim-context requires --checkpoints")
            }
            (None, None, None, None) => anyhow::bail!(
                "challenge build requires --claim <claim_bundle.json> or --checkpoints <checkpoints.txt> --claim-context <claim_bundle.json>"
            ),
            _ => anyhow::bail!(
                "challenge build accepts only one input mode: --claim, --trace, or --checkpoints with --claim-context"
            ),
        };
        let current_exe = std::env::current_exe().context("failed to locate current executable")?;
        let staged_backend = if self.staged_subprocess {
            staged_infer::chain_runner::StagedExecutionBackend::Subprocess
        } else {
            staged_infer::chain_runner::StagedExecutionBackend::InProcess
        };
        let mut options =
            ChallengeBuildOptions::from_current_dir(current_exe, staged_backend, input, self.run)?;
        options.direct_output_ids = self.direct_output_ids;
        Ok(options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_phase_one_workflows() {
        parse_for_test(["iris", "model", "import", "--model", "model-bundles/model"]).unwrap();
        parse_for_test(["iris", "claim", "build", "--run", "inference.toml"]).unwrap();
        parse_for_test([
            "iris",
            "claim",
            "corrupt",
            "--checkpoints",
            "checkpoints.txt",
            "--index",
            "1",
        ])
        .unwrap();
        parse_for_test([
            "iris",
            "challenge",
            "build",
            "--trace",
            "checkpoint_trace.json",
        ])
        .unwrap();
        parse_for_test([
            "iris",
            "challenge",
            "build",
            "--checkpoints",
            "checkpoints.txt",
            "--claim-context",
            "claim_bundle.json",
        ])
        .unwrap();
        parse_for_test(["iris", "infer", "--run", "inference.toml"]).unwrap();
        parse_for_test([
            "iris",
            "model",
            "import",
            "--model",
            "model-bundles/model",
            "--only-direct",
        ])
        .unwrap();
    }

    #[test]
    fn challenge_run_spec_defaults_and_overrides_reach_options() {
        for run in [None, Some("specs/model-b.toml")] {
            let mut argv = vec!["iris", "challenge", "build", "--claim", "claim_bundle.json"];
            if let Some(path) = run {
                argv.extend(["--run", path]);
            }
            let cli = Cli::try_parse_from(argv).unwrap();
            let Some(Commands::Challenge {
                command: ChallengeCommand::Build(args),
            }) = cli.command
            else {
                panic!("expected challenge build");
            };
            let options = args.into_options().unwrap();
            assert_eq!(
                options.run_spec_path,
                PathBuf::from(run.unwrap_or("inference.toml"))
            );
        }
    }

    #[test]
    fn infer_requires_an_imported_workspace() {
        let error = execute_from(["iris", "infer"]).unwrap_err();

        assert!(format!("{error:#}").contains("failed to load run spec"));
    }

    #[test]
    fn direct_infer_timing_summary_uses_phase_language() {
        let timings = staged_infer::InferenceTimings {
            total_duration: std::time::Duration::from_millis(10),
            stages: vec![
                direct_timing("load_direct_model", 1),
                direct_timing("prompt_prepare", 1),
                direct_timing("input_embedding", 1),
                direct_timing("prefill_layer", 3),
                direct_timing("prefill", 3),
                direct_timing("decode_layer", 2),
                direct_timing("decode", 2),
                direct_timing("output_finalize", 1),
            ],
            aux_waves: Vec::new(),
        };

        let summary = render_infer_timing_summary(&timings);

        assert!(summary.contains("direct-infer timing:"));
        assert!(summary.contains("model load: 1ms"));
        assert!(summary.contains("phase records: 6"));
        assert!(summary.contains("accounted phase time: 9ms"));
        assert!(!summary.contains("validation"));
        assert!(!summary.contains("stages:"));
        assert!(!summary.contains("input synthesis:"));
        assert!(!summary.contains("encode:"));
    }

    #[test]
    fn token_trace_labels_special_terminal_byte_unknown_and_text_tokens() {
        let decoder = TokenDecoder {
            tokens: std::collections::BTreeMap::from([
                (1, String::from("<eos>")),
                (2, String::from("<bos>")),
                (3, String::from("late")),
                (10, String::from("<0xC3>")),
                (150917, String::from("Ciao")),
            ]),
            special_ids: std::collections::BTreeSet::from([1, 2]),
            terminal_ids: std::collections::BTreeSet::from([1]),
        };
        let result = staged_infer::InferenceResult {
            generated_token_count: 6,
            generated_token_ids: vec![2, 150917, 10, 999, 1, 3],
            generated_token_ids_sha256: String::from("sha"),
            generated_text: String::from("Ciao"),
            stop_reason: String::from("eos"),
            ..Default::default()
        };

        let trace = render_token_trace(&result, &decoder);

        assert!(trace.contains(r#"0: id=2 token="<bos>" kind=special rendered=<skipped>"#));
        assert!(trace.contains(r#"1: id=150917 token="Ciao" kind=text rendered="Ciao""#));
        assert!(trace.contains(r#"2: id=10 token="<0xC3>" kind=byte rendered=<byte 0xC3>"#));
        assert!(trace.contains("3: id=999 token=<unknown> kind=unknown rendered=<unknown>"));
        assert!(trace.contains(r#"4: id=1 token="<eos>" kind=terminal rendered=<stops text>"#));
        assert!(trace.contains(r#"5: id=3 token="late" kind=after_stop rendered=<skipped>"#));
    }

    #[test]
    fn model_import_args_build_typed_config() {
        let config = ModelImportArgs {
            model_dir: PathBuf::from("model"),
            manifest: Some(PathBuf::from("Raster.toml")),
            artifact_root: PathBuf::from(inference_artifacts::MODEL_ARTIFACTS_DIR),
            model_id: Some(String::from("gemma")),
            only_tokenizer: true,
            only_layers: false,
            only_embedding: true,
            only_ple: false,
            only_direct: false,
            no_stage_fixtures: false,
            range_size: model_import::DEFAULT_RANGE_SIZE,
        }
        .into_import_config();

        assert_eq!(
            config,
            model_import::ImportConfig {
                model_dir: PathBuf::from("model"),
                manifest: Some(PathBuf::from("Raster.toml")),
                artifact_root: PathBuf::from(inference_artifacts::MODEL_ARTIFACTS_DIR),
                model_id: Some(String::from("gemma")),
                only_tokenizer: true,
                only_layers: false,
                only_embedding: true,
                only_ple: false,
                only_direct: false,
                no_stage_fixtures: false,
                range_size: model_import::DEFAULT_RANGE_SIZE,
            }
        );
    }

    fn direct_timing(stage: &str, millis: u64) -> staged_infer::InferStageTiming {
        let duration = std::time::Duration::from_millis(millis);
        staged_infer::InferStageTiming {
            stage: stage.to_string(),
            routine: String::from("direct-infer"),
            input_synthesis_duration: std::time::Duration::ZERO,
            input_load_duration: std::time::Duration::ZERO,
            kernel_duration: duration,
            encode_duration: std::time::Duration::ZERO,
            total_duration: duration,
        }
    }
}
