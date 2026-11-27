//! Writing the record of a run, and checking one.
//!
//! A record (`procedure_definition::run_record`) is what the party that made
//! a run hands to anyone who wants to make it again: what it states (the
//! procedure by its id, the inputs the run was given, the result), and what
//! backs that, one entry for each method the run was made for. The
//! run this module records is a pipeline run, which leaves a checkpoint at
//! every stage, so its record carries the fault provable backing.
//!
//! Checking one is a pipeline run on the record's inputs under the checker's
//! own specification, which has to describe the record's procedure. The
//! checker generates nothing: the stages hold the output the record states
//! to the model's own choice, token by token. The result is compared with
//! the one the record states, and, when the record carries the fault
//! provable backing, every checkpoint with the record's, in order. A record
//! that holds says what the procedure gives on those inputs, whoever ran it.
//!
//! Where two runs differ at a stage, this names the first. It does not
//! settle who is right: with a blockchain, a challenge and a fault proof do.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, ensure, Context, Result};
use procedure_definition::divergence::locate_divergence;
use procedure_definition::registry::{Procedure, Registered};
use procedure_definition::run_record::{
    Backing, RunRecord, RECORD_PROCEDURE_JSON, RUN_RECORD_JSON,
};
use procedure_definition::{
    read_case, read_checkpoint_list, read_json, read_specification, write_json, Divergence,
    InferenceResult, CHECKPOINTS_TXT, CHECKPOINT_LIST_JSON,
};
use procedure_native::absolute;
use procedure_pipeline::pipeline_runner::NativeStageBackend;
use procedure_prepare::pipeline_run::PipelineRun;
use procedure_prepare::run_inputs;

use crate::{differing_field, execute, first_unapproved, CheckpointedRun};

/// Writes the record of a pipeline run of `procedure` into `dir`, with the
/// fault provable backing its checkpoints are, and returns the path of its
/// `record.json`. A run with a stage the procedure does not approve is not a
/// run of it, and has no record.
pub fn write_record(run: &CheckpointedRun, procedure: &Procedure, dir: &Path) -> Result<PathBuf> {
    let id = procedure.id();
    if let Some(stage) = first_unapproved(run, procedure)? {
        bail!("stage `{stage}` ran a program procedure {id} does not approve");
    }
    let run_dir = run
        .manifest_path
        .parent()
        .context("the run's pipeline has no directory")?;
    let inputs = run_inputs::load(run_dir).context("the run left no inputs to record")?;
    let list = read_checkpoint_list(&run.checkpoint_list_path)?;

    fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    run_inputs::save(&inputs, dir)?;
    for (from, name) in [
        (&run.checkpoint_list_path, CHECKPOINT_LIST_JSON),
        (&run.checkpoint_hashes_path, CHECKPOINTS_TXT),
    ] {
        let to = dir.join(name);
        fs::copy(from, &to)
            .with_context(|| format!("failed to copy {} to {}", from.display(), to.display()))?;
    }
    write_json(
        &dir.join(RECORD_PROCEDURE_JSON),
        &Registered {
            id: id.clone(),
            procedure: procedure.clone(),
        },
    )?;
    let path = dir.join(RUN_RECORD_JSON);
    write_json(
        &path,
        &RunRecord::of(
            id,
            run.results.clone(),
            run.consensus,
            vec![Backing::fault_provable(&list)?],
        ),
    )?;
    Ok(path)
}

/// What a record is checked from.
#[derive(Debug, Clone)]
pub struct CheckOptions {
    /// Where the check's files are written, under `target/`. A relative
    /// specification or record path is taken from here.
    pub base_dir: PathBuf,
    /// The checker's own specification of the record's procedure: its own
    /// imports of the models.
    pub specification_path: PathBuf,
    /// The record's directory.
    pub record_dir: PathBuf,
    /// The binary the pipeline runner starts its workers from.
    pub current_exe: PathBuf,
    pub native_stage_backend: NativeStageBackend,
}

/// How a record came out against a run of its inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordCheck {
    /// The id of the procedure the record and the checker's specification
    /// both name.
    pub procedure: String,
    /// The checker's run on the record's inputs.
    pub run: CheckpointedRun,
    /// Whether the record carries the fault provable backing, so that its
    /// checkpoints were compared with the run's.
    pub checkpoints_compared: bool,
    /// The first stage where the record's checkpoints are not the run's.
    pub divergence: Option<Divergence>,
    /// The first thing the record states of a result that the run does not.
    pub result_difference: Option<String>,
}

impl RecordCheck {
    /// Whether the record states what the run on its inputs gives.
    pub fn holds(&self) -> bool {
        self.divergence.is_none() && self.result_difference.is_none()
    }
}

/// Makes a recorded run again and compares it with the record.
///
/// It fails, before anything runs, when the record is not one: its procedure
/// does not hash to the id it states, the checker's specification describes
/// another procedure, or the checkpoint list beside a fault provable backing
/// is not the one the backing states. A record that is one and states
/// something the run does not is no failure here: [`RecordCheck`] says where
/// they differ.
pub fn check_record(options: &CheckOptions) -> Result<RecordCheck> {
    let record_dir = absolute(&options.base_dir, &options.record_dir);
    let record = RunRecord::read(&record_dir)?;
    let procedure_path = record_dir.join(RECORD_PROCEDURE_JSON);
    let held: Registered = read_json(&procedure_path)?;
    ensure!(
        held.id == record.procedure && held.procedure.id() == held.id,
        "the procedure in {} does not hash to the id the record states, {}",
        procedure_path.display(),
        record.procedure
    );
    let specification_path = absolute(&options.base_dir, &options.specification_path);
    let procedure = procedure_prepare::procedure::spec_procedure(&specification_path, None)?;
    ensure!(
        procedure.id() == record.procedure,
        "the record is of procedure {}, and {} describes {}",
        record.procedure,
        specification_path.display(),
        procedure.id()
    );
    // The checkpoints are the fault provable backing's. A record without it
    // states a result and no stage to compare.
    let record_list_path = record_dir.join(CHECKPOINT_LIST_JSON);
    let record_list = match record.fault_provable() {
        Some(stated) => {
            let list = read_checkpoint_list(&record_list_path)?;
            if let Some(mismatch) = stated.list_mismatch(&list)? {
                bail!(
                    "{} is not the record's checkpoint list: {mismatch}",
                    record_list_path.display()
                );
            }
            Some(list)
        }
        None => None,
    };

    // The run again, on the record's inputs: every file a stage is given
    // from the run is written from the record's own value.
    let inputs = run_inputs::load(&record_dir)?;
    let case_path = run_inputs::write_case(&inputs, &work_dir(&options.base_dir))?;
    let case = read_case(&case_path)?;
    let specification = read_specification(&specification_path)?;
    let generated = |_: usize, _: Option<&InferenceResult>| -> Result<InferenceResult> {
        bail!("a record's check generates nothing: the record states the output")
    };
    let prepared = procedure_prepare::pipeline_run::prepare_run(PipelineRun {
        base_dir: &options.base_dir,
        specification_path: specification_path.clone(),
        specification: &specification,
        case_path: Some(case_path),
        case: &case,
        frozen: None,
        given: Some(&inputs),
        given_ids: None,
        check_claimed_length: false,
        discover: &generated,
    })?;
    let run = execute(
        &options.base_dir,
        &options.current_exe,
        options.native_stage_backend,
        &prepared,
        specification.procedure_kind,
    )?;
    if let Some(stage) = first_unapproved(&run, &procedure)? {
        bail!(
            "stage `{stage}` ran a program procedure {} does not approve: the stage was locked \
             again after the model was imported, so import the model again",
            record.procedure
        );
    }
    let divergence = match &record_list {
        Some(record_list) => locate_divergence(
            &record_list_path,
            record_list,
            &run.checkpoint_list_path,
            &read_checkpoint_list(&run.checkpoint_list_path)?,
        ),
        None => None,
    };
    let result_difference = stated_difference(&record, &run)?;
    Ok(RecordCheck {
        procedure: record.procedure,
        run,
        checkpoints_compared: record_list.is_some(),
        divergence,
        result_difference,
    })
}

/// The first thing a record states of a result that a run does not.
fn stated_difference(record: &RunRecord, run: &CheckpointedRun) -> Result<Option<String>> {
    if record.results.len() != run.results.len() {
        return Ok(Some(format!(
            "the record states {} result(s), and the run on its inputs {}",
            record.results.len(),
            run.results.len()
        )));
    }
    for (inference, (recorded, stated)) in record.results.iter().zip(&run.results).enumerate() {
        if let Some(difference) = differing_field(recorded, stated)? {
            return Ok(Some(match difference {
                Some((field, recorded, stated)) => format!(
                    "inference {inference}: the record states `{field}` as {recorded}, and the \
                     run on its inputs as {stated}"
                ),
                None => format!(
                    "inference {inference}: the record's result is not that of the run on its \
                     inputs"
                ),
            }));
        }
    }
    Ok((record.consensus != run.consensus).then(|| {
        format!(
            "the record states the consensus {:?}, and the run on its inputs {:?}",
            record.consensus, run.consensus
        )
    }))
}

/// A new directory for the case a check reads the record's inputs as.
fn work_dir(base_dir: &Path) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    base_dir
        .join("target/procedure-record")
        .join(format!("{nanos}-pid{}", std::process::id()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use procedure_definition::ProcedureKind;

    fn run(results: Vec<InferenceResult>) -> CheckpointedRun {
        CheckpointedRun {
            procedure_kind: ProcedureKind::single(),
            manifest_path: PathBuf::new(),
            pipeline_dir: PathBuf::new(),
            checkpoint_list_path: PathBuf::new(),
            checkpoint_hashes_path: PathBuf::new(),
            checkpoints: 0,
            results,
            consensus: None,
        }
    }

    #[test]
    fn a_record_that_states_another_result_is_told_which_field() {
        let stated = InferenceResult {
            generated_text: String::from("yes"),
            valid: true,
            result_code: 1,
            ..Default::default()
        };
        let record = RunRecord::of("ab".repeat(32), vec![stated.clone()], None, Vec::new());
        assert_eq!(
            stated_difference(&record, &run(vec![stated.clone()])).unwrap(),
            None
        );

        // The record says code 1, and the run on its inputs gives code 0.
        let mut other = stated.clone();
        other.result_code = 0;
        let difference = stated_difference(&record, &run(vec![other]))
            .unwrap()
            .unwrap();
        assert!(
            difference.contains("`result_code` as 1") && difference.ends_with("as 0"),
            "{difference}"
        );

        let difference = stated_difference(&record, &run(Vec::new()))
            .unwrap()
            .unwrap();
        assert!(
            difference.starts_with("the record states 1 result(s)"),
            "{difference}"
        );
    }
}
