//! The record of a run: what the party that made a run hands to anyone who
//! wants to make it again.
//!
//! A record has two parts. The statement is the same whatever method the run
//! was made for: the id of the procedure the run is of, the inputs it was
//! given, and each inference's result. The backings are what stands behind
//! the statement besides making the run again: one entry for each method the
//! run was made for, named by its method. A record with none is backed by
//! making the run again, and by nothing else.
//!
//! One method has a backing today. A run made to be fault provable leaves a
//! checkpoint at every stage, and its backing is the count and root of those
//! checkpoints: what a claim commits to on a blockchain, and what a challenge
//! opens one stage of.
//!
//! A record is a directory. `record.json` is this file's [`RunRecord`].
//! Beside it are the procedure itself (`procedure.json`), the inputs the run
//! was given as values (`posted_run.bin`: the text, a source's documents with
//! their attestations, the forms, and the output each inference is held to),
//! and what a backing is of: for the fault provable one, the checkpoint list.
//!
//! Nothing in a record needs a blockchain. A claim posts the same inputs and
//! the same checkpoints, and names the same procedure id.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

use crate::checkpoint::{checkpoint_hashes, checkpoints_root, CheckpointList};
use crate::inference::{ConsensusOutcome, InferenceResult};

pub const RUN_RECORD_JSON: &str = "record.json";
/// The procedure a record's run is of, with its id, as a registry holds one.
pub const RECORD_PROCEDURE_JSON: &str = "procedure.json";
pub const RUN_RECORD_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RunRecord {
    pub version: u32,
    /// The id of the procedure the run is of.
    pub procedure: String,
    /// Each inference's result, in the order the procedure kind runs them.
    pub results: Vec<InferenceResult>,
    /// What a consensus run states of its two inferences.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub consensus: Option<ConsensusOutcome>,
    /// What stands behind the statement besides making the run again: at
    /// most one entry for each method. With none, the record is backed by
    /// making the run again alone.
    pub backings: Vec<Backing>,
}

/// What one method leaves behind a record's statement. The method is the
/// entry's tag, so a record says which methods its run was made for.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum Backing {
    /// The run left a checkpoint at every stage, so that one stage of it can
    /// be proven wrong: Raster's method, and what a claim posts.
    FaultProvable(StatedCheckpoints),
}

/// The checkpoints a fault provable run states: enough to hold a checkpoint
/// list to, and what a claim commits to.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StatedCheckpoints {
    /// How many stages ran: one checkpoint each.
    pub checkpoints: u32,
    /// The root of the tree over the checkpoints' hashes, in order.
    pub checkpoints_root: String,
}

impl Backing {
    /// The backing of a run that left these checkpoints.
    pub fn fault_provable(list: &CheckpointList) -> Result<Self> {
        Ok(Self::FaultProvable(StatedCheckpoints::of(list)?))
    }

    /// The method's name, as a record writes it.
    pub fn method(&self) -> &'static str {
        match self {
            Self::FaultProvable(_) => "fault_provable",
        }
    }
}

impl StatedCheckpoints {
    /// What a run that left these checkpoints states of them.
    pub fn of(list: &CheckpointList) -> Result<Self> {
        Ok(Self {
            checkpoints: list.checkpoints.len() as u32,
            checkpoints_root: checkpoints_root(&checkpoint_hashes(list)?)?,
        })
    }

    /// Why this checkpoint list is not the one stated, if it is not: another
    /// count, or another root.
    pub fn list_mismatch(&self, list: &CheckpointList) -> Result<Option<String>> {
        if list.checkpoints.len() as u32 != self.checkpoints {
            return Ok(Some(format!(
                "it holds {} checkpoint(s), and the record states {}",
                list.checkpoints.len(),
                self.checkpoints
            )));
        }
        let root = checkpoints_root(&checkpoint_hashes(list)?)?;
        Ok((root != self.checkpoints_root).then(|| {
            format!(
                "its root is {root}, and the record states {}",
                self.checkpoints_root
            )
        }))
    }
}

impl RunRecord {
    /// The record of a run of this procedure that states these results, with
    /// what the run left behind them.
    pub fn of(
        procedure: String,
        results: Vec<InferenceResult>,
        consensus: Option<ConsensusOutcome>,
        backings: Vec<Backing>,
    ) -> Self {
        Self {
            version: RUN_RECORD_VERSION,
            procedure,
            results,
            consensus,
            backings,
        }
    }

    /// The record in this directory.
    pub fn read(dir: &Path) -> Result<Self> {
        let path = dir.join(RUN_RECORD_JSON);
        let record: Self = crate::read_json(&path)
            .with_context(|| format!("{} is not a run record of this build", path.display()))?;
        ensure!(
            record.version == RUN_RECORD_VERSION,
            "{} is run record version {}, and this build reads version {RUN_RECORD_VERSION}",
            path.display(),
            record.version
        );
        record
            .one_backing_for_each_method()
            .with_context(|| format!("{} is not a run record", path.display()))?;
        Ok(record)
    }

    /// The checkpoints the record states, when its run was made to be fault
    /// provable.
    pub fn fault_provable(&self) -> Option<&StatedCheckpoints> {
        self.backings
            .iter()
            .map(|Backing::FaultProvable(stated)| stated)
            .next()
    }

    /// A record states one thing for each method: two backings of one method
    /// could state two.
    fn one_backing_for_each_method(&self) -> Result<()> {
        let mut methods = BTreeSet::new();
        for backing in &self.backings {
            ensure!(
                methods.insert(backing.method()),
                "it holds two backings of the method `{}`",
                backing.method()
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::Checkpoint;

    fn list(outputs: &[&str]) -> CheckpointList {
        CheckpointList {
            checkpoints: outputs
                .iter()
                .enumerate()
                .map(|(index, output)| Checkpoint {
                    stage: format!("stage_{index}"),
                    program_commitment: "ab".repeat(32),
                    input_commitment: "cd".repeat(32),
                    output_commitment: output.repeat(32),
                    output_sha256: "ef".repeat(32),
                    ..Checkpoint::default()
                })
                .collect(),
        }
    }

    #[test]
    fn a_record_states_the_checkpoints_of_its_run_and_no_others() {
        let own = list(&["01", "02"]);
        let stated = StatedCheckpoints::of(&own).unwrap();
        assert_eq!(stated.checkpoints, 2);
        assert_eq!(stated.list_mismatch(&own).unwrap(), None);

        // Another output at one stage is another root.
        let mismatch = stated.list_mismatch(&list(&["01", "03"])).unwrap().unwrap();
        assert!(mismatch.starts_with("its root is "), "{mismatch}");
        // A stage more is another count.
        let mismatch = stated
            .list_mismatch(&list(&["01", "02", "03"]))
            .unwrap()
            .unwrap();
        assert!(
            mismatch.starts_with("it holds 3 checkpoint(s)"),
            "{mismatch}"
        );
    }

    #[test]
    fn a_record_names_the_method_of_each_backing() {
        let own = list(&["01", "02"]);
        let backing = Backing::fault_provable(&own).unwrap();
        let record = RunRecord::of("ab".repeat(32), Vec::new(), None, vec![backing.clone()]);
        assert_eq!(
            record.fault_provable(),
            Some(&StatedCheckpoints::of(&own).unwrap())
        );
        record.one_backing_for_each_method().unwrap();

        // It reads back as written, with the method as the backing's tag.
        let text = serde_json::to_string(&record).unwrap();
        assert!(text.contains(r#""method":"fault_provable""#), "{text}");
        assert_eq!(serde_json::from_str::<RunRecord>(&text).unwrap(), record);

        // An unknown field is refused, in the record and in a backing, and
        // so is a method this build has no check for.
        let extra = text.replacen('{', "{\"signed\":true,", 1);
        assert!(serde_json::from_str::<RunRecord>(&extra).is_err());
        let extra = text.replace("\"method\":", "\"signed\":true,\"method\":");
        assert!(serde_json::from_str::<RunRecord>(&extra).is_err());
        let unknown = text.replace("fault_provable", "attested");
        assert!(serde_json::from_str::<RunRecord>(&unknown).is_err());

        // Two backings of one method are not one record.
        let twice = RunRecord::of(
            "ab".repeat(32),
            Vec::new(),
            None,
            vec![backing.clone(), backing],
        );
        assert!(twice.one_backing_for_each_method().is_err());
    }

    #[test]
    fn a_record_with_no_backing_is_the_statement_alone() {
        let record = RunRecord::of("ab".repeat(32), Vec::new(), None, Vec::new());
        assert_eq!(record.fault_provable(), None);
        record.one_backing_for_each_method().unwrap();
        let text = serde_json::to_string(&record).unwrap();
        assert!(text.contains(r#""backings":[]"#), "{text}");
        assert_eq!(serde_json::from_str::<RunRecord>(&text).unwrap(), record);
    }
}
