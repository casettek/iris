//! The chain as a claim arranges it: every stage bound to the inputs the
//! claim's checkpoint says it was given.
//!
//! A dispute is settled against the inputs a checkpoint names, so a
//! challenger runs each stage on those and on nothing else. For a claim that
//! gives its stages what the chain template gives them this is the template's
//! own expansion. A claim can give a stage another value it is allowed to
//! name, such as another layer's weights, the marks of another range or the
//! output of another earlier stage: the stage's own checks say so in its
//! output, and a claim that states another output for it is shown false by a
//! replay of the stage on exactly those inputs.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use inference_artifacts::checks::read_input_manifest;
use inference_artifacts::CheckpointList;
use sha2::{Digest, Sha256};
use staged_infer::chain_runner::{expanded_stages, ExpandedInput, ExpandedStage};

/// The arranged chain's file name, beside the run's own manifest.
pub(crate) const ARRANGED_MANIFEST: &str = "claim_arranged.toml";

/// A chain manifest that binds every stage as a claim's checkpoints do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Arranged {
    pub manifest: String,
    /// The inputs the claim gives a stage otherwise than the chain does.
    pub rearranged: Vec<Rearranged>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Rearranged {
    pub checkpoint: usize,
    pub stage: String,
    pub input: String,
}

/// What an audit of the claim settles, said wherever a claim names something
/// no run can be made of.
const AUDIT: &str = "Audit the claim (`iris claim audit`): a program outside the approved set, or \
                     an input that is not an earlier stage's output, a pinned commitment or a \
                     posted value, voids it with no replay";

/// Bind every stage of `claimed` to what its checkpoint names.
///
/// `run_manifest` is the chain prepared from the claim's posted inputs: it
/// says which project each stage runs and holds every file a stage may be
/// given, the model's and the run's. `claim_dir` holds each stage's input
/// list as the claim posted it.
///
/// An input is bound to the stage the checkpoint points at when that stage's
/// stated output is the commitment the list loads, and otherwise to the file
/// the chain holds under that commitment. A pointer that does not lead to
/// what the stage loads is not followed: where the procedure takes the
/// input from an earlier stage the link comparison voids such a claim, and
/// elsewhere the pointer says nothing of what the stage ran on.
pub(crate) fn arrange(
    run_manifest: &Path,
    claimed: &CheckpointList,
    claim_dir: &Path,
) -> Result<Arranged> {
    let template = expanded_stages(run_manifest)?;
    let by_name: BTreeMap<&str, &ExpandedStage> = template
        .iter()
        .map(|stage| (stage.name.as_str(), stage))
        .collect();
    // Every file the chain holds, by its commitment.
    let mut held: BTreeMap<&str, &ExpandedInput> = BTreeMap::new();
    for stage in &template {
        for binding in stage.inputs.values() {
            if let ExpandedInput::External { commitment, .. } = binding {
                held.entry(commitment.as_str()).or_insert(binding);
            }
        }
    }

    let mut stages = Vec::with_capacity(claimed.checkpoints.len());
    let mut rearranged = Vec::new();
    let mut seen = BTreeSet::new();
    for (index, checkpoint) in claimed.checkpoints.iter().enumerate() {
        let stage = checkpoint.stage.as_str();
        ensure!(
            seen.insert(stage),
            "the claim lists the stage `{stage}` twice, and a run has one directory per stage"
        );
        let Some(expected) = by_name.get(stage) else {
            bail!(
                "checkpoint {index} of the claim is for a stage `{stage}`, which the chain of \
                 this run does not have, so there is no program here to run it with"
            );
        };
        let list_path = claim_dir.join(stage).join("input_manifest.json");
        let list = fs::read(&list_path)
            .with_context(|| format!("failed to read {}", list_path.display()))?;
        ensure!(
            hex::encode(Sha256::digest(&list)) == checkpoint.input_commitment,
            "{} is not the input list checkpoint {index} (`{stage}`) commits to",
            list_path.display()
        );
        let Some(loads) = read_input_manifest(&list) else {
            bail!(
                "the input list of checkpoint {index} (`{stage}`) is not in its canonical form. \
                 {AUDIT}"
            );
        };
        ensure!(
            loads.keys().eq(expected.inputs.keys()),
            "checkpoint {index} (`{stage}`) lists the inputs {:?}, and the stage takes {:?}. \
             {AUDIT}",
            loads.keys().collect::<Vec<_>>(),
            expected.inputs.keys().collect::<Vec<_>>()
        );

        let mut inputs = BTreeMap::new();
        for (input, loaded) in &loads {
            let produced = checkpoint
                .input_sources
                .get(input)
                .map(|&producer| producer as usize)
                .filter(|&producer| producer < index)
                .map(|producer| &claimed.checkpoints[producer])
                .filter(|producer| &producer.output_commitment == loaded);
            let own = &expected.inputs[input];
            let binding = if let Some(producer) = produced {
                ExpandedInput::From(producer.stage.clone())
            } else if matches!(own, ExpandedInput::External { commitment, .. } if commitment == loaded)
            {
                own.clone()
            } else if let Some(file) = held.get(loaded.as_str()) {
                (*file).clone()
            } else {
                bail!(
                    "checkpoint {index} (`{stage}`) gives `{input}` the value {loaded}, which is \
                     not the output of the stage it points at and is no value the chain holds. \
                     {AUDIT}"
                );
            };
            if !same_binding(own, &binding) {
                rearranged.push(Rearranged {
                    checkpoint: index,
                    stage: stage.to_string(),
                    input: input.clone(),
                });
            }
            inputs.insert(input.clone(), binding);
        }
        stages.push(ExpandedStage {
            name: stage.to_string(),
            project: expected.project.clone(),
            inputs,
        });
    }
    Ok(Arranged {
        manifest: render(run_manifest, &stages)?,
        rearranged,
    })
}

/// Whether two bindings give a stage the same thing: the same producer, or a
/// file of the same commitment.
fn same_binding(left: &ExpandedInput, right: &ExpandedInput) -> bool {
    match (left, right) {
        (ExpandedInput::From(left), ExpandedInput::From(right)) => left == right,
        (
            ExpandedInput::External {
                commitment: left, ..
            },
            ExpandedInput::External {
                commitment: right, ..
            },
        ) => left == right,
        _ => false,
    }
}

/// The stages as a chain manifest with nothing to expand.
fn render(run_manifest: &Path, stages: &[ExpandedStage]) -> Result<String> {
    let document: toml::Value = toml::from_str(
        &fs::read_to_string(run_manifest)
            .with_context(|| format!("failed to read {}", run_manifest.display()))?,
    )
    .with_context(|| format!("failed to parse {}", run_manifest.display()))?;
    let header = |key: &str| -> Result<String> {
        Ok(quoted(
            document
                .get("chain")
                .and_then(|chain| chain.get(key))
                .and_then(toml::Value::as_str)
                .with_context(|| format!("{} has no chain {key}", run_manifest.display()))?,
        ))
    };
    let mut out = format!(
        "[chain]\nname = {}\nversion = {}\n",
        header("name")?,
        header("version")?
    );
    for stage in stages {
        out.push_str(&format!(
            "\n[[chain.stage]]\nname = {}\nproject = {}\n",
            quoted(&stage.name),
            quoted(&stage.project)
        ));
        for (input, binding) in &stage.inputs {
            ensure!(
                input
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-'),
                "`{input}` is no input name"
            );
            let bound = match binding {
                ExpandedInput::From(from) => format!("{{ from = {} }}", quoted(from)),
                ExpandedInput::External {
                    path,
                    index_path,
                    commitment,
                } => {
                    let index_path = index_path
                        .as_ref()
                        .map(|path| format!("index_path = {}, ", quoted(path)))
                        .unwrap_or_default();
                    format!(
                        "{{ external = {{ path = {}, {index_path}commitment = {} }} }}",
                        quoted(path),
                        quoted(commitment)
                    )
                }
            };
            out.push_str(&format!("inputs.{input} = {bound}\n"));
        }
    }
    Ok(out)
}

fn quoted(text: &str) -> String {
    toml::Value::String(text.to_string()).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use inference_artifacts::Checkpoint;

    const RUN: &str = r#"[chain]
name = "iris"
version = "0.1.0"

[chain.input.weights]
index = "l"
path = "model/layer{l}.rastered"
index_path = "model/layer{l}.rindex"
commitments = ["aa", "bb"]

[[chain.stage]]
name = "first"
project = "stages/first"
inputs.text = { external = { path = "run/text.rastered", index_path = "run/text.rindex", commitment = "11" } }

[[chain.repeat]]
name = "layers"
index = "l"
count = 2

  [[chain.repeat.stage]]
name = "layer_l{l}"
project = "stages/layer"
inputs.layer = { input = "weights_{l}" }
inputs.state = { from = "layer_l{l-1}", first = "first" }
"#;

    /// A run directory with the manifest above, and a claim directory with
    /// the input list of every stage as `lists` gives it.
    fn fixture(
        name: &str,
        lists: &[(&str, &[(&str, &str)], &[(&str, u32)])],
    ) -> (std::path::PathBuf, CheckpointList, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "iris-arrangement-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let run_manifest = dir.join("Raster.toml");
        fs::write(&run_manifest, RUN).unwrap();
        let claim_dir = dir.join("claim");
        let mut checkpoints = Vec::new();
        for (stage, inputs, sources) in lists {
            let list: BTreeMap<String, BTreeMap<&str, &str>> = inputs
                .iter()
                .map(|(input, commitment)| {
                    (
                        input.to_string(),
                        BTreeMap::from([
                            ("commitment", *commitment),
                            ("encoding", "raster"),
                            ("type", "sha256"),
                        ]),
                    )
                })
                .collect();
            let bytes = serde_json::to_vec_pretty(&list).unwrap();
            fs::create_dir_all(claim_dir.join(stage)).unwrap();
            fs::write(claim_dir.join(stage).join("input_manifest.json"), &bytes).unwrap();
            checkpoints.push(Checkpoint {
                stage: stage.to_string(),
                program_commitment: String::from("p"),
                input_commitment: hex::encode(Sha256::digest(&bytes)),
                input_sources: sources
                    .iter()
                    .map(|(input, producer)| (input.to_string(), *producer))
                    .collect(),
                output_commitment: format!("out-{stage}"),
                output_sha256: format!("sha-{stage}"),
            });
        }
        let list = CheckpointList { checkpoints };
        (run_manifest, list, claim_dir)
    }

    fn written(name: &str, arranged: &Arranged) -> Vec<ExpandedStage> {
        let path = std::env::temp_dir().join(format!(
            "iris-arrangement-{name}-{}.toml",
            std::process::id()
        ));
        fs::write(&path, &arranged.manifest).unwrap();
        expanded_stages(&path).unwrap()
    }

    #[test]
    fn a_claim_arranged_as_the_chain_is_the_chain() {
        let (run, list, claim) = fixture(
            "honest",
            &[
                ("first", &[("text", "11")], &[]),
                (
                    "layer_l0",
                    &[("layer", "aa"), ("state", "out-first")],
                    &[("state", 0)],
                ),
                (
                    "layer_l1",
                    &[("layer", "bb"), ("state", "out-layer_l0")],
                    &[("state", 1)],
                ),
            ],
        );
        let arranged = arrange(&run, &list, &claim).unwrap();
        assert!(arranged.rearranged.is_empty());
        assert_eq!(written("honest", &arranged), expanded_stages(&run).unwrap());
    }

    #[test]
    fn a_stage_is_bound_to_what_its_checkpoint_names() {
        // The second layer is given the first layer's weights and the first
        // stage's output: both are values the chain holds.
        let (run, list, claim) = fixture(
            "rearranged",
            &[
                ("first", &[("text", "11")], &[]),
                (
                    "layer_l0",
                    &[("layer", "aa"), ("state", "out-first")],
                    &[("state", 0)],
                ),
                (
                    "layer_l1",
                    &[("layer", "aa"), ("state", "out-first")],
                    &[("state", 0)],
                ),
            ],
        );
        let arranged = arrange(&run, &list, &claim).unwrap();
        assert_eq!(
            arranged
                .rearranged
                .iter()
                .map(|found| (found.checkpoint, found.input.as_str()))
                .collect::<Vec<_>>(),
            [(2, "layer"), (2, "state")]
        );
        let stages = written("rearranged", &arranged);
        assert_eq!(
            stages[2].inputs["state"],
            ExpandedInput::From(String::from("first"))
        );
        assert_eq!(
            stages[2].inputs["layer"],
            ExpandedInput::External {
                path: String::from("model/layer0.rastered"),
                index_path: Some(String::from("model/layer0.rindex")),
                commitment: String::from("aa"),
            }
        );
    }

    #[test]
    fn a_pointer_that_does_not_lead_to_what_is_loaded_is_not_followed() {
        // The text is a file of the run. A pointer beside it says nothing of
        // what the stage ran on.
        let (run, list, claim) = fixture(
            "pointer",
            &[
                ("first", &[("text", "11")], &[]),
                (
                    "layer_l0",
                    &[("layer", "aa"), ("state", "out-first")],
                    &[("layer", 0), ("state", 0)],
                ),
            ],
        );
        let arranged = arrange(&run, &list, &claim).unwrap();
        assert!(arranged.rearranged.is_empty());
    }

    #[test]
    fn a_value_nobody_holds_is_left_to_the_audit() {
        let (run, list, claim) = fixture(
            "unheld",
            &[
                ("first", &[("text", "11")], &[]),
                (
                    "layer_l0",
                    &[("layer", "cc"), ("state", "out-first")],
                    &[("state", 0)],
                ),
            ],
        );
        let refused = arrange(&run, &list, &claim).unwrap_err().to_string();
        assert!(refused.contains("is no value the chain holds"), "{refused}");
    }
}
