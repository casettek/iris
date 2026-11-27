//! The procedure kinds of this build, and the id of the procedure a
//! specification describes: `iwebs procedure`.
//!
//! A caller that runs procedures holds no list of kinds. It asks here which
//! exist and what each is: how many models it runs, which inference its
//! result is of, and what its specification has besides its models.

use crate::*;
use procedure_definition::ProcedureKind;
use serde_json::{json, Map, Value};

#[derive(Debug, Subcommand)]
pub(crate) enum ProcedureCommand {
    /// List the procedure kinds of this build, each with what `describe`
    /// says of it.
    List {
        /// Print the descriptions as one JSON array.
        #[arg(long)]
        json: bool,
    },
    /// Describe one procedure kind: its steps in order, how many models it
    /// runs, and which inference its result is of.
    Describe {
        /// The kind's name, as a specification writes `procedure_kind`.
        kind: String,
        /// Print the description as JSON.
        #[arg(long)]
        json: bool,
    },
    /// The id of the procedure a specification describes: a hash over its
    /// kind's stage programs, its models, its parameters and what it requires
    /// of a run's source and result. Two parties that name one id have
    /// agreed to one procedure. No blockchain is read.
    Id {
        /// The specification of the procedure.
        #[arg(long, default_value = procedure_definition::SPECIFICATION_TOML)]
        specification: PathBuf,
        /// The pins of the kind's programs and models, as `iwebs claim pins`
        /// wrote them. Without it, those the models' imports wrote.
        #[arg(long)]
        pins: Option<PathBuf>,
        /// Print the id with the whole procedure, as a registry holds one.
        #[arg(long)]
        json: bool,
    },
}

/// What a caller needs of a procedure kind to run one and read its result.
fn description(kind: ProcedureKind) -> Value {
    let shape = kind.shape();
    let inferences = kind.inferences();
    let steps: Vec<Value> = shape
        .steps
        .iter()
        .map(|step| {
            let inputs: Map<String, Value> = step
                .inputs
                .iter()
                .map(|(input, from)| (input.to_string(), json!(shape.steps[*from].name)))
                .collect();
            json!({
                "name": step.name,
                "segment": step.segment.name(),
                "program": step.program.map(|program| program.name),
                "inputs": inputs,
            })
        })
        .collect();
    json!({
        "name": kind.name(),
        // One model per inference, in the order a specification names them.
        "inferences": inferences,
        // The inference whose output is the procedure's answer.
        "answer": kind.answer(),
        "agreement": kind.result().states_agreement(),
        "result": {
            "program": kind.result_program(),
            "reads": shape
                .result_reads()
                .iter()
                .map(|(input, inference)| json!({ "input": input, "inference": inference }))
                .collect::<Vec<_>>(),
        },
        "relays": (0..inferences)
            .filter_map(|inference| {
                kind.relayed_from(inference)
                    .map(|from| json!({ "inference": inference, "from": from }))
            })
            .collect::<Vec<_>>(),
        // The tables a specification of the kind has besides its models: a
        // `[source]` it may have, and a `[relay]` it must.
        "specification": {
            "source": shape.runs("prompt-source"),
            "relay": kind.relays(),
        },
        "steps": steps,
    })
}

fn print(kind: ProcedureKind) {
    let shape = kind.shape();
    println!("procedure kind: {}", kind.name());
    println!("inferences: {}", kind.inferences());
    println!(
        "result: {}, of inference {}{}",
        kind.result_program(),
        kind.answer(),
        if kind.result().states_agreement() {
            " and those it agrees with"
        } else {
            ""
        }
    );
    println!("steps:");
    for step in &shape.steps {
        let inputs: Vec<String> = step
            .inputs
            .iter()
            .map(|(input, from)| format!("{input} from {}", shape.steps[*from].name))
            .collect();
        println!(
            "  {:10} {:10} {:17} {}",
            step.name,
            step.segment.name(),
            step.program.map_or("", |program| program.name),
            inputs.join(", ")
        );
    }
}

pub(crate) fn run(command: ProcedureCommand) -> Result<ExitCode> {
    match command {
        ProcedureCommand::List { json } => {
            let kinds = ProcedureKind::all();
            if json {
                let described: Vec<Value> = kinds.into_iter().map(description).collect();
                println!("{}", serde_json::to_string_pretty(&described)?);
            } else {
                for kind in kinds {
                    println!(
                        "{:12} {} model(s), result of inference {}",
                        kind.name(),
                        kind.inferences(),
                        kind.answer()
                    );
                }
            }
        }
        ProcedureCommand::Describe { kind, json } => {
            let kind = ProcedureKind::named(&kind)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&description(kind))?);
            } else {
                print(kind);
            }
        }
        ProcedureCommand::Id {
            specification,
            pins,
            json,
        } => {
            let base_dir =
                std::env::current_dir().context("failed to resolve current directory")?;
            let procedure = procedure_prepare::procedure::spec_procedure(
                &absolute(&base_dir, &specification),
                pins.map(|path| absolute(&base_dir, &path)).as_deref(),
            )?;
            let id = procedure.id();
            if json {
                let registered = procedure_definition::registry::Registered { id, procedure };
                println!("{}", serde_json::to_string_pretty(&registered)?);
            } else {
                println!("procedure: {id}");
            }
        }
    }
    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_description_says_how_many_models_and_which_inference_the_result_is_of() {
        let described = |name: &str| description(ProcedureKind::named(name).unwrap());
        let single = described("single");
        assert_eq!(single["inferences"], 1);
        assert_eq!(single["answer"], 0);
        assert_eq!(single["agreement"], false);
        assert_eq!(single["relays"], json!([]));
        assert_eq!(single["specification"]["relay"], false);

        let consensus = described("consensus");
        assert_eq!(consensus["inferences"], 2);
        assert_eq!(consensus["agreement"], true);
        assert_eq!(
            consensus["result"]["reads"],
            json!([{"input": "first", "inference": 0}, {"input": "second", "inference": 1}])
        );

        // A relay's result is of its second inference, whose prompt holds
        // the first one's output, and its specification has a relay table.
        let relay = described("relay");
        assert_eq!(relay["inferences"], 2);
        assert_eq!(relay["answer"], 1);
        assert_eq!(relay["relays"], json!([{"inference": 1, "from": 0}]));
        assert_eq!(relay["specification"]["relay"], true);
        assert_eq!(relay["result"]["program"], "claim-result");
        let steps: Vec<&str> = relay["steps"]
            .as_array()
            .unwrap()
            .iter()
            .map(|step| step["name"].as_str().unwrap())
            .collect();
        assert_eq!(steps, ["source", "first", "relay", "second", "result"]);
        assert_eq!(relay["steps"][2]["inputs"], json!({"check": "first"}));
        assert_eq!(relay["steps"][1]["program"], Value::Null);
    }
}
