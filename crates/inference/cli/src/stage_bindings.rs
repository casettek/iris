//! Every value a stage's tiles read must be pinned by the stage's control-flow
//! schema.
//!
//! Two spellings in a sequence compile, run, commit and audit, and still leave
//! a value to whoever writes the trace:
//!
//! - a literal passed to a tile, a sequence or a recur. The schema records the
//!   argument as inline and holds no bytes for it, and the transition guest
//!   checks only that the argument is inline;
//! - a recur's `state = …` seed that is a `select!` into a larger value. The
//!   guest ties a loop's first step to its seed only when the seed is a whole
//!   stored value, and otherwise takes the first step's word for it.
//!
//! Raster's check ladder sees neither, so this reads each stage's schema, built
//! from source the way `cargo raster cfs` builds it, and refuses both. A
//! constant or a seed a sequence needs is returned by a tile and passed whole.
//!
//! The parity gate runs the check before anything else, and so does
//! `cargo test`.

use anyhow::{ensure, Context, Result};
use raster_compiler::{CfsBuilder, Project};
use serde_json::Value;
use std::path::{Path, PathBuf};

/// The stage programs: every crate under `raster-stages`.
fn stage_dirs(root: &Path) -> Result<Vec<PathBuf>> {
    let stages = root.join("raster-stages");
    let mut dirs = Vec::new();
    for entry in std::fs::read_dir(&stages).with_context(|| format!("read {}", stages.display()))? {
        let path = entry?.path();
        if path.join("Cargo.toml").is_file() {
            dirs.push(path);
        }
    }
    dirs.sort();
    Ok(dirs)
}

/// Whether a binding, or anything it indexes by, is an inline value.
fn has_inline(binding: &Value) -> bool {
    if binding
        .get("Direct")
        .is_some_and(|source| source == "Inline")
    {
        return true;
    }
    binding.get("Indexed").is_some_and(|indexed| {
        has_inline(&indexed["value"])
            || indexed["indexes"]
                .as_array()
                .is_some_and(|indexes| indexes.iter().any(has_inline))
    })
}

/// Whether a binding is the whole result of an earlier step of the sequence.
fn is_whole_result(binding: &Value) -> bool {
    binding
        .get("PriorItemOutput")
        .and_then(|prior| prior["path"].as_array())
        .is_some_and(Vec::is_empty)
}

/// Every binding of a schema that no proof pins, as `sequence[item] tile: what`.
fn unpinned_bindings(cfs: &Value) -> Vec<String> {
    let mut found = Vec::new();
    for sequence in cfs["sequences"].as_array().unwrap() {
        let name = sequence["id"].as_str().unwrap();
        for (index, item) in sequence["items"].as_array().unwrap().iter().enumerate() {
            let (kind, body) = item.as_object().unwrap().iter().next().unwrap();
            let at = format!("{name}[{index}] {}", body["id"].as_str().unwrap());
            let sources = body["sources"].as_array().unwrap();
            let mut arguments = 0;
            if kind == "RecurTile" || kind == "RecurSequence" {
                if has_inline(&sources[0]) {
                    found.push(format!("{at}: the recur's input is an inline value"));
                }
                arguments = 1;
                if body["carries_state"] == true {
                    if !is_whole_result(&sources[arguments]) {
                        found.push(format!(
                            "{at}: the state seed is not the whole result of a step ({})",
                            sources[arguments]
                        ));
                    }
                    arguments += 1;
                }
                if !body["output"].is_null() {
                    // A site that creates its object has an inline slot here,
                    // and the schema states the root it opens at.
                    if body["output"]["derives"] == true && !is_whole_result(&sources[arguments]) {
                        found.push(format!(
                            "{at}: the output base is not the whole result of a step ({})",
                            sources[arguments]
                        ));
                    }
                    arguments += 1;
                }
            }
            for (position, source) in sources.iter().enumerate().skip(arguments) {
                if has_inline(source) {
                    found.push(format!("{at}: argument {position} is an inline value"));
                }
            }
        }
    }
    found
}

/// Refuses a stage program that reads a value its schema does not pin.
/// `root` is the repository.
pub(crate) fn check_stage_programs(root: &Path) -> Result<()> {
    let stages = stage_dirs(root)?;
    ensure!(stages.len() >= 6, "found {} stage programs", stages.len());
    let mut found = Vec::new();
    for stage in stages {
        let name = stage.file_name().unwrap().to_string_lossy().into_owned();
        let project = Project::new(stage.clone())
            .map_err(|error| anyhow::anyhow!("open stage program {name}: {error}"))?;
        let cfs = CfsBuilder::new(&project)
            .build()
            .map_err(|error| anyhow::anyhow!("build the schema of {name}: {error}"))?;
        let cfs = serde_json::to_value(&cfs)?;
        ensure!(
            cfs["sequences"]
                .as_array()
                .is_some_and(|all| !all.is_empty()),
            "{name} has no sequences"
        );
        found.extend(
            unpinned_bindings(&cfs)
                .into_iter()
                .map(|what| format!("{name}: {what}")),
        );
    }
    ensure!(
        found.is_empty(),
        "stage programs read values no proof pins; return each from a tile and pass the whole          result:
{}",
        found.join("
")
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn no_stage_reads_a_value_its_schema_does_not_pin() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .canonicalize()
            .unwrap();
        check_stage_programs(&root).unwrap();
    }

    /// `prompt-check` unrolls its merge rounds in its sequences, where no
    /// constant can count them. The importer admits a merge result only if
    /// two such tokens merge within the rounds of a pass, so the rounds the
    /// schema holds must be the ones that bound is computed from.
    #[test]
    fn prompt_check_unrolls_the_rounds_the_importer_counts_on() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .canonicalize()
            .unwrap();
        let project = Project::new(root.join("raster-stages/prompt-check")).unwrap();
        let cfs = serde_json::to_value(CfsBuilder::new(&project).build().unwrap()).unwrap();
        let calls = |sequence: &str, callee: &str| {
            cfs["sequences"]
                .as_array()
                .unwrap()
                .iter()
                .find(|entry| entry["id"] == sequence)
                .unwrap_or_else(|| panic!("no sequence {sequence}"))["items"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|item| item["Sequence"]["id"] == callee)
                .count()
        };
        let rounds = calls("check_pass", "check_eight") * calls("check_eight", "check_round");
        assert_eq!(rounds, 2 * (run_prep::MAX_MERGED_TOKEN_CHARS - 1));
        // Both passes, and the scan that closes each.
        assert_eq!(calls("main", "check_pass"), 2);
        assert_eq!(calls("main", "closing_scan"), 2);
    }

    fn schema(item: Value) -> Value {
        json!({ "sequences": [{ "id": "main", "items": [item] }] })
    }

    #[test]
    fn a_literal_argument_is_refused() {
        let prior = json!({ "PriorItemOutput": { "intra_sequence_item_index": 0, "path": [] } });
        let literal = json!({ "Direct": "Inline" });
        let tile = schema(json!({ "Tile": { "id": "f", "sources": [prior, literal] } }));
        assert_eq!(
            unpinned_bindings(&tile),
            ["main[0] f: argument 1 is an inline value"]
        );
        let recur = schema(json!({ "RecurTile": {
            "id": "sweep", "sources": [prior, prior, literal],
            "chunk": 64, "output": null, "state_is_output": true, "carries_state": true,
        } }));
        assert_eq!(
            unpinned_bindings(&recur),
            ["main[0] sweep: argument 2 is an inline value"]
        );
        let index = schema(json!({ "Tile": { "id": "g", "sources": [
            { "Indexed": { "value": prior, "indexes": [literal] } }
        ] } }));
        assert_eq!(
            unpinned_bindings(&index),
            ["main[0] g: argument 0 is an inline value"]
        );
    }

    #[test]
    fn a_selected_seed_is_refused_and_a_created_output_is_not() {
        let whole = json!({ "PriorItemOutput": { "intra_sequence_item_index": 1, "path": [] } });
        let selected = json!({ "PriorItemOutput": {
            "intra_sequence_item_index": 1, "path": [{ "Field": "carry" }, { "Field": "text" }],
        } });
        let created = json!({ "Direct": "Inline" });
        let output = json!({ "schema_hash": [], "empty_root": [], "derives": false });
        let item = |seed: &Value, output: &Value, slot: &Value| {
            schema(json!({ "RecurTile": {
                "id": "fold", "sources": [whole, seed, slot, whole],
                "chunk": null, "output": output, "state_is_output": false, "carries_state": true,
            } }))
        };
        assert!(unpinned_bindings(&item(&whole, &output, &created)).is_empty());
        let found = unpinned_bindings(&item(&selected, &output, &created));
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(found[0].starts_with("main[0] fold: the state seed is not the whole result"));
        assert_eq!(
            unpinned_bindings(&item(&created, &output, &created)).len(),
            1
        );
        let derived = json!({ "schema_hash": [], "empty_root": [], "derives": true });
        assert!(unpinned_bindings(&item(&whole, &derived, &whole)).is_empty());
        assert_eq!(
            unpinned_bindings(&item(&whole, &derived, &selected)).len(),
            1
        );
    }
}
