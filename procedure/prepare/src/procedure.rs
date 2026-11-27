//! The procedure a specification describes: its kind's programs and its
//! models as their imports pin them, its parameters, and what it requires of
//! a run's source and result. Its id names it, with a blockchain or without
//! one.

use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use inference_manifest::ModelManifest;
use procedure_definition::pins::{
    ContractPins, RequiredOrigin, RequiredParameters, RequiredSource, CONTRACT_PINS_JSON,
};
use procedure_definition::registry::Procedure;
use procedure_definition::{read_json, read_specification, ProcedureKind, Specification};

fn absolute(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

/// A pins file, held to the procedure kind and the models it is used for.
fn load_pins(
    path: &Path,
    procedure_kind: ProcedureKind,
    models: &[&ModelManifest],
) -> Result<ContractPins> {
    ensure!(
        path.exists(),
        "{} does not exist: write it with `iwebs claim pins`, or import the model again from \
         the iwebs root, where the stage projects and their Raster.lock files are",
        path.display()
    );
    let pins: ContractPins = read_json(path).with_context(|| {
        format!(
            "{} is not a contract pins file of this build: write it again with `iwebs claim \
             pins`, or import the model again",
            path.display()
        )
    })?;
    if let Some(mismatch) = pins.mismatch(procedure_kind, models) {
        anyhow::bail!("{} are not this run's pins: {mismatch}", path.display());
    }
    Ok(pins)
}

/// The imports a specification names: each model's manifest path, the manifest
/// and the model's segment of a pipeline.
fn run_imports(
    specification_path: &Path,
    specification: &Specification,
) -> Result<Vec<(PathBuf, ModelManifest, String)>> {
    let specification_dir = specification_path
        .parent()
        .context("specification has no parent directory")?;
    specification
        .model_manifests()
        .into_iter()
        .map(|manifest| {
            let path = absolute(specification_dir, manifest);
            let manifest = ModelManifest::load_verified(&path)?;
            let segment = manifest.verified_raster_template(&path)?.1;
            Ok((path, manifest, segment))
        })
        .collect()
}

/// What a contract of a specification's procedure kind over its models pins, made from
/// the imports and the stage projects' locks as they are now.
pub fn run_pins(base_dir: &Path, specification_path: &Path) -> Result<ContractPins> {
    let specification_path = absolute(base_dir, specification_path);
    let specification = read_specification(&specification_path)?;
    let imports = run_imports(&specification_path, &specification)?;
    let imports: Vec<(&ModelManifest, &str)> = imports
        .iter()
        .map(|(_, manifest, segment)| (manifest, segment.as_str()))
        .collect();
    ContractPins::of(specification.procedure_kind, &imports)
}

/// What a specification requires of where a claim's prompt came from and of the
/// forms its output is read by. A contract holds these as it holds the
/// parameters: they are the terms the claim was asked for under.
fn required_origin(specification: &Specification) -> RequiredOrigin {
    // A relay's recipe is the specification's own text, so a procedure pins
    // it. The hash is over the recipe alone, so the relay step over no
    // output states it.
    let relay_recipe_sha256 = specification.relay.as_ref().map(|relay| {
        source_phase::relay::run_relay_steps(&relay.recipe(), "", &[], &[], true)
            .record
            .recipe_sha256
    });
    let Some(terms) = crate::source::source_terms(specification) else {
        return RequiredOrigin {
            source: None,
            forms: Vec::new(),
            relay_recipe_sha256,
        };
    };
    RequiredOrigin {
        source: Some(RequiredSource {
            signer: terms.signer,
            // The base prompt and the documents' domains are not pinned yet:
            // a procedure accepts any recipe, and the claim's statement
            // names the one it used.
            recipe_sha256: None,
        }),
        forms: terms.forms,
        relay_recipe_sha256,
    }
}

/// The procedure a specification describes. `Ok(None)` when no pins file is given
/// and a model's import has none: it was made where the stage projects are
/// not locked.
///
/// The pins are those of the file given. Without one, a contract of the
/// `single` procedure kind pins what the model's import wrote, and one of several
/// inferences what [`run_pins`] makes from the imports now.
pub fn procedure_of(
    specification_path: &Path,
    specification: &Specification,
    pins_path: Option<&Path>,
) -> Result<Option<Procedure>> {
    let procedure_kind = specification.procedure_kind;
    let imports = run_imports(specification_path, specification)?;
    let manifests: Vec<&ModelManifest> = imports.iter().map(|(_, manifest, _)| manifest).collect();
    let pins = match (pins_path, imports.as_slice()) {
        (Some(path), _) => load_pins(path, procedure_kind, &manifests)?,
        (None, imports) => {
            if imports
                .iter()
                .any(|(path, _, _)| !path.with_file_name(CONTRACT_PINS_JSON).exists())
            {
                return Ok(None);
            }
            match imports {
                [(path, _, _)] => load_pins(
                    &path.with_file_name(CONTRACT_PINS_JSON),
                    procedure_kind,
                    &manifests,
                )?,
                imports => {
                    let imports: Vec<(&ModelManifest, &str)> = imports
                        .iter()
                        .map(|(_, manifest, segment)| (manifest, segment.as_str()))
                        .collect();
                    ContractPins::of(procedure_kind, &imports)?
                }
            }
        }
    };
    Ok(Some(Procedure::of(
        pins,
        RequiredParameters {
            raw_prompt: specification.raw_prompt,
            max_new_tokens: specification.max_new_tokens,
        },
        required_origin(specification),
    )))
}

/// The procedure a specification describes, with the pins of this file or,
/// without one, as `iwebs claim check` finds them.
pub fn spec_procedure(specification_path: &Path, pins_path: Option<&Path>) -> Result<Procedure> {
    let specification = read_specification(specification_path)?;
    procedure_of(specification_path, &specification, pins_path)?.with_context(|| {
        format!(
            "a model {} names has no {CONTRACT_PINS_JSON}: import it again from the iwebs root, \
             where the stage projects and their Raster.lock files are",
            specification_path.display()
        )
    })
}
