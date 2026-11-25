//! The gate's source scenario: a prompt built from signed documents.
//!
//! One honest claim with a source is run three ways, as every case is: direct,
//! native stages, and the whole chain through Raster. Its recipe names the
//! output as one of its forms, so the result stage states that form's code.
//! Then sources that are
//! wrong are run through the `source` stage and the binds, in Raster and
//! natively: each must finish, give the same bytes both ways, and record the
//! fault it should.

use std::fs;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use model_manifest::ModelManifest;
use procedure_definition::{
    Case, CaseDocument, PathStep, PreparedSourceInputs, ProcedureKind, Specification,
    SpecificationDocument, SpecificationField, SpecificationForm, SpecificationModel,
    SpecificationSource, CASE_TOML, SPECIFICATION_TOML,
};
use run_prep::source::{LoadedDocument, LoadedSource};
use serde_json::{json, Value};
use source_core::{
    hex, sha256, sign_attestation, signer_key, Attestation, CIPHER_NONE, FAULT_DOCUMENT,
    FAULT_FIELD_MISSING, FAULT_FIELD_NOT_SCALAR, FAULT_NONE, FAULT_SERVER, FAULT_SIGNATURE,
    SCHEME_SECP256K1,
};
use staged_infer::chain_runner::StagedExecutionBackend;

use super::compare::{compare, read_boundaries};
use super::{case_beside, command, logged, prompt_only_manifest};
use blockchain_host::claim::{prepare_claim_run, ClaimBuildOptions};

const SERVER: &str = "example.org";
const EVENT_TARGET: &str = "/event/us1.json";
const COMMENTS_TARGET: &str = "/product/p1/comments.json";
/// The gate's signing key. It signs nothing outside a gate run.
const SECRET: [u8; 32] = [7; 32];
/// The code of the form the honest claim's output is.
const OUTPUT_CODE: u32 = 7;

fn attest(secret: &[u8], target: &str, body: &[u8]) -> Result<Attestation> {
    let mut attestation = Attestation {
        scheme: SCHEME_SECP256K1,
        cipher: CIPHER_NONE,
        time: 1_763_300_000,
        server: SERVER.to_string(),
        target: target.to_string(),
        records_sha256: hex(&sha256(body)),
        records_length: body.len() as u64,
        signature: String::new(),
    };
    sign_attestation(secret, &mut attestation).context("the gate's attestation was not signed")?;
    Ok(attestation)
}

fn key(steps: &[&str]) -> Vec<PathStep> {
    steps
        .iter()
        .map(|step| match step.parse::<u32>() {
            Ok(index) => PathStep::Index(index),
            Err(_) => PathStep::Key(step.to_string()),
        })
        .collect()
}

/// Two documents and a recipe that takes values from both, through an array
/// and a key with a dot in it. The event is padded so it spans several chunks.
pub(super) fn honest_source() -> Result<LoadedSource> {
    let event = serde_json::to_vec(&json!({
        "type": "Feature",
        "properties": {
            "mag": 5.4,
            "place": "10 km N of Lata",
            "products": {"losspager": [
                {"id": "p0"},
                {"id": "p1", "contents": {"comments.json": {"url": "https://example.org/c"}}},
            ]},
            "pad": "x".repeat(9000),
        },
        "id": "us1",
    }))?;
    let comments = serde_json::to_vec(&json!({"impact1": "Low risk.", "impact2": ""}))?;
    let field = |lead: &str, document: u32, path: &[&str]| SpecificationField {
        lead: lead.to_string(),
        document,
        path: key(path),
    };
    Ok(LoadedSource {
        specification: SpecificationSource {
            base: "Is mag 5 or more?\n".to_string(),
            tail: "\n".to_string(),
            signer: signer_key(&SECRET).context("the gate's key is no key")?,
            documents: vec![
                SpecificationDocument {
                    server: SERVER.to_string(),
                    target: EVENT_TARGET.to_string(),
                },
                SpecificationDocument {
                    server: SERVER.to_string(),
                    target: COMMENTS_TARGET.to_string(),
                },
            ],
            fields: vec![
                field("id: ", 0, &["id"]),
                field("\nmag: ", 0, &["properties", "mag"]),
                field(
                    "\nlink: ",
                    0,
                    &[
                        "properties",
                        "products",
                        "losspager",
                        "1",
                        "contents",
                        "comments.json",
                        "url",
                    ],
                ),
                field("\nnote: ", 1, &["impact1"]),
            ],
        },
        forms: vec![SpecificationForm {
            text: "yes".to_string(),
            code: 1,
        }],
        documents: vec![
            LoadedDocument {
                attestation: attest(&SECRET, EVENT_TARGET, &event)?,
                body: event,
            },
            LoadedDocument {
                attestation: attest(&SECRET, COMMENTS_TARGET, &comments)?,
                body: comments,
            },
        ],
    })
}

/// Writes a run of this source in `dir`: a specification of these models with
/// the source and its forms, and beside it the case, with each document and
/// its attestation as a file. Returns the specification's path.
pub(super) fn write_run(
    dir: &Path,
    source: &LoadedSource,
    procedure_kind: ProcedureKind,
    models: &[&Path],
) -> Result<std::path::PathBuf> {
    fs::create_dir_all(dir)?;
    let mut documents = Vec::new();
    for (index, document) in source.documents.iter().enumerate() {
        let file = std::path::PathBuf::from(format!("document_{index}.bin"));
        let attestation = std::path::PathBuf::from(format!("document_{index}.json"));
        fs::write(dir.join(&file), &document.body)?;
        procedure_definition::write_json(&dir.join(&attestation), &document.attestation)?;
        documents.push(CaseDocument { file, attestation });
    }
    let specification = Specification {
        procedure_kind,
        raw_prompt: false,
        max_new_tokens: 4,
        models: models
            .iter()
            .map(|model| SpecificationModel {
                manifest: model.to_path_buf(),
            })
            .collect(),
        source: Some(source.specification.clone()),
        forms: source.forms.clone(),
    };
    let path = dir.join(SPECIFICATION_TOML);
    fs::write(&path, toml::to_string(&specification)?)?;
    fs::write(
        dir.join(CASE_TOML),
        toml::to_string(&Case {
            documents,
            ..Case::default()
        })?,
    )?;
    Ok(path)
}

/// What the model answers the source's prompt with, from direct inference
/// alone. The forms are no part of the prompt, so naming this text as a form
/// afterwards leaves the answer as it is.
fn direct_output(
    root: &Path,
    dir: &Path,
    exe: &Path,
    model_path: &Path,
    source: &LoadedSource,
) -> Result<String> {
    let specification = write_run(dir, source, ProcedureKind::Single, &[model_path])?;
    let mut run = command(exe, root);
    run.args(["parity-worker", "--specification"])
        .arg(&specification)
        .arg("--case")
        .arg(case_beside(&specification))
        .arg("--output")
        .arg(dir.join("native-direct"))
        .arg("--direct");
    logged(run, &dir.join("native-direct.log")).context("direct inference")?;
    let result: Value = procedure_definition::read_json(&dir.join("native-direct/result.json"))?;
    Ok(result["generated_text"]
        .as_str()
        .context("direct inference recorded no text")?
        .to_string())
}

/// The honest claim, three ways.
fn run_honest(
    root: &Path,
    dir: &Path,
    exe: &Path,
    raster: &Path,
    model_path: &Path,
    source: &LoadedSource,
) -> Result<(Value, blockchain_host::claim::PreparedClaimRun)> {
    let specification = write_run(dir, source, ProcedureKind::Single, &[model_path])?;
    let prepared = prepare_claim_run(&ClaimBuildOptions {
        base_dir: dir.to_path_buf(),
        case_path: Some(case_beside(&specification)),
        specification_path: specification.clone(),
        current_exe: exe.to_path_buf(),
        staged_backend: StagedExecutionBackend::InProcess,
        claimed_token_ids: None,
        direct_output_ids: None,
        claimer: None,
    })?;
    let raster_dir = dir.join("raster");
    fs::create_dir(&raster_dir)?;
    let mut run = command(raster, root);
    run.args(["raster", "chain", "run"])
        .arg(&prepared.manifest_path)
        .args(["--no-auth", "--run"])
        .arg(&raster_dir);
    logged(run, &dir.join("raster.log")).context("Raster-staged execution")?;
    for (name, direct) in [("native-staged", false), ("native-direct", true)] {
        let mut run = command(exe, root);
        run.args(["parity-worker", "--specification"])
            .arg(&specification)
            .arg("--case")
            .arg(case_beside(&specification))
            .arg("--output")
            .arg(dir.join(name));
        if direct {
            run.arg("--direct");
        }
        logged(run, &dir.join(format!("{name}.log"))).with_context(|| name.to_string())?;
    }
    let staged: Value = procedure_definition::read_json(&dir.join("native-staged/result.json"))?;
    let direct: Value = procedure_definition::read_json(&dir.join("native-direct/result.json"))?;
    let native_dir = std::path::PathBuf::from(
        staged["chain_dir"]
            .as_str()
            .context("native-staged recorded no chain directory")?,
    );
    let order = procedure_definition::read_stage_order(&native_dir)?
        .context("native chain recorded no stage inventory")?;
    ensure!(
        order.first().map(String::as_str) == Some("source"),
        "a claim with a source does not start at the source stage"
    );
    let raster_outputs = read_boundaries(&raster_dir, &order, false).context("Raster-staged")?;
    let native = read_boundaries(&native_dir, &order, false).context("native-staged")?;
    for stage in &order {
        compare(stage, &raster_outputs[stage], &native[stage])
            .context("Raster-staged ↔ native-staged")?;
    }

    // Direct inference states the record the chain's last check does.
    let record = &staged["result"];
    ensure!(
        record == &direct,
        "direct inference and the claim state different records: {record} and {direct}"
    );
    let expected = run_prep::source_id(Some(&source.run().record));
    ensure!(
        record["source_sha256"] == json!(expected) && expected != "00".repeat(32),
        "the claim names source {}, the source step gives {expected}",
        record["source_sha256"]
    );
    ensure!(
        record["valid"] == json!(true),
        "an honest claim with a source is not valid: {record}"
    );
    // The result stage: last in the chain, the same bytes through Raster and
    // natively (compared above), and the code of the form the output is.
    let stated = &raster_outputs[order.last().expect("the order is not empty")].value;
    ensure!(
        order.last().map(String::as_str) == Some("result")
            && stated["code"] == json!(OUTPUT_CODE)
            && record["result_code"] == json!(OUTPUT_CODE)
            && stated["forms_sha256"] == record["result_forms_sha256"]
            && stated["statement"]["bytes"] == record["statement_sha256"],
        "the result stage does not state the output's form: {stated} for {record}"
    );
    ensure!(
        raster_outputs["source"].value["fault"] == json!(FAULT_NONE)
            && raster_outputs["source"].value["text_sha256"] == record["input_text_sha256"],
        "the source stage's record is not the claim's text: {}",
        raster_outputs["source"].value
    );

    // What a contract would check, and every native checkpoint against the
    // boundary the Raster run committed.
    let contract = blockchain_host::claim_check::check_claim(
        blockchain_host::claim_check::ClaimCheckOptions {
            base_dir: dir.to_path_buf(),
            specification_path: specification.clone(),
            submission_path: native_dir.join(blockchain_rules::checks::CLAIM_SUBMISSION_JSON),
            raster_run: Some(raster_dir),
            pins_path: None,
            registry_path: None,
            da_dir: None,
            submitted_at: None,
        },
    )
    .context("contract checks")?;
    for line in &contract.lines {
        ensure!(
            line.rejection.is_none(),
            "a contract would reject an honest claim with a source at `{}`: {}",
            line.check,
            line.rejection.as_deref().unwrap_or_default()
        );
    }

    // The challenger's side: read back from the layer into an empty
    // directory, the claim's source is the documents it posted, and the case
    // written over them gives the source record the claim names.
    let fetched =
        blockchain_host::posted::fetch_claim(blockchain_host::posted::FetchClaimOptions {
            submission_path: native_dir.join(blockchain_rules::checks::CLAIM_SUBMISSION_JSON),
            specification_path: specification.clone(),
            pins_path: None,
            da_dir: None,
            out_dir: dir.join("fetched"),
        })
        .context("reading the claim back from the layer")?;
    let fetched_source = run_prep::source::load_source(
        &procedure_definition::read_specification(&specification)?,
        fetched
            .case_path
            .parent()
            .context("the fetched case has no directory")?,
        &procedure_definition::read_case(&fetched.case_path)?,
    )?
    .context("the specification of a claim with a source names none")?;
    ensure!(
        fetched_source.documents.len() == source.documents.len()
            && fetched_source
                .documents
                .iter()
                .zip(&source.documents)
                .all(|(fetched, posted)| fetched.body == posted.body),
        "the documents read back from the layer are not the ones the claim was built from"
    );
    ensure!(
        run_prep::source_id(Some(&fetched_source.run().record)) == expected,
        "the source read back from the layer does not give the record the claim names"
    );
    Ok((
        json!({
            "checkpoints": order.len(),
            "source_sha256": expected,
            "statement_sha256": record["statement_sha256"],
        }),
        prepared,
    ))
}

/// One source that is wrong, or a text that is not its source's: the source
/// stage and the binds, through Raster and natively.
#[allow(clippy::too_many_arguments)]
fn run_fault(
    root: &Path,
    dir: &Path,
    exe: &Path,
    raster: &Path,
    template: &str,
    name: &str,
    inputs: &PreparedSourceInputs,
    text: &str,
    raw_prompt: bool,
    range_size: usize,
    tokenizer: &prompt_prep::PromptTokenizer,
    expected: (u32, u32),
) -> Result<Value> {
    let files = dir.join("prompt");
    let bound = prompt_prep::bound_ranges(text, raw_prompt, range_size, tokenizer)?;
    let texts = run_prep::write_bound_text(&bound.ranges, &files)?;
    let text_digest = run_prep::write_text_digest(&bound.text_digest, &files)?;
    let manifest = dir.join("Raster.toml");
    fs::write(
        &manifest,
        prompt_only_manifest(template, &texts, &[], Some(&text_digest), Some(inputs))?,
    )?;

    let raster_dir = dir.join("raster");
    fs::create_dir(&raster_dir)?;
    let mut run = command(raster, root);
    run.args(["raster", "chain", "run"])
        .arg(&manifest)
        .args(["--no-auth", "--run"])
        .arg(&raster_dir);
    logged(run, &dir.join("raster.log"))
        .with_context(|| format!("source fault `{name}`: a Raster stage did not finish"))?;
    let native_dir = staged_infer::CheckpointedInferenceExecutor
        .run(staged_infer::CheckpointedInferenceConfig {
            base_dir: dir.to_path_buf(),
            manifest_path: manifest,
            current_exe: exe.to_path_buf(),
            staged_backend: StagedExecutionBackend::InProcess,
            parity_policy: staged_infer::ParityPolicy::Skip,
        })
        .with_context(|| format!("source fault `{name}`: a native stage did not finish"))?
        .chain_dir;

    let order = procedure_definition::read_stage_order(&raster_dir)?
        .with_context(|| format!("source fault `{name}`: no stage inventory"))?;
    ensure!(
        order.first().map(String::as_str) == Some("source") && order.len() == texts.len() + 1,
        "source fault `{name}`: expected the source stage and {} binds, ran {order:?}",
        texts.len()
    );
    let raster_outputs = read_boundaries(&raster_dir, &order, false)?;
    let native = read_boundaries(&native_dir, &order, false)?;
    for stage in &order {
        compare(stage, &raster_outputs[stage], &native[stage])
            .with_context(|| format!("source fault `{name}`"))?;
    }
    let record = &raster_outputs["source"].value;
    let context = &raster_outputs[order.last().expect("the order is not empty")].value["context"];
    ensure!(
        record["fault"] == json!(expected.0) && context["fault"] == json!(expected.1),
        "source fault `{name}`: expected source fault {} and prompt fault {}; got {record} and {context}",
        expected.0,
        expected.1
    );
    // Whatever went wrong, the statement names the record the stage gave.
    ensure!(
        context["source_sha256"] != json!("00".repeat(32)),
        "source fault `{name}`: the bind names no source"
    );
    Ok(json!({
        "case": name,
        "stages": order.len(),
        "source_fault": record["fault"],
        "prompt_fault": context["fault"],
    }))
}

pub(super) fn run_source_scenario(
    root: &Path,
    run_dir: &Path,
    exe: &Path,
    raster: &Path,
    model_path: &Path,
    model: &ModelManifest,
) -> Result<Value> {
    use prompt_prep::{FAULT_SOURCE, FAULT_TEXT_SOURCE};
    let dir = run_dir.join("source");
    let mut honest = honest_source()?;
    // The answer is learned first and named as the second form, so the claim's
    // result stage passes one form by and stops at the next.
    let output = direct_output(root, &dir.join("answer"), exe, model_path, &honest)
        .context("the answer to the source's prompt")?;
    honest.forms.push(SpecificationForm {
        text: output.trim().to_string(),
        code: OUTPUT_CODE,
    });
    let (claim, prepared) = run_honest(root, &dir.join("honest"), exe, raster, model_path, &honest)
        .context("the honest claim with a source")?;

    // The honest run's own chain, cut down to the source stage and the binds.
    let template = fs::read_to_string(&prepared.manifest_path)?;
    let prompt = &prepared.prepared_run.first()?.prompt;
    let raw_prompt = prepared.prepared_run.raw_prompt;
    let range_size = model.range_size as usize;
    let tokenizer: Value = procedure_definition::read_json(
        &model_path
            .parent()
            .context("model manifest has no parent directory")?
            .join(&model.bundle.tokenizer_path),
    )?;
    let tokenizer = prompt_prep::prompt_tokenizer(&tokenizer)?;

    let other_secret = [9_u8; 32];
    let edited = |edit: &dyn Fn(&mut LoadedSource) -> Result<()>| -> Result<LoadedSource> {
        let mut source = honest.clone();
        edit(&mut source)?;
        Ok(source)
    };
    let own_text = prompt.resolved_prompt.clone();
    let cases: Vec<(&str, LoadedSource, String, (u32, u32))> = vec![
        (
            "untampered",
            honest.clone(),
            own_text.clone(),
            (FAULT_NONE, 0),
        ),
        (
            "altered-document",
            edited(&|source| {
                let body = String::from_utf8(source.documents[0].body.clone())?;
                source.documents[0].body = body.replace("5.4", "9.9").into_bytes();
                Ok(())
            })?,
            own_text.clone(),
            (FAULT_DOCUMENT, FAULT_SOURCE),
        ),
        (
            "other-signer",
            edited(&|source| {
                for document in &mut source.documents {
                    let target = document.attestation.target.clone();
                    document.attestation = attest(&other_secret, &target, &document.body)?;
                }
                Ok(())
            })?,
            own_text.clone(),
            (FAULT_SIGNATURE, FAULT_SOURCE),
        ),
        (
            "swapped-documents",
            edited(&|source| {
                source.documents.swap(0, 1);
                Ok(())
            })?,
            own_text.clone(),
            (FAULT_SERVER, FAULT_SOURCE),
        ),
        (
            "missing-document",
            edited(&|source| {
                source.documents.pop();
                Ok(())
            })?,
            own_text.clone(),
            (FAULT_SERVER, FAULT_SOURCE),
        ),
        (
            "missing-field",
            edited(&|source| {
                source.specification.fields[1].path = key(&["properties", "depth"]);
                Ok(())
            })?,
            own_text.clone(),
            (FAULT_FIELD_MISSING, FAULT_SOURCE),
        ),
        (
            "object-for-a-value",
            edited(&|source| {
                source.specification.fields[1].path = key(&["properties", "products"]);
                Ok(())
            })?,
            own_text.clone(),
            (FAULT_FIELD_NOT_SCALAR, FAULT_SOURCE),
        ),
        // The source is honest, and the model is given another text.
        (
            "other-text",
            honest.clone(),
            own_text.replace("5.4", "9.9"),
            (FAULT_NONE, FAULT_TEXT_SOURCE),
        ),
    ];
    let mut faults = Vec::new();
    for (name, source, text, expected) in cases {
        let case_dir = dir.join(name);
        fs::create_dir_all(&case_dir)?;
        ensure!(
            name == "untampered" || name == "other-text" || source.run().record.fault == expected.0,
            "source fault `{name}`: the source step gives fault {}, the case expects {}",
            source.run().record.fault,
            expected.0
        );
        let inputs = run_prep::write_source_inputs(&source, &case_dir.join("prompt"))?;
        faults.push(run_fault(
            root, &case_dir, exe, raster, &template, name, &inputs, &text, raw_prompt, range_size,
            &tokenizer, expected,
        )?);
    }
    Ok(json!({"claim": claim, "faults": faults}))
}
