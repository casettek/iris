//! `iris attest` and `iris source`: the host signer, and a run's source read
//! without running the model.
//!
//! The signer stands in for a notary. It signs whatever hash it is given, so
//! an attestation it makes is worth exactly the trust placed in whoever ran it.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, ensure, Context, Result};
use clap::{Args, Subcommand};
use run_prep::source::{fault_description, load_source, LoadedSource, SourceRecord, FAULT_NONE};
use source_core::{
    hex, sha256, sign_attestation, signer_key, unhex, verify_attestation, Attestation, CIPHER_NONE,
    SCHEME_SECP256K1,
};

pub const SIGNER_SECRET_FILE: &str = "signer.key";
pub const SIGNER_PUBLIC_FILE: &str = "signer.json";

#[derive(Debug, Subcommand)]
pub enum AttestCommand {
    /// Make a signing key: `signer.key` (secret) and `signer.json` (public).
    Keygen {
        /// Directory to write the two files in. An existing key is kept.
        #[arg(long)]
        out: PathBuf,
    },
    /// Sign that a server answered a request with a document.
    Sign(SignArgs),
}

#[derive(Debug, Args)]
pub struct SignArgs {
    /// The secret key file `keygen` wrote.
    #[arg(long)]
    key: PathBuf,
    /// The host name the document came from.
    #[arg(long)]
    server: String,
    /// The request path and query.
    #[arg(long)]
    target: String,
    /// The response body, exactly as received.
    #[arg(long)]
    document: PathBuf,
    /// Where to write the attestation.
    #[arg(long)]
    out: PathBuf,
    /// Unix seconds to state. The clock's, when not given.
    #[arg(long)]
    time: Option<u64>,
}

#[derive(Debug, Subcommand)]
pub enum SourceCommand {
    /// Check a run's source and print its record. Exits with 1 on a fault.
    Render {
        #[arg(long, default_value = inference_artifacts::SPECIFICATION_TOML)]
        specification: PathBuf,
        #[arg(long, default_value = inference_artifacts::CASE_TOML)]
        case: PathBuf,
        /// Write the prompt text the source gives to this file.
        #[arg(long)]
        text_out: Option<PathBuf>,
    },
    /// Write the `prompt-source` stage's inputs for a run, with the
    /// `input.json` and `input_manifest.json` that run the stage on its own.
    Inputs {
        #[arg(long, default_value = inference_artifacts::SPECIFICATION_TOML)]
        specification: PathBuf,
        #[arg(long, default_value = inference_artifacts::CASE_TOML)]
        case: PathBuf,
        /// Directory to write the files in.
        #[arg(long)]
        out: PathBuf,
    },
}

fn read_secret(path: &Path) -> Result<Vec<u8>> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read signing key {}", path.display()))?;
    unhex(text.trim()).with_context(|| format!("{} is not a signing key", path.display()))
}

pub fn keygen(out: &Path) -> Result<()> {
    fs::create_dir_all(out).with_context(|| format!("failed to create {}", out.display()))?;
    let secret_path = out.join(SIGNER_SECRET_FILE);
    let public_path = out.join(SIGNER_PUBLIC_FILE);
    if !secret_path.exists() {
        let mut random = fs::File::open("/dev/urandom").context("failed to open /dev/urandom")?;
        let mut secret = [0_u8; 32];
        // Nearly every 32 bytes are a key. Draw again for the few that are not.
        loop {
            random.read_exact(&mut secret)?;
            if signer_key(&secret).is_some() {
                break;
            }
        }
        let mut options = fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options
            .open(&secret_path)
            .with_context(|| format!("failed to create {}", secret_path.display()))?;
        writeln!(file, "{}", hex(&secret))?;
    }
    let public = signer_key(&read_secret(&secret_path)?)
        .with_context(|| format!("{} is not a signing key", secret_path.display()))?;
    fs::write(&public_path, serde_json::to_string_pretty(&public)? + "\n")
        .with_context(|| format!("failed to write {}", public_path.display()))?;
    println!("signing key: {}", secret_path.display());
    println!("signer: {}", public_path.display());
    println!("signer key: {}", public.key);
    Ok(())
}

pub fn sign(args: SignArgs) -> Result<()> {
    let secret = read_secret(&args.key)?;
    let document = fs::read(&args.document)
        .with_context(|| format!("failed to read document {}", args.document.display()))?;
    let time = match args.time {
        Some(time) => time,
        None => SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("the clock is before 1970")?
            .as_secs(),
    };
    let mut attestation = Attestation {
        scheme: SCHEME_SECP256K1,
        cipher: CIPHER_NONE,
        time,
        server: args.server,
        target: args.target,
        records_sha256: hex(&sha256(&document)),
        records_length: document.len() as u64,
        signature: String::new(),
    };
    sign_attestation(&secret, &mut attestation)
        .context("the attestation cannot be signed: check the key, and that the server and target are under 65,536 bytes")?;
    let signer = signer_key(&secret).context("the signing key is not a key")?;
    ensure!(
        verify_attestation(&signer, &attestation) == FAULT_NONE,
        "the attestation just signed does not verify"
    );
    fs::write(
        &args.out,
        serde_json::to_string_pretty(&attestation)? + "\n",
    )
    .with_context(|| format!("failed to write {}", args.out.display()))?;
    println!("attestation: {}", args.out.display());
    println!("document sha256: {}", attestation.records_sha256);
    println!("signer key: {}", signer.key);
    Ok(())
}

/// A run's source, when its specification has one, over the case's documents.
pub fn run_source_of(specification_path: &Path, case_path: &Path) -> Result<Option<LoadedSource>> {
    let base_dir = std::env::current_dir().context("failed to read current directory")?;
    let specification_path = crate::absolute(&base_dir, specification_path);
    let specification = inference_artifacts::read_specification(&specification_path)?;
    let case_path = crate::absolute(&base_dir, case_path);
    let case = inference_artifacts::read_case(&case_path)?;
    let case_dir = case_path.parent().context("case has no parent directory")?;
    load_source(&specification, case_dir, &case)
}

pub fn render(specification: &Path, case: &Path, text_out: Option<&Path>) -> Result<SourceRecord> {
    let Some(source) = run_source_of(specification, case)? else {
        bail!(
            "specification {} has no `[source]`",
            specification.display()
        );
    };
    let run = source.run();
    if let Some(path) = text_out {
        fs::write(path, &run.text)
            .with_context(|| format!("failed to write {}", path.display()))?;
    }
    println!("{}", serde_json::to_string_pretty(&run.record)?);
    if run.record.fault != FAULT_NONE {
        eprintln!(
            "iris: source fault {} at {}: {}",
            run.record.fault,
            run.record.fault_position,
            fault_description(run.record.fault)
        );
    }
    Ok(run.record)
}

pub fn stage_inputs(specification: &Path, case: &Path, out: &Path) -> Result<()> {
    let Some(source) = run_source_of(specification, case)? else {
        bail!(
            "specification {} has no `[source]`",
            specification.display()
        );
    };
    fs::create_dir_all(out).with_context(|| format!("failed to create {}", out.display()))?;
    let out = fs::canonicalize(out)?;
    let files = run_prep::write_source_inputs(&source, &out)?;
    let mut input = serde_json::Map::new();
    let mut manifest = serde_json::Map::new();
    for (name, file) in [
        ("signer", &files.signer),
        ("recipe", &files.recipe),
        ("documents", &files.documents),
    ] {
        input.insert(
            name.to_string(),
            serde_json::json!({
                "path": file.path,
                "index_path": file.index_path,
                "load_preference": "mmap",
            }),
        );
        manifest.insert(
            name.to_string(),
            serde_json::json!({
                "commitment": file.commitment,
                "encoding": "raster",
                "type": "sha256",
            }),
        );
    }
    for (name, value) in [("input.json", input), ("input_manifest.json", manifest)] {
        let path = out.join(name);
        fs::write(&path, serde_json::to_string_pretty(&value)? + "\n")
            .with_context(|| format!("failed to write {}", path.display()))?;
        println!("{}", path.display());
    }
    Ok(())
}

/// The lines `iris infer` prints for a run with a source, before the result.
pub fn print_source_lines(source: &LoadedSource) {
    let record = source.run().record;
    for (index, document) in source.documents.iter().enumerate() {
        let attestation = &document.attestation;
        println!(
            "source document {index}: https://{}{} sha256 {} time {}",
            attestation.server, attestation.target, attestation.records_sha256, attestation.time
        );
    }
    println!("source signer: {}", record.signer);
    println!("source documents sha256: {}", record.documents_sha256);
    println!("source recipe sha256: {}", record.recipe_sha256);
    println!("source fault: {}", record.fault);
}

#[cfg(test)]
mod tests {
    use super::*;
    use inference_artifacts::{Case, CaseDocument, Specification};
    use run_prep::source::sourced_prompt;

    const EVENT: &str = r#"{"type":"Feature","properties":{"mag":5.1,"products":{"losspager":[{"id":"p1","contents":{"comments.json":{"url":"https://example.org/p1/comments.json"}}}]}},"id":"us7000abcd"}"#;
    const COMMENTS: &str = r#"{"impact1": "Green alert.", "impact2": ""}"#;
    /// A specification with a source. `KEY` stands for the signer's key.
    const SPECIFICATION: &str = r#"
[[model]]
manifest = "unused.json"

[source]
base = "Are losses expected to be minimal?\n\n"
tail = "\n"

[source.signer]
scheme = 1
key = "KEY"

[[source.document]]
server = "example.org"
target = "/event/us7000abcd.json"

[[source.document]]
server = "example.org"
target = "/p1/comments.json"

[[source.field]]
lead = "event_id: "
path = ["id"]

[[source.field]]
lead = "\nmagnitude: "
path = ["properties", "mag"]

[[source.field]]
lead = "\ncomments: "
path = ["properties", "products", "losspager", 0, "contents", "comments.json", "url"]

[[source.field]]
lead = "\nimpact1: "
document = 1
path = ["impact1"]

[[form]]
text = "{\"verdict\":\"confirmed\"}"
code = 1

[[form]]
text = "{\"verdict\":\"denied\"}"
code = 2
"#;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("iris-source-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The public key `keygen` wrote in a directory.
    fn public_key(dir: &Path) -> source_core::SignerKey {
        serde_json::from_str(&fs::read_to_string(dir.join(SIGNER_PUBLIC_FILE)).unwrap()).unwrap()
    }

    /// A specification whose source names this directory's signer, and the
    /// case of two documents that signer attests.
    fn signed_run(dir: &Path) -> (Specification, Case) {
        keygen(dir).unwrap();
        let mut documents = Vec::new();
        for (name, target, body) in [
            ("event", "/event/us7000abcd.json", EVENT),
            ("comments", "/p1/comments.json", COMMENTS),
        ] {
            fs::write(dir.join(format!("{name}.bin")), body).unwrap();
            sign(SignArgs {
                key: dir.join(SIGNER_SECRET_FILE),
                server: "example.org".to_string(),
                target: target.to_string(),
                document: dir.join(format!("{name}.bin")),
                out: dir.join(format!("{name}.json")),
                time: Some(1_763_300_000),
            })
            .unwrap();
            documents.push(CaseDocument {
                file: PathBuf::from(format!("{name}.bin")),
                attestation: PathBuf::from(format!("{name}.json")),
            });
        }
        let specification: Specification =
            toml::from_str(&SPECIFICATION.replace("KEY", &public_key(dir).key)).unwrap();
        specification.validate().unwrap();
        (
            specification,
            Case {
                documents,
                ..Case::default()
            },
        )
    }

    fn load(dir: &Path, specification: &Specification, case: &Case) -> LoadedSource {
        load_source(specification, dir, case).unwrap().unwrap()
    }

    #[test]
    fn signed_documents_give_the_recipes_prompt() {
        let dir = scratch("good");
        let (specification, case) = signed_run(&dir);
        let source = load(&dir, &specification, &case);
        let text = sourced_prompt(&source).unwrap();
        assert_eq!(
            text,
            "Are losses expected to be minimal?\n\nevent_id: \"us7000abcd\"\nmagnitude: 5.1\ncomments: \"https://example.org/p1/comments.json\"\nimpact1: \"Green alert.\"\n"
        );
        let record = source.run().record;
        assert_eq!(record.fault, FAULT_NONE);
        assert_eq!(record.documents, 2);
        assert_eq!(record.text_sha256, hex(&sha256(text.as_bytes())));
        assert_eq!(source.result_code("{\"verdict\":\"denied\"}\n"), 2);
        assert_eq!(source.result_code("maybe"), 0);

        // A second keygen keeps the key, so the attestations still verify.
        keygen(&dir).unwrap();
        assert!(sourced_prompt(&load(&dir, &specification, &case)).is_ok());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_changed_document_or_another_signer_gives_no_prompt() {
        let dir = scratch("bad");
        let (mut specification, case) = signed_run(&dir);
        let prompt = |specification: &Specification, case: &Case| {
            format!(
                "{:#}",
                sourced_prompt(&load(&dir, specification, case)).unwrap_err()
            )
        };
        fs::write(dir.join("comments.bin"), COMMENTS.replace("Green", "Red")).unwrap();
        let error = prompt(&specification, &case);
        assert!(error.contains("fault 3 at 0"), "{error}");
        fs::write(dir.join("comments.bin"), COMMENTS).unwrap();

        // A case of fewer documents than the source names is the source
        // step's fault to state, and no refusal before it.
        let mut fewer = case.clone();
        fewer.documents.pop();
        let error = prompt(&specification, &fewer);
        assert!(error.contains("fault 4"), "{error}");

        let other = scratch("bad-other-signer");
        keygen(&other).unwrap();
        specification.source.as_mut().unwrap().signer = public_key(&other);
        let error = prompt(&specification, &case);
        assert!(error.contains("fault 2 at 0"), "{error}");
        fs::remove_dir_all(dir).unwrap();
        fs::remove_dir_all(other).unwrap();
    }
}
