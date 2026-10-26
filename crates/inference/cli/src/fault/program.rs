//! Iris identity adapter following raster-cli program.rs; never writes Raster.lock.
//! Host-side assembly of a raster program's identity artifacts.
//!
//! Turns a project (source + optional `Raster.toml`) into a
//! [`ProgramDefinition`] (see `docs/proposals/program-identity.md`), writes the
//! canonical `program.bin` frame and the `Raster.lock` claim, and verifies a
//! reassembled definition against a previously written lock (drift check).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use raster_compiler::Project;
use raster_core::cfs::ControlFlowSchema;
use raster_core::input::ExternalEncoding;
use raster_core::program::{ImageId, InterfaceDecl, ProgramDefinition, ProgramManifest};
use raster_core::{Error, Result};
use raster_prover::replay::Replayer;
use serde::{Deserialize, Serialize};

/// `Raster.toml` document shape (authored program interface).
#[derive(Debug, Deserialize)]
struct RasterToml {
    program: ProgramSection,
    #[serde(default)]
    inputs: BTreeMap<String, TomlInterface>,
    #[serde(default)]
    output: Option<TomlInterface>,
}

#[derive(Debug, Deserialize)]
struct ProgramSection {
    name: String,
    #[serde(default = "default_version")]
    version: String,
}

fn default_version() -> String {
    "0.0.0".to_string()
}

#[derive(Debug, Deserialize)]
struct TomlInterface {
    #[serde(rename = "type")]
    type_path: String,
    #[serde(default)]
    encoding: ExternalEncoding,
}

impl From<TomlInterface> for InterfaceDecl {
    fn from(t: TomlInterface) -> Self {
        InterfaceDecl {
            type_path: t.type_path,
            encoding: t.encoding,
            schema_hash: [0u8; 32],
        }
    }
}

/// The derived lock: the identity claim tying `program_commitment` to a source
/// revision, plus the per-tile image ids and build toolchain. Cargo.lock
/// semantics — deterministic, checked in.
#[derive(Debug, Serialize, Deserialize)]
pub struct RasterLock {
    pub format: u32,
    pub program_commitment: String,
    pub tiles: BTreeMap<String, LockTile>,
    pub toolchain: LockToolchain,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LockTile {
    pub image_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_hash: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LockToolchain {
    pub risc0: String,
    pub build_mode: String,
}

/// Path of `Raster.toml` at the project root.
fn raster_toml_path(project: &Project) -> PathBuf {
    project.root_dir.join("Raster.toml")
}

/// Path of `Raster.lock` at the project root.
pub fn raster_lock_path(project: &Project) -> PathBuf {
    project.root_dir.join("Raster.lock")
}

/// Load `main`'s declared interface: parse `Raster.toml` if present, otherwise
/// synthesize it from the CFS + main's signature (types from the AST). Either
/// way the result is enforced against the CFS by [`ProgramDefinition::assemble`].
pub fn load_or_synthesize_manifest(
    project: &Project,
    cfs: &ControlFlowSchema,
) -> Result<ProgramManifest> {
    let path = raster_toml_path(project);
    if path.exists() {
        let text = std::fs::read_to_string(&path)
            .map_err(|e| Error::Other(format!("failed to read {}: {e}", path.display())))?;
        let doc: RasterToml = toml::from_str(&text)
            .map_err(|e| Error::Other(format!("failed to parse {}: {e}", path.display())))?;
        Ok(ProgramManifest {
            name: doc.program.name,
            version: doc.program.version,
            inputs: doc.inputs.into_iter().map(|(k, v)| (k, v.into())).collect(),
            output: doc.output.map(Into::into),
        })
    } else {
        Ok(synthesize_manifest(project, cfs))
    }
}

/// Build a manifest from the CFS + main's AST signature: real parameter/return
/// type paths, `raster` encoding. Used when no `Raster.toml` is authored.
fn synthesize_manifest(project: &Project, cfs: &ControlFlowSchema) -> ProgramManifest {
    let main_seq = cfs.sequences.iter().find(|s| s.id == "main");
    let entry_arguments = main_seq
        .map(|s| s.entry_arguments.clone())
        .unwrap_or_default();
    let produces_output = main_seq.map(|s| s.produces_output).unwrap_or(false);

    let main_ast = project.ast.functions.iter().find(|f| f.name == "main");
    let type_of = |name: &str| -> String {
        main_ast
            .and_then(|f| {
                f.input_names
                    .iter()
                    .position(|n| n == name)
                    .and_then(|i| f.inputs.get(i).cloned())
            })
            .unwrap_or_default()
    };

    let inputs = entry_arguments
        .into_iter()
        .map(|name| {
            let type_path = type_of(&name);
            (
                name,
                InterfaceDecl {
                    type_path,
                    encoding: ExternalEncoding::Raster,
                    schema_hash: [0u8; 32],
                },
            )
        })
        .collect();

    let output = produces_output.then(|| InterfaceDecl {
        type_path: main_ast.and_then(|f| f.output.clone()).unwrap_or_default(),
        encoding: ExternalEncoding::Raster,
        schema_hash: [0u8; 32],
    });

    ProgramManifest {
        name: cfs.project.clone(),
        version: "0.0.0".to_string(),
        inputs,
        output,
    }
}

/// Compile every CFS tile to its image id (the program's tile registry).
fn build_registry(
    cfs: &ControlFlowSchema,
    replayer: &Replayer,
) -> Result<BTreeMap<String, ImageId>> {
    let mut tiles = BTreeMap::new();
    for tile in &cfs.tiles {
        let image_id = replayer.tile_image_id(&tile.id)?;
        tiles.insert(tile.id.clone(), image_id);
    }
    Ok(tiles)
}

/// Assemble the full `ProgramDefinition` for a project: CFS + tile registry +
/// manifest, validated by [`ProgramDefinition::assemble`].
pub fn assemble_program(
    project: &Project,
    cfs: &ControlFlowSchema,
    replayer: &Replayer,
) -> Result<ProgramDefinition> {
    let mut manifest = load_or_synthesize_manifest(project, cfs)?;
    raster_compiler::schema_walk::fill_schema_hashes(&project.ast, &mut manifest)?;
    let tiles = build_registry(cfs, replayer)?;
    ProgramDefinition::assemble(manifest, cfs.clone(), tiles)
}

pub fn reassemble_and_verify(
    project: &Project,
    cfs: &ControlFlowSchema,
    replayer: &Replayer,
) -> Result<ProgramDefinition> {
    let program = assemble_program(project, cfs, replayer)?;
    check_lock_drift(&program, &raster_lock_path(project))?;
    Ok(program)
}

/// Read and parse `Raster.lock`. A missing lock is an error here (callers that
/// tolerate absence check existence first, e.g. [`check_lock_drift`]).
fn read_lock(lock_path: &Path) -> Result<RasterLock> {
    let text = std::fs::read_to_string(lock_path).map_err(|_| {
        Error::Other(format!(
            "{} not found — run `cargo raster build --backend risc0`",
            lock_path.display()
        ))
    })?;
    serde_json::from_str(&text)
        .map_err(|e| Error::Other(format!("failed to parse {}: {e}", lock_path.display())))
}

/// If `lock_path` exists, require the program's recomputed commitment to equal
/// the lock's (the stale-lock drift check). Absent lock is not an error.
/// The remedy named in these errors is `--backend risc0` on purpose: plain
/// `cargo raster build` defaults to the native backend, and
/// `commands::build` emits `program.bin` + `Raster.lock` **only** under risc0
/// (image ids need compiled guests). Advising the bare command sent a reader
/// in a circle — it reports success, changes nothing, and the same drift
/// error returns.
fn check_lock_drift(program: &ProgramDefinition, lock_path: &Path) -> Result<()> {
    if !lock_path.exists() {
        return Ok(());
    }
    let lock = read_lock(lock_path)?;
    let recomputed = hex::encode(program.commitment());
    if recomputed != lock.program_commitment {
        return Err(Error::Other(format!(
            "program identity drift: reassembled commitment {recomputed} != Raster.lock {} — run `cargo raster build --backend risc0`",
            lock.program_commitment
        )));
    }
    Ok(())
}
