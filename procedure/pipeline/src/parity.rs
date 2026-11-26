use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use raster_core::input::payload_structural_root;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::segments::{self, StageKind};

const PARITY_DIR: &str = "procedure-pipeline-parity";
const NATIVE_OUTPUT_BIN: &str = "procedure-pipeline-output.bin";
const NATIVE_OUTPUT_RINDEX: &str = "procedure-pipeline-output.rindex";
const REPORT_TXT: &str = "report.txt";
const REPORT_JSON: &str = "report.json";
pub const EXECUTION_TIMES_JSON: &str = "execution-times.json";

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ParityAuthority {
    NonAuthoritative,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RasterSourceMode {
    Unauthenticated,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParityReport {
    pub version: u32,
    pub stage: String,
    pub native_stage: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance: Option<usize>,
    pub authority: ParityAuthority,
    pub raster_source_mode: RasterSourceMode,
    pub matched: bool,
    pub input_load_duration_ns: u128,
    pub kernel_duration_ns: u128,
    pub encode_write_duration_ns: u128,
    pub native_stage_duration_ns: u128,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct PipelineExecutionTimes {
    pub version: u32,
    pub stages: Vec<StageExecutionTime>,
    pub total_exec_duration_ns: u128,
    #[serde(default)]
    pub total_wall_duration_ns: Option<u128>,
    #[serde(default)]
    pub aux_waves: Vec<AuxWaveExecutionTime>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct StageExecutionTime {
    pub name: String,
    pub exec_duration_ns: u128,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct AuxWaveExecutionTime {
    pub name: String,
    pub first_stage: String,
    pub last_stage: String,
    pub stage_count: usize,
    pub parallelism: usize,
    pub wall_duration_ns: u128,
    pub stage_duration_sum_ns: u128,
}

pub fn run_hidden_stage_compare(stage_dir: &Path) -> Result<ParityReport> {
    let stage_name = stage_name(stage_dir)?;
    let kind = StageKind::from_stage_dir(stage_dir)?;
    validate_stage_artifacts(stage_dir)?;

    let native_run = segments::run_for_compare(&kind)?;

    let parity_dir = stage_dir.join(PARITY_DIR);
    fs::create_dir_all(&parity_dir)
        .with_context(|| format!("failed to create {}", parity_dir.display()))?;
    let native_bin = parity_dir.join(NATIVE_OUTPUT_BIN);
    let native_rindex = parity_dir.join(NATIVE_OUTPUT_RINDEX);
    write_atomic(&native_bin, &native_run.encoded.data)?;
    write_atomic(&native_rindex, &native_run.encoded.index)?;

    let raster_bin = stage_dir.join("output.bin");
    let raster_bytes = fs::read(&raster_bin)
        .with_context(|| format!("failed to read {}", raster_bin.display()))?;
    let comparison = compare_bytes(&raster_bytes, &native_run.encoded.data);
    let raster_sha = sha256_hex(&raster_bytes);
    let native_sha = sha256_hex(&native_run.encoded.data);
    let raster_structural = structural_hex(&raster_bytes)?;
    let native_structural = native_run.encoded.structural_commitment;
    let result = ParityReport {
        version: 4,
        stage: stage_name,
        native_stage: kind.native_stage().to_string(),
        instance: kind.instance(),
        authority: ParityAuthority::NonAuthoritative,
        raster_source_mode: RasterSourceMode::Unauthenticated,
        matched: comparison.is_match(),
        input_load_duration_ns: native_run.timings.input_load_duration.as_nanos(),
        kernel_duration_ns: native_run.timings.kernel_duration.as_nanos(),
        encode_write_duration_ns: native_run.timings.encode_write_duration.as_nanos(),
        native_stage_duration_ns: native_run.timings.native_stage_duration.as_nanos(),
    };

    let report = render_report(
        &result.stage,
        &comparison,
        raster_bytes.len(),
        native_run.encoded.data.len(),
        &raster_sha,
        &native_sha,
        &raster_structural,
        &native_structural,
        &raster_bin,
        &native_bin,
        &result,
    );
    fs::write(parity_dir.join(REPORT_TXT), &report)
        .with_context(|| format!("failed to write {}", parity_dir.join(REPORT_TXT).display()))?;
    let report_json = serde_json::to_vec_pretty(&result).context("failed to encode report.json")?;
    write_atomic(&parity_dir.join(REPORT_JSON), &report_json)?;
    print!("{report}");

    Ok(result)
}

pub fn run_hidden_stage_native(stage_dir: &Path) -> Result<()> {
    let stage_name = stage_name(stage_dir)?;
    let kind = StageKind::from_stage_dir(stage_dir)?;
    validate_stage_input_artifacts(stage_dir)?;

    let native_run = segments::run_and_publish(&kind)?;
    fs::write(
        stage_dir.join("pipeline-timings.json"),
        serde_json::to_vec(&native_run.timings)?,
    )?;
    println!(
        "procedure-pipeline {stage_name}: output {} structural={} stage={}",
        native_run.artifact.data_path.display(),
        native_run.artifact.commitment,
        format_duration_ns(native_run.timings.native_stage_duration.as_nanos())
    );

    Ok(())
}

fn stage_name(stage_dir: &Path) -> Result<String> {
    stage_dir
        .file_name()
        .and_then(|name| name.to_str())
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("selected stage has no valid UTF-8 directory name"))
}

fn validate_stage_artifacts(stage_dir: &Path) -> Result<()> {
    validate_stage_input_artifacts(stage_dir)?;
    for name in ["output.bin", "output.rindex", "output_manifest.json"] {
        let path = stage_dir.join(name);
        if !path.is_file() {
            bail!(
                "selected stage is missing required artifact {}",
                path.display()
            );
        }
    }
    Ok(())
}

fn validate_stage_input_artifacts(stage_dir: &Path) -> Result<()> {
    StageKind::from_stage_dir(stage_dir)?;
    for name in ["input.json", "input_manifest.json"] {
        let path = stage_dir.join(name);
        if !path.is_file() {
            bail!(
                "selected stage is missing required artifact {}",
                path.display()
            );
        }
    }
    let run_dir = stage_dir
        .parent()
        .ok_or_else(|| anyhow::anyhow!("selected stage has no parent run directory"))?;
    let canonical_run = run_dir
        .canonicalize()
        .with_context(|| format!("failed to canonicalize {}", run_dir.display()))?;
    let expected_roots = allowed_no_auth_roots()?;
    if !expected_roots
        .iter()
        .any(|expected_root| canonical_run.starts_with(expected_root))
    {
        bail!(
            "selected stage run {} is not under an unauthenticated pipeline root ({})",
            canonical_run.display(),
            expected_roots
                .iter()
                .map(|root| root.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    let authenticated_artifacts = [
        run_dir.join("pipeline-commitment"),
        stage_dir.join("trace.bin"),
        stage_dir.join("commit.bin"),
    ];
    if let Some(path) = authenticated_artifacts.iter().find(|path| path.exists()) {
        bail!(
            "selected no-auth stage contains authenticated artifacts: {}",
            path.display()
        );
    }
    Ok(())
}

fn allowed_no_auth_roots() -> Result<Vec<PathBuf>> {
    let current_dir = std::env::current_dir().context("failed to read current directory")?;
    let roots = [
        current_dir
            .join("target")
            .join("raster")
            .join("pipelines-no-auth"),
        current_dir
            .join("target")
            .join("procedure-pipeline")
            .join("runs"),
    ];
    Ok(roots
        .into_iter()
        .filter_map(|root| root.canonicalize().ok())
        .collect())
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension(format!(
        "{}.tmp-{}",
        path.extension().and_then(|s| s.to_str()).unwrap_or("out"),
        std::process::id()
    ));
    fs::write(&tmp, bytes).with_context(|| format!("failed to write {}", tmp.display()))?;
    fs::rename(&tmp, path)
        .with_context(|| format!("failed to rename {} to {}", tmp.display(), path.display()))?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ByteComparison {
    Match,
    Mismatch {
        offset: usize,
        raster: Option<u8>,
        native_run: Option<u8>,
    },
}

impl ByteComparison {
    fn is_match(&self) -> bool {
        matches!(self, Self::Match)
    }
}

fn compare_bytes(raster: &[u8], native_run: &[u8]) -> ByteComparison {
    for (idx, (left, right)) in raster.iter().zip(native_run.iter()).enumerate() {
        if left != right {
            return ByteComparison::Mismatch {
                offset: idx,
                raster: Some(*left),
                native_run: Some(*right),
            };
        }
    }
    if raster.len() == native_run.len() {
        ByteComparison::Match
    } else {
        let offset = raster.len().min(native_run.len());
        ByteComparison::Mismatch {
            offset,
            raster: raster.get(offset).copied(),
            native_run: native_run.get(offset).copied(),
        }
    }
}

fn render_report(
    stage: &str,
    comparison: &ByteComparison,
    raster_len: usize,
    native_len: usize,
    raster_sha: &str,
    native_sha: &str,
    raster_structural: &str,
    native_structural: &str,
    raster_artifact: &Path,
    native_artifact: &Path,
    result: &ParityReport,
) -> String {
    match comparison {
        ByteComparison::Match => format!(
            "NON-AUTHORITATIVE procedure-pipeline computational parity {stage}: MATCH\n\
             Raster source mode: unauthenticated (--no-auth)\n\
             output.bin bytes: {raster_len}\n\
             raster sha256: {raster_sha}\n\
             procedure-pipeline sha256: {native_sha}\n\
             structural commitment: {raster_structural}\n\
             procedure-pipeline stage: {}\n\
             procedure-pipeline kernel: {}\n",
            format_duration_ns(result.native_stage_duration_ns),
            format_duration_ns(result.kernel_duration_ns),
        ),
        ByteComparison::Mismatch {
            offset,
            raster,
            native_run,
        } => format!(
            "NON-AUTHORITATIVE procedure-pipeline computational parity {stage}: MISMATCH\n\
             Raster source mode: unauthenticated (--no-auth)\n\
             raster bytes: {raster_len}\n\
             procedure-pipeline bytes: {native_len}\n\
             first differing byte offset: {offset}\n\
             raster byte: {}\n\
             procedure-pipeline byte: {}\n\
             raster sha256: {raster_sha}\n\
             procedure-pipeline sha256: {native_sha}\n\
             raster structural commitment: {raster_structural}\n\
             procedure-pipeline structural commitment: {native_structural}\n\
             raster artifact: {}\n\
             procedure-pipeline artifact: {}\n\
             procedure-pipeline stage: {}\n\
             procedure-pipeline kernel: {}\n",
            render_byte(*raster),
            render_byte(*native_run),
            raster_artifact.display(),
            native_artifact.display(),
            format_duration_ns(result.native_stage_duration_ns),
            format_duration_ns(result.kernel_duration_ns),
        ),
    }
}

fn render_byte(value: Option<u8>) -> String {
    value
        .map(|byte| format!("0x{byte:02x}"))
        .unwrap_or_else(|| String::from("EOF"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn structural_hex(bytes: &[u8]) -> Result<String> {
    payload_structural_root(bytes)
        .map(hex::encode)
        .ok_or_else(|| anyhow::anyhow!("artifact is not a well-formed Raster payload"))
}

pub fn read_parity_report(stage_dir: &Path) -> Result<ParityReport> {
    let path = parity_dir(stage_dir).join(REPORT_JSON);
    let report: ParityReport = serde_json::from_slice(
        &fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to decode {}", path.display()))?;
    if report.version != 4 {
        bail!(
            "unsupported procedure-pipeline report version {} in {}",
            report.version,
            path.display()
        );
    }
    Ok(report)
}

pub fn read_pipeline_execution_times(pipeline_dir: &Path) -> Result<PipelineExecutionTimes> {
    let path = pipeline_dir.join(EXECUTION_TIMES_JSON);
    let timings: PipelineExecutionTimes = serde_json::from_slice(
        &fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to decode {}", path.display()))?;
    if !matches!(timings.version, 1 | 2) {
        bail!(
            "unsupported pipeline execution-times version {} in {}",
            timings.version,
            path.display()
        );
    }
    Ok(timings)
}

pub fn render_timing_summary(
    timings: &PipelineExecutionTimes,
    parity: &ParityReport,
) -> Result<String> {
    let stage_width = timings
        .stages
        .iter()
        .map(|timing| timing.name.len())
        .chain(std::iter::once("stage".len()))
        .max()
        .unwrap_or_default();
    let mut output = String::new();
    writeln!(
        output,
        "\nNON-AUTHORITATIVE timing comparison (pipeline runner: --no-auth; stage execution only)"
    )?;
    writeln!(
        output,
        "{:<stage_width$}  {:>12}  {:>12}  {:>12}  {:>12}  {:>8}  {:>8}",
        "stage", "pipeline exec", "native stage", "kernel", "saved", "speedup", "parity"
    )?;

    let mut selected_count = 0usize;
    for timing in &timings.stages {
        if timing.name == parity.stage {
            selected_count += 1;
            let speedup = if parity.native_stage_duration_ns == 0 {
                String::from("—")
            } else {
                format!(
                    "{:.2}x",
                    timing.exec_duration_ns as f64 / parity.native_stage_duration_ns as f64
                )
            };
            writeln!(
                output,
                "{:<stage_width$}  {:>12}  {:>12}  {:>12}  {:>12}  {:>8}  {:>8}",
                timing.name,
                format_duration_ns(timing.exec_duration_ns),
                format_duration_ns(parity.native_stage_duration_ns),
                format_duration_ns(parity.kernel_duration_ns),
                format_saved(timing.exec_duration_ns, parity.native_stage_duration_ns),
                speedup,
                if parity.matched { "MATCH" } else { "MISMATCH" },
            )?;
        } else {
            writeln!(
                output,
                "{:<stage_width$}  {:>12}  {:>12}  {:>12}  {:>12}  {:>8}  {:>8}",
                timing.name,
                format_duration_ns(timing.exec_duration_ns),
                "—",
                "—",
                "—",
                "—",
                "—",
            )?;
        }
    }
    if selected_count != 1 {
        bail!(
            "expected exactly one timing for `{}`, found {selected_count}",
            parity.stage
        );
    }
    writeln!(
        output,
        "{:<stage_width$}  {:>12}",
        "stage sum",
        format_duration_ns(timings.total_exec_duration_ns)
    )?;
    if let Some(wall_duration_ns) = timings.total_wall_duration_ns {
        writeln!(
            output,
            "{:<stage_width$}  {:>12}",
            "wall time",
            format_duration_ns(wall_duration_ns)
        )?;
    }
    for wave in &timings.aux_waves {
        writeln!(
            output,
            "aux wave {} ({}..{}): wall={} stage-sum={} parallelism={} effective={}",
            wave.name,
            wave.first_stage,
            wave.last_stage,
            format_duration_ns(wave.wall_duration_ns),
            format_duration_ns(wave.stage_duration_sum_ns),
            wave.parallelism,
            format_speedup(wave.stage_duration_sum_ns, wave.wall_duration_ns),
        )?;
    }
    writeln!(
        output,
        "procedure-pipeline parity overhead: {}",
        format_duration_ns(parity.native_stage_duration_ns)
    )?;
    Ok(output)
}

fn format_saved(raster_ns: u128, native_ns: u128) -> String {
    if raster_ns >= native_ns {
        format_duration_ns(raster_ns - native_ns)
    } else {
        format!("-{}", format_duration_ns(native_ns - raster_ns))
    }
}

fn format_speedup(numerator_ns: u128, denominator_ns: u128) -> String {
    if denominator_ns == 0 {
        String::from("—")
    } else {
        format!("{:.2}x", numerator_ns as f64 / denominator_ns as f64)
    }
}

fn format_duration_ns(ns: u128) -> String {
    if ns < 1_000 {
        format!("{ns}ns")
    } else if ns < 1_000_000 {
        format!("{:.2}µs", ns as f64 / 1_000.0)
    } else if ns < 1_000_000_000 {
        format!("{:.2}ms", ns as f64 / 1_000_000.0)
    } else {
        format!("{:.2}s", ns as f64 / 1_000_000_000.0)
    }
}

pub fn parity_dir(stage_dir: &Path) -> PathBuf {
    stage_dir.join(PARITY_DIR)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    static NEXT_TEST_RUN: AtomicUsize = AtomicUsize::new(0);

    struct TestStage {
        run_dir: PathBuf,
        stage_dir: PathBuf,
    }

    impl TestStage {
        fn no_auth(stage_name: &str) -> Self {
            let id = NEXT_TEST_RUN.fetch_add(1, Ordering::Relaxed);
            let run_dir = std::env::current_dir()
                .unwrap()
                .join("target")
                .join("raster")
                .join("pipelines-no-auth")
                .join(format!("parity-test-{}-{id}", std::process::id()));
            let stage_dir = run_dir.join(stage_name);
            fs::create_dir_all(&stage_dir).unwrap();
            for name in [
                "input.json",
                "input_manifest.json",
                "output.bin",
                "output.rindex",
                "output_manifest.json",
            ] {
                fs::write(stage_dir.join(name), []).unwrap();
            }
            Self { run_dir, stage_dir }
        }
    }

    impl Drop for TestStage {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.run_dir).unwrap();
        }
    }

    #[test]
    fn accepts_complete_no_auth_stage_without_pipeline_commitment() {
        let stage = TestStage::no_auth("range_p0_l3");

        validate_stage_artifacts(&stage.stage_dir).unwrap();
    }

    #[test]
    fn rejects_authenticated_artifacts_in_no_auth_stage() {
        let stage = TestStage::no_auth("range_p0_l3");
        fs::write(stage.run_dir.join("pipeline-commitment"), []).unwrap();
        fs::write(stage.stage_dir.join("trace.bin"), []).unwrap();
        fs::write(stage.stage_dir.join("commit.bin"), []).unwrap();

        let error = validate_stage_artifacts(&stage.stage_dir).unwrap_err();
        assert!(error.to_string().contains("authenticated artifacts"));
    }

    #[test]
    fn finds_same_length_difference() {
        assert_eq!(
            compare_bytes(b"abc", b"axc"),
            ByteComparison::Mismatch {
                offset: 1,
                raster: Some(b'b'),
                native_run: Some(b'x')
            }
        );
    }

    #[test]
    fn finds_length_difference_as_eof() {
        assert_eq!(
            compare_bytes(b"abc", b"ab"),
            ByteComparison::Mismatch {
                offset: 2,
                raster: Some(b'c'),
                native_run: None
            }
        );
    }
}
