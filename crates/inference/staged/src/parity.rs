use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use raster_core::input::payload_structural_root;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::routines::{self, StageKind};

const PARITY_DIR: &str = "staged-infer-parity";
const DIRECT_OUTPUT_BIN: &str = "staged-infer-output.bin";
const DIRECT_OUTPUT_RINDEX: &str = "staged-infer-output.rindex";
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
    pub routine: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance: Option<usize>,
    pub authority: ParityAuthority,
    pub raster_source_mode: RasterSourceMode,
    pub matched: bool,
    pub input_load_duration_ns: u128,
    pub kernel_duration_ns: u128,
    pub encode_write_duration_ns: u128,
    pub direct_stage_duration_ns: u128,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct ChainExecutionTimes {
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
    let kind = StageKind::from_stage_name(&stage_name)?;
    validate_stage_artifacts(stage_dir)?;

    let direct = routines::run_for_compare(&kind)?;

    let parity_dir = stage_dir.join(PARITY_DIR);
    fs::create_dir_all(&parity_dir)
        .with_context(|| format!("failed to create {}", parity_dir.display()))?;
    let direct_bin = parity_dir.join(DIRECT_OUTPUT_BIN);
    let direct_rindex = parity_dir.join(DIRECT_OUTPUT_RINDEX);
    write_atomic(&direct_bin, &direct.encoded.data)?;
    write_atomic(&direct_rindex, &direct.encoded.index)?;

    let raster_bin = stage_dir.join("output.bin");
    let raster_bytes = fs::read(&raster_bin)
        .with_context(|| format!("failed to read {}", raster_bin.display()))?;
    let comparison = compare_bytes(&raster_bytes, &direct.encoded.data);
    let raster_sha = sha256_hex(&raster_bytes);
    let direct_sha = sha256_hex(&direct.encoded.data);
    let raster_structural = structural_hex(&raster_bytes)?;
    let direct_structural = direct.encoded.structural_commitment;
    let result = ParityReport {
        version: 3,
        stage: stage_name,
        routine: kind.routine().to_string(),
        instance: kind.instance(),
        authority: ParityAuthority::NonAuthoritative,
        raster_source_mode: RasterSourceMode::Unauthenticated,
        matched: comparison.is_match(),
        input_load_duration_ns: direct.timings.input_load_duration.as_nanos(),
        kernel_duration_ns: direct.timings.kernel_duration.as_nanos(),
        encode_write_duration_ns: direct.timings.encode_write_duration.as_nanos(),
        direct_stage_duration_ns: direct.timings.direct_stage_duration.as_nanos(),
    };

    let report = render_report(
        &result.stage,
        &comparison,
        raster_bytes.len(),
        direct.encoded.data.len(),
        &raster_sha,
        &direct_sha,
        &raster_structural,
        &direct_structural,
        &raster_bin,
        &direct_bin,
        &result,
    );
    fs::write(parity_dir.join(REPORT_TXT), &report)
        .with_context(|| format!("failed to write {}", parity_dir.join(REPORT_TXT).display()))?;
    let report_json = serde_json::to_vec_pretty(&result).context("failed to encode report.json")?;
    write_atomic(&parity_dir.join(REPORT_JSON), &report_json)?;
    print!("{report}");

    Ok(result)
}

pub fn run_hidden_stage_direct(stage_dir: &Path) -> Result<()> {
    let stage_name = stage_name(stage_dir)?;
    let kind = StageKind::from_stage_name(&stage_name)?;
    validate_stage_input_artifacts(stage_dir)?;

    let direct = routines::run_and_publish(&kind)?;
    println!(
        "staged-infer {stage_name}: output {} structural={} stage={}",
        direct.artifact.data_path.display(),
        direct.artifact.commitment,
        format_duration_ns(direct.timings.direct_stage_duration.as_nanos())
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
            "selected stage run {} is not under an unauthenticated chain root ({})",
            canonical_run.display(),
            expected_roots
                .iter()
                .map(|root| root.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    let authenticated_artifacts = [
        run_dir.join("chain-commitment"),
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
            .join("chains-no-auth"),
        current_dir.join("target").join("staged-infer").join("runs"),
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
        direct: Option<u8>,
    },
}

impl ByteComparison {
    fn is_match(&self) -> bool {
        matches!(self, Self::Match)
    }
}

fn compare_bytes(raster: &[u8], direct: &[u8]) -> ByteComparison {
    for (idx, (left, right)) in raster.iter().zip(direct.iter()).enumerate() {
        if left != right {
            return ByteComparison::Mismatch {
                offset: idx,
                raster: Some(*left),
                direct: Some(*right),
            };
        }
    }
    if raster.len() == direct.len() {
        ByteComparison::Match
    } else {
        let offset = raster.len().min(direct.len());
        ByteComparison::Mismatch {
            offset,
            raster: raster.get(offset).copied(),
            direct: direct.get(offset).copied(),
        }
    }
}

fn render_report(
    stage: &str,
    comparison: &ByteComparison,
    raster_len: usize,
    direct_len: usize,
    raster_sha: &str,
    direct_sha: &str,
    raster_structural: &str,
    direct_structural: &str,
    raster_artifact: &Path,
    direct_artifact: &Path,
    result: &ParityReport,
) -> String {
    match comparison {
        ByteComparison::Match => format!(
            "NON-AUTHORITATIVE staged-infer computational parity {stage}: MATCH\n\
             Raster source mode: unauthenticated (--no-auth)\n\
             output.bin bytes: {raster_len}\n\
             raster sha256: {raster_sha}\n\
             staged-infer sha256: {direct_sha}\n\
             structural commitment: {raster_structural}\n\
             staged-infer stage: {}\n\
             staged-infer kernel: {}\n",
            format_duration_ns(result.direct_stage_duration_ns),
            format_duration_ns(result.kernel_duration_ns),
        ),
        ByteComparison::Mismatch {
            offset,
            raster,
            direct,
        } => format!(
            "NON-AUTHORITATIVE staged-infer computational parity {stage}: MISMATCH\n\
             Raster source mode: unauthenticated (--no-auth)\n\
             raster bytes: {raster_len}\n\
             staged-infer bytes: {direct_len}\n\
             first differing byte offset: {offset}\n\
             raster byte: {}\n\
             staged-infer byte: {}\n\
             raster sha256: {raster_sha}\n\
             staged-infer sha256: {direct_sha}\n\
             raster structural commitment: {raster_structural}\n\
             staged-infer structural commitment: {direct_structural}\n\
             raster artifact: {}\n\
             staged-infer artifact: {}\n\
             staged-infer stage: {}\n\
             staged-infer kernel: {}\n",
            render_byte(*raster),
            render_byte(*direct),
            raster_artifact.display(),
            direct_artifact.display(),
            format_duration_ns(result.direct_stage_duration_ns),
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
    if report.version != 3 {
        bail!(
            "unsupported staged-infer report version {} in {}",
            report.version,
            path.display()
        );
    }
    Ok(report)
}

pub fn read_chain_execution_times(chain_dir: &Path) -> Result<ChainExecutionTimes> {
    let path = chain_dir.join(EXECUTION_TIMES_JSON);
    let timings: ChainExecutionTimes = serde_json::from_slice(
        &fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to decode {}", path.display()))?;
    if !matches!(timings.version, 1 | 2) {
        bail!(
            "unsupported chain execution-times version {} in {}",
            timings.version,
            path.display()
        );
    }
    Ok(timings)
}

pub fn render_timing_summary(
    timings: &ChainExecutionTimes,
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
        "\nNON-AUTHORITATIVE timing comparison (chain runner: --no-auth; stage execution only)"
    )?;
    writeln!(
        output,
        "{:<stage_width$}  {:>12}  {:>12}  {:>12}  {:>12}  {:>8}  {:>8}",
        "stage", "pipeline exec", "direct stage", "kernel", "saved", "speedup", "parity"
    )?;

    let mut selected_count = 0usize;
    for timing in &timings.stages {
        if timing.name == parity.stage {
            selected_count += 1;
            let speedup = if parity.direct_stage_duration_ns == 0 {
                String::from("—")
            } else {
                format!(
                    "{:.2}x",
                    timing.exec_duration_ns as f64 / parity.direct_stage_duration_ns as f64
                )
            };
            writeln!(
                output,
                "{:<stage_width$}  {:>12}  {:>12}  {:>12}  {:>12}  {:>8}  {:>8}",
                timing.name,
                format_duration_ns(timing.exec_duration_ns),
                format_duration_ns(parity.direct_stage_duration_ns),
                format_duration_ns(parity.kernel_duration_ns),
                format_saved(timing.exec_duration_ns, parity.direct_stage_duration_ns),
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
        "staged-infer parity overhead: {}",
        format_duration_ns(parity.direct_stage_duration_ns)
    )?;
    Ok(output)
}

fn format_saved(raster_ns: u128, direct_ns: u128) -> String {
    if raster_ns >= direct_ns {
        format_duration_ns(raster_ns - direct_ns)
    } else {
        format!("-{}", format_duration_ns(direct_ns - raster_ns))
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
                .join("chains-no-auth")
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
    fn accepts_complete_no_auth_stage_without_chain_commitment() {
        let stage = TestStage::no_auth("prefill_range_l3");

        validate_stage_artifacts(&stage.stage_dir).unwrap();
    }

    #[test]
    fn rejects_authenticated_artifacts_in_no_auth_stage() {
        let stage = TestStage::no_auth("prefill_range_l3");
        fs::write(stage.run_dir.join("chain-commitment"), []).unwrap();
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
                direct: Some(b'x')
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
                direct: None
            }
        );
    }
}
