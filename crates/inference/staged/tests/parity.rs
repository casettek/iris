use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use staged_infer::parity::{
    render_timing_summary, ChainExecutionTimes, ParityAuthority, ParityReport, RasterSourceMode,
};

#[test]
fn parity_artifacts_live_under_stage_directory() {
    let stage = Path::new("target/raster/chains/run/prefill_range_l3");
    assert_eq!(
        staged_infer::parity::parity_dir(stage),
        stage.join("staged-infer-parity")
    );
}

#[test]
fn timing_summary_annotates_selected_stage() {
    let timings: ChainExecutionTimes = serde_json::from_str(
        r#"{
            "version": 1,
            "stages": [{
                "name": "prefill_range_l3",
                "exec_duration_ns": 2000000000
            }],
            "total_exec_duration_ns": 2000000000
        }"#,
    )
    .unwrap();
    let parity = ParityReport {
        version: 3,
        stage: "prefill_range_l3".into(),
        routine: "prefill_range".into(),
        instance: Some(3),
        authority: ParityAuthority::NonAuthoritative,
        raster_source_mode: RasterSourceMode::Unauthenticated,
        matched: true,
        input_load_duration_ns: 10_000_000,
        kernel_duration_ns: 200_000_000,
        encode_write_duration_ns: 20_000_000,
        direct_stage_duration_ns: 250_000_000,
    };
    let encoded = serde_json::to_vec(&parity).unwrap();
    let json = String::from_utf8(encoded.clone()).unwrap();
    assert!(json.contains(r#""authority":"non_authoritative""#));
    assert!(json.contains(r#""raster_source_mode":"unauthenticated""#));
    let parity: ParityReport = serde_json::from_slice(&encoded).unwrap();

    let summary = render_timing_summary(&timings, &parity).unwrap();
    assert!(summary.contains("prefill_range_l3"));
    assert!(summary.contains("250.00ms"));
    assert!(summary.contains("8.00x"));
    assert!(summary.contains("MATCH"));
    assert!(summary.contains("NON-AUTHORITATIVE"));
    assert!(summary.contains("chain runner: --no-auth"));
}

#[test]
fn timing_summary_reports_wall_time_and_aux_wave_speedup() {
    let timings: ChainExecutionTimes = serde_json::from_str(
        r#"{
            "version": 2,
            "stages": [{
                "name": "prefill_range_l3",
                "exec_duration_ns": 2000000000
            }],
            "total_exec_duration_ns": 2000000000,
            "total_wall_duration_ns": 1200000000,
            "aux_waves": [{
                "name": "prefill_prepare_aux",
                "first_stage": "prefill_prepare_aux_l0",
                "last_stage": "prefill_prepare_aux_l34",
                "stage_count": 35,
                "parallelism": 4,
                "wall_duration_ns": 5000000000,
                "stage_duration_sum_ns": 20000000000
            }]
        }"#,
    )
    .unwrap();
    let parity = ParityReport {
        version: 3,
        stage: "prefill_range_l3".into(),
        routine: "prefill_range".into(),
        instance: Some(3),
        authority: ParityAuthority::NonAuthoritative,
        raster_source_mode: RasterSourceMode::Unauthenticated,
        matched: true,
        input_load_duration_ns: 10_000_000,
        kernel_duration_ns: 200_000_000,
        encode_write_duration_ns: 20_000_000,
        direct_stage_duration_ns: 250_000_000,
    };

    let summary = render_timing_summary(&timings, &parity).unwrap();

    assert!(summary.contains("stage sum"));
    assert!(summary.contains("wall time"));
    assert!(summary.contains("1.20s"));
    assert!(summary.contains("aux wave prefill_prepare_aux"));
    assert!(summary.contains("parallelism=4"));
    assert!(summary.contains("effective=4.00x"));
}

#[test]
#[ignore = "runs the full staged-infer chain twice"]
fn full_chain_parallel_aux_matches_single_worker_outputs() {
    let single_worker = run_chain_with_aux_parallelism("1");
    let parallel = run_chain_with_aux_parallelism("4");

    for layer in 0..35 {
        assert_stage_output_matches(
            &single_worker,
            &parallel,
            &format!("prefill_prepare_aux_l{layer}"),
        );
    }
    assert_stage_output_matches(&single_worker, &parallel, "prefill_range_l0");
    assert_stage_output_matches(&single_worker, &parallel, "decode_select_token");
}

fn run_chain_with_aux_parallelism(parallelism: &str) -> PathBuf {
    let output = Command::new(full_chain_binary())
        .current_dir(Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap())
        .args(["chain", "run", "--staged-subprocess"])
        .env("STAGED_INFER_AUX_PARALLELISM", parallelism)
        .output()
        .expect("staged-infer chain run should spawn");
    assert!(
        output.status.success(),
        "staged-infer chain run failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find_map(|line| line.trim().strip_prefix("dir: ").map(PathBuf::from))
        .expect("staged-infer output should print chain run dir")
}

fn full_chain_binary() -> PathBuf {
    if let Ok(path) = std::env::var("STAGED_INFER_FULL_CHAIN_BIN") {
        return PathBuf::from(path);
    }

    let release_binary = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("release")
        .join("staged-infer");
    assert!(
        release_binary.is_file(),
        "build the release binary first with `cargo build --release -p staged-infer`, or set STAGED_INFER_FULL_CHAIN_BIN"
    );
    release_binary
}

fn assert_stage_output_matches(left_run: &Path, right_run: &Path, stage: &str) {
    let left = std::fs::read(left_run.join(stage).join("output.bin"))
        .expect("left stage output should exist");
    let right = std::fs::read(right_run.join(stage).join("output.bin"))
        .expect("right stage output should exist");
    assert_eq!(left, right, "stage {stage} output changed");
}
