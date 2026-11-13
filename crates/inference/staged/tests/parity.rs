use std::path::Path;

use staged_infer::parity::{
    render_timing_summary, ChainExecutionTimes, ParityAuthority, ParityReport, RasterSourceMode,
};

#[test]
fn parity_artifacts_live_under_stage_directory() {
    let stage = Path::new("target/raster/chains/run/range_p0_l3");
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
                "name": "range_p0_l3",
                "exec_duration_ns": 2000000000
            }],
            "total_exec_duration_ns": 2000000000
        }"#,
    )
    .unwrap();
    let parity = ParityReport {
        version: 3,
        stage: "range_p0_l3".into(),
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
    assert!(summary.contains("range_p0_l3"));
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
                "name": "range_p0_l3",
                "exec_duration_ns": 2000000000
            }],
            "total_exec_duration_ns": 2000000000,
            "total_wall_duration_ns": 1200000000,
            "aux_waves": [{
                "name": "prefill_prepare_aux",
                "first_stage": "aux_p0_l0",
                "last_stage": "aux_p0_l34",
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
        stage: "range_p0_l3".into(),
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
