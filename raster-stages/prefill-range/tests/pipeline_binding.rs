use model_identity::{pipeline_core::Execution, ModelContext};
use prefill_range::{cache_coverage, consume_cache_position, validate_cache_coverage};

fn context(start: u32, end: u32) -> ModelContext {
    let mut model = ModelContext::default();
    model.model_id = "22".repeat(32);
    model.bind_statement(&"11".repeat(32), false, 16, 8);
    let mut execution: Execution = model.execution_context().unwrap();
    for _ in 0..start {
        execution.end.append(0).unwrap();
    }
    execution.before = execution.end;
    for _ in start..end {
        execution.end.append(0).unwrap();
    }
    model.set_execution(&execution);
    model.producer = model_identity::context::TRANSFORMER;
    model
}

#[test]
fn cache_coverage_rejects_gaps_duplicates_reordering_and_extra_positions() {
    let model = context(4, 8);
    for (positions, clean) in [
        (vec![0, 1, 2, 3, 4, 5, 6, 7], true),
        (vec![0, 1, 2, 4, 5, 6, 7], false),
        (vec![0, 1, 2, 2, 3, 4, 5, 6, 7], false),
        (vec![0, 1, 3, 2, 4, 5, 6, 7], false),
        (vec![0, 1, 2, 3, 4, 5, 6], false),
        (vec![0, 1, 2, 3, 4, 5, 6, 7, 8], false),
    ] {
        let mut coverage = cache_coverage(&model, 0);
        for position in positions {
            consume_cache_position(&mut coverage, position);
        }
        assert_eq!(
            validate_cache_coverage(model.clone(), coverage).fault == 0,
            clean
        );
    }
}

#[test]
fn window_coverage_uses_the_producing_ranges_retention_boundary() {
    let model = context(4, 8);
    let mut coverage = cache_coverage(&model, 3);
    assert_eq!(coverage.next, 2);
    for position in 2..8 {
        consume_cache_position(&mut coverage, position);
    }
    assert_eq!(validate_cache_coverage(model, coverage).fault, 0);
}
