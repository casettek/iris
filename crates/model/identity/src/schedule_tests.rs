//! A whole run under a stage schedule, and every way of leaving it.
use crate::{ModelContext, ModelDescriptor};
use alloc::{format, string::String, vec, vec::Vec};
use pipeline_core::{StageSchedule, Step};

/// Two prompt ranges, two claimed ranges, two layers.
fn schedule() -> StageSchedule {
    StageSchedule {
        prompt_ranges: 2,
        claimed_ranges: 2,
    }
}
fn origin() -> ModelContext {
    ModelContext::default()
}
fn statement() -> ModelContext {
    let d = ModelDescriptor {
        version: 1,
        tokenizer: [1; 32],
        decoder: [2; 32],
        embedding: [3; 32],
        head: [4; 32],
        auxiliary: vec![[5; 32]; 2],
        transformer: vec![[6; 32]; 2],
    };
    let id = d
        .model_id()
        .unwrap()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    let mut context = ModelContext::bind(&d.encode().unwrap(), &id);
    context.bind_statement(&"11".repeat(32), false, 8, 4);
    context
}
fn bind(schedule: StageSchedule) -> ModelContext {
    let mut context = statement();
    context.bind_schedule(schedule);
    context
}
/// A range's token check as its stage opens it, before it reads a token.
fn open(previous: &ModelContext, bound: &ModelContext) -> ModelContext {
    let mut context = previous.clone();
    context.begin_prompt_range();
    context.require_bound(bound);
    context
}
/// Every range holds one token: range `j` of the prompt is token `j`.
fn close(opened: &ModelContext, range: u32, complete: bool) -> ModelContext {
    let mut context = opened.clone();
    context.append_consumed_token(range);
    context.finish_prompt_range(complete, range + 1, 0);
    context
}
/// What a range's token check publishes.
fn tokens(
    previous: &ModelContext,
    bound: &ModelContext,
    range: u32,
    complete: bool,
) -> ModelContext {
    close(&open(previous, bound), range, complete)
}
fn embed(prompt: &ModelContext, previous: &ModelContext, position: u32) -> ModelContext {
    let claimed = prompt.is_origin();
    let mut context = ModelContext::embedding(prompt, previous);
    context.append_consumed_token(position);
    context.finish_embedding_execution(position, u64::from(!claimed), u64::from(claimed));
    context
}
fn layer(
    input: &ModelContext,
    embedded: &ModelContext,
    prior: &ModelContext,
    index: u32,
) -> ModelContext {
    input.layer(
        index,
        &embedded.auxiliary(index),
        prior,
        &origin(),
        &origin(),
        -1,
        -1,
        -1,
    )
}
fn step(context: &ModelContext) -> Step {
    context.execution_context().unwrap().step
}

/// The contexts an honest run publishes, stage by stage.
struct Run {
    bound: ModelContext,
    /// Each prompt range's checked tokens.
    prepared: Vec<ModelContext>,
    /// Prompt ranges 0 and 1, then claimed ranges 0 and 1.
    embeds: Vec<ModelContext>,
    /// For each of those four ranges, layers 0 and 1.
    layers: Vec<[ModelContext; 2]>,
    checks: Vec<ModelContext>,
}

fn run() -> Run {
    let bound = bind(schedule());
    let (mut prepared, mut embeds) = (vec![], vec![]);
    let mut previous = bound.clone();
    for range in 0..2 {
        let ready = tokens(&previous, &bound, range, range == 1);
        let before = embeds.last().cloned().unwrap_or_else(origin);
        embeds.push(embed(&ready, &before, range));
        previous = ready.clone();
        prepared.push(ready);
    }
    for range in 0..2 {
        let before = embeds.last().cloned().unwrap();
        embeds.push(embed(&origin(), &before, 2 + range));
    }
    let mut layers: Vec<[ModelContext; 2]> = vec![];
    for range in 0..4 {
        let prior = |index: usize| match range {
            0 => origin(),
            _ => layers[range - 1][index].clone(),
        };
        let lower = layer(&embeds[range], &embeds[range], &prior(0), 0);
        let upper = layer(&lower, &embeds[range], &prior(1), 1);
        layers.push([lower, upper]);
    }
    let first = layers[2][1].head(&layers[1][1], &prepared[1], &origin());
    let second = layers[3][1].head(&layers[2][1], &prepared[1], &first);
    Run {
        bound,
        prepared,
        embeds,
        layers,
        checks: vec![first, second],
    }
}

#[test]
fn an_honest_run_follows_its_schedule_and_records_every_step() {
    let r = run();
    let all = [&r.bound]
        .into_iter()
        .chain(&r.prepared)
        .chain(&r.embeds)
        .chain(r.layers.iter().flatten())
        .chain(&r.checks);
    for context in all {
        assert_eq!(context.fault, 0, "{:?}", step(context));
    }
    assert_eq!(r.checks[1].final_fault(&r.prepared[1], &r.prepared[1]), 0);
    assert_eq!(step(&r.bound), Step::bind());
    assert_eq!(step(&r.prepared[0]), Step::tokens(0));
    assert_eq!(step(&r.prepared[1]), Step::tokens(1));
    assert_eq!(step(&r.embeds[1]), Step::embed(false, 1));
    assert_eq!(step(&r.embeds[3]), Step::embed(true, 1));
    assert_eq!(step(&r.embeds[2].auxiliary(1)), Step::auxiliary(true, 0, 1));
    assert_eq!(step(&r.layers[3][1]), Step::layer(true, 1, 1));
    assert_eq!(step(&r.checks[1]), Step::check(1));
    // The recorded steps are exactly the schedule's, each once.
    let mut recorded: Vec<Step> = [&r.bound]
        .into_iter()
        .chain(&r.prepared)
        .chain(&r.embeds)
        .chain(r.layers.iter().flatten())
        .chain(&r.checks)
        .map(step)
        .collect();
    for (range, embedded) in r.embeds.iter().enumerate() {
        recorded.extend([0, 1].map(|index| step(&embedded.auxiliary(index))));
        assert_eq!(embedded.auxiliary(1).fault, 0, "range {range}");
    }
    recorded.push(Step::finalize());
    let scheduled: Vec<Step> = schedule().steps(2).collect();
    assert_eq!(recorded.len(), scheduled.len());
    for expected in scheduled {
        assert!(recorded.contains(&expected), "{expected:?} never ran");
    }
}

#[test]
fn a_prompt_stage_out_of_place_faults() {
    let r = run();
    for (case, context) in [
        (
            "a range opened twice",
            open(&open(&r.bound, &r.bound), &r.bound),
        ),
        (
            "a range closed without being opened",
            close(&r.bound, 0, false),
        ),
        ("a range closed twice", close(&r.prepared[0], 1, true)),
        (
            "a range past the schedule",
            tokens(&r.prepared[1], &r.bound, 2, false),
        ),
        (
            "the first range completing",
            tokens(&r.bound, &r.bound, 0, true),
        ),
        (
            "the last range not completing",
            tokens(&r.prepared[0], &r.bound, 1, false),
        ),
        (
            "a range bound to something else",
            open(&r.bound, &r.prepared[0]),
        ),
    ] {
        assert_ne!(context.execution_fault, 0, "{case}");
    }
}

#[test]
fn an_embedding_or_layer_out_of_place_faults() {
    let r = run();
    for (case, context) in [
        (
            "a range embedded with nothing before it",
            embed(&r.prepared[1], &origin(), 1),
        ),
        (
            "a range embedded after itself",
            embed(&r.prepared[0], &r.embeds[0], 0),
        ),
        (
            "a range embedded before its tokens are checked",
            embed(&open(&r.bound, &r.bound), &origin(), 0),
        ),
        (
            "a claimed range past the schedule",
            embed(&origin(), &r.embeds[3], 4),
        ),
        (
            "claimed ids before the prompt is embedded",
            embed(&origin(), &r.embeds[0], 1),
        ),
        ("a layer the model lacks", r.embeds[0].auxiliary(2)),
        (
            "auxiliary rows from a layer output",
            r.layers[0][0].auxiliary(1),
        ),
        (
            "a layer skipped",
            layer(&r.embeds[0], &r.embeds[0], &origin(), 1),
        ),
        (
            "a layer repeated",
            layer(&r.layers[0][0], &r.embeds[0], &origin(), 0),
        ),
        (
            "another range's auxiliary rows",
            layer(&r.embeds[1], &r.embeds[0], &r.layers[0][0], 0),
        ),
        (
            "a later range with no cache before it",
            layer(&r.embeds[1], &r.embeds[1], &origin(), 0),
        ),
        (
            "a cache from the wrong range",
            layer(&r.embeds[2], &r.embeds[2], &r.layers[0][0], 0),
        ),
        (
            "a cache from the wrong layer",
            layer(&r.layers[1][0], &r.embeds[1], &r.layers[0][0], 1),
        ),
        (
            "a donor from another range",
            r.layers[0][0].layer(
                1,
                &r.embeds[0].auxiliary(1),
                &origin(),
                &r.layers[1][0],
                &origin(),
                0,
                0,
                -1,
            ),
        ),
    ] {
        assert_ne!(context.fault, 0, "{case}");
    }
    let borrowed = r.layers[0][0].layer(
        1,
        &r.embeds[0].auxiliary(1),
        &origin(),
        &r.layers[0][0],
        &origin(),
        0,
        0,
        -1,
    );
    assert_eq!(borrowed.fault, 0);
}

#[test]
fn a_check_or_the_final_stage_out_of_place_faults() {
    let r = run();
    for (case, context) in [
        (
            "a check before the last layer",
            r.layers[2][0].head(&r.layers[1][0], &r.prepared[1], &origin()),
        ),
        (
            "the second check without the first",
            r.layers[3][1].head(&r.layers[2][1], &r.prepared[1], &origin()),
        ),
        (
            "a check repeated",
            r.layers[2][1].head(&r.layers[1][1], &r.prepared[1], &r.checks[0]),
        ),
        (
            "a check against an earlier prompt range",
            r.layers[2][1].head(&r.layers[1][1], &r.prepared[0], &origin()),
        ),
        (
            "a check over a prompt range",
            r.layers[1][1].head(&r.layers[0][1], &r.prepared[1], &origin()),
        ),
        (
            "a check after the wrong range",
            r.layers[3][1].head(&r.layers[1][1], &r.prepared[1], &r.checks[0]),
        ),
    ] {
        assert_ne!(context.fault, 0, "{case}");
    }
    let last = &r.prepared[1];
    assert_ne!(r.checks[0].final_fault(last, last), 0, "a check not run");
    assert_ne!(
        r.checks[1].final_fault(&r.prepared[0], &r.prepared[0]),
        0,
        "an earlier prompt range"
    );
    assert_ne!(
        r.layers[3][1].final_fault(last, last),
        0,
        "the final stage in a check's place"
    );
}

#[test]
fn a_run_that_disagrees_with_its_counts_faults() {
    let with = |edit: fn(&mut StageSchedule)| {
        let mut changed = schedule();
        edit(&mut changed);
        bind(changed)
    };
    // Scheduled for three prompt ranges, the prompt completes in two.
    let bound = with(|s| s.prompt_ranges = 3);
    let first = tokens(&bound, &bound, 0, false);
    assert_eq!(first.fault, 0);
    let second = tokens(&first, &bound, 1, true);
    assert_ne!(second.execution_fault, 0);
    // Scheduled for one prompt range, the prompt needs two.
    let bound = with(|s| s.prompt_ranges = 1);
    let first = tokens(&bound, &bound, 0, false);
    assert_ne!(first.execution_fault, 0);
    // Scheduled for one claimed range, two run.
    let r = run();
    let bound = with(|s| s.claimed_ranges = 1);
    let mut previous = bound.clone();
    let mut embedded = origin();
    for range in 0..2 {
        previous = tokens(&previous, &bound, range, range == 1);
        embedded = embed(&previous, &embedded, range);
    }
    let claimed = embed(&origin(), &embedded, 2);
    assert_eq!(claimed.fault, 0);
    assert_ne!(embed(&origin(), &claimed, 3).execution_fault, 0);
    // An input from a run under another schedule, and a run with none.
    assert_ne!(open(&r.prepared[0], &bound).execution_fault, 0);
    let unscheduled = statement();
    assert_ne!(open(&unscheduled, &unscheduled).execution_fault, 0);
}
