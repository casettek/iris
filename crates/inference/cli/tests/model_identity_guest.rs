//! Explicit guest execution gate; requires the installed RISC Zero toolchain.
use model_identity::ModelDescriptor;
use output_finalize::input::{DecoderProof, DecoderTable, DecoderToken, FinalizeState};
use raster::{BytesPage, List};
use raster_backend::ExecutionMode;
use raster_backend_risc0::Risc0Backend;
use raster_compiler::{builder::Builder, tile::TileDiscovery, Project};

fn page(bytes: Vec<u8>) -> BytesPage {
    BytesPage::__from_parts(0, 0, bytes)
}

#[test]
#[ignore = "builds and executes a RISC Zero guest"]
fn decoder_substitution_is_rejected_by_guest_execution() {
    let stage = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../raster-stages/output-finalize")
        .canonicalize()
        .unwrap();
    let project = Project::new(stage.clone()).unwrap();
    let discovery = TileDiscovery::new(&project);
    let tile = discovery.get("consume_authenticated_decoder").unwrap();
    let backend = Risc0Backend::new(stage.join("target/raster"))
        .with_user_crate(stage)
        .with_guest_stdout(false);
    let backend = raster_compiler::backend::BackendImpl::Risc0(backend);
    let builder = Builder::new(&project, &backend);
    let runner = builder.build_tile_runner(&tile).unwrap();

    let honest = DecoderToken {
        token: "hello".into(),
        special: false,
        terminal: false,
    };
    let table = DecoderTable {
        tokens: List::from(vec![honest.clone()]),
    };
    let root = raster::core::tree::value_root(&table).unwrap();
    let descriptor = ModelDescriptor {
        version: 1,
        tokenizer: [1; 32],
        decoder: root,
        embedding: [2; 32],
        head: [3; 32],
        auxiliary: vec![[4; 32]],
        transformer: vec![[5; 32]],
    };
    let id = hex::encode(descriptor.model_id().unwrap());
    let descriptor = page(descriptor.encode().unwrap());
    let begin = builder
        .build_tile_runner(&discovery.get("begin_authenticated_finalize").unwrap())
        .unwrap();
    for expected_id in [&id, &"00".repeat(32)] {
        let execution = begin
            .run(
                &postcard::to_allocvec(&(&descriptor, expected_id)).unwrap(),
                ExecutionMode::Estimate,
            )
            .unwrap();
        let state: FinalizeState = postcard::from_bytes(&execution.output).unwrap();
        assert_eq!(
            state,
            output_finalize::authenticated_finalize_state(&descriptor, expected_id)
        );
        eprintln!(
            "begin model fault {}: {:?} cycles",
            state.model_fault, execution.cycles
        );
    }
    let proof = DecoderProof {
        siblings: page(vec![]),
    };
    let mut altered_text = honest.clone();
    altered_text.token = "different".into();
    let mut altered_special = honest.clone();
    altered_special.special = true;
    let mut altered_eos = honest.clone();
    altered_eos.terminal = true;
    for (token, fault) in [
        (honest.clone(), 0),
        (altered_text, 1),
        (altered_special, 1),
        (altered_eos, 1),
    ] {
        let state = output_finalize::authenticated_finalize_state(&descriptor, &id);
        let expected = output_finalize::advance_authenticated_finalize(
            state.clone(),
            &descriptor,
            &id,
            0,
            1,
            &proof.siblings,
            token.clone(),
        );
        let encoded =
            postcard::to_allocvec(&(state, &descriptor, &id, 0u32, 1u64, &proof, &token)).unwrap();
        let execution = runner.run(&encoded, ExecutionMode::Estimate).unwrap();
        let output: FinalizeState = postcard::from_bytes(&execution.output).unwrap();
        assert_eq!(output, expected);
        assert_eq!(output.model_fault, fault);
        assert_eq!(output.text, if fault == 0 { "hello" } else { "" });
        assert_eq!(output.count, 1);
        eprintln!("decoder model fault {fault}: {:?} cycles", execution.cycles);
    }
}

#[test]
#[ignore = "builds and executes RISC Zero guests for all component roles"]
fn component_roles_and_sticky_context_are_enforced_in_guests() {
    let d = ModelDescriptor {
        version: 1,
        tokenizer: [1; 32],
        decoder: [2; 32],
        embedding: [3; 32],
        head: [4; 32],
        auxiliary: vec![[5; 32], [6; 32]],
        transformer: vec![[7; 32], [8; 32]],
    };
    let mut context = model_identity::ModelContext::bind(
        &d.encode().unwrap(),
        &hex::encode(d.model_id().unwrap()),
    );
    context.bind_statement(&"11".repeat(32), false, 4, 4);
    context.bind_schedule(model_identity::pipeline_core::StageSchedule {
        prompt_ranges: 1,
        claimed_ranges: 1,
    });
    context.begin_prompt_range();
    context.append_consumed_token(1);
    context.finish_prompt_range(true, 1, 0);
    for (stage, role, root, indexed) in [
        (
            "prompt-check",
            model_identity::Role::Tokenizer,
            [1; 32],
            false,
        ),
        (
            "input-embedding",
            model_identity::Role::Embedding,
            [3; 32],
            false,
        ),
        (
            "prefill-prepare-aux",
            model_identity::Role::Auxiliary(1),
            [6; 32],
            true,
        ),
        (
            "prefill-range",
            model_identity::Role::Transformer(1),
            [8; 32],
            true,
        ),
        ("sequence-check", model_identity::Role::Head, [4; 32], false),
    ] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../raster-stages")
            .join(stage)
            .canonicalize()
            .unwrap();
        let project = Project::new(path.clone()).unwrap();
        let backend = raster_compiler::backend::BackendImpl::Risc0(
            Risc0Backend::new(path.join("target/raster"))
                .with_user_crate(path)
                .with_guest_stdout(false),
        );
        let discovery = TileDiscovery::new(&project);
        let builder = Builder::new(&project, &backend);
        let tile_name = match stage {
            "prompt-check" => "identity_prompt",
            "input-embedding" => "identity_embedding_context",
            _ => "identity_verify",
        };
        let runner = builder
            .build_tile_runner(&discovery.get(tile_name).unwrap())
            .unwrap();
        for (actual, prior_fault, index) in [
            (root, 0, 1u32),
            ([99; 32], 0, 1),
            (root, 7, 1),
            (root, 0, 0),
        ] {
            let mut input = context.clone();
            input.fault = prior_fault;
            let root = page(actual.into());
            let prompt = output_finalize::prompt::PromptContext {
                model: input.clone(),
                model_id: input.model_id.clone(),
                ..Default::default()
            };
            let bytes = if indexed {
                postcard::to_allocvec(&(&input, &root, index)).unwrap()
            } else if stage == "prompt-check" {
                postcard::to_allocvec(&(&prompt, &root)).unwrap()
            } else if stage == "input-embedding" {
                postcard::to_allocvec(&(
                    &input,
                    &model_identity::ModelContext::default(),
                    &root,
                    (0u32,),
                    0u64,
                ))
                .unwrap()
            } else {
                postcard::to_allocvec(&(&input, &root)).unwrap()
            };
            let execution = runner.run(&bytes, ExecutionMode::Estimate).unwrap();
            let output: model_identity::ModelContext = if stage == "prompt-check" {
                postcard::from_bytes::<output_finalize::prompt::PromptContext>(&execution.output)
                    .unwrap()
                    .model
            } else {
                postcard::from_bytes(&execution.output).unwrap()
            };
            let input = if stage == "input-embedding" {
                model_identity::ModelContext::embedding(&input, &Default::default())
            } else {
                input
            };
            let expected_role = match role {
                model_identity::Role::Auxiliary(_) => model_identity::Role::Auxiliary(index),
                model_identity::Role::Transformer(_) => model_identity::Role::Transformer(index),
                r => r,
            };
            assert_eq!(
                output,
                model_identity::stream::verify(input, &root, expected_role)
            );
            assert_eq!(
                output.fault == 0,
                actual != [99; 32] && prior_fault == 0 && (!indexed || index == 1)
            );
            eprintln!(
                "{stage} component fault {}: {:?} cycles",
                output.fault, execution.cycles
            );
        }
    }
}

#[test]
#[ignore = "executes bounded page hashing and model-context joins in RISC Zero"]
fn page_bytes_and_final_context_are_checked_in_guests() {
    let stage = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../raster-stages/input-embedding")
        .canonicalize()
        .unwrap();
    let project = Project::new(stage.clone()).unwrap();
    let backend = raster_compiler::backend::BackendImpl::Risc0(
        Risc0Backend::new(stage.join("target/raster"))
            .with_user_crate(stage)
            .with_guest_stdout(false),
    );
    let builder = Builder::new(&project, &backend);
    let discovery = TileDiscovery::new(&project);
    let runner = builder
        .build_tile_runner(&discovery.get("identity_page").unwrap())
        .unwrap();
    let mut fold = model_identity::stream::empty();
    for (index, bytes) in [vec![1, 2, 3, 4], vec![5, 6], vec![7; 195_840]]
        .into_iter()
        .enumerate()
    {
        let value = BytesPage::__from_parts(index as u64, index as u64 * 4, bytes);
        let input = raster::RecurInput::new(value.clone(), index as u64, 3);
        let state = raster::RecurState::new(fold.clone());
        let execution = runner
            .run(
                &postcard::to_allocvec(&(input, state)).unwrap(),
                ExecutionMode::Estimate,
            )
            .unwrap();
        let journal: raster::core::draft::TileReplayJournal =
            postcard::from_bytes(execution.journal.as_ref().unwrap()).unwrap();
        let transition = journal.recur.unwrap().state.unwrap();
        assert_eq!(
            transition.state_in,
            raster::core::tree::value_root(&fold).unwrap()
        );
        fold = model_identity::stream::append(&fold, &model_identity::stream::digest(&value));
        assert_eq!(
            transition.state_out,
            raster::core::tree::value_root(&fold).unwrap()
        );
        eprintln!("page hash: {:?} cycles", execution.cycles);
    }
    let stage = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../raster-stages/output-finalize")
        .canonicalize()
        .unwrap();
    let project = Project::new(stage.clone()).unwrap();
    let backend = raster_compiler::backend::BackendImpl::Risc0(
        Risc0Backend::new(stage.join("target/raster"))
            .with_user_crate(stage)
            .with_guest_stdout(false),
    );
    let builder = Builder::new(&project, &backend);
    let discovery = TileDiscovery::new(&project);
    let runner = builder
        .build_tile_runner(&discovery.get("verify_final_model_context").unwrap())
        .unwrap();
    let d = ModelDescriptor {
        version: 1,
        tokenizer: [1; 32],
        decoder: [2; 32],
        embedding: [3; 32],
        head: [4; 32],
        auxiliary: vec![[5; 32]],
        transformer: vec![[6; 32]],
    };
    let descriptor = page(d.encode().unwrap());
    let id = hex::encode(d.model_id().unwrap());
    let (p, previous, current) = execution_fixture();
    let prompt = output_finalize::prompt::PromptContext {
        model: p,
        model_id: id.clone(),
        input_text_sha256: "11".repeat(32),
        max_new_tokens: 4,
        range_size: 4,
        prompt_tokens: 2,
        complete: true,
        ..Default::default()
    };
    let honest = current.head(&previous, &prompt.model, &Default::default());
    for case in 0..4 {
        let mut edge = honest.clone();
        match case {
            1 => edge.model_id = "00".repeat(32),
            2 => edge.completed_layers = 0,
            3 => edge.fault = 4,
            _ => (),
        }
        let state = output_finalize::authenticated_finalize_state(&descriptor, &id);
        let execution = runner
            .run(
                &postcard::to_allocvec(&(&state, &edge, &prompt, &descriptor, true, 2u32, 3u32))
                    .unwrap(),
                ExecutionMode::Estimate,
            )
            .unwrap();
        let result: FinalizeState = postcard::from_bytes(&execution.output).unwrap();
        assert_eq!(
            result.model_fault,
            output_finalize::final_context_fault(&edge, &prompt, &descriptor)
        );
        assert_eq!(result.model_fault == 0, case == 0);
        eprintln!(
            "final context fault {}: {:?} cycles",
            result.model_fault, execution.cycles
        );
    }
}

#[test]
#[ignore = "executes sparse membership checks in the four production Raster guests"]
fn sparse_membership_is_checked_by_production_guests() {
    use model_identity::{sparse, stream, ModelContext};
    let values = vec![
        page(vec![1; 1024]),
        page(vec![2; 1024]),
        page(vec![3; 1024]),
    ];
    let witness = sparse::witness(&values).unwrap();
    for stage in [
        "input-embedding",
        "prefill-prepare-aux",
        "prompt-bind",
        "prompt-check",
    ] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../raster-stages")
            .join(stage)
            .canonicalize()
            .unwrap();
        let project = Project::new(path.clone()).unwrap();
        let backend = raster_compiler::backend::BackendImpl::Risc0(
            Risc0Backend::new(path.join("target/raster"))
                .with_user_crate(path)
                .with_guest_stdout(false),
        );
        let builder = Builder::new(&project, &backend);
        let prompt_stage = stage.starts_with("prompt-");
        let tile = if prompt_stage {
            "identity_sparse_context"
        } else {
            "identity_sparse_page"
        };
        let discovery = TileDiscovery::new(&project);
        let runner = builder
            .build_tile_runner(&discovery.get(tile).unwrap())
            .unwrap();
        for case in 0..7 {
            let mut context = ModelContext::default();
            context.fault = if case == 6 { 7 } else { 0 };
            let mut header = witness.header.clone();
            let mut proof = witness.paths.as_slice()[0].clone();
            let index = if case == 2 { 1u64 } else { 0 };
            let value = if case == 1 {
                values[1].clone()
            } else {
                values[0].clone()
            };
            if case == 3 {
                header.count += 1;
            }
            if case == 4 {
                proof.siblings = page(vec![0; 31]);
            }
            if case == 5 {
                header.root = page(vec![0; 32]);
            }
            let input = if prompt_stage {
                let prompt = output_finalize::prompt::PromptContext {
                    model: context,
                    ..Default::default()
                };
                postcard::to_allocvec(&(
                    prompt,
                    stream::digest(&value),
                    index as u32,
                    header,
                    proof,
                ))
                .unwrap()
            } else {
                postcard::to_allocvec(&(context, value, index, header, proof)).unwrap()
            };
            let execution = runner.run(&input, ExecutionMode::Estimate).unwrap();
            let output = if prompt_stage {
                postcard::from_bytes::<output_finalize::prompt::PromptContext>(&execution.output)
                    .unwrap()
                    .model
            } else {
                postcard::from_bytes::<ModelContext>(&execution.output).unwrap()
            };
            assert_eq!(output.fault == 0, case == 0, "{stage}, case {case}");
            if case == 6 {
                assert_eq!(output.fault, 7);
            }
            eprintln!("sparse {stage} case {case}: {:?} cycles", execution.cycles);
        }
        if !prompt_stage {
            let pages = (0..3u64)
                .map(|i| BytesPage::__from_parts(i, i * 195_840, vec![i as u8; 195_840]))
                .collect::<Vec<_>>();
            let full = sparse::witness(&pages).unwrap();
            let input = postcard::to_allocvec(&(
                ModelContext::default(),
                &pages[1],
                1u64,
                &full.header,
                &full.paths.as_slice()[1],
            ))
            .unwrap();
            let execution = runner.run(&input, ExecutionMode::Estimate).unwrap();
            assert_eq!(
                postcard::from_bytes::<ModelContext>(&execution.output)
                    .unwrap()
                    .fault,
                0
            );
            eprintln!(
                "sparse {stage} full 195840-byte page: {:?} cycles",
                execution.cycles
            );
        }
    }
}

#[test]
#[ignore = "builds and executes the production prompt-check guest"]
fn prompt_statement_substitution_is_rejected_in_guest() {
    use output_finalize::prompt::{join_prompt_models, PromptContext, FAULT_CONTEXT_MISMATCH};
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../raster-stages/prompt-check")
        .canonicalize()
        .unwrap();
    let project = Project::new(path.clone()).unwrap();
    let backend = raster_compiler::backend::BackendImpl::Risc0(
        Risc0Backend::new(path.join("target/raster"))
            .with_user_crate(path)
            .with_guest_stdout(false),
    );
    let discovery = TileDiscovery::new(&project);
    let runner = Builder::new(&project, &backend)
        .build_tile_runner(&discovery.get("join_prompt_contexts").unwrap())
        .unwrap();
    let descriptor = ModelDescriptor {
        version: 1,
        tokenizer: [1; 32],
        decoder: [2; 32],
        embedding: [3; 32],
        head: [4; 32],
        auxiliary: vec![[5; 32]],
        transformer: vec![[6; 32]],
    };
    let model_id = hex::encode(descriptor.model_id().unwrap());
    let mut model = model_identity::ModelContext::bind(&descriptor.encode().unwrap(), &model_id);
    model.bind_statement(&"11".repeat(32), false, 8, 4);
    model.bind_schedule(model_identity::pipeline_core::StageSchedule {
        prompt_ranges: 1,
        claimed_ranges: 1,
    });
    let bound = PromptContext {
        input_text_sha256: "11".repeat(32),
        model,
        model_id,
        max_new_tokens: 8,
        range_size: 4,
        ..Default::default()
    };
    let mut changed = bound.clone();
    changed.input_text_sha256 = "substituted".into();
    let mut config = bound.clone();
    config.max_new_tokens += 1;
    for (previous, fault) in [
        (bound.clone(), 0),
        (changed, FAULT_CONTEXT_MISMATCH),
        (config, FAULT_CONTEXT_MISMATCH),
    ] {
        let expected = join_prompt_models(previous.clone(), &bound);
        let execution = runner
            .run(
                &postcard::to_allocvec(&(previous, &bound)).unwrap(),
                ExecutionMode::Estimate,
            )
            .unwrap();
        let actual: PromptContext = postcard::from_bytes(&execution.output).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(actual.fault, fault);
        assert_eq!(actual.model.fault, 0);
    }
}

#[test]
#[ignore = "builds and executes the production prompt-bind guest"]
fn stage_schedule_is_checked_and_stored_in_guest() {
    use model_identity::pipeline_core::MAX_RANGES;
    use output_finalize::prompt::{bind_run_schedule, PromptContext, RunSchedule};
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../raster-stages/prompt-bind")
        .canonicalize()
        .unwrap();
    let project = Project::new(path.clone()).unwrap();
    let backend = raster_compiler::backend::BackendImpl::Risc0(
        Risc0Backend::new(path.join("target/raster"))
            .with_user_crate(path)
            .with_guest_stdout(false),
    );
    let discovery = TileDiscovery::new(&project);
    let runner = Builder::new(&project, &backend)
        .build_tile_runner(&discovery.get("bind_schedule").unwrap())
        .unwrap();
    let descriptor = ModelDescriptor {
        version: 1,
        tokenizer: [1; 32],
        decoder: [2; 32],
        embedding: [3; 32],
        head: [4; 32],
        auxiliary: vec![[5; 32]],
        transformer: vec![[6; 32]],
    };
    let model_id = hex::encode(descriptor.model_id().unwrap());
    let mut model = model_identity::ModelContext::bind(&descriptor.encode().unwrap(), &model_id);
    model.bind_statement(&"11".repeat(32), false, 8, 4);
    let bound = PromptContext {
        input_text_sha256: "11".repeat(32),
        model,
        model_id,
        max_new_tokens: 8,
        range_size: 4,
        ..Default::default()
    };
    // Eight tokens in ranges of four allow two claimed ranges.
    let honest = RunSchedule {
        prompt_ranges: 2,
        claimed_ranges: 2,
    };
    let over_budget = RunSchedule {
        claimed_ranges: 3,
        ..honest.clone()
    };
    let no_ranges = RunSchedule {
        prompt_ranges: 0,
        ..honest.clone()
    };
    let past_limit = RunSchedule {
        prompt_ranges: MAX_RANGES + 1,
        ..honest.clone()
    };
    for (schedule, faulted) in [
        (honest, false),
        (over_budget, true),
        (no_ranges, true),
        (past_limit, true),
    ] {
        let expected = bind_run_schedule(bound.clone(), &schedule);
        let execution = runner
            .run(
                &postcard::to_allocvec(&(&bound, &schedule)).unwrap(),
                ExecutionMode::Estimate,
            )
            .unwrap();
        let actual: PromptContext = postcard::from_bytes(&execution.output).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(actual.model.execution_fault != 0, faulted);
        // The stage finishes and keeps the schedule it was given either way.
        let stored = actual.model.execution_context().unwrap().schedule;
        assert_eq!(
            (stored.prompt_ranges, stored.claimed_ranges),
            (schedule.prompt_ranges, schedule.claimed_ranges)
        );
    }
}

fn execution_fixture() -> (
    model_identity::ModelContext,
    model_identity::ModelContext,
    model_identity::ModelContext,
) {
    use model_identity::ModelContext;
    let d = ModelDescriptor {
        version: 1,
        tokenizer: [1; 32],
        decoder: [2; 32],
        embedding: [3; 32],
        head: [4; 32],
        auxiliary: vec![[5; 32]],
        transformer: vec![[6; 32]],
    };
    let mut p = ModelContext::bind(&d.encode().unwrap(), &hex::encode(d.model_id().unwrap()));
    p.bind_statement(&"11".repeat(32), false, 4, 4);
    p.bind_schedule(model_identity::pipeline_core::StageSchedule {
        prompt_ranges: 1,
        claimed_ranges: 1,
    });
    p.begin_prompt_range();
    p.append_consumed_token(1);
    p.append_consumed_token(2);
    p.finish_prompt_range(true, 2, 0);
    let empty = ModelContext::default();
    let mut e = ModelContext::embedding(&p, &empty);
    e.append_consumed_token(1);
    e.append_consumed_token(2);
    e.finish_embedding_execution(0, 2, 0);
    let prev = e.layer(0, &e.auxiliary(0), &empty, &empty, &empty, -1, -1, -1);
    let mut c = ModelContext::embedding(&empty, &e);
    c.append_consumed_token(3);
    c.finish_embedding_execution(2, 0, 1);
    let current = c.layer(0, &c.auxiliary(0), &prev, &empty, &empty, -1, -1, -1);
    assert_eq!(current.fault, 0);
    (p, prev, current)
}

#[test]
#[ignore = "builds and executes production sequence-check and finalization guests"]
fn pipeline_history_and_transcript_substitutions_match_native_in_guest() {
    use model_identity::ModelContext;
    let (prompt, previous, current) = execution_fixture();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../raster-stages/sequence-check")
        .canonicalize()
        .unwrap();
    let project = Project::new(path.clone()).unwrap();
    let backend = raster_compiler::backend::BackendImpl::Risc0(
        Risc0Backend::new(path.join("target/raster"))
            .with_user_crate(path)
            .with_guest_stdout(false),
    );
    let discovery = TileDiscovery::new(&project);
    let runner = Builder::new(&project, &backend)
        .build_tile_runner(&discovery.get("head_model_context").unwrap())
        .unwrap();
    let empty = ModelContext::default();
    let mut foreign = previous.clone();
    let mut history = foreign.execution_context().unwrap();
    history.end.digest[0] ^= 1;
    foreign.set_execution(&history);
    let mut foreign_prompt = prompt.clone();
    let mut statement = foreign_prompt.execution_context().unwrap();
    statement.statement.max_new_tokens += 1;
    foreign_prompt.set_execution(&statement);
    let mut late = current.clone();
    let mut history = late.execution_context().unwrap();
    history.before = history.end;
    history.end.append(4).unwrap();
    late.set_execution(&history);
    for (c, p, s, clean) in [
        (current.clone(), previous.clone(), prompt.clone(), true),
        (current.clone(), foreign, prompt.clone(), false),
        (current.clone(), previous.clone(), foreign_prompt, false),
        (late, current.clone(), prompt.clone(), false),
    ] {
        let expected = c.head(&p, &s, &empty);
        let actual = runner
            .run(
                &postcard::to_allocvec(&(c, p, s, &empty)).unwrap(),
                ExecutionMode::Estimate,
            )
            .unwrap();
        let actual: ModelContext = postcard::from_bytes(&actual.output).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(actual.fault == 0, clean);
    }
    let checked = current.head(&previous, &prompt, &empty);
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../raster-stages/output-finalize")
        .canonicalize()
        .unwrap();
    let project = Project::new(path.clone()).unwrap();
    let backend = raster_compiler::backend::BackendImpl::Risc0(
        Risc0Backend::new(path.join("target/raster"))
            .with_user_crate(path)
            .with_guest_stdout(false),
    );
    let discovery = TileDiscovery::new(&project);
    let runner = Builder::new(&project, &backend)
        .build_tile_runner(&discovery.get("finish_finalize_state").unwrap())
        .unwrap();
    for (ids, clean) in [
        (vec![3], true),
        (vec![], false),
        (vec![4], false),
        (vec![3, 3], false),
    ] {
        let mut state = output_finalize::initial_finalize_state();
        output_finalize::begin_final_history(&mut state, &checked, true, 2, 3);
        for id in ids {
            state = output_finalize::advance_finalize_state(
                state,
                id,
                DecoderToken {
                    token: "a".into(),
                    special: false,
                    terminal: false,
                },
            );
        }
        let expected = output_finalize::finished_state(state.clone());
        let actual = runner
            .run(
                &postcard::to_allocvec(&(state,)).unwrap(),
                ExecutionMode::Estimate,
            )
            .unwrap();
        let actual: FinalizeState = postcard::from_bytes(&actual.output).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(actual.model_fault == 0, clean);
    }
}
