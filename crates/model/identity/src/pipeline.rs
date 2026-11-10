//! Semantic transition checks over authenticated predecessor metadata.
use crate::{context::parse_id, ModelContext};
use pipeline_core::{
    Execution, Input, Operation, Source, StageSchedule, Statement, Step, BOUND, CHECKED, EMBEDDED,
    PREPARED, TOKENIZING,
};
impl ModelContext {
    pub fn require_execution(&mut self, condition: bool) {
        if !condition {
            self.execution_fault = 1;
            self.fault = 1;
        }
    }
    pub fn execution_context(&self) -> Option<Execution> {
        Execution::decode(self.execution.as_slice())
    }
    pub fn set_execution(&mut self, value: &Execution) {
        self.execution = raster::BytesPage::__from_parts(0, 0, value.encode());
    }
    pub fn bind_statement(&mut self, input: &str, raw: bool, budget: u32, range: u32) {
        if let (Some(input_text), Some(model)) = (parse_id(input), parse_id(&self.model_id)) {
            self.set_execution(&Execution::bind(Statement {
                input_text,
                model,
                raw_prompt: raw,
                max_new_tokens: budget,
                range_size: range,
            }));
            self.require_execution(range > 0 && budget > 0);
        } else {
            self.require_execution(false);
        }
    }
    /// Stores the run's stage schedule beside its statement: once, and before
    /// any range begins. A schedule that fails its checks is kept with a fault.
    pub fn bind_schedule(&mut self, schedule: StageSchedule) {
        if let Some(mut e) = self.execution_context() {
            self.require_execution(
                e.phase == BOUND
                    && e.step == Step::bind()
                    && e.schedule == StageSchedule::default()
                    && schedule.validate(&e.statement).is_ok(),
            );
            e.schedule = schedule;
            self.set_execution(&e);
        } else {
            self.require_execution(false);
        }
    }
    /// Records `step` as this stage's, after checking that the schedule runs it
    /// and puts the step this context arrived with on `input`.
    fn take_step(&mut self, e: &mut Execution, step: Step, input: Input) {
        self.require_execution(
            e.schedule.before(&step, input, self.layers) == Some(Source::Step(e.step)),
        );
        e.step = step;
    }
    /// Requires another input to carry what the schedule puts on it for this
    /// stage's step, under the same statement and schedule.
    fn require_source(&mut self, e: &Execution, input: Input, source: &Self) {
        let held = match e.schedule.before(&e.step, input, self.layers) {
            Some(Source::Step(expected)) => source.execution_context().is_some_and(|s| {
                s.step == expected && s.schedule == e.schedule && s.statement == e.statement
            }),
            Some(Source::Origin) => source.is_origin(),
            None => false,
        };
        self.require_execution(held);
    }
    /// Requires `bound` to be the run's `prompt-bind` output, for a range's
    /// token check.
    pub fn require_bound(&mut self, bound: &Self) {
        match self.execution_context() {
            Some(e) => self.require_source(&e, Input::Bound, bound),
            None => self.require_execution(false),
        }
    }
    /// Opens one prompt range's token check and records its step.
    pub fn begin_prompt_range(&mut self) {
        if let Some(mut e) = self.execution_context() {
            self.require_execution((e.phase == BOUND || e.phase == PREPARED) && e.prompt.is_none());
            // The first range follows the bind, each later one the range before.
            let range = match e.step.operation {
                Operation::Tokens => e.step.range.saturating_add(1),
                _ => 0,
            };
            self.take_step(&mut e, Step::tokens(range), Input::Previous);
            e.before = e.end;
            e.phase = TOKENIZING;
            self.set_execution(&e);
        } else {
            self.require_execution(false);
        }
    }
    pub fn append_consumed_token(&mut self, id: u32) {
        if let Some(mut e) = self.execution_context() {
            self.require_execution(e.end.append(id).is_ok());
            self.set_execution(&e);
        } else {
            self.require_execution(false);
        }
    }
    pub fn finish_prompt_range(&mut self, complete: bool, tokens: u32, fault: u32) {
        if let Some(mut e) = self.execution_context() {
            self.require_execution(
                e.phase == TOKENIZING
                    && e.nonempty_bounded()
                    && e.end.position == tokens
                    && fault == 0,
            );
            // The schedule's last range completes the prompt, and no earlier one.
            self.require_execution(
                complete == (e.step.range.saturating_add(1) == e.schedule.prompt_ranges),
            );
            e.phase = PREPARED;
            if complete {
                e.prompt = Some(e.end);
            }
            self.set_execution(&e);
        } else {
            self.require_execution(false);
        }
    }
    pub fn begin_embedding_execution(&mut self, prompt: &Self, previous: &Self) {
        if prompt.is_origin() {
            if let Some(mut e) = previous.execution_context() {
                self.require_execution(e.phase == EMBEDDED && e.prompt.is_some());
                // The first claimed range follows the prompt's last embedding.
                let range = if e.step.claimed {
                    e.step.range.saturating_add(1)
                } else {
                    0
                };
                self.take_step(&mut e, Step::embed(true, range), Input::Previous);
                e.before = e.end;
                e.claimed = true;
                e.expected = None;
                self.set_execution(&e);
            } else {
                self.require_execution(false);
            }
        } else if let Some(mut e) = prompt.execution_context() {
            self.require_execution(e.phase == PREPARED && !e.claimed);
            let step = Step::embed(false, e.step.range);
            self.take_step(&mut e, step, Input::Prompt);
            self.require_source(&e, Input::Previous, previous);
            if previous.is_origin() {
                self.require_execution(
                    e.before.position == 0
                        && e.before == pipeline_core::Prefix::empty(&e.statement),
                );
            } else if let Some(p) = previous.execution_context() {
                self.require_execution(
                    p.phase == EMBEDDED && !p.claimed && e.same_statement(&p) && p.end == e.before,
                );
            } else {
                self.require_execution(false);
            }
            e.expected = Some(e.end);
            e.end = e.before;
            e.phase = EMBEDDED;
            self.set_execution(&e);
        } else {
            self.require_execution(false);
        }
    }
    pub fn finish_embedding_execution(
        &mut self,
        start: u32,
        prompt_count: u64,
        claimed_count: u64,
    ) {
        if let Some(mut e) = self.execution_context() {
            self.require_execution(
                e.phase == EMBEDDED && e.before.position == start && e.nonempty_bounded(),
            );
            self.require_execution(if e.claimed {
                prompt_count == 0 && claimed_count > 0
            } else {
                prompt_count > 0 && claimed_count == 0
            });
            self.require_execution(
                u64::from(e.end.position - e.before.position)
                    == prompt_count.saturating_add(claimed_count),
            );
            if let Some(expected) = e.expected {
                self.require_execution(e.end == expected);
            }
            e.expected = None;
            self.set_execution(&e);
        } else {
            self.require_execution(false);
        }
    }
    /// One layer's auxiliary inputs for the range this context embeds.
    pub fn auxiliary_execution(&mut self, layer: u32) {
        if let Some(mut e) = self.execution_context() {
            let step = Step::auxiliary(e.step.claimed, e.step.range, layer);
            self.take_step(&mut e, step, Input::Embedded);
            self.set_execution(&e);
        } else {
            self.require_execution(false);
        }
    }
    pub fn validate_rows(&mut self, start: u32, count: u64) {
        if let Some(e) = self.execution_context() {
            self.require_execution(
                e.phase == EMBEDDED
                    && e.before.position == start
                    && u64::from(e.end.position - e.before.position) == count
                    && e.nonempty_bounded(),
            );
        } else {
            self.require_execution(false);
        }
    }
    pub fn layer_execution(
        &mut self,
        aux: &Self,
        prior: &Self,
        a: &Self,
        b: &Self,
        donor: i32,
        ai: i32,
        bi: i32,
    ) {
        if let Some(e) = self.execution_context() {
            self.require_execution(
                e.phase == EMBEDDED && aux.execution_context().is_some_and(|x| e.same_range(&x)),
            );
            if prior.is_origin() {
                self.require_execution(e.before.position == 0);
            } else {
                self.require_execution(prior.execution_context().is_some_and(|p| {
                    if prior.producer == crate::context::EMBEDDING {
                        e.before.position == 0 && e.same_range(&p)
                    } else {
                        e.same_statement(&p) && p.end == e.before && p.phase == EMBEDDED
                    }
                }));
            }
            for (c, i) in [(a, ai), (b, bi)] {
                if donor >= 0 && donor == i {
                    self.require_execution(c.execution_context().is_some_and(|d| e.same_range(&d)));
                }
            }
        } else {
            self.require_execution(false);
        }
    }
    /// One transformer layer over the range its activations carry: after the
    /// layer below, with this layer's auxiliary rows, the range before's cache
    /// and, where the model names one, the donor layer of the same range.
    #[allow(clippy::too_many_arguments)]
    pub fn layer_step(
        &mut self,
        layer: u32,
        aux: &Self,
        prior: &Self,
        a: &Self,
        b: &Self,
        donor: i32,
        ai: i32,
        bi: i32,
    ) {
        if let Some(mut e) = self.execution_context() {
            let step = Step::layer(e.step.claimed, e.step.range, layer);
            self.take_step(&mut e, step, Input::Activations);
            self.require_source(&e, Input::Ple, aux);
            self.require_source(&e, Input::PriorKv, prior);
            for (c, i) in [(a, ai), (b, bi)] {
                if donor >= 0 && donor == i {
                    let expected = e.schedule.donor(&step, donor as u32, self.layers);
                    self.require_execution(
                        expected.is_some()
                            && c.execution_context().is_some_and(|d| {
                                Some(d.step) == expected && d.schedule == e.schedule
                            }),
                    );
                }
            }
            self.set_execution(&e);
        } else {
            self.require_execution(false);
        }
    }
    pub fn head_execution(&mut self, previous: &Self, prompt: &Self, prior: &Self) {
        if let Some(mut e) = self.execution_context() {
            self.require_execution(e.phase == EMBEDDED && e.claimed && e.nonempty_bounded());
            let step = Step::check(e.step.range);
            self.take_step(&mut e, step, Input::Activations);
            self.require_source(&e, Input::Previous, previous);
            self.require_source(&e, Input::Prompt, prompt);
            self.require_source(&e, Input::Prior, prior);
            self.require_execution(
                previous
                    .execution_context()
                    .is_some_and(|p| e.same_statement(&p) && p.end == e.before),
            );
            self.require_execution(prompt.execution_context().is_some_and(|p| {
                e.same_statement(&p)
                    && p.phase == PREPARED
                    && p.prompt.is_some()
                    && p.prompt == e.prompt
            }));
            if prior.is_origin() {
                self.require_execution(Some(e.before) == e.prompt);
            } else {
                self.require_execution(prior.execution_context().is_some_and(|p| {
                    p.phase == CHECKED
                        && e.same_statement(&p)
                        && p.end == e.before
                        && p.prompt == e.prompt
                }));
            }
            e.phase = CHECKED;
            if let Some(anchor) = e.prompt {
                e.before = anchor;
            }
            self.set_execution(&e);
        } else {
            self.require_execution(false);
        }
    }
    pub fn final_execution(&mut self, prompt: &Self) {
        if let Some(mut e) = self.execution_context() {
            self.require_execution(e.phase == CHECKED && e.claimed && e.prompt == Some(e.before));
            // The last stage: every scheduled check has run.
            self.take_step(&mut e, Step::finalize(), Input::Edge);
            self.require_source(&e, Input::Prompt, prompt);
            self.require_execution(prompt.execution_context().is_some_and(|p| {
                p.phase == PREPARED
                    && e.same_statement(&p)
                    && p.prompt == e.prompt
                    && p.prompt == Some(p.end)
            }));
        } else {
            self.require_execution(false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{format, string::String, vec};
    fn bound() -> ModelContext {
        let descriptor = crate::ModelDescriptor {
            version: 1,
            tokenizer: [1; 32],
            decoder: [2; 32],
            embedding: [3; 32],
            head: [4; 32],
            auxiliary: vec![[5; 32]],
            transformer: vec![[6; 32]],
        };
        let id = descriptor
            .model_id()
            .unwrap()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        let mut p = ModelContext::bind(&descriptor.encode().unwrap(), &id);
        p.bind_statement(&"11".repeat(32), false, 8, 4);
        p
    }
    fn prepared(ids: &[u32]) -> ModelContext {
        let mut p = bound();
        p.bind_schedule(StageSchedule {
            prompt_ranges: 1,
            claimed_ranges: 2,
        });
        p.begin_prompt_range();
        for &id in ids {
            p.append_consumed_token(id);
        }
        p.finish_prompt_range(true, ids.len() as u32, 0);
        p
    }
    #[test]
    fn schedule_binds_once_before_any_range_and_keeps_a_bad_one_with_a_fault() {
        let schedule = StageSchedule {
            prompt_ranges: 2,
            claimed_ranges: 1,
        };
        let bind = |mut context: ModelContext, schedule| {
            context.bind_schedule(schedule);
            context
        };
        let good = bind(bound(), schedule);
        let e = good.execution_context().unwrap();
        assert_eq!(
            (good.fault, e.schedule, e.step),
            (0, schedule, Step::bind())
        );
        // A budget of eight tokens in ranges of four allows two claimed ranges.
        for bad in [
            StageSchedule {
                claimed_ranges: 3,
                ..schedule
            },
            StageSchedule {
                prompt_ranges: 0,
                ..schedule
            },
            StageSchedule {
                prompt_ranges: pipeline_core::MAX_RANGES + 1,
                ..schedule
            },
        ] {
            let context = bind(bound(), bad);
            assert_ne!(context.execution_fault, 0);
            assert_eq!(context.execution_context().unwrap().schedule, bad);
        }
        assert_ne!(bind(good.clone(), schedule).fault, 0);
        let mut started = bound();
        started.begin_prompt_range();
        assert_ne!(bind(started, schedule).fault, 0);
        assert_ne!(bind(ModelContext::default(), schedule).fault, 0);
        // The schedule travels with the context, and the next stage records its step.
        let mut later = good;
        later.begin_prompt_range();
        let e = later.execution_context().unwrap();
        assert_eq!(
            (later.fault, e.schedule, e.step),
            (0, schedule, Step::tokens(0))
        );
    }
    fn embed(
        p: &ModelContext,
        previous: &ModelContext,
        ids: &[u32],
        start: u32,
        claimed: bool,
    ) -> ModelContext {
        let mut e = ModelContext::embedding(p, previous);
        for &id in ids {
            e.append_consumed_token(id);
        }
        e.finish_embedding_execution(
            start,
            if claimed { 0 } else { ids.len() as u64 },
            if claimed { ids.len() as u64 } else { 0 },
        );
        e
    }
    fn layer(e: &ModelContext, aux: &ModelContext, prior: &ModelContext) -> ModelContext {
        e.layer(
            0,
            aux,
            prior,
            &Default::default(),
            &Default::default(),
            -1,
            -1,
            -1,
        )
    }
    #[test]
    fn consumed_ids_and_phase_are_bound() {
        let p = prepared(&[1, 2]);
        let empty = ModelContext::default();
        let e = embed(&p, &empty, &[1, 2], 0, false);
        assert_eq!(e.fault, 0);
        assert_ne!(embed(&p, &empty, &[9, 2], 0, false).fault, 0);
        assert_ne!(embed(&p, &empty, &[1], 0, false).fault, 0);
        assert_ne!(embed(&p, &empty, &[1, 2], 1, false).fault, 0);
        let mut mixed = ModelContext::embedding(&p, &empty);
        mixed.append_consumed_token(1);
        mixed.append_consumed_token(2);
        mixed.finish_embedding_execution(0, 1, 1);
        assert_ne!(mixed.fault, 0);
        let mut restarted = p.clone();
        restarted.begin_prompt_range();
        assert_ne!(restarted.fault, 0);
    }
    #[test]
    fn same_length_same_tail_different_history_cannot_join() {
        let empty = ModelContext::default();
        let p = prepared(&[1, 2]);
        let other = prepared(&[9, 2]);
        let e = embed(&p, &empty, &[1, 2], 0, false);
        let f = embed(&other, &empty, &[9, 2], 0, false);
        let a = layer(&e, &e.auxiliary(0), &empty);
        let b = layer(&f, &f.auxiliary(0), &empty);
        let c = embed(&empty, &e, &[3], 2, true);
        assert_eq!(layer(&c, &c.auxiliary(0), &a).fault, 0);
        assert_ne!(layer(&c, &c.auxiliary(0), &b).fault, 0);
        assert_ne!(layer(&c, &c.auxiliary(0), &empty).fault, 0);
        assert_ne!(layer(&e, &f.auxiliary(0), &empty).fault, 0);
        let checked = layer(&c, &c.auxiliary(0), &a).head(&a, &p, &empty);
        // Two claimed ranges are scheduled, so the first check is not the last.
        assert_ne!(checked.final_fault(&p, &p), 0);
        let d = embed(&empty, &c, &[4], 3, true);
        let c_layer = layer(&c, &c.auxiliary(0), &a);
        let d_layer = layer(&d, &d.auxiliary(0), &c_layer);
        assert_eq!(d_layer.head(&c_layer, &p, &checked).fault, 0);
        assert_ne!(d_layer.head(&c_layer, &p, &empty).fault, 0);
        assert_ne!(d_layer.head(&a, &p, &checked).fault, 0);
        let last = d_layer.head(&c_layer, &p, &checked);
        assert_eq!(last.final_fault(&p, &p), 0);
        assert_ne!(last.final_fault(&other, &other), 0);
    }
}
