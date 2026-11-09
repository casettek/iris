//! The stage schedule: which stages a claim runs, and what each one follows.
//! These are rules only. A stage applies them in a tile to committed inputs.
use crate::{Fault, Statement};
use serde::{Deserialize, Serialize};

/// Most ranges of either kind a schedule may name.
pub const MAX_RANGES: u32 = 1 << 20;
/// Most merge batches a range may run after its seed.
pub const MAX_MERGE_BATCHES: u32 = 1 << 20;

/// The counts that, with the model's layer count, fix every stage and its place.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageSchedule {
    pub prompt_ranges: u32,
    /// Merge batches each prompt range runs after its seed.
    pub merge_batches: u32,
    pub claimed_ranges: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Operation {
    Bind,
    Seed,
    Merge,
    Prepare,
    Embed,
    Auxiliary,
    Layer,
    Check,
    Finalize,
}

/// One stage of a schedule. Coordinates an operation does not use are zero,
/// so a step has exactly one encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Step {
    pub operation: Operation,
    /// `range` counts claimed ranges rather than prompt ranges.
    pub claimed: bool,
    pub range: u32,
    pub layer: u32,
    pub batch: u32,
}

impl Step {
    const fn new(operation: Operation, claimed: bool, range: u32, layer: u32, batch: u32) -> Self {
        Self {
            operation,
            claimed,
            range,
            layer,
            batch,
        }
    }
    pub const fn bind() -> Self {
        Self::new(Operation::Bind, false, 0, 0, 0)
    }
    pub const fn seed(range: u32) -> Self {
        Self::new(Operation::Seed, false, range, 0, 0)
    }
    pub const fn merge(range: u32, batch: u32) -> Self {
        Self::new(Operation::Merge, false, range, 0, batch)
    }
    pub const fn prepare(range: u32) -> Self {
        Self::new(Operation::Prepare, false, range, 0, 0)
    }
    pub const fn embed(claimed: bool, range: u32) -> Self {
        Self::new(Operation::Embed, claimed, range, 0, 0)
    }
    pub const fn auxiliary(claimed: bool, range: u32, layer: u32) -> Self {
        Self::new(Operation::Auxiliary, claimed, range, layer, 0)
    }
    pub const fn layer(claimed: bool, range: u32, layer: u32) -> Self {
        Self::new(Operation::Layer, claimed, range, layer, 0)
    }
    pub const fn check(range: u32) -> Self {
        Self::new(Operation::Check, true, range, 0, 0)
    }
    pub const fn finalize() -> Self {
        Self::new(Operation::Finalize, false, 0, 0, 0)
    }
}

/// An input a stage takes from an earlier stage, named as its `main` names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Input {
    Bound,
    Previous,
    Pieces,
    Prompt,
    Embedded,
    Activations,
    PriorKv,
    Ple,
    Prior,
    Edge,
}

/// What a schedule puts on an input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Step(Step),
    /// The empty value the first stage of its kind starts from.
    Origin,
}

impl StageSchedule {
    /// Checks the counts against their limits and the statement's token budget.
    /// The prompt's two counts depend on tokenization, so only stages can hold
    /// a run to them.
    pub fn validate(&self, statement: &Statement) -> Result<(), Fault> {
        let within = |count: u32, max: u32| (1..=max).contains(&count);
        if !within(self.prompt_ranges, MAX_RANGES)
            || !within(self.merge_batches, MAX_MERGE_BATCHES)
            || !within(self.claimed_ranges, MAX_RANGES)
            || statement.range_size == 0
            || self.claimed_ranges > statement.max_new_tokens.div_ceil(statement.range_size)
        {
            return Err(Fault::Schedule);
        }
        Ok(())
    }

    fn ranges(&self, claimed: bool) -> u32 {
        if claimed {
            self.claimed_ranges
        } else {
            self.prompt_ranges
        }
    }

    /// The range whose stages come directly before this one's: the prompt's
    /// last range precedes the first claimed one, and nothing precedes the
    /// prompt's first.
    fn range_before(&self, claimed: bool, range: u32) -> Option<(bool, u32)> {
        match range.checked_sub(1) {
            Some(range) => Some((claimed, range)),
            None if claimed => Some((false, self.prompt_ranges.checked_sub(1)?)),
            None => None,
        }
    }

    /// Whether the schedule runs this step, for a model of `layers` layers.
    pub fn contains(&self, step: &Step, layers: u32) -> bool {
        let Step {
            operation,
            claimed,
            range,
            layer,
            batch,
        } = *step;
        let in_range = range < self.ranges(claimed);
        match operation {
            Operation::Bind | Operation::Finalize => *step == Step::new(operation, false, 0, 0, 0),
            Operation::Seed | Operation::Prepare => {
                !claimed && in_range && layer == 0 && batch == 0
            }
            Operation::Merge => !claimed && in_range && layer == 0 && batch < self.merge_batches,
            Operation::Embed => in_range && layer == 0 && batch == 0,
            Operation::Auxiliary | Operation::Layer => in_range && layer < layers && batch == 0,
            Operation::Check => claimed && in_range && layer == 0 && batch == 0,
        }
    }

    /// The step, or origin, that must have produced `input` of `step`. `None`
    /// when the schedule does not run the step or the step takes no such input.
    pub fn before(&self, step: &Step, input: Input, layers: u32) -> Option<Source> {
        if !self.contains(step, layers) {
            return None;
        }
        let (claimed, range, layer) = (step.claimed, step.range, step.layer);
        let at = Source::Step;
        Some(match (step.operation, input) {
            (Operation::Seed, Input::Bound) => at(Step::bind()),
            (Operation::Seed, Input::Previous) => {
                at(range.checked_sub(1).map_or(Step::bind(), Step::prepare))
            }
            (Operation::Merge, Input::Pieces) => at(step
                .batch
                .checked_sub(1)
                .map_or(Step::seed(range), |batch| Step::merge(range, batch))),
            (Operation::Prepare, Input::Pieces) => {
                at(Step::merge(range, self.merge_batches.checked_sub(1)?))
            }
            (Operation::Embed, Input::Prompt) if claimed => Source::Origin,
            (Operation::Embed, Input::Prompt) => at(Step::prepare(range)),
            (Operation::Embed, Input::Previous) => self
                .range_before(claimed, range)
                .map_or(Source::Origin, |(claimed, range)| {
                    at(Step::embed(claimed, range))
                }),
            (Operation::Auxiliary, Input::Embedded) => at(Step::embed(claimed, range)),
            (Operation::Layer, Input::Activations) => at(layer
                .checked_sub(1)
                .map_or(Step::embed(claimed, range), |layer| {
                    Step::layer(claimed, range, layer)
                })),
            (Operation::Layer, Input::PriorKv) => self
                .range_before(claimed, range)
                .map_or(Source::Origin, |(claimed, range)| {
                    at(Step::layer(claimed, range, layer))
                }),
            (Operation::Layer, Input::Ple) => at(Step::auxiliary(claimed, range, layer)),
            (Operation::Check, Input::Activations) => {
                at(Step::layer(true, range, layers.checked_sub(1)?))
            }
            (Operation::Check, Input::Previous) => {
                let (claimed, range) = self.range_before(true, range)?;
                at(Step::layer(claimed, range, layers.checked_sub(1)?))
            }
            (Operation::Check, Input::Prior) => range
                .checked_sub(1)
                .map_or(Source::Origin, |range| at(Step::check(range))),
            (Operation::Check | Operation::Finalize, Input::Prompt) => {
                at(Step::prepare(self.prompt_ranges.checked_sub(1)?))
            }
            (Operation::Finalize, Input::Edge) => {
                at(Step::check(self.claimed_ranges.checked_sub(1)?))
            }
            _ => return None,
        })
    }

    /// The step a layer borrows its cache from, when the model gives it a donor.
    pub fn donor(&self, step: &Step, donor_layer: u32, layers: u32) -> Option<Step> {
        (step.operation == Operation::Layer
            && self.contains(step, layers)
            && donor_layer < step.layer)
            .then(|| Step::layer(step.claimed, step.range, donor_layer))
    }

    /// Every step in the order the chain runs them.
    pub fn steps(&self, layers: u32) -> impl Iterator<Item = Step> {
        let schedule = *self;
        let prompt = (0..schedule.prompt_ranges).flat_map(move |range| {
            core::iter::once(Step::seed(range))
                .chain((0..schedule.merge_batches).map(move |batch| Step::merge(range, batch)))
                .chain([Step::prepare(range), Step::embed(false, range)])
        });
        let claimed = (0..schedule.claimed_ranges).map(|range| Step::embed(true, range));
        let ranges = move || {
            (0..schedule.prompt_ranges)
                .map(|range| (false, range))
                .chain((0..schedule.claimed_ranges).map(|range| (true, range)))
        };
        let layers = (0..layers).flat_map(move |layer| {
            ranges()
                .map(move |(claimed, range)| Step::auxiliary(claimed, range, layer))
                .chain(ranges().map(move |(claimed, range)| Step::layer(claimed, range, layer)))
        });
        let checks = (0..schedule.claimed_ranges).map(Step::check);
        core::iter::once(Step::bind())
            .chain(prompt)
            .chain(claimed)
            .chain(layers)
            .chain(checks)
            .chain(core::iter::once(Step::finalize()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    const LAYERS: u32 = 2;
    const INPUTS: [Input; 10] = [
        Input::Bound,
        Input::Previous,
        Input::Pieces,
        Input::Prompt,
        Input::Embedded,
        Input::Activations,
        Input::PriorKv,
        Input::Ple,
        Input::Prior,
        Input::Edge,
    ];
    const OPERATIONS: [Operation; 9] = [
        Operation::Bind,
        Operation::Seed,
        Operation::Merge,
        Operation::Prepare,
        Operation::Embed,
        Operation::Auxiliary,
        Operation::Layer,
        Operation::Check,
        Operation::Finalize,
    ];

    fn schedule() -> StageSchedule {
        StageSchedule {
            prompt_ranges: 2,
            merge_batches: 2,
            claimed_ranges: 2,
        }
    }
    fn statement(max_new_tokens: u32, range_size: u32) -> Statement {
        Statement {
            input_text: [1; 32],
            model: [2; 32],
            raw_prompt: false,
            max_new_tokens,
            range_size,
        }
    }
    fn sequence() -> Vec<Step> {
        schedule().steps(LAYERS).collect()
    }

    #[test]
    fn a_schedule_has_one_step_sequence() {
        let mut expected = Vec::from([
            Step::bind(),
            Step::seed(0),
            Step::merge(0, 0),
            Step::merge(0, 1),
            Step::prepare(0),
            Step::embed(false, 0),
            Step::seed(1),
            Step::merge(1, 0),
            Step::merge(1, 1),
            Step::prepare(1),
            Step::embed(false, 1),
            Step::embed(true, 0),
            Step::embed(true, 1),
        ]);
        let ranges = [(false, 0), (false, 1), (true, 0), (true, 1)];
        for layer in 0..LAYERS {
            expected.extend(ranges.map(|(claimed, range)| Step::auxiliary(claimed, range, layer)));
            expected.extend(ranges.map(|(claimed, range)| Step::layer(claimed, range, layer)));
        }
        expected.extend([Step::check(0), Step::check(1), Step::finalize()]);
        let steps = sequence();
        assert_eq!(steps, expected);
        for (index, step) in steps.iter().enumerate() {
            assert!(!steps[..index].contains(step), "{step:?} repeats");
        }
    }

    #[test]
    fn only_the_sequence_is_in_the_schedule() {
        let steps = sequence();
        let mut seen = 0;
        for operation in OPERATIONS {
            for claimed in [false, true] {
                for range in 0..4 {
                    for layer in 0..4 {
                        for batch in 0..4 {
                            let step = Step::new(operation, claimed, range, layer, batch);
                            let contained = schedule().contains(&step, LAYERS);
                            assert_eq!(contained, steps.contains(&step), "{step:?}");
                            seen += usize::from(contained);
                        }
                    }
                }
            }
        }
        assert_eq!(seen, steps.len());
    }

    #[test]
    fn every_input_comes_from_an_earlier_step_and_no_step_is_unused() {
        let steps = sequence();
        let mut used = Vec::new();
        for (index, step) in steps.iter().enumerate() {
            for input in INPUTS {
                if let Some(Source::Step(source)) = schedule().before(step, input, LAYERS) {
                    assert!(steps[..index].contains(&source), "{step:?} {input:?}");
                    used.push(source);
                }
            }
            for donor_layer in 0..4 {
                match schedule().donor(step, donor_layer, LAYERS) {
                    Some(source) => assert!(steps[..index].contains(&source), "{step:?} donor"),
                    None => assert!(
                        step.operation != Operation::Layer || donor_layer >= step.layer,
                        "{step:?} donor"
                    ),
                }
            }
        }
        for step in &steps[..steps.len() - 1] {
            assert!(used.contains(step), "{step:?} feeds nothing");
        }
    }

    #[test]
    fn inputs_follow_the_chain_wiring() {
        let s = schedule();
        let before = |step, input| s.before(&step, input, LAYERS);
        let at = |step| Some(Source::Step(step));
        assert_eq!(before(Step::seed(0), Input::Previous), at(Step::bind()));
        assert_eq!(before(Step::seed(1), Input::Previous), at(Step::prepare(0)));
        assert_eq!(before(Step::seed(1), Input::Bound), at(Step::bind()));
        assert_eq!(before(Step::merge(1, 0), Input::Pieces), at(Step::seed(1)));
        assert_eq!(
            before(Step::merge(1, 1), Input::Pieces),
            at(Step::merge(1, 0))
        );
        assert_eq!(
            before(Step::prepare(1), Input::Pieces),
            at(Step::merge(1, 1))
        );
        assert_eq!(
            before(Step::embed(false, 0), Input::Previous),
            Some(Source::Origin)
        );
        assert_eq!(
            before(Step::embed(false, 1), Input::Prompt),
            at(Step::prepare(1))
        );
        assert_eq!(
            before(Step::embed(true, 0), Input::Prompt),
            Some(Source::Origin)
        );
        assert_eq!(
            before(Step::embed(true, 0), Input::Previous),
            at(Step::embed(false, 1))
        );
        assert_eq!(
            before(Step::embed(true, 1), Input::Previous),
            at(Step::embed(true, 0))
        );
        assert_eq!(
            before(Step::auxiliary(true, 1, 1), Input::Embedded),
            at(Step::embed(true, 1))
        );
        assert_eq!(
            before(Step::layer(false, 1, 0), Input::Activations),
            at(Step::embed(false, 1))
        );
        assert_eq!(
            before(Step::layer(false, 1, 1), Input::Activations),
            at(Step::layer(false, 1, 0))
        );
        assert_eq!(
            before(Step::layer(false, 0, 1), Input::PriorKv),
            Some(Source::Origin)
        );
        assert_eq!(
            before(Step::layer(true, 0, 1), Input::PriorKv),
            at(Step::layer(false, 1, 1))
        );
        assert_eq!(
            before(Step::layer(true, 1, 1), Input::Ple),
            at(Step::auxiliary(true, 1, 1))
        );
        assert_eq!(
            s.donor(&Step::layer(true, 1, 1), 0, LAYERS),
            Some(Step::layer(true, 1, 0))
        );
        assert_eq!(
            before(Step::check(0), Input::Activations),
            at(Step::layer(true, 0, 1))
        );
        assert_eq!(
            before(Step::check(0), Input::Previous),
            at(Step::layer(false, 1, 1))
        );
        assert_eq!(
            before(Step::check(1), Input::Previous),
            at(Step::layer(true, 0, 1))
        );
        assert_eq!(before(Step::check(0), Input::Prior), Some(Source::Origin));
        assert_eq!(before(Step::check(1), Input::Prior), at(Step::check(0)));
        assert_eq!(before(Step::check(1), Input::Prompt), at(Step::prepare(1)));
        assert_eq!(before(Step::finalize(), Input::Edge), at(Step::check(1)));
        assert_eq!(
            before(Step::finalize(), Input::Prompt),
            at(Step::prepare(1))
        );
        // An input the step does not take, and a step the schedule does not run.
        assert_eq!(before(Step::bind(), Input::Previous), None);
        assert_eq!(before(Step::prepare(0), Input::Edge), None);
        assert_eq!(before(Step::seed(2), Input::Previous), None);
    }

    #[test]
    fn counts_are_held_to_their_limits_and_the_budget() {
        let s = statement(8, 4);
        assert_eq!(schedule().validate(&s), Ok(()));
        let with = |edit: fn(&mut StageSchedule)| {
            let mut changed = schedule();
            edit(&mut changed);
            changed.validate(&statement(u32::MAX, 1))
        };
        assert_eq!(with(|s| s.prompt_ranges = 0), Err(Fault::Schedule));
        assert_eq!(with(|s| s.merge_batches = 0), Err(Fault::Schedule));
        assert_eq!(with(|s| s.claimed_ranges = 0), Err(Fault::Schedule));
        assert_eq!(with(|s| s.prompt_ranges = MAX_RANGES), Ok(()));
        assert_eq!(with(|s| s.merge_batches = MAX_MERGE_BATCHES), Ok(()));
        assert_eq!(with(|s| s.claimed_ranges = MAX_RANGES), Ok(()));
        assert_eq!(
            with(|s| s.prompt_ranges = MAX_RANGES + 1),
            Err(Fault::Schedule)
        );
        assert_eq!(
            with(|s| s.merge_batches = MAX_MERGE_BATCHES + 1),
            Err(Fault::Schedule)
        );
        assert_eq!(
            with(|s| s.claimed_ranges = MAX_RANGES + 1),
            Err(Fault::Schedule)
        );
        // Two ranges of four cover a budget of five to eight tokens, never four.
        assert_eq!(schedule().validate(&statement(5, 4)), Ok(()));
        assert_eq!(schedule().validate(&statement(4, 4)), Err(Fault::Schedule));
        assert_eq!(schedule().validate(&statement(8, 0)), Err(Fault::Schedule));
        assert_eq!(StageSchedule::default().validate(&s), Err(Fault::Schedule));
    }

    #[test]
    fn extreme_counts_and_coordinates_do_not_overflow() {
        let full = StageSchedule {
            prompt_ranges: u32::MAX,
            merge_batches: u32::MAX,
            claimed_ranges: u32::MAX,
        };
        let empty = StageSchedule::default();
        let edge = [0, 1, u32::MAX - 1, u32::MAX];
        for schedule in [full, empty] {
            for layers in [0, u32::MAX] {
                for operation in OPERATIONS {
                    for claimed in [false, true] {
                        for coordinate in edge {
                            let step =
                                Step::new(operation, claimed, coordinate, coordinate, coordinate);
                            let _ = schedule.contains(&step, layers);
                            let _ = schedule.donor(&step, coordinate, layers);
                            for input in INPUTS {
                                let _ = schedule.before(&step, input, layers);
                            }
                        }
                    }
                }
            }
        }
        // With nothing scheduled, the two fixed steps name no predecessor.
        assert!(empty.contains(&Step::finalize(), 0));
        assert_eq!(empty.before(&Step::finalize(), Input::Edge, 0), None);
        assert_eq!(empty.before(&Step::finalize(), Input::Prompt, 0), None);
        assert_eq!(empty.steps(0).count(), 2);
    }
}
