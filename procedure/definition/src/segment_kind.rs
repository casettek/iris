//! The segment kinds a procedure is made of.
//!
//! A segment is the stages of a pipeline that share one purpose. Each kind is
//! declared here once: what a use of it gives a later step, and the stage
//! programs it has, each with the inputs it takes from an earlier step and
//! the inputs a run fills in. A shape ([`crate::shape::Shape`]) arranges uses
//! of these kinds, and is held to what they declare.
//!
//! To add a kind: its logic under `phases/`, its stage programs under
//! `raster-stages/`, its native stages under
//! `procedure/pipeline/src/segments/`, and its entry here.

/// What a step of a shape gives the steps after it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gives {
    /// What a prompt's text was built from.
    SourceRecord,
    /// An inference's last check: its claim record and its output.
    LastCheck,
    /// A procedure's result. Nothing in a pipeline takes it.
    Result,
}

impl Gives {
    /// As a refusal names it.
    pub fn name(self) -> &'static str {
        match self {
            Self::SourceRecord => "a source record",
            Self::LastCheck => "an inference's last check",
            Self::Result => "a result",
        }
    }
}

/// One stage program of a segment kind.
#[derive(Debug, PartialEq, Eq)]
pub struct StageProgram {
    /// The stage project's directory name.
    pub name: &'static str,
    /// The inputs it takes from an earlier step, in the order a pipeline
    /// writes them, and what each must be given.
    pub takes: &'static [(&'static str, Gives)],
    /// The inputs a run fills in when it renders its pipeline: the parameter,
    /// and the name the run's file goes by.
    pub run_inputs: &'static [(&'static str, &'static str)],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SegmentKind {
    /// States what the prompt text was built from.
    Source,
    /// One model's inference, from its first bind to its last check. Its
    /// stages are the ones the model's import wrote.
    Inference,
    /// States the procedure's result.
    Result,
}

const SOURCE_PROGRAMS: &[StageProgram] = &[
    // A prompt built from documents a signer vouches for.
    StageProgram {
        name: "prompt-source",
        takes: &[],
        run_inputs: &[
            ("signer", "source_signer"),
            ("recipe", "source_recipe"),
            ("documents", "source_documents"),
        ],
    },
    // A prompt built around an earlier inference's output.
    StageProgram {
        name: "prompt-relay",
        takes: &[("check", Gives::LastCheck)],
        run_inputs: &[("recipe", "relay_recipe")],
    },
];

const RESULT_PROGRAMS: &[StageProgram] = &[
    StageProgram {
        name: "claim-result",
        takes: &[("check", Gives::LastCheck)],
        run_inputs: &[("forms", "result_forms")],
    },
    StageProgram {
        name: "consensus-result",
        takes: &[("first", Gives::LastCheck), ("second", Gives::LastCheck)],
        run_inputs: &[("forms", "result_forms")],
    },
];

/// What an inference's first bind may take from an earlier step. Without it
/// the bind keeps the constant its import gave it.
pub const INFERENCE_TAKES: &[(&str, Gives)] = &[("source", Gives::SourceRecord)];

impl SegmentKind {
    pub const ALL: &'static [SegmentKind] = &[Self::Source, Self::Inference, Self::Result];

    /// The name a shape uses.
    pub fn name(self) -> &'static str {
        match self {
            Self::Source => "source",
            Self::Inference => "inference",
            Self::Result => "result",
        }
    }

    pub fn named(name: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|kind| kind.name() == name)
    }

    /// What a use of this kind gives the steps after it.
    pub fn gives(self) -> Gives {
        match self {
            Self::Source => Gives::SourceRecord,
            Self::Inference => Gives::LastCheck,
            Self::Result => Gives::Result,
        }
    }

    /// The stage programs a shape may name for a use of this kind. None for
    /// an inference, whose programs are its model family's.
    pub fn programs(self) -> &'static [StageProgram] {
        match self {
            Self::Source => SOURCE_PROGRAMS,
            Self::Inference => &[],
            Self::Result => RESULT_PROGRAMS,
        }
    }

    pub fn program(self, name: &str) -> Option<&'static StageProgram> {
        self.programs().iter().find(|program| program.name == name)
    }
}

/// What a procedure's last stage states: one of the result segment's
/// programs, with what the host must know of it to read its output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResultKind {
    /// The code of the form one output is.
    OneOutput,
    /// The code of each of two outputs, and whether they agree.
    Agreement,
}

impl ResultKind {
    /// Every result kind, in the order an output that does not say which it
    /// holds is read as one: their records have no field in common at the top.
    pub const ALL: &'static [ResultKind] = &[ResultKind::Agreement, ResultKind::OneOutput];

    /// The stage program that states this result: the stage project's
    /// directory name.
    pub fn program(self) -> &'static str {
        match self {
            Self::OneOutput => "claim-result",
            Self::Agreement => "consensus-result",
        }
    }

    /// The result kind a stage program states, or `None` for a program that
    /// is not the result segment's.
    pub fn of_program(program: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|result| result.program() == program)
    }

    /// Whether the result says anything of several outputs together.
    pub fn states_agreement(self) -> bool {
        self == Self::Agreement
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_result_program_is_a_result_kind() {
        let programs: Vec<&str> = SegmentKind::Result
            .programs()
            .iter()
            .map(|program| program.name)
            .collect();
        let mut kinds: Vec<&str> = ResultKind::ALL.iter().map(|kind| kind.program()).collect();
        kinds.sort_unstable();
        assert_eq!(programs, kinds);
        for program in &programs {
            assert!(ResultKind::of_program(program).is_some());
        }
        assert_eq!(ResultKind::of_program("prompt-source"), None);
    }

    #[test]
    fn a_segment_kind_is_found_by_its_name() {
        for kind in SegmentKind::ALL {
            assert_eq!(SegmentKind::named(kind.name()), Some(*kind));
        }
        assert_eq!(SegmentKind::named("judgment"), None);
        // An inference's programs are its model family's, so a shape names none.
        assert!(SegmentKind::Inference.programs().is_empty());
        assert!(SegmentKind::Source.program("prompt-source").is_some());
        // The relay program is the source segment's too: it states the same
        // record, from an inference's last check.
        let relay = SegmentKind::Source.program("prompt-relay").unwrap();
        assert_eq!(relay.takes, [("check", Gives::LastCheck)]);
        assert!(SegmentKind::Source.program("claim-result").is_none());
    }
}
