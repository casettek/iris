# Glossary

The words IWEBS uses, each with one meaning. Where the code still says
something else, the last column gives it: the docs change first and the code
follows, so this page says which is which until they agree.

## The protocol

| Word | Meaning | In the code today, where it differs |
| --- | --- | --- |
| Adjudication harness | What IWEBS is: it runs a model as the judge of a case, under a procedure the parties agreed to beforehand, and the verdict it gives does not depend on who ran it | |
| Procedure | The way a judgment is made, fixed before any case: the stage programs, the models, the parameters, and what a run must say of its source and result. Parties agree to one by its id. On a blockchain a contract registers it, and judges a claim by the procedure the claim names | |
| Procedure kind | The arrangement a procedure's segments follow. Three exist: `single`, one inference on one model; `consensus`, the same inference on two models and whether they agree; and `relay`, one model's output set in a second model's prompt. Each is a file of `procedures/` | |
| Shape | A procedure kind's steps in order: what its file in `procedures/` lists. Each step is a use of a segment kind, with the earlier step each of its inputs comes from | |
| Specification | The document that defines a procedure | |
| Case | One run of a procedure: the question and the evidence a verdict is of. On a blockchain, what a claim is about | |
| Claim | What a claimer posts about a case: the statement, the checkpoint list and what the procedure requires beside them | |
| Claimer, challenger, defender | Who posts a claim, who disputes one of its stages, and who answers that dispute | |
| Phase | A part of a procedure with one purpose: the source, an inference, the result. A phase's logic is written once, under `phases/`, and every way of running the procedure uses it | |
| Segment | The stages that share one purpose: what a phase becomes in a pipeline. A procedure's pipeline has a source segment, one inference segment per model, and a result segment. Raster's word | |
| Stage | One program's place in a pipeline | |
| Native stage | A stage's code as an ordinary host program runs it. It must give, byte for byte, what the stage's Raster program gives. The native stages are under `procedure/pipeline/src/segments/` | |
| Native run | A procedure run the ordinary way, with nothing shaped for stages: every phase in turn, with inference generating its output. It gives the result as fast as possible, and it is the only run that generates. `iwebs infer` makes one | |
| Pipeline run | A procedure run as its Raster pipeline: stage by stage, each verifying its part of an output that was stated beforehand, with a checkpoint per stage. Its stages run as native stages or as Raster programs. A claim, a challenge and `iwebs run` each make one | |
| Checkpoint | What a claim commits to for one stage: its program, its inputs and its output | |
| Checkpoint list | A claim's checkpoints, one per stage, in order | |
| Checkpoint hashes | One hash per checkpoint of the list, in order (`checkpoints.txt`). A claim commits to its checkpoint list by the root of the tree over them | |
| Text digest | What a run states of its input text before any stage has read the text: its SHA-256, its length, and the span the template's trim keeps. Every prompt range uses it, and the range the text ends in checks it | |
| Procedure id | The hash over a procedure: its stage programs, its models, its parameters and what it requires of a run's source and result. Two parties that name one id have agreed to one procedure. The question, the criteria and the evidence of a case are not in it: they are inputs of a run. A contract registers a procedure under this id | |
| Run record | What the party that made a run hands to anyone who wants to make it again: the procedure id with the procedure, the inputs the run was given and each inference's result, with the backings of that statement. A directory `iwebs run --record` writes and `iwebs record check` reads. It needs no blockchain | |
| Method | A way a verdict is backed for someone who did not make the run. Making the run again needs only the record. One other is built: fault provable, where the run leaves a checkpoint for each stage so that one stage can be proven wrong | |
| Backing | What one method leaves behind what a run record states, besides making the run again. A record names the method of each. One is built: `fault_provable`, the count and root of the run's checkpoints | |
| Challenge period | How long a claim stays open to challenge after it is submitted | |
| Fault-proof period | How long a defender has to answer a challenge | |
| Scenario | One test of the parity gate | |

## Words kept apart

Each of these had two meanings. It now has the one given here.

| Word | Means | Does not mean |
| --- | --- | --- |
| Pipeline | A procedure's stage programs in order, each output linked to the next input. Raster's word. `procedure/pipeline/` holds the run of one | A procedure kind |
| Piped input | An input a stage takes from an earlier stage's output. Raster's word | |
| Chain | A hash chain: each hash taken over the one before it and one more item | A pipeline, which Raster and IWEBS used to call a chain. Nor Celestia or an EVM network: those are blockchains. "On chain" stays, for where a contract runs |
| Blockchain | Celestia, or the EVM network the contracts are on | |
| Fault | What a claim has when a stage of it is wrong, and what a fault proof proves. Raster's word, which used to be fraud | |
| Segment | Stages that share one purpose | A piece of a prompt's text. The tokenizer's code says segment for that, and keeps it |
| Native | Ordinary compiled code, as opposed to a Raster program: the native run, a native stage | Without a blockchain. A native run and a pipeline run are each made with one or without |
| Contract | Code on a blockchain | A file format |
| Format | The layout of a file or a record IWEBS writes. [Formats and versions](protocol/formats.md) lists them | |
| Window | A span of trace items, as Raster proves them | A span of time. That is a period |
| Trace | The events one run of a program emits. Raster's word | A claim's checkpoints. That is the checkpoint list |

## The words for what IWEBS is used for

A second set of words describes a procedure by what it does for the people who
rely on it. They are for this page and for pages written for users. The code
keeps the word on the right.

| Word | Meaning | In the code |
| --- | --- | --- |
| Evidence | What a case is judged on | Source, documents |
| Question | What the judge is asked | Recipe |
| Answers | The verdicts a procedure allows | Forms |
| Verdict | The answer a case reaches | Result |
| Judge | The model, pinned by its id | Model |
| Record | What is posted for a case | Submission, on a blockchain. Run record, with none |
| Relying party | Who acts on a verdict | Contract |

Four more are named and not built:

| Word | Meaning |
| --- | --- |
| Condition | One contract's terms, fixed in advance |
| Finding | A final verdict used as evidence in another case |
| Precedent | Past cases that guide a similar one |
| Review | A case that judges another case |
