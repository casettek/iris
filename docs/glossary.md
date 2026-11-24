# Glossary

The words iris uses, each with one meaning. Where the code still says
something else, the last column gives it: the docs change first and the code
follows, so this page says which is which until they agree.

## The protocol

| Word | Meaning | In the code today, where it differs |
| --- | --- | --- |
| Procedure | The registered way a judgment is made: the stage programs, the models and the parameters a claim must have run under. A contract judges a claim by the procedure the claim names | |
| Procedure kind | The arrangement a procedure's segments follow. Two exist: `single`, one inference on one model, and `consensus`, the same inference on two models and whether they agree | |
| Specification | The document that defines a procedure | |
| Case | One run of a procedure: what a claim is about | |
| Claim | What a claimer posts about a case: the statement, the checkpoint list and what the procedure requires beside them | |
| Claimer, challenger, defender | Who posts a claim, who disputes one of its stages, and who answers that dispute | |
| Segment | The stages that share one purpose. A procedure has a source segment, one inference segment per model, and a result segment | |
| Stage | One program's place in a chain | |
| Checkpoint | What a claim commits to for one stage: its program, its inputs and its output | |
| Checkpoint list | A claim's checkpoints, one per stage, in order | |
| Checkpoint hashes | One hash per checkpoint of the list, in order (`checkpoints.txt`). A claim commits to its checkpoint list by the root of the tree over them | |
| Text digest | What a run states of its input text before any stage has read the text: its SHA-256, its length, and the span the template's trim keeps. Every prompt range uses it, and the range the text ends in checks it | |
| Challenge period | How long a claim stays open to challenge after it is submitted | |
| Fault-proof period | How long a defender has to answer a challenge | |
| Scenario | One test of the parity gate | |

## Words kept apart

Each of these had two meanings. It now has the one given here.

| Word | Means | Does not mean |
| --- | --- | --- |
| Chain | A Raster chain: programs run in order, each output linked to the next input | Celestia or an EVM network. Those are blockchains. "On chain" stays, for where a contract runs |
| Blockchain | Celestia, or the EVM network the contracts are on | |
| Pipeline | Nothing in iris for now. Raster will take the word for what is a chain today, and iris will follow | A procedure kind |
| Contract | Code on a blockchain | A file format |
| Format | The layout of a file or a record iris writes. [Formats and versions](protocol/formats.md) lists them | |
| Window | A span of trace items, as Raster proves them | A span of time. That is a period |
| Trace | The events one run of a program emits. Raster's word | A claim's checkpoints. That is the checkpoint list |

## The words for what iris is used for

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
| Record | What is posted for a case | Submission |
| Relying party | Who acts on a verdict | Contract |
| Phase | A part of a procedure with one purpose | Segment |

Four more are named and not built:

| Word | Meaning |
| --- | --- |
| Condition | One contract's terms, fixed in advance |
| Finding | A final verdict used as evidence in another case |
| Precedent | Past cases that guide a similar one |
| Review | A case that judges another case |
