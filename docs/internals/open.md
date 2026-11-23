# What is open

Every known gap, in one place. Nothing on another page is closed by being
missing here: if a gap is found, it is added to this list.

The first group is left open on purpose. The second and third are not
decided. The last is what Raster would have to change.

## Left open on purpose

- **What follows a claim.** Nothing reads a void claim, an unanswered
  challenge or a dismissal. A claim with an unanswered challenge and a claim
  with none are held the same way. The clocks that close a claim or a
  challenge, bonds, and what an unchallenged claim becomes are not built.
  `submitFaultProof` reads no clock: the fault-proof window is used only for
  how long a challenge's trace must stay retrievable.
- **Several challenges of one claim.** Each is admitted and dismissed alone.
  How they are settled together belongs with the clocks and bonds.
- **The trusted hosts.** The attester signs what each posted blob holds, the
  relayer posts Celestia's block roots, and a source's signer signs that a
  server answered a request with a document
  ([what replaces each](overview.md#what-is-trusted-today)). The signer today
  is the host itself, and which documents to fetch is its choice.
- **The proof that replaces the attester.** The stand-in runs host code: the
  source step, the checkpoint package check, the reading of records, and
  Raster's trace header. A receipt needs that code in a crate a guest can
  build. The source step already is one, in `source-core`.
- **Short stages.** A stage whose trace is shorter than a replay window
  cannot be disputed ([the issue](../issues/short-stage-fraud-window.md)).
  The claimer chooses the input, so it can make one occur. Three cases are
  short whatever the claimer does: the result stage, which is three steps and
  one per form; a source of a few hundred bytes; and, on small prompts, the
  bind and check stages. The contract does not rest on the result stage: it
  takes the result on the submitted text itself, and for a consensus the
  agreement on the two. A result step that grows past what a contract can
  recompute has to be a stage long enough to replay.
- **Two commitments to one output.** A checkpoint holds the hash of a
  stage's output bytes, which a dispute is about, and its structural
  commitment, which the input comparison uses. Raster's proof guest checks
  the output against both at the end of a stage and publishes only the byte
  hash. So a claimer can list a true byte hash beside the structural
  commitment of a made-up value, feed that value to a later stage, and pass
  the comparison. Closing it needs a proof: either Raster publishes the
  structural commitment in the window's journal, or a proof of this
  repository's own ties the two.

## Open in the protocol

- **A stage that cannot run on what it is given.** A replay shows a lie only
  when the stage finishes. A claimer can give a stage a value it is allowed to
  name but cannot run on: a run input of the wrong shape, or a held value
  that stops the program. The stages are written to state a fault and finish,
  and the tests replay each kind of misplaced value (another layer's
  weights, another range's marks, another stage's output) on the small model
  and on gemma E2B. No test gives every stage every value. As Raster's stage
  challenge is designed, a trace that does not reach the program's end is
  refuted as a misstated one, so a challenger of such a stage has nothing to
  win with. Read from the design, not run.
- **A true trace with a false label.** Raster has a second defence: the
  challenger's trace is an honest run, and ends in another output than the
  challenger stated for it (`TerminalMatch`). The contract does not take it,
  so such a challenge cannot be dismissed on chain. It needs the stated
  output compared with the one the window ends in.
- **Raster's admission of a challenge.** `iris challenge check` asks Raster
  whether it admits the challenge (`verify_challenge`). The contract makes
  the trace-shape checks of that admission and not the rest, and the attester
  signs only that the posted body is a well-formed trace with the header the
  challenge names.
- **Reading an output by its index.** The stand-in reads a record out of an
  output with the output's index, given by the claimer, after checking that
  the index and the payload have the same root. A proof program should not
  rest on an index the prover chose: Raster's selection proofs tie a selected
  value to the payload's own root, and the record can be compared by encoding
  the submitted one, with no decoding at all.
- **The signer's key is a run input.** The `source` stage checks attestations
  against the key the run gives it and states that key in its record. No
  stage input pins the key. The contract compares the key in the submitted
  record with the one it requires, which holds as long as the `source`
  checkpoint is true.
- **Upgrades.** A deployment takes proofs of one build of Raster and one
  release of RISC Zero's circuits.

## Open in the tools

- **A claim whose list of stages is not the chain's.** `challenge build`
  runs a claim whose every stage is one the chain of its run has, each named
  once. It refuses a claim that repeats a stage name or names one the chain
  lacks. The stages' schedule checks refuse such a run, so its final record
  is not valid, but no command shows that.
- **No fault proof of a full-size stage.** The proofs made so far are of a
  small fixture and of the small test case. How long a window of a real stage
  takes to prove is not measured.
- **`iris fault prove` and the contract ask slightly different things.**
  `fault prove` holds its proof to the checkpoint of the claim bundle it is
  given, which in a staged dispute is the honest claim's, while the challenge
  on chain is of the lying claim's checkpoint. The two have the same program
  and inputs, which is all a fault proof names, so the proof serves both.
  `fault prove` also does not ask which authorization program the journal
  names; `iris evm fault-proof` and the contract do.
- **The forged verdict is built for one inference.** `iris test
  corrupt-claim --fault verdict` refuses a consensus claim.
- **A consensus claim posts its text once per inference.** Each model cuts
  the text into its own slices, and the slices are what is posted.
- **`claim build` checks its own consensus claim against pins made at that
  moment.** A contract's are the file it was given. The two differ only if a
  stage was re-locked in between.
- **Formatting stops at the stage code.** `just fmt` leaves the stage crates
  and what they are built from, because formatting them changes every program
  id ([testing](../workflows/testing.md#formatting)). They are formatted at
  the next re-lock.
- **Nothing runs automatically.** There is no CI.

## What Raster would have to change

[`docs/issues/`](../issues/README.md) holds the gaps that are Raster's to
close, each with what iris does in the meantime.
