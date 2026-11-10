# Checking claimed prompt tokens instead of recomputing them

Implemented November 10, 2026. The stage is `raster-stages/prompt-check`. It replaced
`prompt-seed`, `prompt-merge` and `prompt-prepare`, which have been removed.
[What was built](#what-was-built) lists where the result differs from the plan below. The
soundness argument in [Why the pair check is enough](#why-the-pair-check-is-enough) is still
the part to review first.

## Goal

**Give the pipeline the prompt's tokenization as an input and have one stage per range
check it, in place of the stages that recompute it by replaying every merge.**

The chain already treats the output this way: the claimed output ids are an input, and
`sequence-check` verifies them. This proposal does the same for the prompt. What the claim
supplies is where each token starts; the stage looks the ids up itself.

It is about the number of stages and the cost of a challenge. It is not needed for
time-to-claim; the native tokenizer fix of the same day covers that without a stage change.

## What does not change

- The challenge: rerun honestly, find the first mismatching checkpoint, replay that stage.
- `prompt_bind`. It still hashes the original text, renders the template and splits the
  result into pieces and segments. The text hash keeps coming from there.
- Everything from `embed_p{j}` onward.
- A wrong claim never aborts a stage. It publishes a fault, and the final record rejects
  the claim.
- Raster.

## Today

For each prompt range the chain runs `prompt_seed`, then `B` merge batches of 64 rounds,
then `prompt_prepare`. A merge round scans every adjacent pair in the range, looks each one
up in the merge table, and applies the pair with the lowest rank.

On the 1,209-token test prompt that is 3 ranges of 32 stages each: 96 of the chain's 385
stages. A range starts at about 1,900 pieces, so a round is about 1,900 authenticated
lookups, and a range needs about 1,500 rounds.

## The design

### The token boundaries become an input

Run preparation already tokenizes the prompt natively to choose the ranges. It would write,
for each range `j`, a committed input `marks_{j}`: one number per piece of the range, 1
where the piece starts a token and 0 elsewhere.

The claim supplies no ids and no text. The stage looks each token's id up in the
vocabulary, as `prompt_prepare` does today, so a claim cannot name a wrong id. The marks
are plain numbers, so nothing the claim supplies has a size it controls except the length
of the list, which the stage only iterates.

### One stage per range checks them

`prompt_check_p{j}` (name open) takes `prompt_bind`'s pieces, the previous range's context
and `marks_{j}`. The marks cover consecutive pieces, starting where the previous range
ended, and cut them into tokens. For each token the stage requires:

1. **It lies inside one segment.** Pieces of different segments never merge.
2. **With a left neighbour in its segment, the pair run gives back the two tokens.** The
   left neighbour is the token before it, which for the first token of a range is the last
   token of the previous range. Run the ranked merge on the two tokens' pieces alone. The
   result must be exactly two pieces: the first made of exactly the neighbour's pieces, the
   second of exactly this token's.
3. **With no left neighbour in its segment, it merges to one piece** when its pieces are
   run alone.
4. **The resulting text is a vocabulary entry.** Its id is the token's id.

Every neighbouring pair inside a segment is therefore run once, including pairs that
straddle two ranges, and every token is covered by at least one run. The stage publishes
the range's ids and the context, as `prompt_prepare` does today.

A failed requirement records a fault in `PromptContext.fault` and the stage finishes. Two
new fault codes would cover it: tokens that do not fit the pieces, and tokens that are not
the merge's result.

### What this removes

- `prompt-seed`, `prompt-merge` and `prompt-prepare` are replaced by one program.
- The merge batch count `B` leaves the stage schedule, which becomes prompt ranges and
  claimed ranges.
- Ranges become any run of whole tokens up to the range size. The rule that a range must
  end where no merge can cross, the boundary table behind it, and the failure when one
  stretch is longer than a range are no longer needed.

## Why the pair check is enough

Checking that the tokens spell the text is not enough. With rules `a+b` at rank 1 and
`b+c` at rank 2, the text `abc` merges to `[ab, c]`, but `[a, bc]` also spells it, uses
only vocabulary tokens and has no pair left to merge. The check has to confirm the merge
order, and the claim below is that the pair run does.

### What the argument assumes about the merge

Each holds in the current stage code (`prompt-merge`'s `scan_merge_rules`,
`finish_ranked_step` and `apply_ranked_merge`):

- **A1.** A merge replaces two adjacent pieces of the same segment with one piece.
- **A2.** Each round applies the available pair with the lowest rank, and among pairs of
  equal rank the leftmost.
- **A3.** Whether two pieces can merge, and at what rank, depends only on their two texts.
- **A4.** The run ends when no pair is available.

It does not assume that a merged piece's text is the two texts joined.

### Terms

Fix one segment and the claimed tokens in it. Each token's pieces are its **group**. A
merge is **internal** if both pieces are in one group and **crossing** otherwise. The
**full run** is the ranked merge over the whole segment, which is what the chain computes
today. A group's **solo run** is the ranked merge over that group's pieces alone. The
**pair run** for two neighbouring groups is the ranked merge over their pieces alone.

### Claim 1: if every pair check passes, the full run produces exactly the claimed tokens

1. A crossing merge joins pieces of two groups for good, since merges never split. A pair
   run that ends as two pieces split at the group boundary therefore had no crossing merge,
   and each of its two groups merged down to its token by internal merges.
2. While a run has made no crossing merge, each group moves through the states of its own
   solo run, in order. The run picks the best pair overall; if that pair is inside a group,
   it is also the best pair inside that group, which is the solo run's choice (A2, A3).
3. Suppose the full run makes a crossing merge, and take the first one, between groups `i`
   and `i+1`, when they are in solo states `a` and `b`. The crossing pair is then the best
   pair in the whole segment, so it beats the best pair inside `a` and inside `b`.
4. The pair run for `i` and `i+1` reaches the same states `a` and `b`. Whenever the full
   run advances one of the two groups, the pair it applies beats the other group's best
   pair and the crossing pair. The pair run chooses among those same three candidates under
   the same rule, so it makes the same move. Steps the full run spends on other groups do
   not change these two.
5. At `a` and `b` the pair run faces the same three candidates as in step 3, so it applies
   the crossing pair. That contradicts step 1. So the full run makes no crossing merge.
6. With no crossing merge, every group follows its solo run to the end (step 2), which is
   its token (step 1), and the full run ends with exactly those tokens.

The full run is deterministic, so no other tokenization can pass.

### Claim 2: the true tokenization always passes

Take the groups to be the full run's own final tokens. Every merge of the full run is then
internal. Step 4 above shows the pair run follows the full run on its two groups, so both
merge down to their tokens. At the end the two tokens are adjacent in the full run too,
and the full run stopped, so no rule joins them (A3, A4) and the pair run stops as well.

### What the argument leaves to check

- It is an argument, not a machine-checked proof, and it is mine. It should be read
  critically, especially steps 2 and 4 where ties in rank are decided by position.
- A neighbour is a token whose last piece is in the same segment. The first native version
  looked at the neighbour's first piece, which differs only when the previous range's last
  token is itself malformed; modelling the stage exposed it.
- A token that starts its segment has no left neighbour, which is why requirement 3 exists.
- The stage must tie the result to the claimed token boundary, not only to the two texts.
  Otherwise wrong marks could pass when the texts happen to match.
- Byte fallback, unknown-token fusing and special tokens are decided in `prompt_bind`,
  before any merge. The argument starts from its pieces and segments and does not cover
  that step, which this proposal leaves as it is.

## Tests of the rule

The check is written natively in `inference-kernels`
([prompt_check.rs](../../crates/inference/kernels/src/kernels/prompt_check.rs)). No stage
uses it. Each test below compares it with the current full merge.

| Test | Scope | Result |
| --- | --- | --- |
| Random merge tables | 400 tables over three letters, every string up to 7 letters, every grouping, with and without a segment break: 134,367,600 groupings | A grouping passes exactly when it is the full merge's |
| Same, wrong but plausible | The 994,168 of those groupings that spell vocabulary tokens with no pair left to merge | All refused |
| Production Gemma table, agreement | 12 pinned reference cases and 3,000 texts built by joining vocabulary entries, whole and cut into two ranges at every token | The full merge's tokens always pass |
| Production Gemma table, exhaustion | 20,000 short built texts, every grouping: 14,507,984 groupings | A grouping passes exactly when it is the full merge's |
| Same, wrong but plausible | The 27,269 of those groupings that spell vocabulary tokens with no pair left to merge | All refused |
| Wrong id | The right grouping with one id changed, 20,000 times | All refused |
| Ranges | On random tables, a wrong grouping cut into two ranges at any point | One of the two ranges always fails |
| Staged rounds, random tables | The stage's rounds as modelled natively, on every grouping of every string up to 5 letters over 30 tables, whole and as a second range at every cut | Accepts and refuses exactly as the pair check does |
| Staged rounds, production table | The same model on all 14,507,984 groupings above | Accepts and refuses exactly as the pair check does |

Half of the random tables have their ranks shuffled, so a rule can outrank the rules that
build its parts. The production table has 514,906 rules for 236,339 distinct results, so
most tokens can be formed from more than one pair of pieces.

What the tests do not cover: texts longer than 12 pieces are not exhausted, only checked
for agreement; every tested rule's result is its two texts joined, as import enforces; and
the tests start from `prompt_bind`'s pieces, like the argument.

Run them with:

```sh
cargo test -p inference-kernels prompt_check
cargo test --release -p inference-kernels -- --ignored exactly_the_merge_grouping_passes_exhaustive
cargo test --release -p staged-infer --test prompt_check -- --ignored --nocapture
```

## The stage

Designed against the Raster authoring rules and modelled natively
(`prompt_check::rounds`). Not authored: no tile or sequence exists, and the check ladder
has not been run. The first authoring attempt may force changes.

### What the rules allow

- A sequence cannot loop on data. The recur reference gives one honest way to run a merge
  to its end: unroll a fixed number of rounds and finish with a check that nothing is left
  to merge. `prompt-merge` does this today with 64 rounds.
- A loop driven by a list of merge steps from the claim is the pattern the reference
  forbids: the trip count would be an input.
- A tile cannot look anything up, and a sequence cannot select a range whose bounds are
  computed. So a pair of tokens cannot be handed to a tile as a window, and each merge
  lookup has to be one step of a recur over pieces.

### Windows are marks on the pieces, not separate lists

Number the range's tokens in order. Two passes pair them differently:

| Pass | Windows |
| --- | --- |
| A | tokens (0, 1), (2, 3), (4, 5), ... |
| B | token 0 alone, then (1, 2), (3, 4), ... |

Every neighbouring pair shares a window in exactly one pass. Within a pass the pieces stay
one list in their original order, and a piece may merge only with a piece of the same
window and segment. Merging a window in isolation is the pair run of the rule, so the two
passes together run every pair once. No piece is duplicated and no window is ever built.

### The steps

1. **Count the marks.** A fold over `marks_{j}` gives the number of pieces the range covers
   and the number of tokens it claims.
2. **Copy and number the pieces.** A recur over `prompt_bind`'s pieces copies that many,
   starting at the previous range's last token when it is a neighbour, and reads each
   piece's mark to give it a token number.
3. **Pass A, then pass B, 64 rounds each.** A round is two recurs over the pieces:
   - *Scan.* One merge-table lookup per piece, for the pair it forms with the piece before
     it, as `scan_pair` does today. The best pair of each window is appended to a list as
     the window ends.
   - *Apply.* Each piece reads its window's best pair from that list and is copied, merged
     or dropped. Merging two pieces of different tokens records a fault.
4. **Check the end state**, once per pass. One more scan must find no pair to merge, and
   every token must be exactly one piece.
5. **Look up the ids** of pass A's pieces in the vocabulary and publish them with the
   context, leaving out the previous range's token.

Loop state stays small throughout: the previous piece's text, a few counters, and one best
candidate. Everything that grows is a recur output.

### Nothing a claim supplies can abort the stage

- The marks are numbers. A wrong mark changes the token numbering and the check fails.
- The only indexed read of a claim list is a piece reading its mark. The stage copies at
  most as many pieces as it counted marks, so that index is below the list's length by
  construction. With no marks, no piece is copied and nothing is read.
- The list of best pairs is built by the scan with one entry per window, and the apply
  step reads it by window number.
- Marks that run past the prompt, a token spanning two segments, more tokens than the
  range size and an empty range are recorded as faults.

### The round count is a limit on token length

A window of two tokens needs two merges fewer than it has pieces. 64 rounds therefore
cover tokens of up to 33 pieces each. The production vocabulary's longest entry is 31
characters, and only 45 of its 262,144 entries are longer than 16 (runs of spaces, tabs
and newlines). A window that is still merging after the last round is a fault, so the
limit can refuse a true tokenization but never accept a false one. Import should reject a
vocabulary with a longer entry, so that no honest claim meets the limit.

### Measured on the native model

For a 6,000-piece prompt of 1,148 tokens in three ranges, on the production table:

| | Today | As designed |
| --- | --- | --- |
| Tokenizer stages per range | about 27 | 1 |
| Merge-table lookups, whole prompt | 7,275,202 | 206,509 |

That is 35 times fewer lookups. One new stage does about 69,000 of them, comparable to one of
today's merge batches. Most are spent in rounds after every window has finished; dropping
finished windows from the list would cut them further and is left out for now.

The apply step reads one list entry per piece, where today's apply walks 256 pieces per
step. Its cost in the guest is not measured.

### Open points for authoring

- Whether the scan can append a window's best pair as the window ends, and the apply step
  read it back by number, as cleanly as described. This is the part with no precedent in
  the current stages.
- The length of one stage's trace, and whether the two passes should be two stages.
- Whether to keep `prompt_bind` unchanged, which costs a full pass over its pieces in
  every range as `prompt_seed` does today, or have it publish a piece count.

## What would change in the repository

- Three stage programs replaced by one; nine programs in place of eleven.
- `StageSchedule` loses `merge_batches`, so the `Execution` record's version changes.
- Run preparation writes `marks_{j}` and stops writing cuts and the batch count.
- Import and prepared-run formats move to version 8, the model is re-imported, and the
  backend's pinned version follows.
- `test-parity` gains a case whose marks cut the prompt into vocabulary tokens that are not
  the merge's result, and the cut-fault cases are replaced.
- Import rejects a vocabulary with an entry longer than the rounds can check.
- `raster-chain.md`, `claim-build.md` and `test-parity.md`.

## What was built

The stage, its native twin, run preparation, the import template and the parity gate's
cases are in place. These points differ from the plan above:

- **The schedule and the old programs were cleaned up in a second step.** The stage first
  shipped recording a seed, one merge step and a prepare step under the old three-count
  schedule, with the three old programs still in the repository. Later the same day the
  merge-batch count and the step's batch coordinate were removed (`Execution` version 3),
  the three programs and their fixture chain were deleted, every remaining program was
  re-locked, and import and prepared-run formats moved to version 9. Direct inference
  keeps its own whole-prompt tokenization in `inference-kernels`.
- **The previous range's last token is always kept as a neighbour**, including when the
  range starts a new segment. It then merges alone, which changes nothing for a token the
  previous range already checked, and saves the stage a decision.
- **Every list of pieces ends with an end piece.** A recur that builds a list gives its
  caller the list and not its final state, so the faults a pass finds travel on the last
  element. The end piece is also what closes the last pair's window.
- **A range hands the next one its last token as pieces**, in the `pieces` field of its
  output, which `prompt-prepare` left empty. Nothing downstream reads that field.
- **Two fault codes were added** to the shared prompt source, 8 for a token whose pieces
  are in two segments and 9 for tokens that are not what the merge produces. Code 4, a cut
  a merge could cross, is no longer recorded by any stage.
- **The tokenizer's boundary table was removed** in the same cleanup. It was what
  `prompt-seed` checked cuts against, and no stage read it once ranges could end between
  any two tokens. The tokenizer artifact is now the vocabulary, the merge table and the
  profile, so the model id changed.
- **Import refuses a merge table whose results are longer than 33 characters**, the most
  the stage's rounds can check.

The stage's logic is one set of plain functions in
[check.rs](../../raster-stages/prompt-check/src/check.rs), called by its tiles and by the
native twin
([prompt_check_stage.rs](../../crates/inference/kernels/src/kernels/prompt_check_stage.rs)).

## Questions for you

- Is fewer stages and cheaper challenges enough reason to do this, given that
  time-to-claim no longer depends on it?
- Should the boundary table and range-cut rules be retired in the same change, or kept
  until the new stage has been through a full gate?
- The design needs no table derived at import. Is one stage of 128 rounds acceptable, or
  should the two passes be separate stages from the start?
