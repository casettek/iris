# Checking claimed prompt tokens instead of recomputing them

Draft for review. November 9, 2026. No stage is implemented. A native version of the check
exists and its tests pass ([Tests of the rule](#tests-of-the-rule)). The soundness argument in
[Why the pair check is enough](#why-the-pair-check-is-enough) is the part to review first.

## Goal

**Give the pipeline the prompt's token ids as an input and have one stage per range check
them, in place of the stages that recompute them by replaying every merge.**

The chain already treats the output this way: the claimed output ids are an input, and
`sequence-check` verifies them. This proposal does the same for the prompt.

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

### The token ids become an input

Run preparation already tokenizes the prompt natively to choose the ranges. It would write,
for each range `j`, a committed input `tokens_{j}`:

```text
ClaimedToken {
    id,       // the token id
    pieces,   // how many of prompt_bind's pieces this token is made of
}
```

`pieces` tells the stage where each token starts without matching strings. It is a hint
and is not trusted: a wrong count fails the check below.

### One stage per range checks them

`prompt_check_p{j}` (name open) takes `prompt_bind`'s pieces, the previous range's context
and `tokens_{j}`. The tokens of a range cover consecutive pieces, starting where the
previous range ended. For each token it requires:

1. **It lies inside one segment.** Pieces of different segments never merge.
2. **With a left neighbour in its segment, the pair run gives back the two tokens.** The
   left neighbour is the token before it, which for the first token of a range is the last
   token of the previous range. Run the ranked merge on the two tokens' pieces alone. The
   result must be exactly two pieces: the first made of exactly the neighbour's pieces, the
   second of exactly this token's.
3. **With no left neighbour in its segment, it merges to one piece** when its pieces are
   run alone.
4. **The resulting text is the vocabulary entry for the claimed id.**

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
- A token that starts its segment has no left neighbour, which is why requirement 3 exists.
- The stage must tie the result to the claimed group boundary, not only to the two texts.
  Otherwise a wrong `pieces` count could pass when the texts happen to match.
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

## Authoring the stage: open

I have not designed the Raster program, and this is where the remaining risk to the plan
sits, as distinct from the risk to soundness.

- A sequence cannot loop on data, and a tile cannot look anything up; lookups are
  authenticated selections made by the sequence. Today's merge stage handles this by
  unrolling 64 rounds, each a recur over the range's pieces.
- A pair run is small, but its length depends on the two tokens. The stage needs a bound
  on pieces per token, or a shape whose work items are the pieces themselves.
- The recur rules forbid driving a loop with a counter list and require loop state to stay
  small, which rules out the most direct translations.

Shapes to evaluate, none yet checked against the authoring rules in detail:

- **Unrolled pair runs.** A fixed number of rounds per pair, sized for the longest token.
  Simple, but the cost per pair is set by the worst case. The production vocabulary's
  longest entry is 31 characters, so a pair can hold 62 pieces and need 60 rounds, while
  only 45 of its 262,144 entries are longer than 16 characters (runs of spaces, tabs and
  newlines).
- **One pass over the pieces.** A recur over the range's pieces that carries the current
  pair's small window as state. Work proportional to the pieces, but the state has to stay
  small enough for the recur rules.
- **Per-token facts committed at import.** A token's solo run depends only on the
  vocabulary, so whether it merges to itself could be a committed table, as the boundary
  table is today. That would leave only the crossing check per pair.

By my rough count the check needs 50 to 100 times fewer authenticated lookups per range
than today, because each run covers a few pieces instead of the range. That estimate
assumes a shape whose cost follows the actual token lengths.

## What would change in the repository

- Three stage programs replaced by one; nine programs in place of eleven.
- `StageSchedule` loses `merge_batches`, so the `Execution` record's version changes.
- Run preparation writes `tokens_{j}` and stops writing cuts and the batch count.
- Import and prepared-run formats move to version 8, the model is re-imported, and the
  backend's pinned version follows.
- `test-parity` gains a case whose tokens spell the prompt but are not the merge's result,
  and the cut-fault cases are replaced.
- `raster-chain.md`, `claim-build.md` and `test-parity.md`.

## Suggested order

1. Review this argument.
2. Native check and tests. Done; see [Tests of the rule](#tests-of-the-rule).
3. Pick the stage shape, using the token lengths above.
4. Author the stage and take it through the check ladder.
5. Formats, re-import, parity cases, full gate.

Nothing before step 4 touches the chain.

## Questions for you

- Is fewer stages and cheaper challenges enough reason to do this, given that
  time-to-claim no longer depends on it?
- Should the boundary table and range-cut rules be retired in the same change, or kept
  until the new stage has been through a full gate?
- Is a committed per-token table derived at import acceptable, as the boundary table is
  today, or should the stage establish everything from the merge table itself?
