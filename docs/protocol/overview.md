# How iris works

A claimer says what a model answered. Anyone can check it, and a contract acts
on it if nobody shows it wrong. This page is the whole of that from the top:
what exists before a claim, what a claim is, the one rule that makes a claim
checkable, and what catches each way a claim can be wrong. The other pages
under `protocol/` and `blockchain/` each go into one part.

## Who takes part

| Party | What it does |
| --- | --- |
| Claimer | Runs the model, posts the claim's data, submits the claim. |
| Challenger | Reads a claim back, checks it, and either points at a fault or replays the stage that is wrong. Anyone can be one. |
| The contracts | Hold what is pinned, take claims and challenges, void a claim or dismiss a challenge. They run no stage. |
| Trusted hosts | Stand where a proof belongs today: one signs what each posted blob holds, one relays Celestia's block roots, one signs a source's documents. |

## What is there before any claim

A contract is given a **procedure**, registered under its id
([availability](availability.md#the-procedure-registry)). A claim names
one, and is judged by it.

- **The pipeline**: the kinds of stage a run is made of, each a program with
  an identity.
- **For each program, where each of its inputs comes from**: an earlier
  stage's output and which programs may produce it, a pinned commitment such
  as the model's weights, or the run.
- **The models**, each by the id its weights, tokenizer and descriptor hash
  to, and **the parameters** a run must have used.
- **What a claim must say of its source and its result**, when the
  procedure requires either: the signer of the documents the prompt is
  built from, the recipe that builds it, and the forms the answer is read by.

The claims contract is also deployed with the ids of Raster's two proof
programs and RISC Zero's verifier.

## What a claim is

A run is a pipeline of stages ([the pipeline](raster-pipeline.md)): the prompt is
bound and tokenized, each layer runs over each range of tokens, the output is
checked token by token against the model's own choice, and a last stage states
the result. A claim commits to one **checkpoint** per stage: which program
ran, the list of inputs it loaded, which earlier stage each piped input came
from, and its output.

| Where | What |
| --- | --- |
| Posted to Celestia | Every input the run gave its stages (the text, the token marks, the claimed output ids, the settings, a source's documents), the checkpoint hashes, every checkpoint, and every stage's input list. |
| In the transaction | The procedure's id, the claimer, the input text's hash, the source record, each inference's output text with its record and result, the checkpoint count and root, and where each blob is with the host's signature over what it holds. |
| Kept by the contract | The claim's procedure, claimer, checkpoint count and root, when it arrived, and one hash of the commitments of the run inputs it posted. |

On arrival the contract checks what it can from the transaction alone
([the contract's rules](checks.md#on-submission)): the procedure is
registered, the statement is the procedure's model and parameters over
the claimed input, the record is valid, the source and result are what the
procedure requires, and the data is posted and stays retrievable. It does
not read the checkpoints.

## The one rule

**Every input a checkpoint names is something a challenger also holds**: the
output of an earlier checkpoint, a commitment the procedure pins, or a
value the claim posted.

A dispute about a stage is settled against the inputs the claimer's own
checkpoint names. So when two parties disagree about a stage, they ran it on
the same inputs, and a replay of that one stage settles it. A claimer that
named an input nobody else had could answer any honest challenger, which is
why the rule is checked and not assumed.

What a stage does with its inputs is the stage's to check, inside the
program: that the weights are the model's, that the marks are the
tokenizer's, that a claimed token is the model's choice, that it is the right
stage at the right place in the schedule. A stage that finds something wrong
says so in its output and finishes, and every later stage carries that into
the final record.

## What catches each way a claim can be wrong

| What is wrong | What catches it | How it ends |
| --- | --- | --- |
| The claim does not state what its procedure requires, or its data is not posted | The contract, when the claim arrives (`submitClaim`) | The claim is refused |
| A stage names a program outside the approved set | Anyone, pointing at the checkpoint (`voidClaim`) | The claim is void, in that transaction |
| A stage names an input that is not an earlier stage's output, a pinned commitment or a posted value | Anyone, pointing at the checkpoint and the input (`voidClaim`) | The claim is void, in that transaction |
| A stage's output is not what its program gives on its inputs | A challenger, replaying that stage and posting the trace (`submitChallenge`) | The challenge stands unless the claimer answers it |
| A challenge's trace is itself wrong | The claimer, with a Raster fault proof of one window of it (`submitFaultProof`) | The challenge is dismissed |

The fourth row covers more than a miscomputed layer. A stage given a value
from the wrong place (another layer's weights, another range's marks) states
a fault when it is run honestly, so a claim that states a clean output for it
has a false output. So has a claim whose output tokens are not the model's:
the check stage says so, and a claim that states a valid verdict lies about
that stage. Both are found by running every stage on the inputs its
checkpoint names and replaying the first whose output differs
([challenge build](../workflows/challenge-build.md)).

The second and third rows need no replay and have no answer: the contract
reads the claim's own checkpoint, its procedure and what it posted, and
there is nothing to wait for. They are two questions of one call, with five
recorded faults ([the blockchains](../blockchain/overview.md#what-voidclaim-checks)).

There is one proof in all of this: Raster's fault proof, in the last row.

## What is trusted today

| Trusted | For what | What replaces it |
| --- | --- | --- |
| The attester | That each posted blob is on Celestia and holds what the claim says | A proof of inclusion and of the blob's contents |
| The relayer | Celestia's block roots, posted to an oracle contract | A bridge that proves them |
| A source's signer | That a server answered a request with a document | A notary |

## What is not there

Nothing yet acts on a void claim or an unanswered challenge: there are no
clocks that close a claim, no bonds, and no finality. Stages too short to
replay cannot be disputed. These and every other gap are in
[one list](open.md).
