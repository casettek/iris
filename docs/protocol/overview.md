# How IWEBS works

IWEBS is an adjudication harness. It runs a language model as the judge of a
case, under a procedure the parties agreed to beforehand, and the verdict it
gives does not depend on who ran it.

This page is the whole of that from the top, in two halves. The first needs no
blockchain: what a procedure, a case and a verdict are, what a run shows, and
the record that carries it. The second is the one way a verdict is settled on
a blockchain today: what a claim is, the one rule that makes a claim
checkable, and what catches each way a claim can be wrong. The other pages
under `protocol/` and `blockchain/` each go into one part.

Judge, verdict, evidence and question are the
[glossary's words for users](../glossary.md#the-words-for-what-iwebs-is-used-for).
The code says model, result, source and recipe.

## Procedure, case, verdict

A **procedure** is how a judgment is made, fixed before any case. It holds:

- **The models**, each by the id its weights, tokenizer and descriptor hash
  to, and **the parameters** a run must have used.
- **What a run must say of its source and its result**, when the procedure
  requires either: the signer of the documents the prompt is built from, and
  the forms the answer is read by.
- **The pipeline**: the kinds of stage a run is made of, each a program with
  an identity, and for each program where each of its inputs comes from: an
  earlier stage's output and which programs may produce it, a pinned
  commitment such as the model's weights, or the run.

A procedure's **id** is a hash over all of that (`iwebs procedure id`). Two
parties that name one id have agreed to one procedure. The
[procedure kinds](procedure-kinds.md) are the arrangements one can take: one
model, two models that must agree, or one model's output set in a second
model's prompt.

A **case** is one run of a procedure: the question, and the evidence it is
judged on. They are inputs of the run and are not in the procedure's id. A
procedure made from a specification names the evidence's signer and no
recipe, so the question is the case's.

The **verdict** is what the procedure gives on the case: each inference's
output, the result the procedure's forms read from it, and for a consensus
whether the two models agree.

## What a run shows

Inference is integer arithmetic with every rounding and overflow rule fixed
([numerics](numerics.md)), so the same inputs give the same bytes whoever runs
them.

There are two sets of procedures, and a run is of one
([run](../workflows/run.md#two-sets-of-procedures)):

| Set | What a run does | How it runs |
| --- | --- | --- |
| Generation | Finds the verdict: each model run forward, each token its own choice. | Natively, with no stages. |
| Verification | Holds a verdict stated beforehand to each model's own choice, token by token. | Stage by stage, with a checkpoint for each stage. |

They are different logic, so they are different procedures with different
ids, and nothing pairs one with the other. Whoever needs a verdict verified
runs a generation first and states what it found in the case of a
verification. The parity gate holds that the two give the same result.

A run that someone else can make again shows one thing: this procedure, on
these inputs, gives this verdict. So the party that relies on a verdict need
not trust the party that ran the model. Three things a run does not show:

- **That the answer is true.** That rests on the procedure the parties
  chose and the model it names.
- **That the evidence is what a server said.** A source's signer says so, and
  is trusted ([what is trusted today](#what-is-trusted-today)).
- **That the inputs can be had.** The model's weights and the case's inputs
  are assumed to be available to whoever checks.

## The record, and what backs it

A **record** is what the party that ran a case hands to everyone else
([the record of a run](../workflows/run.md#the-record-of-a-run)). It has two
parts.

**What it states** is the procedure's id, the case's inputs and the verdict.
That is the same whatever method the run was made for.

**Its backings** are what stands behind that, besides making the run again.
A record names the method of each:

| Method | What backs the verdict | Built |
| --- | --- | --- |
| Making the run again | Nothing but the record. Whoever doubts it runs the whole case and compares. How a generation's record is checked | Yes: `iwebs record check` |
| Fault provable (`fault_provable`) | The count and root of a verification's checkpoints, one for each stage, so that a dispute comes down to proving one stage wrong. Raster's method, and what a claim commits to | Yes: `iwebs run --record` of a verification writes it |

An enclave's attestation and a validity proof would each be a further method.
Neither is built.

## With no blockchain

1. The parties agree on a procedure and put its id in whatever binds them.
2. A case comes up, and any of them runs it: `iwebs run --procedure <id> --record <dir>`,
   of a generation procedure.
3. The record is handed to the others, by any means.
4. Whoever doubts it runs `iwebs record check` under their own specification
   of the procedure, which makes the generation again. The record holds, or
   the check names the first field that differs.

IWEBS stops there. Three things are the parties' own:

- **Enforcement.** Nothing acts on a verdict.
- **A place and a clock.** There is nowhere a record is posted and no time by
  which an objection must come.
- **Who made a record.** A record is not signed.

Where two runs differ, a third settles which is the procedure's.

## On a blockchain

The rest of this page is the fault provable method with a blockchain as the
place a verdict is settled. Every procedure in it is a verification procedure:
the claimer has found the output beforehand, with a generation procedure or
any other way. A contract is given a procedure, registered under
its id ([availability](availability.md#the-procedure-registry)). A claim
names one, and is judged by it. The claims contract is also deployed with the
ids of Raster's two proof programs and RISC Zero's verifier.

### Who takes part

| Party | What it does |
| --- | --- |
| Claimer | Runs the model, posts the claim's data, submits the claim. |
| Challenger | Reads a claim back, checks it, and either points at a fault or replays the stage that is wrong. Anyone can be one. |
| The contracts | Hold what is pinned, take claims and challenges, void a claim or dismiss a challenge. They run no stage. |
| Trusted hosts | Stand where a proof belongs today: one signs what each posted blob holds, one relays Celestia's block roots, one signs a source's documents. |

### What a claim is

A claim posts what a record holds: the same inputs, the same checkpoints and
the same procedure id.

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

### The one rule

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

### What catches each way a claim can be wrong

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

### What is trusted today

| Trusted | For what | What replaces it |
| --- | --- | --- |
| The attester | That each posted blob is on Celestia and holds what the claim says | A proof of inclusion and of the blob's contents |
| The relayer | Celestia's block roots, posted to an oracle contract | A bridge that proves them |
| A source's signer | That a server answered a request with a document | A notary |

The first two are there only with a blockchain. A source's signer is trusted
with one or without.

### What is not there

Nothing yet acts on a void claim or an unanswered challenge: there are no
clocks that close a claim, no bonds, and no finality. Stages too short to
replay cannot be disputed. These and every other gap are in
[one list](open.md).
