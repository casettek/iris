#!/usr/bin/env python3
"""Hold the local contracts to iris's own verdict, case by case.

  replay.py claim --run <spec.toml> --submission <claim_submission.json>
  replay.py challenge --run <spec.toml> --submission <claim_submission.json> \\
                      --challenge <challenge_submission.json>

`claim` takes a claim that `iris claim post` has posted. It asks `iris claim
check` and the contract about the claim as it is, then about one changed copy
after another, and compares the two answers: both take it, or both refuse it
at the same check. The claim itself is then submitted.

`challenge` does the same for a challenge that `iris challenge post` has
posted: it compares `iris challenge check` with the contract on the challenge
and on changed copies, and submits the challenge. The claim it disputes is
submitted first if the contract does not hold it.

A changed copy is only ever called, never sent, so nothing but the claim and
the challenge themselves reaches the contract's state. A few cases are the
contract's alone, where it checks something the Rust model has no part for
yet: the host's signature, and what the contract already holds. They are
listed as such.

Exits 0 when every case agrees. The stack must be up and the contracts
deployed, as for submit.py.
"""

import argparse
import copy
import json
import pathlib
import subprocess
import sys

import submit
from submit import DEPLOYMENT, ENV, IRIS, foundry, iris

ERRORS = DEPLOYMENT.get("errors", {})

# The check of `iris claim check` and `iris challenge check` that each
# contract error belongs to: the start of the line the check is printed on.
CHECK_OF = {
    "UnregisteredConfiguration": "submission: configuration",
    "AnotherShape": "submission: pipeline",
    "StatementMismatch": "submission: statement",
    "OutputMismatch": "submission: statement",
    "NotValid": "submission: statement",
    "AnotherSourceRecord": "submission: source record",
    "UnexpectedSource": "submission: source record",
    "MissingSource": "submission: source record",
    "SourceFault": "submission: source record",
    "AnotherSigner": "submission: source record",
    "AnotherRecipe": "submission: source record",
    "AnotherForms": "submission: source record",
    "AnotherResult": "submission: source record",
    "AnotherConsensus": "submission: source record",
    "NothingPosted": "posted",
    "NotIncluded": "posted",
    "ExpiresTooSoon": "posted",
    "NotProven": "posted",
    "AnotherSubmitter": "posted",
    "SourceNotPosted": "posted",
    "PromptTextNotPosted": "posted",
    "CheckpointsNotPosted": "posted",
    "TraceBodyNotPosted": "posted",
    "RecordsNotStated": "submission: records and results",
    "NotTheResultStage": "submission: records and results",
    "AnotherStatedResult": "submission: records and results",
    "NoCheck": "submission: records and results",
    "NotACheck": "submission: records and results",
    "AnotherStatedRecord": "submission: records and results",
    "NotTheCheckpoint": "challenge: of the claim's checkpoint",
    "UnapprovedProgram": "challenge admission: the stage names an approved program",
    "OutputNotDisputed": "challenge: Raster admits it",
}


def check_name(line):
    """The name a printed check goes by in CHECK_OF."""
    line = line.replace("the first inference's ", "").replace("the second inference's ", "")
    if "posted, proven and retrievable" in line:
        return "posted"
    if line.startswith("challenge: about the claim's checkpoint"):
        return "challenge: of the claim's checkpoint"
    for name in set(CHECK_OF.values()):
        if line.startswith(name):
            return name
    return line


def rust_verdict(arguments):
    """What iris decides: None when it takes the submission, otherwise the
    first check it refuses it at and why."""
    done = subprocess.run([IRIS, *arguments], capture_output=True, text=True, env=ENV)
    refused = None
    seen = False
    for line in done.stdout.splitlines():
        state, _, rest = line.partition(" ")
        if state not in ("ok", "REJECTED"):
            continue
        seen = True
        rest = rest.strip()
        # The checks of a submission, and not the comparisons a later
        # challenge of the claim would meet.
        if rest.startswith("challenge admission: every "):
            continue
        if state == "REJECTED" and refused is None:
            check, _, reason = rest.partition(": ")
            head, _, tail = reason.partition(": ")
            # A check's name can hold a colon of its own.
            if check in ("submission", "challenge", "challenge admission"):
                check, reason = f"{check}: {head}", tail
            refused = {"check": check_name(check), "reason": reason}
    if not seen:
        sys.exit(f"iris {' '.join(arguments)} made no checks:\n{done.stderr}")
    return refused


def contract_verdict(function, argument, at=None):
    """What the contract decides, by a call that changes nothing: None when
    it takes the submission, otherwise the error it reverts with."""
    done = foundry(
        "cast", "call", DEPLOYMENT["claims"], f"{function}(bytes)(bytes32)", argument,
        "--from", ENV["IRIS_SENDER_ADDRESS"], "--rpc-url", ENV["IRIS_EVM_RPC"],
        *(["--block.time", str(at)] if at is not None else []),
    )
    if done.returncode == 0:
        return None
    reason = submit.revert_reason(done.stderr + done.stdout, ERRORS)["error"]
    return reason.partition("(")[0]


def contract_id(function, argument):
    """The id the contract gives a submission it takes."""
    done = foundry(
        "cast", "call", DEPLOYMENT["claims"], f"{function}(bytes)(bytes32)", argument,
        "--from", ENV["IRIS_SENDER_ADDRESS"], "--rpc-url", ENV["IRIS_EVM_RPC"],
    )
    return done.stdout.strip() if done.returncode == 0 else None


class Replay:
    def __init__(self):
        self.rows = []

    def same_id(self, iris_id, contract):
        """The id iris works out for a claim is the one the contract gives it."""
        agree = contract is not None and iris_id.lower() == contract.lower()
        self.rows.append({"case": "id", "iris": iris_id, "contract": contract, "agree": agree})
        print(f"{'ok  ' if agree else 'DIFF'}  {'id':24} iris: {iris_id}  contract: {contract}", flush=True)

    def record(self, name, rust, contract, contract_only=None):
        """One case: the two verdicts, and whether they are the same."""
        if contract_only is not None:
            agree = contract == contract_only
            rust_text = "(no part)"
        else:
            rust_text = "accepted" if rust is None else f"refused at {rust['check']}"
            agree = (rust is None) == (contract is None) and (
                rust is None or CHECK_OF.get(contract) == rust["check"]
            )
        self.rows.append({
            "case": name,
            "iris": rust_text,
            "contract": contract or "accepted",
            "agree": agree,
        })
        print(f"{'ok  ' if agree else 'DIFF'}  {name:24} iris: {rust_text:58} contract: {contract or 'accepted'}",
              flush=True)

    def finish(self):
        differing = [row["case"] for row in self.rows if not row["agree"]]
        print(f"{len(self.rows) - len(differing)} of {len(self.rows)} cases agree")
        if differing:
            sys.exit(f"the contract and iris differ on: {', '.join(differing)}")


def changed(path, name, original, change):
    """Write a changed copy of a submission beside it. Its path."""
    copied = copy.deepcopy(original)
    change(copied)
    out = path.with_name(f"replay-{name}.{path.name}")
    out.write_text(json.dumps(copied, indent=2))
    return out


def block_time(height):
    return json.loads(iris("da", "block-root", "--height", str(height)))["time"]


def accepted_at(claim_id):
    """When the contract accepted a claim, in Unix seconds."""
    held = submit.call(DEPLOYMENT["claims"], "claims(bytes32)", claim_id,
                       returns="(bytes32,address,uint32,bytes32,uint64)")
    if held is None:
        sys.exit(f"the contract does not answer for claim {claim_id}")
    return int(held.splitlines()[4].split()[0])


def unsign(submission):
    for blob in submission["availability"]:
        blob["proof"].pop("attestation", None)


def replay_claim(arguments):
    spec = arguments.run
    path = pathlib.Path(arguments.submission).resolve()
    replay = Replay()

    iris("da", "attest", "--submission", str(path))
    honest = json.loads(path.read_text())
    built = json.loads(iris("evm", "claim", "--submission", str(path)))
    submit.register(spec)
    submit.relay(built["heights"])

    def verdicts(submission_path, at=None, ask_iris=True):
        rust = ask_iris and rust_verdict(
            ["claim", "check", "--run", spec, "--submission", str(submission_path),
             *(["--submitted-at", str(at)] if at is not None else [])])
        argument = json.loads(iris("evm", "claim", "--submission", str(submission_path)))["argument"]
        return rust, contract_verdict("submitClaim", argument, at)

    rust, contract = verdicts(path)
    already = contract == "AlreadySubmitted"
    replay.record("as built", rust, None if already else contract)
    if contract is None:
        replay.same_id(built["claim"], contract_id("submitClaim", built["argument"]))

    def case(name, change, at=None, attest=False, contract_only=None):
        copy_path = changed(path, name, honest, change)
        if attest:
            iris("da", "attest", "--submission", str(copy_path))
        rust, contract = verdicts(copy_path, at, contract_only is None)
        replay.record(name, rust, contract, contract_only)

    first = honest["inferences"][0]
    blob = honest["availability"][0]
    case("other-configuration", lambda s: s.__setitem__("configuration", "00" * 32))
    case("other-output", lambda s: s["inferences"][0].__setitem__("output_text", first["output_text"] + "!"))
    case("not-valid", lambda s: s["inferences"][0]["record"].__setitem__("valid", False))
    case("other-statement", lambda s: s.__setitem__("input_text_sha256", "11" * 32))
    case("other-result", lambda s: s["inferences"][0]["result"].__setitem__("code", first["result"]["code"] + 7))
    if len(honest["inferences"]) > 1:
        case("other-second-output",
             lambda s: s["inferences"][1].__setitem__("output_text", s["inferences"][1]["output_text"] + "!"))
    if honest.get("source"):
        case("other-source-record",
             lambda s: s["source"].__setitem__("documents", s["source"]["documents"] + 1))
        case("missing-source", lambda s: s.pop("source"))
    if honest.get("consensus"):
        case("other-agreement",
             lambda s: s["consensus"].__setitem__("agree", not s["consensus"]["agree"]))
        case("no-agreement", lambda s: s.pop("consensus"))
    case("copied", lambda s: s.__setitem__("claimer", "0x00000000000000000000000000000000000c0b1e"))
    case("nothing-posted", lambda s: s.__setitem__("availability", []))
    case("not-included", lambda s: s["availability"][0]["pointer"].__setitem__("height", blob["pointer"]["height"] + 1))
    case("expires",
         lambda s: None,
         at=block_time(blob["pointer"]["height"]) + DEPLOYMENT["retention_secs"]
         - DEPLOYMENT["challenge_period_secs"] + 1)
    # The contract's alone: iris reads the claim's directory, and stops before
    # its checks when the list there is not the one the submission names.
    case("other-root", lambda s: s.__setitem__("checkpoints_root", "22" * 32),
         contract_only="CheckpointsNotPosted")
    case("overstated-contents", lambda s: s["availability"][0]["contents"].setdefault("prompt_texts", []).append("33" * 32))

    def unstate(s):
        for posted in s["availability"]:
            posted["contents"]["records"] = []
            posted["proof"]["outputs"] = []
        unsign(s)

    # The host signs what the blobs hold when no output is read: no records.
    case("records-not-stated", unstate, attest=True)
    # The contract's alone: the host's signature is how it learns what the
    # Rust model reads from the layer itself.
    case("unsigned", unsign, contract_only="NotIncluded")

    if not already:
        ok, result = submit.send(DEPLOYMENT["claims"], "submitClaim(bytes)", built["argument"])
        if not ok:
            sys.exit(f"the contract refused the claim it had taken in a call: {result}")
    # The contract's alone: the Rust model holds no earlier claims.
    replay.record("again", None, contract_verdict("submitClaim", built["argument"]),
                  contract_only="AlreadySubmitted")
    print(json.dumps({"claim": built["claim"]}))
    replay.finish()


def replay_challenge(arguments):
    spec = arguments.run
    claim_path = pathlib.Path(arguments.submission).resolve()
    path = pathlib.Path(arguments.challenge).resolve()
    replay = Replay()

    # The disputed claim, as the contract holds it. A contract takes a claim
    # with a false checkpoint: that is what a challenge is for.
    iris("da", "attest", "--submission", str(claim_path))
    claim = json.loads(iris("evm", "claim", "--submission", str(claim_path)))
    submit.register(spec)
    submit.relay(claim["heights"])
    held = contract_verdict("submitClaim", claim["argument"])
    if held is None:
        ok, result = submit.send(DEPLOYMENT["claims"], "submitClaim(bytes)", claim["argument"])
        if not ok:
            sys.exit(f"the contract refused the claim it had taken in a call: {result}")
    elif held != "AlreadySubmitted":
        sys.exit(f"the contract does not take the disputed claim: {held}")

    iris("da", "attest", "--submission", str(path))
    honest = json.loads(path.read_text())

    def build(submission_path, of_claim=claim_path):
        return json.loads(iris("evm", "challenge", "--challenge", str(submission_path),
                               "--submission", str(of_claim)))

    built = build(path)
    submit.relay(built["heights"])

    def verdicts(submission_path, at=None, ask_iris=True):
        rust = ask_iris and rust_verdict(
            ["challenge", "check", "--run", spec, "--challenge", str(submission_path),
             "--submission", str(claim_path),
             *(["--submitted-at", str(at)] if at is not None else [])])
        return rust, contract_verdict("submitChallenge", build(submission_path)["argument"], at)

    rust, contract = verdicts(path)
    already = contract == "AlreadySubmitted"
    replay.record("as built", rust, None if already else contract)

    def case(name, change, at=None, contract_only=None):
        rust, contract = verdicts(changed(path, name, honest, change), at, contract_only is None)
        replay.record(name, rust, contract, contract_only)

    case("copied", lambda s: s.__setitem__("challenger", "0x00000000000000000000000000000000000c0b1e"))
    case("nothing-posted", lambda s: s.__setitem__("availability", []))
    # The output the claim's own checkpoint states: nothing is disputed.
    stated = json.loads(claim_path.with_name("checkpoint_trace.json").read_text())
    stated = (stated["checkpoints"] if isinstance(stated, dict) else stated)[honest["stage"]]["output_sha256"]
    case("undisputed",
         lambda s: s["challenge"].__setitem__("claimed_output", list(bytes.fromhex(stated))))
    case("overstated-contents",
         lambda s: s["availability"][0]["contents"]["trace_bodies"].append("33" * 32))
    case("unsigned", unsign, contract_only="NotIncluded")
    # The contract's alone: a challenge names its claim by id on chain, and
    # the Rust model holds no claims to look one up in.
    unknown = changed(claim_path, "unknown", json.loads(claim_path.read_text()),
                      lambda s: s.__setitem__("claimer", "0x00000000000000000000000000000000000c0b1e"))
    replay.record("unknown-claim", None,
                  contract_verdict("submitChallenge", build(path, unknown)["argument"]),
                  contract_only="UnknownClaim")
    # The contract's alone: the Rust model keeps no clock for a claim yet.
    # The period runs from when the contract accepted the claim, which can be
    # long after the challenge's trace was posted.
    case("period-over", lambda s: None,
         at=accepted_at(built["claim"]) + DEPLOYMENT["challenge_period_secs"] + 3600,
         contract_only="ChallengePeriodOver")

    if not already:
        ok, result = submit.send(DEPLOYMENT["claims"], "submitChallenge(bytes)", built["argument"])
        if not ok:
            sys.exit(f"the contract refused the challenge it had taken in a call: {result}")
    # The contract's alone: the Rust model holds no earlier challenges.
    replay.record("again", None, contract_verdict("submitChallenge", built["argument"]),
                  contract_only="AlreadySubmitted")
    print(json.dumps({"claim": built["claim"], "stage": honest["stage"]}))
    replay.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=["claim", "challenge"])
    parser.add_argument("--run", required=True, help="the run spec of the configuration")
    parser.add_argument("--submission", required=True, help="the claim's claim_submission.json")
    parser.add_argument("--challenge", help="the challenge_submission.json that `iris challenge post` wrote")
    arguments = parser.parse_args()
    if arguments.what == "claim":
        replay_claim(arguments)
    else:
        if not arguments.challenge:
            parser.error("a challenge needs --challenge")
        replay_challenge(arguments)


if __name__ == "__main__":
    main()
