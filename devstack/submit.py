#!/usr/bin/env python3
"""Send a claim, a challenge or a fault proof to the local contracts, as the
trusted host and the sender would.

  submit.py claim --run <spec.toml> --submission <claim_submission.json>
  submit.py challenge --submission <claim_submission.json> \\
                      --challenge <challenge_submission.json>
  submit.py fault-proof --submission <claim_submission.json> \\
                        --challenge <challenge_submission.json> --receipt <fault.receipt>
  submit.py fault --calls <calls.json> --call <n>

For a claim: the attester signs each posted blob, the relayer posts the block
roots those blobs are under, the configuration is registered if it is not, and
the claim is submitted. For a challenge: the same for its trace body, then the
challenge is submitted. A fault proof answers a challenge the contract holds:
it is sent as it is, and the contract verifies it. A fault is something wrong
with a claim that no replay shows: `iris claim audit --calls` writes the call
that points at each one it finds, and `fault` sends the one named. The
contract checks it against the claim's own checkpoints and holds the claim
void.

Prints one JSON object: whether the contract took it, the id it gave it, or
the error it reverted with. iris itself sends nothing: it writes the bytes,
and this script sends them with `cast`. iris-backend does the same job for a
deployed service.

The stack must be up (`docker compose up -d --wait`) and the contracts
deployed (`deploy.sh`). `--no-attest` and `--no-relay` leave those steps out,
to send a submission as it is.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent


def environment():
    """The stack's endpoints and accounts, from env.sh."""
    out = subprocess.run(
        ["bash", "-c", f"source {HERE / 'env.sh'} && env"],
        check=True, capture_output=True, text=True,
    ).stdout
    env = dict(os.environ)
    for line in out.splitlines():
        name, _, value = line.partition("=")
        if name.startswith("IRIS_"):
            env[name] = value
    return env


ENV = environment()
IRIS = os.environ.get("IRIS_BINARY", str(ROOT / "target/release/iris"))
DEPLOYMENT = json.loads((HERE / "deployment.json").read_text())
# The contracts' errors by selector, as deploy.sh listed them.
ERRORS = DEPLOYMENT.get("errors", {})


def run(command, **kwargs):
    return subprocess.run(command, capture_output=True, text=True, env=ENV, **kwargs)


def iris(*arguments):
    done = run([IRIS, *arguments])
    if done.returncode != 0:
        sys.exit(f"iris {' '.join(arguments)} failed:\n{done.stderr}")
    return done.stdout


def foundry(*arguments):
    return run([str(HERE / "foundry.sh"), *arguments])


def revert_reason(text, errors):
    """The error a failed call reverted with, by name when it is one of the
    contract's."""
    for word in text.replace('"', " ").replace(",", " ").split():
        if word.startswith("0x") and len(word) >= 10:
            selector = word[2:10]
            if selector in errors:
                return {"error": errors[selector], "data": word}
    return {"error": text.strip().splitlines()[-1] if text.strip() else "reverted"}


def call(to, signature, *arguments, returns=None):
    """A call that changes nothing: its result, or None when it reverts."""
    done = foundry("cast", "call", to, signature + (returns or ""), *arguments,
                   "--rpc-url", ENV["IRIS_EVM_RPC"])
    return done.stdout.strip() if done.returncode == 0 else None


def send(to, signature, *arguments):
    """Send a transaction. Returns (True, receipt text) or (False, reason)."""
    done = foundry("cast", "send", to, signature, *arguments,
                   "--rpc-url", ENV["IRIS_EVM_RPC"], "--private-key", ENV["IRIS_SENDER_KEY"])
    if done.returncode == 0 and "status               1" in done.stdout:
        return True, done.stdout
    return False, revert_reason(done.stderr + done.stdout, ERRORS)


def gas_used(receipt):
    """The gas a sent transaction used, from the receipt `cast send` prints."""
    for line in receipt.splitlines():
        name, _, value = line.partition(" ")
        if name == "gasUsed":
            return int(value.strip())
    return None


def relay(heights):
    """Post each block's root to the oracle, as the trusted relayer."""
    for height in heights:
        root = json.loads(iris("da", "block-root", "--height", str(height)))
        ok, reason = send(
            DEPLOYMENT["oracle"], "post(uint64,bytes32,uint64)",
            str(height), "0x" + root["data_root"], str(root["time"]),
        )
        if not ok:
            sys.exit(f"the oracle refused block {height}: {reason}")


def register(run_spec):
    """Register the run's configuration unless it is registered. Its id."""
    configuration = json.loads(iris("evm", "configuration", "--run", run_spec))
    registered = call(DEPLOYMENT["registry"], "isRegistered(bytes32)",
                      configuration["configuration"], returns="(bool)")
    if registered != "true":
        ok, reason = send(DEPLOYMENT["registry"], "register(bytes)", configuration["argument"])
        if not ok:
            sys.exit(f"the registry refused the configuration: {reason}")
    return configuration["configuration"]


def submit(function, built, arguments):
    if not arguments.no_relay:
        relay(built["heights"])
    ok, result = send(DEPLOYMENT["claims"], f"{function}(bytes)", built["argument"])
    return {"accepted": True} if ok else {"accepted": False, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=["claim", "challenge", "fault-proof", "fault"])
    parser.add_argument("--run", help="the run spec of a claim's configuration")
    parser.add_argument("--submission", help="the claim's claim_submission.json")
    parser.add_argument("--calls", help="the calls `iris claim audit --calls` wrote")
    parser.add_argument("--call", type=int, help="which of them to send")
    parser.add_argument("--challenge", help="the challenge's challenge_submission.json")
    parser.add_argument("--receipt", help="the fault proof `iris fault prove` wrote")
    parser.add_argument("--no-attest", action="store_true")
    parser.add_argument("--no-relay", action="store_true")
    parser.add_argument("--no-register", action="store_true")
    arguments = parser.parse_args()

    if arguments.what == "fault":
        if not arguments.calls or arguments.call is None:
            parser.error("a fault needs --calls and --call")
        fault = json.loads(pathlib.Path(arguments.calls).read_text())[arguments.call]
        outcome = {"checkpoint": fault["checkpoint"], "stage": fault["stage"], "input": fault["input"],
                   "expected": fault["expected"]}
        ok, result = send(DEPLOYMENT["claims"], fault["function"], fault["argument"])
        outcome.update({"accepted": True, "gas_used": gas_used(result)} if ok
                       else {"accepted": False, **result})
        print(json.dumps(outcome, indent=2))
        sys.exit(0 if outcome["accepted"] else 1)
    if not arguments.submission:
        parser.error("a claim, a challenge and a fault proof need --submission")

    if arguments.what == "claim":
        if not arguments.no_attest:
            iris("da", "attest", "--submission", arguments.submission)
        built = json.loads(iris("evm", "claim", "--submission", arguments.submission))
        outcome = {"claim": built["claim"]}
        if not arguments.no_register:
            if not arguments.run:
                parser.error("registering a claim's configuration needs --run")
            outcome["configuration"] = register(arguments.run)
        outcome.update(submit("submitClaim", built, arguments))
    elif arguments.what == "fault-proof":
        if not arguments.challenge or not arguments.receipt:
            parser.error("a fault proof needs --challenge and --receipt")
        built = json.loads(iris("evm", "fault-proof", "--receipt", arguments.receipt,
                                "--challenge", arguments.challenge,
                                "--submission", arguments.submission))
        # What iris expects the contract to say of the journal, beside what
        # the contract said.
        outcome = {"challenge": built["challenge"], "expected_rejection": built["rejection"]}
        ok, result = send(DEPLOYMENT["claims"], "submitFaultProof(bytes)", built["argument"])
        outcome.update({"accepted": True} if ok else {"accepted": False, **result})
    else:
        if not arguments.challenge:
            parser.error("a challenge needs --challenge")
        if not arguments.no_attest:
            iris("da", "attest", "--submission", arguments.challenge)
        built = json.loads(iris("evm", "challenge", "--challenge", arguments.challenge,
                                "--submission", arguments.submission))
        outcome = {"claim": built["claim"], "challenge": built["challenge"]}
        outcome.update(submit("submitChallenge", built, arguments))
    print(json.dumps(outcome, indent=2))
    sys.exit(0 if outcome["accepted"] else 1)


if __name__ == "__main__":
    main()
