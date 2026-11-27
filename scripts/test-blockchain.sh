#!/usr/bin/env bash
# Blockchain tier, a few minutes: claims against the contracts on the local
# blockchains (blockchain/devstack/README.md). An honest claim must stand, and
# a claim that lies in each known way must be caught the way the protocol
# catches it:
#
#   - a checkpoint that names an unapproved program, or an input that is not
#     an earlier stage's output, a pinned commitment or a posted value: the
#     contract takes the claim, and voids it on the spot when pointed at the
#     checkpoint;
#   - a stage given a value the claim may name in a place the pipeline does not
#     give it, and a verdict stated valid that the check did not reach: the
#     contract takes the claim and refuses every on-the-spot question, and a
#     challenger's replay of the stage on the inputs its checkpoint names is a
#     challenge the contract admits.
#
# Before those, the contracts' own tests run (a real fault proof against
# `submitFaultProof`), and the contract is held to IWEBS on changed copies of
# the honest claim and of the first challenge.
#
# Last, a claim of the `relay` procedure kind, on the run the gate's relay
# scenario made: it must stand, and a lie in its second inference must be
# challenged by a replay of a run rebuilt from what the claim posted.
#
# usage: test-blockchain.sh [--deploy] [--real-model <specification>]
#   --deploy       build and deploy the contracts first (new addresses: claims
#                  made before are not on them)
#   --real-model   also build a claim of this specification's model, on the
#                  `case.toml` beside it, and make the two replayed lies of it. Its check stage is long enough to
#                  replay, which the small model's is not. A few minutes on a
#                  GPU build for a short prompt; no proof is made.
#
# Needs the local blockchains up (`just devstack-up`), the contracts deployed, and
# a parity run of the gemma family's small model for the small claims (`just
# test-parity`); reads the newest such run under target/parity, or the one
# IWEBS_PARITY_RUN names. The cases are written for that model: on the other
# family's small model a layer stage given another layer's weights is under
# the replay window, which `--real-model` covers at full size. Every run
# makes its claims for fresh addresses, so it can be repeated on one
# deployment. Everything is kept under target/blockchain-test/<run>.
# See docs/workflows/testing.md.
source "$(dirname "${BASH_SOURCE[0]}")/test-env.sh"
source "$repo_root/blockchain/devstack/env.sh"

deploy=0
real_specification=""
while [ $# -gt 0 ]; do
    case $1 in
    --deploy) deploy=1 ;;
    --real-model)
        real_specification=$(realpath "${2:?--real-model needs a specification}")
        shift
        ;;
    *)
        echo "usage: test-blockchain.sh [--deploy] [--real-model <specification>]" >&2
        exit 2
        ;;
    esac
    shift
done

echo "== blockchain tier"

rpc_answers() {
    curl -s -m 5 -o /dev/null -X POST -H 'content-type: application/json' \
        --data '{"jsonrpc":"2.0","id":1,"method":"eth_chainId","params":[]}' "$IWEBS_EVM_RPC" &&
        curl -s -m 5 -o /dev/null "$IWEBS_CELESTIA_RPC"
}
if ! rpc_answers; then
    echo "the local blockchains do not answer at $IWEBS_EVM_RPC and $IWEBS_CELESTIA_RPC: run \`just devstack-up\`" >&2
    exit 1
fi
# The deployment file is written by the binary, so it is built first.
step blockchain-build "release binary built" \
    cargo build --release --locked -p iwebs-cli --bin iwebs "${features[@]}"
if [ $deploy -eq 1 ]; then
    step blockchain-deploy "contracts built and deployed" bash "$repo_root/blockchain/devstack/deploy.sh"
fi
# The EVM node keeps nothing across a restart, so a deployment file can
# outlive its contracts.
deployed() {
    local claims code
    claims=$(sed -n 's/.*"claims": *"\(0x[0-9a-fA-F]*\)".*/\1/p' "$repo_root/blockchain/devstack/deployment.json" 2> /dev/null | head -1)
    [ -n "$claims" ] || return 1
    code=$(curl -s -m 5 -X POST -H 'content-type: application/json' \
        --data "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"eth_getCode\",\"params\":[\"$claims\",\"latest\"]}" \
        "$IWEBS_EVM_RPC")
    [[ "$code" == *'"result":"0x'[0-9a-f]* ]]
}
if ! deployed; then
    echo "no claims contract is deployed on the local EVM node: run \`just devstack-deploy\`, or this tier with --deploy" >&2
    exit 1
fi

# The newest parity run of the gemma family's small model.
parity_run="${IWEBS_PARITY_RUN:-}"
if [ -z "$parity_run" ]; then
    for run in $(ls -td "$repo_root"/target/parity/*/ 2> /dev/null); do
        if [ -d "$run/model-artifacts/parity-gemma" ]; then
            parity_run="${run%/}"
            break
        fi
    done
fi
small_specification="$parity_run/over-window/specification.toml"
relay_specification="$parity_run/relay/run/specification.toml"
if [ -z "$parity_run" ] || [ ! -f "$small_specification" ]; then
    echo "no parity run of the gemma family's small model under target/parity: run the parity gate first (\`just test-parity\`)" >&2
    exit 1
fi

run_id=$(date +%s)
out="$repo_root/target/blockchain-test/$run_id"
mkdir -p "$out"
echo "   kept under target/blockchain-test/$run_id"
iwebs="$repo_root/target/release/iwebs"
export IWEBS_BINARY="$iwebs"
# The cases run in directories of their own.
export IWEBS_BLOCKCHAIN_DEPLOYMENT="$repo_root/blockchain/devstack/deployment.json"

# A fresh address per run and name: twenty bytes of the hash of both.
address() {
    printf '0x%s' "$(printf '%s' "$run_id $1" | sha256sum | cut -c1-40)"
}

# absent <file> <fixed text>: no line of the file contains the text.
absent() {
    if grep -qF -- "$2" "$1"; then
        echo "did not expect \`$2\` in $1" >&2
        return 1
    fi
}

# Each case below is a function whose body is a subshell that stops at the
# first failing command.

# expect <file> <fixed text>: the file has a line containing the text.
expect() {
    grep -qF -- "$2" "$1" || {
        echo "expected \`$2\` in $1" >&2
        return 1
    }
}

# The case of a specification here: `case.toml` beside it.
case_of() {
    printf '%s/case.toml' "$(dirname "$1")"
}

# build_claim <name> <specification> [ids]: a claim built in its own directory,
# of the model's own output or of the ids given; writes the submission's path
# to <name>.submission.
build_claim() (
    set -euo pipefail
    local name=$1 specification=$2 ids=${3:-}
    local build=(claim build) case
    case=$(case_of "$specification")
    [ -n "$ids" ] && build=(test claim --ids "$ids")
    mkdir -p "$out/$name"
    cd "$out/$name"
    # A claim states its output. For the model's own, the output is found
    # first, by the procedure that finds one.
    [ -n "$ids" ] || case=$(python3 "$repo_root/scripts/state-output.py" "$iwebs" "$specification" "$case" "$out/$name/stated")
    "$iwebs" "${build[@]}" --specification "$specification" --case "$case" --claimer "$(address "$name")" > build.log 2>&1 || {
        tail -5 build.log >&2
        exit 1
    }
    sed -n 's/^claim submission: //p' build.log | tail -1 > "$out/$name.submission"
    [ -s "$out/$name.submission" ]
)

# Ids that are not a model's output: as many as the specification's token budget.
other_ids() {
    local tokens
    tokens=$(sed -n 's/^max_new_tokens *= *\([0-9]*\).*/\1/p' "$1" | head -1)
    seq -s, 11 $((10 + ${tokens:-1}))
}

# The last check stage of a claim: the stage its verdict is stated by.
last_check() {
    grep -o '"stage": *"[A-Za-z0-9_.-]*check_c[0-9]*"' "$(dirname "$1")/checkpoint_list.json" |
        tail -1 | sed 's/.*"\([^"]*\)"$/\1/'
}

# submitted <claim_submission.json> <specification>: posted, and taken by the
# contract.
submitted() {
    "$iwebs" claim post --submission "$1"
    "$iwebs" test devstack submit claim --specification "$2" --submission "$1" > "$1.arrival.json"
    expect "$1.arrival.json" '"accepted": true'
}

# audited <fetched dir> <specification> <faults>: every on-the-spot question
# asked of the contract, which answers each as IWEBS does and holds this many
# faults.
audited() {
    "$iwebs" test devstack replay faults --specification "$2" \
        --submission "$1/claim_submission.json" | tee "$1.faults.log"
    if [ "$3" = none ]; then
        expect "$1.faults.log" '"faults": 0}'
    else
        expect "$1.faults.log" "$3"
        absent "$1.faults.log" '"faults": 0}'
    fi
}

honest_claim() (
    set -euo pipefail
    local claim specification=$2
    claim=$(cat "$out/$1.submission")
    "$iwebs" claim post --submission "$claim"
    # IWEBS and the contract on the claim and on changed copies of it, then
    # the claim itself submitted.
    "$iwebs" test devstack replay claim --specification "$specification" --submission "$claim"
    "$iwebs" claim fetch --submission "$claim" --specification "$specification" --out "$out/$1-fetched"
    audited "$out/$1-fetched" "$specification" none
)

# voided_lie <name> <source claim> <specification> <expected fault> <corrupt arguments>
voided_lie() (
    set -euo pipefail
    local name=$1 claim specification=$3 fault=$4
    claim=$(cat "$out/$2.submission")
    shift 4
    "$iwebs" test corrupt-claim --submission "$claim" --claimer "$(address "$name")" --output "$out/$name-lie" "$@"
    submitted "$out/$name-lie/claim_submission.json" "$specification"
    "$iwebs" claim fetch --submission "$out/$name-lie/claim_submission.json" --specification "$specification" --out "$out/$name"
    audited "$out/$name" "$specification" "$fault"
)

# replayed_lie <name> <source claim> <specification> <stage> <must replay: yes|no>
#              <hold the challenge to IWEBS: yes|no> <corrupt arguments>
replayed_lie() (
    set -euo pipefail
    local name=$1 claim specification=$3 stage=$4 must_replay=$5 held=$6
    claim=$(cat "$out/$2.submission")
    shift 6
    "$iwebs" test corrupt-claim --submission "$claim" --claimer "$(address "$name")" --output "$out/$name-lie" \
        --stage "$stage" "$@"
    # A lie a replay shows arrives as any claim does, here with a valid record.
    expect "$out/$name-lie/claim_submission.json" '"valid": true'
    absent "$out/$name-lie/claim_submission.json" '"valid": false'
    submitted "$out/$name-lie/claim_submission.json" "$specification"
    "$iwebs" claim fetch --submission "$out/$name-lie/claim_submission.json" --specification "$specification" --out "$out/$name"
    audited "$out/$name" "$specification" none

    mkdir -p "$out/$name-challenge"
    cd "$out/$name-challenge"
    local built=0
    # The challenger's own specification. The case is the one `claim fetch`
    # wrote beside the submission.
    "$iwebs" challenge build --specification "$specification" \
        --submission "$out/$name/claim_submission.json" > build.log 2>&1 || built=$?
    grep -v '^IWEBS_CHALLENGE_PROGRESS' build.log | tail -12
    expect build.log "stage to replay: $stage"
    if [ $built -ne 0 ]; then
        # The one reason a replay may not be built here: the stage is under
        # the replay window on the small model
        # (docs/issues/short-stage-fault-window.md).
        if [ "$must_replay" = yes ]; then
            echo "no replay of $stage was built" >&2
            exit 1
        fi
        expect build.log "trace length outside commitment limits"
        echo "found at $stage, which is too short to replay on this model"
        exit 0
    fi
    expect build.log "divergence found: $stage"
    local bundle
    bundle=$(sed -n 's/^challenge bundle: //p' build.log | tail -1)
    "$iwebs" challenge post --challenge "$bundle" --submission "$out/$name/claim_submission.json" \
        --specification "$specification" --challenger "$(address "$name challenger")"
    local challenge
    challenge="$(dirname "$bundle")/challenge_submission.json"
    if [ "$held" = yes ]; then
        # IWEBS and the contract on the challenge and on changed copies of
        # it, then the challenge itself submitted.
        "$iwebs" test devstack replay challenge --specification "$specification" \
            --submission "$out/$name/claim_submission.json" --challenge "$challenge"
    else
        "$iwebs" test devstack submit challenge \
            --submission "$out/$name/claim_submission.json" --challenge "$challenge" | tee challenge.json
        expect challenge.json '"accepted": true'
    fi
)

step blockchain-contracts "contracts' own tests, with a real fault proof" \
    bash "$repo_root/blockchain/devstack/foundry.sh" forge test

# build_claims <prefix> <specification>: an honest claim, and a claim of ids that
# are not the model's output.
build_claims() (
    set -euo pipefail
    build_claim "$1honest" "$2"
    build_claim "$1other-ids" "$2" "$(other_ids "$2")"
)
step blockchain-claims "small model: an honest claim and one of other ids built" \
    build_claims "" "$small_specification"
step blockchain-honest "honest claim: taken, and every question refused" \
    honest_claim honest "$small_specification"

stage=prompt_check_p1
step blockchain-void-program "unapproved program: voided on the spot" \
    voided_lie void-program honest "$small_specification" UnapprovedProgram --stage $stage --fault program
step blockchain-void-piped "input not its producer's output: voided on the spot" \
    voided_lie void-piped honest "$small_specification" NotAnEarlierOutput --stage $stage --fault input
step blockchain-void-pinned "input not a pinned commitment: voided on the spot" \
    voided_lie void-pinned honest "$small_specification" NotAPinnedValue --stage $stage --fault input --input tokenizer
step blockchain-void-posted "input not a posted value: voided on the spot" \
    voided_lie void-posted honest "$small_specification" NotAPostedValue --stage $stage --fault input --input marks

step blockchain-arranged-pinned "another layer's weights: challenged by a replay" \
    replayed_lie arranged-pinned honest "$small_specification" range_p1_l2 yes yes --fault arrangement --input layer
step blockchain-arranged-posted "another range's marks: challenged by a replay" \
    replayed_lie arranged-posted honest "$small_specification" $stage yes no --fault arrangement --input marks
step blockchain-arranged-piped "another stage's output: challenged by a replay" \
    replayed_lie arranged-piped honest "$small_specification" $stage yes no --fault arrangement --input bound
step blockchain-arranged-empty "empty value for an earlier output: challenged by a replay" \
    replayed_lie arranged-empty honest "$small_specification" $stage yes no --fault arrangement --input previous

forged_verdict() {
    local check
    check=$(last_check "$(cat "$out/$2.submission")")
    replayed_lie "$1" "$2" "$3" "$check" "$4" no --fault verdict
}
step blockchain-forged-verdict "forged verdict: arrives valid, found at the last check" \
    forged_verdict forged-verdict other-ids "$small_specification" no

# A relay: two models, the second on the first one's output. Its submission
# holds the relay's record, which the contract holds to the first inference's
# claim record and to the recipe the procedure pins.
if [ -f "$relay_specification" ]; then
    step blockchain-relay-claim "relay: an honest claim built" \
        build_claim relay-honest "$relay_specification"
    step blockchain-relay-honest "relay: taken, and every question refused" \
        honest_claim relay-honest "$relay_specification"
    step blockchain-relay-arranged "relay, another range's marks in the second inference: challenged by a replay" \
        replayed_lie relay-arranged relay-honest "$relay_specification" m1_prompt_check_p1 yes no --fault arrangement --input marks
else
    echo "the parity run has no relay scenario at $relay_specification: run the parity gate again" >&2
    failed=1
fi

if [ -n "$real_specification" ]; then
    step blockchain-real-claims "real model: an honest claim and one of other ids built" \
        build_claims real- "$real_specification"
    step blockchain-real-arranged "real model, another layer's weights: challenged by a replay" \
        replayed_lie real-arranged real-honest "$real_specification" range_p0_l1 yes no --fault arrangement --input layer
    step blockchain-real-verdict "real model, forged verdict: challenged by a replay" \
        forged_verdict real-verdict real-other-ids "$real_specification" yes
fi

exit $failed
