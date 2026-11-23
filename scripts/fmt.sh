#!/usr/bin/env bash
# Format iris's host code with rustfmt, and nothing else.
#
# usage: fmt.sh [--check]
#
# `cargo fmt --all` must not be run here. It follows path dependencies into
# the Raster checkout and rewrites it, and it rewrites the code the stage
# programs are built from. A stage program's identity covers its sources down
# to the line numbers in their panic messages, so formatting that code changes
# every program id, which means a re-lock of every stage, new contract pins
# and a new real-model reference.
#
# So this formats only what no stage program is built from: every tracked Rust
# file except the stage crates, the crates they depend on, and the files they
# take by symbolic link. Each file is formatted alone through standard input,
# because rustfmt given a path also formats the modules that file declares,
# and some of those are the stage programs' files. The stage code is formatted
# only together with a re-lock.
#
# With --check nothing is written: the files that are not formatted are listed
# and the exit status is 1 when there are any.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

check=0
if [ "${1:-}" = "--check" ]; then
    check=1
elif [ $# -gt 0 ]; then
    echo "usage: fmt.sh [--check]" >&2
    exit 2
fi

# What the stage programs are built from: the stage crates, the crates they
# depend on, and whatever a symbolic link in either leads to.
guest_dirs=(raster-stages crates/det-num crates/model/identity crates/inference/pipeline-core)

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
guest_list="$work/guest-files"
{
    git ls-files --cached --others --exclude-standard -- "${guest_dirs[@]}" | while read -r file; do
        # A link to a directory brings in every file under it.
        if [ -d "$file" ]; then
            find -L "$file" -name '*.rs' -type f
        else
            echo "$file"
        fi
    done | while read -r file; do
        realpath --relative-to="$repo_root" "$file"
    done
} | sort -u > "$guest_list"

unformatted=0
formatted=0
while read -r file; do
    # A tracked link is the file it leads to, listed under its own path.
    [ -L "$file" ] && continue
    # Deleted here and not yet committed.
    [ -f "$file" ] || continue
    grep -qxF "$file" "$guest_list" && continue
    if ! rustfmt --edition 2021 < "$file" > "$work/formatted.rs" 2> "$work/rustfmt.err"; then
        echo "rustfmt failed on $file:" >&2
        cat "$work/rustfmt.err" >&2
        exit 2
    fi
    if cmp -s "$file" "$work/formatted.rs"; then
        continue
    elif [ $check -eq 1 ]; then
        echo "not formatted: $file"
        unformatted=$((unformatted + 1))
    else
        # Keep the file and its mode: write over it in place.
        cat "$work/formatted.rs" > "$file"
        echo "formatted: $file"
        formatted=$((formatted + 1))
    fi
done < <(git ls-files --cached --others --exclude-standard -- "*.rs")

if [ $check -eq 1 ]; then
    if [ $unformatted -gt 0 ]; then
        echo "$unformatted file(s) not formatted: run \`just fmt\`" >&2
        exit 1
    fi
    echo "every host file is formatted"
else
    echo "$formatted file(s) formatted"
fi
