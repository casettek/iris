#!/usr/bin/env python3
"""Run the procedure that finds a case's output, and write the case a
verification procedure is given: the same case, with that output stated.

usage: state-output.py <iwebs binary> <specification> <case> <out dir>

The specification is a verification procedure's. Nothing in IWEBS pairs it
with a generation procedure, so this script does what anyone does who needs an
output verified: it writes the specification again with the kind that
generates, runs it, and states what it found in a copy of the case. It prints
the path of that case.

Written in the out dir: `generation.toml`, `generation.log`,
`generation-record/` and `case.toml`.
"""
import json
import pathlib
import re
import subprocess
import sys


def absolute(text, keys, base):
    """Paths under these keys, made absolute: the files move, what they name does not."""

    def fix(match):
        path = pathlib.Path(match.group(2))
        return f'{match.group(1)} = "{path if path.is_absolute() else base / path}"'

    return re.sub(rf'^({"|".join(keys)}) = "([^"]*)"', fix, text, flags=re.M)


def main():
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    iwebs, specification, case, out = (pathlib.Path(arg).resolve() for arg in sys.argv[1:])
    out.mkdir(parents=True, exist_ok=True)

    text = specification.read_text()
    named = re.search(r'^procedure_kind = "([a-z0-9-]+)"', text, flags=re.M)
    kind = named.group(1) if named else "verify-single"
    if not kind.startswith("verify-"):
        sys.exit(f"{specification} names the `{kind}` procedure kind, which verifies no output")
    generates = "generate-" + kind[len("verify-"):]
    text = re.sub(r'^procedure_kind = "[a-z0-9-]+"\n', "", text, flags=re.M)
    text = f'procedure_kind = "{generates}"\n' + absolute(text, ["manifest"], specification.parent)
    generation = out / "generation.toml"
    generation.write_text(text)

    record = out / "generation-record"
    with (out / "generation.log").open("w") as log:
        ran = subprocess.run(
            [iwebs, "run", "--specification", generation, "--case", case, "--record", record],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if ran.returncode != 0:
        sys.stderr.write((out / "generation.log").read_text()[-2000:])
        sys.exit(f"the generation failed: see {out / 'generation.log'}")

    stated = absolute(case.read_text(), ["file", "attestation", "prompt_file"], case.parent)
    for result in json.loads((record / "record.json").read_text())["results"]:
        stated += f'\n[[output]]\nids = {json.dumps(result["generated_token_ids"])}\n'
    path = out / "case.toml"
    path.write_text(stated)
    print(path)


if __name__ == "__main__":
    main()
