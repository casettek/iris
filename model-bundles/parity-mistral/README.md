# Synthetic parity Ministral

A four-layer `mistral-dense` model small enough to run every path of the parity gate.
All weights are synthetic and generated locally; no downloaded model is needed.
The stable test inputs are the checked-in `model.detwgt`, `config.json`,
`tokenizer.json`, and `cases.json`, bound by `checksums.json`.

It is shaped as Ministral 3 is: full attention in every layer, a gated-SiLU
feed-forward block, YaRN rotation of consecutive pairs, no per-layer inputs, a
head tied to the embedding table, and a byte-level tokenizer with the tekken
word split. Its corpus has a forty-byte merged token, blanks after line
breaks, and capitals after a letter with no case.

`generate.py` uses integer xorshift32 with seed `0x3A7E51C9` and emits DETWGT v3,
numeric specification v1: `Q8_0` blocks for the embedding table and the layer
matrices, and signed 32-bit Q16.16 coefficients for the norms. Regeneration
needs only Python's standard library. `python3 generate.py --check` verifies
the exact bytes without changing them. The corpus's input token IDs come from
the small independent tokenizer oracle in the generator.

See [the parity workflow](../../docs/workflows/test-parity.md) for coverage,
execution, comparison rules, and artifact locations.
