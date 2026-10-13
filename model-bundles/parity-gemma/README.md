# Synthetic parity Gemma

All weights are synthetic and generated locally; no downloaded model is needed.
The stable test inputs are the checked-in `model.detwgt`, `config.json`,
`tokenizer.json`, and `cases.json`, bound by `checksums.json`.

`generate.py` uses integer xorshift32 with seed `0x51A7E123` and emits DETWGT v2,
numeric specification v1, signed 32-bit Q16.16 coefficients. Regeneration needs
only Python's standard library. `python3 generate.py --check` verifies the exact
bytes without changing them. The corpus's input token IDs come from the small
independent tokenizer oracle in the generator.

See [the parity workflow](../../docs/workflows/test-parity.md) for coverage,
execution, comparison rules, and artifact locations.
