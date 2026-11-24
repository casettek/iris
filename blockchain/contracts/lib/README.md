# Vendored contract sources

Sources iris's contracts import, copied unchanged. Nothing here is iris's.

| Directory | From | Release | Licence |
| --- | --- | --- | --- |
| `risc0-ethereum/src` | https://github.com/risc0/risc0-ethereum, `contracts/src` | v3.0.1 | Apache-2.0 (`risc0-ethereum/LICENSE`) |
| `openzeppelin/contracts` | https://github.com/OpenZeppelin/openzeppelin-contracts, `contracts` | v5.1.0 | MIT (`openzeppelin/LICENSE`) |

From `risc0-ethereum`: the Groth16 verifier (`groth16/RiscZeroGroth16Verifier.sol`,
`groth16/Groth16Verifier.sol`, `groth16/ControlID.sol`) and what it imports
(`IRiscZeroVerifier.sol`, `IRiscZeroSelectable.sol`, `StructHash.sol`,
`Util.sol`). From OpenZeppelin: `utils/math/SafeCast.sol`, which the verifier
imports.

The verifier takes proofs of one release of RISC Zero's circuits, fixed by
`ControlID.CONTROL_ROOT`. It has to be the release Raster is built with:
`ALLOWED_CONTROL_ROOT` in the `risc0-circuit-recursion` crate that iris's
`Cargo.lock` names is the same value. `contracts/test/FaultProof.t.sol`
verifies a real fault proof with it.

To move to another release, replace the files from the matching tag and run
`just contracts-test` with a fault proof made by the new build.
