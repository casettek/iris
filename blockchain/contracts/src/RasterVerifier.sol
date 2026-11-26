// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

// RISC Zero's verifier of a Groth16 proof, unchanged, from lib/risc0-ethereum.
// Imported here so that it is built with IWEBS's contracts and can be deployed
// beside them.
import {RiscZeroGroth16Verifier} from "risc0/groth16/RiscZeroGroth16Verifier.sol";
import {ControlID} from "risc0/groth16/ControlID.sol";

/// The verifier with the control ids of the vendored release, which fix the
/// release of RISC Zero's circuits it takes proofs of, so that a deployment
/// gives it nothing to get wrong.
contract RasterVerifier is RiscZeroGroth16Verifier(ControlID.CONTROL_ROOT, ControlID.BN254_CONTROL_ID) {}
