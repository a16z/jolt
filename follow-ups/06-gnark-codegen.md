# Gnark Groth16 Wrapper

## What

Generate gnark (Go) circuit code that implements a verifier for the Spartan-HyperKZG wrapper proof (05). The gnark circuit compiles to Groth16, producing a constant-size proof (~128 bytes) verifiable on-chain in a smart contract.

This is the second wrapping layer. The Spartan-HyperKZG wrapper (05) is a native Jolt protocol that runs on the jolt-zkvm runtime. The gnark wrapper is external codegen targeting a different proving system (Groth16 via gnark in Go).

## Why

The Spartan-HyperKZG wrapper proof is already much smaller than the inner Jolt proof, but on-chain verification still requires sumcheck verification + HyperKZG pairing checks. Groth16 compresses this to ~3 pairings regardless of circuit size, which is the cheapest possible on-chain verification.

## Scope

**Input:** The wrapper Module's `VerifierSchedule` (from 05) — this describes the Spartan-HyperKZG verifier's computation.

**Output:** Go source code using the gnark constraint API.

**What gets codegen'd:**
1. **Transcript simulation:** `Squeeze` → gnark algebraic hash gadget (MiMC or Poseidon). Blake2b is not efficient in-circuit, so the wrapper's transcript must use an algebraic hash.
2. **Sumcheck verification:** Loop over rounds, constrain round_poly consistency.
3. **Claim formulas:** `ClaimFormula` evaluation as gnark multiplications/additions.
4. **HyperKZG verification:** KZG pairing check — gnark has native BN254 pairing gadgets.

**Pipeline:**
```
wrapper Module (05)  →  gnark codegen  →  Go circuit  →  gnark compile
                                                              ↓
                                                     Groth16 proving key
                                                     Groth16 verifying key
                                                              ↓
                                                     Solidity verifier (ExportSolidity)
```

This is a one-time codegen + compilation step per protocol version. The Groth16 keys are reused across all proofs.

**Alternatively:** The gnark codegen could consume the R1CS from Module (02) directly — walking the R1CS constraint matrices and emitting gnark `api.AssertIsEqual` for each constraint. This avoids re-encoding the verifier schedule and instead just translates the already-derived R1CS. Whether to walk the Module or the R1CS is a design choice.

## How It Fits

This lives in `jolt-wrapper` (currently a stub). It's a codegen tool, not a runtime component — it produces Go source files that are compiled separately by gnark.

The gnark circuit encodes the **wrapper** verifier (05), not the inner Jolt verifier. The wrapper is small (Spartan over a modest R1CS with HyperKZG), so the gnark circuit is small.

## Dependencies

- Wrapper protocol (05) — the gnark circuit verifies the wrapper proof
- R1CS from Module (02) — either directly consumed, or indirectly via the wrapper Module

## Unblocks

- On-chain verification (the final step)
- jolt-zkvm composition (07)

## Open Questions

- **Walk Module vs. walk R1CS:** Should codegen traverse the `VerifierSchedule` ops (higher-level, more structure to exploit) or the derived R1CS matrices (lower-level, more mechanical)?
- **Transcript hash:** The gnark circuit needs an algebraic hash for Fiat-Shamir. This means the Spartan-HyperKZG wrapper (05) must also use an algebraic hash in its transcript (so the gnark circuit can reproduce it). Does the `Transcript` trait support pluggable hash functions?
- **Circuit size:** What's the target Groth16 constraint count? The wrapper's R1CS size determines this. Groth16 at 10M constraints takes ~10s to prove.
- **Alternative backends:** Should we also support Circom/snarkjs or Plonk for broader compatibility?
