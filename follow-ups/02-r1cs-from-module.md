# R1CS from Module

## What

A compiler pass that takes a Module (pre-linking, since linking doesn't affect semantics) and mechanically derives R1CS constraint matrices encoding the verifier's computation. Both BlindFold and the Spartan-HyperKZG wrapper consume these constraints.

## Why

Today, BlindFold manually constructs its verifier R1CS from `StageConfig` and `BakedPublicInputs` — hand-written constraint generation that must stay synchronized with the verifier schedule. The wrapper similarly needs an R1CS encoding of the verifier. Both are doing the same thing: "given a verifier schedule, what R1CS checks it?"

A generic `Module → R1CS` pass eliminates this duplication and makes both ZK and wrapping automatic consequences of the protocol definition.

## Scope

The verifier schedule (`VerifierSchedule`) is a sequence of `VerifierOp`s: absorb commitments, squeeze challenges, verify sumcheck, record evals, check output formulas, verify openings. The R1CS derivation would:

1. **Transcript simulation:** Each `Squeeze` produces a challenge variable. Each `Absorb*` constrains the transcript state transition. The R1CS encodes the full Fiat-Shamir derivation.

2. **Sumcheck verification:** Each `VerifySumcheck` expands into constraints checking that round polynomial evaluations at 0 and 1 sum to the claimed value, and that the next claim is derived correctly from the challenge.

3. **Claim formulas:** Each `CheckOutput` evaluates a `ClaimFormula` — a sum of products over polynomial evaluations, challenges, and constants. These become multiplication + addition constraints.

4. **Opening verification (PCS-parametric via primitives):** PCS verification is inherently PCS-specific (Dory: pairings + GT exponentiation, HyperKZG: KZG pairing check, hash PCS: Merkle path verification). Rather than hardcoding one PCS, the R1CS derivation uses a **primitive extension mechanism**.

   The Protocol IR gains a notion of **primitives** — opaque operations with declared input/output types. Each PCS registers its verification logic as a set of primitives (e.g., `Pairing(G1, G2) → Gt`, `GtExp(Gt, scalar) → Gt`, `MerkleVerify(root, leaf, path) → bool`). The R1CS emitter accepts a pluggable `PrimitiveConstraintGenerator` that knows how to encode each primitive as R1CS constraints.

   The Module says *what* to verify (which polynomial openings, at which points). The primitive spec says *how* to verify it (what constraints encode the PCS verification equation). This keeps the Module PCS-agnostic while making the R1CS derivation PCS-complete.

   Primitives are a lightweight extension point. If a PCS's verification is complex enough to warrant its own sumcheck PIOPs (as the recursion paper does for Dory — see 14), the primitive can later be promoted to a full sub-Protocol. But primitives are the right starting abstraction.

The output is `ConstraintMatrices<F>` (the A, B, C sparse matrices) that can be consumed by `R1csKey::new()`.

## How It Fits

Input: `Module` (from `jolt-compiler`)
Output: `ConstraintMatrices<F>` (from `jolt-r1cs`)

This would likely live in `jolt-r1cs` as a new module (e.g., `jolt-r1cs/src/from_module.rs`) or as a separate thin crate if the dependency on `jolt-compiler` is undesirable in `jolt-r1cs`.

BlindFold (`jolt-blindfold`) currently has its own `VerifierR1CS` and `VerifierR1CSBuilder`. These would be replaced by consuming the output of this pass.

The Spartan-HyperKZG wrapper (`wrapper.jolt`) would use the same output to build its proving/verification keys.

## Dependencies

- Compiler full protocol coverage (01) — the Module must represent the full protocol for the derived R1CS to be complete
- `jolt-r1cs` must support the constraint structure (it already does via `ConstraintMatrices`)

## Unblocks

- BlindFold rewrite (04)
- Wrapper protocol (05)
- Gnark codegen (06) — indirectly, since gnark needs to verify the same R1CS

## Open Questions

- **Primitive granularity:** How fine-grained should primitives be? A single `VerifyDoryOpening` primitive is simple but opaque to the R1CS emitter. Individual primitives (`Pairing`, `GtExp`, `G1ScalarMul`) are composable but require the Module to spell out the PCS verification algorithm. Where's the right level?
- **Primitive constraint generators:** Should each PCS crate ship its own `PrimitiveConstraintGenerator` impl, or should there be a shared library of common crypto-operation constraints (pairing, hash, EC arithmetic)?
- BlindFold's R1CS includes Nova folding constraints (relaxed R1CS, cross-term). Should those be part of this pass, or layered on top?
- What's the witness assignment strategy? The R1CS variables are transcript state, challenges, round polynomial evaluations, claims. The prover knows all of these. Does the witness builder live here or in the consumer (BlindFold/wrapper)?
