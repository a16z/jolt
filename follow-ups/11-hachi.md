# Hachi — Lattice Jolt

## What

Integrate the existing [`hachi-pcs`](https://github.com/LayerZero-Labs/hachi) crate as a new PCS backend for Jolt, replacing Dory with a lattice-based polynomial commitment scheme. Hachi provides post-quantum security, transparent setup (no trusted ceremony), and additive homomorphism — meaning BlindFold works without protocol changes.

## Why

- **Post-quantum security** — lattice assumptions (Module-SIS/LWE) resist quantum attacks, unlike BN254 pairings
- **Transparent setup** — public parameters only, no toxic waste
- **Additive homomorphism** — lattice module commitments support linear combinations, preserving BlindFold's Nova folding (unlike hash commitments which require a fundamentally different ZK protocol)
- **Native one-hot support** — Hachi already supports one-hot polynomial commitments, which Jolt uses extensively for RA polynomials

## What Hachi Provides

Hachi (`hachi-pcs`) is a concrete, existing Rust crate:

- **Field:** Small pseudo-Mersenne primes (q = 2^k - offset) for CRT/NTT arithmetic. This is a different field than BN254's scalar field.
- **Commitment:** Lattice module elements (vectors over Z_q). Additively homomorphic — `commit(a) + commit(b) = commit(a + b)`.
- **Opening protocol:** Sumcheck-based with recursive level scheduling. `HachiProof` contains `HachiLevelProof` layers + `HachiProofTail`.
- **Ring-switch proofs:** Field-to-ring opening reduction.
- **Dependencies:** Purely symmetric crypto (Blake2, SHA-3, AES-CTR). No pairing or elliptic curve libraries.
- **Features:** `parallel` (Rayon), `disk-persistence` for large commitments.

## Scope

**1. Integration as a `CommitmentScheme` impl**

Wrap `hachi-pcs` in a new crate (`jolt-hachi` or adapt `hachi-pcs` directly) implementing the `CommitmentScheme` trait from `jolt-openings`:

```rust
impl CommitmentScheme for HachiCommitmentScheme {
    type Field = HachiField;  // pseudo-Mersenne prime field
    type Commitment = HachiCommitment;
    type ProverSetup = HachiParams;
    type VerifierSetup = HachiParams;  // transparent — same params
    type Proof = HachiProof;
    // ...
}
```

**2. Field integration**

Hachi uses its own field (pseudo-Mersenne primes). The `Field` trait in `jolt-field` must accommodate this. Options:
- Implement `jolt_field::Field` for Hachi's field type
- Or use a newtype wrapper that bridges Hachi's field API to `jolt-field`'s trait

The rest of the pipeline (compiler, compute, witness, runtime) is generic over `F: Field`, so they work unchanged.

**3. Compute backend support**

Hachi's field arithmetic (NTT over small CRT primes, modular reduction) differs from BN254. The CPU backend needs kernels for Hachi's field operations. The lattice commitment (matrix-vector products) may also benefit from GPU acceleration.

**4. BlindFold with Hachi**

BlindFold should be generic over additively homomorphic commitments (see 04). Hachi's lattice commitments are additively homomorphic, so BlindFold's Nova folding works without protocol changes. The commitment operations change (lattice module operations instead of EC scalar mul/add), but the protocol structure is identical.

## How It Fits

```rust
// Standard Jolt (current)
prove::<CpuBackend, BN254Fr, Blake2bTranscript, DoryPCS>(module, ...)

// Hachi Jolt (new)
prove::<CpuBackend, HachiField, Blake2bTranscript, HachiPCS>(module, ...)
```

Same Module, same compiler, same runtime. Different field and PCS type parameters.

Crate structure:
```
jolt-hachi (wraps hachi-pcs, implements CommitmentScheme)
jolt-field (extended with HachiField impl, or bridge newtype)
jolt-blindfold (generic over AdditivelyHomomorphicCommitment → works with both Pedersen and Hachi)
```

## Dependencies

- BlindFold rewrite (04) — BlindFold must be generic over commitment scheme
- Compiler full protocol (01) — the Module must be complete

## Unblocks

- Post-quantum Jolt proving with efficient ZK (BlindFold preserved)
- Demonstrates the modularity of the architecture

## Open Questions

- **Field compatibility:** Hachi's pseudo-Mersenne field is different from BN254's scalar field. Does this affect the Protocol IR or witness construction? The IR is field-agnostic, but the witness extraction (`jolt-host`) may have BN254-specific assumptions.
- **Proof size:** Lattice proofs are larger than Dory proofs. What's the concrete overhead? Is wrapping (05) still necessary/feasible with Hachi?
- **Performance profile:** Hachi's NTT-based operations have different performance characteristics than Dory's MSMs. Which is faster for Jolt's specific polynomial sizes and commitment patterns?
- **On-chain verification:** Hachi proofs can't be verified by a BN254 pairing check. What's the on-chain verification story? Does the wrapper (05) need a Hachi-specific variant, or does extended P/V (14) not apply here?
- **Dual instantiation:** Should the codebase support running both Dory and Hachi simultaneously (e.g., user chooses at compile time via feature flags)?
