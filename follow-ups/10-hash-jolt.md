# Hash Jolt

## What

An instantiation of the Jolt protocol using a hash-based polynomial commitment scheme instead of Dory. Same Protocol IR, same Module, same schedule-driven prover/verifier — different PCS.

## Why

Hash-based PCS (e.g., FRI-based, Ligero-style, or Brakedown) offers different tradeoffs:
- **No trusted setup** — hash-based schemes are transparent
- **Post-quantum security** — no reliance on discrete log or pairing assumptions
- **Different performance profile** — potentially faster proving (no MSMs), larger proofs, different verification costs
- **Field flexibility** — some hash-based PCS work over any field, not just pairing-friendly curves

## Scope

This is a research + engineering track with two phases:

**Phase 1 — Research: PCS selection**

Evaluate candidate hash-based PCS for compatibility with Jolt's structure:

| Candidate | Proof Size | Prover | Verifier | Field Requirement |
|-----------|-----------|--------|----------|-------------------|
| FRI (STARK-style) | O(log²n) | O(n log n) | O(log²n) | needs smooth domain (2-adic field) |
| Ligero / Brakedown | O(√n) | O(n) | O(√n) | any field |
| Basefold | O(log²n) | O(n log n) | O(log²n) | any field |

Key questions:
- Does Jolt's sumcheck structure (multilinear polynomials, many small openings) favor one scheme over another?
- Jolt uses BN254 scalar field. FRI needs a 2-adic field — BN254's 2-adicity is 2²⁸, which is sufficient but not as large as Goldilocks. Is this a practical concern?
- Opening batching: Jolt batches many polynomial openings via RLC reduction. Does the hash PCS support efficient batched openings?

**Phase 2 — Engineering: implementation**

Implement the chosen PCS as a new crate (e.g., `jolt-fri` or `jolt-brakedown`) that implements the `CommitmentScheme` trait from `jolt-openings`:

```rust
pub trait CommitmentScheme {
    type Field: Field;
    type Commitment;
    type ProverSetup;
    type VerifierSetup;
    type Proof;
    
    fn commit(polys: &[&[Self::Field]], setup: &Self::ProverSetup) -> Vec<Self::Commitment>;
    fn prove(claims: &[ProverClaim<Self>], setup: &Self::ProverSetup, transcript: &mut T) -> Self::Proof;
    fn verify(claims: &[VerifierClaim<Self>], setup: &Self::VerifierSetup, proof: &Self::Proof, transcript: &mut T) -> Result<()>;
}
```

Then: `prove::<CpuBackend, Fr, Blake2bTranscript, HashPCS>(module, ...)` — the rest of the pipeline is unchanged.

## How It Fits

The architecture already supports this cleanly:
- `Module` is PCS-agnostic — it describes the protocol structure, not the commitment scheme
- The prover runtime dispatches `Op::Commit` and `Op::OpeningProof` to the PCS via the generic type parameter
- The verifier's `VerifyOpenings` op calls `PCS::verify`

The only architectural question is whether the hash PCS introduces operations that don't fit the current `CommitmentScheme` trait (e.g., FRI requires domain evaluations that Dory doesn't).

## Dependencies

None — independent research track. Only requires the stable `CommitmentScheme` trait interface.

## Unblocks

- Transparent (no-trusted-setup) Jolt proofs
- Post-quantum Jolt (when combined with hash BlindFold for ZK)
- Performance exploration (hash PCS may be faster for large traces)

## Open Questions

- **Field choice:** Should hash Jolt stick with BN254's scalar field, or should we explore Goldilocks / Mersenne31 for better FRI performance? Changing the field has deep implications (all field arithmetic, all polynomial types).
- **Proof size vs. verification cost:** Hash PCS proofs are larger than Dory proofs. For on-chain use, the wrapper (05) compresses this. For off-chain use, is the proof size acceptable?
- **Commitment structure:** Dory commits to multilinear polynomials natively. FRI commits to univariates and evaluates at arbitrary points. How does this interact with Jolt's multilinear polynomial structure?
- **Streaming compatibility:** Dory supports streaming commitment (tier-1 chunks → tier-2 aggregation). Does the hash PCS need streaming support for memory efficiency?
