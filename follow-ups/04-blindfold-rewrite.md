# BlindFold Rewrite

## What

Rewrite the BlindFold zero-knowledge protocol (`jolt-blindfold`) to consume R1CS constraints derived from the Module rather than hand-building its own verifier R1CS. This makes BlindFold a generic ZK layer over any schedule-driven protocol.

## Why

The current BlindFold implementation (on the `refactor/crates` branch, currently a stub) was tightly coupled to jolt-core's hardcoded stage structure. It manually constructed `VerifierR1CS` from `StageConfig` and `BakedPublicInputs`, with hand-written constraints that had to stay synchronized with the prover/verifier — the "critical invariant" documented in CLAUDE.md. Any protocol change required matching constraint updates.

With R1CS-from-Module (02), BlindFold becomes generic: it receives constraint matrices and a witness assignment, applies Nova folding, and proves the folded instance via Spartan. It no longer needs to know what protocol it's making zero-knowledge.

## Scope

**Core protocol (unchanged in principle):**
1. During sumcheck stages, the prover sends commitments to round polynomial coefficients (using an additively homomorphic scheme) instead of cleartext coefficients
2. The verifier's checks (round poly consistency, claim derivation) are encoded as R1CS constraints
3. Nova folds the real instance with a random satisfying instance (using the homomorphism)
4. Spartan proves the folded relaxed R1CS
5. Hyrax-style openings verify witness evaluations

**What changes:**
- The R1CS comes from the Module-derived pass (02) instead of being hand-built
- `StageConfig` and `BakedPublicInputs` are replaced by the Module + transcript-derived public inputs
- The `input_claim_constraint()` / `output_claim_constraint()` synchronization requirement disappears — constraints are mechanically derived from the same source the verifier uses
- `SumcheckInstanceParams` no longer needs constraint methods; it only needs `input_claim()` for the non-ZK path

**Generic over additively homomorphic commitments:**

BlindFold must be generic over the commitment scheme used for round polynomials. Both elliptic curve commitments (Pedersen) and lattice commitments (Hachi) are additively homomorphic, so the same protocol works with both. This requires a trait:

```rust
trait AdditivelyHomomorphicCommitment {
    type Commitment;
    type Key;
    type Opening;
    
    fn commit(key: &Self::Key, values: &[F], randomness: &Self::Opening) -> Self::Commitment;
    fn add(a: &Self::Commitment, b: &Self::Commitment) -> Self::Commitment;
    fn scale(c: &Self::Commitment, scalar: F) -> Self::Commitment;
    fn verify(key: &Self::Key, commitment: &Self::Commitment, values: &[F], opening: &Self::Opening) -> bool;
}
```

Pedersen (over BN254/Grumpkin) and Hachi (lattice modules) both implement this. BlindFold is generic over it. Hash commitments do NOT implement this — they require a fundamentally different ZK approach (13).

**Witness assignment:**
The BlindFold witness consists of: transcript state at each step, challenge values, round polynomial coefficients, claimed sums, polynomial evaluations. All of these are available to the prover during execution. The witness builder maps runtime values to R1CS variable indices.

## How It Fits

`jolt-blindfold` depends on:
- `jolt-r1cs` — for constraint matrices and R1CS key
- `jolt-crypto` — for Pedersen commitments, curve operations
- `jolt-compiler` — for Module (to derive R1CS via the pass from 02)
- `jolt-sumcheck` — for sumcheck proof types

The prover runtime (`jolt-zkvm`) conditionally invokes BlindFold at the end of proving (feature-gated behind `zk`). The verifier receives a `BlindFoldProof` instead of cleartext claims.

## Dependencies

- R1CS from Module (02) — provides the constraint matrices
- Compiler full protocol (01) — the Module must be complete

## Unblocks

- jolt-zkvm ZK composition (07)
- Lattice BlindFold (12) — same protocol structure, different commitment scheme
- Hash BlindFold (13) — different protocol, but informed by this design

## Open Questions

- The Nova folding step requires a random satisfying instance. Currently this is sampled and proved. With the generic R1CS, does the random instance generation change?
- Performance: the current BlindFold R1CS is optimized for the specific Jolt verifier structure (known sparsity, known variable layout). A generic derivation may produce denser constraints. Is this acceptable, or do we need optimization passes on the derived R1CS?
- The `AdditivelyHomomorphicCommitment` trait needs to be expressive enough for both Pedersen (EC) and Hachi (lattice). Are there operations that one supports but the other doesn't (e.g., efficient multi-scalar commitment)?
