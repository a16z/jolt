# impl-jolt-openings: Clean-room implementation of jolt-openings

**Scope:** crates/jolt-openings/

**Depends:** impl-jolt-poly

**Verifier:** ./verifiers/scoped.sh /workdir jolt-openings

**Context:**

Implement the `jolt-openings` crate — abstract commitment scheme interfaces, opening proof accumulation, and batch reduction logic. Designed to support homomorphic (Dory, KZG), lattice-based, and hash-based (FRI) commitment schemes.

**This is a clean-room rewrite.** Study `jolt-core/src/poly/commitment/`, `jolt-core/src/poly/opening_proof.rs`, and the `CommitmentScheme` trait in `jolt-core` for algorithmic reference. Design the API from scratch.

**Dependencies:** `jolt-field`, `jolt-poly`, `jolt-transcript`.

### Reference material

The old code lives in:
- `jolt-core/src/poly/commitment/` — CommitmentScheme trait + Dory/KZG/HyperKZG impls
- `jolt-core/src/poly/opening_proof.rs` (690 LOC) — ProverOpeningAccumulator, VerifierOpeningAccumulator, RLC reduction

Also read the Jolt Book: https://jolt.a16zcrypto.com/ — sections on polynomial commitment schemes and batched openings.

### Public API contract — Trait hierarchy

The key design decision: a three-tier trait hierarchy that supports future lattice and hash-based schemes.

```rust
/// Base trait: every commitment scheme implements this.
/// No assumptions about algebraic structure of commitments.
pub trait CommitmentScheme: Clone + Send + Sync + 'static {
    type Field: Field;
    type Commitment: Clone + Send + Sync + Debug + Serialize + DeserializeOwned;
    type Proof: Clone + Send + Sync + Serialize + DeserializeOwned;
    type ProverSetup: Clone + Send + Sync;
    type VerifierSetup: Clone + Send + Sync + Serialize + DeserializeOwned;

    fn protocol_name() -> &'static str;
    fn setup_prover(max_size: usize) -> Self::ProverSetup;
    fn setup_verifier(max_size: usize) -> Self::VerifierSetup;

    fn commit(
        poly: &impl MultilinearPolynomial<Self::Field>,
        setup: &Self::ProverSetup,
    ) -> Self::Commitment;

    fn prove(
        poly: &impl MultilinearPolynomial<Self::Field>,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        transcript: &mut impl Transcript,
    ) -> Self::Proof;

    fn verify(
        commitment: &Self::Commitment,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript,
    ) -> Result<(), OpeningsError>;
}

/// Extension: additively homomorphic schemes (Dory, KZG).
/// Commitments can be combined linearly, enabling batch opening via RLC.
pub trait HomomorphicCommitmentScheme: CommitmentScheme {
    type BatchedProof: Clone + Send + Sync + Serialize + DeserializeOwned;

    fn combine_commitments(
        commitments: &[Self::Commitment],
        scalars: &[Self::Field],
    ) -> Self::Commitment;

    fn batch_prove(
        polynomials: &[&dyn MultilinearPolynomial<Self::Field>],
        points: &[Vec<Self::Field>],
        evals: &[Self::Field],
        setup: &Self::ProverSetup,
        transcript: &mut impl Transcript,
    ) -> Self::BatchedProof;

    fn batch_verify(
        commitments: &[Self::Commitment],
        points: &[Vec<Self::Field>],
        evals: &[Self::Field],
        proof: &Self::BatchedProof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript,
    ) -> Result<(), OpeningsError>;
}

/// Extension: streaming commitment for memory-efficient large polynomials.
pub trait StreamingCommitmentScheme: CommitmentScheme {
    type PartialCommitment: Clone + Send + Sync;

    fn begin_streaming(setup: &Self::ProverSetup) -> Self::PartialCommitment;
    fn stream_chunk(partial: &mut Self::PartialCommitment, chunk: &[Self::Field]);
    fn finalize_streaming(partial: Self::PartialCommitment) -> Self::Commitment;
}
```

### Public API contract — Opening accumulators

```rust
/// Accumulates opening claims during proving, then reduces them
/// to a minimal set of opening proofs via random linear combination.
pub struct ProverOpeningAccumulator<F: Field> { ... }

impl<F: Field> ProverOpeningAccumulator<F> {
    pub fn new() -> Self;
    pub fn accumulate(
        &mut self,
        poly: &dyn MultilinearPolynomial<F>,
        point: Vec<F>,
        eval: F,
    );
    pub fn reduce_and_prove<PCS: HomomorphicCommitmentScheme<Field = F>>(
        self,
        setup: &PCS::ProverSetup,
        transcript: &mut impl Transcript,
    ) -> Vec<PCS::BatchedProof>;
}

/// Verifier-side accumulator.
pub struct VerifierOpeningAccumulator<F: Field> { ... }

impl<F: Field> VerifierOpeningAccumulator<F> {
    pub fn new() -> Self;
    pub fn accumulate(
        &mut self,
        commitment: impl Into<Box<dyn Any + Send + Sync>>,
        point: Vec<F>,
        eval: F,
    );
    pub fn reduce_and_verify<PCS: HomomorphicCommitmentScheme<Field = F>>(
        self,
        setup: &PCS::VerifierSetup,
        transcript: &mut impl Transcript,
    ) -> Result<(), OpeningsError>;
}
```

### Error type

```rust
#[derive(Debug, thiserror::Error)]
pub enum OpeningsError {
    #[error("opening proof verification failed")]
    VerificationFailed,
    #[error("commitment mismatch: expected {expected}, got {actual}")]
    CommitmentMismatch { expected: String, actual: String },
    #[error("invalid setup parameters: {0}")]
    InvalidSetup(String),
    #[error("polynomial size {poly_size} exceeds setup max {setup_max}")]
    PolynomialTooLarge { poly_size: usize, setup_max: usize },
}
```

### Implementation notes

- The accumulator tracks a list of `(polynomial, point, eval)` triples. `reduce_and_prove` groups them by opening point, then uses RLC to combine polynomials at the same point.
- The RLC (random linear combination) reduction: draw random `rho` from transcript, compute `combined_poly = sum_i rho^i * poly_i`, then open `combined_poly` at the shared point.
- For different points, produce separate batched proofs.
- The accumulator's DAG optimization (from jolt-core's opening_proof.rs) efficiently handles the case where many polynomials share the same point — study the old code's approach but implement it cleanly.
- `VerifierOpeningAccumulator` stores commitments type-erased (as `Box<dyn Any>`) and downcasts when `reduce_and_verify` is called with a concrete PCS type.

### Mock PCS for testing

Include a `mock` module (behind `#[cfg(test)]`):

```rust
/// Trivial commitment scheme for testing: commitment = hash of evaluations.
/// Not cryptographically secure.
#[cfg(test)]
pub struct MockCommitmentScheme;
```

This enables testing the accumulator and reduction logic without depending on Dory.

### File structure

```
jolt-openings/src/
├── lib.rs
├── traits.rs           # CommitmentScheme, HomomorphicCommitmentScheme, StreamingCommitmentScheme
├── accumulator.rs      # ProverOpeningAccumulator, VerifierOpeningAccumulator
├── reduction.rs        # RLC batch reduction logic
├── error.rs            # OpeningsError
└── mock.rs             # MockCommitmentScheme (test-only)
```

**Acceptance:**

- All three trait tiers implemented (CommitmentScheme, Homomorphic, Streaming)
- ProverOpeningAccumulator and VerifierOpeningAccumulator fully functional
- RLC reduction correctly batches multiple openings at the same point
- MockCommitmentScheme for testing round-trips (commit → prove → verify)
- Error types cover all failure modes
- All public types are `Serialize + Deserialize` (where appropriate)
- `parallel` feature enables rayon in reduction hot paths
- No file exceeds 500 lines
- Rustdoc on all public items with LaTeX math
- `cargo clippy` clean
- Unit tests for accumulator, reduction, and mock PCS round-trip
