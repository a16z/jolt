# jolt-core Refactoring Specification

**Status:** Draft
**Authors:** Markos Georghiades
**Date:** 2026-02-22

## 1. Motivation

jolt-core is a ~64K LOC monolith that implements the entire Jolt zkVM proving system. The [RFC](./rfc.md) identifies 11 structural problems — arkworks coupling, global mutable state, unclear trait boundaries, lack of unit testing, and tight coupling between the PIOP and commitment scheme. These problems make the codebase difficult to audit, difficult to extend (lattices, hashes), and hostile to AI-assisted development.

This spec defines a **clean-room rewrite** of jolt-core into a modular crate workspace. Each crate is written from scratch with a fresh API design, using the existing jolt-core as a correctness reference but never copying code. The existing jolt-core remains intact throughout the process and is only removed as the final step.

### Goals

1. **Audit readiness** — well-encapsulated crates with clear API boundaries and isolated test suites
2. **Backend flexibility** — arkworks behind clean abstractions; field, commitment, and polynomial traits designed for multiple backends (arkworks, lattice, hash)
3. **AI friendliness** — small files (<10K tokens), explicit dependencies, well-defined traits, comprehensive tests
4. **Reusability** — crates like `jolt-field`, `jolt-sumcheck`, `jolt-poly` usable as standalone dependencies in other projects
5. **Production quality** — idiomatic Rust, extensive doc comments with LaTeX math, property-based testing, fuzzing

### Non-Goals

- Rewriting `common`, `tracer`, `jolt-sdk`, or `jolt-platform` (out of scope, though minor modifications are allowed)
- Performance regression — the rewrite must match or exceed current performance
- New features — no new cryptographic functionality, just restructuring

---

## 2. Principles

### 2.1 Clean-Room Methodology

Each crate is designed and implemented from first principles:

1. **Study** the corresponding jolt-core module to understand the mathematical operations and data flow
2. **Design** a clean API (traits, types, methods) without inheriting jolt-core's structural decisions
3. **Implement** from scratch, referencing jolt-core for algorithmic correctness but not copying code
4. **Test** independently with unit tests, property-based tests, and integration tests
5. **Verify** against jolt-core's end-to-end test suite once integrated

### 2.2 Style Reference

The existing `jolt-transcript` and `jolt-field` crates (already completed) define the quality bar:

- **Trait-driven architecture** — minimal concrete types, maximum abstraction
- **Extensive documentation** — module-level and item-level rustdoc with LaTeX math (`$...$`, `$$...$$`)
- **Performance-first** — inline optimizations, unreduced arithmetic paths, specialized types
- **Test infrastructure** — `#[cfg(test)]` features, comparison modes
- **Type safety** — rich trait bounds, const generics where appropriate
- **Zero unnecessary dependencies** — each crate depends only on what it needs
- **README per crate** — explaining the module's role, linking to the [Jolt Book](https://jolt.a16zcrypto.com/)

### 2.3 Cross-Cutting Conventions

| Concern | Decision |
|---------|----------|
| **Serialization** | `serde` everywhere. No `CanonicalSerialize`/`CanonicalDeserialize` in public APIs. Arkworks types get custom `Serialize`/`Deserialize` impls that delegate to arkworks internally. |
| **Error handling** | `thiserror` per crate. Each crate defines its own error enum. `From` conversions at crate boundaries. |
| **Parallelism** | `rayon` available in all crates. Feature flag `parallel` (default on) per crate to allow disabling for WASM/embedded. |
| **Logging** | `tracing` for instrumentation. No `println!` or `eprintln!`. |
| **RNG** | `rand` + `rand_core` traits. No concrete RNG types in public APIs. |
| **Arkworks** | Arkworks types *implement* Jolt traits. Arkworks never appears in trait bounds or public API signatures. Contained in `arkworks` submodules behind the crate's own traits. |
| **Feature flags** | `parallel` (rayon, default on), `arkworks` (arkworks backend, default on), `serde` (serialization, default on). Crate-specific flags as needed. |

---

## 3. Crate Decomposition

### 3.1 Dependency Graph

```
jolt-transcript              (no deps)
        │
        ▼
   jolt-field                (jolt-transcript)
        │
        ▼
   jolt-poly                 (jolt-field)
        │
        ├──────────────────────────┐
        ▼                          ▼
jolt-openings              jolt-sumcheck
(jolt-field, jolt-poly,    (jolt-field, jolt-poly,
 jolt-transcript)           jolt-transcript)
        │                          │
        ├──────────┬───────────────┘
        ▼          ▼
   jolt-spartan
   (jolt-sumcheck, jolt-openings)
        │
        ▼
   jolt-instructions
   (jolt-field)
        │
        ▼
   jolt-zkvm
   (jolt-spartan, jolt-sumcheck,
    jolt-openings, jolt-instructions,
    jolt-field, jolt-poly, jolt-transcript)
        │
        ├──► jolt-dory     (jolt-openings, dory-pcs)
        ├──► jolt-kzg      (jolt-openings, ark-poly-commit)  [future]
        └──► jolt-hyperkzg (jolt-openings)                   [future]
```

### 3.2 Crate Summary

| Crate | Purpose | Reusable | LOC Estimate |
|-------|---------|----------|--------------|
| `jolt-transcript` | Fiat-Shamir transcripts | Yes | ~500 (done) |
| `jolt-field` | Field arithmetic traits + arkworks impl | Yes | ~2000 (done) |
| `jolt-poly` | Polynomial types and operations | Yes | ~3000 |
| `jolt-openings` | Commitment scheme traits + opening accumulators | Yes | ~2500 |
| `jolt-sumcheck` | Sumcheck protocol engine | Yes | ~2000 |
| `jolt-spartan` | R1CS + Spartan prover/verifier | Yes | ~3000 |
| `jolt-instructions` | RISC-V instruction set + lookup tables | No | ~8000 |
| `jolt-dory` | Dory commitment scheme impl | No | ~1500 |
| `jolt-zkvm` | zkVM prover/verifier orchestration | No | ~10000 |

---

## 4. Per-Crate Specifications

### 4.1 `jolt-transcript` — **DONE**

Already completed. Defines `Transcript` and `AppendToTranscript` traits with Blake2b and Keccak implementations. See `crates/jolt-transcript/`.

### 4.2 `jolt-field` — **DONE**

Already completed. Defines `Field`, `UnreducedOps`, `ReductionOps`, `Challenge`, `WithChallenge`, `OptimizedMul`, accumulation traits (`FMAdd`, `BarrettReduce`, `MontgomeryReduce`), and arkworks BN254 implementation. See `crates/jolt-field/`.

---

### 4.3 `jolt-poly` — Polynomial Library

**Purpose:** Generic polynomial types and operations for multilinear, univariate, and specialized polynomials. This crate is backend-agnostic and reusable outside Jolt.

**Dependencies:** `jolt-field`

#### Public API

```rust
// ── Core trait ──────────────────────────────────────────────

/// A multilinear polynomial over `F` in `num_vars` variables,
/// represented by its evaluations over the Boolean hypercube $\{0,1\}^n$.
pub trait MultilinearPolynomial<F: Field>: Send + Sync {
    /// Number of variables (dimension of the hypercube).
    fn num_vars(&self) -> usize;

    /// Number of evaluations ($2^{\text{num\_vars}}$).
    fn len(&self) -> usize;

    /// Evaluate at a point $r \in F^n$ via multilinear extension.
    fn evaluate(&self, point: &[F]) -> F;

    /// Bind the first variable to `scalar`, reducing to $n-1$ variables.
    /// Returns a dense polynomial of half the size.
    fn bind(&self, scalar: F) -> DensePolynomial<F>;

    /// Read-only access to evaluations (may allocate if compressed).
    fn evaluations(&self) -> Cow<[F]>;
}

// ── Concrete types ──────────────────────────────────────────

/// Dense multilinear polynomial: stores all $2^n$ evaluations as `Vec<F>`.
pub struct DensePolynomial<F: Field> {
    evaluations: Vec<F>,
    num_vars: usize,
}

impl<F: Field> DensePolynomial<F> {
    pub fn new(evaluations: Vec<F>) -> Self;
    pub fn zeros(num_vars: usize) -> Self;
    pub fn random(num_vars: usize, rng: &mut impl RngCore) -> Self;

    /// Bind first variable in-place (mutates, halves length).
    pub fn bind_in_place(&mut self, scalar: F);

    /// Evaluate and consume (avoids clone for owned polys).
    pub fn evaluate_and_consume(self, point: &[F]) -> F;

    /// Mutable access to evaluations.
    pub fn evaluations_mut(&mut self) -> &mut [F];
}

/// Compact multilinear polynomial: stores small scalars (`u8`, `u16`, etc.)
/// and converts to `F` on demand. Reduces memory by up to 32x for Boolean polys.
pub struct CompactPolynomial<S: SmallScalar, F: Field> {
    scalars: Vec<S>,
    num_vars: usize,
    _marker: PhantomData<F>,
}

/// Trait for scalar types that can be stored compactly and promoted to `F`.
pub trait SmallScalar: Copy + Send + Sync + Into<u64> + 'static {
    fn from_field<F: Field>(f: F) -> Option<Self>;
}

/// Equality polynomial: $\widetilde{eq}(x, r) = \prod_{i=1}^{n}(r_i x_i + (1-r_i)(1-x_i))$.
pub struct EqPolynomial<F: Field> {
    point: Vec<F>,
}

impl<F: Field> EqPolynomial<F> {
    pub fn new(point: Vec<F>) -> Self;

    /// Compute all $2^n$ evaluations over the Boolean hypercube.
    pub fn evaluations(&self) -> Vec<F>;

    /// Evaluate at a single point (without materializing all evaluations).
    pub fn evaluate(&self, point: &[F]) -> F;
}

/// Univariate polynomial in coefficient form.
pub struct UnivariatePoly<F: Field> {
    coefficients: Vec<F>,
}

impl<F: Field> UnivariatePoly<F> {
    pub fn new(coefficients: Vec<F>) -> Self;
    pub fn degree(&self) -> usize;
    pub fn evaluate(&self, point: F) -> F;
    pub fn interpolate(points: &[(F, F)]) -> Self;
}

/// Identity polynomial: $\tilde{I}(x) = \sum_{i} i \cdot \widetilde{eq}(x, i)$.
pub struct IdentityPolynomial {
    num_vars: usize,
}

/// Lagrange basis polynomial.
pub struct LagrangePolynomial<F: Field> { /* ... */ }
```

#### Testing

- **Unit tests:** Evaluate known polynomials at known points, verify binding correctness
- **Property tests:** For random polynomials, `evaluate(point) == bind_sequentially(point)`. Schwartz-Zippel: two distinct polys disagree at a random point with high probability.
- **Fuzz targets:** `DensePolynomial::new` with arbitrary byte inputs, evaluate with arbitrary points

#### File Structure

```
jolt-poly/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Re-exports, module declarations
│   ├── traits.rs           # MultilinearPolynomial trait
│   ├── dense.rs            # DensePolynomial (with unit tests)
│   ├── compact.rs          # CompactPolynomial + SmallScalar (with unit tests)
│   ├── eq.rs               # EqPolynomial (with unit tests)
│   ├── univariate.rs       # UnivariatePoly (with unit tests)
│   ├── identity.rs         # IdentityPolynomial (with unit tests)
│   └── lagrange.rs         # LagrangePolynomial (with unit tests)
├── tests/                  # Integration tests
│   ├── polynomial_api.rs   # Test public API usage patterns
│   ├── type_interop.rs     # Test conversions between polynomial types
│   └── math_properties.rs  # Verify mathematical invariants
├── fuzz/
│   └── fuzz_targets/
│       ├── dense_new.rs
│       └── evaluate.rs
└── benches/
    ├── binding.rs
    └── evaluation.rs
```

---

### 4.4 `jolt-openings` — Commitment Scheme Traits & Opening Accumulators

**Purpose:** Abstract commitment scheme interfaces, opening proof accumulation, and batch reduction logic. Designed to support homomorphic (Dory, KZG), lattice-based, and hash-based (FRI) commitment schemes.

**Dependencies:** `jolt-field`, `jolt-poly`, `jolt-transcript`

#### Commitment Scheme Trait Hierarchy

```rust
// ── Base trait: every commitment scheme implements this ─────

/// A polynomial commitment scheme (PCS).
///
/// The minimal interface: commit to a polynomial, produce an opening proof
/// at a point, and verify the proof. No assumptions about algebraic structure
/// of commitments.
pub trait CommitmentScheme: Clone + Send + Sync + 'static {
    /// The field over which polynomials are defined.
    type Field: Field;

    /// Opaque commitment value (e.g., group element, Merkle root, lattice vector).
    type Commitment: Clone + Send + Sync + Debug + Serialize + DeserializeOwned;

    /// Opening proof for a single polynomial at a single point.
    type Proof: Clone + Send + Sync + Serialize + DeserializeOwned;

    /// Prover-side setup parameters.
    type ProverSetup: Clone + Send + Sync;

    /// Verifier-side setup parameters.
    type VerifierSetup: Clone + Send + Sync + Serialize + DeserializeOwned;

    /// Human-readable protocol name (for transcript domain separation).
    fn protocol_name() -> &'static str;

    /// Generate prover parameters for polynomials of at most `max_size` evaluations.
    fn setup_prover(max_size: usize) -> Self::ProverSetup;

    /// Generate verifier parameters.
    fn setup_verifier(max_size: usize) -> Self::VerifierSetup;

    /// Commit to a polynomial.
    fn commit(
        poly: &impl MultilinearPolynomial<Self::Field>,
        setup: &Self::ProverSetup,
    ) -> Self::Commitment;

    /// Produce an opening proof: the polynomial evaluates to `eval` at `point`.
    fn prove(
        poly: &impl MultilinearPolynomial<Self::Field>,
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        transcript: &mut impl Transcript,
    ) -> Self::Proof;

    /// Verify an opening proof.
    fn verify(
        commitment: &Self::Commitment,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript,
    ) -> Result<(), OpeningsError>;
}

// ── Extension: additively homomorphic schemes ───────────────

/// Commitment scheme where commitments can be combined linearly.
///
/// This enables batch opening proofs via random linear combination (RLC):
/// given commitments $C_1, \ldots, C_k$ and a random $\rho$, the verifier
/// checks a single opening of $C_1 + \rho C_2 + \cdots + \rho^{k-1} C_k$.
pub trait HomomorphicCommitmentScheme: CommitmentScheme {
    /// Batched opening proof for multiple polynomials.
    type BatchedProof: Clone + Send + Sync + Serialize + DeserializeOwned;

    /// Linearly combine commitments: $\sum_i \text{scalars}_i \cdot C_i$.
    fn combine_commitments(
        commitments: &[Self::Commitment],
        scalars: &[Self::Field],
    ) -> Self::Commitment;

    /// Produce a batched opening proof for multiple polynomials at possibly
    /// different points.
    fn batch_prove(
        polynomials: &[&dyn MultilinearPolynomial<Self::Field>],
        points: &[Vec<Self::Field>],
        evals: &[Self::Field],
        setup: &Self::ProverSetup,
        transcript: &mut impl Transcript,
    ) -> Self::BatchedProof;

    /// Verify a batched opening proof.
    fn batch_verify(
        commitments: &[Self::Commitment],
        points: &[Vec<Self::Field>],
        evals: &[Self::Field],
        proof: &Self::BatchedProof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript,
    ) -> Result<(), OpeningsError>;
}

// ── Extension: streaming commitment ─────────────────────────

/// Commitment scheme that supports chunked/streaming commitment for
/// memory efficiency with large polynomials.
pub trait StreamingCommitmentScheme: CommitmentScheme {
    /// Intermediate state for a partial commitment.
    type PartialCommitment: Clone + Send + Sync;

    /// Begin a streaming commitment.
    fn begin_streaming(setup: &Self::ProverSetup) -> Self::PartialCommitment;

    /// Feed a chunk of evaluations.
    fn stream_chunk(
        partial: &mut Self::PartialCommitment,
        chunk: &[Self::Field],
    );

    /// Finalize the streaming commitment.
    fn finalize_streaming(partial: Self::PartialCommitment) -> Self::Commitment;
}
```

#### Opening Accumulator

```rust
/// Accumulates opening claims during proving, then reduces them to a
/// minimal set of opening proofs via random linear combination.
///
/// The accumulator tracks a DAG of opening points and polynomials,
/// batching where possible to minimize the number of PCS calls.
pub struct ProverOpeningAccumulator<F: Field> { /* ... */ }

impl<F: Field> ProverOpeningAccumulator<F> {
    pub fn new() -> Self;

    /// Register a polynomial + point + evaluation for later batching.
    pub fn accumulate(
        &mut self,
        poly: &dyn MultilinearPolynomial<F>,
        point: Vec<F>,
        eval: F,
    );

    /// Reduce all accumulated claims and produce opening proofs.
    pub fn reduce_and_prove<PCS: HomomorphicCommitmentScheme<Field = F>>(
        self,
        setup: &PCS::ProverSetup,
        transcript: &mut impl Transcript,
    ) -> Vec<PCS::BatchedProof>;
}

/// Verifier-side accumulator: collects commitments + claimed evaluations,
/// then batch-verifies.
pub struct VerifierOpeningAccumulator<F: Field> { /* ... */ }

impl<F: Field> VerifierOpeningAccumulator<F> {
    pub fn new() -> Self;

    pub fn accumulate(
        &mut self,
        commitment: &impl Clone, // type-erased, stored as Any
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

#### Error Type

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

#### Testing

- **Unit tests:** Accumulate mock openings, verify reduction produces correct batches
- **Property tests:** For any set of random polynomials and random points, `accumulate → reduce → verify` succeeds. Modifying any evaluation causes verification failure.
- **Integration tests:** Round-trip with `jolt-dory` (prove + verify)

#### File Structure

```
jolt-openings/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs
│   ├── traits.rs           # CommitmentScheme, HomomorphicCommitmentScheme, StreamingCommitmentScheme
│   ├── accumulator.rs      # ProverOpeningAccumulator, VerifierOpeningAccumulator
│   ├── reduction.rs        # RLC batch reduction logic
│   └── error.rs            # OpeningsError
├── tests/                  # Integration tests
│   ├── commitment_api.rs   # Test trait implementations
│   ├── accumulator.rs      # Test accumulator round-trips
│   └── batching.rs         # Test batch operations
├── fuzz/
│   └── fuzz_targets/
│       ├── verify_proof.rs
│       └── accumulator.rs
└── benches/
    └── opening.rs
```

---

### 4.5 `jolt-sumcheck` — Sumcheck Protocol Engine

**Purpose:** Generic implementation of the sum-check protocol, including batched and streaming variants. Reusable for any sum-check application, not just Jolt.

**Dependencies:** `jolt-field`, `jolt-poly`, `jolt-transcript`

#### Public API

```rust
// ── Core traits ─────────────────────────────────────────────

/// A sumcheck claim: the prover asserts that
/// $$\sum_{x \in \{0,1\}^n} g(x) = \text{claimed\_sum}$$
/// where $g$ is implicitly defined by the witness.
pub struct SumcheckClaim<F: Field> {
    pub num_vars: usize,
    pub degree: usize,
    pub claimed_sum: F,
}

/// Prover-side interface for a single sumcheck instance.
///
/// The sumcheck engine calls `round_polynomial` repeatedly, once per variable.
/// After each round, the engine calls `bind` with the verifier's challenge,
/// fixing one variable and halving the instance.
pub trait SumcheckInstanceProver<F: Field>: Send + Sync {
    /// Produce the round polynomial for the current round.
    ///
    /// Returns a univariate polynomial of degree ≤ `claim.degree` representing
    /// $$s_i(X_i) = \sum_{x_{i+1}, \ldots, x_n \in \{0,1\}} g(r_1, \ldots, r_{i-1}, X_i, x_{i+1}, \ldots, x_n)$$
    fn round_polynomial(&self) -> UnivariatePoly<F>;

    /// Bind the current variable to `challenge`, reducing the instance by one variable.
    fn bind(&mut self, challenge: F);
}

/// Verifier-side interface for checking a sumcheck transcript.
pub trait SumcheckInstanceVerifier<F: Field> {
    /// Verify a single round: check that $s_i(0) + s_i(1) = \text{expected\_sum}$,
    /// then return $s_i(r_i)$ as the next expected sum.
    fn verify_round(
        round_poly: &UnivariatePoly<F>,
        expected_sum: F,
        challenge: F,
    ) -> Result<F, SumcheckError>;

    /// Final check: verify the claimed evaluation at the fully-bound point.
    fn verify_final(
        claimed_eval: F,
        expected: F,
    ) -> Result<(), SumcheckError>;
}

// ── Protocol engine ─────────────────────────────────────────

/// Non-interactive sumcheck (Fiat-Shamir) prover.
pub struct SumcheckProver;

impl SumcheckProver {
    /// Run the sumcheck protocol for a single instance, appending
    /// round polynomials to the transcript and returning the proof.
    pub fn prove<F: Field>(
        claim: &SumcheckClaim<F>,
        witness: &mut impl SumcheckInstanceProver<F>,
        transcript: &mut impl Transcript,
    ) -> SumcheckProof<F>;

    /// Run batched sumcheck for multiple instances simultaneously.
    ///
    /// Uses Posen's front-loaded batching: instances with fewer variables
    /// are scaled by powers of 2 to match the largest instance.
    pub fn prove_batched<F: Field>(
        claims: &[SumcheckClaim<F>],
        witnesses: &mut [Box<dyn SumcheckInstanceProver<F>>],
        transcript: &mut impl Transcript,
    ) -> BatchedSumcheckProof<F>;
}

/// Non-interactive sumcheck verifier.
pub struct SumcheckVerifier;

impl SumcheckVerifier {
    pub fn verify<F: Field>(
        claim: &SumcheckClaim<F>,
        proof: &SumcheckProof<F>,
        transcript: &mut impl Transcript,
    ) -> Result<(F, Vec<F>), SumcheckError>;

    pub fn verify_batched<F: Field>(
        claims: &[SumcheckClaim<F>],
        proof: &BatchedSumcheckProof<F>,
        transcript: &mut impl Transcript,
    ) -> Result<(Vec<F>, Vec<Vec<F>>), SumcheckError>;
}

// ── Proof types ─────────────────────────────────────────────

/// Proof transcript for a single sumcheck instance.
pub struct SumcheckProof<F: Field> {
    pub round_polynomials: Vec<UnivariatePoly<F>>,
}

pub struct BatchedSumcheckProof<F: Field> {
    pub round_polynomials: Vec<Vec<UnivariatePoly<F>>>,
}

// ── Error ───────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum SumcheckError {
    #[error("round {round}: expected sum {expected}, got {actual}")]
    RoundCheckFailed { round: usize, expected: String, actual: String },

    #[error("final evaluation mismatch")]
    FinalEvalMismatch,

    #[error("degree bound exceeded: round poly degree {got}, max {max}")]
    DegreeBoundExceeded { got: usize, max: usize },
}
```

#### Testing

- **Unit tests:** Sumcheck over known polynomial sums (e.g., $\sum_{x \in \{0,1\}^3} eq(x, r)$ should equal 1)
- **Property tests:**
  - For random multilinear $f$: `prove(sum(f)) → verify` succeeds
  - Modifying any round polynomial causes verification failure
  - Batched sumcheck: each sub-claim verifies independently
- **Fuzz targets:** Random claims + random round polynomials → verifier rejects (soundness)

#### File Structure

```
jolt-sumcheck/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs
│   ├── claim.rs            # SumcheckClaim
│   ├── prover.rs           # SumcheckProver, SumcheckInstanceProver trait
│   ├── verifier.rs         # SumcheckVerifier, SumcheckInstanceVerifier trait
│   ├── proof.rs            # SumcheckProof, BatchedSumcheckProof
│   ├── batched.rs          # Batching logic (front-loaded, power-of-2 scaling)
│   ├── streaming.rs        # Streaming sumcheck variant (memory-efficient)
│   └── error.rs            # SumcheckError
├── tests/                  # Integration tests
│   ├── protocol.rs         # Test complete protocol round-trips
│   ├── batching.rs         # Test batched sumcheck
│   └── streaming.rs        # Test streaming variant
├── fuzz/
│   └── fuzz_targets/
│       ├── round_polys.rs
│       └── verify.rs
└── benches/
    └── sumcheck.rs
```

---

### 4.6 `jolt-spartan` — R1CS + Spartan Prover/Verifier

**Purpose:** Spartan-based SNARK for R1CS constraint systems. Generic over the commitment scheme and field. Usable for any R1CS system, not just Jolt.

**Dependencies:** `jolt-sumcheck`, `jolt-openings`, `jolt-field`, `jolt-poly`, `jolt-transcript`

#### Public API

```rust
// ── R1CS representation ─────────────────────────────────────

/// A Rank-1 Constraint System: matrices $A$, $B$, $C$ such that
/// $Az \circ Bz = Cz$ for a valid witness $z$.
pub trait R1CS<F: Field> {
    /// Number of constraints.
    fn num_constraints(&self) -> usize;

    /// Number of variables (including input/output and auxiliary).
    fn num_variables(&self) -> usize;

    /// Multiply the witness by the constraint matrices,
    /// returning $(Az, Bz, Cz)$.
    fn multiply_witness(
        &self,
        witness: &[F],
    ) -> (Vec<F>, Vec<F>, Vec<F>);
}

/// Uniform (structured) R1CS where the constraint matrices have
/// repeating structure, enabling more efficient proving.
pub struct UniformR1CS<F: Field> { /* ... */ }

/// Key material derived from the R1CS structure (precomputed).
pub struct SpartanKey<F: Field> { /* ... */ }

impl<F: Field> SpartanKey<F> {
    pub fn from_r1cs(r1cs: &impl R1CS<F>) -> Self;
}

// ── Prover / Verifier ───────────────────────────────────────

pub struct SpartanProver;

impl SpartanProver {
    pub fn prove<F, PCS>(
        key: &SpartanKey<F>,
        witness: &[F],
        pcs_setup: &PCS::ProverSetup,
        transcript: &mut impl Transcript,
    ) -> Result<SpartanProof<F, PCS>, SpartanError>
    where
        F: Field,
        PCS: HomomorphicCommitmentScheme<Field = F>;
}

pub struct SpartanVerifier;

impl SpartanVerifier {
    pub fn verify<F, PCS>(
        key: &SpartanKey<F>,
        proof: &SpartanProof<F, PCS>,
        pcs_setup: &PCS::VerifierSetup,
        transcript: &mut impl Transcript,
    ) -> Result<(), SpartanError>
    where
        F: Field,
        PCS: HomomorphicCommitmentScheme<Field = F>;
}

// ── Proof + Error ───────────────────────────────────────────

pub struct SpartanProof<F: Field, PCS: CommitmentScheme> { /* ... */ }

#[derive(Debug, thiserror::Error)]
pub enum SpartanError {
    #[error("R1CS constraint violation at index {0}")]
    ConstraintViolation(usize),
    #[error("sumcheck failed: {0}")]
    Sumcheck(#[from] SumcheckError),
    #[error("opening proof failed: {0}")]
    Opening(#[from] OpeningsError),
}
```

#### Univariate Skip Optimization

The current Spartan implementation uses a univariate skip optimization for the first sumcheck round. This should be implemented as an optional strategy:

```rust
/// Strategy for the first sumcheck round in Spartan.
pub enum FirstRoundStrategy {
    /// Standard first round (no optimization).
    Standard,
    /// Univariate skip: precompute domain evaluations to skip
    /// the first round's full polynomial evaluation.
    UnivariateSkip { domain_size: usize },
}
```

#### Testing

- **Unit tests:** Small R1CS instances (e.g., $x^2 = y$), verify proof generation and verification
- **Property tests:** Random satisfiable R1CS → prove → verify succeeds. Unsatisfiable witness → prove → verify fails.
- **Integration:** Round-trip with `jolt-dory`

#### File Structure

```
jolt-spartan/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs
│   ├── r1cs.rs             # R1CS trait, UniformR1CS
│   ├── key.rs              # SpartanKey
│   ├── prover.rs           # SpartanProver
│   ├── verifier.rs         # SpartanVerifier
│   ├── proof.rs            # SpartanProof
│   ├── uni_skip.rs         # Univariate skip optimization
│   └── error.rs            # SpartanError
├── tests/                  # Integration tests
│   ├── r1cs_proving.rs     # Test R1CS satisfaction proving
│   ├── uniform_r1cs.rs     # Test uniform R1CS
│   └── uni_skip.rs         # Test univariate skip optimization
├── fuzz/
│   └── fuzz_targets/
│       ├── r1cs_verify.rs
│       └── witness.rs
└── benches/
    └── spartan.rs
```

---

### 4.7 `jolt-instructions` — RISC-V Instruction Set & Lookup Tables

**Purpose:** Defines the Jolt instruction set (RISC-V base + virtual instructions) and their decomposition into lookup tables. This is Jolt-specific and not intended for reuse outside the project.

**Dependencies:** `jolt-field`

#### Public API

```rust
// ── Instruction trait ───────────────────────────────────────

/// A Jolt instruction: a function from operands to a result,
/// decomposed into lookup table queries for the prover.
pub trait Instruction<F: Field>: Send + Sync + 'static {
    /// Unique opcode identifier.
    fn opcode(&self) -> u32;

    /// Human-readable name.
    fn name(&self) -> &'static str;

    /// Execute the instruction on concrete operands.
    fn execute(&self, operands: &[u64]) -> u64;

    /// Decompose into lookup table indices for the prover.
    fn lookups(&self, operands: &[u64]) -> Vec<LookupQuery>;
}

/// A single lookup table query.
pub struct LookupQuery {
    pub table: TableId,
    pub input: u64,
}

/// Identifies a lookup table.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub struct TableId(pub u16);

// ── Lookup table trait ──────────────────────────────────────

/// A lookup table: a function from a small input domain to field elements.
pub trait LookupTable<F: Field>: Send + Sync {
    fn id(&self) -> TableId;
    fn name(&self) -> &'static str;

    /// Size of the input domain.
    fn size(&self) -> usize;

    /// Evaluate the table at `input`.
    fn evaluate(&self, input: u64) -> F;

    /// Materialize the full table (for commitment).
    fn materialize(&self) -> Vec<F>;
}

// ── Instruction set ─────────────────────────────────────────

/// The complete Jolt instruction set.
pub struct JoltInstructionSet { /* ... */ }

impl JoltInstructionSet {
    pub fn new() -> Self;
    pub fn instruction(&self, opcode: u32) -> Option<&dyn Instruction<F>>;
    pub fn tables(&self) -> &[Box<dyn LookupTable<F>>];
}
```

#### Instruction Categories

Standard RISC-V (RV32I/RV64I):
- Arithmetic: `ADD`, `ADDI`, `SUB`, `LUI`, `AUIPC`
- Logic: `AND`, `ANDI`, `OR`, `ORI`, `XOR`, `XORI`
- Shift: `SLL`, `SLLI`, `SRL`, `SRLI`, `SRA`, `SRAI`
- Compare: `SLT`, `SLTI`, `SLTU`, `SLTIU`
- Branch: `BEQ`, `BNE`, `BLT`, `BGE`, `BLTU`, `BGEU`
- Jump: `JAL`, `JALR`
- Load: `LB`, `LBU`, `LH`, `LHU`, `LW`, `LWU`, `LD`
- Store: `SB`, `SH`, `SW`, `SD`
- System: `ECALL`, `EBREAK`, `FENCE`

Virtual (Jolt-specific):
- `ASSERT_EQ`, `ASSERT_LTE`
- `POW2`, `MOVSIGN`
- `ROTRIW`, `XOR_ROT`, etc.

Lookup Table Categories:
- **Prefix tables:** Operate on upper bits (AND, OR, XOR, LT, GT, shifts, etc.)
- **Suffix tables:** Operate on lower bits (corresponding suffix operations)
- **Virtual tables:** Composite operations

#### Testing

- **Unit tests:** Every instruction tested: `execute(operands) == expected_result`
- **Property tests:** For arithmetic instructions, `execute` matches Rust's native arithmetic (wrapping). For decomposition: `lookups(operands) → reconstruct == execute(operands)`.
- **Exhaustive tests:** For small-domain tables, verify every entry

#### File Structure

```
jolt-instructions/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs
│   ├── traits.rs           # Instruction, LookupTable traits
│   ├── instruction_set.rs  # JoltInstructionSet
│   ├── rv/                 # Standard RISC-V instructions
│   │   ├── mod.rs
│   │   ├── arithmetic.rs   # ADD, SUB, etc.
│   │   ├── logic.rs        # AND, OR, XOR
│   │   ├── shift.rs        # SLL, SRL, SRA
│   │   ├── compare.rs      # SLT, SLTU
│   │   ├── branch.rs       # BEQ, BNE, etc.
│   │   ├── jump.rs         # JAL, JALR
│   │   ├── load.rs         # LB, LW, LD, etc.
│   │   ├── store.rs        # SB, SW, SD
│   │   └── system.rs       # ECALL, EBREAK, FENCE
│   ├── virtual_/           # Virtual instructions
│   │   ├── mod.rs
│   │   ├── assert.rs
│   │   ├── bitwise.rs
│   │   └── arithmetic.rs
│   └── tables/             # Lookup tables
│       ├── mod.rs
│       ├── prefix/         # Prefix decomposition tables
│       ├── suffix/         # Suffix decomposition tables
│       └── virtual_/       # Virtual lookup tables
├── tests/                  # Integration tests
│   ├── instruction_set.rs  # Test complete instruction set
│   ├── lookup_tables.rs    # Test lookup table consistency
│   └── decomposition.rs    # Test instruction decomposition
├── fuzz/
│   └── fuzz_targets/
│       ├── execute.rs
│       └── decode.rs
└── benches/
    └── lookups.rs
```

---

### 4.8 `jolt-dory` — Dory Commitment Scheme

**Purpose:** Implements `CommitmentScheme` and `HomomorphicCommitmentScheme` from `jolt-openings` using the Dory polynomial commitment scheme. Wraps the external `dory-pcs` crate. All parameters are instance-local (no globals).

**Dependencies:** `jolt-openings`, `jolt-field`, `jolt-poly`, `jolt-transcript`, `dory-pcs`

#### Public API

```rust
/// Dory polynomial commitment scheme.
///
/// All configuration is instance-local — no global state.
/// Parameters are passed at construction and threaded through
/// all operations.
pub struct DoryScheme {
    params: DoryParams,
}

/// Instance-local Dory parameters (replaces the old globals).
pub struct DoryParams {
    pub t: usize,
    pub max_num_rows: usize,
    pub num_columns: usize,
}

impl DoryScheme {
    pub fn new(params: DoryParams) -> Self;
}

impl CommitmentScheme for DoryScheme {
    type Field = ark_bn254::Fr; // via jolt-field arkworks impl
    type Commitment = DoryCommitment;
    type Proof = DoryProof;
    type ProverSetup = DoryProverSetup;
    type VerifierSetup = DoryVerifierSetup;
    // ... all trait methods
}

impl HomomorphicCommitmentScheme for DoryScheme {
    type BatchedProof = DoryBatchedProof;
    // ... all trait methods
}

impl StreamingCommitmentScheme for DoryScheme {
    type PartialCommitment = DoryPartialCommitment;
    // ... all trait methods
}
```

#### Testing

- **Unit tests:** Commit → open → verify round-trip for small polynomials
- **Integration tests:** End-to-end with `jolt-sumcheck` and `jolt-spartan`
- **Property tests:** Random polynomial, random point → prove → verify

#### File Structure

```
jolt-dory/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs
│   ├── scheme.rs           # DoryScheme, impl CommitmentScheme
│   ├── params.rs           # DoryParams (instance-local)
│   ├── commitment.rs       # DoryCommitment, DoryProof types
│   ├── streaming.rs        # StreamingCommitmentScheme impl
│   └── error.rs
├── tests/                  # Integration tests
│   ├── commitment.rs       # Test commitment round-trips
│   ├── streaming.rs        # Test streaming commitment
│   └── batching.rs         # Test batched operations
├── fuzz/
│   └── fuzz_targets/
│       └── verify.rs
└── benches/
    └── dory.rs
```

---

### 4.9 `jolt-zkvm` — zkVM Prover/Verifier

**Purpose:** The top-level zkVM that orchestrates all sub-crates into a complete proving system. This is Jolt-specific and replaces the old `jolt-core`. It is the last crate to be built.

**Dependencies:** All other `jolt-*` crates

#### Public API

```rust
// ── Core prover/verifier ────────────────────────────────────

pub struct JoltProver<PCS: CommitmentScheme> { /* ... */ }

impl<PCS: HomomorphicCommitmentScheme> JoltProver<PCS> {
    pub fn new(config: ProverConfig, pcs_setup: PCS::ProverSetup) -> Self;

    pub fn prove<T: Transcript>(
        &self,
        trace: ExecutionTrace,
        transcript: &mut T,
    ) -> Result<JoltProof<PCS>, JoltError>;
}

pub struct JoltVerifier<PCS: CommitmentScheme> { /* ... */ }

impl<PCS: HomomorphicCommitmentScheme> JoltVerifier<PCS> {
    pub fn new(pcs_setup: PCS::VerifierSetup) -> Self;

    pub fn verify<T: Transcript>(
        &self,
        proof: &JoltProof<PCS>,
        transcript: &mut T,
    ) -> Result<(), JoltError>;
}

// ── Configuration ───────────────────────────────────────────

pub struct ProverConfig {
    pub memory_layout: MemoryLayout,
    pub first_round_strategy: FirstRoundStrategy,
    // ... other config
}

// ── Proof type ──────────────────────────────────────────────

pub struct JoltProof<PCS: CommitmentScheme> { /* ... */ }

// ── Error ───────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum JoltError {
    #[error("spartan error: {0}")]
    Spartan(#[from] SpartanError),
    #[error("sumcheck error: {0}")]
    Sumcheck(#[from] SumcheckError),
    #[error("opening error: {0}")]
    Opening(#[from] OpeningsError),
    #[error("instruction error: {0}")]
    Instruction(String),
    #[error("memory error: {0}")]
    Memory(String),
}
```

#### Internal Modules

The zkVM contains Jolt-specific protocol logic that implements `SumcheckInstanceProver` for various sub-protocols:

- **RAM checking** — read/write memory consistency via sumcheck
- **Register checking** — register read/write consistency
- **Bytecode checking** — program code verification
- **Claim reductions** — batching claims from different sub-protocols
- **Instruction lookups** — connecting execution trace to lookup tables

Each of these implements the `SumcheckInstanceProver` trait from `jolt-sumcheck`, keeping the sumcheck engine generic while the witness-generation logic is Jolt-specific.

#### File Structure

```
jolt-zkvm/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs
│   ├── prover.rs           # JoltProver
│   ├── verifier.rs         # JoltVerifier
│   ├── config.rs           # ProverConfig
│   ├── proof.rs            # JoltProof, serialization
│   ├── trace.rs            # ExecutionTrace
│   ├── error.rs            # JoltError
│   ├── ram/                # RAM consistency checking
│   │   ├── mod.rs
│   │   ├── read_write.rs
│   │   └── output.rs
│   ├── registers/          # Register checking
│   │   ├── mod.rs
│   │   └── read_write.rs
│   ├── bytecode/           # Bytecode verification
│   │   ├── mod.rs
│   │   └── read_checking.rs
│   ├── claim_reductions/   # Claim batching
│   │   ├── mod.rs
│   │   ├── advice.rs
│   │   ├── hamming.rs
│   │   ├── increments.rs
│   │   └── lookups.rs
│   └── instruction_lookups/ # Lookup table integration
│       ├── mod.rs
│       └── checking.rs
├── tests/                  # Integration tests
│   ├── simple_programs.rs  # Test small RISC-V programs
│   ├── claim_reduction.rs  # Test claim reduction logic
│   ├── memory_checking.rs  # Test RAM/register consistency
│   └── e2e_proving.rs      # End-to-end proof generation
├── fuzz/
│   └── fuzz_targets/
│       └── trace_verify.rs
└── benches/
    └── proving.rs
```

---

## 5. Testing Strategy

### 5.1 Three-Tier Per-Crate Testing

Every crate implements three levels of testing:

1. **Unit Tests** (`src/**/*.rs`)
   - Co-located with source via `#[cfg(test)] mod tests`
   - Test internal logic and private functions
   - Fast, focused on individual components
   - Rule: **if a function has non-trivial logic, it has a test**

2. **Integration Tests** (`tests/`)
   - Test public API from external perspective
   - Verify trait implementations work correctly
   - Test interactions between modules within the crate
   - Each crate has 3-5 integration test files

3. **Fuzz Tests** (`fuzz/`)
   - Security-critical paths
   - Parser/deserializer robustness
   - Property verification with arbitrary inputs

### 5.2 Test Coverage by Crate

| Crate | Unit Tests | Integration Tests | Fuzz Targets |
|-------|------------|-------------------|--------------|
| `jolt-poly` | Eval, bind, eq-poly identities | API patterns, type interop, math properties | Polynomial construction, evaluation |
| `jolt-openings` | Accumulator logic, RLC batching | Commitment round-trips, batch operations | Proof verification |
| `jolt-sumcheck` | Round verification, degree checks | Protocol completeness, batching, streaming | Arbitrary round polynomials |
| `jolt-spartan` | R1CS satisfaction, key gen | E2E R1CS proving, uniform R1CS, uni-skip | R1CS verification, witness |
| `jolt-instructions` | Each instruction's `execute` | Instruction set coverage, lookup consistency | Instruction decoding |
| `jolt-dory` | Basic commit/open | Full protocol flows, streaming commitment | Commitment verification |
| `jolt-zkvm` | Subprotocol logic | Small program proving, claim reductions | Trace verification |

### 5.3 Property-Based Testing

Use `proptest` for generating random inputs and checking invariants:

| Crate | Properties |
|-------|-----------|
| `jolt-poly` | `evaluate(bind_all(r)) == evaluate(r)`, Schwartz-Zippel |
| `jolt-sumcheck` | Completeness (honest prover always passes), soundness (cheating fails w.h.p.) |
| `jolt-spartan` | Satisfiable witness → valid proof, unsatisfiable → rejection |
| `jolt-instructions` | `execute` matches wrapping native ops, lookup decomposition reconstructs correctly |

### 5.4 Fuzzing

Use `cargo-fuzz` with `libFuzzer` for security-critical code:

| Crate | Fuzz Targets |
|-------|-------------|
| `jolt-field` | Deserialization of arbitrary bytes |
| `jolt-poly` | `DensePolynomial::new` from arbitrary evaluations + evaluate at arbitrary points |
| `jolt-sumcheck` | Verifier with arbitrary round polynomials (soundness) |
| `jolt-openings` | Verifier with arbitrary proofs |

### 5.5 Cross-Crate Integration Tests

Cross-crate integration tests verify interactions between multiple crates:

1. **Sumcheck + Dory:** Prove a polynomial sum claim with real Dory commitments
2. **Spartan + Dory:** End-to-end R1CS proof with Dory opening proofs
3. **Full zkVM:** Execute a small RISC-V program, generate proof, verify

These are separate from per-crate integration tests and focus on the boundaries between crates.

### 5.6 End-to-End Test Infrastructure

Address RFC finding 11 — reduce e2e test boilerplate:

```rust
/// Macro to generate an e2e test from a guest program.
/// Compiles the guest, traces execution, proves, and verifies.
#[macro_export]
macro_rules! jolt_e2e_test {
    ($name:ident, guest = $guest:expr $(, config = $config:expr)?) => { /* ... */ };
}
```

---

## 6. ferris-software Execution Plan

### 6.1 Task Dependency DAG

The ferris-software harness executes tasks in dependency order. The DAG follows role-based task naming conventions:

```
Phase 0 (setup):
  implement-scaffold-workspace

Phase 1a (implementation, parallel):
  implement-jolt-poly
  implement-jolt-instructions     (needs jolt-field only)

Phase 1b (testing, after 1a):
  test-jolt-poly-integration
  test-jolt-instructions-integration
  test-fuzz-jolt-poly

Phase 2a (implementation, after jolt-poly):
  implement-jolt-sumcheck         (needs jolt-poly)
  implement-jolt-openings         (needs jolt-poly)

Phase 2b (testing, after 2a):
  test-jolt-sumcheck-integration
  test-jolt-openings-integration
  test-fuzz-jolt-sumcheck
  test-fuzz-jolt-openings

Phase 3a (implementation, after 2a):
  implement-jolt-spartan          (needs jolt-sumcheck + jolt-openings)
  implement-jolt-dory             (needs jolt-openings)

Phase 3b (testing, after 3a):
  test-jolt-spartan-integration
  test-jolt-dory-integration
  test-cross-crate-integration-1  # Spartan+Dory integration

Phase 4a (implementation, after 3a):
  implement-jolt-zkvm             (needs all crates)

Phase 4b (testing, after 4a):
  test-jolt-zkvm-integration
  test-cross-crate-integration-full  # Full stack integration

Phase 5 (integration):
  implement-integrate-workspace   (wire everything together)

Phase 6 (quality):
  quality-documentation-cleanup   (doc comments, clippy, fmt, READMEs)
  quality-benchmark-suite        (performance benchmarks for all crates)
```

### 6.2 Task Spec File (for `gen-tasks.sh`)

```spec
# Setup tasks
workspace           : implement-scaffold-workspace      : none                     : Create crate workspace structure

# Implementation tasks (implementer role)
src/poly/           : implement-jolt-poly               : none                     : Polynomial types and operations
src/zkvm/instruction: implement-jolt-instructions       : none                     : RISC-V instructions + lookup tables
src/subprotocols/   : implement-jolt-sumcheck           : jolt-poly                : Sumcheck protocol engine
src/poly/commitment : implement-jolt-openings           : jolt-poly                : Commitment scheme traits + accumulators
src/zkvm/spartan    : implement-jolt-spartan            : jolt-sumcheck,jolt-openings : Spartan R1CS prover/verifier
src/poly/dory       : implement-jolt-dory               : jolt-openings            : Dory commitment scheme
src/zkvm/           : implement-jolt-zkvm               : jolt-spartan,jolt-sumcheck,jolt-openings,jolt-instructions,jolt-dory : zkVM prover/verifier
integration         : implement-integrate-workspace     : all                      : Wire crates together in workspace

# Test tasks (tester role)
tests               : test-jolt-poly-integration        : jolt-poly                : Integration tests for jolt-poly
tests               : test-jolt-instructions-integration : jolt-instructions        : Integration tests for jolt-instructions
tests               : test-jolt-sumcheck-integration    : jolt-sumcheck            : Integration tests for jolt-sumcheck
tests               : test-jolt-openings-integration    : jolt-openings            : Integration tests for jolt-openings
tests               : test-jolt-spartan-integration     : jolt-spartan             : Integration tests for jolt-spartan
tests               : test-jolt-dory-integration        : jolt-dory                : Integration tests for jolt-dory
tests               : test-jolt-zkvm-integration        : jolt-zkvm                : Integration tests for jolt-zkvm

# Fuzz test tasks (tester role)
fuzz                : test-fuzz-jolt-poly               : jolt-poly                : Fuzz testing for jolt-poly
fuzz                : test-fuzz-jolt-sumcheck           : jolt-sumcheck            : Fuzz testing for jolt-sumcheck
fuzz                : test-fuzz-jolt-openings           : jolt-openings            : Fuzz testing for jolt-openings

# Cross-crate test tasks (tester role)
tests               : test-cross-crate-integration-1    : jolt-spartan,jolt-dory   : Spartan+Dory integration
tests               : test-cross-crate-integration-full : jolt-zkvm                : Full stack integration

# Quality tasks (quality role)
quality             : quality-documentation-cleanup     : all                      : Documentation, clippy, fmt, READMEs
benchmarks          : quality-benchmark-suite           : all                      : Performance benchmarks for all crates
```

### 6.3 Agent Roles

Following ferris-software conventions:

| Role | Count | Task Glob | Description |
|------|-------|-----------|-------------|
| implementer | 4 | `implement-*` | Clean-room implementation of each crate |
| tester | 3 | `test-*` | Write all tests (unit, integration, fuzz) |
| quality | 2 | `quality-*` | Documentation, code quality, benchmarks |

Role assignment in `roles.conf`:
```
implementer:4:templates/implementer.md:verifiers/scoped.sh:implement-*
tester:3:templates/tester.md:verifiers/scoped.sh:test-*
quality:2:templates/quality.md:verifiers/scoped.sh:quality-*
```

### 6.4 Verification

Each task is verified by the scoped verifier:

```bash
./verifiers/scoped.sh /workdir jolt-poly jolt-sumcheck  # etc.
```

The default verifier runs: `cargo check` → `cargo clippy` → `cargo fmt --check` → `cargo nextest run`

### 6.5 CLAUDE.md for Agents

The project CLAUDE.md template will include:

- This spec (or a digest) as the single source of truth
- The crate dependency graph
- Per-crate API contracts (trait signatures)
- Style guide (reference `jolt-transcript` and `jolt-field`)
- Constraints: no arkworks in public APIs, serde everywhere, thiserror per crate
- File size limit: no file >500 lines, prefer <300

---

## 7. Migration Plan

### 7.1 Phased Approach

| Phase | Action | Outcome |
|-------|--------|---------|
| **0** | Write this spec + generate ferris-software tasks | Spec approved, tasks in `tasks/available/` |
| **1** | Build leaf crates (`jolt-poly`, `jolt-instructions`) | Standalone crates with tests, no jolt-core dependency |
| **2** | Build mid-tier (`jolt-openings`, `jolt-sumcheck`, `jolt-dory`) | Protocol crates with integration tests |
| **3** | Build top-tier (`jolt-spartan`) | R1CS proving works end-to-end |
| **4** | Build `jolt-zkvm` | Full zkVM prover/verifier |
| **5** | Wire `jolt-zkvm` into the workspace | `jolt-sdk`, `jolt-inlines`, examples depend on new crates |
| **6** | Delete `jolt-core/` | Old code removed, `jolt-zkvm` is the new entry point |

### 7.2 Coexistence Strategy

During phases 1–5, both `jolt-core/` and the new `crates/` coexist in the workspace. The workspace `Cargo.toml` includes both. No code in the new crates imports from `jolt-core`. The old `jolt-core` is used only as a reference for correctness (reading the code, not depending on it).

### 7.3 Workspace Structure (Final State)

```
Cargo.toml              # workspace root
crates/
├── jolt-transcript/    # (done)
├── jolt-field/         # (done)
├── jolt-poly/
├── jolt-openings/
├── jolt-sumcheck/
├── jolt-spartan/
├── jolt-instructions/
├── jolt-dory/
├── jolt-kzg/           # (future)
├── jolt-hyperkzg/      # (future)
└── jolt-zkvm/          # replaces jolt-core
tracer/                 # unchanged
common/                 # unchanged (minor mods allowed)
jolt-sdk/               # updated to depend on jolt-zkvm
jolt-inlines/           # unchanged (separate from jolt-instructions)
examples/               # updated to use new crate APIs
```

---

## 8. Open Questions

These should be resolved during implementation, not before:

1. **Streaming sumcheck placement** — does streaming belong in `jolt-sumcheck` (as a variant) or `jolt-zkvm` (as a Jolt-specific optimization)? Tentatively in `jolt-sumcheck` with a `StreamingSumcheckProver` trait.

2. **Univariate skip generality** — is the univariate skip optimization general enough for `jolt-spartan`, or is it inherently tied to Jolt's R1CS structure? Tentatively in `jolt-spartan` as an optional strategy.

3. **CompactPolynomial in jolt-poly vs jolt-zkvm** — the compressed polynomial variants are a Jolt optimization. Should `jolt-poly` define the `SmallScalar` trait and `CompactPolynomial`, or should this be pushed to `jolt-zkvm`? Tentatively in `jolt-poly` since it's a general optimization.

4. **Exact Dory wrapping boundary** — how much of `dory-pcs` gets re-exposed vs. hidden behind the trait? Resolved during `jolt-dory` implementation.

5. **Shared constants between `jolt-instructions` and `tracer`** — the RFC flags duplicated opcodes. A shared `jolt-opcodes` crate or a constants module in `common/` could solve this. Deferred unless it becomes a blocker.
