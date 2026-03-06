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
- New features — no new cryptographic functionality. `jolt-ir` consolidates existing representations (not new crypto), and the kernel IR is deferred.

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
        ├───────────┐
        ▼           │
   jolt-crypto      │         (jolt-field, jolt-transcript)
        │           │
        ├───────────┼─────────────────┐
        ▼           ▼                 ▼
   jolt-poly   jolt-ir          jolt-blindfold
   (jolt-field) (jolt-field)    (jolt-crypto, jolt-sumcheck)
        │           │
        ├───────────┤
        │           │
        ├──────┐    │
        ▼      ▼    │
jolt-openings  jolt-sumcheck ◄────┘
(jolt-crypto,  (jolt-field, jolt-poly,
 jolt-field,    jolt-transcript,
 jolt-poly,     jolt-ir)
 jolt-trscpt)
        │              │
        ├──────┬───────┘
        ▼      ▼
   jolt-spartan
   (jolt-sumcheck, jolt-openings,
    jolt-ir)
        │
        ▼
   jolt-instructions
   (jolt-field)
        │
        ▼
   jolt-zkvm
   (jolt-spartan, jolt-sumcheck,
    jolt-openings, jolt-instructions,
    jolt-ir, jolt-field, jolt-crypto,
    jolt-poly, jolt-transcript)
        │
        ├──► jolt-dory     (jolt-openings, jolt-crypto, dory-pcs)
        ├──► jolt-kzg      (jolt-openings, jolt-crypto)  [future]
        └──► jolt-hyperkzg (jolt-openings)                [future]
```

### 3.2 Crate Summary

| Crate | Purpose | Reusable | LOC Estimate |
|-------|---------|----------|--------------|
| `jolt-transcript` | Fiat-Shamir transcripts | Yes | ~500 (done) |
| `jolt-field` | Field arithmetic traits + arkworks impl | Yes | ~2000 (done) |
| `jolt-crypto` | Group, pairing, and commitment abstractions | Yes | ~1200 (done) |
| `jolt-poly` | Polynomial types and operations | Yes | ~1500 (done) |
| `jolt-openings` | Commitment scheme traits + opening accumulators | Yes | ~800 (done) |
| `jolt-sumcheck` | Sumcheck protocol engine | Yes | ~1100 (done) |
| `jolt-spartan` | R1CS + Spartan prover/verifier | Yes | ~925 (done) |
| `jolt-ir` | Symbolic expression IR for sumcheck claims | Yes | ~1500 |
| `jolt-instructions` | RISC-V instruction set + lookup tables | No | ~3000 (done) |
| `jolt-dory` | Dory commitment scheme impl | No | ~5000 (done) |
| `jolt-zkvm` | zkVM prover/verifier orchestration | No | ~10000 (in progress) |

---

## 4. Per-Crate Specifications

### 4.1 `jolt-transcript` — **DONE**

Already completed. Defines `Transcript` and `AppendToTranscript` traits with Blake2b and Keccak implementations. See `crates/jolt-transcript/`.

### 4.2 `jolt-field` — **DONE**

Already completed. Defines `Field`, `UnreducedOps`, `ReductionOps`, `Challenge`, `WithChallenge`, `OptimizedMul`, accumulation traits (`FMAdd`, `BarrettReduce`, `MontgomeryReduce`), and arkworks BN254 implementation. See `crates/jolt-field/`.

---

### 4.2b `jolt-crypto` — Backend-Agnostic Cryptographic Primitives — **DONE**

**Purpose:** Backend-agnostic cryptographic group and commitment abstractions for the Jolt proving system. Defines the core group traits (`JoltGroup`, `PairingGroup`) and a vector commitment interface (`JoltCommitment`) that the rest of the stack programs against. The BN254 implementation wraps arkworks internally, but no arkworks types appear in the public API.

**Dependencies:** `jolt-field`, `jolt-transcript`

**Dependency position:**

```
jolt-field ← jolt-crypto ← jolt-openings, jolt-dory, jolt-blindfold, jolt-zkvm
             jolt-transcript ↗
```

**Design decisions:**
- **Additive notation everywhere.** `JoltGroup` uses `Add`/`Sub`/`Neg` for the group operation, even for GT (where "addition" maps to Fq12 multiplication). This gives uniform code for any group without the caller needing to know the underlying algebra.
- **`identity()`/`is_identity()` naming.** Instead of `zero()`/`is_zero()`, to stay notation-agnostic.
- **MSM lives on `JoltGroup`.** Every group gets `msm()` — no need for separate `g1_msm`/`g2_msm` methods on the pairing type.
- **Generators and randomness are PCS concerns.** `Bn254::g1_generator()`, `random_g1()` are inherent methods on the concrete `Bn254` type, not on `PairingGroup`. Different PCS have different generator requirements; the trait shouldn't prescribe them.
- **`AppendToTranscript` supertrait on `JoltGroup`.** Group elements must be absorbable into Fiat-Shamir transcripts (needed for committed sumcheck, Pedersen commitments in BlindFold).
- **`JoltCommitment` is a separate trait.** Defines `Setup`, `Commitment`, `capacity()`, `commit()`, `verify()`. Protocol code is generic over this trait, enabling Pedersen, hash-based, or lattice commitments.
- **`Pedersen<G>` blanket impl.** Any `JoltGroup` automatically gets Pedersen commitments via `impl<G: JoltGroup> JoltCommitment for Pedersen<G>`. No per-backend boilerplate.
- **`PairingGroup` has `type ScalarField: Field`.** Associates the scalar field with the pairing, enabling generic code to work with field elements without a separate generic parameter.
- **`bn254` feature flag (default).** All arkworks dependencies are optional, gated behind `bn254`. Without it, the crate compiles to trait definitions only — enforcing that the public API is genuinely backend-agnostic.
- **`#[repr(transparent)]` newtypes.** `Bn254G1`, `Bn254G2`, `Bn254GT` wrap arkworks projective points with zero overhead. Custom `Serialize`/`Deserialize` impls delegate to arkworks canonical serialization internally while presenting serde externally.

#### Core Traits

```rust
/// Cryptographic group suitable for commitments.
/// Additive notation — the underlying algebra may be multiplicative (GT).
pub trait JoltGroup:
    Copy + Debug + Default + Eq + Send + Sync + 'static
    + Add<Output = Self> + Sub<Output = Self> + Neg<Output = Self>
    + for<'a> Add<&'a Self, Output = Self> + for<'a> Sub<&'a Self, Output = Self>
    + AddAssign + SubAssign
    + Serialize + for<'de> Deserialize<'de>
    + AppendToTranscript
{
    fn identity() -> Self;
    fn is_identity(&self) -> bool;
    fn double(&self) -> Self;
    fn scalar_mul<F: Field>(&self, scalar: &F) -> Self;
    fn msm<F: Field>(bases: &[Self], scalars: &[F]) -> Self;
}

/// Pairing-friendly group for bilinear-map-based schemes (Dory, KZG).
pub trait PairingGroup: Clone + Sync + Send + 'static {
    type ScalarField: Field;
    type G1: JoltGroup;
    type G2: JoltGroup;
    type GT: JoltGroup;

    fn pairing(g1: &Self::G1, g2: &Self::G2) -> Self::GT;
    fn multi_pairing(g1s: &[Self::G1], g2s: &[Self::G2]) -> Self::GT;
}

/// Backend-agnostic vector commitment.
pub trait JoltCommitment: Clone + Send + Sync + 'static {
    type Setup: Clone + Send + Sync;
    type Commitment: Copy + Debug + Default + Eq + Send + Sync + 'static
        + Serialize + for<'de> Deserialize<'de> + AppendToTranscript;

    fn capacity(setup: &Self::Setup) -> usize;
    fn commit<F: Field>(setup: &Self::Setup, values: &[F], blinding: &F) -> Self::Commitment;
    fn verify<F: Field>(setup: &Self::Setup, commitment: &Self::Commitment,
        values: &[F], blinding: &F) -> bool;
}
```

#### Pedersen Commitment

```rust
/// Blanket JoltCommitment for any JoltGroup.
/// C = Σᵢ values[i] * generators[i] + blinding * H
impl<G: JoltGroup> JoltCommitment for Pedersen<G> {
    type Setup = PedersenSetup<G>;  // message_generators + blinding_generator
    type Commitment = G;
    // ...
}
```

#### BN254 Concrete Types

| Type | Wraps | Notes |
|------|-------|-------|
| `Bn254` | — | `PairingGroup` impl; inherent: `g1_generator()`, `g2_generator()`, `random_g1()` |
| `Bn254G1` | `G1Projective` | `JoltGroup` with arkworks MSM (`msm_bigint`) |
| `Bn254G2` | `G2Projective` | `JoltGroup` with arkworks MSM |
| `Bn254GT` | `Fq12` | `JoltGroup` with additive notation (Add=mul, Neg=inv, identity=ONE) |

#### Testing

64 tests across 4 test modules (group laws, pairing, Pedersen, serialization), 14 criterion benchmarks, 3 fuzz targets (deserialization safety, group arithmetic invariants, Pedersen commitment properties).

#### Directory Layout

```
crates/jolt-crypto/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Re-exports
│   ├── group.rs            # JoltGroup trait
│   ├── pairing.rs          # PairingGroup trait
│   ├── commitment.rs       # JoltCommitment trait
│   ├── pedersen.rs         # Pedersen<G> + PedersenSetup<G>
│   └── arkworks/
│       └── bn254/
│           ├── mod.rs      # Bn254, PairingGroup impl, field_to_fr bridge
│           ├── g1.rs       # Bn254G1: JoltGroup
│           ├── g2.rs       # Bn254G2: JoltGroup
│           └── gt.rs       # Bn254GT: JoltGroup (additive notation)
├── tests/
│   ├── group_laws.rs       # G1/G2 algebraic properties (22 tests)
│   ├── pairing.rs          # Bilinearity, GT group laws (17 tests)
│   ├── pedersen.rs         # Commit/verify, homomorphism, binding (11 tests)
│   └── serialization.rs   # JSON/bincode roundtrips (14 tests)
├── benches/crypto.rs       # 14 criterion benchmarks
└── fuzz/fuzz_targets/      # 3 fuzz targets
```

---

### 4.3 `jolt-poly` — Polynomial Library — **DONE**

**Purpose:** Generic polynomial types and operations for multilinear, univariate, and specialized polynomials. This crate is backend-agnostic and reusable outside Jolt.

**Dependencies:** `jolt-field`

**Deviations from spec:**
- `CompactPolynomial<S, F>` replaced by unified `Polynomial<T>` with `bind_to_field<F: From<T>>()` — simpler, one generic struct for all scalar types
- `SmallScalar` trait dropped — `From<T>` bound used instead for field promotion
- `LagrangePolynomial` not implemented (not needed by downstream crates)
- `bind_in_place` renamed to `bind` (in-place is the default; allocating bind is `bind_to_field`)
- `CompressedPoly<F>` added (univariate with hint-based linear term recovery, used by sumcheck)
- `UnivariatePolynomial` trait added alongside `UnivariatePoly` concrete type

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
    fn bind(&self, scalar: F) -> Polynomial<F>;

    /// Read-only access to evaluations (may allocate if compressed).
    fn evaluations(&self) -> Cow<[F]>;
}

// ── Concrete types ──────────────────────────────────────────

/// Dense multilinear polynomial: stores all $2^n$ evaluations as `Vec<F>`.
pub struct Polynomial<F: Field> {
    evaluations: Vec<F>,
    num_vars: usize,
}

impl<F: Field> Polynomial<F> {
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
- **Fuzz targets:** `Polynomial::new` with arbitrary byte inputs, evaluate with arbitrary points

#### File Structure

```
jolt-poly/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Re-exports, module declarations
│   ├── traits.rs           # MultilinearPolynomial trait
│   ├── dense.rs            # Polynomial (with unit tests)
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

### 4.4 `jolt-openings` — Polynomial Commitment Scheme Traits & Opening Reduction — **REDESIGN**

**Purpose:** Abstract polynomial commitment scheme (PCS) interfaces, opening claim types, and reduction traits for batching claims. Designed to support homomorphic (EC, lattice), and hash-based (FRI) commitment schemes without leaking implementation details. The crate is intentionally thin — it defines abstractions and provides reusable primitives, not protocol-level orchestration.

**Dependencies:** `jolt-field`, `jolt-poly`, `jolt-crypto`, `jolt-transcript`

**Key design decisions (vs. previous spec):**
- **No accumulators.** The old `ProverOpeningAccumulator` / `VerifierOpeningAccumulator` are removed. Claims are plain data (`Vec<ProverClaim>`, `Vec<VerifierClaim>`), collected by the caller and passed to reduction functions. No statefulness.
- **Reduction is separate from proving.** The `OpeningReduction` trait defines a *claim transformation* (many claims → fewer claims), not a proving step. After reduction, the caller opens each remaining claim individually via `CommitmentScheme::open` / `verify`. Reductions compose.
- **No batching baked into PCS traits.** `batch_prove` / `batch_verify` are removed from the trait hierarchy. Batching is a reduction strategy (`RlcReduction`), not a PCS property. The PCS only ever opens single claims.
- **`Commitment` base trait from jolt-crypto.** `CommitmentScheme` extends `jolt_crypto::Commitment`, sharing the `Output` associated type. This avoids duplicating commitment output bounds and connects the PCS to the vector commitment hierarchy.
- **`VerifierClaim` is fully typed.** No `Box<dyn Any>` type erasure. The commitment type is a generic parameter, enforced at compile time.

#### Trait Hierarchy

`CommitmentScheme` extends `jolt_crypto::Commitment`, which provides the shared `Output` type (the commitment value). The full hierarchy across jolt-crypto and jolt-openings:

```
                    Commitment              (jolt-crypto: just Output type)
                   /          \
    VectorCommitment            CommitmentScheme    (jolt-openings: + open/verify)
          |                          |
  BlindableCommitment        AdditivelyHomomorphic  (+ combine)
                                     |
                             StreamingCommitment    (+ incremental)
```

#### PCS Traits

```rust
use jolt_crypto::Commitment;

/// Multilinear polynomial commitment scheme.
///
/// Extends `Commitment` (from jolt-crypto) with opening proofs: the ability
/// to prove and verify that a committed polynomial evaluates to a claimed
/// value at a given point. No algebraic structure is assumed on commitments —
/// this trait covers group-based (Dory, KZG), lattice-based, and hash-based
/// (FRI, Brakedown) schemes.
pub trait CommitmentScheme: Commitment {
    type Field: Field;
    type Proof: Clone + Send + Sync + Serialize + DeserializeOwned;
    type ProverSetup: Clone + Send + Sync;
    type VerifierSetup: Clone + Send + Sync + Serialize + DeserializeOwned;

    fn commit(evals: &[Self::Field], setup: &Self::ProverSetup) -> Self::Output;

    fn open(
        evals: &[Self::Field],
        point: &[Self::Field],
        eval: Self::Field,
        setup: &Self::ProverSetup,
        transcript: &mut impl Transcript,
    ) -> Self::Proof;

    fn verify(
        commitment: &Self::Output,
        point: &[Self::Field],
        eval: Self::Field,
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut impl Transcript,
    ) -> Result<(), OpeningsError>;
}

/// Commitments live in an additive group and can be linearly combined.
///
/// This is the algebraic property that enables batch reduction via RLC.
/// For EC-based PCS, `combine` is MSM. For lattice-based PCS, the
/// operation may be more involved.
pub trait AdditivelyHomomorphic: CommitmentScheme {
    fn combine(
        commitments: &[Self::Output],
        scalars: &[Self::Field],
    ) -> Self::Output;
}

/// Commit incrementally without materializing the full evaluation table.
pub trait StreamingCommitment: CommitmentScheme {
    type Partial: Clone + Send + Sync;

    fn begin(setup: &Self::ProverSetup) -> Self::Partial;
    fn feed(partial: &mut Self::Partial, chunk: &[Self::Field]);
    fn finish(partial: Self::Partial) -> Self::Output;
}
```

#### Claim Types

Plain data, no behavior. Collected by the caller (typically `jolt-zkvm`) into `Vec`s.

```rust
/// A leaf claim the prover needs to open via PCS.
pub struct ProverClaim<F: Field> {
    pub evaluations: Vec<F>,
    pub point: Vec<F>,
    pub eval: F,
}

/// A leaf claim the verifier needs to check via PCS.
pub struct VerifierClaim<F: Field, C> {
    pub commitment: C,
    pub point: Vec<F>,
    pub eval: F,
}
```

#### Opening Reduction Trait

Reduction is a *claim transformation*, not a proving step. It takes many claims and produces fewer equivalent claims. The PCS then opens each remaining claim individually. Reductions compose because the output type equals the input type.

```rust
/// Transforms multiple opening claims into fewer equivalent claims.
///
/// The reduction may draw Fiat-Shamir challenges (e.g., RLC) or run
/// sub-protocols (e.g., sumcheck-based reduction). Implementations
/// define the strategy. The trait is implemented by callers (jolt-zkvm)
/// for protocol-specific reductions, with `RlcReduction` provided as
/// a default for homomorphic schemes.
pub trait OpeningReduction<PCS: CommitmentScheme> {
    /// Proof artifact from the reduction itself, if any.
    /// `()` for deterministic reductions like RLC.
    type ReductionProof: Clone + Send + Sync + Serialize + DeserializeOwned;

    fn reduce_prover(
        claims: Vec<ProverClaim<PCS::Field>>,
        transcript: &mut impl Transcript,
    ) -> (Vec<ProverClaim<PCS::Field>>, Self::ReductionProof);

    fn reduce_verifier(
        claims: Vec<VerifierClaim<PCS::Field, PCS::Output>>,
        proof: &Self::ReductionProof,
        transcript: &mut impl Transcript,
    ) -> Result<Vec<VerifierClaim<PCS::Field, PCS::Output>>, OpeningsError>;
}
```

Example reductions:

| Reduction | Bound | Input | Output | `ReductionProof` |
|-----------|-------|-------|--------|------------------|
| `RlcReduction` | `PCS: AdditivelyHomomorphic` | N claims at K points | K claims (one per point) | `()` |
| `SumcheckReduction` (jolt-zkvm) | `PCS: CommitmentScheme` | K claims at K points | 1 claim at a single point | `SumcheckProof<F>` |

#### RLC Utilities

```rust
/// RLC of polynomial evaluation tables using Horner's method.
pub fn rlc_combine<F: Field>(polynomials: &[&[F]], rho: F) -> Vec<F>;

/// RLC of scalar evaluations using Horner's method.
pub fn rlc_combine_scalars<F: Field>(evals: &[F], rho: F) -> F;
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

#### Protocol Usage (in jolt-zkvm)

```rust
// Each protocol stage returns its leaf claims
let (proof1, leaves1) = stage1::prove(&witnesses, &mut transcript);
let (proof2, leaves2) = stage2::prove(&stage1.forwarded, &mut transcript);

// Collect all leaf claims
let all_leaves = [leaves1, leaves2].concat();

// Reduce: RLC groups claims by point, combines via homomorphism
let (reduced, _) = RlcReduction::reduce_prover(all_leaves, &mut transcript);

// Open each reduced claim individually via PCS
let opening_proofs: Vec<_> = reduced.iter().map(|claim| {
    PCS::open(&claim.evaluations, &claim.point, claim.eval, &setup, &mut transcript)
}).collect();
```

#### Testing

- **Unit tests:** RLC combine correctness (Horner, commutativity with evaluation), reduction grouping, claim round-trips
- **Property tests:** For any random claims, `reduce → open → verify` succeeds; tampered evaluations cause rejection
- **Integration tests:** Round-trip with `MockCommitmentScheme`, round-trip with `jolt-dory`
- **Benchmarks:** `rlc_combine`, `rlc_combine_scalars` at various polynomial counts and sizes

#### File Structure

```
jolt-openings/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # re-exports
│   ├── traits.rs           # CommitmentScheme, AdditivelyHomomorphic, StreamingCommitment
│   ├── claims.rs           # ProverClaim, VerifierClaim
│   ├── reduction.rs        # OpeningReduction trait + RlcReduction impl
│   ├── rlc.rs              # rlc_combine, rlc_combine_scalars
│   ├── error.rs            # OpeningsError
│   └── mock.rs             # MockCommitmentScheme (test-utils feature)
├── tests/
│   └── reduction.rs        # Integration tests for reduction round-trips
├── fuzz/
│   └── fuzz_targets/
│       └── rlc.rs
└── benches/
    └── rlc.rs
```

---

### 4.5 `jolt-sumcheck` — Sumcheck Protocol Engine — **DONE**

**Purpose:** Generic implementation of the sum-check protocol, including batched and streaming variants. Reusable for any sum-check application, not just Jolt.

**Dependencies:** `jolt-field`, `jolt-poly`, `jolt-transcript`

**Deviations from spec:**
- `SumcheckInstanceProver` / `SumcheckInstanceVerifier` traits replaced by simpler `SumcheckWitness` trait (witness provides `round_polynomial` + `bind`)
- `BatchedSumcheckProof` not a separate type — batched prover returns `SumcheckProof<F>` (same structure)
- Prover/verifier accept a `challenge_fn` closure for Fiat-Shamir flexibility instead of `&mut impl Transcript` directly
- `WrongNumberOfRounds` error variant added (not in original spec)
- 27 test functions (exceeds the spec's testing requirements)

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

### 4.6 `jolt-spartan` — Generic Spartan SNARK — **DONE**

**Purpose:** Generic Spartan-based SNARK for arbitrary R1CS constraint systems. Proves satisfaction of R1CS constraints produced by any source (jolt-ir R1CS backend, hand-written, etc.). Generic over the commitment scheme and field.

**Scope:** `jolt-spartan` is a general-purpose Spartan implementation for arbitrary R1CS. It does **not** implement the main zkVM's outer sumcheck, which uses a specialized evaluation path (`UniformSpartanKey` with lazy constraint evaluation, two constraint groups, univariate skip with Jolt-specific domain sizes). The main zkVM outer sumcheck stays as custom code in `jolt-zkvm`, using `jolt-sumcheck` for the protocol and `jolt-ir` for constraint definitions.

`jolt-spartan` serves three consumers:
1. **BlindFold** — proves the verifier R1CS after Nova folding. Requires relaxed R1CS support.
2. **Recursive verification** — proving Jolt verifier execution as R1CS (constraints produced by jolt-ir).
3. **Standalone use** — any R1CS satisfaction proof outside Jolt.

**Dependencies:** `jolt-sumcheck`, `jolt-openings`, `jolt-field`, `jolt-poly`, `jolt-transcript`

**Deviations from spec:**
- `UniformR1CS` not implemented (will be added during `jolt-zkvm` integration)
- `SimpleR1CS` added as a sparse-triple R1CS for testing (not in original spec)
- `SpartanProver::prove` takes the R1CS instance directly (not just the key)
- `EvaluationMismatch` error variant added
- `FirstRoundStrategy::UnivariateSkip` defined but implementation deferred (enum stub only)

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

### 4.7 `jolt-instructions` — RISC-V Instruction Set & Lookup Tables — **DONE**

**Purpose:** Defines the Jolt instruction set (RISC-V base + virtual instructions) and their decomposition into lookup tables. This is Jolt-specific and not intended for reuse outside the project.

**Dependencies:** `jolt-field`

**Deviations from spec:**
- `Instruction` trait takes `(x: u64, y: u64)` instead of `&[u64]` — more explicit for the two-operand RISC-V ISA
- `JoltInstructionSet` is a concrete registry struct (not trait), uses O(1) opcode dispatch
- `define_instruction!` macro generates uniform implementations — not in original spec but ensures consistency
- `JAL`, `JALR` not included (handled by VM control flow, not as data instructions)
- W-suffix variants (`ADDW`, `SUBW`, etc.) added for RV64 compatibility
- 68 instructions total (exceeds original spec's list)
- Lookup table implementations deferred (stub `lookups()` returns empty vec)

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

### 4.8 `jolt-dory` — Dory Commitment Scheme — **DONE**

**Purpose:** Implements `CommitmentScheme` and `HomomorphicCommitmentScheme` from `jolt-openings` using the Dory polynomial commitment scheme. Wraps the external `dory-pcs` crate. All parameters are instance-local (no globals).

**Dependencies:** `jolt-openings`, `jolt-field`, `jolt-poly`, `jolt-transcript`, `dory-pcs`

**Deviations from spec:**
- `optimizations/` module added (ported from `jolt-optimizations` in the arkworks fork — GLV, batch addition, vector ops)
- `DoryStreamingCommitter` added as a separate helper struct (not a trait method on `DoryScheme`)
- `StreamingCommitmentScheme` trait impl is a stub (streaming done via `DoryStreamingCommitter` directly)
- `transcript.rs` added for `JoltToDoryTranscript` adapter
- `types.rs` added with field conversion utilities (`jolt_fr_to_ark`, `ark_to_jolt_fr`) and newtype wrappers
- ~5000 LOC (larger than spec's ~1500 estimate due to optimizations module)

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

#### Constraint Definitions

`jolt-zkvm` defines all uniform R1CS constraints and sumcheck claim formulas using `jolt-ir`'s `ExprBuilder`. This replaces the compile-time `lc!` / `r1cs_eq_conditional!` macro system with natural arithmetic expressions. At initialization:

1. Each constraint is built as a `ClaimDefinition` via `ExprBuilder`
2. Degree-2 constraints are factored into bilinear `(a, b)` pairs for the fused evaluator
3. `SumOfProducts` form is cached for BlindFold R1CS emission

The main zkVM outer sumcheck uses the factored bilinear pairs with a specialized evaluation path (`evaluate_inner_sum_product_at_point`), NOT `jolt-spartan`. This is a performance-critical design decision: the lazy evaluation avoids materializing full R1CS matrices.

#### Internal Modules

The zkVM contains Jolt-specific protocol logic that implements `SumcheckInstanceProver` for various sub-protocols:

- **RAM checking** — read/write memory consistency via sumcheck
- **Register checking** — register read/write consistency
- **Bytecode checking** — program code verification
- **Claim reductions** — batching claims from different sub-protocols
- **Instruction lookups** — connecting execution trace to lookup tables
- **Spartan outer** — specialized outer sumcheck with univariate skip (custom, not `jolt-spartan`)

Each sub-protocol implements `SumcheckInstanceProver` from `jolt-sumcheck` and provides its claim formula as a `ClaimDefinition` from `jolt-ir`. The sumcheck engine is generic; witness-generation and constraint evaluation are Jolt-specific.

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
│   ├── r1cs/               # Uniform R1CS (jolt-ir-based constraint definitions)
│   │   ├── mod.rs
│   │   ├── constraints.rs   # ClaimDefinitions for all 19 constraints
│   │   ├── inputs.rs        # Variable map (OpeningBinding metadata)
│   │   └── key.rs           # UniformSpartanKey (lazy evaluation)
│   ├── spartan/             # Specialized outer sumcheck (NOT jolt-spartan)
│   │   ├── mod.rs
│   │   ├── outer.rs         # Outer sumcheck with univariate skip
│   │   ├── product.rs       # Product virtualization sumcheck
│   │   └── shift.rs         # Shift sumcheck
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

### 4.10 `jolt-ir` — Symbolic Expression IR for Sumcheck Claims

**Purpose:** Provide a single expression IR that serves as the **source of truth** for all sumcheck claim formulas across every backend: runtime evaluation, BlindFold ZK (R1CS generation), formal verification (Lean4), circuit transpilation (gnark/Groth16), and GPU kernel compilation. Eliminates the sync hazard between redundant hand-written representations (see RFC finding 12).

**Dependencies:** `jolt-field`

**Motivation and prior art:**

Today, every sumcheck instance in Jolt encodes its claim formula in up to four separate, incompatible representations:

1. `SumcheckInstanceParams::input_claim()` — imperative Rust computing a scalar `F` value at runtime (`jolt-core/src/subprotocols/sumcheck_verifier.rs:54`)
2. `SumcheckInstanceParams::input_claim_constraint()` / `output_claim_constraint()` — hand-written `OutputClaimConstraint` structs (`jolt-core/src/subprotocols/blindfold/output_constraint.rs`) using `ProductTerm` / `ValueSource` for BlindFold R1CS generation
3. `MleAst` — a symbolic field type (`zklean-extractor/src/mle_ast.rs`) that implements `JoltField` and records operations as AST nodes in a global mutable arena, used for Lean4 extraction and extended by the gnark transpiler (PR [#1322](https://github.com/a16z/jolt/pull/1322))
4. `ClaimExpr<F>` — an expression tree (`jolt-core/src/subprotocols/sumcheck_claim.rs:139-145`) with `Constant`, `Var`, `Add`, `Mul`, `Sub` nodes, used by `SumcheckFrontend` for Lean4 claim extraction

These representations must all compute the same formula. Desynchronization between (1) and (2) causes BlindFold R1CS unsatisfiability — a critical invariant called out in CLAUDE.md. Each new backend (GPU kernels, recursion circuits) would require adding yet another representation.

`jolt-ir` consolidates all four into a single IR. Developers define each formula **once** using an `ExprBuilder` API. All backends consume that definition via a `ExprVisitor` trait. The formula is pure data — no field arithmetic, no side effects, no global state.

#### Philosophy

**The IR is the source of truth for claim-level expressions.** The standard verifier's runtime evaluation, BlindFold's R1CS constraints, the Lean4 extractor, and the circuit transpiler all derive their output from the same `Expr`. No hand-written parallel implementations.

**The IR does NOT own verifier orchestration logic.** Transcript operations, stage sequencing, commitment verification, and point normalization remain as Rust code in `jolt-sumcheck` / `jolt-zkvm`. The IR covers the ~20 formulas (one per sumcheck instance) that define claim composition, not the entire verifier pipeline.

**Tracing is complementary, not replaced.** For full-verifier capture (gnark transpilation, comprehensive Lean4 extraction), the `MleAst` tracing approach (running the verifier with a symbolic field type) remains appropriate. But tracing now *consumes* the canonical `Expr` when it encounters a claim check, rather than re-deriving the formula from scratch. The tracing infrastructure (`TracingField`) may live in a separate `jolt-ir-trace` crate that depends on `jolt-field` and `jolt-ir`.

#### Public API

```rust
// ── Variables ───────────────────────────────────────────────

/// A symbolic variable in a sumcheck claim expression.
///
/// Variables are late-bound: they carry an identifier but no
/// concrete value. Each backend resolves variables differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Var {
    /// The evaluation of a polynomial at the sumcheck challenge point.
    /// The `u32` is an opaque identifier assigned by the sumcheck framework.
    Opening(u32),
    /// A Fiat-Shamir challenge value (batching coefficient, etc.).
    Challenge(u32),
}

// ── Expression nodes ────────────────────────────────────────

/// A node in a symbolic field expression DAG.
///
/// Constants are `i128`, covering all practical field constants
/// without making the IR generic over `F`. Backends promote to
/// the target field during evaluation.
#[derive(Debug, Clone, Copy)]
pub enum ExprNode {
    Constant(i128),
    Var(Var),
    Neg(ExprId),
    Add(ExprId, ExprId),
    Mul(ExprId, ExprId),
    Sub(ExprId, ExprId),
}

/// Stable index into an expression arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExprId(u32);

/// Instance-local expression arena. No global state.
pub struct ExprArena {
    nodes: Vec<ExprNode>,
}

/// A complete symbolic expression: arena + root node.
pub struct Expr {
    arena: ExprArena,
    root: ExprId,
}

// ── Builder (ergonomic construction) ────────────────────────

/// Builder for constructing expressions with operator overloading.
///
/// Usage:
/// ```
/// let mut b = ExprBuilder::new();
/// let h = b.opening(0);
/// let gamma = b.challenge(0);
/// let expr = b.build(gamma * (h * h - h));
/// ```
pub struct ExprBuilder {
    arena: ExprArena,
}

impl ExprBuilder {
    pub fn new() -> Self;
    pub fn opening(&self, id: u32) -> ExprHandle<'_>;
    pub fn challenge(&self, id: u32) -> ExprHandle<'_>;
    pub fn constant(&self, val: i128) -> ExprHandle<'_>;
    pub fn zero(&self) -> ExprHandle<'_>;
    pub fn one(&self) -> ExprHandle<'_>;
    pub fn build(self, root: ExprHandle<'_>) -> Expr;
}

/// A handle to a node in the builder, supporting `+`, `-`, `*`.
#[derive(Clone, Copy)]
pub struct ExprHandle<'a> { /* builder ref + ExprId */ }

impl Add for ExprHandle<'_> { type Output = Self; /* ... */ }
impl Sub for ExprHandle<'_> { type Output = Self; /* ... */ }
impl Mul for ExprHandle<'_> { type Output = Self; /* ... */ }
impl Neg for ExprHandle<'_> { type Output = Self; /* ... */ }

// ── Claim definition ────────────────────────────────────────

/// A sumcheck claim formula with variable binding metadata.
///
/// This is the primary artifact that sumcheck instances produce.
/// It replaces the dual `input_claim()` + `input_claim_constraint()`
/// pattern with a single definition.
pub struct ClaimDefinition {
    /// The symbolic expression.
    pub expr: Expr,
    /// Maps Opening(id) → external polynomial identity.
    pub opening_bindings: Vec<OpeningBinding>,
    /// Maps Challenge(id) → how to obtain the challenge value.
    pub challenge_bindings: Vec<ChallengeBinding>,
}

pub struct OpeningBinding {
    pub var_id: u32,
    /// Opaque tag identifying the polynomial (interpreted by jolt-zkvm).
    pub polynomial_tag: u64,
    /// Opaque tag identifying the sumcheck this opening belongs to.
    pub sumcheck_tag: u64,
}

pub struct ChallengeBinding {
    pub var_id: u32,
    pub source: ChallengeSource,
}

pub enum ChallengeSource {
    /// A batching coefficient (gamma power, RLC scalar, etc.)
    BatchingCoefficient(usize),
    /// A sumcheck challenge from a specific round.
    SumcheckChallenge(usize),
    /// A runtime-computed value passed by the caller.
    Derived,
}

// ── Visitor trait (backend extension point) ─────────────────

/// Walk an expression tree, producing a result per node.
///
/// Backends implement this to consume expressions:
/// - `EvaluateVisitor<F>`: concrete field evaluation
/// - `SopVisitor`: sum-of-products normalization
/// - `LeanVisitor`: Lean4 code emission
/// - `CircuitVisitor`: circuit constraint emission
pub trait ExprVisitor {
    type Output;
    fn visit_constant(&mut self, val: i128) -> Self::Output;
    fn visit_var(&mut self, var: Var) -> Self::Output;
    fn visit_neg(&mut self, inner: Self::Output) -> Self::Output;
    fn visit_add(&mut self, lhs: Self::Output, rhs: Self::Output) -> Self::Output;
    fn visit_sub(&mut self, lhs: Self::Output, rhs: Self::Output) -> Self::Output;
    fn visit_mul(&mut self, lhs: Self::Output, rhs: Self::Output) -> Self::Output;
}

impl Expr {
    /// Apply a visitor to the expression DAG (bottom-up).
    pub fn visit<V: ExprVisitor>(&self, visitor: &mut V) -> V::Output;
}

// ── Normalization ───────────────────────────────────────────

/// Sum-of-products normal form.
///
/// Every expression can be mechanically expanded to:
/// $\sum_i c_i \cdot \prod_j f_{ij}$
///
/// This form maps directly to R1CS multiplication gates and
/// replaces the hand-written `OutputClaimConstraint`.
pub struct SumOfProducts {
    pub terms: Vec<SopTerm>,
}

pub struct SopTerm {
    pub coefficient: SopValue,
    pub factors: Vec<SopValue>,
}

pub enum SopValue {
    Constant(i128),
    Opening(u32),
    Challenge(u32),
}

impl Expr {
    /// Expand into sum-of-products form via distribution.
    pub fn to_sum_of_products(&self) -> SumOfProducts;
    /// Fold constant sub-expressions.
    pub fn fold_constants(&self) -> Expr;
    /// Deduplicate structurally identical subtrees.
    pub fn eliminate_common_subexpressions(&self) -> Expr;
}
```

#### Backends

**1. Evaluate** (`backends/evaluate.rs`)

Evaluates an `Expr` to a concrete `F` value given opening and challenge slices. This replaces hand-written `input_claim()` methods on `SumcheckInstanceParams`.

```rust
impl Expr {
    pub fn evaluate<F: Field>(&self, openings: &[F], challenges: &[F]) -> F;
}
```

**2. R1CS** (`backends/r1cs.rs`)

Converts a `SumOfProducts` to R1CS constraints. This backend serves two consumers:

1. **BlindFold verifier R1CS** — replaces the `R1csConstraintVisitor` in `jolt-core/src/subprotocols/blindfold/r1cs.rs`. Emits sparse A/B/C matrices consumed by `jolt-spartan`.
2. **Main zkVM uniform R1CS** — replaces the compile-time `LC` / `lc!` / `r1cs_eq_conditional!` constraint system in `jolt-core/src/zkvm/r1cs/`. See "Replacing compile-time R1CS constraints" below.

```rust
impl SumOfProducts {
    pub fn to_r1cs_constraints(&self, ...) -> Vec<R1csConstraint>;
    pub fn estimate_aux_var_count(&self) -> usize;
}
```

For degree-2 expressions (which all current uniform constraints are), the backend can also extract the bilinear factorization `(LC_a, LC_b)` directly, enabling the fused dot-product evaluation path used by `UniformSpartanKey::evaluate_inner_sum_product_at_point`.

**3. Lean4** (`backends/lean.rs`)

Emits Lean4 syntax for an `Expr`. Replaces the `MleAst` → string formatting pipeline in `zklean-extractor`. Instance-local (no global arena).

```rust
impl Expr {
    pub fn to_lean4(&self, config: &LeanConfig) -> String;
}
```

**4. Circuit** (`backends/circuit.rs`)

Emits circuit constraints (gnark Go code, or a generic circuit IR) from an `Expr`. Replaces the `MemoizedCodeGen` pipeline in PR [#1322](https://github.com/a16z/jolt/pull/1322)'s transpiler for the claim-formula portion.

```rust
impl Expr {
    pub fn to_circuit<E: CircuitEmitter>(&self, emitter: &mut E);
}
```

#### How `SumcheckInstanceParams` Changes

In `jolt-zkvm`, the trait that defines sumcheck parameters currently requires dual implementations:

```rust
// BEFORE (jolt-core, current):
pub trait SumcheckInstanceParams<F: JoltField> {
    fn input_claim(&self, acc: &dyn OpeningAccumulator<F>) -> F;         // imperative
    fn input_claim_constraint(&self) -> InputClaimConstraint;            // symbolic (BlindFold)
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint>;  // symbolic (BlindFold)
    // ... challenge value methods ...
}
```

With `jolt-ir`, this becomes:

```rust
// AFTER (jolt-zkvm, with jolt-ir):
pub trait SumcheckInstanceParams<F: Field> {
    fn input_claim_definition(&self) -> ClaimDefinition;
    fn output_claim_definition(&self) -> Option<ClaimDefinition>;

    // Derived — not hand-written:
    fn input_claim(&self, openings: &[F], challenges: &[F]) -> F {
        self.input_claim_definition().expr.evaluate(openings, challenges)
    }
}
```

The sync hazard is eliminated structurally: there is exactly one definition per claim, and all backends derive from it.

#### Replacing compile-time R1CS constraints

Beyond sumcheck claim formulas, `jolt-ir` also replaces the compile-time constraint authoring system used by the main zkVM's Spartan outer sumcheck. Today, this system consists of:

- **`LC` enum** (`jolt-core/src/zkvm/r1cs/ops.rs`) — 12 variants (`Zero`, `Const`, `Terms1`..`Terms5`, `Terms1Const`..`Terms5Const`) to be `const fn`-compatible. Every operation (`dot_product`, `mul_by_const`, `accumulate_evaluations`) is a 12-arm match.
- **`lc!` macro** — builds `LC` values at compile time from `JoltR1CSInputs`.
- **`r1cs_eq_conditional!` macro** — builds equality-conditional constraints: `condition * (left - right) = 0`.
- **`R1CS_CONSTRAINTS` static array** — 19 named constraints, each an `(LC, LC)` pair.
- **`JoltR1CSInputs` enum** — 37 input variables with a hand-maintained canonical ordering.

With `jolt-ir`, each constraint is authored with `ExprBuilder`:

```rust
// BEFORE (macro DSL, ~4 lines but hard to read):
r1cs_eq_conditional!(
    label: R1CSConstraintLabel::RamAddrEqRs1PlusImmIfLoadStore,
    if { { JoltR1CSInputs::OpFlags(CircuitFlags::Load) }
       + { JoltR1CSInputs::OpFlags(CircuitFlags::Store) } }
    => ( { JoltR1CSInputs::RamAddress } )
       == ( { JoltR1CSInputs::Rs1Value } + { JoltR1CSInputs::Imm } )
)

// AFTER (jolt-ir, natural arithmetic):
fn ram_addr_if_load_store(v: &VarMap) -> ClaimDefinition {
    let b = ExprBuilder::new();
    let load  = b.opening(v.load);
    let store = b.opening(v.store);
    let addr  = b.opening(v.ram_addr);
    let rs1   = b.opening(v.rs1);
    let imm   = b.opening(v.imm);
    let expr = b.build((load + store) * (addr - rs1 - imm));
    ClaimDefinition { expr, opening_bindings: /* ... */, challenge_bindings: vec![] }
}
```

**Evaluation performance is preserved.** Every current constraint is degree 2 (root is `Mul(linear, linear)`). At init time, the `Expr` is factored back into `(Vec<(var_id, coeff)>, Vec<(var_id, coeff)>)` — the bilinear form. The `UniformSpartanKey::evaluate_inner_sum_product_at_point` method uses these factored pairs with the same fused dot-product logic. The 12-variant `LC` enum is replaced by a simpler `SmallVec`-based representation since `const fn` compatibility is no longer required.

**What jolt-ir replaces vs. what stays:**

| Current | jolt-ir replacement |
|---------|--------------------|
| `LC` enum (12 variants) | `Expr` → factored at init to bilinear pairs |
| `lc!` macro | `ExprBuilder` arithmetic operators |
| `r1cs_eq_conditional!` macro | `ExprBuilder::build(condition * (left - right))` |
| `R1CS_CONSTRAINTS` static array | `Vec<ClaimDefinition>` built at init |
| `JoltR1CSInputs` enum (as indices) | `OpeningBinding` metadata on `ClaimDefinition` |
| `OutputClaimConstraint` / `ProductTerm` | `SumOfProducts` (mechanically derived) |
| `ValueSource` enum | `SopValue` enum (isomorphic) |

**What is NOT replaced:**
- `R1CSCycleInputs` / witness materialization from trace (runtime data, not constraints)
- `evaluate_inner_sum_product_at_point` evaluation logic (stays, consumes jolt-ir-derived data)
- Constraint grouping metadata (first/second group labels for univariate skip)
- `UniformSpartanKey` struct (sources constraints differently, same evaluation strategy)
- `ProductConstraint` definitions (trivially expressed as `left * right - output` in jolt-ir)

#### jolt-spartan's role

`jolt-spartan` is a **generic Spartan SNARK over arbitrary R1CS**. It does NOT implement the main zkVM's outer sumcheck, which is too specialized (lazy evaluation via `UniformSpartanKey`, univariate skip, streaming rounds, two constraint groups).

`jolt-spartan` serves:
1. **BlindFold** — proves the verifier R1CS (produced by jolt-ir's R1CS backend) after Nova folding. Requires relaxed R1CS support (`Az ∘ Bz = u·Cz + E`).
2. **Recursive verification** — expressing the Jolt verifier as R1CS via jolt-ir and proving it with standard Spartan.
3. **Any future "prove R1CS via Spartan" use case** — the crate is genuinely reusable.

The main zkVM outer sumcheck stays as custom code in `jolt-zkvm`, using `jolt-sumcheck` for the protocol and `jolt-ir` for the constraint definitions, but with its own specialized evaluation path.

The dependency flow:
```
jolt-ir (constraint definitions, R1CS backend)
    ↓
jolt-spartan (generic Spartan prover/verifier)
    ↓
jolt-blindfold (ZK orchestration: folding + committed Spartan)
```

The main zkVM bypasses `jolt-spartan` but uses `jolt-ir` and `jolt-sumcheck`.

#### Concrete example: Booleanity

The booleanity sumcheck (`jolt-core/src/subprotocols/booleanity.rs`) proves that all RA polynomials are Boolean: $0 = \sum_{k,j} eq(r,k) \cdot eq(r,j) \cdot \sum_i \gamma^i \cdot (H_i^2 - H_i)$.

**Before** (4 representations):
```rust
// (1) Runtime value — hand-written
fn input_claim(&self, _acc: &dyn OpeningAccumulator<F>) -> F { F::zero() }

// (2) BlindFold constraint — hand-written, must match (1)
fn input_claim_constraint(&self) -> InputClaimConstraint { InputClaimConstraint::default() }

// (2b) BlindFold output — hand-written, 30 lines of ProductTerm construction
fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
    let mut terms = Vec::with_capacity(2 * n);
    for (i, poly) in self.polynomial_types.iter().enumerate() {
        let opening = OpeningId::committed(*poly, SumcheckId::Booleanity);
        terms.push(ProductTerm::scaled(ValueSource::Challenge(2*i),
            vec![ValueSource::Opening(opening), ValueSource::Opening(opening)]));
        terms.push(ProductTerm::scaled(ValueSource::Challenge(2*i + 1),
            vec![ValueSource::Opening(opening)]));
    }
    Some(OutputClaimConstraint::sum_of_products(terms))
}
```

**After** (1 definition):
```rust
fn input_claim_definition(&self) -> ClaimDefinition {
    // Input is zero — the booleanity relation sums to 0
    let b = ExprBuilder::new();
    ClaimDefinition { expr: b.build(b.zero()), ..default() }
}

fn output_claim_definition(&self) -> Option<ClaimDefinition> {
    let mut b = ExprBuilder::new();
    let mut sum = b.zero();
    for i in 0..self.polynomial_types.len() {
        let h = b.opening(i as u32);
        let gamma_sq  = b.challenge(2 * i as u32);      // eq_eval · γ²ⁱ
        let neg_gamma = b.challenge(2 * i as u32 + 1);   // -eq_eval · γ²ⁱ
        sum = sum + gamma_sq * h * h + neg_gamma * h;
    }
    Some(ClaimDefinition {
        expr: b.build(sum),
        opening_bindings: /* one per polynomial */,
        challenge_bindings: /* one per challenge */,
    })
}
```

This single definition feeds all backends:
- Standard verifier: `def.expr.evaluate(&opening_values, &challenge_values)` → `F`
- BlindFold: `def.expr.to_sum_of_products().to_r1cs_constraints(...)` → R1CS gates
- Lean4: `def.expr.to_lean4(&config)` → Lean4 code
- Circuit: `def.expr.to_circuit(&mut gnark_emitter)` → Go code

#### Kernel IR (optional, for GPU compilation)

Separately from claim-level expressions, `jolt-ir` may also host a **kernel IR** for describing how sumcheck round polynomials are computed from evaluation tables. This covers the ~70% of sumchecks that follow the pattern:

$$\text{round\_poly}(t) = \sum_{x} f(\text{table}_0[2x+t],\; \text{table}_1[2x+t],\; \ldots,\; \text{eq}[2x+t])$$

The kernel IR reuses the same `Expr` type but with different variable semantics (table positions instead of scalar openings):

```rust
pub struct SumcheckKernel {
    /// Per-point combination expression (variables are table lookups).
    pub combination: Expr,
    /// Metadata about each polynomial table.
    pub tables: Vec<TableDescriptor>,
    /// Maximum degree of the round polynomial.
    pub degree: usize,
}

pub struct TableDescriptor {
    pub var_id: u32,
    pub role: TableRole,
}

pub enum TableRole {
    /// Full evaluation table, bound in-place each round.
    Dense,
    /// Eq polynomial (supports split-eq / sqrt-space optimization).
    Eq,
    /// Compact table (small scalar type, promoted on bind).
    Compact { scalar_bytes: usize },
    /// Precomputed table (invariant across rounds).
    Static,
}
```

A CPU executor interprets the kernel in a map-reduce loop. A GPU executor compiles it into a Metal/CUDA kernel. Hand-tuned kernels (for the ~30% of sumchecks with non-standard patterns like `SharedRaPolynomials` or multi-phase binding) implement `SumcheckWitness` directly, bypassing the kernel IR.

The kernel IR is optional and may be deferred to a later phase. The claim IR is the priority.

#### Testing

- **Unit tests:** Expression construction, operator overloading, arena integrity
- **Property tests:** For random expressions, `evaluate()` matches manual computation. `to_sum_of_products()` evaluates to the same value as the original expression.
- **Round-trip tests:** `Expr` → `SumOfProducts` → `evaluate()` ≡ `Expr` → `evaluate()`
- **Backend consistency tests:** For each sumcheck claim definition, verify that `evaluate()`, `to_sum_of_products().evaluate()`, and `to_lean4()` (parsed and evaluated) all agree.
- **CSE tests:** Verify that `eliminate_common_subexpressions()` produces a smaller DAG but identical evaluation.

#### Delivery phases

The crate is built incrementally (see `crates/tasks/`):

1. **Core IR** (task 01) — expr, builder, claim, visitor, normalize. No backends, no `jolt-field` dep.
2. **Evaluate backend** (task 02) — `Expr::evaluate<F>()`. Adds `jolt-field` dep. Enables standard-mode verifier.
3. **R1CS backend** (task 03) — `SumOfProducts::emit_r1cs()`. Enables BlindFold ZK mode.
4. **Lean4 backend** (task 04) — `Expr::to_lean4()`. Enables zklean-extractor migration.
5. **Circuit backend** (task 05) — `CircuitEmitter` trait. Enables gnark transpiler migration.
6. **Downstream integration** (task 06) — migrate `SumcheckInstanceParams` in jolt-zkvm.

The kernel IR (`SumcheckKernel`, `TableDescriptor`) is deferred until GPU integration begins.

#### File Structure

```
jolt-ir/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Re-exports: Expr, ExprBuilder, ClaimDefinition, ExprVisitor
│   ├── expr.rs             # ExprNode, ExprId, ExprArena, Expr
│   ├── builder.rs          # ExprBuilder, ExprHandle, operator impls
│   ├── claim.rs            # ClaimDefinition, OpeningBinding, ChallengeBinding
│   ├── visitor.rs          # ExprVisitor trait, Expr::visit()
│   ├── normalize.rs        # SumOfProducts, to_sum_of_products(), fold_constants(), CSE
│   └── backends/
│       ├── mod.rs
│       ├── evaluate.rs     # EvaluateVisitor — Expr → F
│       ├── r1cs.rs         # SoP → R1CS constraints
│       ├── lean.rs         # Expr → Lean4 string
│       └── circuit.rs      # Expr → circuit IR
├── tests/
│   ├── expr_eval.rs        # Expression evaluation tests
│   ├── sop_consistency.rs  # SoP matches direct evaluation
│   └── backend_agree.rs    # All backends agree on same input
└── benches/
    └── normalize.rs        # SoP expansion benchmarks
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
| `jolt-ir` | Expr construction, operator overloads, arena | SoP consistency, backend agreement | — |
| `jolt-instructions` | Each instruction's `execute` | Instruction set coverage, lookup consistency | Instruction decoding |
| `jolt-dory` | Basic commit/open | Full protocol flows, streaming commitment | Commitment verification |
| `jolt-zkvm` | Subprotocol logic | Small program proving, claim reductions | Trace verification |

### 5.3 Property-Based Testing

Use `proptest` for generating random inputs and checking invariants:

| Crate | Properties |
|-------|-----------|
| `jolt-poly` | `evaluate(bind_all(r)) == evaluate(r)`, Schwartz-Zippel |
| `jolt-ir` | `Expr::evaluate() == Expr::to_sum_of_products().evaluate()` for random expressions |
| `jolt-sumcheck` | Completeness (honest prover always passes), soundness (cheating fails w.h.p.) |
| `jolt-spartan` | Satisfiable witness → valid proof, unsatisfiable → rejection |
| `jolt-instructions` | `execute` matches wrapping native ops, lookup decomposition reconstructs correctly |

### 5.4 Fuzzing

Use `cargo-fuzz` with `libFuzzer` for security-critical code:

| Crate | Fuzz Targets |
|-------|-------------|
| `jolt-field` | Deserialization of arbitrary bytes |
| `jolt-poly` | `Polynomial::new` from arbitrary evaluations + evaluate at arbitrary points |
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
  implement-jolt-ir         (needs jolt-field only)
  implement-jolt-instructions     (needs jolt-field only)

Phase 1b (testing, after 1a):
  test-jolt-poly-integration
  test-jolt-ir-integration
  test-jolt-instructions-integration
  test-fuzz-jolt-poly

Phase 2a (implementation, after jolt-poly + jolt-ir):
  implement-jolt-sumcheck         (needs jolt-poly, jolt-ir)
  implement-jolt-openings         (needs jolt-poly)

Phase 2b (testing, after 2a):
  test-jolt-sumcheck-integration
  test-jolt-openings-integration
  test-fuzz-jolt-sumcheck
  test-fuzz-jolt-openings

Phase 3a (implementation, after 2a):
  implement-jolt-spartan          (needs jolt-sumcheck + jolt-openings + jolt-ir)
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
ir/                 : implement-jolt-ir           : none                     : Symbolic expression IR for sumcheck claims
src/zkvm/instruction: implement-jolt-instructions       : none                     : RISC-V instructions + lookup tables
src/subprotocols/   : implement-jolt-sumcheck           : jolt-poly,jolt-ir  : Sumcheck protocol engine
src/poly/commitment : implement-jolt-openings           : jolt-poly                : Commitment scheme traits + accumulators
src/zkvm/spartan    : implement-jolt-spartan            : jolt-sumcheck,jolt-openings,jolt-ir : Spartan R1CS prover/verifier
src/poly/dory       : implement-jolt-dory               : jolt-openings            : Dory commitment scheme
src/zkvm/           : implement-jolt-zkvm               : jolt-spartan,jolt-sumcheck,jolt-openings,jolt-instructions,jolt-dory,jolt-ir : zkVM prover/verifier
integration         : implement-integrate-workspace     : all                      : Wire crates together in workspace

# Test tasks (tester role)
tests               : test-jolt-poly-integration        : jolt-poly                : Integration tests for jolt-poly
tests               : test-jolt-ir-integration    : jolt-ir            : Integration tests for jolt-ir
tests               : test-jolt-instructions-integration : jolt-instructions        : Integration tests for jolt-instructions
tests               : test-jolt-sumcheck-integration    : jolt-sumcheck            : Integration tests for jolt-sumcheck
tests               : test-jolt-openings-integration    : jolt-openings            : Integration tests for jolt-openings
tests               : test-jolt-spartan-integration     : jolt-spartan             : Integration tests for jolt-spartan
tests               : test-jolt-dory-integration        : jolt-dory                : Integration tests for jolt-dory
tests               : test-jolt-zkvm-integration        : jolt-zkvm                : Integration tests for jolt-zkvm

# Fuzz test tasks (tester role)
fuzz                : test-fuzz-jolt-poly               : jolt-poly                : Fuzz testing for jolt-poly
fuzz                : test-fuzz-jolt-ir           : jolt-ir            : Fuzz testing for jolt-ir
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
├── jolt-ir/
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

6. **Kernel IR scope in jolt-ir** — the kernel IR (`SumcheckKernel`, `TableDescriptor`) for GPU compilation is optional. It reuses the same `Expr` type as claim definitions but with different variable semantics (table positions vs. scalar openings). Decision: include the types in `jolt-ir` but defer backend implementations (CPU executor, GPU compiler) until the GPU integration work begins.

7. **TracingField placement** — the `MleAst`-style symbolic field type (for capturing full verifier traces) could live in `jolt-ir` or a separate `jolt-ir-trace` crate. Tentatively a separate crate to keep `jolt-ir` dependency-minimal (just `jolt-field`). The tracing crate would depend on both `jolt-field` and `jolt-ir`.

8. **OpeningBinding / ChallengeBinding opacity** — `jolt-ir` uses opaque `u64` tags for polynomial and sumcheck identifiers. These are interpreted by `jolt-zkvm` which maps them to concrete `CommittedPolynomial`/`VirtualPolynomial`/`SumcheckId` enums. The boundary should be clean: `jolt-ir` never imports from `jolt-zkvm`.
