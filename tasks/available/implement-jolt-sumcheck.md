# impl-jolt-sumcheck: Clean-room implementation of jolt-sumcheck

**Scope:** crates/jolt-sumcheck/

**Depends:** impl-jolt-poly

**Verifier:** ./verifiers/scoped.sh /workdir jolt-sumcheck

**Context:**

Implement the `jolt-sumcheck` crate — a generic sumcheck protocol engine including batched and streaming variants. This crate is reusable for any sumcheck application, not just Jolt.

**This is a clean-room rewrite.** Study `jolt-core/src/subprotocols/sumcheck.rs`, `sumcheck_prover.rs`, `sumcheck_verifier.rs`, `sumcheck_claim.rs`, `streaming_sumcheck.rs`, and `streaming_schedule.rs` for algorithmic reference. Design the API from scratch.

**Dependencies:** `jolt-field`, `jolt-poly`, `jolt-transcript`.

### Reference material

The old code lives in:
- `jolt-core/src/subprotocols/sumcheck.rs` (12.8 KB) — `BatchedSumcheck` with prove/verify
- `jolt-core/src/subprotocols/sumcheck_prover.rs` — `SumcheckInstanceProver` trait
- `jolt-core/src/subprotocols/sumcheck_verifier.rs` — `SumcheckInstanceVerifier` trait
- `jolt-core/src/subprotocols/sumcheck_claim.rs` (8.6 KB) — `SumcheckClaim<F>`
- `jolt-core/src/subprotocols/streaming_sumcheck.rs` — streaming variant
- `jolt-core/src/subprotocols/streaming_schedule.rs` — scheduling for streaming

Also read the Jolt Book appendix on sumcheck: https://jolt.a16zcrypto.com/
And the Formal Algorithms for Transformers paper for mathematical background on sumcheck if needed: see `docs/papers/processed/`.

### Public API contract

```rust
/// A sumcheck claim: the prover asserts that
/// sum_{x in {0,1}^n} g(x) = claimed_sum
/// where g is implicitly defined by the witness.
pub struct SumcheckClaim<F: Field> {
    pub num_vars: usize,
    pub degree: usize,
    pub claimed_sum: F,
}

/// Prover-side interface for a single sumcheck instance.
///
/// The sumcheck engine calls `round_polynomial` repeatedly, once per variable.
/// After each round, the engine calls `bind` with the verifier's challenge.
pub trait SumcheckInstanceProver<F: Field>: Send + Sync {
    /// Produce the round polynomial for the current round.
    /// Returns a univariate poly of degree <= claim.degree.
    fn round_polynomial(&self) -> UnivariatePoly<F>;

    /// Bind the current variable to `challenge`, reducing by one variable.
    fn bind(&mut self, challenge: F);
}

/// Verifier-side interface for checking a sumcheck round.
pub trait SumcheckInstanceVerifier<F: Field> {
    fn verify_round(
        round_poly: &UnivariatePoly<F>,
        expected_sum: F,
        challenge: F,
    ) -> Result<F, SumcheckError>;

    fn verify_final(
        claimed_eval: F,
        expected: F,
    ) -> Result<(), SumcheckError>;
}

/// Non-interactive sumcheck prover (Fiat-Shamir).
pub struct SumcheckProver;

impl SumcheckProver {
    /// Prove a single sumcheck instance.
    pub fn prove<F: Field>(
        claim: &SumcheckClaim<F>,
        witness: &mut impl SumcheckInstanceProver<F>,
        transcript: &mut impl Transcript,
    ) -> SumcheckProof<F>;

    /// Prove batched sumcheck for multiple instances simultaneously.
    /// Uses front-loaded batching (Posen): instances with fewer variables
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
    /// Verify a single sumcheck proof.
    /// Returns (final_eval, challenge_vector).
    pub fn verify<F: Field>(
        claim: &SumcheckClaim<F>,
        proof: &SumcheckProof<F>,
        transcript: &mut impl Transcript,
    ) -> Result<(F, Vec<F>), SumcheckError>;

    /// Verify a batched sumcheck proof.
    pub fn verify_batched<F: Field>(
        claims: &[SumcheckClaim<F>],
        proof: &BatchedSumcheckProof<F>,
        transcript: &mut impl Transcript,
    ) -> Result<(Vec<F>, Vec<Vec<F>>), SumcheckError>;
}

/// Proof transcript for a single sumcheck instance.
#[derive(Clone, Serialize, Deserialize)]
pub struct SumcheckProof<F: Field> {
    pub round_polynomials: Vec<UnivariatePoly<F>>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct BatchedSumcheckProof<F: Field> {
    pub round_polynomials: Vec<Vec<UnivariatePoly<F>>>,
}

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

### Implementation notes — Core protocol

The sumcheck protocol for $\sum_{x \in \{0,1\}^n} g(x) = C$:

1. **Round 1:** Prover sends $s_1(X_1) = \sum_{x_2,\ldots,x_n \in \{0,1\}} g(X_1, x_2, \ldots, x_n)$. Verifier checks $s_1(0) + s_1(1) = C$, sends random $r_1$.
2. **Round $i$:** Prover sends $s_i(X_i)$. Verifier checks $s_i(0) + s_i(1) = s_{i-1}(r_{i-1})$, sends random $r_i$.
3. **Final:** Verifier checks claimed evaluation $g(r_1, \ldots, r_n)$.

The `SumcheckInstanceProver` trait decouples the protocol engine from the specific polynomial/witness representation. Jolt's zkVM will implement this trait for its various sub-protocols (RAM, registers, etc.).

### Implementation notes — Batching

Posen's front-loaded batching: given $k$ instances with possibly different `num_vars`, pad shorter instances to match the longest by multiplying their claimed sums by $2^{\Delta}$ where $\Delta$ is the difference in variables. All instances then run for the same number of rounds.

### Implementation notes — Streaming

The streaming variant processes evaluations in chunks rather than materializing the entire polynomial. This is critical for memory efficiency. Include a `StreamingSumcheckProver` trait:

```rust
pub trait StreamingSumcheckProver<F: Field>: Send + Sync {
    fn begin_round(&mut self);
    fn process_chunk(&mut self, chunk: &[F]);
    fn finish_round(&mut self) -> UnivariatePoly<F>;
    fn bind(&mut self, challenge: F);
}
```

### File structure

```
jolt-sumcheck/src/
├── lib.rs
├── claim.rs            # SumcheckClaim
├── prover.rs           # SumcheckProver, SumcheckInstanceProver trait
├── verifier.rs         # SumcheckVerifier, SumcheckInstanceVerifier trait
├── proof.rs            # SumcheckProof, BatchedSumcheckProof
├── batched.rs          # Batching logic (front-loaded, power-of-2 scaling)
├── streaming.rs        # StreamingSumcheckProver trait + engine
└── error.rs            # SumcheckError
```

**Acceptance:**

- Single-instance sumcheck: prove → verify round-trip succeeds for known polynomial sums
- Batched sumcheck: multiple instances with different `num_vars` prove/verify correctly
- Streaming sumcheck: trait defined, basic implementation works for simple cases
- Verifier rejects proofs with modified round polynomials
- Verifier rejects proofs with wrong claimed sum
- Degree bound checking works
- All proof types are `Serialize + Deserialize`
- `parallel` feature enables rayon in round polynomial computation
- No file exceeds 500 lines
- Rustdoc on all public items with LaTeX math for the protocol
- `cargo clippy` clean
- Unit tests inline in each source file
