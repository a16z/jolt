# impl-jolt-spartan: Clean-room implementation of jolt-spartan

**Scope:** crates/jolt-spartan/

**Depends:** impl-jolt-sumcheck, impl-jolt-openings

**Verifier:** ./verifiers/scoped.sh /workdir jolt-spartan

**Context:**

Implement the `jolt-spartan` crate — a Spartan-based SNARK for R1CS constraint systems. Generic over the commitment scheme and field. Usable for any R1CS system, not just Jolt.

**This is a clean-room rewrite.** Study `jolt-core/src/zkvm/spartan/` and `jolt-core/src/zkvm/r1cs/` for algorithmic reference. Design the API from scratch.

**Dependencies:** `jolt-sumcheck`, `jolt-openings`, `jolt-field`, `jolt-poly`, `jolt-transcript`.

### Reference material

The old code lives in:
- `jolt-core/src/zkvm/spartan/mod.rs` (49 LOC) — public API
- `jolt-core/src/zkvm/spartan/outer.rs` (58.5 KB) — outer sumcheck with univariate skip
- `jolt-core/src/zkvm/spartan/product.rs` (31.3 KB) — product subprotocol
- `jolt-core/src/zkvm/spartan/shift.rs` (33.3 KB) — shift constraints
- `jolt-core/src/zkvm/spartan/instruction_input.rs` (25 KB) — instruction input handling
- `jolt-core/src/zkvm/r1cs/constraints.rs` (25.8 KB) — R1CS constraints
- `jolt-core/src/zkvm/r1cs/evaluation.rs` (49.2 KB) — constraint evaluation
- `jolt-core/src/zkvm/r1cs/inputs.rs` (29.7 KB) — constraint inputs
- `jolt-core/src/zkvm/r1cs/key.rs` (6.2 KB) — UniformSpartanKey
- `jolt-core/src/zkvm/r1cs/ops.rs` (26.6 KB) — constraint operations

Also read the Jolt Book architecture overview and Spartan section: https://jolt.a16zcrypto.com/

### Public API contract

```rust
/// A Rank-1 Constraint System: matrices A, B, C such that
/// Az ∘ Bz = Cz for a valid witness z.
pub trait R1CS<F: Field> {
    fn num_constraints(&self) -> usize;
    fn num_variables(&self) -> usize;
    fn multiply_witness(&self, witness: &[F]) -> (Vec<F>, Vec<F>, Vec<F>);
}

/// Uniform (structured) R1CS with repeating constraint patterns.
pub struct UniformR1CS<F: Field> { ... }

/// Key material derived from the R1CS structure.
pub struct SpartanKey<F: Field> { ... }

impl<F: Field> SpartanKey<F> {
    pub fn from_r1cs(r1cs: &impl R1CS<F>) -> Self;
}

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

pub struct SpartanProof<F: Field, PCS: CommitmentScheme> { ... }

/// Strategy for the first sumcheck round.
pub enum FirstRoundStrategy {
    Standard,
    UnivariateSkip { domain_size: usize },
}

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

### Implementation notes

Spartan reduces R1CS satisfaction to sumcheck:

1. **Outer sumcheck:** Verify that $\sum_x \widetilde{A}(x) \cdot \widetilde{B}(x) - \widetilde{C}(x) = 0$ where $\widetilde{A}, \widetilde{B}, \widetilde{C}$ are multilinear extensions of the constraint matrix-vector products.
2. **Inner sumcheck:** Reduce the multilinear extension evaluations to polynomial openings.
3. **Opening proofs:** Via the `HomomorphicCommitmentScheme` from `jolt-openings`.

The `SpartanProver` implements `SumcheckInstanceProver` from `jolt-sumcheck` for the outer and inner sumcheck rounds. This is the key integration point.

The univariate skip optimization (from `outer.rs`) precomputes first-round evaluations to skip a full polynomial evaluation. Implement as an optional `FirstRoundStrategy`.

The "dark arts" in `outer.rs` are largely due to tight coupling with Jolt-specific state. By making Spartan generic over `R1CS`, the outer sumcheck becomes cleaner.

### File structure

```
jolt-spartan/src/
├── lib.rs
├── r1cs.rs             # R1CS trait, UniformR1CS
├── key.rs              # SpartanKey
├── prover.rs           # SpartanProver
├── verifier.rs         # SpartanVerifier
├── proof.rs            # SpartanProof
├── uni_skip.rs         # Univariate skip optimization
└── error.rs            # SpartanError
```

**Acceptance:**

- R1CS trait is generic and usable for arbitrary constraint systems
- SpartanProver produces valid proofs for satisfiable R1CS instances
- SpartanVerifier rejects proofs for unsatisfiable witnesses
- Univariate skip optimization works as an optional first-round strategy
- Error types compose correctly (SumcheckError, OpeningsError flow through)
- Works with MockCommitmentScheme from `jolt-openings` for testing
- `parallel` feature enables rayon in sumcheck rounds
- No file exceeds 500 lines
- Rustdoc on all public items with LaTeX math
- `cargo clippy` clean
- Unit tests: small R1CS (e.g., x^2 = y), prove → verify round-trip
