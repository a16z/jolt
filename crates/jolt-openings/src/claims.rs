//! Stateless claim types for polynomial commitment scheme operations.
//!
//! Claims are plain data collected by the protocol orchestrator (e.g., `jolt-zkvm`)
//! during multi-round interactive proofs. They are then passed to a reduction
//! strategy ([`OpeningReduction`](crate::OpeningReduction)) and finally to the
//! PCS for opening/verification.

use jolt_field::Field;

/// A prover-side opening claim: polynomial evaluations, point, and claimed value.
///
/// The prover holds the full evaluation table of the polynomial, the evaluation
/// point, and the claimed evaluation. After reduction, the prover opens the
/// reduced claims via [`CommitmentScheme::open`](crate::CommitmentScheme::open).
#[derive(Clone, Debug)]
pub struct ProverClaim<F: Field> {
    /// Full evaluation table of the polynomial over the Boolean hypercube.
    pub evaluations: Vec<F>,
    /// Evaluation point $r \in \mathbb{F}^n$.
    pub point: Vec<F>,
    /// Claimed evaluation $f(r)$.
    pub eval: F,
}

/// A verifier-side opening claim: commitment, point, and claimed value.
///
/// Fully typed — no `dyn Any` type erasure. The commitment type `C` is
/// typically `PCS::Output` from a [`CommitmentScheme`](crate::CommitmentScheme).
#[derive(Clone, Debug)]
pub struct VerifierClaim<F: Field, C> {
    /// Binding commitment to the polynomial.
    pub commitment: C,
    /// Evaluation point $r \in \mathbb{F}^n$.
    pub point: Vec<F>,
    /// Claimed evaluation $f(r)$.
    pub eval: F,
}
