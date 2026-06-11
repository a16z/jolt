//! Sumcheck protocol: claims, proofs, and verification.
//!
//! Verifier-side types and logic for the sumcheck protocol.
//!
//! # Protocol overview
//!
//! The sumcheck protocol reduces the verification of a claim
//! $$\sum_{x \in \{0,1\}^n} g(x) = C$$
//! to a single evaluation query $g(r_1, \ldots, r_n) = v$ via $n$
//! rounds of interaction. In round $i$ the prover sends a univariate
//! polynomial $s_i(X)$ and the verifier checks $s_i(0) + s_i(1)$ against
//! the running sum, then sets $r_i$ and recurses.
//!
//! # Crate structure
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`claim`] | [`SumcheckClaim`] (input statement) and [`EvaluationClaim`] (reduction output) |
//! | [`proof`] | [`ClearProof`], [`ClearSumcheckProof`], [`CompressedSumcheckProof`], and [`SumcheckProof`] — serializable proofs |
//! | [`verifier`] | [`SumcheckVerifier`] engine |
//! | [`batched_verifier`] | [`BatchedSumcheckVerifier`] — batched verification via RLC |
//! | [`domain`] | [`SumcheckDomain`] implementations for round-sum checks |
//! | `r1cs` | R1CS lowering for sumcheck verifier equations (`r1cs` feature) |
//! | [`round_proof`] | [`RoundMessage`] and [`ClearRound`] traits |
//! | [`committed`] | Commitment-backed round messages |
//! | [`error`] | [`SumcheckError`] variants |
//!
//! # Public API
//!
//! ## Types
//! - [`SumcheckClaim<F>`] — the public statement: `num_vars`, `degree`, and `claimed_sum`.
//! - [`SumcheckStatement`] — round count and degree bound without a claimed sum.
//! - [`EvaluationClaim<F>`] — the oracle evaluation claim `g(r) = v` produced by a
//!   successful reduction; the caller MUST discharge it against the polynomial oracle.
//! - [`ClearProof<F>`] — clear proof wire representation, either full or compressed.
//! - [`ClearSumcheckProof<F>`] — a sequence of full univariate round polynomials, one per variable.
//! - [`CompressedSumcheckProof<F>`] — owned wire form omitting each linear coefficient.
//! - [`SumcheckProof<F, C>`] — clear or committed sumcheck proof data.
//! - [`CommittedSumcheckProof<C>`] — committed round messages and output-claim commitments.
//! - [`BooleanHypercube`] — the standard `{0,1}` sumcheck round domain.
//! - [`CenteredIntegerDomain`] — centered consecutive-integer sumcheck round domain.
//! - [`SumcheckError`] — error variants: `RoundCheckFailed`, `DegreeBoundExceeded`,
//!   `WrongNumberOfRounds`, `EmptyClaims`.
//!
//! ## Verifiers
//! - [`SumcheckVerifier`] — single-instance verifier. Replays the Fiat-Shamir
//!   transcript and checks each round.
//! - [`BatchedSumcheckVerifier`] — batched verification via random linear
//!   combination. Supports claims with different `num_vars` and `degree` bounds
//!   via front-loaded padding.
//!
//! ## Per-round proof types
//! - [`RoundDegree`] — field-agnostic degree bound.
//! - [`RoundMessage`] — transcript absorption (over a challenge field `F`).
//! - [`ClearRound<F>`] — clear round polynomial evaluation and well-formedness.
//! - [`UnivariatePoly<F>`](jolt_poly::UnivariatePoly) — raw round-coefficient absorb.
//! - [`LabeledRoundPoly`] — borrowed round-polynomial wrapper.
//! - [`CompressedLabeledRoundPoly`] — borrowed wrapper using the compressed
//!   wire format (omits the linear coefficient).
//!
//! # Dependency position
//!
//! ```text
//! jolt-field      ─┐
//! jolt-poly       ─┼─> jolt-sumcheck
//! jolt-transcript ─┤
//! jolt-crypto     ─┘
//! ```
//!

pub mod batched_verifier;
pub mod claim;
pub mod committed;
pub mod domain;
pub mod error;
pub mod proof;
#[cfg(feature = "r1cs")]
pub mod r1cs;
pub mod round_proof;
pub mod scalar;
pub mod verifier;

#[cfg(test)]
mod tests;

/// Absorbs a sumcheck claim scalar. Like jolt-core, no domain-separation label
/// is absorbed — claims are separated positionally and by the transcript's
/// one-time `DomainSeparator`/instance.
pub fn append_sumcheck_claim<F, T>(transcript: &mut T, claim: &F)
where
    F: jolt_field::Field,
    T: jolt_transcript::FsTranscript<F>,
{
    transcript.absorb_field(claim);
}

pub use batched_verifier::{
    BatchedCommittedSumcheckConsistency, BatchedEvaluationClaim, BatchedSumcheckVerifier,
};
pub use claim::{EvaluationClaim, SumcheckClaim, SumcheckStatement};
pub use committed::{
    CommittedOutputClaims, CommittedRound, CommittedRoundWitness, CommittedSumcheckConsistency,
    CommittedSumcheckProof, VerifiedCommittedRound,
};
pub use domain::{BooleanHypercube, CenteredIntegerDomain, SumcheckDomain, SumcheckDomainSpec};
pub use error::SumcheckError;
pub use proof::{ClearProof, ClearSumcheckProof, CompressedSumcheckProof, SumcheckProof};
#[cfg(feature = "r1cs")]
pub use r1cs::{
    allocate_sumcheck_r1cs_layout, append_sumcheck_r1cs_constraints,
    append_sumcheck_r1cs_constraints_for_domain, SumcheckR1csError, SumcheckR1csLayout,
    SumcheckR1csRound, SumcheckR1csRoundLayout,
};
pub use round_proof::{
    ClearRound, CompressedLabeledRoundPoly, LabeledRoundPoly, RoundDegree, RoundMessage,
};
pub use scalar::SumcheckScalar;
pub use verifier::SumcheckVerifier;
