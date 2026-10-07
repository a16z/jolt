//! Sumcheck protocol: claims, proofs, verification, and the prove-side
//! recording seam.
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
//! | [`batch`] | [`BatchPrelude`] — the batched head shared by verify and prove drivers |
//! | [`verifier`] | [`SumcheckVerifier`] engine, reading rounds from the proof transcript |
//! | [`prover`] | [`ProveRounds`], [`prove_batch`], and the uni-skip provers — the prove-side engine |
//! | [`prover`] | [`RoundScheduler`] / [`SequentialRounds`] — the per-round member-traversal seam |
//! | [`recorder`] | [`SumcheckRecorder`] — the clear/ZK proof-recording seam |
//! | [`domain`] | [`SumcheckDomain`] implementations for round-sum checks |
//! | `r1cs` | R1CS lowering for sumcheck verifier equations (`r1cs` feature) |
//! | [`round_proof`] | Round polynomials on the wire: fixed-width full and compressed forms |
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
//! - [`CommittedOutputClaims<C>`] — row commitments to a committed sumcheck's output claims.
//! - [`BooleanHypercube`] — the standard `{0,1}` sumcheck round domain.
//! - [`CenteredIntegerDomain`] — centered consecutive-integer sumcheck round domain.
//! - [`SumcheckError`] — error variants, including `RoundCheckFailed`,
//!   `DegreeBoundExceeded`, and `Transcript`.
//!
//! ## Proofs live in the transcript
//!
//! There is no sumcheck proof type. Provers write round polynomials (or their
//! commitments) into a [`ProverTranscript`](jolt_transcript::ProverTranscript)
//! through a [`SumcheckRecorder`]; [`SumcheckVerifier`] reads them back from a
//! [`VerifierTranscript`](jolt_transcript::VerifierTranscript). Every round is
//! sent at its public degree bound, so no length or label travels with it.
//!
//! # Dependency position
//!
//! ```text
//! jolt-field      ─┐
//! jolt-poly       ─┼─> jolt-sumcheck
//! jolt-transcript ─┘
//!
//! optional: jolt-crypto (`committed`), jolt-r1cs (`r1cs`)
//! ```
//!
//! Polynomial and clear sumcheck arithmetic is generic over
//! [`Field`](jolt_field::Field); transcript paths additionally require
//! [`CanonicalEncoding`](jolt_field::CanonicalEncoding) to send and draw field
//! elements. Optimized Jolt kernels and commitment backends retain their
//! stronger capability bounds at their own integration points.
//!

// In the jolt-verifier runtime closure: stricter panic and unsafe discipline
// than the workspace lints (specs/verifier-closure-lints.md).
#![forbid(unsafe_code)]
#![deny(
    clippy::indexing_slicing,
    clippy::get_unwrap,
    clippy::string_slice,
    clippy::fallible_impl_from,
    clippy::mem_forget,
    clippy::exit,
    clippy::panic_in_result_fn,
    clippy::let_underscore_must_use,
    clippy::host_endian_bytes,
    clippy::wildcard_enum_match_arm
)]

pub mod batch;
pub mod claim;
pub mod committed;
pub mod domain;
pub mod error;
pub mod prover;
#[cfg(feature = "r1cs")]
pub mod r1cs;
pub mod recorder;
pub mod round_proof;
pub mod verifier;

#[cfg(all(test, feature = "committed"))]
mod round_scheduler_tests;
#[cfg(all(test, feature = "committed"))]
mod tests;

pub use batch::{BatchMember, BatchPrelude};
pub use claim::{EvaluationClaim, SumcheckClaim, SumcheckStatement};
#[cfg(feature = "committed")]
pub use committed::CommittedSumcheckBuilder;
pub use committed::{
    BatchedCommittedSumcheckConsistency, CommittedOutputClaims, CommittedSumcheckConsistency,
    CommittedSumcheckWitness, VerifiedCommittedRound,
};
pub use domain::{BooleanHypercube, CenteredIntegerDomain, SumcheckDomain, SumcheckDomainSpec};
pub use error::SumcheckError;
pub use prover::{
    prove_batch, prove_uniskip_clear, MemberFinish, MemberRound, ProveRounds, ProvedBatch,
    ProvedUniskip, RoundScheduler, SequentialRounds,
};
#[cfg(feature = "committed")]
pub use prover::{prove_uniskip_committed, ProvedUniskipCommitted};
#[cfg(feature = "r1cs")]
pub use r1cs::{
    allocate_sumcheck_r1cs_layout, append_sumcheck_r1cs_constraints,
    append_sumcheck_r1cs_constraints_for_domain, SumcheckR1csError, SumcheckR1csLayout,
    SumcheckR1csRound, SumcheckR1csRoundLayout,
};
#[cfg(feature = "committed")]
pub use recorder::CommittedSumcheckRecorder;
pub use recorder::{ClearSumcheckRecorder, SumcheckRecorder};
pub use round_proof::{
    padded_coefficients, receive_compressed_round, receive_full_round, send_compressed_round,
    send_full_round,
};
pub use verifier::SumcheckVerifier;
