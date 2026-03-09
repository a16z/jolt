//! Verifier stage interface.
//!
//! Each sumcheck stage (S2–S7) has a verifier-side counterpart that constructs
//! sumcheck claims, checks the final evaluation against claimed polynomial
//! evaluations, and produces [`VerifierClaim`]s for the batch opening phase.
//!
//! Implementations are config-driven — constructed by the caller (jolt-zkvm)
//! using [`ClaimDefinition`](jolt_ir::ClaimDefinition)s and stage metadata.
//! The verifier pipeline consumes them via the [`VerifierStage`] trait.

use jolt_field::Field;
use jolt_openings::VerifierClaim;
use jolt_sumcheck::SumcheckClaim;
use jolt_transcript::Transcript;

use crate::error::JoltError;

/// Verifier-side counterpart to a prover stage.
///
/// Each stage:
/// 1. Builds sumcheck claims from prior opening claims and transcript state
/// 2. After sumcheck verification, checks claimed evaluations against the
///    sumcheck final evaluation and produces opening claims for S8
///
/// Generic over `F` (field), `C` (commitment type), and `T` (transcript)
/// for dyn-compatibility when all three are fixed by the pipeline caller.
///
/// # Implementing
///
/// Implementations should be config-driven. A single generic struct
/// (e.g., `ClaimReductionVerifier`) can handle all claim reduction stages
/// by parameterizing on degree, num_vars, and a [`ClaimDefinition`](jolt_ir::ClaimDefinition).
pub trait VerifierStage<F: Field, C: Clone, T: Transcript> {
    /// Constructs sumcheck claims for this stage.
    ///
    /// `prior_claims` contains all [`VerifierClaim`]s produced by previous
    /// stages. The implementation selects the relevant claims, computes
    /// the `claimed_sum` for each sumcheck instance, and returns the
    /// fully-populated claims.
    ///
    /// Implementations may sample additional challenges from the transcript
    /// (e.g., batching coefficients) — the same challenges will be sampled
    /// by the prover.
    fn build_claims(
        &mut self,
        prior_claims: &[VerifierClaim<F, C>],
        transcript: &mut T,
    ) -> Vec<SumcheckClaim<F>>;

    /// Checks claimed evaluations and produces opening claims.
    ///
    /// Called after the batched sumcheck verifier returns `(final_eval, challenges)`.
    /// The implementation must:
    ///
    /// 1. Verify that the claimed `evaluations` are consistent with
    ///    `final_eval` using the stage's claim formula
    /// 2. Pair each evaluation with its polynomial commitment from `commitments`
    /// 3. Return the resulting [`VerifierClaim`]s
    ///
    /// # Arguments
    ///
    /// * `final_eval` — sumcheck final evaluation from the verifier
    /// * `challenges` — sumcheck challenge vector (= evaluation point)
    /// * `evaluations` — claimed polynomial evaluations from the proof
    /// * `commitments` — all polynomial commitments from the proof
    fn check_and_extract(
        &mut self,
        final_eval: F,
        challenges: &[F],
        evaluations: &[F],
        commitments: &[C],
    ) -> Result<Vec<VerifierClaim<F, C>>, JoltError>;
}
