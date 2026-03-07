//! Verifier stage trait.
//!
//! Each sumcheck stage in the Jolt pipeline has a verifier-side
//! counterpart that checks the sumcheck proof and extracts opening claims.

use jolt_openings::{CommitmentScheme, VerifierClaim};
use jolt_sumcheck::{SumcheckClaim, SumcheckProof};
use jolt_transcript::Transcript;

use crate::error::JoltError;

/// Verifier-side counterpart to a prover stage.
///
/// Each stage verifies the sumcheck proof produced by the prover,
/// checks the output claim, and produces `VerifierClaim`s that
/// feed into stage 8's opening reduction.
///
/// Stages are stateless — all data flows through method parameters.
///
/// Generic over both `PCS` (commitment scheme) and `T` (transcript)
/// to remain dyn-compatible.
pub trait VerifierStage<PCS: CommitmentScheme, T: Transcript> {
    /// Construct sumcheck claims from the proof and prior opening claims.
    ///
    /// The returned claims must match those produced by the prover's
    /// `ProverStage::build()` (same `num_vars`, `degree`, `claimed_sum`).
    fn build_claims(
        &self,
        prior_claims: &[VerifierClaim<PCS::Field, PCS::Output>],
        transcript: &mut T,
    ) -> Vec<SumcheckClaim<PCS::Field>>;

    /// Verify the sumcheck proof and extract opening claims.
    ///
    /// This is the main verification logic:
    /// 1. Verify the sumcheck proof against the claims
    /// 2. Check the output claim (expected vs. actual)
    /// 3. Record opening claims for stage 8
    #[allow(clippy::type_complexity)]
    fn verify(
        &self,
        claims: &[SumcheckClaim<PCS::Field>],
        proof: &SumcheckProof<PCS::Field>,
        prior_claims: &[VerifierClaim<PCS::Field, PCS::Output>],
        transcript: &mut T,
    ) -> Result<Vec<VerifierClaim<PCS::Field, PCS::Output>>, JoltError>;
}
