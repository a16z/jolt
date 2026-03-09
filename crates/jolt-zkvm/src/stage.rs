//! Prover stage trait and batch output.
//!
//! Each sumcheck stage in the Jolt pipeline implements [`ProverStage`],
//! producing claims and witnesses for [`BatchedSumcheckProver`](jolt_sumcheck::BatchedSumcheckProver).

use jolt_field::Field;
use jolt_ir::ClaimDefinition;
use jolt_openings::ProverClaim;
use jolt_sumcheck::claim::SumcheckClaim;
use jolt_sumcheck::prover::SumcheckCompute;
use jolt_transcript::Transcript;

/// Output of a stage's [`build()`](ProverStage::build) method.
///
/// Contains paired sumcheck claims and witnesses, ready to feed into
/// [`BatchedSumcheckProver::prove`](jolt_sumcheck::BatchedSumcheckProver::prove).
pub struct StageBatch<F: Field> {
    /// Sumcheck claims (one per instance in this stage).
    pub claims: Vec<SumcheckClaim<F>>,
    /// Sumcheck witnesses (element-wise paired with `claims`).
    pub witnesses: Vec<Box<dyn SumcheckCompute<F>>>,
}

/// A proving stage that contributes batched sumcheck instances.
///
/// Each of the 7 sumcheck stages in the Jolt pipeline implements this trait.
/// The prover pipeline calls [`build()`](Self::build) to construct claims and
/// witnesses, runs the batched sumcheck prover, then calls
/// [`extract_claims()`](Self::extract_claims) to produce opening claims for
/// downstream stages and stage 8.
///
/// Generic over `T: Transcript` for dyn-compatibility — the pipeline fixes
/// a concrete transcript type and uses `dyn ProverStage<F, T>`.
pub trait ProverStage<F: Field, T: Transcript> {
    /// Constructs sumcheck claims and witnesses for this stage.
    ///
    /// `prior_claims` contains opening claims from all previous stages,
    /// used to derive input claims for this stage's sumcheck instances.
    fn build(&mut self, prior_claims: &[ProverClaim<F>], transcript: &mut T) -> StageBatch<F>;

    /// Extracts opening claims after sumcheck completes.
    ///
    /// Given the sumcheck challenge vector and final evaluation, produces
    /// `ProverClaim`s that:
    /// - Feed into subsequent stages as input claims
    /// - Feed into stage 8 for RLC reduction and PCS opening proofs
    ///
    /// Implementations typically evaluate the original polynomials at the
    /// challenge point and pair evaluations with the evaluation tables
    /// (moved from the [`WitnessStore`](crate::witness::WitnessStore)).
    fn extract_claims(&mut self, challenges: &[F], final_eval: F) -> Vec<ProverClaim<F>>;

    /// IR-based claim definitions for this stage's sumcheck instances.
    ///
    /// Returns the same formulas used by [`build()`](Self::build) to
    /// construct claims, in symbolic form. Used by BlindFold (ZK mode)
    /// to build verifier R1CS constraints, and by tests to verify that
    /// claim formulas produce correct evaluations.
    fn claim_definitions(&self) -> Vec<ClaimDefinition>;
}
