//! Sumcheck-based claim reduction: trait and utilities.
//!
//! A [`SumcheckReduction`] transforms opening claims into sumcheck instances,
//! runs the sumcheck protocol, and extracts new (reduced) claims from the
//! resulting challenges. This is the sumcheck analogue of
//! [`OpeningReduction`](jolt_openings::OpeningReduction) in jolt-openings.
//!
//! # Composability
//!
//! Reductions compose: the output `ProverClaim`s of one reduction can feed
//! into the input of another. The advice two-phase reduction is modeled as
//! two composed `SumcheckReduction`s — phase 1 binds cycle variables and
//! produces an intermediate claim whose `evaluations` carries the
//! partially-bound polynomial; phase 2 consumes that claim.
//!
//! # Relationship to `OpeningReduction`
//!
//! | | `OpeningReduction` | `SumcheckReduction` |
//! |---|---|---|
//! | Mechanism | Algebraic (RLC) | Interactive (sumcheck) |
//! | Proof artifact | `()` or small proof | `SumcheckProof<F>` |
//! | Claims in | `Vec<ProverClaim<F>>` | `Vec<ProverClaim<F>>` |
//! | Claims out | `Vec<ProverClaim<F>>` | `Vec<ProverClaim<F>>` |
//! | Lives in | `jolt-openings` | `jolt-sumcheck` |

use jolt_field::Field;
use jolt_openings::{ProverClaim, VerifierClaim};

use crate::claim::SumcheckClaim;
use crate::prover::SumcheckCompute;

/// Output of [`SumcheckReduction::build_witnesses`]: paired claims and witnesses.
pub type SumcheckWitnessBatch<F> = (Vec<SumcheckClaim<F>>, Vec<Box<dyn SumcheckCompute<F>>>);

/// Transforms opening claims via a sumcheck-based reduction.
///
/// Implementors define how to construct sumcheck witnesses from input claims
/// (prover side) and how to extract reduced claims from the sumcheck
/// challenge vector (both sides).
///
/// The protocol orchestrator calls the methods in this order:
///
/// ```text
/// Prover:
///   1. build_witnesses(input_claims)  → (sumcheck_claims, witnesses)
///   2. BatchedSumcheckProver::prove(sumcheck_claims, witnesses, ...)  → proof, challenges
///   3. extract_prover_claims(input_claims, challenges, final_eval)  → output_claims
///
/// Verifier:
///   1. build_verifier_claims(input_claims)  → sumcheck_claims
///   2. BatchedSumcheckVerifier::verify(sumcheck_claims, proof, ...)  → final_eval, challenges
///   3. extract_verifier_claims(input_claims, challenges, final_eval)  → output_claims
/// ```
///
/// # Composability
///
/// Output claims have the same type as input claims, so reductions chain:
///
/// ```text
/// claims → SumcheckReduction₁ → intermediate_claims → SumcheckReduction₂ → final_claims
/// ```
///
/// This is how the advice two-phase reduction works: phase 1 produces an
/// intermediate `ProverClaim` whose `evaluations` is the partially-bound
/// polynomial. Phase 2 constructs its witness from that evaluation table.
pub trait SumcheckReduction<F: Field> {
    /// Constructs sumcheck claims and witnesses from input opening claims.
    ///
    /// The returned witnesses implement [`SumcheckCompute`] and are fed
    /// directly to [`BatchedSumcheckProver::prove`](crate::BatchedSumcheckProver::prove).
    ///
    /// The `SumcheckClaim` vector and witness vector must have the same length
    /// and correspond element-wise.
    fn build_witnesses(
        &self,
        claims: &[ProverClaim<F>],
    ) -> SumcheckWitnessBatch<F>;

    /// Constructs sumcheck claims from verifier-side input claims.
    ///
    /// The verifier doesn't have polynomial evaluations, only commitments.
    /// This method builds the same `SumcheckClaim`s that the prover builds
    /// (same `num_vars`, `degree`, `claimed_sum`) so the verifier can check
    /// the proof.
    fn build_verifier_claims<C: Clone>(
        &self,
        claims: &[VerifierClaim<F, C>],
    ) -> Vec<SumcheckClaim<F>>;

    /// Extracts reduced prover claims after sumcheck completes.
    ///
    /// Given the original input claims, the sumcheck challenge vector, and
    /// the final evaluation, produces new `ProverClaim`s. These typically
    /// contain the partially-bound polynomial evaluations and the derived
    /// opening point.
    fn extract_prover_claims(
        &self,
        input_claims: &[ProverClaim<F>],
        challenges: &[F],
        final_eval: F,
    ) -> Vec<ProverClaim<F>>;

    /// Extracts reduced verifier claims after sumcheck completes.
    ///
    /// Mirror of [`extract_prover_claims`](Self::extract_prover_claims) for
    /// the verifier side. The verifier derives the same opening points and
    /// evaluation from the challenges, paired with the original commitments.
    fn extract_verifier_claims<C: Clone>(
        &self,
        input_claims: &[VerifierClaim<F, C>],
        challenges: &[F],
        final_eval: F,
    ) -> Vec<VerifierClaim<F, C>>;
}
