//! Top-level proving orchestration.
//!
//! [`prove`] sequences the three protocol phases — uniform Spartan result
//! intake, batched sumcheck stages S2–S7, and batch opening proofs — into a
//! complete [`JoltProof`].

use jolt_openings::{AdditivelyHomomorphic, OpeningReduction, RlcReduction};
use jolt_spartan::SpartanError;
use jolt_transcript::Transcript;

use crate::pipeline::prove_stages;
use crate::proof::{BatchOpeningProofs, JoltProof, JoltProvingKey};
use crate::stage::ProverStage;
use crate::stages::s1_spartan::UniformSpartanResult;

/// Errors that can occur during the full proving pipeline.
#[derive(Debug)]
pub enum ProveError {
    /// Spartan R1CS proving failed (constraint violation or sumcheck error).
    Spartan(SpartanError),
}

impl From<SpartanError> for ProveError {
    fn from(e: SpartanError) -> Self {
        ProveError::Spartan(e)
    }
}

impl std::fmt::Display for ProveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProveError::Spartan(e) => write!(f, "spartan: {e}"),
        }
    }
}

impl std::error::Error for ProveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ProveError::Spartan(e) => Some(e),
        }
    }
}

/// Runs the Jolt proving pipeline from a completed uniform Spartan result.
///
/// The caller is responsible for:
/// 1. Running S1 via [`UniformSpartanStage::prove`](crate::stages::s1_spartan::UniformSpartanStage::prove)
/// 2. Using `UniformSpartanResult::r_x` / `r_y` to construct stages S2–S7 with
///    the correct eq-points and gamma powers
/// 3. Committing to all polynomials before calling this function
/// 4. Passing the stages, Spartan result, and commitments here
///
/// This function:
/// 1. Runs S2–S7 via [`prove_stages`]
/// 2. Collects all opening claims (including the witness claim from S1)
/// 3. Runs S8 via RLC reduction + PCS opening proofs
///
/// # Arguments
///
/// * `spartan_result` — completed uniform Spartan proof and challenge vectors from S1
/// * `stages` — S2–S7 stages, pre-constructed with inter-stage challenges
/// * `key` — preprocessed proving key (contains PCS setup)
/// * `commitments` — commitments to all polynomials (created during witness gen)
/// * `trace_length` — number of execution cycles
/// * `transcript` — Fiat-Shamir transcript, in the same state the verifier
///   will have after verifying S1
/// * `challenge_fn` — converts transcript challenges to field elements
#[tracing::instrument(skip_all, name = "prove")]
pub fn prove<PCS, T>(
    spartan_result: UniformSpartanResult<PCS::Field, PCS>,
    stages: &mut [Box<dyn ProverStage<PCS::Field, T>>],
    key: &JoltProvingKey<PCS::Field, PCS>,
    commitments: Vec<PCS::Output>,
    trace_length: usize,
    transcript: &mut T,
    challenge_fn: impl Fn(T::Challenge) -> PCS::Field + Copy,
) -> JoltProof<PCS::Field, PCS>
where
    PCS: AdditivelyHomomorphic,
    T: Transcript,
{
    let (stage_proofs, mut opening_claims) = prove_stages(stages, transcript, challenge_fn);

    opening_claims.push(spartan_result.witness_opening_claim);

    // S8: RLC reduction + PCS opening proofs
    let (reduced, ()) = <RlcReduction as OpeningReduction<PCS>>::reduce_prover(
        opening_claims,
        transcript,
        &challenge_fn,
    );

    let proofs = reduced
        .into_iter()
        .map(|claim| {
            let poly: PCS::Polynomial = claim.evaluations.into();
            PCS::open(
                &poly,
                &claim.point,
                claim.eval,
                &key.pcs_prover_setup,
                None,
                transcript,
            )
        })
        .collect();

    JoltProof {
        spartan_proof: spartan_result.proof,
        stage_proofs,
        opening_proofs: BatchOpeningProofs { proofs },
        commitments,
        trace_length,
    }
}
