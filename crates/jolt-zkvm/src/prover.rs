//! Top-level proving orchestration.
//!
//! [`prove`] sequences the three protocol phases — Spartan result intake,
//! batched sumcheck stages S2–S7, and batch opening proofs — into a complete
//! [`JoltProof`].

use jolt_openings::AdditivelyHomomorphic;
use jolt_spartan::SpartanError;
use jolt_transcript::Transcript;

use crate::pipeline::prove_stages;
use crate::proof::{JoltProof, JoltProvingKey};
use crate::stage::ProverStage;
use crate::stages::s1_spartan::SpartanResult;
use crate::stages::s8_opening::OpeningStage;

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

/// Runs the Jolt proving pipeline from a completed Spartan result.
///
/// The caller is responsible for:
/// 1. Running S1 via [`SpartanStage::prove`](crate::stages::s1_spartan::SpartanStage::prove)
/// 2. Using `SpartanResult::r_x` / `r_y` to construct stages S2–S7 with
///    the correct eq-points and gamma powers
/// 3. Passing the stages and Spartan result here
///
/// This function:
/// 1. Runs S2–S7 via [`prove_stages`]
/// 2. Collects all opening claims (including the witness claim from S1)
/// 3. Runs S8 via [`OpeningStage::prove`] to produce batch opening proofs
///
/// # Arguments
///
/// * `spartan_result` — completed Spartan proof and challenge vectors from S1
/// * `stages` — S2–S7 stages, pre-constructed with inter-stage challenges
/// * `key` — preprocessed proving key (contains PCS setup)
/// * `transcript` — Fiat-Shamir transcript, in the same state the verifier
///   will have after verifying S1
/// * `challenge_fn` — converts transcript challenges to field elements
#[tracing::instrument(skip_all, name = "prove")]
pub fn prove<PCS, T>(
    spartan_result: SpartanResult<PCS::Field, PCS>,
    stages: &mut [Box<dyn ProverStage<PCS::Field, T>>],
    key: &JoltProvingKey<PCS::Field, PCS>,
    transcript: &mut T,
    challenge_fn: impl Fn(T::Challenge) -> PCS::Field + Copy,
) -> JoltProof<PCS::Field, PCS>
where
    PCS: AdditivelyHomomorphic,
    T: Transcript,
{
    let (sumcheck_proofs, mut opening_claims) = prove_stages(stages, transcript, challenge_fn);

    opening_claims.push(spartan_result.witness_opening_claim);

    let opening_proofs = OpeningStage::<PCS>::prove(
        opening_claims,
        &key.pcs_prover_setup,
        transcript,
        challenge_fn,
    );

    JoltProof {
        spartan_proof: spartan_result.proof,
        sumcheck_proofs,
        opening_proofs,
    }
}
