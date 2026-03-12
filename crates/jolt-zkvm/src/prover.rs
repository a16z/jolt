//! Top-level proving orchestration.
//!
//! [`prove`] runs the complete Jolt proving pipeline — from witness commitment
//! through Spartan, sumcheck stages, and batch opening proofs.

use jolt_field::WithChallenge;
use jolt_openings::{AdditivelyHomomorphic, OpeningReduction, RlcReduction};
use jolt_spartan::SpartanError;
use jolt_transcript::Transcript;

use crate::pipeline::prove_stages;
use crate::preprocessing::interleave_witnesses;
use crate::proof::{JoltProof, JoltProvingKey};
use crate::stage::ProverStage;
use crate::stages::s1_spartan::UniformSpartanStage;
use jolt_verifier::ProverConfig;

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

/// Runs the complete Jolt proving pipeline.
///
/// Orchestrates all phases:
///
/// 1. Interleave per-cycle witnesses into the flat R1CS witness
/// 2. Commit the witness and append commitment to transcript
/// 3. Run uniform Spartan (S1) to produce the R1CS proof
/// 4. Build sumcheck stages via `build_stages(r_x, r_y)`
/// 5. Run S2–S7 sumcheck stages
/// 6. Collect all opening claims (stages + Spartan witness)
/// 7. RLC-reduce and produce batch PCS opening proofs (S8)
///
/// # Arguments
///
/// * `key` — preprocessed proving key (Spartan key + PCS setup)
/// * `cycle_witnesses` — per-cycle variable assignments (one `Vec<F>` per cycle)
/// * `poly_commitments` — commitments to stage polynomials (not the witness)
/// * `build_stages` — closure receiving `(r_x, r_y, transcript)` from Spartan,
///   returns prover stages for S2–S7. Transcript access allows squeezing
///   batching challenges (e.g., γ) at the correct Fiat-Shamir state.
/// * `transcript` — Fiat-Shamir transcript
#[tracing::instrument(skip_all, name = "prove")]
pub fn prove<PCS, T>(
    key: &JoltProvingKey<PCS::Field, PCS>,
    cycle_witnesses: &[Vec<PCS::Field>],
    poly_commitments: Vec<PCS::Output>,
    build_stages: impl FnOnce(
        &[PCS::Field],
        &[PCS::Field],
        &mut T,
    ) -> Vec<Box<dyn ProverStage<PCS::Field, T>>>,
    transcript: &mut T,
) -> Result<JoltProof<PCS::Field, PCS>, ProveError>
where
    PCS: AdditivelyHomomorphic,
    PCS::Field: WithChallenge,
    <PCS::Field as WithChallenge>::Challenge: From<T::Challenge>,
    T: Transcript,
{
    // S0: Interleave per-cycle witnesses and commit.
    let (flat_witness, witness_commitment) = {
        let _span = tracing::info_span!("S0_witness_commit").entered();
        let flat_witness = interleave_witnesses(&key.spartan_key, cycle_witnesses);
        let (witness_commitment, _) = PCS::commit(&flat_witness, &key.pcs_prover_setup);
        transcript.append_bytes(format!("{witness_commitment:?}").as_bytes());
        tracing::info!(
            num_cycles = key.spartan_key.num_cycles,
            witness_len = flat_witness.len(),
            "witness committed"
        );
        (flat_witness, witness_commitment)
    };

    // S1: Uniform Spartan PIOP.
    let spartan_result = {
        let _span = tracing::info_span!("S1_spartan").entered();
        UniformSpartanStage::prove(&key.spartan_key, &flat_witness, &flat_witness, transcript)?
    };

    // Build stages using Spartan challenge vectors. Transcript is passed so
    // the factory can squeeze batching challenges at the correct Fiat-Shamir state.
    let mut stages = build_stages(&spartan_result.r_x, &spartan_result.r_y, transcript);

    // S2–S7: Sumcheck stages.
    let (stage_proofs, mut opening_claims) = prove_stages(&mut stages, transcript);

    // Spartan witness opening claim — added last to match verifier ordering.
    opening_claims.push(spartan_result.witness_opening_claim);

    // S8: RLC reduction + PCS opening proofs.
    let proofs = {
        let _span = tracing::info_span!("S8_opening_proofs").entered();
        tracing::info!(total_claims = opening_claims.len(), "reducing and opening");

        let (reduced, ()) = <RlcReduction as OpeningReduction<PCS>>::reduce_prover(
            opening_claims,
            transcript,
        );

        tracing::info!(reduced_claims = reduced.len(), "opening PCS proofs");

        reduced
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
            .collect()
    };

    Ok(JoltProof {
        config: ProverConfig {
            trace_length: key.spartan_key.num_cycles,
            ram_k: 0,
            one_hot_config: jolt_verifier::OneHotConfig::new(
                key.spartan_key.num_cycles.trailing_zeros() as usize,
            ),
            rw_config: jolt_verifier::ReadWriteConfig::new(
                key.spartan_key.num_cycles.trailing_zeros() as usize,
                0,
            ),
        },
        spartan_proof: spartan_result.proof,
        stage_proofs,
        opening_proofs: proofs,
        witness_commitment,
        commitments: poly_commitments,
    })
}
