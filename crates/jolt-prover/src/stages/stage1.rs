//! Stage 1: the Spartan outer uni-skip round and the outer remainder
//! sumcheck.
//!
//! Pure orchestration: the challenge draws, batch head, point derivation,
//! final-claim fold, and absorb order are `jolt-verifier`'s generated
//! drivers; all compute (input-table materialization, the brute-forced
//! uni-skip polynomial, the remainder rounds) is behind the backend's
//! `spartan_outer_uniskip` and `spartan_outer_remainder` slots.

use jolt_claims::protocols::jolt::geometry::spartan::SpartanOuterDimensions;
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_kernels::{JoltBackend, ProofSession};
use jolt_openings::CommitmentScheme;
use jolt_r1cs::constraints::jolt::{
    SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE,
};
#[cfg(feature = "zk")]
use jolt_sumcheck::CommittedSumcheckWitness;
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::Transcript;
use jolt_verifier::stages::stage1::outer_remainder::{
    outer_remainder_input_values_from_uniskip_output, OuterRemainder,
};
use jolt_verifier::stages::stage1::outputs::{
    Stage1BatchInputClaims, Stage1BatchSumchecks, Stage1ClearOutput, Stage1OutputClaims,
};
use jolt_verifier::stages::uniskip::draw_spartan_outer_tau;
use jolt_witness::JoltWitnessPlane;

use crate::recorder::ProofMode;
use crate::{ProverError, StageProver as _};

/// Stage 1's outputs: the two wire proofs, the wire claims, and the
/// verifier-typed cross-stage carrier downstream stages consume.
pub struct Stage1ProverOutput<F: Field, C> {
    pub uniskip_proof: SumcheckProof<F, C>,
    pub sumcheck_proof: SumcheckProof<F, C>,
    pub claims: Stage1OutputClaims<F>,
    pub clear_output: Stage1ClearOutput<F>,
    #[cfg(feature = "zk")]
    pub uniskip_witness: CommittedSumcheckWitness<F>,
    #[cfg(feature = "zk")]
    pub committed_witness: CommittedSumcheckWitness<F>,
}

/// Prove stage 1 on `transcript` (positioned at the stage-0 boundary).
#[tracing::instrument(skip_all)]
pub fn prove_stage1<F, PCS, VC, T>(
    backend: &JoltBackend<F, PCS>,
    session: &mut ProofSession,
    mode: &ProofMode<'_, VC>,
    log_t: usize,
    witness: &dyn JoltWitnessPlane<F>,
    transcript: &mut T,
) -> Result<Stage1ProverOutput<F, VC::Output>, ProverError<F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
    let tau = draw_spartan_outer_tau(transcript, log_t);
    // Backend-neutral kernel-seam spans at the call boundary, so every
    // `UniskipKernel` implementation inherits them — see the taxonomy's
    // kernel-seam contract.
    tracing::info_span!("SpartanOuterUniskip::prepare").in_scope(|| {
        backend
            .spartan_outer_uniskip
            .prepare(session, log_t, &tau, witness)
    })?;

    let uniskip_poly = tracing::info_span!("SpartanOuterUniskip::first_round_poly")
        .in_scope(|| backend.spartan_outer_uniskip.first_round_poly(session, &[]))?;
    // The COMPOSED jolt-r1cs uni-skip shape (feature-aware): identical to the
    // jolt-claims RV64-only constants FR-off, the FR-extended row domain under
    // `field-inline` — the shape the verifier's stage-1 uni-skip checks.
    let proved_uniskip = mode.prove_uniskip(
        uniskip_poly,
        F::zero(),
        SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE,
        SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE,
        transcript,
    )?;
    let uniskip_challenge = proved_uniskip.challenge;

    // The generated stage drivers, on the verifier's own batch type.
    let sumchecks = Stage1BatchSumchecks {
        outer_remainder: OuterRemainder::new(
            SpartanOuterDimensions::rv64(log_t),
            tau,
            uniskip_challenge,
        ),
    };
    let challenges = sumchecks.draw_challenges(transcript)?;
    let input_points = sumchecks.empty_input_points();
    let inputs = Stage1BatchInputClaims {
        outer_remainder: outer_remainder_input_values_from_uniskip_output(
            proved_uniskip.output_claim,
        ),
    };

    let mut scheduler = backend.round_scheduler.build(session);
    let proved = sumchecks.prove(
        backend,
        session,
        &mut *scheduler,
        witness,
        &inputs,
        &input_points,
        &challenges,
        mode.recorder()?,
        transcript,
    )?;
    #[cfg(feature = "zk")]
    let (sumcheck_proof, committed_witness) = crate::recorder::split_recorded(proved.recorded)?;
    #[cfg(not(feature = "zk"))]
    let sumcheck_proof = proved.recorded.proof;

    Ok(Stage1ProverOutput {
        uniskip_proof: proved_uniskip.proof,
        sumcheck_proof,
        claims: Stage1OutputClaims::new(proved_uniskip.output_claim, proved.output_claims.clone()),
        clear_output: Stage1ClearOutput::new(proved.output_claims, proved.output_points),
        #[cfg(feature = "zk")]
        uniskip_witness: proved_uniskip.witness,
        #[cfg(feature = "zk")]
        committed_witness,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FR-off, the composed jolt-r1cs outer uni-skip constants equal the
    /// jolt-claims RV64-only constants this recipe previously passed — the
    /// swap is byte-neutral.
    #[cfg(not(feature = "field-inline"))]
    #[test]
    fn outer_uniskip_constants_match_the_rv64_only_values() {
        use jolt_claims::protocols::jolt::geometry::dimensions::{
            OUTER_UNISKIP_DOMAIN_SIZE, OUTER_UNISKIP_FIRST_ROUND_DEGREE,
        };

        assert_eq!(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, OUTER_UNISKIP_DOMAIN_SIZE);
        assert_eq!(
            SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE,
            OUTER_UNISKIP_FIRST_ROUND_DEGREE
        );
    }

    /// FR-on, the composed outer domain carries the appended FR rows — the
    /// spec's 14-point domain and its degree-39 first round.
    #[cfg(feature = "field-inline")]
    #[test]
    fn outer_uniskip_constants_are_the_composed_fr_domains() {
        assert_eq!(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, 14);
        assert_eq!(SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE, 39);
    }
}
