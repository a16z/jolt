//! Stage 1: the Spartan outer uni-skip round and the outer remainder
//! sumcheck.
//!
//! Pure orchestration: the challenge draws, batch head, point derivation,
//! final-claim fold, and absorb order are `jolt-verifier`'s generated
//! drivers; all compute (input-table materialization, the brute-forced
//! uni-skip polynomial, the remainder rounds) is behind the backend's
//! `spartan_outer` slot.

use jolt_claims::protocols::jolt::geometry::dimensions::{
    OUTER_UNISKIP_DOMAIN_SIZE, OUTER_UNISKIP_FIRST_ROUND_DEGREE,
};
use jolt_claims::protocols::jolt::geometry::spartan::SpartanOuterDimensions;
use jolt_field::Field;
use jolt_kernels::{JoltBackend, ProofSession};
use jolt_openings::CommitmentScheme;
use jolt_sumcheck::{
    prove_batch, prove_uniskip_clear, ClearSumcheckRecorder, ProveRounds, SumcheckProof,
    SumcheckRecorder,
};
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::stages::stage1::outer_remainder::{
    outer_remainder_input_values_from_uniskip_output, OuterRemainder,
};
use jolt_verifier::stages::stage1::outputs::{
    Stage1BatchInputClaims, Stage1BatchOutputClaims, Stage1BatchSumchecks, Stage1ClearOutput,
    Stage1OutputClaims,
};
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::ProverError;

/// Stage 1's outputs: the two wire proofs, the wire claims, and the
/// verifier-typed cross-stage carrier downstream stages consume.
pub struct Stage1ProverOutput<F: Field, C> {
    pub uniskip_proof: SumcheckProof<F, C>,
    pub sumcheck_proof: SumcheckProof<F, C>,
    pub claims: Stage1OutputClaims<F>,
    pub clear_output: Stage1ClearOutput<F>,
}

/// Prove stage 1 on `transcript` (positioned at the stage-0 boundary).
pub fn prove_stage1<F, PCS, C, T>(
    backend: &JoltBackend<F, PCS>,
    session: &mut ProofSession,
    log_t: usize,
    witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    transcript: &mut T,
) -> Result<Stage1ProverOutput<F, C>, ProverError<F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
    C: Clone + AppendToTranscript,
    T: Transcript<Challenge = F>,
{
    let tau = transcript.challenge_vector(log_t + 2);
    let instance = backend
        .spartan_outer
        .prepare(session, log_t, &tau, witness)?;

    let uniskip_poly = instance.uniskip_first_round_poly()?;
    let proved_uniskip = prove_uniskip_clear::<F, C, T>(
        uniskip_poly,
        F::zero(),
        OUTER_UNISKIP_FIRST_ROUND_DEGREE,
        OUTER_UNISKIP_DOMAIN_SIZE,
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

    let mut recorder = ClearSumcheckRecorder::<F, C>::new();
    let (batch, coefficients) =
        sumchecks.begin_batch(&inputs, &challenges, &mut recorder, transcript)?;

    let mut member = instance.into_remainder(uniskip_challenge)?;
    let mut members: Vec<&mut dyn ProveRounds<F>> = vec![&mut *member];
    let proved = prove_batch(&batch, &mut members, &mut recorder, transcript)?;

    let output_points = sumchecks.derive_opening_points(&proved.challenges, &input_points)?;
    member.validate_derived_tables(
        &sumchecks.outer_remainder,
        &input_points.outer_remainder,
        &output_points.outer_remainder,
        &challenges.outer_remainder,
    )?;
    let output_values = Stage1BatchOutputClaims {
        outer_remainder: member.output_claims()?,
    };
    sumchecks.validate_output_claims(&output_values)?;
    let expected = sumchecks.expected_final_claim(
        &coefficients,
        &input_points,
        &output_values,
        &output_points,
        &challenges,
    )?;
    if expected != proved.final_claim {
        return Err(ProverError::FinalClaimMismatch {
            stage: "stage1",
            expected,
            got: proved.final_claim,
        });
    }

    let recorded = recorder.finish(&sumchecks.opening_values(&output_values), transcript)?;

    Ok(Stage1ProverOutput {
        uniskip_proof: proved_uniskip.proof,
        sumcheck_proof: recorded.proof,
        claims: Stage1OutputClaims {
            uniskip_output_claim: proved_uniskip.output_claim,
            outer: output_values.clone(),
        },
        clear_output: Stage1ClearOutput {
            output_values,
            output_points,
        },
    })
}
