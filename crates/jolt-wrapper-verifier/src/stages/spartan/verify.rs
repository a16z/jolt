use jolt_claims::protocols::wrapper_spartan_hyperkzg::{
    SpartanInnerBatchingCoefficients, SpartanInnerStatement, SpartanOuterEvaluationClaims,
    SpartanOuterReduction, SpartanWitnessOpeningStatement,
    WRAPPER_SPARTAN_INNER_BATCHING_TRANSCRIPT_LABEL, WRAPPER_SPARTAN_INNER_SUMCHECK_DEGREE,
    WRAPPER_SPARTAN_INNER_SUMCHECK_TRANSCRIPT_LABEL, WRAPPER_SPARTAN_OUTER_SUMCHECK_DEGREE,
    WRAPPER_SPARTAN_OUTER_SUMCHECK_TRANSCRIPT_LABEL, WRAPPER_SPARTAN_TAU_TRANSCRIPT_LABEL,
};
use jolt_crypto::PairingGroup;
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_field::{Field, FromPrimitiveInt};
use jolt_poly::EqPolynomial;
use jolt_poly::{Point, HIGH_TO_LOW};
#[cfg(feature = "zk")]
use jolt_sumcheck::SumcheckStatement;
use jolt_sumcheck::{BooleanHypercube, SumcheckClaim};
use jolt_transcript::{AppendToTranscript, Label, Transcript};

use crate::{stages::spartan::inputs::SpartanInputs, WrapperError};

use super::outputs::SpartanOutput;
#[cfg(feature = "zk")]
use super::{inputs::SpartanZkInputs, outputs::SpartanZkOutput};

#[cfg(feature = "zk")]
use crate::stages::zk::committed;

#[cfg(feature = "zk")]
const OUTER_OUTPUT_CLAIMS: usize = 3;
#[cfg(feature = "zk")]
const INNER_OUTPUT_CLAIMS: usize = 1;

pub fn verify<P, T>(
    inputs: SpartanInputs<'_, '_, P>,
    transcript: &mut T,
) -> Result<SpartanOutput<P::ScalarField>, WrapperError>
where
    P: PairingGroup,
    P::ScalarField: Field + AppendToTranscript,
    T: Transcript<Challenge = P::ScalarField>,
{
    let dimensions = inputs.deps.relation.statement_facts.spartan;
    let tau = tau_point(dimensions.num_constraint_rounds(), transcript);
    let claim = SumcheckClaim::new(
        dimensions.num_constraint_rounds(),
        WRAPPER_SPARTAN_OUTER_SUMCHECK_DEGREE,
        P::ScalarField::from_u64(0),
    );

    transcript.append(&Label(WRAPPER_SPARTAN_OUTER_SUMCHECK_TRANSCRIPT_LABEL));
    let outer_reduction = inputs
        .proof
        .spartan
        .outer_sumcheck
        .verify(
            &claim,
            BooleanHypercube,
            jolt_sumcheck::SUMCHECK_ROUND_TRANSCRIPT_LABEL,
            transcript,
        )
        .map_err(|error| WrapperError::SpartanSumcheckFailed {
            reason: error.to_string(),
        })?;
    let outer =
        SpartanOuterReduction::new(tau, outer_reduction.point.clone(), outer_reduction.value);
    let outer_evaluation_claims = inputs.proof.spartan.outer_evaluation_claims;
    check_outer_reduction(&outer, outer_evaluation_claims)?;

    let inner_batching = inner_batching_coefficients(outer_evaluation_claims, transcript);
    let inner_claim = SumcheckClaim::new(
        dimensions.num_var_rounds(),
        WRAPPER_SPARTAN_INNER_SUMCHECK_DEGREE,
        inner_batching.combine(outer_evaluation_claims),
    );

    transcript.append(&Label(WRAPPER_SPARTAN_INNER_SUMCHECK_TRANSCRIPT_LABEL));
    let inner_reduction = inputs
        .proof
        .spartan
        .inner_sumcheck
        .verify(
            &inner_claim,
            BooleanHypercube,
            jolt_sumcheck::SUMCHECK_ROUND_TRANSCRIPT_LABEL,
            transcript,
        )
        .map_err(|error| WrapperError::SpartanSumcheckFailed {
            reason: error.to_string(),
        })?;
    let inner =
        SpartanInnerStatement::new(outer.rx.clone(), inner_batching, inner_claim.claimed_sum);
    check_inner_reduction(&inputs, inner_batching, &outer, &inner_reduction)?;
    let witness_opening = SpartanWitnessOpeningStatement::new(
        inner_reduction.point.clone(),
        inputs.proof.spartan.witness_opening_claim,
    );

    Ok(SpartanOutput {
        dimensions,
        outer_reduction,
        outer,
        outer_evaluation_claims,
        inner_batching,
        inner_reduction,
        inner,
        witness_opening,
    })
}

#[cfg(feature = "zk")]
pub fn verify_zk<P, VC, T>(
    inputs: SpartanZkInputs<'_, '_, P, VC>,
    transcript: &mut T,
) -> Result<SpartanZkOutput<P::ScalarField, VC::Output>, WrapperError>
where
    P: PairingGroup,
    P::ScalarField: Field + AppendToTranscript,
    VC: VectorCommitment<Field = P::ScalarField>,
    VC::Output: Clone + AppendToTranscript,
    T: Transcript<Challenge = P::ScalarField>,
{
    let dimensions = inputs.deps.relation.statement_facts.spartan;
    let tau = tau_point(dimensions.num_constraint_rounds(), transcript);
    let outer_statement = SumcheckStatement::new(
        dimensions.num_constraint_rounds(),
        WRAPPER_SPARTAN_OUTER_SUMCHECK_DEGREE,
    );

    transcript.append(&Label(WRAPPER_SPARTAN_OUTER_SUMCHECK_TRANSCRIPT_LABEL));
    let outer_consistency = inputs
        .proof
        .spartan
        .outer_sumcheck
        .verify_committed_consistency(outer_statement, transcript)
        .map_err(|error| WrapperError::SpartanSumcheckFailed {
            reason: error.to_string(),
        })?;
    let outer_output_claims = committed::verify_output_claim_commitments::<VC>(
        inputs.vc_setup,
        "wrapper-spartan-outer",
        &inputs.proof.spartan.outer_sumcheck,
        OUTER_OUTPUT_CLAIMS,
    )?;
    let outer_rx = Point::high_to_low(outer_consistency.challenges());
    let eq_tau_rx = EqPolynomial::<P::ScalarField>::mle(tau.as_slice(), outer_rx.as_slice());

    let inner_batching = inner_batching_coefficients_zk(transcript);
    let inner_statement = SumcheckStatement::new(
        dimensions.num_var_rounds(),
        WRAPPER_SPARTAN_INNER_SUMCHECK_DEGREE,
    );

    transcript.append(&Label(WRAPPER_SPARTAN_INNER_SUMCHECK_TRANSCRIPT_LABEL));
    let inner_consistency = inputs
        .proof
        .spartan
        .inner_sumcheck
        .verify_committed_consistency(inner_statement, transcript)
        .map_err(|error| WrapperError::SpartanSumcheckFailed {
            reason: error.to_string(),
        })?;
    let inner_output_claims = committed::verify_output_claim_commitments::<VC>(
        inputs.vc_setup,
        "wrapper-spartan-inner",
        &inputs.proof.spartan.inner_sumcheck,
        INNER_OUTPUT_CLAIMS,
    )?;
    let inner_ry = Point::high_to_low(inner_consistency.challenges());
    let combined_matrix_eval = combined_matrix_eval(&inputs, inner_batching, &outer_rx, &inner_ry)?;

    Ok(SpartanZkOutput {
        dimensions,
        tau,
        outer_statement,
        outer_consistency,
        outer_rx,
        eq_tau_rx,
        outer_output_claims,
        inner_batching,
        inner_statement,
        inner_consistency,
        inner_ry,
        combined_matrix_eval,
        inner_output_claims,
    })
}

fn check_outer_reduction<F: Field>(
    outer: &SpartanOuterReduction<F>,
    claims: SpartanOuterEvaluationClaims<F>,
) -> Result<(), WrapperError> {
    let eq_tau_rx = EqPolynomial::<F>::mle(outer.tau.as_slice(), outer.rx.as_slice());
    let expected = eq_tau_rx * (claims.a * claims.b - claims.c);
    if outer.final_claim != expected {
        return Err(WrapperError::SpartanOuterReductionMismatch);
    }
    Ok(())
}

fn inner_batching_coefficients<F, T>(
    claims: SpartanOuterEvaluationClaims<F>,
    transcript: &mut T,
) -> SpartanInnerBatchingCoefficients<F>
where
    F: Field + AppendToTranscript,
    T: Transcript<Challenge = F>,
{
    transcript.append(&Label(WRAPPER_SPARTAN_INNER_BATCHING_TRANSCRIPT_LABEL));
    transcript.append(&claims.a);
    transcript.append(&claims.b);
    transcript.append(&claims.c);
    SpartanInnerBatchingCoefficients {
        a: transcript.challenge_scalar(),
        b: transcript.challenge_scalar(),
        c: transcript.challenge_scalar(),
    }
}

#[cfg(feature = "zk")]
fn inner_batching_coefficients_zk<F, T>(transcript: &mut T) -> SpartanInnerBatchingCoefficients<F>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    transcript.append(&Label(WRAPPER_SPARTAN_INNER_BATCHING_TRANSCRIPT_LABEL));
    SpartanInnerBatchingCoefficients {
        a: transcript.challenge_scalar(),
        b: transcript.challenge_scalar(),
        c: transcript.challenge_scalar(),
    }
}

fn check_inner_reduction<P>(
    inputs: &SpartanInputs<'_, '_, P>,
    batching: SpartanInnerBatchingCoefficients<P::ScalarField>,
    outer: &SpartanOuterReduction<P::ScalarField>,
    inner_reduction: &jolt_sumcheck::EvaluationClaim<P::ScalarField>,
) -> Result<(), WrapperError>
where
    P: PairingGroup,
    P::ScalarField: Field,
{
    let matrix_evals = inputs
        .deps
        .relation
        .relation
        .evaluate_matrix_mles(&outer.rx, &inner_reduction.point)
        .map_err(|error| WrapperError::R1csMatrixEvaluationFailed {
            reason: error.to_string(),
        })?;
    let combined_matrix_eval =
        batching.a * matrix_evals.a + batching.b * matrix_evals.b + batching.c * matrix_evals.c;
    let expected = combined_matrix_eval * inputs.proof.spartan.witness_opening_claim;
    if inner_reduction.value != expected {
        return Err(WrapperError::SpartanInnerReductionMismatch);
    }
    Ok(())
}

#[cfg(feature = "zk")]
fn combined_matrix_eval<P, VC>(
    inputs: &SpartanZkInputs<'_, '_, P, VC>,
    batching: SpartanInnerBatchingCoefficients<P::ScalarField>,
    outer_rx: &Point<HIGH_TO_LOW, P::ScalarField>,
    inner_ry: &Point<HIGH_TO_LOW, P::ScalarField>,
) -> Result<P::ScalarField, WrapperError>
where
    P: PairingGroup,
    P::ScalarField: Field,
    VC: VectorCommitment<Field = P::ScalarField>,
{
    let matrix_evals = inputs
        .deps
        .relation
        .relation
        .evaluate_matrix_mles(outer_rx, inner_ry)
        .map_err(|error| WrapperError::R1csMatrixEvaluationFailed {
            reason: error.to_string(),
        })?;
    Ok(batching.a * matrix_evals.a + batching.b * matrix_evals.b + batching.c * matrix_evals.c)
}

fn tau_point<F, T>(num_rounds: usize, transcript: &mut T) -> Point<HIGH_TO_LOW, F>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    transcript.append(&Label(WRAPPER_SPARTAN_TAU_TRANSCRIPT_LABEL));
    Point::high_to_low(transcript.challenge_vector(num_rounds))
}
