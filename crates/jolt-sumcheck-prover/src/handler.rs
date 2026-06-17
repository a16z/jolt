use jolt_field::{Field, Invertible, MulPow2};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{
    BatchedEvaluationClaim, BatchedSumcheckVerifier, CompressedSumcheckProof, SumcheckClaim,
};
use jolt_transcript::Transcript;

use crate::backend::SumcheckBackend;
use crate::error::{BackendError, ProverError};
use crate::recorder::SumcheckProofRecorder;
use crate::spec::BatchedSumcheckSpec;

/// Observable output of one batched sumcheck prove invocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchedSumcheckProveResult<F: Field> {
    pub proof: CompressedSumcheckProof<F>,
    pub batching_coefficients: Vec<F>,
    pub challenges: Vec<F>,
    pub instance_final_claims: Vec<F>,
    pub batched_claim: F,
}

/// Canonical batched-sumcheck prover handler.
///
/// Mirrors [`BatchedSumcheckVerifier::verify_compressed`]: front-loaded dummy
/// rounds, claim scaling by `2^{max_num_vars - num_vars}`, and RLC of per-instance
/// round polynomials before each transcript absorption.
pub fn prove_sumcheck<F, B, T, R>(
    spec: &BatchedSumcheckSpec<F>,
    backend: &mut B,
    transcript: &mut T,
    recorder: &mut R,
) -> Result<BatchedSumcheckProveResult<F>, ProverError<F>>
where
    F: Field + MulPow2 + Invertible,
    B: SumcheckBackend<F>,
    T: Transcript<Challenge = F>,
    R: SumcheckProofRecorder<F, Proof = CompressedSumcheckProof<F>>,
{
    if spec.instances.is_empty() {
        return Err(ProverError::EmptyBatch);
    }

    let num_rounds = spec.num_rounds();
    let max_num_vars = spec.max_num_vars();
    let _max_degree = spec.max_degree();
    let two_inv = F::from_u64(2).inv_or_zero();

    let input_claims: Vec<F> = spec
        .instances
        .iter()
        .map(|instance| instance.input_claim)
        .collect();
    recorder.absorb_input_claims(&input_claims, transcript);
    let batching_coefficients = (0..spec.instances.len())
        .map(|_| transcript.challenge_scalar())
        .collect::<Vec<_>>();

    let mut individual_claims: Vec<F> = spec
        .instances
        .iter()
        .map(|instance| {
            instance
                .input_claim
                .mul_pow_2(max_num_vars - instance.num_vars)
        })
        .collect();

    let mut running_claim = individual_claims
        .iter()
        .zip(&batching_coefficients)
        .map(|(claim, coefficient)| *claim * *coefficient)
        .sum();
    let mut challenges = Vec::with_capacity(num_rounds);
    let mut state = backend.start(spec)?;

    for round in 0..num_rounds {
        let active: Vec<usize> = spec
            .instances
            .iter()
            .enumerate()
            .filter(|(_, instance)| instance.is_active_at_round(round))
            .map(|(index, _)| index)
            .collect();
        let active_claims: Vec<F> = active
            .iter()
            .map(|&index| individual_claims[index])
            .collect();
        let backend_polys = if active.is_empty() {
            Vec::new()
        } else {
            backend.round_polynomials(&state, round, &active, &active_claims)?
        };
        if backend_polys.len() != active.len() {
            return Err(ProverError::Backend(
                crate::error::BackendError::RoundPolynomialCountMismatch {
                    expected: active.len(),
                    got: backend_polys.len(),
                },
            ));
        }

        let mut backend_poly_idx = 0;
        let mut round_poly = UnivariatePoly::zero();
        for (instance_index, instance) in spec.instances.iter().enumerate() {
            let instance_poly = if instance.is_active_at_round(round) {
                let poly = backend_polys[backend_poly_idx].clone();
                backend_poly_idx += 1;
                poly
            } else {
                UnivariatePoly::new(vec![individual_claims[instance_index] * two_inv])
            };
            round_poly += &(&instance_poly * batching_coefficients[instance_index]);
        }

        let round_sum = round_poly.evaluate(F::zero()) + round_poly.evaluate(F::one());
        if round_sum != running_claim {
            return Err(ProverError::RoundCheckFailed {
                round,
                expected: running_claim,
                got: round_sum,
            });
        }

        let challenge = recorder.absorb_round(&round_poly, transcript)?;
        running_claim = round_poly.evaluate(challenge);
        challenges.push(challenge);

        for (instance_index, instance) in spec.instances.iter().enumerate() {
            if instance.is_active_at_round(round) {
                let active_position = active.iter().position(|&idx| idx == instance_index).ok_or(
                    ProverError::Backend(BackendError::InvalidActiveIndex {
                        index: instance_index,
                        batch_size: spec.instances.len(),
                    }),
                )?;
                individual_claims[instance_index] =
                    backend_polys[active_position].evaluate(challenge);
                backend.bind(&mut state, round, instance_index, challenge)?;
            } else {
                individual_claims[instance_index] *= two_inv;
            }
        }
    }

    let _final_outputs = backend.finish(state)?;
    let proof = recorder.finish();

    Ok(BatchedSumcheckProveResult {
        proof,
        batching_coefficients,
        challenges,
        instance_final_claims: individual_claims,
        batched_claim: running_claim,
    })
}

/// Convenience: prove then verify through the existing modular verifier engine.
pub fn prove_and_verify_compressed<F, B, T, R>(
    spec: &BatchedSumcheckSpec<F>,
    backend: &mut B,
    prover_transcript: &mut T,
    verifier_transcript: &mut T,
    recorder: &mut R,
) -> Result<BatchedEvaluationClaim<F>, ProverError<F>>
where
    F: Field + MulPow2 + Invertible,
    B: SumcheckBackend<F>,
    T: Transcript<Challenge = F>,
    R: SumcheckProofRecorder<F, Proof = CompressedSumcheckProof<F>>,
{
    let result = prove_sumcheck(spec, backend, prover_transcript, recorder)?;
    let claims: Vec<SumcheckClaim<F>> = spec
        .instances
        .iter()
        .map(|instance| {
            SumcheckClaim::new(instance.num_vars, instance.degree, instance.input_claim)
        })
        .collect();
    let verified =
        BatchedSumcheckVerifier::verify_compressed(&claims, &result.proof, verifier_transcript)?;
    Ok(verified)
}
