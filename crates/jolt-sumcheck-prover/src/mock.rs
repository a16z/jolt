#![cfg(test)]

use jolt_field::{Field, FromPrimitiveInt};
use jolt_poly::UnivariatePoly;

use crate::backend::SumcheckBackend;
use crate::error::BackendError;
use crate::spec::BatchedSumcheckSpec;

/// Dense multilinear reference backend: folds hypercube evaluations round-by-round.
pub struct DenseMultilinearBackend<F: Field> {
    evals: Vec<Vec<F>>,
}

impl<F: Field> DenseMultilinearBackend<F> {
    pub fn new(evals: Vec<Vec<F>>) -> Self {
        Self { evals }
    }
}

pub struct DenseMultilinearState<F: Field> {
    buffers: Vec<Vec<F>>,
}

impl<F: Field> SumcheckBackend<F> for DenseMultilinearBackend<F> {
    type State = DenseMultilinearState<F>;

    fn start(&mut self, spec: &BatchedSumcheckSpec<F>) -> Result<Self::State, BackendError> {
        if self.evals.len() != spec.instances.len() {
            return Err(BackendError::RoundPolynomialCountMismatch {
                expected: spec.instances.len(),
                got: self.evals.len(),
            });
        }
        Ok(DenseMultilinearState {
            buffers: self.evals.clone(),
        })
    }

    fn round_polynomials(
        &mut self,
        state: &Self::State,
        _round: usize,
        active: &[usize],
        _claims: &[F],
    ) -> Result<Vec<UnivariatePoly<F>>, BackendError> {
        active
            .iter()
            .map(|&instance| round_polynomial_from_buffer(&state.buffers[instance]))
            .collect()
    }

    fn bind(
        &mut self,
        state: &mut Self::State,
        _round: usize,
        instance: usize,
        challenge: F,
    ) -> Result<(), BackendError> {
        bind_buffer_high_to_low(&mut state.buffers[instance], challenge);
        Ok(())
    }
}

fn round_polynomial_from_buffer<F: Field>(buffer: &[F]) -> Result<UnivariatePoly<F>, BackendError> {
    let half = buffer.len().checked_div(2).filter(|&half| half > 0).ok_or(
        BackendError::UnsupportedRelation {
            label: "dense_multilinear",
        },
    )?;

    let mut eval_0 = F::zero();
    let mut eval_1 = F::zero();
    for index in 0..half {
        eval_0 += buffer[index];
        eval_1 += buffer[index + half];
    }

    Ok(UnivariatePoly::new(vec![eval_0, eval_1 - eval_0]))
}

fn bind_buffer_high_to_low<F: Field>(buffer: &mut Vec<F>, challenge: F) {
    let half = buffer.len() / 2;
    for index in 0..half {
        buffer[index] = buffer[index] + challenge * (buffer[index + half] - buffer[index]);
    }
    buffer.truncate(half);
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests may unwrap on assertion failures")]

    use jolt_field::Fr;
    use jolt_sumcheck::SumcheckClaim;
    use jolt_transcript::{Blake2bTranscript, Transcript};

    use crate::handler::prove_sumcheck;
    use crate::recorder::ClearCompressedRecorder;
    use crate::spec::{BatchedSumcheckSpec, RoundOffset, SumcheckInstance};

    use super::*;

    fn compute_sum(evals: &[Fr]) -> Fr {
        evals.iter().copied().sum()
    }

    #[test]
    fn single_instance_matches_verifier() {
        let evals: Vec<Fr> = (1..=8).map(Fr::from_u64).collect();
        let claim = compute_sum(&evals);
        let spec = BatchedSumcheckSpec::new(
            "single",
            vec![SumcheckInstance::new("a", claim, 3, 1, RoundOffset::ZERO)],
        );

        let mut backend = DenseMultilinearBackend::new(vec![evals]);
        let mut prover_transcript = Blake2bTranscript::new(b"sumcheck-test");
        let mut verifier_transcript = Blake2bTranscript::new(b"sumcheck-test");
        let mut recorder = ClearCompressedRecorder::new();

        let result =
            prove_sumcheck(&spec, &mut backend, &mut prover_transcript, &mut recorder).unwrap();
        let claims = vec![SumcheckClaim::new(3, 1, claim)];
        let verified = jolt_sumcheck::BatchedSumcheckVerifier::verify_compressed(
            &claims,
            &result.proof,
            &mut verifier_transcript,
        )
        .unwrap();

        assert_eq!(verified.batching_coefficients.len(), 1);
        assert_eq!(verified.max_num_vars, 3);
        assert_eq!(result.challenges.len(), 3);
        assert_eq!(verified.reduction.point.len(), 3);
    }

    #[test]
    fn front_loaded_batch_matches_verifier() {
        let evals_a: Vec<Fr> = (1..=8).map(Fr::from_u64).collect();
        let evals_b: Vec<Fr> = (1..=4).map(Fr::from_u64).collect();
        let claims = vec![
            SumcheckClaim::new(3, 1, compute_sum(&evals_a)),
            SumcheckClaim::new(2, 1, compute_sum(&evals_b)),
        ];
        let spec = BatchedSumcheckSpec::new(
            "front_loaded",
            vec![
                SumcheckInstance::new("a", claims[0].claimed_sum, 3, 1, RoundOffset::ZERO),
                SumcheckInstance::new("b", claims[1].claimed_sum, 2, 1, RoundOffset::new(1)),
            ],
        );

        let mut backend = DenseMultilinearBackend::new(vec![evals_a, evals_b]);
        let mut prover_transcript = Blake2bTranscript::new(b"sumcheck-test");
        let mut verifier_transcript = Blake2bTranscript::new(b"sumcheck-test");
        let mut recorder = ClearCompressedRecorder::new();

        let result =
            prove_sumcheck(&spec, &mut backend, &mut prover_transcript, &mut recorder).unwrap();
        let verified = jolt_sumcheck::BatchedSumcheckVerifier::verify_compressed(
            &claims,
            &result.proof,
            &mut verifier_transcript,
        )
        .unwrap();

        assert_eq!(verified.batching_coefficients, result.batching_coefficients);
        assert_eq!(verified.max_num_vars, 3);
        assert_eq!(verified.instance_point(2), &verified.reduction.point[1..]);
    }

    #[test]
    fn empty_batch_is_rejected() {
        let spec = BatchedSumcheckSpec::<Fr>::new("empty", vec![]);
        let mut backend = DenseMultilinearBackend::new(vec![]);
        let mut transcript = Blake2bTranscript::new(b"sumcheck-test");
        let mut recorder = ClearCompressedRecorder::new();
        let err = prove_sumcheck(&spec, &mut backend, &mut transcript, &mut recorder).unwrap_err();
        assert!(matches!(err, crate::error::ProverError::EmptyBatch));
    }
}
