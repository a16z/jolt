//! Dense multilinear reference backend: materializes hypercube tables from spec
//! bindings and folds them round-by-round. Correct by construction; use for
//! equivalence tests against CPU and GPU algebra.

use std::marker::PhantomData;

use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_transcript::Transcript;

use crate::backend::SumcheckBackend;
use crate::error::{BackendError, ProverError};
use crate::handler::{prove_sumcheck, BatchedSumcheckProveResult};
use crate::recorder::SumcheckProofRecorder;
use crate::spec::{BatchedSumcheckSpec, SumcheckInstance, WitnessBinding};

/// Reference sumcheck oracle: reads per-instance [`WitnessBinding`] from the spec.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReferenceSumcheckBackend<F: Field> {
    _marker: PhantomData<F>,
}

impl<F: Field> ReferenceSumcheckBackend<F> {
    pub fn new() -> Self {
        Self::default()
    }
}

pub struct ReferenceSumcheckState<F: Field> {
    buffers: Vec<Vec<F>>,
}

impl<F: Field> SumcheckBackend<F> for ReferenceSumcheckBackend<F> {
    type State = ReferenceSumcheckState<F>;

    fn start(&mut self, spec: &BatchedSumcheckSpec<F>) -> Result<Self::State, BackendError> {
        let buffers = spec
            .instances
            .iter()
            .map(|instance| materialize_dense_buffer(instance))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ReferenceSumcheckState { buffers })
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

/// Prove using witness bindings embedded in `spec.instances`.
pub fn prove_reference<F, T, R>(
    spec: &BatchedSumcheckSpec<F>,
    transcript: &mut T,
    recorder: &mut R,
) -> Result<BatchedSumcheckProveResult<F>, ProverError<F>>
where
    F: Field + jolt_field::Invertible + jolt_field::MulPow2,
    T: Transcript<Challenge = F>,
    R: SumcheckProofRecorder<F, Proof = jolt_sumcheck::CompressedSumcheckProof<F>>,
{
    let mut backend = ReferenceSumcheckBackend::new();
    prove_sumcheck(spec, &mut backend, transcript, recorder)
}

fn materialize_dense_buffer<F: Field>(
    instance: &SumcheckInstance<F>,
) -> Result<Vec<F>, BackendError> {
    match &instance.bindings {
        WitnessBinding::DenseMultilinear(evals) => {
            validate_dense_binding(instance, evals)?;
            Ok(evals.clone())
        }
        WitnessBinding::None => Err(BackendError::MissingBinding {
            label: instance.label,
        }),
    }
}

pub(crate) fn validate_dense_binding<F: Field>(
    instance: &SumcheckInstance<F>,
    evals: &[F],
) -> Result<(), BackendError> {
    let expected =
        1usize
            .checked_shl(instance.num_vars as u32)
            .ok_or(BackendError::UnsupportedRelation {
                label: instance.label,
            })?;
    if evals.len() != expected {
        return Err(BackendError::BindingLengthMismatch {
            label: instance.label,
            num_vars: instance.num_vars,
            expected,
            got: evals.len(),
        });
    }
    let binding_sum: F = evals.iter().copied().sum();
    if binding_sum != instance.input_claim {
        return Err(BackendError::BindingClaimMismatch {
            label: instance.label,
        });
    }
    Ok(())
}

pub(crate) fn round_polynomial_from_buffer<F: Field>(
    buffer: &[F],
) -> Result<UnivariatePoly<F>, BackendError> {
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

pub(crate) fn bind_buffer_high_to_low<F: Field>(buffer: &mut Vec<F>, challenge: F) {
    let half = buffer.len() / 2;
    for index in 0..half {
        buffer[index] = buffer[index] + challenge * (buffer[index + half] - buffer[index]);
    }
    buffer.truncate(half);
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests may unwrap on assertion failures")]

    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_sumcheck::SumcheckClaim;
    use jolt_transcript::{Blake2bTranscript, Transcript};

    use crate::handler::prove_sumcheck;
    use crate::recorder::ClearCompressedRecorder;
    use crate::spec::{BatchedSumcheckSpec, RoundOffset, SumcheckInstance};

    use super::*;

    fn compute_sum(evals: &[Fr]) -> Fr {
        evals.iter().copied().sum()
    }

    fn dense_instance(
        label: &'static str,
        evals: Vec<Fr>,
        num_vars: usize,
        alignment: RoundOffset,
    ) -> SumcheckInstance<Fr> {
        let claim = compute_sum(&evals);
        SumcheckInstance::new(label, claim, num_vars, 1, alignment).with_dense_bindings(evals)
    }

    #[test]
    fn single_instance_matches_verifier() {
        let evals: Vec<Fr> = (1..=8).map(Fr::from_u64).collect();
        let spec = BatchedSumcheckSpec::new(
            "single",
            vec![dense_instance("a", evals.clone(), 3, RoundOffset::ZERO)],
        );

        let mut backend = ReferenceSumcheckBackend::new();
        let mut prover_transcript = Blake2bTranscript::new(b"sumcheck-test");
        let mut verifier_transcript = Blake2bTranscript::new(b"sumcheck-test");
        let mut recorder = ClearCompressedRecorder::new();

        let result =
            prove_sumcheck(&spec, &mut backend, &mut prover_transcript, &mut recorder).unwrap();
        let claims = vec![SumcheckClaim::new(3, 1, compute_sum(&evals))];
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
                dense_instance("a", evals_a, 3, RoundOffset::ZERO),
                dense_instance("b", evals_b, 2, RoundOffset::new(1)),
            ],
        );

        let mut backend = ReferenceSumcheckBackend::new();
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
    fn prove_reference_matches_prove_sumcheck() {
        let evals: Vec<Fr> = (1..=8).map(Fr::from_u64).collect();
        let spec = BatchedSumcheckSpec::new(
            "single",
            vec![dense_instance("a", evals, 3, RoundOffset::ZERO)],
        );

        let mut transcript_a = Blake2bTranscript::new(b"sumcheck-test");
        let mut transcript_b = Blake2bTranscript::new(b"sumcheck-test");
        let mut recorder_a = ClearCompressedRecorder::new();
        let mut recorder_b = ClearCompressedRecorder::new();

        let via_trait = {
            let mut backend = ReferenceSumcheckBackend::new();
            prove_sumcheck(&spec, &mut backend, &mut transcript_a, &mut recorder_a).unwrap()
        };
        let via_helper = prove_reference(&spec, &mut transcript_b, &mut recorder_b).unwrap();

        assert_eq!(via_trait.proof, via_helper.proof);
        assert_eq!(via_trait.challenges, via_helper.challenges);
        assert_eq!(
            via_trait.batching_coefficients,
            via_helper.batching_coefficients
        );
    }

    #[test]
    fn missing_binding_is_rejected() {
        let spec = BatchedSumcheckSpec::new(
            "missing",
            vec![SumcheckInstance::new(
                "a",
                Fr::from_u64(10),
                3,
                1,
                RoundOffset::ZERO,
            )],
        );
        let mut backend = ReferenceSumcheckBackend::new();
        let mut transcript = Blake2bTranscript::new(b"sumcheck-test");
        let mut recorder = ClearCompressedRecorder::new();
        let err = prove_sumcheck(&spec, &mut backend, &mut transcript, &mut recorder).unwrap_err();
        assert!(matches!(
            err,
            ProverError::Backend(BackendError::MissingBinding { label: "a" })
        ));
    }

    #[test]
    fn binding_claim_mismatch_is_rejected() {
        let evals: Vec<Fr> = (1..=8).map(Fr::from_u64).collect();
        let spec = BatchedSumcheckSpec::new(
            "mismatch",
            vec![
                SumcheckInstance::new("a", Fr::from_u64(0), 3, 1, RoundOffset::ZERO)
                    .with_dense_bindings(evals),
            ],
        );
        let mut backend = ReferenceSumcheckBackend::new();
        let mut transcript = Blake2bTranscript::new(b"sumcheck-test");
        let mut recorder = ClearCompressedRecorder::new();
        let err = prove_sumcheck(&spec, &mut backend, &mut transcript, &mut recorder).unwrap_err();
        assert!(matches!(
            err,
            ProverError::Backend(BackendError::BindingClaimMismatch { label: "a", .. })
        ));
    }

    #[test]
    fn binding_length_mismatch_is_rejected() {
        let evals: Vec<Fr> = (1..=4).map(Fr::from_u64).collect();
        let spec = BatchedSumcheckSpec::new(
            "short",
            vec![dense_instance("a", evals, 3, RoundOffset::ZERO)],
        );
        let mut backend = ReferenceSumcheckBackend::new();
        let mut transcript = Blake2bTranscript::new(b"sumcheck-test");
        let mut recorder = ClearCompressedRecorder::new();
        let err = prove_sumcheck(&spec, &mut backend, &mut transcript, &mut recorder).unwrap_err();
        assert!(matches!(
            err,
            ProverError::Backend(BackendError::BindingLengthMismatch {
                label: "a",
                num_vars: 3,
                expected: 8,
                got: 4,
            })
        ));
    }

    #[test]
    fn empty_batch_is_rejected() {
        let spec = BatchedSumcheckSpec::<Fr>::new("empty", vec![]);
        let mut backend = ReferenceSumcheckBackend::new();
        let mut transcript = Blake2bTranscript::new(b"sumcheck-test");
        let mut recorder = ClearCompressedRecorder::new();
        let err = prove_sumcheck(&spec, &mut backend, &mut transcript, &mut recorder).unwrap_err();
        assert!(matches!(err, crate::error::ProverError::EmptyBatch));
    }

    #[test]
    fn round_polynomial_matches_hypercube_fold() {
        let evals: Vec<Fr> = (1..=8).map(Fr::from_u64).collect();
        let poly = round_polynomial_from_buffer(&evals).unwrap();
        let expected_0: Fr = evals[..4].iter().copied().sum();
        let expected_1: Fr = evals[4..].iter().copied().sum();
        assert_eq!(poly.evaluate(Fr::from_u64(0)), expected_0);
        assert_eq!(poly.evaluate(Fr::from_u64(1)), expected_1);
        assert_eq!(
            poly.evaluate(Fr::from_u64(0)) + poly.evaluate(Fr::from_u64(1)),
            compute_sum(&evals)
        );
    }
}
