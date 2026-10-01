//! The stock clear prover and verifier over non-BN254 prime fields, through
//! the NARG transcript.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests may panic on assertion failures and index fixture data"
)]

use jolt_field::{CanonicalEncoding, Field, Prime128Offset275, Prime32Offset99, Prime64Offset59};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{
    prove_batch, prove_uniskip_clear, BatchMember, BatchPrelude, CenteredIntegerDomain,
    ClearSumcheckRecorder, ProveRounds, SequentialRounds, SumcheckClaim, SumcheckError,
    SumcheckRecorder, SumcheckVerifier,
};
use jolt_transcript::{Blake2b512, Channel, ProtocolId, ProverTranscript, VerifierTranscript};

const PROTOCOL: ProtocolId = ProtocolId::new::<Blake2b512>("jolt-sumcheck/tests/fields");

struct ProductOfAffines<F: Field> {
    num_rounds: usize,
    left: Vec<F>,
    right: Vec<F>,
}

impl<F: Field> ProductOfAffines<F> {
    fn new(left: Vec<F>, right: Vec<F>) -> Self {
        assert_eq!(left.len(), right.len());
        assert!(left.len().is_power_of_two());
        Self {
            num_rounds: left.len().ilog2() as usize,
            left,
            right,
        }
    }

    fn bind(table: &mut Vec<F>, challenge: F) {
        for index in 0..table.len() / 2 {
            let low = table[2 * index];
            let high = table[2 * index + 1];
            table[index] = low + challenge * (high - low);
        }
        table.truncate(table.len() / 2);
    }
}

impl<F: Field> ProveRounds<F> for ProductOfAffines<F> {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            Self::bind(&mut self.left, challenge);
            Self::bind(&mut self.right, challenge);
        }

        let mut coefficients = [F::zero(); 3];
        for (left, right) in self.left.chunks_exact(2).zip(self.right.chunks_exact(2)) {
            let left_delta = left[1] - left[0];
            let right_delta = right[1] - right[0];
            coefficients[0] += left[0] * right[0];
            coefficients[1] += left[0] * right_delta + left_delta * right[0];
            coefficients[2] += left_delta * right_delta;
        }
        let polynomial = UnivariatePoly::new(coefficients.to_vec());
        assert_eq!(
            polynomial.evaluate(F::zero()) + polynomial.evaluate(F::one()),
            previous_claim
        );
        Ok(polynomial)
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        Self::bind(&mut self.left, bind);
        Self::bind(&mut self.right, bind);
        Ok(())
    }
}

/// `c0 + c1 x0 + c2 x1`; the member binds `x0` (the low bit) first.
fn affine<F: Field>(coefficients: [F; 3], point: [F; 2]) -> F {
    coefficients[0] + coefficients[1] * point[0] + coefficients[2] * point[1]
}

fn affine_table<F: Field>(coefficients: [F; 3]) -> Vec<F> {
    [
        [F::zero(), F::zero()],
        [F::one(), F::zero()],
        [F::zero(), F::one()],
        [F::one(), F::one()],
    ]
    .into_iter()
    .map(|point| affine(coefficients, point))
    .collect()
}

fn run_batched_roundtrip<F: Field + CanonicalEncoding>(
    left_coefficients: [F; 3],
    right_coefficients: [F; 3],
) {
    let left = affine_table(left_coefficients);
    let right = affine_table(right_coefficients);
    let input_claim: F = left.iter().zip(&right).map(|(&a, &b)| a * b).sum();

    let mut prover_transcript = ProverTranscript::<Blake2b512>::new(&PROTOCOL, b"batched");
    let mut recorder = ClearSumcheckRecorder::<F>::new();
    recorder.absorb_input_claims(&[input_claim], &mut prover_transcript);
    let coefficient: F = prover_transcript.challenge_small();
    let prelude = BatchPrelude::try_new(
        vec![BatchMember {
            input_claim,
            coefficient,
            rounds: 2,
            offset: 0,
        }],
        2,
        2,
    )
    .unwrap();
    let mut product = ProductOfAffines::new(left, right);
    let mut members: [&mut dyn ProveRounds<F>; 1] = [&mut product];
    let proved = prove_batch(
        &prelude,
        &mut members,
        &mut SequentialRounds,
        &mut recorder,
        &mut prover_transcript,
    )
    .unwrap();
    recorder
        .finish(&proved.member_claims, &mut prover_transcript)
        .unwrap();
    let prover_state: [u8; 32] = prover_transcript.preview().squeeze();
    let narg = prover_transcript.finish();
    assert_eq!(narg.len(), (2 * 2 + 1) * F::NUM_BYTES);

    let mut verifier_transcript =
        VerifierTranscript::<Blake2b512>::new(&PROTOCOL, b"batched", &narg);
    verifier_transcript.public(&input_claim);
    let verifier_coefficient: F = verifier_transcript.challenge_small();
    assert_eq!(verifier_coefficient, coefficient);
    let reduced = SumcheckVerifier::verify_compressed(
        &SumcheckClaim::new(2, 2, coefficient * input_claim),
        &mut verifier_transcript,
    )
    .unwrap();
    let member_claim: F = verifier_transcript.receive().unwrap();

    let point: [F; 2] = reduced.point.as_slice().try_into().unwrap();
    let expected = affine(left_coefficients, point) * affine(right_coefficients, point);
    assert_eq!(member_claim, expected);
    assert_eq!(proved.member_claims, vec![expected]);
    assert_eq!(reduced.value, coefficient * expected);
    assert_eq!(proved.final_claim, reduced.value);
    assert_eq!(verifier_transcript.preview().squeeze::<32>(), prover_state);
    verifier_transcript.finish().unwrap();
}

fn run_uniskip_roundtrip<F: Field + CanonicalEncoding>(coefficients: [F; 3]) {
    let polynomial = UnivariatePoly::new(coefficients.to_vec());
    let input_claim = [-1, 0, 1]
        .into_iter()
        .map(|point| polynomial.evaluate(F::from_i64(point)))
        .sum();
    let mut prover_transcript = ProverTranscript::<Blake2b512>::new(&PROTOCOL, b"uniskip");
    let proved = prove_uniskip_clear(
        polynomial.clone(),
        input_claim,
        2,
        3,
        &mut prover_transcript,
    )
    .unwrap();
    let prover_state: [u8; 32] = prover_transcript.preview().squeeze();
    let narg = prover_transcript.finish();

    let mut verifier_transcript =
        VerifierTranscript::<Blake2b512>::new(&PROTOCOL, b"uniskip", &narg);
    let reduced = SumcheckVerifier::verify(
        &SumcheckClaim::new(1, 2, input_claim),
        CenteredIntegerDomain::new(3),
        &mut verifier_transcript,
    )
    .unwrap();
    let output_claim: F = verifier_transcript.receive().unwrap();

    assert_eq!(reduced.point.as_slice(), &[proved.challenge]);
    assert_eq!(reduced.value, polynomial.evaluate(proved.challenge));
    assert_eq!(output_claim, proved.output_claim);
    assert_eq!(reduced.value, proved.output_claim);
    assert_eq!(verifier_transcript.preview().squeeze::<32>(), prover_state);
    verifier_transcript.finish().unwrap();
}

fn run_roundtrips<F: Field + CanonicalEncoding>() {
    let value = |v: u64| F::from_u64(v);
    run_batched_roundtrip(
        [value(1), value(3), value(5)],
        [value(7), value(9), value(11)],
    );
    run_uniskip_roundtrip([value(2), value(5), value(11)]);
}

#[test]
fn prime32_runs_prover_and_verifier() {
    run_roundtrips::<Prime32Offset99>();
}

#[test]
fn prime64_runs_prover_and_verifier() {
    run_roundtrips::<Prime64Offset59>();
}

#[test]
fn prime128_runs_prover_and_verifier() {
    run_roundtrips::<Prime128Offset275>();
}
