#![expect(clippy::unwrap_used, reason = "tests may panic on assertion failures")]

use jolt_field::{Ext2, Field, FpExt4, Prime32Offset99, Prime64Offset59, Ring};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{
    prove_batch, prove_uniskip_clear, BatchMember, BatchPrelude, BooleanHypercube,
    CenteredIntegerDomain, ClearSumcheckRecorder, CompressedSumcheckProof, ProveRounds,
    SequentialRounds, SumcheckClaim, SumcheckError, SumcheckProof, SumcheckRecorder,
    SumcheckVerifier,
};
use jolt_transcript::{
    prover_transcript, verifier_transcript, Blake2b512, ChallengeCodec, Encoding, FieldTranscript,
    Transcript,
};
use serde::{Deserialize, Serialize};

/// A field the challenge codecs cover, absorbed through its encoding.
trait ChallengeField: Field + Encoding + ChallengeCodec + 'static {}

impl<F: Field + Encoding + ChallengeCodec + 'static> ChallengeField for F {}

type Quadratic = Ext2<Prime64Offset59>;
type Quartic = FpExt4<Prime32Offset99>;

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

fn run_batched_roundtrip<F>(left_coefficients: [F; 3], right_coefficients: [F; 3])
where
    F: ChallengeField + Serialize + for<'de> Deserialize<'de>,
{
    let left = affine_table(left_coefficients);
    let right = affine_table(right_coefficients);
    let input_claim: F = left.iter().zip(&right).map(|(&a, &b)| a * b).sum();

    let mut prover_state =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"extension-sumcheck"), b"");
    let mut recorder = ClearSumcheckRecorder::<F>::new();
    recorder.absorb_input_claims(&[input_claim], &mut prover_state);
    let coefficient = prover_state.challenge().unwrap();
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
        &mut prover_state,
    )
    .unwrap();
    let recorded = recorder
        .finish(&proved.member_claims, &mut prover_state)
        .unwrap();
    let proof = recorded.proof;

    let wire = bincode::serde::encode_to_vec(&proof, bincode::config::standard()).unwrap();
    let (proof, read): (CompressedSumcheckProof<F>, usize) =
        bincode::serde::decode_from_slice(&wire, bincode::config::standard()).unwrap();
    assert_eq!(read, wire.len());

    let mut verifier_transcript = verifier_transcript::<Blake2b512>(
        &jolt_transcript::protocol_tag(b"extension-sumcheck"),
        b"",
        &[],
    );
    verifier_transcript.append(&input_claim);
    let verifier_coefficient: F = verifier_transcript.challenge().unwrap();
    assert_eq!(verifier_coefficient, coefficient);
    let reduced = SumcheckVerifier::verify_compressed(
        &SumcheckClaim::new(2, 2, coefficient * input_claim),
        &proof,
        BooleanHypercube,
        &mut verifier_transcript,
    )
    .unwrap();
    for claim in &proved.member_claims {
        verifier_transcript.append(claim);
    }

    let point: [F; 2] = reduced.point.as_slice().try_into().unwrap();
    let expected = affine(left_coefficients, point) * affine(right_coefficients, point);
    assert_eq!(proved.member_claims, vec![expected]);
    assert_eq!(reduced.value, coefficient * expected);
    assert_eq!(proved.final_claim, reduced.value);
    assert_eq!(
        prover_state.verifier_message::<[u8; 32]>(),
        verifier_transcript.verifier_message::<[u8; 32]>()
    );
}

fn run_uniskip_roundtrip<F>(coefficients: [F; 3])
where
    F: ChallengeField,
{
    let polynomial = UnivariatePoly::new(coefficients.to_vec());
    let input_claim = [-1, 0, 1]
        .into_iter()
        .map(|point| polynomial.evaluate(F::from_i64(point)))
        .sum();
    let mut prover_state =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"extension-uniskip"), b"");
    let proved =
        prove_uniskip_clear::<F, _>(polynomial.clone(), input_claim, 2, 3, &mut prover_state)
            .unwrap();

    let mut verifier_transcript = verifier_transcript::<Blake2b512>(
        &jolt_transcript::protocol_tag(b"extension-uniskip"),
        b"",
        &[],
    );
    let reduced = SumcheckProof::<F, ()>::from(proved.proof)
        .verify(
            &SumcheckClaim::new(1, 2, input_claim),
            CenteredIntegerDomain::new(3),
            &mut verifier_transcript,
        )
        .unwrap();
    verifier_transcript.append(&reduced.value);

    assert_eq!(reduced.point.as_slice(), &[proved.challenge]);
    assert_eq!(reduced.value, polynomial.evaluate(proved.challenge));
    assert_eq!(reduced.value, proved.output_claim);
    assert_eq!(
        prover_state.verifier_message::<[u8; 32]>(),
        verifier_transcript.verifier_message::<[u8; 32]>()
    );
}

#[test]
fn quadratic_encoding_is_the_canonical_little_endian_bytes() {
    let value = Quadratic::new(Prime64Offset59::from_u64(1), Prime64Offset59::from_u64(2));
    assert_eq!(
        value.encode().as_ref(),
        [1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]
    );
}

#[test]
fn quartic_extension_runs_prover_and_verifier() {
    let value = |offset| {
        Quartic::new(std::array::from_fn(|index| {
            Prime32Offset99::from_u64(offset + index as u64 + 1)
        }))
    };
    run_batched_roundtrip(
        [value(0), value(4), value(8)],
        [value(12), value(16), value(20)],
    );
    run_uniskip_roundtrip([value(24), value(28), value(32)]);
}
