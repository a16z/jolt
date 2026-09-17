#![expect(
    clippy::panic,
    clippy::unwrap_used,
    reason = "tests may panic on assertion failures"
)]

use std::marker::PhantomData;

use jolt_field::{Ext2, Field, FpExt4, Prime128Offset275, Prime32Offset99, Prime64Offset59, Ring};
use jolt_poly::{CompressedPoly, UnivariatePoly};
use jolt_sumcheck::{
    prove_batch, prove_uniskip_clear, BatchMember, BatchPrelude, BooleanHypercube,
    CenteredIntegerDomain, ClearProof, ClearSumcheckRecorder, CompressedSumcheckProof, ProveRounds,
    SequentialRounds, SumcheckClaim, SumcheckError, SumcheckProof, SumcheckRecorder,
    SumcheckVerifier, OPENING_CLAIM_TRANSCRIPT_LABEL, SUMCHECK_CLAIM_TRANSCRIPT_LABEL,
    SUMCHECK_ROUND_TRANSCRIPT_LABEL, UNISKIP_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::{AppendToTranscript, Transcript};
use num_traits::{One, Zero};

trait ChallengeField: Field + AppendToTranscript + 'static {
    fn challenge_from_seed(seed: u64) -> Self;
}

type Quadratic = Ext2<Prime64Offset59>;
type Quartic = FpExt4<Prime32Offset99>;

impl ChallengeField for Quadratic {
    fn challenge_from_seed(seed: u64) -> Self {
        Self::new(
            Prime64Offset59::from_u64(seed + 1),
            Prime64Offset59::from_u64(seed + 11),
        )
    }
}

impl ChallengeField for Quartic {
    fn challenge_from_seed(seed: u64) -> Self {
        Self::new(std::array::from_fn(|index| {
            Prime32Offset99::from_u64(seed + 3 * index as u64 + 1)
        }))
    }
}

impl ChallengeField for Prime128Offset275 {
    fn challenge_from_seed(seed: u64) -> Self {
        Self::from_u64(seed + 1)
    }
}

struct ExtensionTranscript<F> {
    state: [u8; 32],
    cursor: usize,
    challenge_index: u64,
    _field: PhantomData<F>,
}

impl<F> Default for ExtensionTranscript<F> {
    fn default() -> Self {
        Self {
            state: [0; 32],
            cursor: 0,
            challenge_index: 0,
            _field: PhantomData,
        }
    }
}

impl<F: ChallengeField> Transcript for ExtensionTranscript<F> {
    type Challenge = F;

    fn new(label: &'static [u8]) -> Self {
        let mut transcript = Self::default();
        transcript.append_bytes(label);
        transcript
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            let index = self.cursor % self.state.len();
            self.state[index] = self.state[index]
                .wrapping_mul(31)
                .wrapping_add(*byte)
                .wrapping_add(1);
            self.cursor += 1;
        }
    }

    fn challenge(&mut self) -> F {
        self.append_bytes(b"extension_challenge");
        let state_seed = u64::from_le_bytes(self.state[..8].try_into().unwrap());
        let seed = state_seed.wrapping_add(self.challenge_index);
        self.challenge_index = self.challenge_index.wrapping_add(1);
        F::challenge_from_seed(seed)
    }

    fn state(&self) -> [u8; 32] {
        self.state
    }
}

#[derive(Default)]
struct CapturingTranscript {
    absorbed: Vec<u8>,
}

impl Transcript for CapturingTranscript {
    type Challenge = Quadratic;

    fn new(label: &'static [u8]) -> Self {
        Self {
            absorbed: label.to_vec(),
        }
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        self.absorbed.extend_from_slice(bytes);
    }

    fn challenge(&mut self) -> Self::Challenge {
        Quadratic::zero()
    }

    fn state(&self) -> [u8; 32] {
        [0; 32]
    }
}

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
    F: ChallengeField + serde::Serialize + for<'de> serde::Deserialize<'de>,
{
    let left = affine_table(left_coefficients);
    let right = affine_table(right_coefficients);
    let input_claim: F = left.iter().zip(&right).map(|(&a, &b)| a * b).sum();

    let mut prover_transcript = ExtensionTranscript::<F>::new(b"extension-sumcheck");
    let mut recorder = ClearSumcheckRecorder::<F>::new();
    recorder.absorb_input_claims(&[input_claim], &mut prover_transcript);
    let coefficient = prover_transcript.challenge_scalar();
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
    let recorded = recorder
        .finish(&proved.member_claims, &mut prover_transcript)
        .unwrap();
    let SumcheckProof::Clear(ClearProof::Compressed(proof)) = recorded.proof else {
        panic!("clear recorder returned a non-compressed proof")
    };

    let wire = bincode::serde::encode_to_vec(&proof, bincode::config::standard()).unwrap();
    let (proof, read): (CompressedSumcheckProof<F>, usize) =
        bincode::serde::decode_from_slice(&wire, bincode::config::standard()).unwrap();
    assert_eq!(read, wire.len());

    let mut verifier_transcript = ExtensionTranscript::<F>::new(b"extension-sumcheck");
    verifier_transcript.append_labeled(SUMCHECK_CLAIM_TRANSCRIPT_LABEL, &input_claim);
    let verifier_coefficient = verifier_transcript.challenge_scalar();
    assert_eq!(verifier_coefficient, coefficient);
    let reduced = SumcheckVerifier::verify_compressed(
        &SumcheckClaim::new(2, 2, coefficient * input_claim),
        &proof,
        BooleanHypercube,
        SUMCHECK_ROUND_TRANSCRIPT_LABEL,
        &mut verifier_transcript,
    )
    .unwrap();
    for claim in &proved.member_claims {
        verifier_transcript.append_labeled(OPENING_CLAIM_TRANSCRIPT_LABEL, claim);
    }

    let point: [F; 2] = reduced.point.as_slice().try_into().unwrap();
    let expected = affine(left_coefficients, point) * affine(right_coefficients, point);
    assert_eq!(proved.member_claims, vec![expected]);
    assert_eq!(reduced.value, coefficient * expected);
    assert_eq!(proved.final_claim, reduced.value);
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
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
    let mut prover_transcript = ExtensionTranscript::<F>::new(b"extension-uniskip");
    let proved = prove_uniskip_clear::<F, (), _>(
        polynomial.clone(),
        input_claim,
        2,
        3,
        &mut prover_transcript,
    )
    .unwrap();

    let mut verifier_transcript = ExtensionTranscript::<F>::new(b"extension-uniskip");
    let reduced = proved
        .proof
        .verify(
            &SumcheckClaim::new(1, 2, input_claim),
            CenteredIntegerDomain::new(3),
            UNISKIP_ROUND_TRANSCRIPT_LABEL,
            &mut verifier_transcript,
        )
        .unwrap();
    verifier_transcript.append_labeled(OPENING_CLAIM_TRANSCRIPT_LABEL, &reduced.value);

    assert_eq!(reduced.point.as_slice(), &[proved.challenge]);
    assert_eq!(reduced.value, polynomial.evaluate(proved.challenge));
    assert_eq!(reduced.value, proved.output_claim);
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
}

#[test]
fn quadratic_extension_runs_prover_and_verifier() {
    let value = |a, b| Quadratic::new(Prime64Offset59::from_u64(a), Prime64Offset59::from_u64(b));
    run_batched_roundtrip(
        [value(1, 2), value(3, 4), value(5, 6)],
        [value(7, 8), value(9, 10), value(11, 12)],
    );
    run_uniskip_roundtrip([value(2, 3), value(5, 7), value(11, 13)]);
}

#[test]
fn quadratic_transcript_absorption_reverses_the_complete_payload() {
    let value = Quadratic::new(Prime64Offset59::from_u64(1), Prime64Offset59::from_u64(2));
    let mut transcript = CapturingTranscript::default();
    value.append_to_transcript(&mut transcript);
    assert_eq!(
        transcript.absorbed,
        [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1]
    );
}

#[test]
fn direct_128_bit_field_runs_prover_and_verifier() {
    let value = Prime128Offset275::from_u64;
    run_batched_roundtrip(
        [value(1), value(3), value(5)],
        [value(7), value(9), value(11)],
    );
    run_uniskip_roundtrip([value(2), value(5), value(11)]);
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

#[test]
fn tampered_quartic_proof_fails_final_evaluation_check() {
    let left_coefficients = [
        Quartic::new(std::array::from_fn(|index| {
            Prime32Offset99::from_u64(index as u64 + 1)
        })),
        Quartic::new(std::array::from_fn(|index| {
            Prime32Offset99::from_u64(index as u64 + 5)
        })),
        Quartic::new(std::array::from_fn(|index| {
            Prime32Offset99::from_u64(index as u64 + 9)
        })),
    ];
    let right_coefficients = [
        Quartic::new(std::array::from_fn(|index| {
            Prime32Offset99::from_u64(index as u64 + 13)
        })),
        Quartic::new(std::array::from_fn(|index| {
            Prime32Offset99::from_u64(index as u64 + 17)
        })),
        Quartic::new(std::array::from_fn(|index| {
            Prime32Offset99::from_u64(index as u64 + 21)
        })),
    ];
    let left = affine_table(left_coefficients);
    let right = affine_table(right_coefficients);
    let input_claim: Quartic = left.iter().zip(&right).map(|(&a, &b)| a * b).sum();

    let mut prover_transcript = ExtensionTranscript::<Quartic>::new(b"tampered-extension");
    let mut recorder = ClearSumcheckRecorder::<Quartic>::new();
    recorder.absorb_input_claims(&[input_claim], &mut prover_transcript);
    let coefficient = prover_transcript.challenge_scalar();
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
    let mut members: [&mut dyn ProveRounds<Quartic>; 1] = [&mut product];
    let proved = prove_batch(
        &prelude,
        &mut members,
        &mut SequentialRounds,
        &mut recorder,
        &mut prover_transcript,
    )
    .unwrap();
    let recorded = recorder
        .finish(&proved.member_claims, &mut prover_transcript)
        .unwrap();
    let SumcheckProof::Clear(ClearProof::Compressed(mut proof)) = recorded.proof else {
        panic!("clear recorder returned a non-compressed proof")
    };
    let mut coefficients = proof.round_polynomials[0]
        .coeffs_except_linear_term()
        .to_vec();
    coefficients[0] += Quartic::new([
        Prime32Offset99::zero(),
        Prime32Offset99::one(),
        Prime32Offset99::zero(),
        Prime32Offset99::zero(),
    ]);
    proof.round_polynomials[0] = CompressedPoly::new(coefficients);

    let mut verifier_transcript = ExtensionTranscript::<Quartic>::new(b"tampered-extension");
    verifier_transcript.append_labeled(SUMCHECK_CLAIM_TRANSCRIPT_LABEL, &input_claim);
    let verifier_coefficient = verifier_transcript.challenge_scalar();
    let reduced = SumcheckVerifier::verify_compressed(
        &SumcheckClaim::new(2, 2, verifier_coefficient * input_claim),
        &proof,
        BooleanHypercube,
        SUMCHECK_ROUND_TRANSCRIPT_LABEL,
        &mut verifier_transcript,
    )
    .unwrap();
    let point: [Quartic; 2] = reduced.point.as_slice().try_into().unwrap();
    let expected =
        verifier_coefficient * affine(left_coefficients, point) * affine(right_coefficients, point);
    assert_ne!(reduced.value, expected);
}
