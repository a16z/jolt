use std::{
    fmt::{Display, Formatter},
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use jolt_field::{AdditiveGroup, Field, Prime64Offset59, Ring};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{
    prove_batch, BatchMember, BatchPrelude, BooleanHypercube, ClearProof,
    ClearSumcheckRecorder, ProveRounds, SequentialRounds, SumcheckClaim, SumcheckError,
    SumcheckProof, SumcheckRecorder, SumcheckVerifier, OPENING_CLAIM_TRANSCRIPT_LABEL,
    SUMCHECK_CLAIM_TRANSCRIPT_LABEL, SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::{AppendToTranscript, Transcript};
use num_traits::{One, Zero};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
struct ExternalField(Prime64Offset59);

impl Display for ExternalField {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, formatter)
    }
}

impl Zero for ExternalField {
    fn zero() -> Self {
        Self(Prime64Offset59::zero())
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl One for ExternalField {
    fn one() -> Self {
        Self(Prime64Offset59::one())
    }

    fn is_one(&self) -> bool {
        self.0.is_one()
    }
}

impl Add for ExternalField {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Add<&Self> for ExternalField {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        self + *rhs
    }
}

impl AddAssign for ExternalField {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for ExternalField {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl Sub<&Self> for ExternalField {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        self - *rhs
    }
}

impl SubAssign for ExternalField {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for ExternalField {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl Mul for ExternalField {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl Mul<&Self> for ExternalField {
    type Output = Self;

    fn mul(self, rhs: &Self) -> Self::Output {
        self * *rhs
    }
}

impl MulAssign for ExternalField {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Sum for ExternalField {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |sum, value| sum + value)
    }
}

impl<'a> Sum<&'a ExternalField> for ExternalField {
    fn sum<I: Iterator<Item = &'a ExternalField>>(iter: I) -> Self {
        iter.copied().sum()
    }
}

impl Product for ExternalField {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |product, value| product * value)
    }
}

impl<'a> Product<&'a ExternalField> for ExternalField {
    fn product<I: Iterator<Item = &'a ExternalField>>(iter: I) -> Self {
        iter.copied().product()
    }
}

impl AdditiveGroup for ExternalField {}

impl Ring for ExternalField {
    fn from_u64(value: u64) -> Self {
        Self(Prime64Offset59::from_u64(value))
    }

    fn from_i64(value: i64) -> Self {
        Self(Prime64Offset59::from_i64(value))
    }

    fn from_u128(value: u128) -> Self {
        Self(Prime64Offset59::from_u128(value))
    }

    fn from_i128(value: i128) -> Self {
        Self(Prime64Offset59::from_i128(value))
    }
}

impl Field for ExternalField {
    fn inverse(&self) -> Option<Self> {
        self.0.inverse().map(Self)
    }

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        Self(Prime64Offset59::random(rng))
    }
}

impl AppendToTranscript for ExternalField {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        self.0.append_to_transcript(transcript);
    }
}

#[derive(Default)]
struct ExternalTranscript {
    state: [u8; 32],
    cursor: usize,
}

impl Transcript for ExternalTranscript {
    type Challenge = ExternalField;

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

    fn challenge(&mut self) -> ExternalField {
        self.append_bytes(b"challenge");
        ExternalField::from_u64(u64::from_le_bytes(self.state[..8].try_into().unwrap()))
    }

    fn state(&self) -> [u8; 32] {
        self.state
    }
}

struct LinearRound;

impl ProveRounds<ExternalField> for LinearRound {
    fn num_rounds(&self) -> usize {
        1
    }

    fn prove_round(
        &mut self,
        bind: Option<ExternalField>,
        round: usize,
        previous_claim: ExternalField,
    ) -> Result<UnivariatePoly<ExternalField>, SumcheckError<ExternalField>> {
        assert!(bind.is_none());
        assert_eq!(round, 0);
        let polynomial = field_only_polynomial([
            ExternalField::from_u64(3),
            ExternalField::from_u64(2),
        ]);
        assert_eq!(
            polynomial.evaluate(ExternalField::zero())
                + polynomial.evaluate(ExternalField::one()),
            previous_claim
        );
        Ok(polynomial)
    }

    fn finish_rounds(
        &mut self,
        _bind: ExternalField,
    ) -> Result<(), SumcheckError<ExternalField>> {
        Ok(())
    }
}

fn field_only_polynomial<F: Field>(coefficients: [F; 2]) -> UnivariatePoly<F> {
    let polynomial = UnivariatePoly::new(coefficients.to_vec());
    let _compressed = polynomial.compress();
    polynomial
}

#[test]
fn external_field_runs_stock_clear_prover_and_verifier() {
    let input_claim = ExternalField::from_u64(8);
    let mut prover_transcript = ExternalTranscript::new(b"external-field");
    let mut recorder = ClearSumcheckRecorder::<ExternalField>::new();
    recorder.absorb_input_claims(&[input_claim], &mut prover_transcript);
    let coefficient = prover_transcript.challenge_scalar();
    let prelude = BatchPrelude::try_new(
        vec![BatchMember {
            input_claim,
            coefficient,
            rounds: 1,
            offset: 0,
        }],
        1,
        1,
    )
    .unwrap();
    let mut round = LinearRound;
    let mut members: [&mut dyn ProveRounds<ExternalField>; 1] = [&mut round];
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

    let mut verifier_transcript = ExternalTranscript::new(b"external-field");
    verifier_transcript.append_labeled(SUMCHECK_CLAIM_TRANSCRIPT_LABEL, &input_claim);
    let verifier_coefficient = verifier_transcript.challenge_scalar();
    let reduced = SumcheckVerifier::verify_compressed(
        &SumcheckClaim::new(1, 1, verifier_coefficient * input_claim),
        &proof,
        BooleanHypercube,
        SUMCHECK_ROUND_TRANSCRIPT_LABEL,
        &mut verifier_transcript,
    )
    .unwrap();
    for claim in &proved.member_claims {
        verifier_transcript.append_labeled(OPENING_CLAIM_TRANSCRIPT_LABEL, claim);
    }

    let point = reduced.point.as_slice()[0];
    let expected = ExternalField::from_u64(3) + ExternalField::from_u64(2) * point;
    assert_eq!(proved.member_claims, vec![expected]);
    assert_eq!(reduced.value, verifier_coefficient * expected);
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
}
