use std::{
    fmt::{Display, Formatter, Result as FmtResult},
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use jolt_field::{
    AdditiveGroup, CanonicalBytes, CanonicalDecode, CanonicalEncoding, Field, Prime64Offset59, Ring,
};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{
    prove_batch, BatchMember, BatchPrelude, ClearSumcheckRecorder, ProveRounds, SequentialRounds,
    SumcheckClaim, SumcheckError, SumcheckRecorder, SumcheckVerifier,
};
use jolt_transcript::{Channel, Keccak, ProtocolId, ProverTranscript, VerifierTranscript};
use num_traits::{One, Zero};
use rand_core::RngCore;
use spongefish::Encoding;
use spongefish::NargDeserialize;
use spongefish::VerificationResult;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
struct ExternalField(Prime64Offset59);

impl Display for ExternalField {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
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

    fn random<R: RngCore>(rng: &mut R) -> Self {
        Self(Prime64Offset59::random(rng))
    }
}

impl CanonicalBytes for ExternalField {
    const NUM_BYTES: usize = Prime64Offset59::NUM_BYTES;

    fn to_bytes_le(&self, out: &mut [u8]) {
        self.0.to_bytes_le(out);
    }
}

impl Encoding<[u8]> for ExternalField {
    fn encode(&self) -> impl AsRef<[u8]> {
        jolt_field::narg::encode(self)
    }
}

impl CanonicalDecode for ExternalField {
    fn from_bytes_le_checked(bytes: &[u8]) -> Option<Self> {
        Prime64Offset59::from_bytes_le_checked(bytes).map(Self)
    }
}

impl NargDeserialize for ExternalField {
    fn deserialize_from_narg(buf: &mut &[u8]) -> VerificationResult<Self> {
        jolt_field::narg::deserialize(buf)
    }
}

impl CanonicalEncoding for ExternalField {
    const MODULUS_BITS: u32 = Prime64Offset59::MODULUS_BITS;

    fn from_bytes_le_reduced(bytes: &[u8]) -> Self {
        Self(Prime64Offset59::from_bytes_le_reduced(bytes))
    }

    fn to_u128_checked(&self) -> Option<u128> {
        self.0.to_u128_checked()
    }

    fn from_u128_checked(value: u128) -> Option<Self> {
        Prime64Offset59::from_u128_checked(value).map(Self)
    }

    fn from_u128_reduced(value: u128) -> Self {
        Self(Prime64Offset59::from_u128_reduced(value))
    }

    fn num_bits(&self) -> u32 {
        self.0.num_bits()
    }

    fn from_scalar_challenge_bytes(bytes: &[u8]) -> Self {
        Self(Prime64Offset59::from_scalar_challenge_bytes(bytes))
    }
}

const PROTOCOL: ProtocolId = ProtocolId::new::<Keccak>("jolt-sumcheck/external-field");
const SESSION: &[u8] = b"external-field";

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
        let polynomial =
            field_only_polynomial([ExternalField::from_u64(3), ExternalField::from_u64(2)]);
        assert_eq!(
            polynomial.evaluate(ExternalField::zero()) + polynomial.evaluate(ExternalField::one()),
            previous_claim
        );
        Ok(polynomial)
    }

    fn finish_rounds(&mut self, _bind: ExternalField) -> Result<(), SumcheckError<ExternalField>> {
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
    let mut prover_transcript = ProverTranscript::<Keccak>::new(&PROTOCOL, SESSION);
    let mut recorder = ClearSumcheckRecorder::<ExternalField>::new();
    recorder.absorb_input_claims(&[input_claim], &mut prover_transcript);
    let coefficient: ExternalField = prover_transcript.challenge_small();
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
    recorder
        .finish(&proved.member_claims, &mut prover_transcript)
        .unwrap();
    let prover_state: [u8; 32] = prover_transcript.challenge_bytes::<32>();
    let proof = prover_transcript.finish();

    let mut verifier_transcript = VerifierTranscript::<Keccak>::new(&PROTOCOL, SESSION, &proof);
    verifier_transcript.public(&input_claim);
    let verifier_coefficient: ExternalField = verifier_transcript.challenge_small();
    let reduced = SumcheckVerifier::verify_compressed(
        &SumcheckClaim::new(1, 1, verifier_coefficient * input_claim),
        &mut verifier_transcript,
    )
    .unwrap();
    let opening_claims: Vec<ExternalField> = verifier_transcript.receive_n(1).unwrap();
    assert_eq!(verifier_transcript.challenge_bytes::<32>(), prover_state);
    verifier_transcript.finish().unwrap();

    let point = reduced.point.as_slice()[0];
    let expected = ExternalField::from_u64(3) + ExternalField::from_u64(2) * point;
    assert_eq!(opening_claims, vec![expected]);
    assert_eq!(proved.member_claims, vec![expected]);
    assert_eq!(reduced.value, verifier_coefficient * expected);
}
