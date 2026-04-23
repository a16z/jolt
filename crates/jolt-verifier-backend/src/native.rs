use std::marker::PhantomData;

use jolt_field::Field;
use jolt_openings::{CommitmentScheme, OpeningsError};
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, LabelWithCount, Transcript};

use crate::backend::{CommitmentOrigin, FieldBackend, ScalarOrigin};
use crate::commitment::CommitmentBackend;
use crate::error::BackendError;

/// Zero-overhead [`FieldBackend`] backed by the underlying field directly.
///
/// `Scalar = F`. Every method is `#[inline(always)]` and forwards to a single
/// field operator, so monomorphization erases the trait machinery and the
/// generated code is byte-identical to handwritten `F` arithmetic.
///
/// Use this for production verification where the verifier is executed in
/// the clear, on real hardware, with no recording or constraint generation.
#[derive(Copy, Clone, Debug, Default)]
pub struct Native<F: Field>(PhantomData<F>);

impl<F: Field> Native<F> {
    /// Constructs a new `Native` backend.
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<F: Field> FieldBackend for Native<F> {
    type F = F;
    type Scalar = F;
    type Transcript = Blake2bTranscript<F>;

    #[inline(always)]
    fn wrap(&mut self, value: F, _origin: ScalarOrigin, _label: &'static str) -> F {
        value
    }

    #[inline(always)]
    fn const_i128(&mut self, v: i128) -> F {
        F::from_i128(v)
    }

    #[inline(always)]
    fn const_zero(&mut self) -> F {
        F::zero()
    }

    #[inline(always)]
    fn const_one(&mut self) -> F {
        F::one()
    }

    #[inline(always)]
    fn add(&mut self, a: &F, b: &F) -> F {
        *a + *b
    }

    #[inline(always)]
    fn sub(&mut self, a: &F, b: &F) -> F {
        *a - *b
    }

    #[inline(always)]
    fn mul(&mut self, a: &F, b: &F) -> F {
        *a * *b
    }

    #[inline(always)]
    fn neg(&mut self, a: &F) -> F {
        -*a
    }

    #[inline(always)]
    fn square(&mut self, a: &F) -> F {
        a.square()
    }

    #[inline(always)]
    fn inverse(&mut self, a: &F, ctx: &'static str) -> Result<F, BackendError> {
        a.inverse().ok_or(BackendError::InverseOfZero(ctx))
    }

    #[inline(always)]
    fn assert_eq(&mut self, a: &F, b: &F, ctx: &'static str) -> Result<(), BackendError> {
        if a == b {
            Ok(())
        } else {
            Err(BackendError::AssertionFailed(ctx))
        }
    }

    #[inline(always)]
    fn unwrap(&self, scalar: &F) -> Option<F> {
        Some(*scalar)
    }

    #[inline(always)]
    fn new_transcript(&mut self, label: &'static [u8]) -> Self::Transcript {
        Blake2bTranscript::<F>::new(label)
    }

    #[inline(always)]
    fn squeeze(&mut self, transcript: &mut Self::Transcript, _label: &'static str) -> (F, F) {
        let v = transcript.challenge();
        (v, v)
    }
}

/// Zero-overhead [`CommitmentBackend`] impl for [`Native`].
///
/// Identity wrap, direct transcript absorb, direct PCS verify. Every
/// method is `#[inline(always)]`; monomorphization erases the trait
/// dispatch and produces code byte-identical to a verifier that calls
/// `PCS::verify` directly.
impl<F, PCS> CommitmentBackend<PCS> for Native<F>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
    PCS::Output: AppendToTranscript,
{
    type Commitment = PCS::Output;

    #[inline(always)]
    fn wrap_commitment(
        &mut self,
        value: PCS::Output,
        _origin: CommitmentOrigin,
        _label: &'static str,
    ) -> PCS::Output {
        value
    }

    #[inline(always)]
    fn absorb_commitment(
        &mut self,
        transcript: &mut Self::Transcript,
        commitment: &PCS::Output,
        label: &'static [u8],
    ) {
        // Standard two-step inline absorb:
        //   1. append a LabelWithCount header so the verifier-side
        //      domain separation matches the prover's serialised stream;
        //   2. forward the commitment's own AppendToTranscript impl.
        transcript.append(&LabelWithCount(label, commitment.serialized_len()));
        commitment.append_to_transcript(transcript);
    }

    #[inline(always)]
    fn verify_opening(
        &mut self,
        vk: &PCS::VerifierSetup,
        commitment: &PCS::Output,
        point: &[F],
        claim: &F,
        proof: &PCS::Proof,
        transcript: &mut Self::Transcript,
    ) -> Result<(), OpeningsError> {
        PCS::verify(commitment, point, *claim, proof, vk, transcript)
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]

    use super::*;
    use jolt_field::Fr;

    #[test]
    fn native_arithmetic_roundtrip() {
        let mut b = Native::<Fr>::new();
        let two = b.const_i128(2);
        let three = b.const_i128(3);
        let six = b.mul(&two, &three);
        let expected = b.const_i128(6);
        b.assert_eq(&six, &expected, "2*3==6").unwrap();
    }

    #[test]
    fn native_assert_eq_failure_returns_error() {
        let mut b = Native::<Fr>::new();
        let a = b.const_i128(2);
        let bv = b.const_i128(3);
        let err = b.assert_eq(&a, &bv, "ne").unwrap_err();
        assert!(matches!(err, BackendError::AssertionFailed("ne")));
    }

    #[test]
    fn native_inverse_of_zero_errors() {
        let mut b = Native::<Fr>::new();
        let z = b.const_zero();
        let err = b.inverse(&z, "z").unwrap_err();
        assert!(matches!(err, BackendError::InverseOfZero("z")));
    }

    #[test]
    fn native_square_matches_mul() {
        let mut b = Native::<Fr>::new();
        let v = b.wrap_proof(Fr::from_u64(7), "x");
        let s = b.square(&v);
        let m = b.mul(&v, &v);
        b.assert_eq(&s, &m, "square==mul").unwrap();
    }

    /// Native [`CommitmentBackend`] is the identity wrapper:
    /// `wrap_commitment` returns its input by value, `absorb_commitment`
    /// performs the standard label-with-count + `AppendToTranscript`
    /// two-step, and `verify_opening` forwards directly to
    /// `<PCS as CommitmentScheme>::verify`. Round-tripping against
    /// `MockCommitmentScheme` (no curves required) catches any drift in
    /// the trait wiring.
    #[test]
    fn native_commitment_backend_round_trip_against_mock() {
        use jolt_openings::mock::MockCommitmentScheme;

        let mut backend = Native::<Fr>::new();
        let mut transcript = backend.new_transcript(b"native_commit_test");

        let evaluations = vec![
            Fr::from_u64(1),
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(4),
        ];
        let poly = jolt_poly::Polynomial::new(evaluations.clone());
        let (commitment, _hint) =
            <MockCommitmentScheme<Fr> as CommitmentScheme>::commit(&poly, &());
        let point = vec![Fr::from_u64(5), Fr::from_u64(6)];
        let eval = poly.evaluate(&point);
        let proof = <MockCommitmentScheme<Fr> as CommitmentScheme>::open(
            &poly,
            &point,
            eval,
            &(),
            None,
            &mut transcript,
        );

        let wrapped = <Native<Fr> as CommitmentBackend<MockCommitmentScheme<Fr>>>::wrap_commitment(
            &mut backend,
            commitment.clone(),
            CommitmentOrigin::Proof,
            "C",
        );
        // Identity wrap: no allocation, no rewrap — just the same value back.
        assert_eq!(
            wrapped, commitment,
            "Native::wrap_commitment must be identity"
        );

        <Native<Fr> as CommitmentBackend<MockCommitmentScheme<Fr>>>::absorb_commitment(
            &mut backend,
            &mut transcript,
            &wrapped,
            b"C",
        );

        <Native<Fr> as CommitmentBackend<MockCommitmentScheme<Fr>>>::verify_opening(
            &mut backend,
            &(),
            &wrapped,
            &point,
            &eval,
            &proof,
            &mut transcript,
        )
        .expect("Native::verify_opening must accept a valid mock opening");
    }
}
