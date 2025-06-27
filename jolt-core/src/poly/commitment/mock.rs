use std::borrow::Borrow;
use std::marker::PhantomData;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Valid};

use crate::{
    field::JoltField,
    poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    utils::{
        errors::ProofVerifyError,
        transcript::{AppendToTranscript, Transcript},
    },
};

use super::commitment_scheme::CommitmentScheme;

#[derive(Clone)]
pub struct MockCommitScheme<F: JoltField, ProofTranscript: Transcript> {
    _marker: PhantomData<(F, ProofTranscript)>,
}

#[derive(Default, Debug, PartialEq, Clone)]
pub struct MockCommitment<F: JoltField> {
    poly: MultilinearPolynomial<F>,
}

impl<F: JoltField> Valid for MockCommitment<F> {
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        unimplemented!("satisfy trait bounds")
    }
}

impl<F: JoltField> CanonicalSerialize for MockCommitment<F> {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        _: W,
        _: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        unimplemented!("satisfy trait bounds")
    }

    fn serialized_size(&self, _: ark_serialize::Compress) -> usize {
        unimplemented!("satisfy trait bounds")
    }
}

impl<F: JoltField> CanonicalDeserialize for MockCommitment<F> {
    fn deserialize_with_mode<R: std::io::Read>(
        _: R,
        _: ark_serialize::Compress,
        _: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        unimplemented!("satisfy trait bounds")
    }
}

impl<F: JoltField> AppendToTranscript for MockCommitment<F> {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, transcript: &mut ProofTranscript) {
        transcript.append_message(b"mocker");
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct MockProof<F: JoltField> {
    opening_point: Vec<F>,
}

impl<F, ProofTranscript> CommitmentScheme<ProofTranscript> for MockCommitScheme<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    type Field = F;
    type ProverSetup = ();
    type VerifierSetup = ();
    type Commitment = MockCommitment<F>;
    type Proof = MockProof<F>;
    type BatchedProof = MockProof<F>;

    fn setup_prover(_max_len: usize) -> Self::ProverSetup {}

    fn setup_verifier(_setup: &Self::ProverSetup) -> Self::VerifierSetup {}

    fn srs_size(_setup: &Self::ProverSetup) -> usize {
        1 << 10
    }

    fn commit(
        poly: &MultilinearPolynomial<Self::Field>,
        _setup: &Self::ProverSetup,
    ) -> Self::Commitment {
        MockCommitment { poly: poly.clone() }
    }
    fn batch_commit<P>(polys: &[P], gens: &Self::ProverSetup) -> Vec<Self::Commitment>
    where
        P: Borrow<MultilinearPolynomial<Self::Field>>,
    {
        polys
            .iter()
            .map(|poly| Self::commit(poly.borrow(), gens))
            .collect()
    }
    fn combine_commitments(
        commitments: &[&Self::Commitment],
        coeffs: &[Self::Field],
    ) -> Self::Commitment {
        let polys: Vec<_> = commitments
            .iter()
            .map(|commitment| &commitment.poly)
            .collect();

        MockCommitment {
            poly: MultilinearPolynomial::linear_combination(&polys, coeffs),
        }
    }

    fn prove(
        _setup: &Self::ProverSetup,
        _poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[Self::Field],
        _transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        MockProof {
            opening_point: opening_point.to_owned(),
        }
    }

    fn verify(
        proof: &Self::Proof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut ProofTranscript,
        opening_point: &[Self::Field],
        opening: &Self::Field,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        let evaluation = commitment.poly.evaluate(opening_point);
        assert_eq!(evaluation, *opening);
        assert_eq!(proof.opening_point, opening_point);
        Ok(())
    }

    fn protocol_name() -> &'static [u8] {
        b"mock_commit"
    }
}
