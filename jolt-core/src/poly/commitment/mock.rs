use std::borrow::Borrow;
use std::marker::PhantomData;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

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

#[derive(CanonicalSerialize, CanonicalDeserialize, Default, Debug, PartialEq)]
pub struct MockCommitment<F: JoltField> {
    poly: MultilinearPolynomial<F>,
}

impl<F: JoltField> AppendToTranscript for MockCommitment<F> {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, transcript: &mut ProofTranscript) {
        transcript.append_message(b"mocker");
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct MockProof<F: JoltField> {
    opening_point: Vec<F>,
}

impl<F, ProofTranscript> CommitmentScheme<ProofTranscript> for MockCommitScheme<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    type Field = F;
    type Setup = ();
    type Commitment = MockCommitment<F>;
    type Proof = MockProof<F>;
    type BatchedProof = MockProof<F>;

    fn setup(_max_poly_len: usize) -> Self::Setup {}

    fn srs_size(_setup: &Self::Setup) -> usize {
        1 << 10
    }

    fn commit(poly: &MultilinearPolynomial<Self::Field>, _setup: &Self::Setup) -> Self::Commitment {
        MockCommitment { poly: poly.clone() }
    }
    fn batch_commit<P>(polys: &[P], setup: &Self::Setup) -> Vec<Self::Commitment>
    where
        P: Borrow<MultilinearPolynomial<Self::Field>>,
    {
        polys
            .iter()
            .map(|poly| Self::commit(poly.borrow(), setup))
            .collect()
    }
    fn prove(
        _setup: &Self::Setup,
        _poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[Self::Field],
        _transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        MockProof {
            opening_point: opening_point.to_owned(),
        }
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

    fn verify(
        proof: &Self::Proof,
        _setup: &Self::Setup,
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
