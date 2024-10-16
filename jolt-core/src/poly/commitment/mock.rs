use std::marker::PhantomData;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::{
    field::JoltField,
    poly::dense_mlpoly::DensePolynomial,
    utils::{
        errors::ProofVerifyError,
        transcript::{AppendToTranscript, Transcript},
    },
};

use super::commitment_scheme::{BatchType, CommitShape, CommitmentScheme};

#[derive(Clone)]
pub struct MockCommitScheme<F: JoltField, ProofTranscript: Transcript> {
    _marker: PhantomData<(F, ProofTranscript)>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Default, Debug, PartialEq)]
pub struct MockCommitment<F: JoltField> {
    poly: DensePolynomial<F>,
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

    fn setup(_shapes: &[CommitShape]) -> Self::Setup {}
    fn commit(poly: &DensePolynomial<Self::Field>, _setup: &Self::Setup) -> Self::Commitment {
        MockCommitment {
            poly: poly.to_owned(),
        }
    }
    fn batch_commit(
        evals: &[&[Self::Field]],
        _gens: &Self::Setup,
        _batch_type: BatchType,
    ) -> Vec<Self::Commitment> {
        let polys: Vec<DensePolynomial<F>> = evals
            .iter()
            .map(|poly_evals| DensePolynomial::new(poly_evals.to_vec()))
            .collect();

        polys
            .into_iter()
            .map(|poly| MockCommitment { poly })
            .collect()
    }
    fn commit_slice(evals: &[Self::Field], _setup: &Self::Setup) -> Self::Commitment {
        MockCommitment {
            poly: DensePolynomial::new(evals.to_owned()),
        }
    }
    fn prove(
        _setup: &Self::Setup,
        _poly: &DensePolynomial<Self::Field>,
        opening_point: &[Self::Field],
        _transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        MockProof {
            opening_point: opening_point.to_owned(),
        }
    }
    fn batch_prove(
        _setup: &Self::Setup,
        _polynomials: &[&DensePolynomial<Self::Field>],
        opening_point: &[Self::Field],
        _openings: &[Self::Field],
        _batch_type: BatchType,
        _transcript: &mut ProofTranscript,
    ) -> Self::BatchedProof {
        MockProof {
            opening_point: opening_point.to_owned(),
        }
    }

    fn combine_commitments(
        commitments: &[&Self::Commitment],
        coeffs: &[Self::Field],
    ) -> Self::Commitment {
        let max_size = commitments
            .iter()
            .map(|comm| comm.poly.len())
            .max()
            .unwrap();
        let mut poly = DensePolynomial::new(vec![Self::Field::zero(); max_size]);
        for (commitment, coeff) in commitments.iter().zip(coeffs.iter()) {
            poly.Z
                .iter_mut()
                .zip(commitment.poly.Z.iter())
                .for_each(|(a, b)| {
                    *a += *coeff * b;
                });
        }
        MockCommitment { poly }
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

    fn batch_verify(
        batch_proof: &Self::BatchedProof,
        _setup: &Self::Setup,
        opening_point: &[Self::Field],
        openings: &[Self::Field],
        commitments: &[&Self::Commitment],
        _transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        assert_eq!(batch_proof.opening_point, opening_point);
        assert_eq!(openings.len(), commitments.len());
        for i in 0..openings.len() {
            let evaluation = commitments[i].poly.evaluate(opening_point);
            assert_eq!(evaluation, openings[i]);
        }
        Ok(())
    }

    fn protocol_name() -> &'static [u8] {
        b"mock_commit"
    }
}
