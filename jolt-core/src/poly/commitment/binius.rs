use crate::poly::commitment::commitment_scheme::BatchType;
use crate::poly::commitment::commitment_scheme::CommitShape;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::utils::errors::ProofVerifyError;
use crate::utils::transcript::{AppendToTranscript, ProofTranscript};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

#[derive(Clone)]
pub struct Binius128Scheme {}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct BiniusCommitment {}

impl AppendToTranscript for BiniusCommitment {
    fn append_to_transcript(&self, label: &[u8], transcript: &mut ProofTranscript) {
        todo!()
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct BiniusProof {}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct BiniusBatchedProof {}

#[derive(Clone)]
pub struct None {}

impl CommitmentScheme for Binius128Scheme {
    type Field = crate::field::binius::BiniusField<binius_field::BinaryField128bPolyval>;
    type Setup = None;
    type Commitment = BiniusCommitment;
    type Proof = BiniusProof;
    type BatchedProof = BiniusBatchedProof;

    fn setup(shapes: &[CommitShape]) -> Self::Setup {
        None {}
    }
    fn commit(poly: &DensePolynomial<Self::Field>, setup: &Self::Setup) -> Self::Commitment {
        todo!()
    }
    fn batch_commit(
        evals: &[&[Self::Field]],
        gens: &Self::Setup,
        batch_type: BatchType,
    ) -> Vec<Self::Commitment> {
        todo!()
    }
    fn commit_slice(evals: &[Self::Field], setup: &Self::Setup) -> Self::Commitment {
        todo!()
    }
    fn prove(
        poly: &DensePolynomial<Self::Field>,
        opening_point: &[Self::Field],
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        todo!()
    }
    fn batch_prove(
        polynomials: &[&DensePolynomial<Self::Field>],
        opening_point: &[Self::Field],
        openings: &[Self::Field],
        batch_type: BatchType,
        transcript: &mut ProofTranscript,
    ) -> Self::BatchedProof {
        todo!()
    }

    fn verify(
        proof: &Self::Proof,
        setup: &Self::Setup,
        transcript: &mut ProofTranscript,
        opening_point: &[Self::Field],
        opening: &Self::Field,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        todo!()
    }

    fn batch_verify(
        batch_proof: &Self::BatchedProof,
        setup: &Self::Setup,
        opening_point: &[Self::Field],
        openings: &[Self::Field],
        commitments: &[&Self::Commitment],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        todo!()
    }

    fn protocol_name() -> &'static [u8] {
        b"binius_commit"
    }
}
