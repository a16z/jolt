#![allow(dead_code)]

use crate::poly::commitment::commitment_scheme::BatchType;
use crate::poly::commitment::commitment_scheme::CommitShape;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::utils::errors::ProofVerifyError;
use crate::utils::transcript::{AppendToTranscript, Transcript};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::borrow::Borrow;
use std::marker::PhantomData;

#[derive(Clone)]
pub struct Binius128Scheme<ProofTranscript: Transcript> {
    _phantom: PhantomData<ProofTranscript>,
}

#[derive(Default, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct BiniusCommitment {}

impl AppendToTranscript for BiniusCommitment {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, _transcript: &mut ProofTranscript) {
        todo!()
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct BiniusProof {}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct BiniusBatchedProof {}

#[derive(Clone)]
pub struct None {}

impl<ProofTranscript: Transcript> CommitmentScheme<ProofTranscript>
    for Binius128Scheme<ProofTranscript>
{
    type Field = crate::field::binius::BiniusField<binius_field::BinaryField128bPolyval>;
    type Setup = None;
    type Commitment = BiniusCommitment;
    type Proof = BiniusProof;
    type BatchedProof = BiniusBatchedProof;

    fn setup(_shapes: &[CommitShape]) -> Self::Setup {
        None {}
    }
    fn commit(
        _poly: &MultilinearPolynomial<Self::Field>,
        _setup: &Self::Setup,
    ) -> Self::Commitment {
        todo!()
    }
    fn batch_commit<P>(
        _polys: &[P],
        _gens: &Self::Setup,
        _batch_type: BatchType,
    ) -> Vec<Self::Commitment>
    where
        P: Borrow<MultilinearPolynomial<Self::Field>>,
    {
        todo!()
    }
    fn prove(
        _none: &Self::Setup,
        _poly: &MultilinearPolynomial<Self::Field>,
        _opening_point: &[Self::Field],
        _transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        todo!()
    }
    fn verify(
        _proof: &Self::Proof,
        _setup: &Self::Setup,
        _transcript: &mut ProofTranscript,
        _opening_point: &[Self::Field],
        _opening: &Self::Field,
        _commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        todo!()
    }

    fn protocol_name() -> &'static [u8] {
        b"binius_commit"
    }
}
