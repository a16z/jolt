#![allow(dead_code)]

use crate::field::JoltField;
use crate::msm::Icicle;
use crate::poly::commitment::commitment_scheme::BatchType;
use crate::poly::commitment::commitment_scheme::CommitShape;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::utils::errors::ProofVerifyError;
use crate::utils::transcript::{AppendToTranscript, Transcript};
use ark_ec::CurveGroup;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::marker::PhantomData;

#[derive(Clone)]
pub struct DoryScheme<G: CurveGroup, ProofTranscript: Transcript> {
    _phantom: PhantomData<(G, ProofTranscript)>,
}

#[derive(Default, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryCommitment<G: CurveGroup> {
    pub c: G,
    pub d1: G,
    pub d2: G
}

impl<G: CurveGroup> AppendToTranscript for DoryCommitment<G> {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, _transcript: &mut ProofTranscript) {
        todo!()
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryProof {}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryBatchedProof {}

#[derive(Clone)]
pub struct None {}

impl<F: JoltField, G: CurveGroup<ScalarField = F> + Icicle, ProofTranscript: Transcript> CommitmentScheme<ProofTranscript>
    for DoryScheme<G, ProofTranscript>
{
    type Field = G::ScalarField;
    type Setup = None;
    type Commitment = DoryCommitment<G>;
    type Proof = DoryProof;
    type BatchedProof = DoryBatchedProof;

    fn setup(_shapes: &[CommitShape]) -> Self::Setup {
        None {}
    }
    fn commit(_poly: &DensePolynomial<Self::Field>, _setup: &Self::Setup) -> Self::Commitment {
        todo!()
    }
    fn batch_commit(
        _evals: &[&[Self::Field]],
        _gens: &Self::Setup,
        _batch_type: BatchType,
    ) -> Vec<Self::Commitment> {
        todo!()
    }
    fn commit_slice(_evals: &[Self::Field], _setup: &Self::Setup) -> Self::Commitment {
        todo!()
    }
    fn prove(
        _none: &Self::Setup,
        _poly: &DensePolynomial<Self::Field>,
        _opening_point: &[Self::Field],
        _transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        todo!()
    }
    fn batch_prove(
        _none: &Self::Setup,
        _polynomials: &[&DensePolynomial<Self::Field>],
        _opening_point: &[Self::Field],
        _openings: &[Self::Field],
        _batch_type: BatchType,
        _transcript: &mut ProofTranscript,
    ) -> Self::BatchedProof {
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

    fn batch_verify(
        _batch_proof: &Self::BatchedProof,
        _setup: &Self::Setup,
        _opening_point: &[Self::Field],
        _openings: &[Self::Field],
        _commitments: &[&Self::Commitment],
        _transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        todo!()
    }

    fn protocol_name() -> &'static [u8] {
        b"dory_commit"
    }
}
