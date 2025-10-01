use std::borrow::Borrow;
use std::marker::PhantomData;

use ark_ff::biginteger::S128;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::{
            StreamingCommitmentScheme, StreamingCommitmentScheme_, StreamingProcessChunk,
        },
        compact_polynomial::StreamingCompactWitness,
        dense_mlpoly::StreamingDenseWitness,
        multilinear_polynomial::MultilinearPolynomial,
        one_hot_polynomial::StreamingOneHotWitness,
    },
    transcripts::{AppendToTranscript, Transcript},
    utils::errors::ProofVerifyError,
};

use super::commitment_scheme::CommitmentScheme;

#[derive(Clone)]
pub struct MockCommitScheme<F: JoltField> {
    _marker: PhantomData<F>,
}

#[derive(Default, Debug, PartialEq, Clone, CanonicalDeserialize, CanonicalSerialize)]
pub struct MockCommitment<F: JoltField> {
    _field: PhantomData<F>,
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

impl<F> CommitmentScheme for MockCommitScheme<F>
where
    F: JoltField,
{
    type Field = F;
    type ProverSetup = ();
    type VerifierSetup = ();
    type Commitment = MockCommitment<F>;
    type Proof = MockProof<F>;
    type BatchedProof = MockProof<F>;
    type OpeningProofHint = ();

    fn setup_prover(_num_vars: usize) -> Self::ProverSetup {}

    fn setup_verifier(_setup: &Self::ProverSetup) -> Self::VerifierSetup {}

    fn commit(
        _poly: &MultilinearPolynomial<Self::Field>,
        _setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        (MockCommitment::default(), ())
    }

    fn batch_commit<P>(polys: &[P], gens: &Self::ProverSetup) -> Vec<Self::Commitment>
    where
        P: Borrow<MultilinearPolynomial<Self::Field>>,
    {
        polys
            .iter()
            .map(|poly| Self::commit(poly.borrow(), gens).0)
            .collect()
    }

    fn combine_commitments<C: Borrow<Self::Commitment>>(
        _commitments: &[C],
        _coeffs: &[Self::Field],
    ) -> Self::Commitment {
        MockCommitment::default()
    }

    fn combine_hints(
        _hints: Vec<Self::OpeningProofHint>,
        _coeffs: &[Self::Field],
    ) -> Self::OpeningProofHint {
    }

    fn prove<ProofTranscript: Transcript>(
        _setup: &Self::ProverSetup,
        _poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[Self::Field],
        _: Self::OpeningProofHint,
        _transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        MockProof {
            opening_point: opening_point.to_owned(),
        }
    }

    fn verify<ProofTranscript: Transcript>(
        proof: &Self::Proof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut ProofTranscript,
        opening_point: &[Self::Field],
        _opening: &Self::Field,
        _commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        assert_eq!(proof.opening_point, opening_point);
        Ok(())
    }

    fn protocol_name() -> &'static [u8] {
        b"mock_commit"
    }
}
impl<F> StreamingCommitmentScheme_ for MockCommitScheme<F>
where
    F: JoltField,
{
    type State<'a> = ();

    type ChunkState = ();

    type SetupCache = ();

    fn cache_setup(setup: &Self::ProverSetup) -> Self::SetupCache {
        todo!()
    }

    fn initialize<'a>(
        poly: crate::poly::multilinear_polynomial::Multilinear,
        size: usize,
        setup: &'a Self::ProverSetup,
        setup_cache: &'a Self::SetupCache,
    ) -> Self::State<'a> {
        todo!()
    }

    fn process<'a>(
        poly: crate::poly::multilinear_polynomial::Multilinear,
        state: Self::State<'a>,
        eval: Self::Field,
    ) -> Self::State<'a> {
        todo!()
    }

    fn process_chunk<'a, T>(state: &Self::State<'a>, chunk: &[T]) -> Self::ChunkState
    where
        Self: super::commitment_scheme::StreamingProcessChunk<T>,
    {
        todo!()
    }

    fn finalize<'a>(
        state: &Self::State<'a>,
        chunks: &[Self::ChunkState],
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        todo!()
    }
}

impl<F> StreamingProcessChunk<StreamingDenseWitness<F>> for MockCommitScheme<F>
where
    F: JoltField,
{
    fn process_chunk_t<'a>(
        s: &Self::State<'a>,
        chunk: &[StreamingDenseWitness<F>],
    ) -> Self::ChunkState {
        todo!()
    }
}

impl<F> StreamingProcessChunk<StreamingCompactWitness<u8, F>> for MockCommitScheme<F>
where
    F: JoltField,
{
    fn process_chunk_t<'a>(
        s: &Self::State<'a>,
        chunk: &[StreamingCompactWitness<u8, F>],
    ) -> Self::ChunkState {
        todo!()
    }
}

impl<F> StreamingProcessChunk<StreamingCompactWitness<u16, F>> for MockCommitScheme<F>
where
    F: JoltField,
{
    fn process_chunk_t<'a>(
        s: &Self::State<'a>,
        chunk: &[StreamingCompactWitness<u16, F>],
    ) -> Self::ChunkState {
        todo!()
    }
}

impl<F> StreamingProcessChunk<StreamingCompactWitness<u32, F>> for MockCommitScheme<F>
where
    F: JoltField,
{
    fn process_chunk_t<'a>(
        s: &Self::State<'a>,
        chunk: &[StreamingCompactWitness<u32, F>],
    ) -> Self::ChunkState {
        todo!()
    }
}

impl<F> StreamingProcessChunk<StreamingCompactWitness<u64, F>> for MockCommitScheme<F>
where
    F: JoltField,
{
    fn process_chunk_t<'a>(
        s: &Self::State<'a>,
        chunk: &[StreamingCompactWitness<u64, F>],
    ) -> Self::ChunkState {
        todo!()
    }
}

impl<F> StreamingProcessChunk<StreamingCompactWitness<i64, F>> for MockCommitScheme<F>
where
    F: JoltField,
{
    fn process_chunk_t<'a>(
        s: &Self::State<'a>,
        chunk: &[StreamingCompactWitness<i64, F>],
    ) -> Self::ChunkState {
        todo!()
    }
}

impl<F> StreamingProcessChunk<StreamingCompactWitness<i128, F>> for MockCommitScheme<F>
where
    F: JoltField,
{
    fn process_chunk_t<'a>(
        s: &Self::State<'a>,
        chunk: &[StreamingCompactWitness<i128, F>],
    ) -> Self::ChunkState {
        todo!()
    }
}

impl<F> StreamingProcessChunk<StreamingCompactWitness<S128, F>> for MockCommitScheme<F>
where
    F: JoltField,
{
    fn process_chunk_t<'a>(
        s: &Self::State<'a>,
        chunk: &[StreamingCompactWitness<S128, F>],
    ) -> Self::ChunkState {
        todo!()
    }
}

impl<F> StreamingProcessChunk<StreamingOneHotWitness<F>> for MockCommitScheme<F>
where
    F: JoltField,
{
    fn process_chunk_t<'a>(
        s: &Self::State<'a>,
        chunk: &[StreamingOneHotWitness<F>],
    ) -> Self::ChunkState {
        todo!()
    }
}
