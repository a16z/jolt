use std::borrow::Borrow;
use std::marker::PhantomData;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::{
    field::JoltField,
    poly::multilinear_polynomial::MultilinearPolynomial,
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
        _poly: &MultilinearPolynomial<Self::Field>,
        _setup: &Self::ProverSetup,
    ) -> Self::Commitment {
        MockCommitment::default()
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
        _commitments: &[&Self::Commitment],
        _coeffs: &[Self::Field],
    ) -> Self::Commitment {
        MockCommitment::default()
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
