use std::{marker::PhantomData, ops::Mul};

use ark_ec::pairing::{Pairing, PairingOutput};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use error::Error;
use params::PublicParams;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use reduce::reduce;
use reduce::DoryProof;
use scalar::ScalarProof;
use scalar::{commit, Commitment, Witness};
use vec_operations::{G1Vec, G2Vec};

use crate::msm::VariableBaseMSM;
use crate::utils::errors::ProofVerifyError;
use crate::{
    field::JoltField,
    poly::multilinear_polynomial::MultilinearPolynomial,
    utils::{math::Math, transcript::Transcript},
};

use super::commitment_scheme::{CommitShape, CommitmentScheme};

mod error;
mod params;
mod reduce;
mod scalar;
mod vec_operations;

#[cfg(test)]
mod tests;

/// G1
pub type G1<Curve> = <Curve as Pairing>::G1;

/// G2
pub type G2<Curve> = <Curve as Pairing>::G2;

/// Cyclic group of integers modulo prime number r
///
/// This is the Domain Set Z
pub type Zr<Curve> = <Curve as Pairing>::ScalarField;

pub type Gt<Curve> = PairingOutput<Curve>;

#[derive(Clone, Default)]
pub struct DoryScheme<P: Pairing, ProofTranscript: Transcript> {
    _data: PhantomData<(P, ProofTranscript)>,
}

fn append_gt<P: Pairing, ProofTranscript: Transcript>(transcript: &mut ProofTranscript, gt: Gt<P>) {
    let mut buf = vec![];
    gt.serialize_uncompressed(&mut buf).unwrap();
    // Serialize uncompressed gives the scalar in LE byte order which is not
    // a natural representation in the EVM for scalar math so we reverse
    // to get an EVM compatible version.
    buf = buf.into_iter().rev().collect();
    transcript.append_bytes(&buf);
}

#[derive(CanonicalDeserialize, CanonicalSerialize)]
pub struct DoryBatchedProof;

impl<P, ProofTranscript> CommitmentScheme<ProofTranscript> for DoryScheme<P, ProofTranscript>
where
    P::G1: VariableBaseMSM,
    P: Pairing + Default,
    ProofTranscript: Transcript,
    G1<P>: Mul<Zr<P>, Output = G1<P>>,
    G2<P>: Mul<Zr<P>, Output = G2<P>>,
    P::ScalarField: JoltField + Default,
{
    type Field = Zr<P>;

    type Setup = Vec<PublicParams<P>>;

    type Commitment = Commitment<P>;

    type Proof = DoryProof<P>;

    type BatchedProof = DoryBatchedProof;

    fn setup(shapes: &[CommitShape]) -> Self::Setup {
        // Dory's setup procedure initializes
        let max_len = shapes
            .iter()
            .map(|shape| shape.input_length.log_2())
            .max()
            .unwrap();
        let mut rng = ark_std::rand::thread_rng();
        PublicParams::generate_public_params(&mut rng, max_len)
            .expect("Length must be greater than 0")
    }

    fn commit(poly: &MultilinearPolynomial<Self::Field>, setup: &Self::Setup) -> Self::Commitment {
        let public_param = setup.first().unwrap();
        let witness = Witness::new(public_param, poly);
        commit(witness, public_param).unwrap()
    }

    fn batch_commit(
        polys: &[&MultilinearPolynomial<Self::Field>],
        setup: &Self::Setup,
        _batch_type: super::commitment_scheme::BatchType,
    ) -> Vec<Self::Commitment> {
        polys
            .into_par_iter()
            .map(|poly| Self::commit(poly, setup))
            .collect()
    }

    fn prove(
        setup: &Self::Setup,
        poly: &MultilinearPolynomial<Self::Field>,
        _opening_point: &[Self::Field], // point at which the polynomial is evaluated
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        let public_param = setup.first().unwrap();

        let witness = Witness::new(public_param, poly);

        let commitment = commit(witness.clone(), public_param).unwrap();

        reduce(transcript, setup.as_slice(), witness, commitment).unwrap()
    }

    fn verify(
        proof: &Self::Proof,
        setup: &Self::Setup,
        transcript: &mut ProofTranscript,
        _opening_point: &[Self::Field], // point at which the polynomial is evaluated
        _opening: &Self::Field,         // evaluation \widetilde{Z}(r)
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        if proof.verify(transcript, setup, *commitment).unwrap() {
            Ok(())
        } else {
            Err(ProofVerifyError::VerificationFailed)
        }
    }

    fn protocol_name() -> &'static [u8] {
        b"dory"
    }
}
