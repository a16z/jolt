use std::{marker::PhantomData, ops::Mul};

use ark_ec::pairing::{Pairing, PairingOutput};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use error::Error;
use params::PublicParams;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use scalar::ScalarProof;
use scalar::{commit, Commitment, Witness};
use vec_operations::{G1Vec, G2Vec};

use crate::utils::errors::ProofVerifyError;
use crate::{
    field::JoltField,
    poly::multilinear_polynomial::MultilinearPolynomial,
    utils::{
        math::Math,
        transcript::{AppendToTranscript, Transcript},
    },
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

impl<P: Pairing> AppendToTranscript for Commitment<P> {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, _transcript: &mut ProofTranscript) {
        todo!()
    }
}

#[derive(CanonicalDeserialize, CanonicalSerialize)]
pub struct DoryBatchedProof;

impl<P, ProofTranscript> CommitmentScheme<ProofTranscript> for DoryScheme<P, ProofTranscript>
where
    P: Pairing + Default,
    ProofTranscript: Transcript,
    G1<P>: Mul<Zr<P>, Output = G1<P>>,
    G2<P>: Mul<Zr<P>, Output = G2<P>>,
    P::ScalarField: JoltField + Default,
{
    type Field = Zr<P>;

    type Setup = PublicParams<P>;

    type Commitment = Commitment<P>;

    type Proof = ScalarProof<P>;

    type BatchedProof = DoryBatchedProof;

    fn setup(shapes: &[CommitShape]) -> Self::Setup {
        // Dory's setup procedure initializes
        let mut max_len: usize = 0;
        for shape in shapes {
            let len = shape.input_length.log_2();
            if len > max_len {
                max_len = len;
            }
        }
        let mut rng = ark_std::rand::thread_rng();
        PublicParams::new(&mut rng, max_len).expect("Length must be greater than 0")
    }

    fn commit(poly: &MultilinearPolynomial<Self::Field>, setup: &Self::Setup) -> Self::Commitment {
        let MultilinearPolynomial::LargeScalars(poly) = poly else {
            panic!("Expected LargeScalars polynomial");
        };
        let witness = Witness::new(setup, poly.evals_ref());
        commit(witness, setup).unwrap()
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
        _transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        let MultilinearPolynomial::LargeScalars(poly) = poly else {
            panic!("Expected LargeScalars polynomial");
        };
        let witness = Witness::new(setup, poly.evals_ref());
        ScalarProof::new(witness)
    }

    fn verify(
        proof: &Self::Proof,
        setup: &Self::Setup,
        _transcript: &mut ProofTranscript,
        _opening_point: &[Self::Field], // point at which the polynomial is evaluated
        _opening: &Self::Field,         // evaluation \widetilde{Z}(r)
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        if proof.verify(setup, commitment).unwrap() {
            Ok(())
        } else {
            Err(ProofVerifyError::VerificationFailed)
        }
    }

    fn protocol_name() -> &'static [u8] {
        b"dory"
    }
}
