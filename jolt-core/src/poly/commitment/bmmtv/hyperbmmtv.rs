//! # Hyper BMMTV extension
//!
//! This is a Reduction of Knowledge (RoK) from Multilinear Polynomial Evaluations (MPE) to
//! Univariate Polynomial Evaluations (UPE), that allows commiting to multilinear polynomials
//! using normal Bmmtv

use std::{borrow::Borrow, marker::PhantomData, sync::Arc};

use ark_ec::pairing::{Pairing, PairingOutput};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::Zero;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use rayon::prelude::*;

use crate::{
    field::JoltField,
    msm::Icicle,
    poly::{
        commitment::{
            bmmtv::poly_commit::UnivariatePolynomialCommitment,
            commitment_scheme::CommitmentScheme,
            kzg::{KZGProverKey, KZGVerifierKey, SRS},
        },
        multilinear_polynomial::MultilinearPolynomial,
        unipoly::UniPoly,
    },
    utils::{transcript::AppendToTranscript, transcript::Transcript},
};

#[derive(Clone)]
pub struct HyperBmmtv<P: Pairing, ProofTranscript: Transcript> {
    _phantom: PhantomData<(P, ProofTranscript)>,
}

#[derive(PartialEq, Eq, Clone, CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct HyperBmmtvCommitment<P: Pairing>(PairingOutput<P>, Vec<P::G1>);

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct HyperBmmtvProof<P: Pairing> {
    pub com: Vec<P::G1Affine>,
    pub w: Vec<P::G1Affine>,
    pub v: Vec<Vec<P::ScalarField>>,
}

impl<P: Pairing> Default for HyperBmmtvCommitment<P> {
    fn default() -> Self {
        todo!()
    }
}

impl<P: Pairing> AppendToTranscript for HyperBmmtvCommitment<P> {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, transcript: &mut ProofTranscript) {
        todo!()
    }
}

impl<F: JoltField> From<&MultilinearPolynomial<F>> for UniPoly<F> {
    fn from(_: &MultilinearPolynomial<F>) -> Self {
        todo!()
    }
}

impl<P: Pairing, ProofTranscript: Transcript> CommitmentScheme<ProofTranscript>
    for HyperBmmtv<P, ProofTranscript>
where
    PairingOutput<P>: CanonicalSerialize,
    P::ScalarField: JoltField,
    P::G1: Icicle,
    P::G2: Icicle,
{
    type Field = P::ScalarField;

    type Setup = (KZGProverKey<P>, KZGVerifierKey<P>);

    type Commitment = HyperBmmtvCommitment<P>;

    type Proof = ();

    type BatchedProof = ();

    fn setup(max_len: usize) -> Self::Setup {
        let mut rng = ChaCha20Rng::from_seed(*b"HyperBMMTV_POLY_COMMITMENT_SCHEM");
        SRS::trim(
            Arc::new(
                UnivariatePolynomialCommitment::<P, ProofTranscript>::setup(&mut rng, max_len)
                    .unwrap(),
            ),
            max_len - 1,
        )
    }

    fn commit(
        poly: &MultilinearPolynomial<Self::Field>,
        (p_srs, _): &Self::Setup,
    ) -> Self::Commitment {
        let field_elements = match poly {
            MultilinearPolynomial::U8Scalars(poly) => poly.coeffs_as_field_elements(),
            MultilinearPolynomial::U16Scalars(poly) => poly.coeffs_as_field_elements(),
            MultilinearPolynomial::U32Scalars(poly) => poly.coeffs_as_field_elements(),
            MultilinearPolynomial::U64Scalars(poly) => poly.coeffs_as_field_elements(),
            MultilinearPolynomial::LargeScalars(poly) => poly.evals(),
            _ => {
                panic!("Unexpected MultilinearPolynomial variant");
            }
        };

        let unipoly = UniPoly::from_coeff(field_elements);
        let commitment =
            UnivariatePolynomialCommitment::<P, ProofTranscript>::commit(p_srs, &unipoly).unwrap();
        HyperBmmtvCommitment(commitment.0, commitment.1)
    }

    fn batch_commit<U>(polys: &[U], gens: &Self::Setup) -> Vec<Self::Commitment>
    where
        U: Borrow<MultilinearPolynomial<Self::Field>> + Sync,
    {
        polys
            .iter()
            .map(|poly| Self::commit(poly.borrow(), gens))
            .collect()
    }

    fn prove(
        setup: &Self::Setup,
        poly: &MultilinearPolynomial<Self::Field>,
        point: &[Self::Field], // point at which the polynomial is evaluated
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        let ell = point.len();
        let n = poly.len();
        assert_eq!(n, 1 << ell); // Below we assume that n is a power of two

        // Phase 1  -- create commitments com_1, ..., com_\ell
        // We do not compute final Pi (and its commitment) as it is constant and equals to 'eval'
        // also known to verifier, so can be derived on its side as well
        let mut polys: Vec<UniPoly<P::ScalarField>> = Vec::with_capacity(ell - 1);
        polys.push(poly.into());
        for i in 0..ell - 1 {
            let previous_poly: &UniPoly<P::ScalarField> = &polys[i];
            let Pi_len = previous_poly.len() / 2;
            let mut Pi = vec![P::ScalarField::zero(); Pi_len];
            Pi.par_iter_mut().enumerate().for_each(|(j, Pi_j)| {
                *Pi_j = point[ell - i - 1] * (previous_poly[2 * j + 1] - previous_poly[2 * j])
                    + previous_poly[2 * j];
            });

            polys.push(UniPoly::from_coeff(Pi));
        }

        assert_eq!(polys.len(), ell);
        assert_eq!(polys[ell - 1].len(), 2);

        // We do not need to commit to the first polynomial as it is already committed.
        let com_list: Vec<_> = (&polys[1..])
            .iter()
            .map(|poly| {
                UnivariatePolynomialCommitment::<P, ProofTranscript>::commit(&setup.0, &poly)
                    .unwrap()
            })
            .collect();

        // Phase 2
        // We do not need to add x to the transcript, because in our context x was obtained from the transcript.
        // We also do not need to absorb `C` and `eval` as they are already absorbed by the transcript by the caller
        // CANT JUST PARALLELIZE transcript is &mut
        com_list.iter().for_each(|g| {
            transcript.append_points(&g.1);
            transcript.append_serializable(&g.0)
        });

        let r: <P as Pairing>::ScalarField = transcript.challenge_scalar();
        // Phase 3 -- create response
        // open all commits

        let eval_proofs = com_list
            .into_iter()
            .zip(&polys[1..])
            .map(|(comm, polynomial)| {
                UnivariatePolynomialCommitment::open(&setup.0, &polynomial, comm.1, &r, transcript)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        todo!()
    }

    fn verify(
        proof: &Self::Proof,
        setup: &Self::Setup,
        transcript: &mut ProofTranscript,
        opening_point: &[Self::Field], // point at which the polynomial is evaluated
        opening: &Self::Field,         // evaluation \widetilde{Z}(r)
        commitment: &Self::Commitment,
    ) -> Result<(), crate::utils::errors::ProofVerifyError> {
        todo!()
    }

    fn protocol_name() -> &'static [u8] {
        b"hyperbmmtv"
    }
}
