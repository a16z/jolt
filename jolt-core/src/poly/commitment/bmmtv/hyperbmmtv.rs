//! # Hyper BMMTV extension
//!
//! This is a Reduction of Knowledge (RoK) from Multilinear Polynomial Evaluations (MPE) to
//! Univariate Polynomial Evaluations (UPE), that allows committing to multilinear polynomials
//! using normal Bmmtv

use std::{borrow::Borrow, marker::PhantomData, sync::Arc};

use crate::{
    field::JoltField,
    msm::Icicle,
    poly::{
        commitment::{
            bmmtv::poly_commit::{OpeningProof, UnivariatePolynomialCommitment},
            commitment_scheme::CommitmentScheme,
            kzg::{KZGProverKey, KZGVerifierKey, SRS},
        },
        multilinear_polynomial::MultilinearPolynomial,
        unipoly::UniPoly,
    },
    utils::{
        errors::ProofVerifyError,
        transcript::{AppendToTranscript, Transcript},
    },
};
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::Zero;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use rayon::prelude::*;

#[derive(Clone)]
pub struct HyperBmmtv<P: Pairing, ProofTranscript: Transcript> {
    _phantom: PhantomData<(P, ProofTranscript)>,
}

#[derive(PartialEq, Eq, Clone, CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct HyperBmmtvCommitment<P: Pairing>(PairingOutput<P>, Vec<P::G1>);

impl<P: Pairing> Default for HyperBmmtvCommitment<P> {
    fn default() -> Self {
        todo!()
    }
}

impl<P: Pairing> AppendToTranscript for HyperBmmtvCommitment<P> {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, _transcript: &mut ProofTranscript) {
        todo!()
    }
}

impl<F: JoltField> From<&MultilinearPolynomial<F>> for UniPoly<F> {
    fn from(poly: &MultilinearPolynomial<F>) -> Self {
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

        UniPoly::from_coeff(field_elements)
    }
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct HyperBmmtvProof<P: Pairing> {
    /// Opening proof for the committed polynomial with the evaluation
    opening: (OpeningProof<P>, P::ScalarField),
    /// All the commitments and opening proofs for sub polynomials
    ///
    /// Doesn't contain the initial commit
    sub_polynomials_proof: Vec<(PairingOutput<P>, OpeningProof<P>, P::ScalarField)>,
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

    type Proof = HyperBmmtvProof<P>;

    type BatchedProof = ();

    #[tracing::instrument(skip_all, name = "HyperBmmtv::setup")]
    fn setup(max_len: usize) -> Self::Setup {
        let mut rng = ChaCha20Rng::from_seed(*b"HyperBMMTV_POLY_COMMITMENTSCHEME");
        let srs =
            UnivariatePolynomialCommitment::<P, ProofTranscript>::setup(&mut rng, max_len - 1)
                .unwrap();
        let powers_len = srs.g1_powers.len();

        SRS::trim(Arc::new(srs), powers_len - 1)
    }

    #[tracing::instrument(skip_all, name = "HyperBmmtv::commit")]
    fn commit(
        poly: &MultilinearPolynomial<Self::Field>,
        (p_srs, _): &Self::Setup,
    ) -> Self::Commitment {
        let unipoly: UniPoly<Self::Field> = poly.into();
        let commitment =
            UnivariatePolynomialCommitment::<P, ProofTranscript>::commit(p_srs, &unipoly).unwrap();
        HyperBmmtvCommitment(commitment.0, commitment.1)
    }

    #[tracing::instrument(skip_all, name = "HyperBmmtv::batch_commit")]
    fn batch_commit<U>(polys: &[U], gens: &Self::Setup) -> Vec<Self::Commitment>
    where
        U: Borrow<MultilinearPolynomial<Self::Field>> + Sync,
    {
        polys
            .iter()
            .map(|poly| Self::commit(poly.borrow(), gens))
            .collect()
    }

    #[tracing::instrument(skip_all, name = "HyperBmmtv::prove")]
    fn prove(
        (p_srs, _): &Self::Setup,
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

        // Convert from Multilinear to UniPoly
        let mut polys: Vec<UniPoly<P::ScalarField>> = Vec::with_capacity(ell);
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

        // Todo: Batch commit
        let com_list: Vec<_> = polys[1..]
            .iter()
            .map(|poly| {
                UnivariatePolynomialCommitment::<P, ProofTranscript>::commit(p_srs, poly).unwrap()
            })
            .collect();

        // Phase 2
        // We do not need to add x to the transcript, because in our context x was obtained from the transcript.
        // We also do not need to absorb `C` and `eval` as they are already absorbed by the transcript by the caller
        // CANT JUST PARALLELIZE transcript is &mut
        com_list.iter().for_each(|g| {
            // Todo: Verify this.
            // transcript.append_points(&g.1);  kVec<G1> is not needed since it's used only for opening
            transcript.append_serializable(&g.0)
        });

        let r: <P as Pairing>::ScalarField = transcript.challenge_scalar();
        // Phase 3 -- create response
        // open all commits

        // TODO How do I get this from the Commitment?
        let (_pairing, kzg_comms) =
            UnivariatePolynomialCommitment::<P, ProofTranscript>::commit(p_srs, &polys[0]).unwrap();

        let eval = polys[0].evaluate(&r);
        let opening =
            UnivariatePolynomialCommitment::open(p_srs, &polys[0], kzg_comms, &r, transcript)
                .unwrap();

        // Todo: Batch opening
        let eval_proofs = com_list
            .into_iter()
            .zip(&polys[1..])
            .map(|(comm, polynomial)| {
                let eval = polynomial.evaluate(&r);
                (
                    comm.0, // pairing
                    UnivariatePolynomialCommitment::open(p_srs, polynomial, comm.1, &r, transcript)
                        .unwrap(), // opening
                    eval,
                )
            })
            .collect::<Vec<_>>();

        HyperBmmtvProof {
            opening: (opening, eval),
            sub_polynomials_proof: eval_proofs,
        }
    }

    #[tracing::instrument(skip_all, name = "HyperBmmtv::verify")]
    fn verify(
        proof: &Self::Proof,
        (_, v_srs): &Self::Setup,
        transcript: &mut ProofTranscript,
        opening_point: &[Self::Field], // point at which the polynomial is evaluated
        _opening: &Self::Field,        // evaluation \widetilde{Z}(r)
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        let ell = opening_point.len();

        // sub polynomials + original reinterpreted
        if proof.sub_polynomials_proof.len() + 1 != ell {
            return Err(ProofVerifyError::InternalError);
        }

        proof.sub_polynomials_proof.iter().for_each(|g| {
            // Todo: Verify this.
            // transcript.append_points(&g.1);  kVec<G1> is not needed since it's used only for opening
            transcript.append_serializable(&g.0)
        });

        let r: P::ScalarField = transcript.challenge_scalar();

        if r == P::ScalarField::zero() {
            return Err(ProofVerifyError::InternalError);
        }

        let n = 1 << ell; // n = 2^ell

        let original_poly = UnivariatePolynomialCommitment::verify(
            v_srs,
            n - 1, // degree = len(coeff) - 1
            commitment.0,
            r,
            proof.opening.1,
            &proof.opening.0,
            transcript,
        )
        .map_err(|_| ProofVerifyError::InternalError)?;
        if !original_poly {
            // failed first check for initial poly
            return Err(ProofVerifyError::InternalError);
        }

        // Todo: Consistency check between generated polynomials and evaluations/commitment

        let eval_proofs = proof
            .sub_polynomials_proof
            .iter()
            .map(|(comm, proof, eval)| {
                // Todo: Bmmtv uses turns a Univariate polynomial in a matrix
                // The number of rows and columns is currently defined by the setup params
                // hence, we are not getting maximal performance
                //
                // let n = n >> (i + 1);
                UnivariatePolynomialCommitment::verify(
                    v_srs,
                    n - 1,
                    *comm,
                    r,
                    *eval,
                    proof,
                    transcript,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| ProofVerifyError::InternalError)?;

        if eval_proofs.into_iter().all(|verified| verified) {
            Ok(())
        } else {
            Err(ProofVerifyError::InternalError)
        }
    }

    fn protocol_name() -> &'static [u8] {
        b"hyperbmmtv"
    }
}

#[cfg(test)]
mod tests {
    use crate::poly::commitment::{
        bmmtv::hyperbmmtv::HyperBmmtv, commitment_scheme::CommitmentScheme,
    };
    use crate::poly::multilinear_polynomial::MultilinearPolynomial;
    use crate::poly::multilinear_polynomial::PolynomialEvaluation;
    use crate::utils::transcript::KeccakTranscript;
    use crate::utils::transcript::Transcript;
    use ark_bn254::Bn254;
    use ark_ec::pairing::Pairing;
    use ark_std::UniformRand;
    use rand_core::SeedableRng;

    #[test]
    fn test_hyper_bmmtv() {
        type HyperTest = HyperBmmtv<Bn254, KeccakTranscript>;
        let ell = 4;
        let n = 1 << ell; // n = 2^ell
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
        let poly_raw = (0..n)
            .map(|_| <Bn254 as Pairing>::ScalarField::rand(&mut rng))
            .collect::<Vec<_>>();
        let poly = MultilinearPolynomial::from(poly_raw.clone());

        let setup = HyperTest::setup(poly.len());

        let commit = HyperTest::commit(&poly, &setup);

        let point = (0..ell)
            .map(|_| <Bn254 as Pairing>::ScalarField::rand(&mut rng))
            .collect::<Vec<_>>();

        let mut transcript = KeccakTranscript::new(b"TestEval");

        let proof = HyperTest::prove(&setup, &poly, &point, &mut transcript);

        let mut transcript = KeccakTranscript::new(b"TestEval");

        let opening = poly.evaluate(&point);
        HyperTest::verify(&proof, &setup, &mut transcript, &point, &opening, &commit).unwrap();
    }
}
