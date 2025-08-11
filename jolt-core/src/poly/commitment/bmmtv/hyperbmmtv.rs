//! # Hyper BMMTV extension
//!
//! This is a Reduction of Knowledge (RoK) from Multilinear Polynomial Evaluations (MPE) to
//! Univariate Polynomial Evaluations (UPE), that allows committing to multilinear polynomials
//! using normal Bmmtv

use crate::transcript::{AppendToTranscript, Transcript};
use crate::{
    field::JoltField,
    poly::{
        commitment::{
            bmmtv::poly_commit::{OpeningProof, UnivariatePolynomialCommitment},
            commitment_scheme::CommitmentScheme,
            kzg::{KZGProverKey, KZGVerifierKey, SRS},
        },
        multilinear_polynomial::MultilinearPolynomial,
        unipoly::UniPoly,
    },
    utils::errors::ProofVerifyError,
};
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{One, Zero};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use rayon::prelude::*;
use std::{borrow::Borrow, marker::PhantomData, sync::Arc};

#[derive(Clone)]
pub struct HyperBmmtv<P: Pairing> {
    _phantom: PhantomData<P>,
}

#[derive(PartialEq, Eq, Clone, CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct HyperBmmtvCommitment<P: Pairing>(PairingOutput<P>, Vec<P::G1>);

impl<P: Pairing> Default for HyperBmmtvCommitment<P> {
    fn default() -> Self {
        HyperBmmtvCommitment(PairingOutput::default(), vec![])
    }
}

impl<P: Pairing> AppendToTranscript for HyperBmmtvCommitment<P> {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, transcript: &mut ProofTranscript) {
        transcript.append_serializable(&self.0);
        // Vec<G1> is only used by the prover
        // transcript.append_points(&self.1);
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
    /// Commitments for sub polynomials
    sub_polynomials_commitments: Vec<PairingOutput<P>>,
    /// Opening proofs for all polynomials (including original)
    polynomials_proof: Vec<SubProof<P>>,
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct SubProof<P: Pairing> {
    proof: OpeningProof<P>,
    y_pos: P::ScalarField,
    y_neg: P::ScalarField,
    y: P::ScalarField,
}

impl<P: Pairing> CommitmentScheme for HyperBmmtv<P>
where
    PairingOutput<P>: CanonicalSerialize,
    P::ScalarField: JoltField,
{
    type Field = P::ScalarField;
    type ProverSetup = KZGProverKey<P>;
    type VerifierSetup = KZGVerifierKey<P>;
    type Commitment = HyperBmmtvCommitment<P>;
    type Proof = HyperBmmtvProof<P>;
    type BatchedProof = ();
    type OpeningProofHint = ();

    #[tracing::instrument(skip_all, name = "HyperBmmtv::setup")]
    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup {
        let mut rng = ChaCha20Rng::from_seed(*b"HyperBMMTV_POLY_COMMITMENTSCHEME");
        let srs =
            UnivariatePolynomialCommitment::<P>::setup(&mut rng, (1 << max_num_vars) - 1).unwrap();
        let powers_len = srs.g1_powers.len();

        SRS::trim(Arc::new(srs), powers_len - 1).0
    }

    #[tracing::instrument(skip_all, name = "HyperBmmtv::setup_verifier")]
    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        KZGVerifierKey::<P>::from(setup)
    }

    #[tracing::instrument(skip_all, name = "HyperBmmtv::commit")]
    fn commit(
        poly: &MultilinearPolynomial<Self::Field>,
        setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        let unipoly: UniPoly<Self::Field> = poly.into();
        let commitment = UnivariatePolynomialCommitment::<P>::commit(setup, &unipoly).unwrap();
        let commitment = HyperBmmtvCommitment(commitment.0, commitment.1);
        (commitment, ())
    }

    #[tracing::instrument(skip_all, name = "HyperBmmtv::batch_commit")]
    fn batch_commit<U>(polys: &[U], gens: &Self::ProverSetup) -> Vec<Self::Commitment>
    where
        U: Borrow<MultilinearPolynomial<Self::Field>> + Sync,
    {
        polys
            .par_iter()
            .map(|poly| Self::commit(poly.borrow(), gens).0)
            .collect()
    }

    #[tracing::instrument(skip_all, name = "HyperBmmtv::prove")]
    fn prove<ProofTranscript: Transcript>(
        setup: &Self::ProverSetup,
        poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[Self::Field], // point at which the polynomial is evaluated
        _: Self::OpeningProofHint,
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        let ell = opening_point.len();
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
            let Pi_len = previous_poly.coeffs.len() / 2;
            let mut Pi = vec![P::ScalarField::zero(); Pi_len];
            let x = opening_point[ell - i - 1];

            Pi.par_iter_mut().enumerate().for_each(|(j, Pi_j)| {
                let Peven = previous_poly[2 * j + 1];
                let Podd = previous_poly[2 * j];
                *Pi_j = x * (Peven - Podd) + Podd;
            });

            polys.push(UniPoly::from_coeff(Pi));
        }

        assert_eq!(polys.len(), ell);
        assert_eq!(polys[ell - 1].coeffs.len(), 2);

        // We do not need to commit to the first polynomial as it is already committed.

        // Todo: Batch commit
        let com_list: Vec<_> = polys[1..]
            .par_iter()
            .map(|poly| UnivariatePolynomialCommitment::<P>::commit(setup, poly).unwrap())
            .collect();

        // Phase 2
        // We do not need to add x to the transcript, because in our context x was obtained from the transcript.
        // We also do not need to absorb `C` and `eval` as they are already absorbed by the transcript by the caller
        com_list
            .iter()
            .for_each(|g| transcript.append_serializable(&g.0));

        let r: <P as Pairing>::ScalarField = transcript.challenge_scalar();

        // Phase 3 -- create response
        // open all commits

        // TODO Get this from the Commitment
        let (_, kzg_comms) = UnivariatePolynomialCommitment::<P>::commit(setup, &polys[0]).unwrap();

        let (sub_polynomials_commitments, com_list): (Vec<_>, Vec<_>) =
            com_list.into_iter().unzip();

        // Todo: Batch opening
        let eval_proofs = std::iter::once(kzg_comms)
            .chain(com_list)
            .zip(&polys)
            .map(|(comm, polynomial)| {
                let y_pos = polynomial.evaluate(&r);
                let y_neg = polynomial.evaluate(&-r);
                let y = polynomial.evaluate(&r.square());
                SubProof {
                    proof: UnivariatePolynomialCommitment::open(
                        setup, polynomial, comm, &r, transcript,
                    )
                    .unwrap(), // opening
                    y,
                    y_pos,
                    y_neg,
                }
            })
            .collect::<Vec<_>>();

        HyperBmmtvProof {
            sub_polynomials_commitments,
            polynomials_proof: eval_proofs,
        }
    }

    #[tracing::instrument(skip_all, name = "HyperBmmtv::verify")]
    fn verify<ProofTranscript: Transcript>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[Self::Field], // point at which the polynomial is evaluated
        opening: &Self::Field,         // evaluation \widetilde{Z}(r)
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        let ell = opening_point.len();

        if proof.polynomials_proof.len() != ell {
            return Err(ProofVerifyError::InternalError);
        }

        // only the sub polynomials commitments are in the proof
        if proof.sub_polynomials_commitments.len() != ell - 1 {
            return Err(ProofVerifyError::InternalError);
        }

        proof
            .sub_polynomials_commitments
            .iter()
            .for_each(|g| transcript.append_serializable(g));

        let r: P::ScalarField = transcript.challenge_scalar();

        if r == P::ScalarField::zero() {
            return Err(ProofVerifyError::InternalError);
        }

        // Consistency check between generated polynomials and evaluations/commitment
        let mut Y = proof
            .polynomials_proof
            .iter()
            .map(|proof| proof.y)
            .collect::<Vec<_>>();
        Y.push(*opening);

        let two = P::ScalarField::from(2u64);
        for (i, SubProof { y_pos, y_neg, .. }) in proof.polynomials_proof.iter().enumerate() {
            let x = opening_point[ell - i - 1];
            let y_pos = *y_pos;
            let y_neg = *y_neg;
            let y_next = Y[i + 1];
            if two * r * y_next
                != (r * (P::ScalarField::one() - x) * (y_pos + y_neg)) + (x * (y_pos - y_neg))
            {
                return Err(ProofVerifyError::InternalError);
            }
            // Note that we don't make any checks about Y[0] here, but our batching
            // check below requires it
        }

        let eval_proofs = proof
            .polynomials_proof
            .iter()
            .zip(std::iter::once(&commitment.0).chain(proof.sub_polynomials_commitments.iter()))
            .enumerate()
            .map(
                |(
                    i,
                    (
                        SubProof {
                            proof, y_pos: eval, .. // y_pos because we want eval for r
                        },
                        commitment,
                    ),
                )| {
                    let n = 1 << (ell - i);
                    UnivariatePolynomialCommitment::verify(
                        setup,
                        n - 1, // degree = len(coeff) - 1
                        *commitment,
                        r,
                        *eval,
                        proof,
                        transcript,
                    )
                },
            )
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
    use crate::transcript::KeccakTranscript;
    use crate::transcript::Transcript;
    use ark_bn254::Bn254;
    use ark_ec::pairing::Pairing;
    use ark_std::UniformRand;
    use rand_core::SeedableRng;

    #[test]
    fn test_hyper_bmmtv() {
        type HyperTest = HyperBmmtv<Bn254>;
        let ell = 4;
        let n = 1 << ell; // n = 2^ell
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
        let poly_raw = (0..n)
            .map(|_| <Bn254 as Pairing>::ScalarField::rand(&mut rng))
            .collect::<Vec<_>>();

        let poly = MultilinearPolynomial::from(poly_raw.clone());

        let setup = HyperTest::setup_prover(poly.len());
        let verifier_setup = HyperTest::setup_verifier(&setup);

        let commit = HyperTest::commit(&poly, &setup).0;

        let point = (0..ell)
            .map(|_| <Bn254 as Pairing>::ScalarField::rand(&mut rng))
            .collect::<Vec<_>>();

        let mut prover_transcript = KeccakTranscript::new(b"TestEval");

        let proof = HyperTest::prove(&setup, &poly, &point, (), &mut prover_transcript);

        let mut verifier_transcript = KeccakTranscript::new(b"TestEval");
        verifier_transcript.compare_to(prover_transcript);

        let opening = poly.evaluate(&point);
        HyperTest::verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            &point,
            &opening,
            &commit,
        )
        .unwrap();
    }
}
