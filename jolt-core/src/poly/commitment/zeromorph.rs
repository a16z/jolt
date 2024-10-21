#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use std::{iter, marker::PhantomData};

use crate::field;
use crate::msm::VariableBaseMSM;
use crate::poly::{dense_mlpoly::DensePolynomial, unipoly::UniPoly};
use crate::utils::mul_0_1_optimized;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::utils::{
    errors::ProofVerifyError,
    transcript::{AppendToTranscript, Transcript},
};
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_ff::{batch_inversion, Field};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{One, Zero};
use itertools::izip;
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};
use rand_core::{CryptoRng, RngCore};
use std::sync::Arc;
use tracing::trace_span;

use rayon::prelude::*;

use super::{
    commitment_scheme::{BatchType, CommitShape, CommitmentScheme},
    kzg::{KZGProverKey, KZGVerifierKey, UnivariateKZG, SRS},
};

pub struct ZeromorphSRS<P: Pairing>(Arc<SRS<P>>);

impl<P: Pairing> ZeromorphSRS<P> {
    pub fn setup<R: RngCore + CryptoRng>(rng: &mut R, max_degree: usize) -> Self {
        Self(Arc::new(SRS::setup(rng, max_degree, max_degree)))
    }

    pub fn trim(self, max_degree: usize) -> (ZeromorphProverKey<P>, ZeromorphVerifierKey<P>) {
        let (commit_pp, kzg_vk) = SRS::trim(self.0.clone(), max_degree);
        let offset = self.0.g1_powers.len() - max_degree;
        let tau_N_max_sub_2_N = self.0.g2_powers[offset];
        let open_pp = KZGProverKey::new(self.0, offset, max_degree);
        (
            ZeromorphProverKey { commit_pp, open_pp },
            ZeromorphVerifierKey {
                kzg_vk,
                tau_N_max_sub_2_N,
            },
        )
    }
}

//TODO: adapt interface to have prover and verifier key
#[derive(Clone, Debug)]
pub struct ZeromorphProverKey<P: Pairing> {
    pub commit_pp: KZGProverKey<P>,
    pub open_pp: KZGProverKey<P>,
}

#[derive(Copy, Clone, Debug)]
pub struct ZeromorphVerifierKey<P: Pairing> {
    pub kzg_vk: KZGVerifierKey<P>,
    pub tau_N_max_sub_2_N: P::G2Affine,
}

#[derive(Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct ZeromorphCommitment<P: Pairing>(P::G1Affine);

impl<P: Pairing> Default for ZeromorphCommitment<P> {
    fn default() -> Self {
        Self(P::G1Affine::zero())
    }
}

impl<P: Pairing> AppendToTranscript for ZeromorphCommitment<P> {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, transcript: &mut ProofTranscript) {
        transcript.append_point(&self.0.into_group());
    }
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct ZeromorphProof<P: Pairing> {
    pub pi: P::G1Affine,
    pub q_hat_com: P::G1Affine,
    pub q_k_com: Vec<P::G1Affine>,
}

fn compute_multilinear_quotients<P: Pairing>(
    poly: &DensePolynomial<P::ScalarField>,
    point: &[P::ScalarField],
) -> (Vec<UniPoly<P::ScalarField>>, P::ScalarField)
where
    <P as Pairing>::ScalarField: field::JoltField,
{
    let num_var = poly.get_num_vars();
    assert_eq!(num_var, point.len());

    let mut remainder = poly.Z.to_vec();
    let mut quotients: Vec<_> = point
        .iter()
        .enumerate()
        .map(|(i, x_i)| {
            let (remainder_lo, remainder_hi) = remainder.split_at_mut(1 << (num_var - 1 - i));
            let mut quotient = vec![P::ScalarField::zero(); remainder_lo.len()];

            quotient
                .par_iter_mut()
                .zip(&*remainder_lo)
                .zip(&*remainder_hi)
                .for_each(|((q, r_lo), r_hi)| {
                    *q = *r_hi - *r_lo;
                });

            remainder_lo
                .par_iter_mut()
                .zip(remainder_hi)
                .for_each(|(r_lo, r_hi)| {
                    *r_lo += (*r_hi - *r_lo) * *x_i;
                });

            remainder.truncate(1 << (num_var - 1 - i));

            UniPoly::from_coeff(quotient)
        })
        .collect();
    quotients.reverse();
    (quotients, remainder[0])
}

// Compute the batched, lifted-degree quotient `\hat{q}`
fn compute_batched_lifted_degree_quotient<P: Pairing>(
    quotients: &[UniPoly<P::ScalarField>],
    y_challenge: &P::ScalarField,
) -> (UniPoly<P::ScalarField>, usize)
where
    <P as Pairing>::ScalarField: field::JoltField,
{
    let num_vars = quotients.len();

    // Compute \hat{q} = \sum_k y^k * X^{N - d_k - 1} * q_k
    let mut scalar = P::ScalarField::one(); // y^k

    // Rather than explicitly computing the shifts of q_k by N - d_k - 1 (i.e. multiplying q_k by X^{N - d_k - 1})
    // then accumulating them, we simply accumulate y^k*q_k into \hat{q} at the index offset N - d_k - 1
    let q_hat = quotients.iter().enumerate().fold(
        vec![P::ScalarField::zero(); 1 << num_vars],
        |mut q_hat, (idx, q)| {
            let q_hat_iter = q_hat[(1 << num_vars) - (1 << idx)..].par_iter_mut();
            q_hat_iter.zip(&q.coeffs).for_each(|(q_hat, q)| {
                *q_hat += scalar * *q;
            });
            scalar *= *y_challenge;
            q_hat
        },
    );

    (UniPoly::from_coeff(q_hat), 1 << (num_vars - 1))
}

fn eval_and_quotient_scalars<P: Pairing>(
    y_challenge: P::ScalarField,
    x_challenge: P::ScalarField,
    z_challenge: P::ScalarField,
    challenges: &[P::ScalarField],
) -> (P::ScalarField, (Vec<P::ScalarField>, Vec<P::ScalarField>))
where
    <P as Pairing>::ScalarField: field::JoltField,
{
    let num_vars = challenges.len();

    // squares of x = [x, x^2, .. x^{2^k}, .. x^{2^num_vars}]
    let squares_of_x: Vec<_> = iter::successors(Some(x_challenge), |&x| Some(x.square()))
        .take(num_vars + 1)
        .collect();

    let offsets_of_x = {
        let mut offsets_of_x = squares_of_x
            .iter()
            .rev()
            .skip(1)
            .scan(P::ScalarField::one(), |acc, pow_x| {
                *acc *= *pow_x;
                Some(*acc)
            })
            .collect::<Vec<_>>();
        offsets_of_x.reverse();
        offsets_of_x
    };

    let vs = {
        let v_numer = squares_of_x[num_vars] - P::ScalarField::one();
        let mut v_denoms = squares_of_x
            .iter()
            .map(|squares_of_x| *squares_of_x - P::ScalarField::one())
            .collect::<Vec<_>>();
        batch_inversion(&mut v_denoms);
        v_denoms
            .iter()
            .map(|v_denom| v_numer * *v_denom)
            .collect::<Vec<_>>()
    };

    let q_scalars = izip!(
        iter::successors(Some(P::ScalarField::one()), |acc| Some(*acc * y_challenge))
            .take(num_vars),
        offsets_of_x,
        squares_of_x,
        &vs,
        &vs[1..],
        challenges.iter().rev()
    )
    .map(|(power_of_y, offset_of_x, square_of_x, v_i, v_j, u_i)| {
        (
            -(power_of_y * offset_of_x),
            -(z_challenge * (square_of_x * *v_j - *u_i * *v_i)),
        )
    })
    .unzip();
    // -vs[0] * z = -z * (x^(2^num_vars) - 1) / (x - 1) = -z Φ_n(x)
    (-vs[0] * z_challenge, q_scalars)
}

#[derive(Clone)]
pub struct Zeromorph<P: Pairing, ProofTranscript: Transcript> {
    _phantom: PhantomData<(P, ProofTranscript)>,
}

impl<P, ProofTranscript> Zeromorph<P, ProofTranscript>
where
    <P as Pairing>::ScalarField: field::JoltField,
    P: Pairing,
    ProofTranscript: Transcript,
{
    pub fn protocol_name() -> &'static [u8] {
        b"Zeromorph"
    }

    pub fn commit(
        pp: &ZeromorphProverKey<P>,
        poly: &DensePolynomial<P::ScalarField>,
    ) -> Result<ZeromorphCommitment<P>, ProofVerifyError> {
        if pp.commit_pp.g1_powers().len() < poly.Z.len() {
            return Err(ProofVerifyError::KeyLengthError(
                pp.commit_pp.g1_powers().len(),
                poly.Z.len(),
            ));
        }
        Ok(ZeromorphCommitment(
            UnivariateKZG::commit(&pp.commit_pp, &UniPoly::from_coeff(poly.Z.clone())).unwrap(),
        ))
    }

    #[tracing::instrument(skip_all, name = "Zeromorph::open")]
    pub fn open(
        pp: &ZeromorphProverKey<P>,
        poly: &DensePolynomial<P::ScalarField>,
        point: &[P::ScalarField],
        // Can be calculated
        eval: &P::ScalarField,
        transcript: &mut ProofTranscript,
    ) -> Result<ZeromorphProof<P>, ProofVerifyError> {
        let protocol_name = Self::protocol_name();
        transcript.append_message(protocol_name);

        if pp.commit_pp.g1_powers().len() < poly.Z.len() {
            return Err(ProofVerifyError::KeyLengthError(
                pp.commit_pp.g1_powers().len(),
                poly.Z.len(),
            ));
        }

        assert_eq!(poly.evaluate(point), *eval);

        let (quotients, remainder): (Vec<UniPoly<P::ScalarField>>, P::ScalarField) =
            compute_multilinear_quotients::<P>(poly, point);
        assert_eq!(quotients.len(), poly.get_num_vars());
        assert_eq!(remainder, *eval);

        // Compute the multilinear quotients q_k = q_k(X_0, ..., X_{k-1})
        let q_k_com: Vec<P::G1Affine> = quotients
            .par_iter()
            .map(|q| UnivariateKZG::commit(&pp.commit_pp, q).unwrap())
            .collect();
        let q_comms: Vec<P::G1> = q_k_com.par_iter().map(|c| c.into_group()).collect();
        q_comms.iter().for_each(|c| transcript.append_point(c));

        // Sample challenge y
        let y_challenge: P::ScalarField = transcript.challenge_scalar();

        // Compute the batched, lifted-degree quotient `\hat{q}`
        // qq_hat = ∑_{i=0}^{num_vars-1} y^i * X^(2^num_vars - d_k - 1) * q_i(x)
        let (q_hat, offset) = compute_batched_lifted_degree_quotient::<P>(&quotients, &y_challenge);

        // Compute and absorb the commitment C_q = [\hat{q}]
        let q_hat_com = UnivariateKZG::commit_offset(&pp.commit_pp, &q_hat, offset)?;
        transcript.append_point(&q_hat_com.into_group());

        // Get x and z challenges
        let x_challenge = transcript.challenge_scalar();
        let z_challenge = transcript.challenge_scalar();

        // Compute batched degree and ZM-identity quotient polynomial pi
        let (eval_scalar, (degree_check_q_scalars, zmpoly_q_scalars)): (
            P::ScalarField,
            (Vec<P::ScalarField>, Vec<P::ScalarField>),
        ) = eval_and_quotient_scalars::<P>(y_challenge, x_challenge, z_challenge, point);
        // f = z * poly.Z + q_hat + (-z * Φ_n(x) * e) + ∑_k (q_scalars_k * q_k)
        let mut f = UniPoly::from_coeff(poly.Z.clone());
        f *= &z_challenge;
        f += &q_hat;
        f[0] += eval_scalar * *eval;
        quotients
            .into_iter()
            .zip(degree_check_q_scalars)
            .zip(zmpoly_q_scalars)
            .for_each(|((mut q, degree_check_scalar), zm_poly_scalar)| {
                q *= &(degree_check_scalar + zm_poly_scalar);
                f += &q;
            });
        debug_assert_eq!(f.evaluate(&x_challenge), P::ScalarField::zero());

        // Compute and send proof commitment pi
        let (pi, _) = UnivariateKZG::open(&pp.open_pp, &f, &x_challenge)?;

        Ok(ZeromorphProof {
            pi,
            q_hat_com,
            q_k_com,
        })
    }

    #[tracing::instrument(skip_all, name = "Zeromorph::batch_open")]
    fn batch_open(
        pk: &ZeromorphProverKey<P>,
        polynomials: &[&DensePolynomial<P::ScalarField>],
        point: &[P::ScalarField],
        evals: &[P::ScalarField],
        transcript: &mut ProofTranscript,
    ) -> ZeromorphProof<P> {
        let num_vars = point.len();
        let n = 1 << num_vars;

        // Generate batching challenge \rho and powers 1,...,\rho^{m-1}
        let rho: P::ScalarField = transcript.challenge_scalar();
        let mut rho_powers = vec![P::ScalarField::one()];
        for i in 1..polynomials.len() {
            rho_powers.push(rho_powers[i - 1] * rho);
        }

        // Compute batching of unshifted polynomials f_i, and batched eval v_i:
        let batched_evaluation = rho_powers
            .iter()
            .zip(evals.iter())
            .map(|(scalar, eval)| *scalar * *eval)
            .sum();

        let span = trace_span!("f_batched");
        let enter = span.enter();
        let num_chunks = rayon::current_num_threads().next_power_of_two();
        let chunk_size = n / num_chunks;
        let f_batched = (0..num_chunks)
            .into_par_iter()
            .flat_map_iter(|chunk_index| {
                let mut chunk = unsafe_allocate_zero_vec::<P::ScalarField>(chunk_size);
                for (coeff, poly) in rho_powers.iter().zip(polynomials.iter()) {
                    for (rlc, poly_eval) in chunk
                        .iter_mut()
                        .zip(poly.evals_ref()[chunk_index * chunk_size..].iter())
                    {
                        *rlc += mul_0_1_optimized(poly_eval, coeff);
                    }
                }
                chunk
            })
            .collect::<Vec<_>>();
        drop(enter);
        drop(span);

        let poly = DensePolynomial::new(f_batched);
        Zeromorph::<P, ProofTranscript>::open(pk, &poly, point, &batched_evaluation, transcript)
            .unwrap()
    }

    fn batch_verify(
        vk: &ZeromorphVerifierKey<P>,
        commitments: &[&ZeromorphCommitment<P>],
        point: &[P::ScalarField],
        evals: &[P::ScalarField],
        batch_proof: &ZeromorphProof<P>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        //TODO(pat): produce powers in parallel using window method
        // Compute batching of unshifted polynomials f_i:
        // Compute powers of batching challenge rho
        let rho: P::ScalarField = transcript.challenge_scalar();
        let mut scalar = P::ScalarField::one();
        let (batched_eval, batched_commitment) = evals.iter().zip(commitments.iter()).fold(
            (P::ScalarField::zero(), P::G1::zero()),
            |(mut batched_evaluation, mut batched_commitment), (opening, commitment)| {
                batched_evaluation += scalar * *opening;
                batched_commitment += commitment.0 * scalar;
                scalar *= rho;
                (batched_evaluation, batched_commitment)
            },
        );
        Zeromorph::<P, ProofTranscript>::verify(
            vk,
            &ZeromorphCommitment(batched_commitment.into_affine()),
            point,
            &batched_eval,
            batch_proof,
            transcript,
        )
    }

    pub fn verify(
        vk: &ZeromorphVerifierKey<P>,
        comm: &ZeromorphCommitment<P>,
        point: &[P::ScalarField],
        eval: &P::ScalarField,
        proof: &ZeromorphProof<P>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let protocol_name = Self::protocol_name();
        transcript.append_message(protocol_name);

        let q_comms: Vec<P::G1> = proof.q_k_com.iter().map(|c| c.into_group()).collect();
        q_comms.iter().for_each(|c| transcript.append_point(c));

        // Challenge y
        let y_challenge: P::ScalarField = transcript.challenge_scalar();

        // Receive commitment C_q_hat
        transcript.append_point(&proof.q_hat_com.into_group());

        // Get x and z challenges
        let x_challenge = transcript.challenge_scalar();
        let z_challenge = transcript.challenge_scalar();

        // Compute batched degree and ZM-identity quotient polynomial pi
        let (eval_scalar, (mut q_scalars, zmpoly_q_scalars)): (
            P::ScalarField,
            (Vec<P::ScalarField>, Vec<P::ScalarField>),
        ) = eval_and_quotient_scalars::<P>(y_challenge, x_challenge, z_challenge, point);
        q_scalars
            .iter_mut()
            .zip(zmpoly_q_scalars)
            .for_each(|(scalar, zm_poly_q_scalar)| {
                *scalar += zm_poly_q_scalar;
            });
        let scalars = [
            vec![P::ScalarField::one(), z_challenge, eval_scalar * *eval],
            q_scalars,
        ]
        .concat();
        let bases = [
            vec![proof.q_hat_com, comm.0, vk.kzg_vk.g1],
            proof.q_k_com.clone(),
        ]
        .concat();
        let zeta_z_com = <P::G1 as VariableBaseMSM>::msm(&bases, &scalars)
            .unwrap()
            .into_affine();

        // e(pi, [tau]_2 - x * [1]_2) == e(C_{\zeta,Z}, -[X^(N_max - 2^n - 1)]_2) <==> e(C_{\zeta,Z} - x * pi, [X^{N_max - 2^n - 1}]_2) * e(-pi, [tau_2]) == 1
        let pairing = P::multi_pairing(
            [zeta_z_com, proof.pi],
            [
                (-vk.tau_N_max_sub_2_N.into_group()).into_affine(),
                (vk.kzg_vk.beta_g2.into_group() - (vk.kzg_vk.g2 * x_challenge)).into(),
            ],
        );
        if pairing.is_zero() {
            Ok(())
        } else {
            Err(ProofVerifyError::InternalError)
        }
    }
}

impl<P: Pairing, ProofTranscript: Transcript> CommitmentScheme<ProofTranscript>
    for Zeromorph<P, ProofTranscript>
where
    <P as Pairing>::ScalarField: field::JoltField,
{
    type Field = P::ScalarField;
    type Setup = (ZeromorphProverKey<P>, ZeromorphVerifierKey<P>);
    type Commitment = ZeromorphCommitment<P>;
    type Proof = ZeromorphProof<P>;
    type BatchedProof = ZeromorphProof<P>;

    fn setup(shapes: &[CommitShape]) -> Self::Setup {
        let max_len = shapes.iter().map(|shape| shape.input_length).max().unwrap();

        ZeromorphSRS(Arc::new(SRS::setup(
            &mut ChaCha20Rng::from_seed(*b"ZEROMORPH_POLY_COMMITMENT_SCHEME"),
            max_len,
            max_len,
        )))
        .trim(max_len)
    }

    fn commit(poly: &DensePolynomial<Self::Field>, setup: &Self::Setup) -> Self::Commitment {
        assert!(
            setup.0.commit_pp.g1_powers().len() > poly.Z.len(),
            "COMMIT KEY LENGTH ERROR {}, {}",
            setup.0.commit_pp.g1_powers().len(),
            poly.Z.len()
        );
        ZeromorphCommitment(
            UnivariateKZG::commit(&setup.0.commit_pp, &UniPoly::from_coeff(poly.Z.clone()))
                .unwrap(),
        )
    }

    fn batch_commit(
        evals: &[&[Self::Field]],
        gens: &Self::Setup,
        _batch_type: BatchType,
    ) -> Vec<Self::Commitment> {
        // TODO: assert lengths are valid
        evals
            .par_iter()
            .map(|evals| {
                assert!(
                    gens.0.commit_pp.g1_powers().len() > evals.len(),
                    "COMMIT KEY LENGTH ERROR {}, {}",
                    gens.0.commit_pp.g1_powers().len(),
                    evals.len()
                );
                ZeromorphCommitment(
                    UnivariateKZG::commit(&gens.0.commit_pp, &UniPoly::from_coeff(evals.to_vec()))
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>()
    }

    fn commit_slice(evals: &[Self::Field], setup: &Self::Setup) -> Self::Commitment {
        ZeromorphCommitment(
            UnivariateKZG::commit(&setup.0.commit_pp, &UniPoly::from_coeff(evals.to_vec()))
                .unwrap(),
        )
    }

    fn prove(
        setup: &Self::Setup,
        poly: &DensePolynomial<Self::Field>,
        opening_point: &[Self::Field], // point at which the polynomial is evaluated
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        let eval = poly.evaluate(opening_point);
        Zeromorph::<P, ProofTranscript>::open(&setup.0, poly, opening_point, &eval, transcript)
            .unwrap()
    }

    fn batch_prove(
        setup: &Self::Setup,
        polynomials: &[&DensePolynomial<Self::Field>],
        opening_point: &[Self::Field],
        openings: &[Self::Field],
        _batch_type: BatchType,
        transcript: &mut ProofTranscript,
    ) -> Self::BatchedProof {
        Zeromorph::<P, ProofTranscript>::batch_open(
            &setup.0,
            polynomials,
            opening_point,
            openings,
            transcript,
        )
    }

    fn combine_commitments(
        commitments: &[&Self::Commitment],
        coeffs: &[Self::Field],
    ) -> Self::Commitment {
        let combined_commitment: P::G1 = commitments
            .iter()
            .zip(coeffs.iter())
            .map(|(commitment, coeff)| commitment.0 * coeff)
            .sum();
        ZeromorphCommitment(combined_commitment.into_affine())
    }

    fn verify(
        proof: &Self::Proof,
        setup: &Self::Setup,
        transcript: &mut ProofTranscript,
        opening_point: &[Self::Field], // point at which the polynomial is evaluated
        opening: &Self::Field,         // evaluation \widetilde{Z}(r)
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        Zeromorph::<P, ProofTranscript>::verify(
            &setup.1,
            commitment,
            opening_point,
            opening,
            proof,
            transcript,
        )
    }

    fn batch_verify(
        batch_proof: &Self::BatchedProof,
        setup: &Self::Setup,
        opening_point: &[Self::Field],
        openings: &[Self::Field],
        commitments: &[&Self::Commitment],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        Zeromorph::<P, ProofTranscript>::batch_verify(
            &setup.1,
            commitments,
            opening_point,
            openings,
            batch_proof,
            transcript,
        )
    }

    fn protocol_name() -> &'static [u8] {
        b"zeromorph"
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::utils::math::Math;
    use crate::utils::transcript::{KeccakTranscript, Transcript};
    use ark_bn254::{Bn254, Fr};
    use ark_ff::{BigInt, Zero};
    use ark_std::{test_rng, UniformRand};
    use rand_core::SeedableRng;

    // Evaluate Phi_k(x) = \sum_{i=0}^k x^i using the direct inefficent formula
    fn phi<P: Pairing>(challenge: &P::ScalarField, subscript: usize) -> P::ScalarField {
        let len = (1 << subscript) as u64;
        (0..len).fold(P::ScalarField::zero(), |mut acc, i| {
            //Note this is ridiculous DevX
            acc += challenge.pow(BigInt::<1>::from(i));
            acc
        })
    }

    /// Test for computing qk given multilinear f
    /// Given 𝑓(𝑋₀, …, 𝑋ₙ₋₁), and `(𝑢, 𝑣)` such that \f(\u) = \v, compute `qₖ(𝑋₀, …, 𝑋ₖ₋₁)`
    /// such that the following identity holds:
    ///
    /// `𝑓(𝑋₀, …, 𝑋ₙ₋₁) − 𝑣 = ∑ₖ₌₀ⁿ⁻¹ (𝑋ₖ − 𝑢ₖ) qₖ(𝑋₀, …, 𝑋ₖ₋₁)`
    #[test]
    fn quotient_construction() {
        // Define size params
        let num_vars = 4;
        let n: u64 = 1 << num_vars;

        // Construct a random multilinear polynomial f, and (u,v) such that f(u) = v
        let mut rng = test_rng();
        let multilinear_f =
            DensePolynomial::new((0..n).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>());
        let u_challenge = (0..num_vars)
            .map(|_| Fr::rand(&mut rng))
            .collect::<Vec<_>>();
        let v_evaluation = multilinear_f.evaluate(&u_challenge);

        // Compute multilinear quotients `qₖ(𝑋₀, …, 𝑋ₖ₋₁)`
        let (quotients, constant_term) =
            compute_multilinear_quotients::<Bn254>(&multilinear_f, &u_challenge);

        // Assert the constant term is equal to v_evaluation
        assert_eq!(
            constant_term, v_evaluation,
            "The constant term equal to the evaluation of the polynomial at challenge point."
        );

        //To demonstrate that q_k was properly constructd we show that the identity holds at a random multilinear challenge
        // i.e. 𝑓(𝑧) − 𝑣 − ∑ₖ₌₀ᵈ⁻¹ (𝑧ₖ − 𝑢ₖ)𝑞ₖ(𝑧) = 0
        let z_challenge = (0..num_vars)
            .map(|_| Fr::rand(&mut rng))
            .collect::<Vec<_>>();

        let mut res = multilinear_f.evaluate(&z_challenge);
        res -= v_evaluation;

        for (k, q_k_uni) in quotients.iter().enumerate() {
            let z_partial = &z_challenge[z_challenge.len() - k..];
            //This is a weird consequence of how things are done.. the univariate polys are of the multilinear commitment in lagrange basis. Therefore we evaluate as multilinear
            let q_k = DensePolynomial::new(q_k_uni.coeffs.clone());
            let q_k_eval = q_k.evaluate(z_partial);

            res -= (z_challenge[z_challenge.len() - k - 1]
                - u_challenge[z_challenge.len() - k - 1])
                * q_k_eval;
        }
        assert!(res.is_zero());
    }

    /// Test for construction of batched lifted degree quotient:
    ///  ̂q = ∑ₖ₌₀ⁿ⁻¹ yᵏ Xᵐ⁻ᵈᵏ⁻¹ ̂qₖ, 𝑑ₖ = deg(̂q), 𝑚 = 𝑁
    #[test]
    fn batched_lifted_degree_quotient() {
        let num_vars = 3;
        let n = 1 << num_vars;

        // Define mock qₖ with deg(qₖ) = 2ᵏ⁻¹
        let q_0 = UniPoly::from_coeff(vec![Fr::one()]);
        let q_1 = UniPoly::from_coeff(vec![Fr::from(2u64), Fr::from(3u64)]);
        let q_2 = UniPoly::from_coeff(vec![
            Fr::from(4u64),
            Fr::from(5u64),
            Fr::from(6u64),
            Fr::from(7u64),
        ]);
        let quotients = vec![q_0, q_1, q_2];

        let mut rng = test_rng();
        let y_challenge = Fr::rand(&mut rng);

        //Compute batched quptient  ̂q
        let (batched_quotient, _) =
            compute_batched_lifted_degree_quotient::<Bn254>(&quotients, &y_challenge);

        //Explicitly define q_k_lifted = X^{N-2^k} * q_k and compute the expected batched result
        let q_0_lifted = UniPoly::from_coeff(vec![
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::one(),
        ]);
        let q_1_lifted = UniPoly::from_coeff(vec![
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::from(2u64),
            Fr::from(3u64),
        ]);
        let q_2_lifted = UniPoly::from_coeff(vec![
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::from(4u64),
            Fr::from(5u64),
            Fr::from(6u64),
            Fr::from(7u64),
        ]);

        //Explicitly compute  ̂q i.e. RLC of lifted polys
        let mut batched_quotient_expected = UniPoly::from_coeff(vec![Fr::zero(); n]);

        batched_quotient_expected += &q_0_lifted;
        batched_quotient_expected += &(q_1_lifted * y_challenge);
        batched_quotient_expected += &(q_2_lifted * (y_challenge * y_challenge));
        assert_eq!(batched_quotient, batched_quotient_expected);
    }

    /// evaluated quotient \zeta_x
    ///
    /// 𝜁 = 𝑓 − ∑ₖ₌₀ⁿ⁻¹𝑦ᵏ𝑥ʷˢ⁻ʷ⁺¹𝑓ₖ  = 𝑓 − ∑_{d ∈ {d₀, ..., dₙ₋₁}} X^{d* - d + 1}  − ∑{k∶ dₖ=d} yᵏ fₖ , where d* = lifted degree
    ///
    /// 𝜁 =  ̂q - ∑ₖ₌₀ⁿ⁻¹ yᵏ Xᵐ⁻ᵈᵏ⁻¹ ̂qₖ, m = N
    #[test]
    fn partially_evaluated_quotient_zeta() {
        let num_vars = 3;
        let n: u64 = 1 << num_vars;

        let mut rng = test_rng();
        let x_challenge = Fr::rand(&mut rng);
        let y_challenge = Fr::rand(&mut rng);

        let challenges: Vec<_> = (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();
        let z_challenge = Fr::rand(&mut rng);

        let (_, (zeta_x_scalars, _)) =
            eval_and_quotient_scalars::<Bn254>(y_challenge, x_challenge, z_challenge, &challenges);

        // To verify we manually compute zeta using the computed powers and expected
        // 𝜁 =  ̂q - ∑ₖ₌₀ⁿ⁻¹ yᵏ Xᵐ⁻ᵈᵏ⁻¹ ̂qₖ, m = N
        assert_eq!(
            zeta_x_scalars[0],
            -x_challenge.pow(BigInt::<1>::from(n - 1))
        );

        assert_eq!(
            zeta_x_scalars[1],
            -y_challenge * x_challenge.pow(BigInt::<1>::from(n - 1 - 1))
        );

        assert_eq!(
            zeta_x_scalars[2],
            -y_challenge * y_challenge * x_challenge.pow(BigInt::<1>::from(n - 3 - 1))
        );
    }

    /// Test efficiently computing 𝛷ₖ(x) = ∑ᵢ₌₀ᵏ⁻¹xⁱ
    /// 𝛷ₖ(𝑥) = ∑ᵢ₌₀ᵏ⁻¹𝑥ⁱ = (𝑥²^ᵏ − 1) / (𝑥 − 1)
    #[test]
    fn phi_n_x_evaluation() {
        const N: u64 = 8u64;
        let log_N = (N as usize).log_2();

        // 𝛷ₖ(𝑥)
        let mut rng = test_rng();
        let x_challenge = Fr::rand(&mut rng);

        let efficient = (x_challenge.pow(BigInt::<1>::from((1 << log_N) as u64)) - Fr::one())
            / (x_challenge - Fr::one());
        let expected: Fr = phi::<Bn254>(&x_challenge, log_N);
        assert_eq!(efficient, expected);
    }

    /// Test efficiently computing 𝛷ₖ(x) = ∑ᵢ₌₀ᵏ⁻¹xⁱ
    /// 𝛷ₙ₋ₖ₋₁(𝑥²^ᵏ⁺¹) = (𝑥²^ⁿ − 1) / (𝑥²^ᵏ⁺¹ − 1)
    #[test]
    fn phi_n_k_1_x_evaluation() {
        const N: u64 = 8u64;
        let log_N = (N as usize).log_2();

        // 𝛷ₖ(𝑥)
        let mut rng = test_rng();
        let x_challenge = Fr::rand(&mut rng);
        let k = 2;

        //𝑥²^ᵏ⁺¹
        let x_pow = x_challenge.pow(BigInt::<1>::from((1 << (k + 1)) as u64));

        //(𝑥²^ⁿ − 1) / (𝑥²^ᵏ⁺¹ − 1)
        let efficient = (x_challenge.pow(BigInt::<1>::from((1 << log_N) as u64)) - Fr::one())
            / (x_pow - Fr::one());
        let expected: Fr = phi::<Bn254>(&x_challenge, log_N - k - 1);
        assert_eq!(efficient, expected);
    }

    /// Test construction of 𝑍ₓ
    /// 𝑍ₓ =  ̂𝑓 − 𝑣 ∑ₖ₌₀ⁿ⁻¹(𝑥²^ᵏ𝛷ₙ₋ₖ₋₁(𝑥ᵏ⁺¹)− 𝑢ₖ𝛷ₙ₋ₖ(𝑥²^ᵏ)) ̂qₖ
    #[test]
    fn partially_evaluated_quotient_z_x() {
        let num_vars = 3;

        // Construct a random multilinear polynomial f, and (u,v) such that f(u) = v.
        let mut rng = test_rng();
        let challenges: Vec<_> = (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();

        let u_rev = {
            let mut res = challenges.clone();
            res.reverse();
            res
        };

        let x_challenge = Fr::rand(&mut rng);
        let y_challenge = Fr::rand(&mut rng);
        let z_challenge = Fr::rand(&mut rng);

        // Construct Z_x scalars
        let (_, (_, z_x_scalars)) =
            eval_and_quotient_scalars::<Bn254>(y_challenge, x_challenge, z_challenge, &challenges);

        for k in 0..num_vars {
            let x_pow_2k = x_challenge.pow(BigInt::<1>::from((1 << k) as u64)); // x^{2^k}
            let x_pow_2kp1 = x_challenge.pow(BigInt::<1>::from((1 << (k + 1)) as u64)); // x^{2^{k+1}}
                                                                                        // x^{2^k} * \Phi_{n-k-1}(x^{2^{k+1}}) - u_k *  \Phi_{n-k}(x^{2^k})
            let mut scalar = x_pow_2k * phi::<Bn254>(&x_pow_2kp1, num_vars - k - 1)
                - u_rev[k] * phi::<Bn254>(&x_pow_2k, num_vars - k);
            scalar *= z_challenge;
            scalar *= Fr::from(-1);
            assert_eq!(z_x_scalars[k], scalar);
        }
    }

    #[test]
    fn zeromorph_commit_prove_verify() {
        for num_vars in [4, 5, 6] {
            let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(num_vars as u64);

            let poly = DensePolynomial::random(num_vars, &mut rng);
            let point: Vec<<Bn254 as Pairing>::ScalarField> = (0..num_vars)
                .map(|_| <Bn254 as Pairing>::ScalarField::rand(&mut rng))
                .collect();
            let eval = poly.evaluate(&point);

            let srs = ZeromorphSRS::<Bn254>::setup(&mut rng, 1 << num_vars);
            let (pk, vk) = srs.trim(1 << num_vars);
            let commitment = Zeromorph::<Bn254, KeccakTranscript>::commit(&pk, &poly).unwrap();

            let mut prover_transcript = KeccakTranscript::new(b"TestEval");
            let proof = Zeromorph::<Bn254, KeccakTranscript>::open(
                &pk,
                &poly,
                &point,
                &eval,
                &mut prover_transcript,
            )
            .unwrap();
            let p_transcipt_squeeze: <Bn254 as Pairing>::ScalarField =
                prover_transcript.challenge_scalar();

            // Verify proof.
            let mut verifier_transcript = KeccakTranscript::new(b"TestEval");
            Zeromorph::<Bn254, KeccakTranscript>::verify(
                &vk,
                &commitment,
                &point,
                &eval,
                &proof,
                &mut verifier_transcript,
            )
            .unwrap();
            let v_transcipt_squeeze: <Bn254 as Pairing>::ScalarField =
                verifier_transcript.challenge_scalar();

            assert_eq!(p_transcipt_squeeze, v_transcipt_squeeze);

            // evaluate bad proof for soundness
            let altered_verifier_point = point
                .iter()
                .map(|s| *s + <Bn254 as Pairing>::ScalarField::one())
                .collect::<Vec<_>>();
            let altered_verifier_eval = poly.evaluate(&altered_verifier_point);
            let mut verifier_transcript = KeccakTranscript::new(b"TestEval");
            assert!(Zeromorph::<Bn254, KeccakTranscript>::verify(
                &vk,
                &commitment,
                &altered_verifier_point,
                &altered_verifier_eval,
                &proof,
                &mut verifier_transcript,
            )
            .is_err())
        }
    }

    #[test]
    fn batched_zeromorph_commit_prove_verify() {
        for num_polys in [1, 2, 5, 7] {
            for num_vars in [4, 5, 6] {
                let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(num_vars as u64);

                let polys: Vec<_> = (0..num_polys)
                    .map(|_| DensePolynomial::random(num_vars, &mut rng))
                    .collect::<Vec<_>>();
                let point: Vec<Fr> = (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();
                let evals: Vec<Fr> = polys.iter().map(|poly| poly.evaluate(&point)).collect();

                let srs = ZeromorphSRS::<Bn254>::setup(&mut rng, 1 << num_vars);
                let (pk, vk) = srs.trim(1 << num_vars);
                let commitments: Vec<_> = polys
                    .iter()
                    .map(|poly| Zeromorph::<Bn254, KeccakTranscript>::commit(&pk, poly).unwrap())
                    .collect();

                let commitments_refs: Vec<_> = commitments.iter().collect();
                let polys_refs: Vec<_> = polys.iter().collect();

                let mut prover_transcript = KeccakTranscript::new(b"TestEval");
                let proof = Zeromorph::<Bn254, KeccakTranscript>::batch_open(
                    &pk,
                    &polys_refs,
                    &point,
                    &evals,
                    &mut prover_transcript,
                );
                let p_transcipt_squeeze: <Bn254 as Pairing>::ScalarField =
                    prover_transcript.challenge_scalar();

                // Verify proof.
                let mut verifier_transcript = KeccakTranscript::new(b"TestEval");
                Zeromorph::<Bn254, KeccakTranscript>::batch_verify(
                    &vk,
                    &commitments_refs,
                    &point,
                    &evals,
                    &proof,
                    &mut verifier_transcript,
                )
                .unwrap();
                let v_transcipt_squeeze: <Bn254 as Pairing>::ScalarField =
                    verifier_transcript.challenge_scalar();

                assert_eq!(p_transcipt_squeeze, v_transcipt_squeeze);

                // evaluate bad proof for completeness
                let altered_verifier_point = point
                    .iter()
                    .map(|s| *s + <Bn254 as Pairing>::ScalarField::one())
                    .collect::<Vec<_>>();
                let altered_verifier_evals: Vec<Fr> = polys
                    .iter()
                    .map(|poly| poly.evaluate(&altered_verifier_point))
                    .collect();
                let mut verifier_transcript = KeccakTranscript::new(b"TestEval");
                assert!(Zeromorph::<Bn254, KeccakTranscript>::batch_verify(
                    &vk,
                    &commitments_refs,
                    &altered_verifier_point,
                    &altered_verifier_evals,
                    &proof,
                    &mut verifier_transcript,
                )
                .is_err())
            }
        }
    }
}
