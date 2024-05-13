#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use std::{iter, marker::PhantomData};

use crate::poly::unipoly::UniPoly;
use crate::poly::{self, dense_mlpoly::DensePolynomial};
use crate::utils::transcript::{AppendToTranscript, ProofTranscript};
use crate::msm::VariableBaseMSM;
use ark_bn254::Bn254;
use ark_ec::scalar_mul::fixed_base::FixedBase;
use ark_ec::{pairing::Pairing, CurveGroup, AffineRepr};
use ark_ff::{PrimeField, batch_inversion, Field};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{UniformRand, iterable::Iterable, One, Zero};
use itertools::Itertools;
use lazy_static::lazy_static;
use rand_chacha::{
    rand_core::SeedableRng,
    ChaCha20Rng,
};
use rand_core::{CryptoRng, RngCore};
use std::sync::Arc;
use thiserror::Error;

#[cfg(feature = "ark-msm")]
use ark_ec::VariableBaseMSM;

use rayon::prelude::*;

//use super::commitment_scheme::{ BatchType, CommitShape, CommitmentScheme};

#[derive(Clone, Debug)]
pub struct SRS<P: Pairing> {
  pub g1_powers: Vec<P::G1Affine>,
  pub g2_powers: Vec<P::G2Affine>
}

impl<P: Pairing> SRS<P> {
  pub fn setup<R: RngCore + CryptoRng>(mut rng: &mut R, max_degree: usize) -> Self {
    let beta = P::ScalarField::rand(&mut rng);
    let g1 = P::G1::rand(&mut rng);
    let g2 = P::G2::rand(&mut rng);

    let beta_powers: Vec<P::ScalarField> = (0..=max_degree).scan(beta, |acc, _| {
      let val = *acc;
      *acc *= beta;
      Some(val)
    }).collect();

    let window_size = FixedBase::get_mul_window_size(max_degree);
    let scalar_bits = P::ScalarField::MODULUS_BIT_SIZE as usize;

    //TODO: gate with rayon
    let (g1_powers_projective, g2_powers_projective) = rayon::join( 
      || {
        let g1_table = FixedBase::get_window_table(scalar_bits, window_size, g1);
        FixedBase::msm(scalar_bits, window_size, &g1_table, &beta_powers)
      },
     || {
        let g2_table = FixedBase::get_window_table(scalar_bits, window_size, g2);
        FixedBase::msm(scalar_bits, window_size, &g2_table, &beta_powers)
      }
    );

    let (g1_powers, g2_powers) = rayon::join( 
      || {
        P::G1::normalize_batch(&g1_powers_projective)

      },
  || {
        P::G2::normalize_batch(&g2_powers_projective)
    });

    Self { g1_powers, g2_powers }
  }

  pub fn trim(params: Arc<Self>, supported_size: usize) -> (KZGProverKey<P>, KZGVerifierKey<P>) {
    assert!(params.g1_powers.len() > 0, "max_degree is 0");
    let g1 = params.g1_powers[0];
    let g2 = params.g2_powers[0];
    let beta_g2 = params.g2_powers[1];
    let pk = KZGProverKey::new(params, 0, supported_size + 1);
    let vk = KZGVerifierKey {g1, g2, beta_g2};
    (pk, vk)
  }

}

// Abstraction around SRS preventing copying. Arc of SRS
#[derive(Clone, Debug)]
pub struct KZGProverKey<P: Pairing> {
  srs: Arc<SRS<P>>,
  // offset to read into SRS
  offset: usize,
  // max size of srs
  supported_size: usize,
}

impl<P: Pairing> KZGProverKey<P> {
  pub fn new(
    srs: Arc<SRS<P>>,
    offset: usize,
    supported_size: usize,
  ) -> Self {
    assert!(
      srs.g1_powers.len() >= offset + supported_size,
      "not enough powers (req: {} from offset {}) in the SRS (length: {})",
      supported_size,
      offset,
      srs.g1_powers.len()
    );
    Self {
      srs,
      offset,
      supported_size,
    }
  }

  pub fn g1_powers(&self) -> &[P::G1Affine] {
    &self.srs.g1_powers[self.offset..self.offset + self.supported_size]
  }
}

// Abstraction around SRS preventing copying. Arc of SRS
#[derive(Clone, Copy, Debug)]
pub struct KZGVerifierKey<P: Pairing> {
  pub g1: P::G1Affine,
  pub g2: P::G2Affine,
  pub beta_g2: P::G2Affine
}

#[derive(Debug, Clone, Eq, PartialEq, Default)]
pub struct UVKZGPCS<P: Pairing> {
  _phantom: PhantomData<P>,
}

impl <P: Pairing> UVKZGPCS<P> 
where
    <P as Pairing>::ScalarField: poly::field::JoltField,
{
  fn commit_offset(pk: &KZGProverKey<P>, poly: &UniPoly<P::ScalarField>, offset: usize) -> Result<P::G1Affine, ZeromorphError> {
    if poly.degree() > pk.g1_powers().len() {
      return Err(ZeromorphError::KeyLengthError(poly.degree(), pk.g1_powers().len()))
    }

    let scalars = poly.as_vec();
    let bases = pk.g1_powers();
    let c = <P::G1 as VariableBaseMSM>::msm(&bases[offset..scalars.len()], &poly.as_vec()[offset..]).unwrap();

    Ok(c.into_affine())
  }

  pub fn commit(pk: &KZGProverKey<P>, poly: &UniPoly<P::ScalarField>) -> Result<P::G1Affine, ZeromorphError> {
    if poly.degree() > pk.g1_powers().len() {
      return Err(ZeromorphError::KeyLengthError(poly.degree(), pk.g1_powers().len()))
    }
    let c = <P::G1 as VariableBaseMSM>::msm( &pk.g1_powers()[..poly.as_vec().len()], &poly.as_vec().as_slice()).unwrap();
    Ok(c.into_affine())
  }

  fn open(
    pk: &KZGProverKey<P>,
    poly: &UniPoly<P::ScalarField>,
    point: &P::ScalarField
  ) -> Result<(P::G1Affine, P::ScalarField), ZeromorphError> 
  where
    <P as ark_ec::pairing::Pairing>::ScalarField: poly::field::JoltField
  {
    let divisor = UniPoly::from_coeff(vec![-*point, P::ScalarField::one()]);
    let (witness_poly, _) = poly.divide_with_q_and_r(&divisor).unwrap();
    let proof = <P::G1 as VariableBaseMSM>::msm(&pk.g1_powers()[..witness_poly.as_vec().len()], &witness_poly.as_vec().as_slice()).unwrap(); 
    let evaluation = poly.evaluate(point);
    Ok((proof.into_affine(), evaluation))
  }

}

const MAX_VARS: usize = 17;

lazy_static! {
    pub static ref ZEROMORPH_SRS: ZeromorphSRS<Bn254> =
        ZeromorphSRS(Arc::new(SRS::setup(
            &mut ChaCha20Rng::from_seed(*b"ZEROMORPH_POLY_COMMITMENT_SCHEME"),
            1 << (MAX_VARS + 1)
        )));
}

pub struct ZeromorphSRS<P: Pairing>(Arc<SRS<P>>);

impl<P: Pairing> ZeromorphSRS<P> {
  pub fn setup<R: RngCore + CryptoRng>(mut rng: &mut R, max_degree: usize) -> Self {
    Self(Arc::new(SRS::setup(rng, max_degree)))
  }

  pub fn trim(self, max_degree: usize) -> (ZeromorphProverKey<P>, ZeromorphVerifierKey<P>) {
    //TODO: remove into()
    let (commit_pp, kzg_vk) = SRS::trim(self.0.clone(), max_degree);
    let offset = self.0.g1_powers.len() - max_degree;
    let tau_N_max_sub_2_N = self.0.g2_powers[offset];
    let open_pp = KZGProverKey::new(self.0, offset, max_degree);
    (
      ZeromorphProverKey {commit_pp, open_pp},
      ZeromorphVerifierKey {kzg_vk, tau_N_max_sub_2_N}
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

#[derive(Error, Debug)]
pub enum ZeromorphError {
    #[error("Length Error: SRS Length: {0}, Key Length: {0}")]
    KeyLengthError(usize, usize),
}

pub struct ZeromorphCommitment<P: Pairing>(P::G1Affine);

impl<P: Pairing> AppendToTranscript for ZeromorphCommitment<P> 
where
  Self: CurveGroup
{
  fn append_to_transcript(&self, label: &'static [u8], transcript: &mut ProofTranscript) {
      transcript.append_point(b"poly_commitment_share", self);
  }
}

#[derive(Clone, CanonicalSerialize,
  CanonicalDeserialize, Debug)]
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
    <P as Pairing>::ScalarField: poly::field::JoltField,
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

            #[cfg(feature = "multicore")]
            let quotient_iter = quotient.par_iter_mut();

            #[cfg(not(feature = "multicore"))]
            let quotient_iter = quotient.iter_mut();

            quotient_iter
                .zip_eq(&*remainder_lo)
                .zip_eq(&*remainder_hi)
                .for_each(|((mut q, r_lo), r_hi)| {
                    *q = *r_hi - *r_lo;
                });

            #[cfg(feature = "multicore")]
            let remainder_lo_iter = remainder_lo.par_iter_mut();

            #[cfg(not(feature = "multicore"))]
            let remainder_lo_iter = remainder_lo.iter_mut();
            remainder_lo_iter.zip_eq(remainder_hi).for_each(|(r_lo, r_hi)| {
                *r_lo += (*r_hi - r_lo as &_) * x_i;
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
    <P as Pairing>::ScalarField: poly::field::JoltField,
{
    let num_vars = quotients.len();

    // Compute \hat{q} = \sum_k y^k * X^{N - d_k - 1} * q_k
    let mut scalar = P::ScalarField::one(); // y^k

    // Rather than explicitly computing the shifts of q_k by N - d_k - 1 (i.e. multiplying q_k by X^{N - d_k - 1})
    // then accumulating them, we simply accumulate y^k*q_k into \hat{q} at the index offset N - d_k - 1
    let q_hat = quotients.iter().enumerate().fold(
        vec![P::ScalarField::zero(); 1 << num_vars],
        |mut q_hat, (idx, q)| {
            #[cfg(feature = "multicore")]
            let q_hat_iter = q_hat[(1 << num_vars) - (1 << idx)..].par_iter_mut();

            #[cfg(not(feature = "multicore"))]
            let q_hat_iter = q_hat[(1 << num_vars) - (1 << idx)..].iter_mut();
            q_hat_iter.zip(&q.as_vec()).for_each(|(q_hat, q)| {
                *q_hat += scalar * q;
            });
            scalar *= y_challenge;
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
    <P as Pairing>::ScalarField: poly::field::JoltField,
{
    let num_vars = challenges.len();

    // squares of x = [x, x^2, .. x^{2^k}, .. x^{2^num_vars}]
    let squares_of_x: Vec<_> = iter::successors(Some(x_challenge), |&x| Some(x.square()))
        .take(num_vars + 1)
        .collect();

    // offsets of x =
    let offsets_of_x = {
        let mut offsets_of_x = squares_of_x
            .iter()
            .rev()
            .skip(1)
            .scan(P::ScalarField::one(), |acc, pow_x| {
                *acc *= pow_x;
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
            .map(|v_denom| v_numer * v_denom)
            .collect::<Vec<_>>()
    };

    let q_scalars = iter::successors(Some(P::ScalarField::one()), |acc| Some(*acc * y_challenge))
        .take(num_vars)
        .zip_eq(offsets_of_x)
        .zip(squares_of_x)
        .zip(&vs)
        .zip_eq(&vs[1..])
        .zip_eq(challenges.iter().rev())
        .map(
            |(((((power_of_y, offset_of_x), square_of_x), v_i), v_j), u_i)| {
                (
                    -(power_of_y * offset_of_x),
                    -(z_challenge * (square_of_x * v_j - *u_i * v_i)),
                )
            },
        )
        .unzip();

    // -vs[0] * z = -z * (x^(2^num_vars) - 1) / (x - 1) = -z Φ_n(x)
    (-vs[0] * z_challenge, q_scalars)
}

#[derive(Clone)]
pub struct Zeromorph<P: Pairing> {
    _phantom: PhantomData<P>,
  }

impl<P: Pairing> Zeromorph<P> 
where
    <P as Pairing>::ScalarField: poly::field::JoltField,
{
  pub fn protocol_name() -> &'static [u8] {
    b"Zeromorph"
  }

  //IDEAS;
  // - extra sampling from transcript??? -> no adding randomness has to due with information leakage not verification

  pub fn commit(pp: &ZeromorphProverKey<P>, poly: &DensePolynomial<P::ScalarField>) -> Result<P::G1Affine, ZeromorphError> {
    if pp.commit_pp.g1_powers().len() < poly.Z.len() {
      return Err(ZeromorphError::KeyLengthError(pp.commit_pp.g1_powers().len(), poly.Z.len()))
    }
    UVKZGPCS::commit(&pp.commit_pp, &UniPoly::from_coeff(poly.Z.clone()))
  }

  pub fn open(pp: &ZeromorphProverKey<P>, comm: &P::G1Affine, poly: &DensePolynomial<P::ScalarField>, point: &[P::ScalarField], eval: &P::ScalarField, transcript: &mut ProofTranscript) -> Result<ZeromorphProof<P>, ZeromorphError> {
    transcript.append_protocol_name(Self::protocol_name());

    if pp.commit_pp.g1_powers().len() < poly.Z.len() {
      return Err(ZeromorphError::KeyLengthError(pp.commit_pp.g1_powers().len(), poly.Z.len()))
    }

    assert_eq!(Self::commit(pp, poly).unwrap(), *comm);
    assert_eq!(poly.evaluate(point), *eval);

    let (quotients, remainder): (Vec<UniPoly<P::ScalarField>>, P::ScalarField) = compute_multilinear_quotients::<P>(poly, point);
    assert_eq!(quotients.len(), poly.get_num_vars());
    assert_eq!(remainder, *eval);
    
    // Compute the multilinear quotients q_k = q_k(X_0, ..., X_{k-1})
    let q_k_com: Vec<P::G1Affine> = quotients.par_iter().map(|q| UVKZGPCS::commit(&pp.commit_pp, q).unwrap()).collect();
    let q_comms: Vec<P::G1> = q_k_com.clone().into_iter().map(|c| c.into_group()).collect();
    //transcript.append_points(b"q_comms", &q_comms);
    q_comms.iter().for_each(|c| transcript.append_point(b"quo", c));

    // Sample challenge y
    let y_challenge: P::ScalarField = transcript.challenge_scalar(b"y");

    // Compute the batched, lifted-degree quotient `\hat{q}`
    // qq_hat = ∑_{i=0}^{num_vars-1} y^i * X^(2^num_vars - d_k - 1) * q_i(x)
    let (q_hat, offset) = compute_batched_lifted_degree_quotient::<P>(&quotients, &y_challenge);

    // Compute and absorb the commitment C_q = [\hat{q}]
    let q_hat_com = UVKZGPCS::commit_offset(&pp.commit_pp, &q_hat, offset)?;
    transcript.append_point(b"q_hat", &q_hat_com.into_group());

    // Get x and z challenges
    let x_challenge = transcript.challenge_scalar(b"x");
    let z_challenge = transcript.challenge_scalar(b"z");

    // Compute batched degree and ZM-identity quotient polynomial pi
    let (eval_scalar, (degree_check_q_scalars, zmpoly_q_scalars)): (P::ScalarField, (Vec<P::ScalarField>, Vec<P::ScalarField>)) = eval_and_quotient_scalars::<P>(y_challenge, x_challenge, z_challenge, point);
    // f = z * poly.Z + q_hat + (-z * Φ_n(x) * e) + ∑_k (q_scalars_k * q_k)
    let mut f = UniPoly::from_coeff(poly.Z.clone());
    f *= &z_challenge;
    f += &q_hat;
    f[0] += eval_scalar * eval;
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
    let (pi, _) = UVKZGPCS::open(&pp.open_pp, &f, &x_challenge)?;

    Ok(ZeromorphProof { pi, q_hat_com, q_k_com })
  }

  pub fn verify(vk: &ZeromorphVerifierKey<P>, comm: &P::G1Affine, point: &[P::ScalarField], eval: &P::ScalarField, proof: &ZeromorphProof<P>, transcript: &mut ProofTranscript) -> Result<bool, ZeromorphError> {
    transcript.append_protocol_name(Self::protocol_name());

    // Receive commitments [q_k]
    //TODO: remove clone
    let q_comms: Vec<P::G1> = proof.q_k_com.clone().into_iter().map(|c| c.into_group()).collect();
    q_comms.iter().for_each(|c| transcript.append_point(b"quo", c));

    // Challenge y
    let y_challenge: P::ScalarField = transcript.challenge_scalar(b"y");

    // Receive commitment C_q_hat
    transcript.append_point(b"q_hat", &proof.q_hat_com.into_group());

    // Get x and z challenges
    let x_challenge = transcript.challenge_scalar(b"x");
    let z_challenge = transcript.challenge_scalar(b"z");

    // Compute batched degree and ZM-identity quotient polynomial pi
    let (eval_scalar, (mut q_scalars, zmpoly_q_scalars)): (P::ScalarField, (Vec<P::ScalarField>, Vec<P::ScalarField>)) = eval_and_quotient_scalars::<P>(y_challenge, x_challenge, z_challenge, point);
    q_scalars
      .iter_mut()
      .zip_eq(zmpoly_q_scalars)
      .for_each(|(scalar, zm_poly_q_scalar)| {
        *scalar += zm_poly_q_scalar;
      });
      let scalars = [vec![P::ScalarField::one(), z_challenge, eval_scalar * eval], q_scalars].concat();
      let bases = [
        vec![proof.q_hat_com, *comm, vk.kzg_vk.g1],
        //TODO: eliminate
        proof.q_k_com.clone()
      ].concat();
      let c = <P::G1 as VariableBaseMSM>::msm(&bases, &scalars).unwrap().into_affine();

      let pairing = P::multi_pairing(
        &[c, proof.pi], 
        &[(-vk.tau_N_max_sub_2_N.into_group()).into_affine(), (vk.kzg_vk.beta_g2.into_group() - (vk.kzg_vk.g2 * x_challenge)).into()]
      );
      Ok(pairing.0.is_one().into())

  }
}

/*
impl<P: Pairing> CommitmentScheme for Zeromorph<P>
where
    <P as Pairing>::ScalarField: poly::field::JoltField,
    ZeromorphCommitment<P>: CurveGroup
{
    type Field =  P::ScalarField;
    type Setup = Vec<(ZeromorphProverKey<P>, ZeromorphVerifierKey<P>)>;
    type Commitment =  ZeromorphCommitment<P>;
    type Proof = ZeromorphProof<P>;
    type BatchedProof = ZeromorphProof<P>;

    fn setup(shapes: &[CommitShape]) -> Self::Setup {
      //TODO: Does using lazy_static! lead to large problems
      todo!()
    }

    fn commit(poly: &DensePolynomial<Self::Field>, setup: &Self::Setup) -> Self::Commitment {
      // TODO: assert lengths are valid
      //ZeromorphCommitment(UnivariateKZG::<P>::commit(setup, &UniPoly::from_coeff(poly.Z.clone())).unwrap())
      todo!()
    }

    fn batch_commit(
        evals: &[&[Self::Field]],
        gens: &Self::Setup,
        batch_type: BatchType,
    ) -> Vec<Self::Commitment> {
      // TODO: assert lengths are valid
      #[cfg(feature = "multicore")]
      let iter = evals.par_iter();
      #[cfg(not(feature = "multicore"))]
      let iter = evals.iter();
      iter
        .map(|poly| ZeromorphCommitment(UnivariateKZG::<P>::commit(gens, &UniPoly::from_coeff(poly.Z.clone())).unwrap()))
        .collect::<Vec<_>>()
    }

    fn commit_slice(evals: &[Self::Field], setup: &Self::Setup) -> Self::Commitment {
      todo!()
    }

    fn prove(
        poly: &DensePolynomial<Self::Field>,
        opening_point: &[Self::Field], // point at which the polynomial is evaluated
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
      todo!()
    }

    fn batch_prove(
        polynomials: &[&DensePolynomial<Self::Field>],
        opening_point: &[Self::Field],
        openings: &[Self::Field],
        batch_type: BatchType,
        transcript: &mut ProofTranscript,
    ) -> Self::BatchedProof {
      todo!()
    }

    fn verify(
        proof: &Self::Proof,
        setup: &Self::Setup,
        transcript: &mut ProofTranscript,
        opening_point: &[Self::Field], // point at which the polynomial is evaluated
        opening: &Self::Field,         // evaluation \widetilde{Z}(r)
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
      todo!()
    }

    fn batch_verify(
        batch_proof: &Self::BatchedProof,
        setup: &Self::Setup,
        opening_point: &[Self::Field],
        openings: &[Self::Field],
        commitments: &[&Self::Commitment],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
      todo!()
    }

    fn protocol_name() -> &'static [u8] {
        b"zeromorph"
    }
}
*/

#[cfg(test)]
mod test {
    use super::*;
    use crate::utils::math::Math;
    use ark_bn254::{Bn254, Fr};
    use ark_ff::{BigInt, Zero};
    use ark_std::{rand::Rng, test_rng, UniformRand};
    use ark_ec::AffineRepr;
    use rand_core::SeedableRng;

    // Evaluate Phi_k(x) = \sum_{i=0}^k x^i using the direct inefficent formula
    fn phi<P: Pairing>(challenge: &P::ScalarField, subscript: usize) -> P::ScalarField {
        let len = (1 << subscript) as u64;
        (0..len)
            .into_iter()
            .fold(P::ScalarField::zero(), |mut acc, i| {
                //Note this is ridiculous DevX
                acc += challenge.pow(BigInt::<1>::from(i));
                acc
            })
    }

    /*
    #[test]
    fn prove_verify_single() {
      let max_vars = 8;
      let mut rng = test_rng();
      let srs = ZEROMORPH_SRS.lock().unwrap();

      for num_vars in 3..max_vars {
        // Setup
        let (pk, vk) = {
          let poly_size = 1 << (num_vars + 1);
          srs.trim(poly_size - 1).unwrap()
        };
        let polys = DensePolynomial::new(
          (0..(1 << num_vars))
            .map(|_| Fr::rand(&mut rng))
            .collect::<Vec<_>>(),
        );
        let challenges = (0..num_vars)
          .map(|_| Fr::rand(&mut rng))
          .collect::<Vec<_>>();
        let evals = polys.evaluate(&challenges);

        // Commit and open
        let commitments = Zeromorph::<Bn254>::commit(&[polys.clone()], &pk.g1_powers).unwrap();

        let mut prover_transcript = Transcript::new(b"example");
        let proof = Zeromorph::<Bn254>::prove(
          &[polys],
          &[evals],
          &challenges,
          &pk,
          &mut prover_transcript,
        )
        .unwrap();

        let mut verifier_transcript = Transcript::new(b"example");
        Zeromorph::<Bn254>::verify(
          &commitments,
          &[evals],
          &challenges,
          &vk,
          &mut verifier_transcript,
          proof,
        )
        .unwrap();

        //TODO: check both random oracles are synced
      }
    }

    #[test]
    fn prove_verify_batched() {
      let max_vars = 16;
      let mut rng = test_rng();
      let num_polys = 8;
      let srs = ZEROMORPH_SRS.lock().unwrap();

      for num_vars in 3..max_vars {
        // Setup
        let (pk, vk) = {
          let poly_size = 1 << (num_vars + 1);
          srs.trim(poly_size - 1).unwrap()
        };
        let polys: Vec<DensePolynomial<Fr>> = (0..num_polys)
          .map(|_| {
            DensePolynomial::new(
              (0..(1 << num_vars))
                .map(|_| Fr::rand(&mut rng))
                .collect::<Vec<_>>(),
            )
          })
          .collect::<Vec<_>>();
        let challenges = (0..num_vars)
          .into_iter()
          .map(|_| Fr::rand(&mut rng))
          .collect::<Vec<_>>();
        let evals = polys
          .clone()
          .into_iter()
          .map(|poly| poly.evaluate(&challenges))
          .collect::<Vec<_>>();

        // Commit and open
        let commitments = Zeromorph::<Bn254>::commit(&polys, &pk.g1_powers).unwrap();

        let mut prover_transcript = Transcript::new(b"example");
        let proof =
          Zeromorph::<Bn254>::prove(&polys, &evals, &challenges, &pk, &mut prover_transcript).unwrap();

        let mut verifier_transcript = Transcript::new(b"example");
        Zeromorph::<Bn254>::verify(
          &commitments,
          &evals,
          &challenges,
          &vk,
          &mut verifier_transcript,
          proof,
        )
        .unwrap();

        //TODO: check both random oracles are synced
      }
    }
    */

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
            .into_iter()
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
            let z_partial = &z_challenge[&z_challenge.len() - k..];
            //This is a weird consequence of how things are done.. the univariate polys are of the multilinear commitment in lagrange basis. Therefore we evaluate as multilinear
            let q_k = DensePolynomial::new(q_k_uni.as_vec());
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
            -x_challenge.pow(BigInt::<1>::from((n - 1) as u64))
        );

        assert_eq!(
            zeta_x_scalars[1],
            -y_challenge * x_challenge.pow(BigInt::<1>::from((n - 1 - 1) as u64))
        );

        assert_eq!(
            zeta_x_scalars[2],
            -y_challenge * y_challenge * x_challenge.pow(BigInt::<1>::from((n - 3 - 1) as u64))
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
        let challenges: Vec<_> = (0..num_vars)
            .into_iter()
            .map(|_| Fr::rand(&mut rng))
            .collect();

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
            let mut scalar = x_pow_2k * &phi::<Bn254>(&x_pow_2kp1, num_vars - k - 1)
                - u_rev[k] * &phi::<Bn254>(&x_pow_2k, num_vars - k);
            scalar *= z_challenge;
            scalar *= Fr::from(-1);
            assert_eq!(z_x_scalars[k], scalar);
        }
    }

    fn kzg_verify<P: Pairing>(
      vk: &KZGVerifierKey<P>,
      commitment: &P::G1Affine,
      point: &P::ScalarField,
      proof: &P::G1Affine,
      evaluation: &P::ScalarField,
    ) -> Result<bool, ZeromorphError> {

        let lhs = P::pairing(
            commitment.into_group() - vk.g1.into_group() * evaluation,
            vk.g2,
        );
        let rhs = P::pairing(proof, vk.beta_g2.into_group() - (vk.g2 * point));
        Ok(lhs == rhs)
    }

    fn random<P: Pairing, R: RngCore>(degree: usize, mut rng: &mut R) -> UniPoly<P::ScalarField>
    where
        <P as Pairing>::ScalarField: poly::field::JoltField,
    {
        let coeffs = (0..=degree)
            .map(|_| P::ScalarField::rand(&mut rng))
            .collect::<Vec<_>>();
        UniPoly::from_coeff(coeffs)
    }

    #[test]
    fn kzg_commit_prove_verify() -> Result<(), ZeromorphError> {
      let seed = b"11111111111111111111111111111111";
      for _ in 0..100 {
          let mut rng = &mut ChaCha20Rng::from_seed(*seed);
          let degree = rng.gen_range(2..20);

          let pp = Arc::new(SRS::<Bn254>::setup(&mut rng, degree));
          let (ck, vk) = SRS::trim(pp, degree);
          let p = random::<Bn254, ChaCha20Rng>(degree, rng);
          let comm = UVKZGPCS::<Bn254>::commit(&ck, &p)?;
          let point = Fr::rand(rng);
          let (proof, value) = UVKZGPCS::<Bn254>::open(&ck, &p, &point)?;
          assert!(
              kzg_verify(&vk, &comm, &point, &proof, &value)?,
              "proof was incorrect for max_degree = {}, polynomial_degree = {}",
              degree,
              p.degree(),
          );
      }
      Ok(())
    }

    #[test]
    fn zeromorph_commit_prove_verify() 
    {
      for num_vars in [4, 5, 6] {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(num_vars as u64);

        let poly = DensePolynomial::random(num_vars, &mut rng);
        let point: Vec<<Bn254 as Pairing>::ScalarField> = (0..num_vars).map(|_| <Bn254 as Pairing>::ScalarField::rand(&mut rng)).collect();
        let eval = poly.evaluate(&point);

        let srs = ZeromorphSRS::<Bn254>::setup(&mut rng, 1 << num_vars);
        let (pk, vk) = srs.trim(1 << num_vars);
        let commitment = Zeromorph::<Bn254>::commit(&pk, &poly).unwrap();

        let mut prover_transcript = ProofTranscript::new(b"TestEval");
        let proof = Zeromorph::<Bn254>::open(&pk, &commitment, &poly, &point, &eval, &mut prover_transcript).unwrap();

        // Verify proof.
        let mut verifier_transcript = ProofTranscript::new(b"TestEval");
        assert!(Zeromorph::<Bn254>::verify(&vk, &commitment, &point, &eval, &proof, &mut verifier_transcript).unwrap())
      } 
    }
}
