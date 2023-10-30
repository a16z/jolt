#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use std::marker::PhantomData;
use std::ops::Index;

use crate::poly::commitments::MultiCommitGens;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::unipoly::{CompressedUniPoly, UniPoly};
use crate::subprotocols::dot_product::DotProductProof;
use crate::utils::errors::ProofVerifyError;
use crate::utils::transcript::{AppendToTranscript, ProofTranscript};
use ark_ec::short_weierstrass::{Affine, SWCurveConfig};
use ark_ec::{CurveConfig, CurveGroup, Group};
use ark_ff::{BigInteger, Field, PrimeField};
use ark_serialize::*;
use ark_std::One;
use merlin::Transcript;

#[cfg(feature = "ark-msm")]
use ark_ec::VariableBaseMSM;

#[cfg(not(feature = "ark-msm"))]
use crate::msm::VariableBaseMSM;

#[cfg(feature = "multicore")]
use rayon::prelude::*;

pub struct Zeromorph<const N: usize, C: CurveGroup + PrimeField> {
  _phantom: PhantomData<C>,
}

/// Compute the powers of a challenge
///
impl<const N: usize, C: CurveGroup + PrimeField> Zeromorph<N, C> {
  fn powers_of_challenge(challenge: C::ScalarField, num_powers: usize) -> Vec<C::ScalarField> {
    (2..num_powers).fold(vec![C::ScalarField::one(), challenge], |mut acc, i| {
      acc.push(acc[i - 1] * challenge);
      acc
    })
  }

  fn compute_multilinear_quotients(
    polynimial: DensePolynomial<C::ScalarField>,
    u_challenge: &[C::ScalarField],
  ) -> Vec<DensePolynomial<C::ScalarField>> {
    let log_N = polynimial.get_num_vars();

    // The size of the multilinear challenge must equal the log of the polynomial size
    assert!(log_N == u_challenge.len());

    // Define vector of quotient polynomials q_k, k = 0, ..., log_N - 1
    let mut quotients = Vec::with_capacity(log_N);

    // Compute the coefficients of q_{n - 1}
    let mut size_q = (1 << (log_N - 1)) as usize;
    let q = DensePolynomial::new((0..size_q).fold(Vec::new(), |mut acc, l| {
      acc.push(polynimial[size_q + l] - polynimial[l]);
      acc
    }));

    //Probs can't avoid this clone
    quotients[log_N - 1] = q.clone();

    let mut f_k = Vec::with_capacity(size_q);

    //We can probably clean this up some but for now we're being explicit
    let mut g = polynimial.clone();

    for k in 1..log_N {
      // Compute f_k
      for l in 0..size_q {
        f_k[l] = g[l] + u_challenge[log_N - k] * q[l];
      }

      size_q = size_q / 2;
      let q = DensePolynomial::new((0..size_q).fold(Vec::new(), |mut acc, l| {
        acc.push(polynimial[size_q + l] - polynimial[l]);
        acc
      }));

      quotients[log_N - k - 1] = q;

      //Would be great to remove this new instantiation probably best way is to just have vectors of coeffs.
      g = DensePolynomial::new(f_k.clone());
    }

    quotients
  }

  fn compute_batched_lifted_degree_quotient(
    quotients: Vec<DensePolynomial<C::ScalarField>>,
    y_challenge: C::ScalarField,
  ) -> DensePolynomial<C::ScalarField> {
    // Batched Lifted Degreee Quotient Polynomials
    let mut res: Vec<C::ScalarField> = Vec::with_capacity(N);

    // Compute \hat{q} = \sum_k y^k * X^{N - d_k - 1} * q_k
    let mut scalar = C::ScalarField::one();
    for (k, quotient) in quotients.iter().enumerate() {
      // Rather than explicitly computing the shifts of q_k by N - d_k - 1 (i.e. multiplying q_k by X^{N - d_k -
      // 1}) then accumulating them, we simply accumulate y^k*q_k into \hat{q} at the index offset N - d_k - 1
      let deg_k = (1 << k) as usize - 1;
      let offset = N - deg_k - 1;
      for i in 0..(deg_k + 1) {
        res[offset + i] += scalar * quotient[i];
      }
      scalar *= y_challenge; // update batching scalar y^k
    }

    DensePolynomial::new(res)
  }

  fn compute_partially_evaluated_degree_check_polynomial(
    batched_quotient: DensePolynomial<C::ScalarField>,
    quotients: Vec<DensePolynomial<C::ScalarField>>,
    y_challenge: C::ScalarField,
    x_challenge: C::ScalarField,
  ) -> DensePolynomial<C::ScalarField> {
    todo!()
  }

  fn compute_partially_evaluated_zeromorph_identity_polynomial(
    f_batched: DensePolynomial<C::ScalarField>,
    g_batched: DensePolynomial<C::ScalarField>,
    quotients: Vec<DensePolynomial<C::ScalarField>>,
    v_evaluation: C::ScalarField,
    u_challenge: &[C::ScalarField],
    x_evaluation: C::ScalarField,
  ) -> DensePolynomial<C::ScalarField> {
    todo!()
  }

  fn compute_batched_evaluation_and_degree_check_quotient(
    zeta_x: DensePolynomial<C::ScalarField>,
    z_x: DensePolynomial<C::ScalarField>,
    x_challenge: C::ScalarField,
    z_challenge: C::ScalarField,
  ) -> DensePolynomial<C::ScalarField> {
    todo!()
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::utils::math::Math;
  use crate::utils::test::TestTranscript;
  use ark_curve25519::{EdwardsProjective as G1Projective, Fr};
  use ark_ff::{BigInt, Zero};

  // Evaluate Phi_k(x) = \sum_{i=0}^k x^i using the direct inefficent formula
  fn phi<C: CurveGroup>(challenge: C::BaseField, subscript: usize) -> C::BaseField {
    let len = (1 << subscript) as u64;
    let res = C::BaseField::zero();
    (0..len)
      .into_iter()
      .fold(C::BaseField::zero(), |mut acc, i| {
        //Note this is ridiculous DevX
        acc += challenge.pow(BigInt::<1>::from(i));
        acc
      });
    res
  }

  #[test]
  fn quotient_construction() {
    todo!()
  }

  #[test]
  fn batched_lifted_degree_quotient() {
    todo!()
  }

  #[test]
  fn partially_evaluated_quotient_zeta() {
    todo!()
  }

  #[test]
  fn phi_evaluation() {
    todo!()
  }

  #[test]
  fn partially_evaluated_quotient_z_x() {
    todo!()
  }

  #[test]
  fn prove_verify_single() {
    todo!()
  }

  #[test]
  fn prove_and_verify_batched_with_shifts() {
    todo!()
  }

  #[test]
  fn test_commit_open_verify() {
    todo!()
  }
}
