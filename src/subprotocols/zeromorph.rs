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
use ark_ec::{CurveConfig, CurveGroup, Group, pairing::Pairing};
use ark_ff::{BigInteger, Field, PrimeField, BigInt};
use ark_serialize::*;
use ark_std::One;
use merlin::Transcript;

#[cfg(feature = "ark-msm")]
use ark_ec::VariableBaseMSM;

#[cfg(not(feature = "ark-msm"))]
use crate::msm::VariableBaseMSM;

#[cfg(feature = "multicore")]
use rayon::prelude::*;

pub struct Proof<P: Pairing> {
  _phantom: PhantomData<P>
}

pub struct CommitmentKey<P: Pairing> {
  _phantom: PhantomData<P>
}

pub struct Zeromorph<const N: usize, P> {
  _phantom: PhantomData<P>,
}

/// Compute the powers of a challenge
///
impl<const N: usize, P: Pairing> Zeromorph<N, P> {
    fn powers_of_challenge(challenge: P::ScalarField, num_powers: usize) -> Vec<P::ScalarField> {
      //TODO: switch to successors
        (2..num_powers).fold(vec![P::ScalarField::one(), challenge], |mut acc, i| {
            acc.push(acc[i - 1] * challenge);
            acc
        })
    }

    fn compute_multilinear_quotients(
    polynimial: DensePolynomial<P::ScalarField>,
    u_challenge: &[P::ScalarField],
    ) -> Vec<DensePolynomial<P::ScalarField>> {
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

        let mut f_k: Vec<P::ScalarField> = Vec::with_capacity(size_q);

        //We can probably clean this up some but for now we're being explicit
        let mut g = Vec::new();//polynimial.clone();

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
            g = f_k.clone();
        }

        quotients
    }

    fn compute_batched_lifted_degree_quotient(
    quotients: Vec<DensePolynomial<P::ScalarField>>,
    y_challenge: P::ScalarField,
    ) -> DensePolynomial<P::ScalarField> {
        // Batched Lifted Degreee Quotient Polynomials
        let mut res: Vec<P::ScalarField> = Vec::with_capacity(N);

        // Compute \hat{q} = \sum_k y^k * X^{N - d_k - 1} * q_k
        let mut scalar = P::ScalarField::one();
        for (k, quotient) in quotients.iter().enumerate() {
            // Rather than explicitly computing the shifts of q_k by N - d_k - 1 (i.e. multiplying q_k by X^{N - d_k -
            // 1}) then accumulating them, we simply accumulate y^k*q_k into \hat{q} at the index offset N - d_k - 1
            //TODO: verify if this is needed as we are not interested in shifts
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
    batched_quotient: DensePolynomial<P::ScalarField>,
    quotients: Vec<DensePolynomial<P::ScalarField>>,
    y_challenge: P::ScalarField,
    x_challenge: P::ScalarField,
    ) -> DensePolynomial<P::ScalarField> {
        let n = batched_quotient.len();
        let log_N = quotients.len();

        // initialize partially evaluated degree check polynomial \zeta_x to \hat{q}
        let mut res = batched_quotient.clone();

        let mut y_power = P::ScalarField::one();
        for k in 0..log_N {
            // Accumulate y^k * x^{N - d_k - 1} * q_k into \hat{q}
            let deg_k = (1 << k) as usize - 1;
            let x_power = x_challenge.pow(BigInt::<1>::from((n - deg_k - 1) as u64));

            // Add poly and scale -> Note this can be parallelized
            // See -> https://github.com/AztecProtocol/barretenberg/blob/master/cpp/src/barretenberg/honk/pcs/zeromorph/zeromorph.hpp#L173
            // https://github.com/AztecProtocol/barretenberg/blob/master/cpp/src/barretenberg/polynomials/polynomial.cpp#L332
            // res += quotient[i] * (-y_power * x_power)
            for i in 0..res.len() {
                res[i] += quotients[k][i] * (-y_power * x_power);
            }

            y_power *= y_challenge; // updated batching scalar y^k
        }

        res
    }

    fn compute_partially_evaluated_zeromorph_identity_polynomial(
    f_batched: DensePolynomial<P::ScalarField>,
    g_batched: DensePolynomial<P::ScalarField>,
    quotients: Vec<DensePolynomial<P::ScalarField>>,
    v_evaluation: P::ScalarField,
    u_challenge: &[P::ScalarField],
    x_challenge: P::ScalarField,
    ) -> DensePolynomial<P::ScalarField> {
      let n = f_batched.len();
      let log_N = quotients.len();

      //Question for non-shifted can we exclude sum_{i=0}^{l-i}
      // Initialize Z_x with x * \sum_{i=0}^{m-1} f_i + /sum_{i=0}^{l-i} * g_i
      let mut res: DensePolynomial<P::ScalarField> = g_batched.clone();

      //add scaled
      for i in 0..res.len() {
          res[i] += f_batched[i] * x_challenge;
      }

      // Compute Z_x -= v * x * \Phi_n(x)
      let phi_numerator = x_challenge.pow(BigInt::<1>::from(n as u64)) - P::ScalarField::one(); //x^N - 1
      let phi_n_x = phi_numerator / (x_challenge - P::ScalarField::one());
      res[0] -= v_evaluation * x_challenge * phi_n_x;

      //Add contribution from q_k polynomials
      for k in 0..log_N {
        let x_power = x_challenge.pow(BigInt::<1>::from((1 << k) as u64)); // x^{2^k}

        // \Phi_{n-k-1}(x^{2^{k + 1}})
        let phi_term_1 = phi_numerator / (x_challenge.pow(BigInt::<1>::from((1 << (k + 1)) as u64)) - P::ScalarField::one());

        // \Phi_{n-k}(x^{2^k})
        let phi_term_2 = phi_numerator / (x_challenge.pow(BigInt::<1>::from((1 << k) as u64)) - P::ScalarField::one());

        // x^{2^k} * \Phi_{n-k-1}(x^{2^{k+1}}) - u_k * \Phi_{n-k}(x^{2^k})
        let mut scalar = x_power * phi_term_1 - u_challenge[k] * phi_term_2;

        scalar *= x_challenge;
        scalar *= -P::ScalarField::one();

        for i in 0..res.len() {
            res[i] += quotients[k][i] * scalar;
        }
      }
      res
    }

    fn compute_batched_evaluation_and_degree_check_quotient(
    zeta_x: DensePolynomial<P::ScalarField>,
    z_x: DensePolynomial<P::ScalarField>,
    x_challenge: P::ScalarField,
    z_challenge: P::ScalarField,
    ) -> DensePolynomial<P::ScalarField> {
      // We cannont commit to polynomials with size > N_max
      let n = zeta_x.len();
      assert!(n <= N);

      // Compute q_{\zeta} and q_Z in place
      let mut batched_quotient = zeta_x;
      for i in 0..batched_quotient.len() {
          batched_quotient[i] += z_x[i] * z_challenge;
      }

      //TODO: finish for batch and non-shifted quotient

      batched_quotient
    }

    pub fn prove(
      f_polynomials: Vec<DensePolynomial<P::ScalarField>>,
      evaluations: Vec<P::ScalarField>,
      multilinear_challenge: Vec<P::ScalarField>,
      commitment_key: CommitmentKey<P>,
      transcript: Transcript, 
    ) -> Proof<P> {
      todo!()
    }

    pub fn verify(

    ) {
      todo!()

    }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::utils::math::Math;
  use crate::utils::test::TestTranscript;
  use ark_bn254::Config;
  use ark_ff::{BigInt, Zero};

  // Evaluate Phi_k(x) = \sum_{i=0}^k x^i using the direct inefficent formula
  fn phi<P: Pairing>(challenge: P::BaseField, subscript: usize) -> P::BaseField {
    let len = (1 << subscript) as u64;
    let res = P::BaseField::zero();
    (0..len)
      .into_iter()
      .fold(P::BaseField::zero(), |mut acc, i| {
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
  fn prove_and_verify_batched() {
    todo!()
  }

  #[test]
  fn test_commit_open_verify() {
    todo!()
  }
}
