#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use std::{borrow::Borrow, marker::PhantomData, ops::Neg};

use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::traits::CommitmentScheme;
use crate::utils::transcript::ProofTranscript;
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_ff::{BigInt, Field};
use ark_std::{iterable::Iterable, One, Zero};
use merlin::Transcript;
use thiserror::Error;

#[cfg(feature = "ark-msm")]
use ark_ec::VariableBaseMSM;

#[cfg(not(feature = "ark-msm"))]
use crate::msm::VariableBaseMSM;

#[cfg(feature = "multicore")]
use rayon::prelude::*;

use super::data_structures::{
  ZeromorphProof, ZeromorphProverKey, ZeromorphVerifierKey, ZEROMORPH_SRS,
};

// Just return vec of P::Scalar
fn compute_multilinear_quotients<P: Pairing>(
  poly: &DensePolynomial<P::ScalarField>,
  u_challenge: &[P::ScalarField],
) -> (Vec<UniPoly<P::ScalarField>>, P::ScalarField) {
  assert_eq!(poly.get_num_vars(), u_challenge.len());

  let mut g = poly.Z.to_vec();
  let mut quotients = u_challenge
    .iter()
    .enumerate()
    .map(|(i, x_i)| {
      let (g_lo, g_hi) = g.split_at_mut(1 << (poly.get_num_vars() - 1 - i));
      let mut quotient = vec![P::ScalarField::zero(); g_lo.len()];

      quotient
        .par_iter_mut()
        .zip(&*g_lo)
        .zip(&*g_hi)
        .for_each(|((mut q, g_lo), g_hi)| {
          *q = *g_hi - *g_lo;
        });
      g_lo.par_iter_mut().zip(g_hi).for_each(|(g_lo, g_hi)| {
        // WHAT IS THIS BLACK MAGIC &_
        *g_lo += (*g_hi - g_lo as &_) * x_i;
      });

      g.truncate(1 << (poly.get_num_vars() - 1 - i));

      UniPoly::from_coeff(quotient)
    })
    .collect::<Vec<UniPoly<P::ScalarField>>>();
  quotients.reverse();
  (quotients, g[0])
}

fn compute_batched_lifted_degree_quotient<const N: usize, P: Pairing>(
  quotients: &Vec<UniPoly<P::ScalarField>>,
  y_challenge: &P::ScalarField,
) -> UniPoly<P::ScalarField> {
  // Batched Lifted Degreee Quotient Polynomials
  let mut res: Vec<P::ScalarField> = vec![P::ScalarField::zero(); N as usize];

  // Compute \hat{q} = \sum_k y^k * X^{N - d_k - 1} * q_k
  let mut scalar = P::ScalarField::one(); // y^k
  for (k, quotient) in quotients.iter().enumerate() {
    // Rather than explicitly computing the shifts of q_k by N - d_k - 1 (i.e. multiplying q_k by X^{N - d_k -
    // 1}) then accumulating them, we simply accumulate y^k*q_k into \hat{q} at the index offset N - d_k - 1
    let deg_k = (1 << k) as usize - 1;
    let offset = N as usize - deg_k - 1;
    for i in 0..(deg_k + 1) {
      res[offset + i] += scalar * quotient[i];
    }
    scalar *= y_challenge; // update batching scalar y^k
  }

  UniPoly::from_coeff(res)
}

fn compute_partially_evaluated_degree_check_polynomial<const N: usize, P: Pairing>(
  batched_quotient: &UniPoly<P::ScalarField>,
  quotients: &Vec<UniPoly<P::ScalarField>>,
  y_challenge: &P::ScalarField,
  x_challenge: &P::ScalarField,
) -> UniPoly<P::ScalarField> {
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
    for i in 0..quotients[k].len() {
      res[i] += quotients[k][i] * (-y_power * x_power);
    }

    y_power *= y_challenge; // updated batching scalar y^k
  }

  res
}

fn compute_partially_evaluated_zeromorph_identity_polynomial<const N: usize, P: Pairing>(
  f_batched: &UniPoly<P::ScalarField>,
  //g_batched: &UniPoly<P::ScalarField>,
  quotients: &Vec<UniPoly<P::ScalarField>>,
  v_evaluation: &P::ScalarField,
  u_challenge: &[P::ScalarField],
  x_challenge: &P::ScalarField,
) -> UniPoly<P::ScalarField> {
  let n = f_batched.len();
  let log_N = quotients.len();

  //Question for non-shifted can we exclude sum_{i=0}^{l-i}
  // Initialize Z_x with x * \sum_{i=0}^{m-1} f_i + /sum_{i=0}^{l-i} * g_i
  //let mut res: UniPoly<P::ScalarField> = g_batched.clone();

  //add scaled
  //for i in 0..res.len() {
  //  res[i] += f_batched[i] * x_challenge;
  //}

  let mut res = f_batched.clone();

  // Compute Z_x -= v * x * \Phi_n(x)
  let phi_numerator = x_challenge.pow(BigInt::<1>::from(n as u64)) - P::ScalarField::one(); //x^N - 1
  let phi_n_x = phi_numerator / (*x_challenge - P::ScalarField::one());
  res[0] -= *v_evaluation * *x_challenge * phi_n_x;

  //Add contribution from q_k polynomials
  for k in 0..log_N {
    let x_power = x_challenge.pow(BigInt::<1>::from((1 << k) as u64)); // x^{2^k}

    // \Phi_{n-k-1}(x^{2^{k + 1}})
    let phi_term_1 = phi_numerator
      / (x_challenge.pow(BigInt::<1>::from((1 << (k + 1)) as u64)) - P::ScalarField::one());

    // \Phi_{n-k}(x^{2^k})
    let phi_term_2 =
      phi_numerator / (x_challenge.pow(BigInt::<1>::from((1 << k) as u64)) - P::ScalarField::one());

    // x^{2^k} * \Phi_{n-k-1}(x^{2^{k+1}}) - u_k * \Phi_{n-k}(x^{2^k})
    let mut scalar = x_power * phi_term_1 - u_challenge[k] * phi_term_2;

    scalar *= x_challenge;
    scalar *= -P::ScalarField::one();

    for i in 0..quotients[k].len() {
      res[i] += quotients[k][i] * scalar;
    }
  }
  res
}

//TODO: Need SRS
fn compute_batched_evaluation_and_degree_check_quotient<const N: usize, P: Pairing>(
  zeta_x: UniPoly<P::ScalarField>,
  z_x: UniPoly<P::ScalarField>,
  x_challenge: P::ScalarField,
  z_challenge: P::ScalarField,
) -> UniPoly<P::ScalarField> {
  // We cannot commit to polynomials with size > N_max
  let n = zeta_x.len();
  assert!(n <= N as usize);

  //Compute quotient polynomials q_{\zeta} and q_Z

  //q_{\zeta} = \zeta_x / (X-x)
  //q_z = Z_x / (X - x)
  //TODO: remove these clones
  let mut q_zeta_x = zeta_x.clone();
  let mut q_z_x = z_x.clone();
  q_zeta_x.factor_roots(&x_challenge);
  q_z_x.factor_roots(&x_challenge);

  // Compute batched quotient q_{\zeta} + z*q_Z in place
  let mut batched_quotient = zeta_x;
  for i in 0..batched_quotient.len() {
    batched_quotient[i] += z_x[i] * z_challenge;
  }

  batched_quotient
}

fn compute_C_zeta_x<const N: usize, P: Pairing>(
  q_hat_com: &P::G1,
  q_k_com: &Vec<P::G1Affine>,
  y_challenge: &P::ScalarField,
  x_challenge: &P::ScalarField,
) -> P::G1 {
  let n = 1 << q_k_com.len();

  let one = P::ScalarField::one();
  let mut scalars = vec![one];
  let mut commitments = vec![q_hat_com.into_affine()];

  for (i, com) in q_k_com.iter().enumerate() {
    let deg_k = (1 << i) - 1;
    // Compute scalar y^k * x^{N - deg_k - 1}
    let mut scalar = y_challenge.pow(BigInt::<1>::from(i as u64));
    scalar *= x_challenge.pow(BigInt::<1>::from((n - deg_k - 1) as u64));
    scalar *= P::ScalarField::one().neg();
    scalars.push(scalar);
    commitments.push(*com);
  }

  <P::G1 as VariableBaseMSM>::msm(&commitments, &scalars).unwrap()
}

fn compute_C_Z_x<const N: usize, P: Pairing>(
  f_commitments: &[P::G1],
  q_k_com: &[P::G1Affine],
  rho: &P::ScalarField,
  batched_evaluation: &P::ScalarField,
  x_challenge: &P::ScalarField,
  u_challenge: &[P::ScalarField],
  g1: &P::G1Affine,
) -> P::G1 {
  let n = 1 < q_k_com.len();

  // Phi_n(x) = (x^N - 1) / (x - 1)
  let phi_numerator = x_challenge.pow(BigInt::<1>::from(n as u64)) - P::ScalarField::one(); //x^N - 1
  let phi_n_x = phi_numerator / (*x_challenge - P::ScalarField::one());

  // Add: -v * x * \Phi_n(x) * [1]_1
  let mut scalars = vec![*batched_evaluation * x_challenge * phi_n_x * P::ScalarField::one().neg()];
  let mut commitments = vec![*g1];

  // Add x * \sum_{i=0}^{m-1} \rho^i*[f_i]
  let mut rho_pow = P::ScalarField::one();
  for com in f_commitments {
    scalars.push(*x_challenge * rho_pow);
    commitments.push(com.into_affine());
    rho_pow *= rho;
  }

  // Add: scalar * [q_k], k = 0, ..., log_N, where
  // scalar = -x * (x^{2^k} * \Phi_{n-k-1}(x^{2^{k+1}}) - u_k * \Phi_{n-k}(x^{2^k}))
  let mut x_pow_2k = *x_challenge; // x^{2^k}
  let mut x_pow_2kp1 = *x_challenge * x_challenge; // x^{2^{k + 1}}

  for k in 0..q_k_com.len() {
    let phi_term_1 = phi_numerator / (x_pow_2kp1 - P::ScalarField::one()); // \Phi_{n-k-1}(x^{2^{k + 1}})
    let phi_term_2 = phi_numerator / (x_pow_2k - P::ScalarField::one()); // \Phi_{n-k-1}(x^{2^k})

    let mut scalar = x_pow_2k * phi_term_1;
    scalar -= u_challenge[k] * phi_term_2;
    scalar *= x_challenge;
    scalar *= P::ScalarField::one().neg();

    scalars.push(scalar);
    commitments.push(q_k_com[k]);

    // update powers of challenge x
    x_pow_2k = x_pow_2kp1;
    x_pow_2kp1 *= x_pow_2kp1;
  }

  <P::G1 as VariableBaseMSM>::msm(&commitments, &scalars).unwrap()
}

#[derive(Error, Debug)]
pub enum ZeromorphError {
  #[error("oh no {0}")]
  ShitIsFucked(String),
}

pub struct Zeromorph<const N: usize, P: Pairing> {
  _phantom: PhantomData<P>,
}

/// Compute the powers of a challenge
///
impl<const N: usize, P: Pairing> CommitmentScheme for Zeromorph<N, P> {
  type Commitment = P::G1;
  type Evaluation = P::ScalarField;
  type Polynomial = DensePolynomial<P::ScalarField>;
  type Challenge = P::ScalarField;
  type Proof = ZeromorphProof<P>;
  type Error = ZeromorphError;

  type ProverKey = ZeromorphProverKey<P>;
  type VerifierKey = ZeromorphVerifierKey<P>;

  fn commit(
    polys: &[Self::Polynomial],
    pk: &Self::ProverKey,
  ) -> Result<Vec<Self::Commitment>, Self::Error> {
    Ok(
      polys
        .into_iter()
        .map(|poly| <P::G1 as VariableBaseMSM>::msm(&pk.g1_powers, &poly.Z).unwrap())
        .collect::<Vec<_>>(),
    )
  }

  fn prove(
    polys: &[Self::Polynomial],
    evals: &[Self::Evaluation],
    challenges: &[Self::Challenge],
    pk: impl Borrow<Self::ProverKey>,
    transcript: &mut Transcript,
  ) -> Result<Self::Proof, Self::Error> {
    // ASSERT evaluations, challenges, and polynomials are the same size
    assert_eq!(evals.len(), challenges.len());
    assert_eq!(evals.len(), polys.len());

    let log_N = challenges.len();
    let n = 1 << log_N;
    let pk = pk.borrow();

    // Generate batching challenge \rho and powers 1,...,\rho^{m-1}
    let rho = <Transcript as ProofTranscript<P::G1>>::challenge_scalar(transcript, b"ZM: rho");
    let rhos = (0..evals.len())
      .scan(P::ScalarField::one(), |acc, _| {
        let val = *acc;
        *acc *= rho;
        Some(val)
      })
      .collect::<Vec<P::ScalarField>>();

    // Compute batching of unshifted polynomials f_i:
    // f_batched = sum_{i=0}^{m-1}\rho^i*f_i
    let mut batched_evaluation = P::ScalarField::zero();
    let mut f_batched = vec![P::ScalarField::zero(); n];
    for (i, f_poly) in polys.iter().enumerate() {
      // add_scaled
      for j in 0..f_batched.len() {
        f_batched[j] = f_poly[j] * rhos[i];
      }

      batched_evaluation += rhos[i] * evals[i];
    }
    let f_polynomial = UniPoly::from_coeff(f_batched.clone());

    // Compute the multilinear quotients q_k = q_k(X_0, ..., X_{k-1})
    let (quotients, _) =
      compute_multilinear_quotients::<P>(&DensePolynomial::new(f_batched.clone()), &challenges);

    // Compute and send commitments C_{q_k} = [q_k], k = 0, ..., d-1
    let label = b"q_k_commitments";
    transcript.append_message(label, b"begin_append_vector");
    let q_k_commitments = (0..log_N).into_iter().fold(Vec::new(), |mut acc, i| {
      let q_k_commitment =
        <P::G1 as VariableBaseMSM>::msm(&pk.g1_powers, &quotients[i].coeffs).unwrap();
      transcript.append_point(label, &q_k_commitment);
      acc.push(q_k_commitment.into_affine());
      acc
    });
    transcript.append_message(label, b"end_append_vector");

    // Get challenge y
    let y_challenge =
      <Transcript as ProofTranscript<P::G1>>::challenge_scalar(transcript, b"ZM: y");

    // Compute the batched, lifted-degree quotient \hat{q}
    let q_hat = compute_batched_lifted_degree_quotient::<N, P>(&quotients, &y_challenge);

    // Compute and send the commitment C_q = [\hat{q}]
    let C_q_hat = <P::G1 as VariableBaseMSM>::msm(&pk.g1_powers, &q_hat.coeffs).unwrap();
    transcript.append_point(b"ZM: C_q_hat", &C_q_hat);

    // Get challenges x and z
    let x_challenge =
      <Transcript as ProofTranscript<P::G1>>::challenge_scalar(transcript, b"ZM: x");
    let z_challenge =
      <Transcript as ProofTranscript<P::G1>>::challenge_scalar(transcript, b"ZM: z");

    // Compute degree check polynomials \zeta partially evaluated at x
    let zeta_x = compute_partially_evaluated_degree_check_polynomial::<N, P>(
      &q_hat,
      &quotients,
      &y_challenge,
      &x_challenge,
    );

    // Compute Zeromorph identity polynomial Z partially evaluated at x
    let Z_x = compute_partially_evaluated_zeromorph_identity_polynomial::<N, P>(
      &f_polynomial,
      &quotients,
      &batched_evaluation,
      &challenges,
      &x_challenge,
    );

    // Compute batched degree-check and ZM-identity quotient polynomial pi
    let pi_poly = compute_batched_evaluation_and_degree_check_quotient::<N, P>(
      zeta_x,
      Z_x,
      x_challenge,
      z_challenge,
    );

    let pi = <P::G1 as VariableBaseMSM>::msm(&pk.g1_powers, &pi_poly.coeffs).unwrap();
    transcript.append_point(b"ZM: C_pi", &pi);

    Ok(ZeromorphProof {
      pi: pi.into_affine(),
      q_hat_com: C_q_hat.into_affine(),
      q_k_com: q_k_commitments,
    })
  }

  fn verify(
    commitments: &[Self::Commitment],
    evals: &[Self::Evaluation],
    challenges: &[Self::Challenge],
    vk: impl Borrow<Self::VerifierKey>,
    transcript: &mut Transcript,
    proof: Self::Proof,
  ) -> Result<bool, Self::Error> {
    let vk = vk.borrow();
    let ZeromorphProof {
      pi,
      q_k_com,
      q_hat_com,
    } = proof;
    let pi = pi.into_group();
    let q_k_com = q_k_com;
    let q_hat_com = q_hat_com.into_group();

    // Compute powers of batching challenge rho
    let rho = <Transcript as ProofTranscript<P::G1>>::challenge_scalar(transcript, b"ZM: rho");
    let rhos = (0..evals.len())
      .scan(P::ScalarField::one(), |acc, _| {
        let val = *acc;
        *acc *= rho;
        Some(val)
      })
      .collect::<Vec<P::ScalarField>>();

    // Construct batched evaluations v = sum_{i=0}^{m-1}\rho^i*f_i(u)
    let mut batched_evaluation = P::ScalarField::zero();
    for (i, val) in evals.iter().enumerate() {
      batched_evaluation += *val * rhos[i];
    }

    // Challenge y
    let y_challenge =
      <Transcript as ProofTranscript<P::G1>>::challenge_scalar(transcript, b"ZM: y");

    // Receive commitment C_{q} -> Since our transcript does not support appending and receiving data we instead store these commitments in a zeromorph proof struct

    // Challenge x, z
    let x_challenge =
      <Transcript as ProofTranscript<P::G1>>::challenge_scalar(transcript, b"ZM: x");
    let z_challenge =
      <Transcript as ProofTranscript<P::G1>>::challenge_scalar(transcript, b"ZM: z");

    let C_zeta_x = compute_C_zeta_x::<N, P>(&q_hat_com, &q_k_com, &y_challenge, &x_challenge);
    let C_Z_x = compute_C_Z_x::<N, P>(
      commitments,
      &q_k_com,
      &rho,
      &batched_evaluation,
      &x_challenge,
      &challenges,
      &vk.g1,
    );

    // Compute commitment C_{\zeta,Z}
    let C_zeta_Z = C_zeta_x + C_Z_x * z_challenge;

    // e(pi, [tau]_2 - x * [1]_2) == e(C_{\zeta,Z}, [X^(N_max - 2^n - 1)]_2) <==> e(C_{\zeta,Z} - x * pi, [X^{N_max - 2^n - 1}]_2) * e(-pi, [tau_2]) == 1
    let lhs = P::pairing(pi, vk.tau_2.into_group() - (vk.g2 * x_challenge));
    let rhs = P::pairing(C_zeta_Z, vk.tau_N_max_sub_2_N);
    Ok(lhs == rhs)
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::utils::{math::Math, transcript};
  use ark_bn254::{Bn254, Fr};
  use ark_ff::{BigInt, Zero};
  use ark_std::{test_rng, UniformRand};

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

  fn execute_zeromorph(num_polys: usize) -> bool {
    const N: usize = 64;
    let log_N = N.log_2();

    let mut rng = test_rng();
    let polys: Vec<DensePolynomial<Fr>> = (0..num_polys)
      .map(|_| DensePolynomial::new((0..N).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>()))
      .collect::<Vec<_>>();
    let challenges = (0..log_N)
      .into_iter()
      .map(|_| Fr::rand(&mut rng))
      .collect::<Vec<_>>();
    let evals = polys
      .clone()
      .into_iter()
      .map(|poly| poly.evaluate(&challenges))
      .collect::<Vec<_>>();

    let srs = ZEROMORPH_SRS.lock().unwrap();
    let pk = srs.get_prover_key();
    let vk = srs.get_verifier_key();
    let mut prover_transcript = Transcript::new(b"example");
    let mut verifier_transcript = Transcript::new(b"example");
    let commitments = Zeromorph::<N, Bn254>::commit(&polys.clone(), &pk).unwrap();

    let proof =
      Zeromorph::<N, Bn254>::prove(&polys, &evals, &challenges, &pk, &mut prover_transcript)
        .unwrap();
    Zeromorph::<N, Bn254>::verify(
      &commitments,
      &evals,
      &challenges,
      &vk,
      &mut verifier_transcript,
      proof,
    )
    .unwrap()
  }

  /// Test for computing qk given multilinear f
  /// Given 𝑓(𝑋₀, …, 𝑋ₙ₋₁), and `(𝑢, 𝑣)` such that \f(\u) = \v, compute `qₖ(𝑋₀, …, 𝑋ₖ₋₁)`
  /// such that the following identity holds:
  ///
  /// `𝑓(𝑋₀, …, 𝑋ₙ₋₁) − 𝑣 = ∑ₖ₌₀ⁿ⁻¹ (𝑋ₖ − 𝑢ₖ) qₖ(𝑋₀, …, 𝑋ₖ₋₁)`
  #[test]
  fn quotient_construction() {
    // Define size params
    const N: u64 = 16u64;
    let log_N = (N as usize).log_2();

    // Construct a random multilinear polynomial f, and (u,v) such that f(u) = v
    let mut rng = test_rng();
    let multilinear_f =
      DensePolynomial::new((0..N).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>());
    let u_challenge = (0..log_N)
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
    let z_challenge = (0..log_N).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();

    let mut res = multilinear_f.evaluate(&z_challenge);
    res -= v_evaluation;

    for (k, q_k_uni) in quotients.iter().enumerate() {
      let z_partial = &z_challenge[z_challenge.len() - k..];
      //This is a weird consequence of how things are done.. the univariate polys are of the multilinear commitment in lagrange basis. Therefore we evaluate as multilinear
      let q_k = DensePolynomial::new(q_k_uni.coeffs.clone());
      let q_k_eval = q_k.evaluate(z_partial);

      res -= (z_challenge[z_challenge.len() - k - 1] - u_challenge[z_challenge.len() - k - 1])
        * q_k_eval;
    }
    assert!(res.is_zero());
  }

  /// Test for construction of batched lifted degree quotient:
  ///  ̂q = ∑ₖ₌₀ⁿ⁻¹ yᵏ Xᵐ⁻ᵈᵏ⁻¹ ̂qₖ, 𝑑ₖ = deg(̂q), 𝑚 = 𝑁
  #[test]
  fn batched_lifted_degree_quotient() {
    const N: usize = 8;

    // Define mock qₖ with deg(qₖ) = 2ᵏ⁻¹
    let data_0 = vec![Fr::one()];
    let data_1 = vec![Fr::from(2u64), Fr::from(3u64)];
    let data_2 = vec![
      Fr::from(4u64),
      Fr::from(5u64),
      Fr::from(6u64),
      Fr::from(7u64),
    ];
    let q_0 = UniPoly::from_coeff(data_0);
    let q_1 = UniPoly::from_coeff(data_1);
    let q_2 = UniPoly::from_coeff(data_2);
    let quotients = vec![q_0, q_1, q_2];

    let mut rng = test_rng();
    let y_challenge = Fr::rand(&mut rng);

    //Compute batched quptient  ̂q
    let batched_quotient =
      compute_batched_lifted_degree_quotient::<N, Bn254>(&quotients, &y_challenge);

    //Explicitly define q_k_lifted = X^{N-2^k} * q_k and compute the expected batched result
    //Note: we've hard programmed in the size of these vectors not the best practice
    let data_0_lifted = vec![
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::one(),
    ];
    let data_1_lifted = vec![
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::from(2u64),
      Fr::from(3u64),
    ];
    let data_2_lifted = vec![
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::from(4u64),
      Fr::from(5u64),
      Fr::from(6u64),
      Fr::from(7u64),
    ];
    let q_0_lifted = UniPoly::from_coeff(data_0_lifted);
    let q_1_lifted = UniPoly::from_coeff(data_1_lifted);
    let q_2_lifted = UniPoly::from_coeff(data_2_lifted);

    //Explicitly compute  ̂q i.e. RLC of lifted polys
    let mut batched_quotient_expected = DensePolynomial::new(vec![Fr::zero(); N as usize]);
    //TODO: implement add and add_scalad
    for i in 0..batched_quotient_expected.len() {
      batched_quotient_expected[i] += q_0_lifted[i];
    }

    for i in 0..batched_quotient_expected.len() {
      batched_quotient_expected[i] += q_1_lifted[i] * y_challenge;
    }

    for i in 0..batched_quotient_expected.len() {
      batched_quotient_expected[i] += q_2_lifted[i] * (y_challenge * y_challenge);
    }

    for i in 0..batched_quotient.len() {
      assert_eq!(batched_quotient[i], batched_quotient_expected[i]);
    }
    // Implement PartialEq in DensePolynomial
  }

  /// evaluated quotient \zeta_x
  ///
  /// 𝜁 = 𝑓 − ∑ₖ₌₀ⁿ⁻¹𝑦ᵏ𝑥ʷˢ⁻ʷ⁺¹𝑓ₖ  = 𝑓 − ∑_{d ∈ {d₀, ..., dₙ₋₁}} X^{d* - d + 1}  − ∑{k∶ dₖ=d} yᵏ fₖ , where d* = lifted degree
  ///
  /// 𝜁 =  ̂q - ∑ₖ₌₀ⁿ⁻¹ yᵏ Xᵐ⁻ᵈᵏ⁻¹ ̂qₖ, m = N
  #[test]
  fn partially_evaluated_quotient_zeta() {
    const N: usize = 8;

    // Define mock qₖ with deg(qₖ) = 2ᵏ⁻¹
    let data_0 = vec![Fr::one()];
    let data_1 = vec![Fr::from(2u64), Fr::from(3u64)];
    let data_2 = vec![
      Fr::from(4u64),
      Fr::from(5u64),
      Fr::from(6u64),
      Fr::from(7u64),
    ];
    let q_0 = UniPoly::from_coeff(data_0);
    let q_1 = UniPoly::from_coeff(data_1);
    let q_2 = UniPoly::from_coeff(data_2);
    let quotients = vec![q_0.clone(), q_1.clone(), q_2.clone()];

    let mut rng = test_rng();
    let y_challenge = Fr::rand(&mut rng);

    //Compute batched quptient  ̂q
    let batched_quotient =
      compute_batched_lifted_degree_quotient::<N, Bn254>(&quotients, &y_challenge);
    println!("batched_quotient.len() {:?}", batched_quotient.len());
    dbg!(quotients.clone());

    let x_challenge = Fr::rand(&mut rng);

    let zeta_x = compute_partially_evaluated_degree_check_polynomial::<N, Bn254>(
      &batched_quotient,
      &quotients,
      &y_challenge,
      &x_challenge,
    );

    // Construct 𝜁ₓ explicitly
    let mut zeta_x_expected = UniPoly::from_coeff(vec![Fr::zero(); N as usize]);

    //TODO: implement add and add_scalad
    for i in 0..zeta_x_expected.len() {
      zeta_x_expected[i] += batched_quotient[i];
    }

    // 𝜁 =  ̂q - ∑ₖ₌₀ⁿ⁻¹ yᵏ Xᵐ⁻ᵈᵏ⁻¹ ̂qₖ, m = N
    for i in 0..q_0.len() {
      zeta_x_expected[i] += q_0[i] * -x_challenge.pow(BigInt::<1>::from((N - 0 - 1) as u64));
    }

    for i in 0..q_1.len() {
      zeta_x_expected[i] +=
        q_1[i] * (-y_challenge * x_challenge.pow(BigInt::<1>::from((N - 1 - 1) as u64)));
    }

    for i in 0..q_2.len() {
      zeta_x_expected[i] += q_2[i]
        * (-y_challenge * y_challenge * x_challenge.pow(BigInt::<1>::from((N - 3 - 1) as u64)));
    }

    for i in 0..zeta_x.len() {
      assert_eq!(zeta_x[i], zeta_x_expected[i]);
    }
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
    let efficient =
      (x_challenge.pow(BigInt::<1>::from((1 << log_N) as u64)) - Fr::one()) / (x_pow - Fr::one());
    let expected: Fr = phi::<Bn254>(&x_challenge, log_N - k - 1);
    assert_eq!(efficient, expected);
  }

  /// Test construction of 𝑍ₓ
  /// 𝑍ₓ =  ̂𝑓 − 𝑣 ∑ₖ₌₀ⁿ⁻¹(𝑥²^ᵏ𝛷ₙ₋ₖ₋₁(𝑥ᵏ⁺¹)− 𝑢ₖ𝛷ₙ₋ₖ(𝑥²^ᵏ)) ̂qₖ
  #[test]
  fn partially_evaluated_quotient_z_x() {
    const N: usize = 8;
    let log_N = (N as usize).log_2();

    // Construct a random multilinear polynomial f, and (u,v) such that f(u) = v.
    let mut rng = test_rng();
    let multilinear_f = (0..N)
      .into_iter()
      .map(|_| Fr::rand(&mut rng))
      .collect::<Vec<_>>();
    let u_challenge = (0..log_N)
      .into_iter()
      .map(|_| Fr::rand(&mut rng))
      .collect::<Vec<_>>();
    let v_evaluation = DensePolynomial::new(multilinear_f.clone()).evaluate(&u_challenge);

    // compute batched polynomial and evaluation
    let f_batched = UniPoly::from_coeff(multilinear_f);

    let v_batched = v_evaluation;

    // Define some mock q_k with deeg(q_k) = 2^k - 1
    let q_0 = UniPoly::from_coeff(
      (0..(1 << 0))
        .into_iter()
        .map(|_| Fr::rand(&mut rng))
        .collect::<Vec<_>>(),
    );
    let q_1 = UniPoly::from_coeff(
      (0..(1 << 1))
        .into_iter()
        .map(|_| Fr::rand(&mut rng))
        .collect::<Vec<_>>(),
    );
    let q_2 = UniPoly::from_coeff(
      (0..(1 << 2))
        .into_iter()
        .map(|_| Fr::rand(&mut rng))
        .collect::<Vec<_>>(),
    );
    let quotients = vec![q_0.clone(), q_1.clone(), q_2.clone()];

    let x_challenge = Fr::rand(&mut rng);

    // Construct Z_x using the prover method
    let Z_x = compute_partially_evaluated_zeromorph_identity_polynomial::<N, Bn254>(
      &f_batched,
      &quotients,
      &v_evaluation,
      &u_challenge,
      &x_challenge,
    );

    // Compute Z_x directly
    let mut Z_x_expected = f_batched;

    Z_x_expected[0] =
      Z_x_expected[0] - v_batched * x_challenge * &phi::<Bn254>(&x_challenge, log_N);

    for k in 0..log_N {
      let x_pow_2k = x_challenge.pow(BigInt::<1>::from((1 << k) as u64)); // x^{2^k}
      let x_pow_2kp1 = x_challenge.pow(BigInt::<1>::from((1 << (k + 1)) as u64)); // x^{2^{k+1}}
                                                                                  // x^{2^k} * \Phi_{n-k-1}(x^{2^{k+1}}) - u_k *  \Phi_{n-k}(x^{2^k})
      let mut scalar = x_pow_2k * &phi::<Bn254>(&x_pow_2kp1, log_N - k - 1)
        - u_challenge[k] * &phi::<Bn254>(&x_pow_2k, log_N - k);
      scalar *= x_challenge;
      scalar *= Fr::from(-1);
      for i in 0..quotients[k].len() {
        Z_x_expected[i] += quotients[k][i] * scalar;
      }
    }

    for i in 0..Z_x.len() {
      assert_eq!(Z_x[i], Z_x_expected[i]);
    }
  }

  #[test]
  fn prove_verify_single() {
    assert!(execute_zeromorph(1));
  }

  #[test]
  fn prove_and_verify_batched() {
    assert!(execute_zeromorph(10));
  }
}
