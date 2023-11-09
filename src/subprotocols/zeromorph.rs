#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use std::marker::PhantomData;
use std::ops::Index;

use crate::poly::commitments::MultiCommitGens;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::unipoly::{CompressedUniPoly, UniPoly};
use crate::subprotocols::dot_product::DotProductProof;
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;
use crate::utils::transcript::{AppendToTranscript, ProofTranscript};
use ark_ec::short_weierstrass::{Affine, SWCurveConfig};
use ark_ec::{pairing::Pairing, CurveConfig, CurveGroup, Group};
use ark_ff::{BigInt, BigInteger, Field, PrimeField};
use ark_serialize::*;
use ark_std::{One, Zero};
use merlin::Transcript;

#[cfg(feature = "ark-msm")]
use ark_ec::VariableBaseMSM;

#[cfg(not(feature = "ark-msm"))]
use crate::msm::VariableBaseMSM;

#[cfg(feature = "multicore")]
use rayon::prelude::*;

pub struct Proof<P: Pairing> {
  _phantom: PhantomData<P>,
}

pub struct CommitmentKey<P: Pairing> {
  _phantom: PhantomData<P>,
}

pub struct Zeromorph<const N: u64, P: Pairing> {
  _phantom: PhantomData<P>,
}

/// Compute the powers of a challenge
///
impl<const N: u64, P: Pairing> Zeromorph<N, P> {
  pub fn new() -> Self {
    Self {
      _phantom: PhantomData::<P>,
    }
  }

  fn powers_of_challenge(challenge: P::ScalarField, num_powers: usize) -> Vec<P::ScalarField> {
    //TODO: switch to successors
    (2..num_powers).fold(vec![P::ScalarField::one(), challenge], |mut acc, i| {
      acc.push(acc[i - 1] * challenge);
      acc
    })
  }

  pub fn compute_multilinear_quotients(
    polynimial: &DensePolynomial<P::ScalarField>,
    u_challenge: &[P::ScalarField],
  ) -> Vec<DensePolynomial<P::ScalarField>> {
    // TODO: can grab from poly
    let log_N = (N as usize).log_2();

    // The size of the multilinear challenge must equal the log of the polynomial size
    assert!(log_N == u_challenge.len());

    // Define vector of quotient polynomials q_k, k = 0, ..., log_N - 1
    let mut quotients = (0..log_N)
      .into_iter()
      .map(|_| DensePolynomial::from_usize(&[0]))
      .collect::<Vec<_>>();
    println!(
      "log_N {:?} N {:?} quotients {:?}",
      log_N,
      N,
      quotients.len()
    );

    // Compute the coefficients of q_{n - 1}
    let mut size_q = (1 << (log_N - 1)) as usize;
    //TODO: check if this is correct. Based on Barretenburg's mle I think it is???
    let q = DensePolynomial::new((0..size_q).fold(Vec::new(), |mut acc, l| {
      acc.push(polynimial[size_q + l] - polynimial[l]);
      acc
    }));

    //Probs can't avoid this clone
    quotients.insert(log_N - 1, q.clone());

    let mut f_k: Vec<P::ScalarField> = vec![P::ScalarField::zero(); size_q];

    //We can probably clean this up some but for now we're being explicit
    let mut g = (0..size_q).fold(Vec::new(), |mut acc, i| {
      acc.push(polynimial[i]);
      acc
    });

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

  pub fn compute_batched_lifted_degree_quotient(
    quotients: &Vec<DensePolynomial<P::ScalarField>>,
    y_challenge: &P::ScalarField,
  ) -> UniPoly<P::ScalarField> {
    // Batched Lifted Degreee Quotient Polynomials
    let mut res: Vec<P::ScalarField> = Vec::with_capacity(N as usize);

    // Compute \hat{q} = \sum_k y^k * X^{N - d_k - 1} * q_k
    let mut scalar = P::ScalarField::one();
    for (k, quotient) in quotients.iter().enumerate() {
      // Rather than explicitly computing the shifts of q_k by N - d_k - 1 (i.e. multiplying q_k by X^{N - d_k -
      // 1}) then accumulating them, we simply accumulate y^k*q_k into \hat{q} at the index offset N - d_k - 1
      //TODO: verify if this is needed as we are not interested in shifts
      let deg_k = (1 << k) as usize - 1;
      let offset = N as usize - deg_k - 1;
      for i in 0..(deg_k + 1) {
        res[offset + i] += scalar * quotient[i];
      }
      scalar *= y_challenge; // update batching scalar y^k
    }

    UniPoly::from_coeff(res)
  }

  pub fn compute_partially_evaluated_degree_check_polynomial(
    batched_quotient: &UniPoly<P::ScalarField>,
    quotients: &Vec<DensePolynomial<P::ScalarField>>,
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
      for i in 0..res.len() {
        res[i] += quotients[k][i] * (-y_power * x_power);
      }

      y_power *= y_challenge; // updated batching scalar y^k
    }

    res
  }

  pub fn compute_partially_evaluated_zeromorph_identity_polynomial(
    f_batched: &UniPoly<P::ScalarField>,
    g_batched: &UniPoly<P::ScalarField>,
    quotients: &Vec<UniPoly<P::ScalarField>>,
    v_evaluation: &P::ScalarField,
    u_challenge: &[P::ScalarField],
    x_challenge: &P::ScalarField,
  ) -> UniPoly<P::ScalarField> {
    let n = f_batched.len();
    let log_N = quotients.len();

    //Question for non-shifted can we exclude sum_{i=0}^{l-i}
    // Initialize Z_x with x * \sum_{i=0}^{m-1} f_i + /sum_{i=0}^{l-i} * g_i
    let mut res: UniPoly<P::ScalarField> = g_batched.clone();

    //add scaled
    for i in 0..res.len() {
      res[i] += f_batched[i] * x_challenge;
    }

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
      let phi_term_2 = phi_numerator
        / (x_challenge.pow(BigInt::<1>::from((1 << k) as u64)) - P::ScalarField::one());

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

  pub fn compute_batched_evaluation_and_degree_check_quotient(
    zeta_x: UniPoly<P::ScalarField>,
    z_x: UniPoly<P::ScalarField>,
    x_challenge: P::ScalarField,
    z_challenge: P::ScalarField,
  ) -> UniPoly<P::ScalarField> {
    // We cannont commit to polynomials with size > N_max
    let n = zeta_x.len();
    assert!(n <= N as usize);

    // Compute q_{\zeta} and q_Z in place
    let mut batched_quotient = zeta_x;
    for i in 0..batched_quotient.len() {
      batched_quotient[i] += z_x[i] * z_challenge;
    }

    //TODO: finish once srs gen is completed

    batched_quotient
  }

  pub fn prove(
    f_polynomials: Vec<DensePolynomial<P::ScalarField>>,
    evaluations: Vec<P::ScalarField>,
    multilinear_challenge: Vec<P::ScalarField>,
    commitment_key: CommitmentKey<P>,
    transcript: Transcript,
  ) -> P::G1 {
    todo!();
  }

  pub fn compute_C_zeta_x(
    C_q: P::G1,
    C_q_k: Vec<P::G1>,
    y_challenge: &P::ScalarField,
    x_challenge: &P::ScalarField,
  ) -> P::G1 {
    todo!()
  }

  pub fn compute_C_z_X(
    f_commitments: Vec<P::G1>,
    g_commitments: Vec<P::G1>,
    C_q_k: &Vec<P::G1>,
    rho: &P::ScalarField,
    batched_evaluation: &P::ScalarField,
    x_challenge: &P::ScalarField,
    u_challenge: &[P::ScalarField],
  ) -> P::G1 {
    todo!();
  }

  pub fn verify() {
    todo!()
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::utils::math::Math;
  use crate::utils::test::TestTranscript;
  use ark_bn254::{Bn254, Fq, Fr, G1Affine, G1Projective};
  use ark_ff::{BigInt, Zero};
  use ark_std::{test_rng, UniformRand};

  // Evaluate Phi_k(x) = \sum_{i=0}^k x^i using the direct inefficent formula
  fn phi<P: Pairing>(challenge: &P::ScalarField, subscript: usize) -> P::ScalarField {
    let len = (1 << subscript) as u64;
    let res = P::ScalarField::zero();
    (0..len)
      .into_iter()
      .fold(P::ScalarField::zero(), |mut acc, i| {
        //Note this is ridiculous DevX
        acc += challenge.pow(BigInt::<1>::from(i));
        acc
      });
    res
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
    let multilinear_f = DensePolynomial::new(
      (0..N)
        .into_iter()
        .map(|_| Fr::rand(&mut rng))
        .collect::<Vec<_>>(),
    );
    let u_challenge = (0..log_N)
      .into_iter()
      .map(|_| Fr::rand(&mut rng))
      .collect::<Vec<_>>();
    let v_evaluation = multilinear_f.evaluate(&u_challenge);

    // Compute multilinear quotients `qₖ(𝑋₀, …, 𝑋ₖ₋₁)`
    let quotients =
      Zeromorph::<N, Bn254>::compute_multilinear_quotients(&multilinear_f, &u_challenge);

    //To demonstrate that q_k was properly constructd we show that the identity holds at a random multilinear challenge
    // i.e. 𝑓(𝑧) − 𝑣 − ∑ₖ₌₀ᵈ⁻¹ (𝑧ₖ − 𝑢ₖ)𝑞ₖ(𝑧) = 0
    let z_challenge = (0..log_N)
      .into_iter()
      .map(|_| Fr::rand(&mut rng))
      .collect::<Vec<_>>();

    let mut res = multilinear_f.evaluate(&z_challenge);
    res -= v_evaluation;
    for k in 0..log_N {
      let q_k_eval;
      if k == 0 {
        // 𝑞₀ = 𝑎₀ is a constant polynomial so it's evaluation is simply its constant coefficient
        q_k_eval = quotients[k][0];
      } else {
        // Construct (𝑢₀, ..., 𝑢ₖ₋₁)
        q_k_eval = quotients[k].evaluate(&z_challenge[..k]);
      }
      // res = res - (𝑧ₖ − 𝑢ₖ) * 𝑞ₖ(𝑢₀, ..., 𝑢ₖ₋₁)
      res -= (z_challenge[k] - u_challenge[k]) * q_k_eval;
    }
    assert_eq!(res, Fr::zero());
  }

  /// Test for construction of batched lifted degree quotient:
  ///  ̂q = ∑ₖ₌₀ⁿ⁻¹ yᵏ Xᵐ⁻ᵈᵏ⁻¹ ̂qₖ, 𝑑ₖ = deg(̂q), 𝑚 = 𝑁
  #[test]
  fn batched_lifted_degree_quotient() {
    const N: u64 = 8u64;

    // Define mock qₖ with deg(qₖ) = 2ᵏ⁻¹
    let data_0 = vec![Fr::one()];
    let data_1 = vec![Fr::from(2u64), Fr::from(3u64)];
    let data_2 = vec![
      Fr::from(4u64),
      Fr::from(5u64),
      Fr::from(6u64),
      Fr::from(7u64),
    ];
    let q_0 = DensePolynomial::new(data_0);
    let q_1 = DensePolynomial::new(data_1);
    let q_2 = DensePolynomial::new(data_2);
    let quotients = vec![q_0, q_1, q_2];

    let mut rng = test_rng();
    let y_challenge = Fr::rand(&mut rng);

    //Compute batched quptient  ̂q
    let batched_quotient =
      Zeromorph::<N, Bn254>::compute_batched_lifted_degree_quotient(&quotients, &y_challenge);

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
    let q_0_lifted = DensePolynomial::new(data_0_lifted);
    let q_1_lifted = DensePolynomial::new(data_1_lifted);
    let q_2_lifted = DensePolynomial::new(data_2_lifted);

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
    const N: u64 = 8u64;

    // Define mock qₖ with deg(qₖ) = 2ᵏ⁻¹
    let data_0 = vec![Fr::one()];
    let data_1 = vec![Fr::from(2u64), Fr::from(3u64)];
    let data_2 = vec![
      Fr::from(4u64),
      Fr::from(5u64),
      Fr::from(6u64),
      Fr::from(7u64),
    ];
    let q_0 = DensePolynomial::new(data_0);
    let q_1 = DensePolynomial::new(data_1);
    let q_2 = DensePolynomial::new(data_2);
    let quotients = vec![q_0.clone(), q_1.clone(), q_2.clone()];

    let mut rng = test_rng();
    let y_challenge = Fr::rand(&mut rng);

    //Compute batched quptient  ̂q
    let batched_quotient =
      Zeromorph::<N, Bn254>::compute_batched_lifted_degree_quotient(&quotients, &y_challenge);

    let x_challenge = Fr::rand(&mut rng);

    let zeta_x = Zeromorph::<N, Bn254>::compute_partially_evaluated_degree_check_polynomial(
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
    for i in 0..zeta_x_expected.len() {
      zeta_x_expected[i] += q_0[i] * -x_challenge.pow(BigInt::<1>::from((N - 0 - 1) as u64));
    }

    for i in 0..zeta_x_expected.len() {
      zeta_x_expected[i] +=
        q_1[i] * (-y_challenge * x_challenge.pow(BigInt::<1>::from((N - 1 - 1) as u64)));
    }

    for i in 0..zeta_x_expected.len() {
      zeta_x_expected[i] += q_1[i]
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
    const N: u64 = 8u64;
    let log_N = (N as usize).log_2();

    // Construct a random multilinear polynomial f, and (u,v) such that f(u) = v.
    let mut rng = test_rng();
    let multilinear_f = (0..N)
      .into_iter()
      .map(|_| Fr::rand(&mut rng))
      .collect::<Vec<_>>();
    let mut multilinear_g = (0..N)
      .into_iter()
      .map(|_| Fr::rand(&mut rng))
      .collect::<Vec<_>>();
    multilinear_g[0] = Fr::zero();
    let u_challenge = (0..log_N)
      .into_iter()
      .map(|_| Fr::rand(&mut rng))
      .collect::<Vec<_>>();
    let v_evaluation = DensePolynomial::new(multilinear_f.clone()).evaluate(&u_challenge);
    let w_evaluation = DensePolynomial::new(multilinear_g.clone()).evaluate(&u_challenge); // This says shifted??? mayhaps exclude -> first ask Kobi

    let rho = Fr::rand(&mut rng);

    // compute batched polynomial and evaluation
    let f_batched = UniPoly::from_coeff(multilinear_f);
    let mut g_batched = UniPoly::from_coeff(multilinear_g);

    for i in 0..g_batched.len() {
      g_batched[i] = g_batched[i] * rho;
    }
    let v_batched = v_evaluation + rho * w_evaluation;

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
    let Z_x = Zeromorph::<N, Bn254>::compute_partially_evaluated_zeromorph_identity_polynomial(
      &f_batched,
      &g_batched,
      &quotients,
      &v_evaluation,
      &u_challenge,
      &x_challenge,
    );

    // Compute Z_x directly
    let mut Z_x_expected = g_batched;
    for i in 0..Z_x_expected.len() {
      Z_x_expected[i] += f_batched[i] * x_challenge;
    }

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
      for i in 0..Z_x_expected.len() {
        Z_x_expected[i] += quotients[k][i] * scalar;
      }
    }

    for i in 0..Z_x.len() {
      assert_eq!(Z_x[i], Z_x_expected[i]);
    }
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
