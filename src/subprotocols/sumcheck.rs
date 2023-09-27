#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::poly::commitments::MultiCommitGens;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::unipoly::{CompressedUniPoly, UniPoly};
use crate::subprotocols::dot_product::DotProductProof;
use crate::utils::errors::ProofVerifyError;
use crate::utils::transcript::{AppendToTranscript, ProofTranscript};
use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_serialize::*;
use ark_std::One;
use merlin::Transcript;

#[cfg(feature = "ark-msm")]
use ark_ec::VariableBaseMSM;

#[cfg(not(feature = "ark-msm"))]
use crate::msm::VariableBaseMSM;

#[cfg(feature = "multicore")]
use rayon::prelude::*;

impl<F: PrimeField> SumcheckInstanceProof<F> {
  #[tracing::instrument(skip_all, name = "Sumcheck.prove_batched")]
  pub fn prove_cubic_batched<Func, G>(
    claim: &F,
    num_rounds: usize,
    poly_vec_par: (
      &mut Vec<&mut DensePolynomial<F>>,
      &mut Vec<&mut DensePolynomial<F>>,
      &mut DensePolynomial<F>,
    ),
    coeffs: &[F],
    comb_func: Func,
    transcript: &mut Transcript,
  ) -> (Self, Vec<F>, (Vec<F>, Vec<F>, F))
  where
    Func: Fn(&F, &F, &F) -> F + Sync,
    G: CurveGroup<ScalarField = F>,
  {
    let (poly_A_vec_par, poly_B_vec_par, poly_C_par) = poly_vec_par;

    let mut e = *claim;
    let mut r: Vec<F> = Vec::new();
    let mut cubic_polys: Vec<CompressedUniPoly<F>> = Vec::new();

    for _j in 0..num_rounds {
      #[cfg(feature = "multicore")]
      let iterator = poly_A_vec_par.par_iter().zip(poly_B_vec_par.par_iter());

      #[cfg(not(feature = "multicore"))]
      let iterator = poly_A_vec_par.iter().zip(poly_B_vec_par.iter());

      let evals: Vec<(F, F, F)> = iterator
        .map(|(poly_A, poly_B)| {
          let mut eval_point_0 = F::zero();
          let mut eval_point_2 = F::zero();
          let mut eval_point_3 = F::zero();

          let len = poly_A.len() / 2;
          for i in 0..len {
            // TODO(#28): Optimize

            // eval 0: bound_func is A(low)
            eval_point_0 += comb_func(&poly_A[i], &poly_B[i], &poly_C_par[i]);

            // eval 2: bound_func is -A(low) + 2*A(high)
            let poly_A_bound_point = poly_A[len + i] + poly_A[len + i] - poly_A[i];
            let poly_B_bound_point = poly_B[len + i] + poly_B[len + i] - poly_B[i];
            let poly_C_bound_point = poly_C_par[len + i] + poly_C_par[len + i] - poly_C_par[i];
            eval_point_2 += comb_func(
              &poly_A_bound_point,
              &poly_B_bound_point,
              &poly_C_bound_point,
            );

            // eval 3: bound_func is -2A(low) + 3A(high); computed incrementally with bound_func applied to eval(2)
            let poly_A_bound_point = poly_A_bound_point + poly_A[len + i] - poly_A[i];
            let poly_B_bound_point = poly_B_bound_point + poly_B[len + i] - poly_B[i];
            let poly_C_bound_point = poly_C_bound_point + poly_C_par[len + i] - poly_C_par[i];

            eval_point_3 += comb_func(
              &poly_A_bound_point,
              &poly_B_bound_point,
              &poly_C_bound_point,
            );
          }

          (eval_point_0, eval_point_2, eval_point_3)
        })
        .collect();

      let evals_combined_0 = (0..evals.len()).map(|i| evals[i].0 * coeffs[i]).sum();
      let evals_combined_2 = (0..evals.len()).map(|i| evals[i].1 * coeffs[i]).sum();
      let evals_combined_3 = (0..evals.len()).map(|i| evals[i].2 * coeffs[i]).sum();

      let evals = vec![
        evals_combined_0,
        e - evals_combined_0,
        evals_combined_2,
        evals_combined_3,
      ];
      let poly = UniPoly::from_evals(&evals);

      // append the prover's message to the transcript
      <UniPoly<F> as AppendToTranscript<G>>::append_to_transcript(&poly, b"poly", transcript);

      //derive the verifier's challenge for the next round
      let r_j =
        <Transcript as ProofTranscript<G>>::challenge_scalar(transcript, b"challenge_nextround");
      r.push(r_j);

      // bound all tables to the verifier's challenege
      for (poly_A, poly_B) in poly_A_vec_par.iter_mut().zip(poly_B_vec_par.iter_mut()) {
        poly_A.bound_poly_var_top(&r_j);
        poly_B.bound_poly_var_top(&r_j);
      }
      poly_C_par.bound_poly_var_top(&r_j);

      e = poly.evaluate(&r_j);
      cubic_polys.push(poly.compress());
    }

    let poly_A_par_final = (0..poly_A_vec_par.len())
      .map(|i| poly_A_vec_par[i][0])
      .collect();
    let poly_B_par_final = (0..poly_B_vec_par.len())
      .map(|i| poly_B_vec_par[i][0])
      .collect();
    let claims_prod = (poly_A_par_final, poly_B_par_final, poly_C_par[0]);

    (SumcheckInstanceProof::new(cubic_polys), r, claims_prod)
  }

  /// Create a sumcheck proof for polynomial(s) of arbitrary degree.
  ///
  /// Params
  /// - `claim`: Claimed sumcheck evaluation (note: currently unused)
  /// - `num_rounds`: Number of rounds of sumcheck, or number of variables to bind
  /// - `polys`: Dense polynomials to combine and sumcheck
  /// - `comb_func`: Function used to combine each polynomial evaluation
  /// - `transcript`: Fiat-shamir transcript
  ///
  /// Returns (SumcheckInstanceProof, r_eval_point, final_evals)
  /// - `r_eval_point`: Final random point of evaluation
  /// - `final_evals`: Each of the polys evaluated at `r_eval_point`
  #[tracing::instrument(skip_all, name = "Sumcheck.prove")]
  pub fn prove_arbitrary<Func, G, T: ProofTranscript<G>, const ALPHA: usize>(
    _claim: &F,
    num_rounds: usize,
    polys: &mut [DensePolynomial<F>; ALPHA],
    comb_func: Func,
    combined_degree: usize,
    transcript: &mut T,
  ) -> (Self, Vec<F>, Vec<F>)
  where
    Func: Fn(&[F; ALPHA]) -> F + Sync,
    G: CurveGroup<ScalarField = F>,
  {
    let mut r: Vec<F> = Vec::new();
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::new();

    for _round in 0..num_rounds {
      // Vector storing evaluations of combined polynomials g(x) = P_0(x) * ... P_{num_polys} (x)
      // for points {0, ..., |g(x)|}
      let mut eval_points = vec![F::zero(); combined_degree + 1];

      let mle_half = polys[0].len() / 2;

      // let mut accum = vec![vec![F::zero(); combined_degree + 1]; mle_half];
      #[cfg(feature = "multicore")]
      let iterator = (0..mle_half).into_par_iter();

      #[cfg(not(feature = "multicore"))]
      let iterator = 0..mle_half;

      let accum: Vec<Vec<F>> = iterator
        .map(|poly_term_i| {
          let mut accum = vec![F::zero(); combined_degree + 1];
          // Evaluate P({0, ..., |g(r)|})

          // TODO(#28): Optimize
          // Tricks can be used here for low order bits {0,1} but general premise is a running sum for each
          // of the m terms in the Dense multilinear polynomials. Formula is:
          // half = | D_{n-1} | / 2
          // D_n(index, r) = D_{n-1}[half + index] + r * (D_{n-1}[half + index] - D_{n-1}[index])

          // eval 0: bound_func is A(low)
          // eval_points[0] += comb_func(&polys.iter().map(|poly| poly[poly_term_i]).collect());
          accum[0] += comb_func(&std::array::from_fn(|j| polys[j][poly_term_i]));

          // TODO(#28): Can be computed from prev_round_claim - eval_point_0
          let eval_at_one: [F; ALPHA] = std::array::from_fn(|j| polys[j][mle_half + poly_term_i]);
          accum[1] += comb_func(&eval_at_one);

          // D_n(index, r) = D_{n-1}[half + index] + r * (D_{n-1}[half + index] - D_{n-1}[index])
          // D_n(index, 0) = D_{n-1} +
          // D_n(index, 1) = D_{n-1} + (D_{n-1}[HIGH] - D_{n-1}[LOW])
          // D_n(index, 2) = D_{n-1} + (D_{n-1}[HIGH] - D_{n-1}[LOW]) + (D_{n-1}[HIGH] - D_{n-1}[LOW])
          // D_n(index, 3) = D_{n-1} + (D_{n-1}[HIGH] - D_{n-1}[LOW]) + (D_{n-1}[HIGH] - D_{n-1}[LOW]) + (D_{n-1}[HIGH] - D_{n-1}[LOW])
          // ...
          let mut existing_term = eval_at_one;
          for acc in accum.iter_mut().skip(2) {
            let mut poly_evals = [F::zero(); ALPHA];
            for poly_i in 0..polys.len() {
              let poly = &polys[poly_i];
              poly_evals[poly_i] =
                existing_term[poly_i] + poly[mle_half + poly_term_i] - poly[poly_term_i];
            }

            *acc += comb_func(&poly_evals);
            existing_term = poly_evals;
          }
          accum
        })
        .collect();

      #[cfg(feature = "multicore")]
      eval_points
        .par_iter_mut()
        .enumerate()
        .for_each(|(poly_i, eval_point)| {
          *eval_point = accum
            .par_iter()
            .take(mle_half)
            .map(|mle| mle[poly_i])
            .sum::<F>();
        });

      #[cfg(not(feature = "multicore"))]
      for (poly_i, eval_point) in eval_points.iter_mut().enumerate() {
        for mle in accum.iter().take(mle_half) {
          *eval_point += mle[poly_i];
        }
      }

      let round_uni_poly = UniPoly::from_evals(&eval_points);

      // append the prover's message to the transcript
      <UniPoly<F> as AppendToTranscript<G>>::append_to_transcript(
        &round_uni_poly,
        b"poly",
        transcript,
      );
      let r_j = transcript.challenge_scalar(b"challenge_nextround");
      r.push(r_j);

      // bound all tables to the verifier's challenege
      for poly in polys.iter_mut() {
        poly.bound_poly_var_top(&r_j);
      }
      compressed_polys.push(round_uni_poly.compress());
    }

    let final_evals = polys.iter().map(|poly| poly[0]).collect();

    (SumcheckInstanceProof::new(compressed_polys), r, final_evals)
  }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct SumcheckInstanceProof<F: PrimeField> {
  compressed_polys: Vec<CompressedUniPoly<F>>,
}

impl<F: PrimeField> SumcheckInstanceProof<F> {
  pub fn new(compressed_polys: Vec<CompressedUniPoly<F>>) -> SumcheckInstanceProof<F> {
    SumcheckInstanceProof { compressed_polys }
  }

  /// Verify this sumcheck proof.
  /// Note: Verification does not execute the final check of sumcheck protocol: g_v(r_v) = oracle_g(r),
  /// as the oracle is not passed in. Expected that the caller will implement.
  ///
  /// Params
  /// - `claim`: Claimed evaluation
  /// - `num_rounds`: Number of rounds of sumcheck, or number of variables to bind
  /// - `degree_bound`: Maximum allowed degree of the combined univariate polynomial
  /// - `transcript`: Fiat-shamir transcript
  ///
  /// Returns (e, r)
  /// - `e`: Claimed evaluation at random point
  /// - `r`: Evaluation point
  pub fn verify<G, T: ProofTranscript<G>>(
    &self,
    claim: F,
    num_rounds: usize,
    degree_bound: usize,
    transcript: &mut T,
  ) -> Result<(F, Vec<F>), ProofVerifyError>
  where
    G: CurveGroup<ScalarField = F>,
  {
    let mut e = claim;
    let mut r: Vec<F> = Vec::new();

    // verify that there is a univariate polynomial for each round
    assert_eq!(self.compressed_polys.len(), num_rounds);
    for i in 0..self.compressed_polys.len() {
      let poly = self.compressed_polys[i].decompress(&e);

      // verify degree bound
      if poly.degree() != degree_bound {
        return Err(ProofVerifyError::InvalidInputLength(
          degree_bound,
          poly.degree(),
        ));
      }

      // check if G_k(0) + G_k(1) = e
      assert_eq!(poly.eval_at_zero() + poly.eval_at_one(), e);

      // append the prover's message to the transcript
      <UniPoly<F> as AppendToTranscript<G>>::append_to_transcript(&poly, b"poly", transcript);

      //derive the verifier's challenge for the next round
      let r_i = transcript.challenge_scalar(b"challenge_nextround");

      r.push(r_i);

      // evaluate the claimed degree-ell polynomial at r_i
      e = poly.evaluate(&r_i);
    }

    Ok((e, r))
  }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct ZKSumcheckInstanceProof<G: CurveGroup> {
  comm_polys: Vec<G>,
  comm_evals: Vec<G>,
  proofs: Vec<DotProductProof<G>>,
}

#[allow(dead_code)]
impl<G: CurveGroup> ZKSumcheckInstanceProof<G> {
  pub fn new(comm_polys: Vec<G>, comm_evals: Vec<G>, proofs: Vec<DotProductProof<G>>) -> Self {
    ZKSumcheckInstanceProof {
      comm_polys,
      comm_evals,
      proofs,
    }
  }

  pub fn verify(
    &self,
    comm_claim: &G,
    num_rounds: usize,
    degree_bound: usize,
    gens_1: &MultiCommitGens<G>,
    gens_n: &MultiCommitGens<G>,
    transcript: &mut Transcript,
  ) -> Result<(G, Vec<G::ScalarField>), ProofVerifyError> {
    // verify degree bound
    assert_eq!(gens_n.n, degree_bound + 1);

    // verify that there is a univariate polynomial for each round
    assert_eq!(self.comm_polys.len(), num_rounds);
    assert_eq!(self.comm_evals.len(), num_rounds);

    let mut r: Vec<G::ScalarField> = Vec::new();
    for i in 0..self.comm_polys.len() {
      let comm_poly = &self.comm_polys[i];

      // append the prover's polynomial to the transcript
      <Transcript as ProofTranscript<G>>::append_point(transcript, b"comm_poly", comm_poly);

      //derive the verifier's challenge for the next round
      let r_i =
        <Transcript as ProofTranscript<G>>::challenge_scalar(transcript, b"challenge_nextround");

      // verify the proof of sum-check and evals
      let res = {
        let comm_claim_per_round = if i == 0 {
          comm_claim
        } else {
          &self.comm_evals[i - 1]
        };
        let comm_eval = &self.comm_evals[i];

        // add two claims to transcript
        <Transcript as ProofTranscript<G>>::append_point(
          transcript,
          b"comm_claim_per_round",
          comm_claim_per_round,
        );
        <Transcript as ProofTranscript<G>>::append_point(transcript, b"comm_eval", comm_eval);

        // produce two weights
        let w = <Transcript as ProofTranscript<G>>::challenge_vector(
          transcript,
          b"combine_two_claims_to_one",
          2,
        );

        // compute a weighted sum of the RHS
        let bases = vec![comm_claim_per_round.into_affine(), comm_eval.into_affine()];

        let comm_target = VariableBaseMSM::msm(bases.as_ref(), w.as_ref()).unwrap();

        let a = {
          // the vector to use to decommit for sum-check test
          let a_sc = {
            let mut a = vec![G::ScalarField::one(); degree_bound + 1];
            a[0] += G::ScalarField::one();
            a
          };

          // the vector to use to decommit for evaluation
          let a_eval = {
            let mut a = vec![G::ScalarField::one(); degree_bound + 1];
            for j in 1..a.len() {
              a[j] = a[j - 1] * r_i;
            }
            a
          };

          // take weighted sum of the two vectors using w
          assert_eq!(a_sc.len(), a_eval.len());
          (0..a_sc.len())
            .map(|i| w[0] * a_sc[i] + w[1] * a_eval[i])
            .collect::<Vec<G::ScalarField>>()
        };

        self.proofs[i]
          .verify(
            gens_1,
            gens_n,
            transcript,
            &a,
            &self.comm_polys[i],
            &comm_target,
          )
          .is_ok()
      };
      if !res {
        return Err(ProofVerifyError::InternalError);
      }

      r.push(r_i);
    }

    Ok((self.comm_evals[self.comm_evals.len() - 1], r))
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::utils::math::Math;
  use crate::utils::test::TestTranscript;
  use ark_curve25519::{EdwardsProjective as G1Projective, Fr};
  use ark_ff::Zero;

  #[test]
  fn sumcheck_arbitrary_cubic() {
    // Create three dense polynomials (all the same)
    let num_vars = 3;
    let num_evals = num_vars.pow2();
    let mut evals: Vec<Fr> = Vec::with_capacity(num_evals);
    for i in 0..num_evals {
      evals.push(Fr::from(8 + i as u64));
    }

    let A: DensePolynomial<Fr> = DensePolynomial::new(evals.clone());
    let B: DensePolynomial<Fr> = DensePolynomial::new(evals.clone());
    let C: DensePolynomial<Fr> = DensePolynomial::new(evals.clone());

    let mut claim = Fr::zero();
    for i in 0..num_evals {
      use crate::utils::index_to_field_bitvector;

      claim += A.evaluate(&index_to_field_bitvector(i, num_vars))
        * B.evaluate(&index_to_field_bitvector(i, num_vars))
        * C.evaluate(&index_to_field_bitvector(i, num_vars));
    }
    let mut polys = [A.clone(), B.clone(), C.clone()];

    let comb_func_prod =
      |polys: &[Fr; 3]| -> Fr { polys.iter().fold(Fr::one(), |acc, poly| acc * *poly) };

    let r = vec![Fr::from(3), Fr::from(1), Fr::from(3)]; // point 0,0,0 within the boolean hypercube

    let mut transcript: TestTranscript<Fr> = TestTranscript::new(r.clone(), vec![]);
    let (proof, prove_randomness, _final_poly_evals) =
      SumcheckInstanceProof::<Fr>::prove_arbitrary::<_, G1Projective, _, 3>(
        &claim,
        num_vars,
        &mut polys,
        comb_func_prod,
        3,
        &mut transcript,
      );

    let mut transcript: TestTranscript<Fr> = TestTranscript::new(r.clone(), vec![]);
    let verify_result = proof.verify::<G1Projective, _>(claim, num_vars, 3, &mut transcript);
    assert!(verify_result.is_ok());

    let (verify_evaluation, verify_randomness) = verify_result.unwrap();
    assert_eq!(prove_randomness, verify_randomness);
    assert_eq!(prove_randomness, r);

    // Consider this the opening proof to a(r) * b(r) * c(r)
    let a = A.evaluate(prove_randomness.as_slice());
    let b = B.evaluate(prove_randomness.as_slice());
    let c = C.evaluate(prove_randomness.as_slice());

    let oracle_query = a * b * c;
    assert_eq!(verify_evaluation, oracle_query);
  }
}
