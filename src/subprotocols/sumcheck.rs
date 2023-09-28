#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::jolt::instruction::{JoltInstruction, Opcode};
use crate::jolt::vm::Jolt;
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
use strum::IntoEnumIterator;

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

  /// Prove Jolt primary sumcheck including instruction collation.
  ///
  /// Computes \sum{ eq(r,x) * [ flags_0(x) * g_0(E(x)) + flags_1(x) * g_1(E(x)) + ... + flags_{NUM_INSTRUCTIONS}(E(x)) * g_{NUM_INSTRUCTIONS}(E(x)) ]}
  /// via the sumcheck protocol.
  /// Note: These E(x) terms differ from term to term depending on the memories used in the instruction.
  ///
  /// Returns: (SumcheckProof, Random evaluation point, claimed evaluations of polynomials)
  ///
  /// Params:
  /// - `claim`: Claimed sumcheck evaluation.
  /// - `num_rounds`: Number of rounds to run sumcheck. Corresponds to the number of free bits or free variables in the polynomials.
  /// - `memory_polys`: Each of the `E` polynomials or "dereferenced memory" polynomials.
  /// - `flag_polys`: Each of the flag selector polynomials describing which instruction is used at a given step of the CPU.
  /// - `degree`: Degree of the inner sumcheck polynomial. Corresponds to number of evaluation points per round.
  /// - `transcript`: Fiat-shamir transcript.
  pub fn prove_jolt<
    G: CurveGroup<ScalarField = F>,
    J: Jolt<F, G> + ?Sized,
    T: ProofTranscript<G>,
  >(
    _claim: &F,
    num_rounds: usize,
    eq_poly: &mut DensePolynomial<F>,
    memory_polys: &mut Vec<DensePolynomial<F>>,
    flag_polys: &mut Vec<DensePolynomial<F>>,
    degree: usize,
    transcript: &mut T,
  ) -> (Self, Vec<F>, (F, Vec<F>, Vec<F>)) {
    // Check all polys are the same size
    let poly_len = eq_poly.len();
    for index in 0..J::NUM_MEMORIES {
      assert_eq!(memory_polys[index].len(), poly_len);
    }
    for index in 0..J::NUM_INSTRUCTIONS {
      assert_eq!(flag_polys[index].len(), poly_len);
    }

    let mut random_vars: Vec<F> = Vec::with_capacity(num_rounds);
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

    let num_eval_points = degree + 1;
    for _round in 0..num_rounds {
      let mle_len = eq_poly.len();
      let mle_half = mle_len / 2;


      // Store evaluations of each polynomial at all poly_size / 2 points
      let mut eq_evals: Vec<Vec<F>> = vec![Vec::with_capacity(num_eval_points); mle_half];
      let mut multi_flag_evals: Vec<Vec<Vec<F>>> =
        vec![vec![Vec::with_capacity(num_eval_points); mle_half]; J::NUM_INSTRUCTIONS];
      let mut multi_memory_evals: Vec<Vec<Vec<F>>> =
        vec![vec![Vec::with_capacity(num_eval_points); mle_half]; J::NUM_MEMORIES];

      let evaluate_mles_iterator = (0..mle_half).into_iter();

      // Loop over half MLE size (size of MLE next round)
      //   - Compute evaluations of eq, flags, E, at p {0, 1, ..., degree}:
      //       eq(p, _boolean_hypercube_), flags(p, _boolean_hypercube_), E(p, _boolean_hypercube_)
      // After: Sum over MLE elements (with combine)

      for mle_leaf_index in evaluate_mles_iterator {
        // 0
        eq_evals[mle_leaf_index].push(eq_poly[mle_leaf_index]);
        for flag_instruction_index in 0..multi_flag_evals.len() {
          multi_flag_evals[flag_instruction_index][mle_leaf_index]
            .push(flag_polys[flag_instruction_index][mle_leaf_index]);
        }
        for memory_index in 0..multi_memory_evals.len() {
          multi_memory_evals[memory_index][mle_leaf_index]
            .push(memory_polys[memory_index][mle_leaf_index]);
        }

        // 1
        eq_evals[mle_leaf_index].push(eq_poly[mle_half + mle_leaf_index]);
        for flag_instruction_index in 0..multi_flag_evals.len() {
          multi_flag_evals[flag_instruction_index][mle_leaf_index]
            .push(flag_polys[flag_instruction_index][mle_half + mle_leaf_index]);
        }
        for memory_index in 0..multi_memory_evals.len() {
          multi_memory_evals[memory_index][mle_leaf_index]
            .push(memory_polys[memory_index][mle_half + mle_leaf_index]);
        }

        // (2, ...)
        for eval_index in 2..num_eval_points {
          let eq_eval = eq_evals[mle_leaf_index][eval_index - 1]
            + eq_poly[mle_half + mle_leaf_index]
            - eq_poly[mle_leaf_index];
          eq_evals[mle_leaf_index].push(eq_eval);

          for flag_instruction_index in 0..multi_flag_evals.len() {
            let flag_eval = multi_flag_evals[flag_instruction_index][mle_leaf_index]
              [eval_index - 1]
              + flag_polys[flag_instruction_index][mle_half + mle_leaf_index]
              - flag_polys[flag_instruction_index][mle_leaf_index];
            multi_flag_evals[flag_instruction_index][mle_leaf_index].push(flag_eval);
          }
          for memory_index in 0..multi_memory_evals.len() {
            let memory_eval = multi_memory_evals[memory_index][mle_leaf_index][eval_index - 1]
              + memory_polys[memory_index][mle_half + mle_leaf_index]
              - memory_polys[memory_index][mle_leaf_index];
            multi_memory_evals[memory_index][mle_leaf_index].push(memory_eval);
          }
        }
      }

      #[cfg(test)]
      {
        // Compute each of these evaluations the slow way to confirm.

        for flag_index in 0..J::NUM_INSTRUCTIONS {
          for eval_index in 0..num_eval_points {
            for point_index in 0..mle_half {
              // TODO: concat: index_to_field_bitvector::<F>(point_index, mle_half.log_2()) for other sizes
              if mle_len != 4 {
                continue;
              }

              // expected evals: f(p, 0, 0), f(p, 0, 1), f(p, 1, 0), f(p, 1, 1)
              let local_eval = flag_polys[flag_index].evaluate(&vec![
                F::from(eval_index as u64),
                F::from(point_index as u64),
              ]);
              let existing_eval = multi_flag_evals[flag_index][point_index][eval_index];
              assert_eq!(local_eval, existing_eval);
            }
          }
        }

        for memory_index in 0..J::NUM_MEMORIES {
          for eval_index in 0..num_eval_points {
            for point_index in 0..mle_half {
              // TODO: concat: index_to_field_bitvector::<F>(point_index, mle_half.log_2()) for other sizes
              if mle_len != 4 {
                continue;
              }

              // expected evals: f(p, 0, 0), f(p, 0, 1), f(p, 1, 0), f(p, 1, 1)
              let local_eval = memory_polys[memory_index].evaluate(&vec![
                F::from(eval_index as u64),
                F::from(point_index as u64),
              ]);
              let existing_eval = multi_memory_evals[memory_index][point_index][eval_index];
              assert_eq!(local_eval, existing_eval);
            }
          }
        }

        for eval_index in 0..num_eval_points {
          for point_index in 0..mle_half {
            // TODO: concat: index_to_field_bitvector::<F>(point_index, mle_half.log_2()) for other sizes
            if mle_len != 4 {
              continue;
            }

            let local_eval = eq_poly.evaluate(&vec![
              F::from(eval_index as u64),
              F::from(point_index as u64),
            ]);
            let existing_eval = eq_evals[point_index][eval_index];
            assert_eq!(local_eval, existing_eval);
          }
        }
      }

      // Accumulate inner terms.
      // S({0,1,... num_eval_points}) = eq * [ INNER TERMS ] = eq * [ flags_0 * g_0(E_0) + flags_1 * g_1(E_1)]
      let mut evaluations: Vec<F> = Vec::with_capacity(num_eval_points);
      for eval_index in 0..num_eval_points {
        evaluations.push(F::zero());
        for instruction in J::InstructionSet::iter() {
          let instruction_index = instruction.to_opcode() as usize;
          let memory_indices: Vec<usize> = J::instruction_to_memory_indices(&instruction);

          for mle_leaf_index in 0..mle_half {
            let mut terms = Vec::with_capacity(memory_indices.len());
            for memory_index in &memory_indices {
              terms.push(multi_memory_evals[*memory_index][mle_leaf_index][eval_index]);
            }

            let instruction_collation_eval = instruction.combine_lookups(&terms, J::C, J::M);
            let flag_eval = multi_flag_evals[instruction_index][mle_leaf_index][eval_index];

            // TODO(sragss): May have an excessive group mul here.
            evaluations[eval_index] +=
              eq_evals[mle_leaf_index][eval_index] * flag_eval * instruction_collation_eval;
          }
        }
      } // End accumulation

      let round_uni_poly = UniPoly::from_evals(&evaluations);
      compressed_polys.push(round_uni_poly.compress());

      <UniPoly<F> as AppendToTranscript<G>>::append_to_transcript(
        &round_uni_poly,
        b"poly",
        transcript,
      );

      let r_j = transcript.challenge_scalar(b"challenge_nextround");
      random_vars.push(r_j);
      println!("round [{_round}] randomness: {r_j:?}");

      // Bind all polys
      eq_poly.bound_poly_var_top(&r_j);
      for flag_instruction_index in 0..flag_polys.len() {
        flag_polys[flag_instruction_index].bound_poly_var_top(&r_j);
      }
      for memory_index in 0..multi_memory_evals.len() {
        memory_polys[memory_index].bound_poly_var_top(&r_j);
      }
    } // End rounds

    // Pass evaluations at point r back in proof:
    // - eq(r)
    // - flags(r) * NUM_INSTRUCTIONS
    // - E(r) * NUM_SUBTABLES

    // Polys are fully defined so we can just take the first (and only) evaluation
    let eq_eval = eq_poly[0];
    let flag_evals = (0..flag_polys.len()).map(|i| flag_polys[i][0]).collect();
    let memory_evals = (0..memory_polys.len())
      .map(|i| memory_polys[i][0])
      .collect();
    let poly_eval_claims = (eq_eval, flag_evals, memory_evals);

    (
      SumcheckInstanceProof::new(compressed_polys),
      random_vars,
      poly_eval_claims,
    )
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
  pub fn prove_arbitrary<Func, G, T: ProofTranscript<G>>(
    _claim: &F,
    num_rounds: usize,
    polys: &mut Vec<DensePolynomial<F>>,
    comb_func: Func,
    combined_degree: usize,
    transcript: &mut T,
  ) -> (Self, Vec<F>, Vec<F>)
  where
    Func: Fn(&[F]) -> F + std::marker::Sync,
    G: CurveGroup<ScalarField = F>,
  {
    let mut r: Vec<F> = Vec::new();
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::new();

    for _round in 0..num_rounds {
      // Vector storing evaluations of combined polynomials g(x) = P_0(x) * ... P_{num_polys} (x)
      // for points {0, ..., |g(x)|}
      let mut eval_points = vec![F::zero(); combined_degree + 1];

      let mle_half = polys[0].len() / 2;

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
          let params_zero: Vec<F> = polys.iter().map(|poly| poly[poly_term_i]).collect();
          accum[0] += comb_func(&params_zero);

          // TODO(#28): Can be computed from prev_round_claim - eval_point_0
          let params_one: Vec<F> = polys
            .iter()
            .map(|poly| poly[mle_half + poly_term_i])
            .collect();
          accum[1] += comb_func(&params_one);

          // D_n(index, r) = D_{n-1}[half + index] + r * (D_{n-1}[half + index] - D_{n-1}[index])
          // D_n(index, 0) = D_{n-1}
          // D_n(index, 1) = D_{n-1} + (D_{n-1}[HIGH] - D_{n-1}[LOW])
          // D_n(index, 2) = D_{n-1} + (D_{n-1}[HIGH] - D_{n-1}[LOW]) + (D_{n-1}[HIGH] - D_{n-1}[LOW])
          // D_n(index, 3) = D_{n-1} + (D_{n-1}[HIGH] - D_{n-1}[LOW]) + (D_{n-1}[HIGH] - D_{n-1}[LOW]) + (D_{n-1}[HIGH] - D_{n-1}[LOW])
          // ...
          let mut existing_term = params_one;
          for eval_i in 2..(combined_degree + 1) {
            let mut poly_evals = vec![F::zero(); polys.len()];
            for poly_i in 0..polys.len() {
              let poly = &polys[poly_i];
              poly_evals[poly_i] =
                existing_term[poly_i] + poly[mle_half + poly_term_i] - poly[poly_term_i];
            }

            accum[eval_i] += comb_func(&poly_evals);
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
  /// Returns (e, r)1
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
      println!("Sumcheck::verify appending UniPoly {:?}", poly);
      <UniPoly<F> as AppendToTranscript<G>>::append_to_transcript(&poly, b"poly", transcript);

      //derive the verifier's challenge for the next round
      let r_i = transcript.challenge_scalar(b"challenge_nextround");
      println!("Sumcheck::verify new randomness {:?}", r_i);

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
    let mut polys = vec![A.clone(), B.clone(), C.clone()];

    let comb_func_prod =
      |polys: &[Fr]| -> Fr { polys.iter().fold(Fr::one(), |acc, poly| acc * *poly) };

    let r = vec![Fr::from(3), Fr::from(1), Fr::from(3)]; // point 0,0,0 within the boolean hypercube

    let mut transcript: TestTranscript<Fr> = TestTranscript::new(r.clone(), vec![]);
    let (proof, prove_randomness, _final_poly_evals) =
      SumcheckInstanceProof::<Fr>::prove_arbitrary::<_, G1Projective, _>(
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
