#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::field::{JoltField, OptimizedMul};
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{
    BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
};
use crate::poly::spartan_interleaved_poly::{
    SpartanInterleavedPolynomial, SpartanInterleavedPolynomialOracle,
};
use crate::poly::split_eq_poly::{GruenSplitEqPolynomial, SplitEqPolynomial};
use crate::poly::unipoly::{CompressedUniPoly, UniPoly};
use crate::r1cs::builder::Constraint;
use crate::r1cs::spartan::ShiftSumCheckOracle;
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;
use crate::utils::mul_0_optimized;

use crate::utils::small_value::svo_helpers::process_svo_sumcheck_rounds;
use crate::utils::streaming::Oracle;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::{AppendToTranscript, Transcript};
use crate::{into_optimal_iter, optimal_iter};

use crate::jolt::vm::JoltProverPreprocessing;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::sparse_interleaved_poly::SparseCoefficient;
use crate::r1cs::inputs::{R1CSInputsOracle, ALL_R1CS_INPUTS};
use ark_serialize::*;
use rayon::prelude::*;
use smallvec::{smallvec, SmallVec};
use std::marker::PhantomData;
use std::time::Duration;
use tokio::time::Instant;
use tracer::instruction::RV32IMCycle;

pub trait Bindable<F: JoltField>: Sync {
    fn bind(&mut self, r: F);
}

/// Batched cubic sumcheck used in grand products
pub trait BatchedCubicSumcheck<F, ProofTranscript>: Bindable<F>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    fn compute_cubic(&self, eq_poly: &SplitEqPolynomial<F>, previous_round_claim: F) -> UniPoly<F>;
    fn final_claims(&self) -> (F, F);

    #[cfg(test)]
    fn sumcheck_sanity_check(&self, eq_poly: &SplitEqPolynomial<F>, round_claim: F);

    #[tracing::instrument(skip_all, name = "BatchedCubicSumcheck::prove_sumcheck")]
    fn prove_sumcheck(
        &mut self,
        claim: &F,
        eq_poly: &mut SplitEqPolynomial<F>,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, (F, F)) {
        let num_rounds = eq_poly.get_num_vars();

        let mut previous_claim = *claim;
        let mut r: Vec<F> = Vec::new();
        let mut cubic_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        for _ in 0..num_rounds {
            #[cfg(test)]
            self.sumcheck_sanity_check(eq_poly, previous_claim);

            let cubic_poly = self.compute_cubic(eq_poly, previous_claim);
            let compressed_poly = cubic_poly.compress();
            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);
            // derive the verifier's challenge for the next round
            let r_j = transcript.challenge_scalar();

            r.push(r_j);
            // bind polynomials to verifier's challenge
            self.bind(r_j);
            eq_poly.bind(r_j);

            previous_claim = cubic_poly.evaluate(&r_j);
            cubic_polys.push(compressed_poly);
        }

        #[cfg(test)]
        self.sumcheck_sanity_check(eq_poly, previous_claim);

        debug_assert_eq!(eq_poly.len(), 1);

        (
            SumcheckInstanceProof::new(cubic_polys),
            r,
            self.final_claims(),
        )
    }
}

/// Trait for a sumcheck instance that can be batched with other instances.
///
/// This trait defines the interface needed to participate in the `BatchedSumcheck` protocol,
/// which reduces verifier cost and proof size by batching multiple sumcheck protocols.
pub trait BatchableSumcheckInstance<F: JoltField, ProofTranscript: Transcript> {
    /// Returns the maximum degree of the sumcheck polynomial.
    fn degree(&self) -> usize;

    /// Returns the number of rounds/variables in this sumcheck instance.
    fn num_rounds(&self) -> usize;

    /// Returns the initial claim of this sumcheck instance, i.e.
    /// input_claim = \sum_{x \in \{0, 1}^N} P(x)
    fn input_claim(&self) -> F;

    /// Computes the prover's message for a specific round of the sumcheck protocol.
    /// Returns the evaluations of the sumcheck polynomial at 0, 2, 3, ..., degree.
    /// The point evaluation at 1 can be interpolated using the previous round's claim.
    fn compute_prover_message(&self, round: usize) -> Vec<F>;

    /// Binds this sumcheck instance to the verifier's challenge from a specific round.
    /// This updates the internal state to prepare for the next round.
    fn bind(&mut self, r_j: F, round: usize);

    /// Caches polynomial opening claims needed after the sumcheck protocol completes.
    /// These openings will later be proven using either an opening proof or another sumcheck.
    fn cache_openings(&mut self);

    /// Computes the expected output claim given the verifier's challenges.
    /// This is used to verify the final result of the sumcheck protocol.
    fn expected_output_claim(&self, r: &[F]) -> F;
}

/// Implements the standard technique for batching parallel sumchecks to reduce
/// verifier cost and proof size.
///
/// For details, refer to Jim Posen's ["Perspectives on Sumcheck Batching"](https://hackmd.io/s/HyxaupAAA).
/// We do what they describe as "front-loaded" batch sumcheck.
pub enum BatchedSumcheck {}
impl BatchedSumcheck {
    pub fn prove<F: JoltField, ProofTranscript: Transcript>(
        mut sumcheck_instances: Vec<&mut dyn BatchableSumcheckInstance<F, ProofTranscript>>,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>) {
        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        let batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());

        // To see why we may need to scale by a power of two, consider a batch of
        // two sumchecks:
        //   claim_a = \sum_x P(x)             where x \in {0, 1}^M
        //   claim_b = \sum_{x, y} Q(x, y)     where x \in {0, 1}^M, y \in {0, 1}^N
        // Then the batched sumcheck is:
        //   \sum_{x, y} A * P(x) + B * Q(x, y)  where A and B are batching coefficients
        //   = A * \sum_y \sum_x P(x) + B * \sum_{x, y} Q(x, y)
        //   = A * \sum_y claim_a + B * claim_b
        //   = A * 2^N * claim_a + B * claim_b
        let mut individual_claims: Vec<F> = sumcheck_instances
            .iter()
            .map(|sumcheck| {
                let num_rounds = sumcheck.num_rounds();
                sumcheck
                    .input_claim()
                    .mul_u64(1 << (max_num_rounds - num_rounds))
            })
            .collect();

        #[cfg(test)]
        let mut batched_claim: F = individual_claims
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(claim, coeff)| *claim * coeff)
            .sum();

        let mut r: Vec<F> = Vec::with_capacity(max_num_rounds);
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(max_num_rounds);

        for round in 0..max_num_rounds {
            let remaining_rounds = max_num_rounds - round;

            let univariate_polys: Vec<UniPoly<F>> = sumcheck_instances
                .iter()
                .zip(individual_claims.iter())
                .map(|(sumcheck, previous_claim)| {
                    let num_rounds = sumcheck.num_rounds();
                    if remaining_rounds > num_rounds {
                        // We haven't gotten to this sumcheck's variables yet, so
                        // the univariate polynomial is just a constant equal to
                        // the input claim, scaled by a power of 2.
                        let num_rounds = sumcheck.num_rounds();
                        let scaled_input_claim = sumcheck
                            .input_claim()
                            .mul_u64(1 << (remaining_rounds - num_rounds - 1));
                        // Constant polynomial
                        UniPoly::from_coeff(vec![scaled_input_claim])
                    } else {
                        let offset = max_num_rounds - sumcheck.num_rounds();
                        let mut univariate_poly_evals =
                            sumcheck.compute_prover_message(round - offset);
                        univariate_poly_evals.insert(1, *previous_claim - univariate_poly_evals[0]);
                        UniPoly::from_evals(&univariate_poly_evals)
                    }
                })
                .collect();

            // Linear combination of individual univariate polynomials
            let batched_univariate_poly: UniPoly<F> =
                univariate_polys.iter().zip(batching_coeffs.iter()).fold(
                    UniPoly::from_coeff(vec![]),
                    |mut batched_poly, (poly, coeff)| {
                        batched_poly += &(poly * coeff);
                        batched_poly
                    },
                );

            let compressed_poly = batched_univariate_poly.compress();

            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);
            let r_j = transcript.challenge_scalar();
            r.push(r_j);

            // Cache individual claims for this round
            individual_claims
                .iter_mut()
                .zip(univariate_polys.into_iter())
                .for_each(|(claim, poly)| {
                    *claim = poly.evaluate(&r_j);
                });

            #[cfg(test)]
            {
                // Sanity check
                let h0 = batched_univariate_poly.evaluate(&F::zero());
                let h1 = batched_univariate_poly.evaluate(&F::one());
                assert_eq!(
                    h0 + h1,
                    batched_claim,
                    "round {}: H(0) + H(1) = {} + {} != {}",
                    round,
                    h0,
                    h1,
                    batched_claim
                );
                batched_claim = batched_univariate_poly.evaluate(&r_j);
            }

            for sumcheck in sumcheck_instances.iter_mut() {
                // If a sumcheck instance has fewer than `max_num_rounds`,
                // we wait until there are <= `sumcheck.num_rounds()` left
                // before binding its variables.
                if remaining_rounds <= sumcheck.num_rounds() {
                    let offset = max_num_rounds - sumcheck.num_rounds();
                    sumcheck.bind(r_j, round - offset);
                }
            }

            compressed_polys.push(compressed_poly);
        }

        for sumcheck in sumcheck_instances.iter_mut() {
            // Cache polynomial opening claims, to be proven using either an
            // opening proof or sumcheck (in the case of virtual polynomials).
            sumcheck.cache_openings();
        }

        (SumcheckInstanceProof::new(compressed_polys), r)
    }

    pub fn verify<F: JoltField, ProofTranscript: Transcript>(
        proof: &SumcheckInstanceProof<F, ProofTranscript>,
        sumcheck_instances: Vec<&dyn BatchableSumcheckInstance<F, ProofTranscript>>,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F>, ProofVerifyError> {
        let max_degree = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.degree())
            .max()
            .unwrap();
        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        let batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());

        // To see why we may need to scale by a power of two, consider a batch of
        // two sumchecks:
        //   claim_a = \sum_x P(x)             where x \in {0, 1}^M
        //   claim_b = \sum_{x, y} Q(x, y)     where x \in {0, 1}^M, y \in {0, 1}^N
        // Then the batched sumcheck is:
        //   \sum_{x, y} A * P(x) + B * Q(x, y)  where A and B are batching coefficients
        //   = A * \sum_y \sum_x P(x) + B * \sum_{x, y} Q(x, y)
        //   = A * \sum_y claim_a + B * claim_b
        //   = A * 2^N * claim_a + B * claim_b
        let claim: F = sumcheck_instances
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(sumcheck, coeff)| {
                let num_rounds = sumcheck.num_rounds();
                sumcheck
                    .input_claim()
                    .mul_u64(1 << (max_num_rounds - num_rounds))
                    * coeff
            })
            .sum();

        let (output_claim, r_sumcheck) =
            proof.verify(claim, max_num_rounds, max_degree, transcript)?;

        let expected_output_claim = sumcheck_instances
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(sumcheck, coeff)| {
                // If a sumcheck instance has fewer than `max_num_rounds`,
                // we wait until there are <= `sumcheck.num_rounds()` left
                // before binding its variables.
                // So, the sumcheck *actually* uses just the last `sumcheck.num_rounds()`
                // values of `r_sumcheck`.
                let r_slice = &r_sumcheck[max_num_rounds - sumcheck.num_rounds()..];
                sumcheck.expected_output_claim(r_slice) * coeff
            })
            .sum();

        if output_claim != expected_output_claim {
            return Err(ProofVerifyError::InternalError);
        }

        Ok(r_sumcheck)
    }
}

impl<F: JoltField, ProofTranscript: Transcript> SumcheckInstanceProof<F, ProofTranscript> {
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
    #[tracing::instrument(skip_all, name = "prove_arbitrary")]
    pub fn prove_arbitrary<Func>(
        claim: &F,
        num_rounds: usize,
        polys: &mut Vec<MultilinearPolynomial<F>>,
        comb_func: Func,
        combined_degree: usize,
        binding_order: BindingOrder,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, Vec<F>)
    where
        Func: Fn(&[F]) -> F + std::marker::Sync,
    {
        let mut previous_claim = *claim;
        let mut r: Vec<F> = Vec::new();
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        #[cfg(test)]
        {
            for poly in polys.iter() {
                assert_eq!(num_rounds, poly.get_num_vars());
            }
            let total_evals = 1 << num_rounds;
            let mut sum = F::zero();
            for i in 0..total_evals {
                let params: Vec<F> = polys.iter().map(|poly| poly.get_coeff(i)).collect();
                sum += comb_func(&params);
            }
            assert_eq!(&sum, claim, "Sumcheck claim is wrong");
        }

        for _round in 0..num_rounds {
            // Vector storing evaluations of combined polynomials g(x) = P_0(x) * ... P_{num_polys} (x)
            // for points {0, ..., |g(x)|}
            let mut eval_points = vec![F::zero(); combined_degree];

            let mle_half = polys[0].len() / 2;

            let accum: Vec<Vec<F>> = (0..mle_half)
                .into_par_iter()
                .map(|poly_term_i| {
                    let mut accum = vec![F::zero(); combined_degree];
                    // TODO(moodlezoup): Optimize
                    let evals: Vec<_> = polys
                        .iter()
                        .map(|poly| {
                            poly.sumcheck_evals(poly_term_i, combined_degree, binding_order)
                        })
                        .collect();
                    for j in 0..combined_degree {
                        let evals_j: Vec<_> = evals.iter().map(|x| x[j]).collect();
                        accum[j] += comb_func(&evals_j);
                    }

                    accum
                })
                .collect();

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

            eval_points.insert(1, previous_claim - eval_points[0]);
            let univariate_poly = UniPoly::from_evals(&eval_points);
            let compressed_poly = univariate_poly.compress();

            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);
            let r_j = transcript.challenge_scalar();
            r.push(r_j);

            // bound all tables to the verifier's challenge
            polys
                .par_iter_mut()
                .for_each(|poly| poly.bind(r_j, binding_order));
            previous_claim = univariate_poly.evaluate(&r_j);
            compressed_polys.push(compressed_poly);
        }

        let final_evals = polys
            .iter()
            .map(|poly| poly.final_sumcheck_claim())
            .collect();

        (SumcheckInstanceProof::new(compressed_polys), r, final_evals)
    }

    #[tracing::instrument(skip_all, name = "Spartan::prove_spartan_small_value")]
    pub fn prove_spartan_small_value<const NUM_SVO_ROUNDS: usize>(
        num_rounds: usize,
        padded_num_constraints: usize,
        uniform_constraints: &[Constraint],
        flattened_polys: &[MultilinearPolynomial<F>],
        tau: &[F],
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, [F; 3]) {
        let mut r = Vec::new();
        let mut polys = Vec::new();
        let mut claim = F::zero();

        // First, precompute the accumulators and also the `SpartanInterleavedPolynomial`
        let (accums_zero, accums_infty, mut az_bz_cz_poly) =
            SpartanInterleavedPolynomial::<NUM_SVO_ROUNDS, F>::new_with_precompute(
                padded_num_constraints,
                uniform_constraints,
                flattened_polys,
                tau,
            );

        let mut eq_poly = GruenSplitEqPolynomial::new(tau);

        process_svo_sumcheck_rounds::<NUM_SVO_ROUNDS, F, ProofTranscript>(
            &accums_zero,
            &accums_infty,
            &mut r,
            &mut polys,
            &mut claim,
            transcript,
            &mut eq_poly,
        );

        // Round NUM_SVO_ROUNDS : do the streaming sumcheck to compute cached values
        az_bz_cz_poly.streaming_sumcheck_round(
            &mut eq_poly,
            transcript,
            &mut r,
            &mut polys,
            &mut claim,
        );

        let shard_len = 1048576;
        let streaming_rounds_start = NUM_SVO_ROUNDS;
        let binding_round = if num_rounds > shard_len.log_2() + padded_num_constraints.log_2() {
            std::cmp::max(
                streaming_rounds_start,
                num_rounds - shard_len.log_2() - padded_num_constraints.log_2(),
            )
        } else {
            streaming_rounds_start
        };

        // Round (NUM_SVO_ROUNDS + 1)..num_rounds : do the linear time sumcheck
        for _ in NUM_SVO_ROUNDS + 1..num_rounds {
            az_bz_cz_poly.remaining_sumcheck_round(
                &mut eq_poly,
                transcript,
                &mut r,
                &mut polys,
                &mut claim,
            );
        }

        (
            SumcheckInstanceProof::new(polys),
            r,
            az_bz_cz_poly.final_sumcheck_evals(),
        )
    }

    #[tracing::instrument(skip_all, name = "prove_spartan_small_value_streaming")]
    pub fn prove_spartan_small_value_streaming<'a, const NUM_SVO_ROUNDS: usize, PCS>(
        num_rounds: usize,
        padded_num_constraints: usize,
        uniform_constraints: &[Constraint],
        trace: &[RV32IMCycle],
        preprocessing: &'a JoltProverPreprocessing<F, PCS, ProofTranscript>,
        shard_len: usize,
        tau: &[F],
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, [F; 3])
    where
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
    {
        let mut r = Vec::new();
        let mut polys = Vec::new();
        let mut claim = F::zero();

        let input_polys_oracle = R1CSInputsOracle::new(shard_len, trace, preprocessing);

        let mut az_bz_poly_oracle = SpartanInterleavedPolynomialOracle::new(
            padded_num_constraints,
            uniform_constraints,
            input_polys_oracle,
        );

        let (accums_zero, accums_infty) = az_bz_poly_oracle.compute_accumulators(
            padded_num_constraints,
            uniform_constraints,
            tau,
            shard_len,
        );

        let mut eq_poly = GruenSplitEqPolynomial::new(tau);
        process_svo_sumcheck_rounds::<NUM_SVO_ROUNDS, F, ProofTranscript>(
            &accums_zero,
            &accums_infty,
            &mut r,
            &mut polys,
            &mut claim,
            transcript,
            &mut eq_poly,
        );

        let streaming_rounds_start = NUM_SVO_ROUNDS;
        let binding_round = if num_rounds > shard_len.log_2() + padded_num_constraints.log_2() {
            std::cmp::max(
                streaming_rounds_start,
                num_rounds - shard_len.log_2() - padded_num_constraints.log_2(),
            )
        } else {
            streaming_rounds_start
        };

        let mut r_rev = r.clone();
        r_rev.reverse();
        let mut eq_r_evals = EqPolynomial::evals(&r_rev);
        let num_shards = az_bz_poly_oracle.get_len() / shard_len;
        az_bz_poly_oracle.streaming_rounds(
            num_shards,
            streaming_rounds_start,
            binding_round,
            &mut eq_poly,
            &mut r,
            &mut eq_r_evals,
            &mut polys,
            &mut claim,
            transcript,
        );

        for i in binding_round + 1..num_rounds {
            az_bz_poly_oracle.remaining_sumcheck_rounds(
                &mut eq_poly,
                transcript,
                &mut r,
                &mut polys,
                &mut claim,
            );
        }

        (
            SumcheckInstanceProof::new(polys),
            r,
            az_bz_poly_oracle.final_sumcheck_evals(),
        )
    }

    #[tracing::instrument(skip_all)]
    // A specialized sumcheck implementation with the 0th round unrolled from the rest of the
    // `for` loop. This allows us to pass in `witness_polynomials` by reference instead of
    // passing them in as a single `DensePolynomial`, which would require an expensive
    // concatenation. We defer the actual instantiation of a `DensePolynomial` to the end of the
    // 0th round.
    pub fn prove_spartan_quadratic(
        claim: &F,
        num_rounds: usize,
        poly_A: &mut DensePolynomial<F>,
        witness_polynomials: &[&MultilinearPolynomial<F>],
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, Vec<F>) {
        let mut r: Vec<F> = Vec::with_capacity(num_rounds);
        let mut polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
        let mut claim_per_round = *claim;

        /*          Round 0 START         */

        let len = poly_A.len() / 2;
        let trace_len = witness_polynomials[0].len();
        // witness_polynomials
        //     .iter()
        //     .for_each(|poly| debug_assert_eq!(poly.len(), trace_len));

        // We don't materialize the full, flattened witness vector, but this closure
        // simulates it
        let witness_value = |index: usize| {
            if index / trace_len >= witness_polynomials.len() {
                F::zero()
            } else {
                witness_polynomials[index / trace_len].get_coeff(index % trace_len)
            }
        };

        let poly = {
            // eval_point_0 = \sum_i A[i] * B[i]
            // where B[i] = witness_value(i) for i in 0..len
            let eval_point_0: F = (0..len)
                .into_par_iter()
                .map(|i| {
                    if poly_A[i].is_zero() || witness_value(i).is_zero() {
                        F::zero()
                    } else {
                        poly_A[i] * witness_value(i)
                    }
                })
                .sum();
            // eval_point_2 = \sum_i (2 * A[len + i] - A[i]) * (2 * B[len + i] - B[i])
            // where B[i] = witness_value(i) for i in 0..len, B[len] = 1, and B[i] = 0 for i > len
            let mut eval_point_2: F = (1..len)
                .into_par_iter()
                .map(|i| {
                    if witness_value(i).is_zero() {
                        F::zero()
                    } else {
                        let poly_A_bound_point = poly_A[len + i] + poly_A[len + i] - poly_A[i];
                        let poly_B_bound_point = -witness_value(i);
                        mul_0_optimized(&poly_A_bound_point, &poly_B_bound_point)
                    }
                })
                .sum();
            eval_point_2 += mul_0_optimized(
                &(poly_A[len] + poly_A[len] - poly_A[0]),
                &(F::from_u8(2) - witness_value(0)),
            );

            let evals = [eval_point_0, claim_per_round - eval_point_0, eval_point_2];
            UniPoly::from_evals(&evals)
        };

        let compressed_poly = poly.compress();
        // append the prover's message to the transcript
        compressed_poly.append_to_transcript(transcript);

        //derive the verifier's challenge for the next round
        let r_i: F = transcript.challenge_scalar();
        r.push(r_i);
        polys.push(compressed_poly);

        // Set up next round
        claim_per_round = poly.evaluate(&r_i);

        // bound all tables to the verifier's challenge
        let (_, mut poly_B) = rayon::join(
            || poly_A.bound_poly_var_top_zero_optimized(&r_i),
            || {
                // Simulates `poly_B.bound_poly_var_top(&r_i)` by
                // iterating over `witness_polynomials`
                // We need to do this because we don't actually have
                // a `DensePolynomial` instance for `poly_B` yet.
                let zero = F::zero();
                let one = [F::one()];
                let W_iter = (0..len).into_par_iter().map(witness_value);
                let Z_iter = W_iter
                    .chain(one.into_par_iter())
                    .chain(rayon::iter::repeatn(zero, len));
                let left_iter = Z_iter.clone().take(len);
                let right_iter = Z_iter.skip(len).take(len);
                let B = left_iter
                    .zip(right_iter)
                    .map(|(a, b)| if a == b { a } else { a + r_i * (b - a) })
                    .collect();
                DensePolynomial::new(B)
            },
        );

        /*          Round 0 END          */

        for _i in 1..num_rounds {
            let poly = {
                let (eval_point_0, eval_point_2) =
                    Self::compute_eval_points_spartan_quadratic(poly_A, &poly_B);

                let evals = [eval_point_0, claim_per_round - eval_point_0, eval_point_2];
                UniPoly::from_evals(&evals)
            };

            let compressed_poly = poly.compress();
            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);

            //derive the verifier's challenge for the next round
            let r_i: F = transcript.challenge_scalar();

            r.push(r_i);
            polys.push(compressed_poly);

            // Set up next round
            claim_per_round = poly.evaluate(&r_i);

            // bound all tables to the verifier's challenge
            rayon::join(
                || poly_A.bound_poly_var_top_zero_optimized(&r_i),
                || poly_B.bound_poly_var_top_zero_optimized(&r_i),
            );
        }

        let evals = vec![poly_A[0], poly_B[0]];
        drop_in_background_thread(poly_B);

        (SumcheckInstanceProof::new(polys), r, evals)
    }

    #[inline]
    #[tracing::instrument(skip_all, name = "Sumcheck::compute_eval_points_spartan_quadratic")]
    pub fn compute_eval_points_spartan_quadratic(
        poly_A: &DensePolynomial<F>,
        poly_B: &DensePolynomial<F>,
    ) -> (F, F) {
        let len = poly_A.len() / 2;
        (0..len)
            .into_par_iter()
            .map(|i| {
                // eval 0: bound_func is A(low)
                let eval_point_0 = if poly_B[i].is_zero() || poly_A[i].is_zero() {
                    F::zero()
                } else {
                    poly_A[i] * poly_B[i]
                };

                // eval 2: bound_func is -A(low) + 2*A(high)
                let poly_B_bound_point = poly_B[len + i] + poly_B[len + i] - poly_B[i];
                let eval_point_2 = if poly_B_bound_point.is_zero() {
                    F::zero()
                } else {
                    let poly_A_bound_point = poly_A[len + i] + poly_A[len + i] - poly_A[i];
                    mul_0_optimized(&poly_A_bound_point, &poly_B_bound_point)
                };

                (eval_point_0, eval_point_2)
            })
            .reduce(|| (F::zero(), F::zero()), |a, b| (a.0 + b.0, a.1 + b.1))
    }

    #[tracing::instrument(skip_all, name = "shift_sumcheck")]
    pub fn shift_sumcheck<Func, PCS>(
        num_rounds: usize,
        stream_poly: &mut ShiftSumCheckOracle<F, PCS, ProofTranscript>,
        comb_func: Func,
        shard_length: usize,
        transcript: &mut ProofTranscript,
        degree: usize,
    ) -> (Self, Vec<F>, Vec<F>)
    where
        Func: Fn(&[F]) -> F + Sync,
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
    {
        let mut r: Vec<F> = Vec::with_capacity(num_rounds);
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

        let log2_shard = shard_length.log_2();
        let split_at = if (1 << num_rounds) == shard_length * shard_length {
            num_rounds - log2_shard
        } else {
            num_rounds - log2_shard + 1
        };

        let mut evals_1 = vec![F::zero(); 1 << split_at];
        let mut evals_2 = vec![F::zero(); 1 << (num_rounds - split_at)];
        evals_1[0] = F::one();
        evals_2[0] = F::one();

        let num_shards = (1 << num_rounds) / shard_length;

        let update_evals = |evals: &mut Vec<F>, r: F, size: usize| {
            for i in 0..size {
                let temp = evals[i];
                evals[size + i] = temp * r;
                evals[i] = temp - evals[size + i];
            }
        };

        let mut eq_eval_idx_s_vec: Vec<SmallVec<[F; 2]>> = vec![smallvec![F::one(); 2]; degree + 1];

        for s in 0..=degree {
            let val = F::from_u64(s as u64);
            eq_eval_idx_s_vec[s][0] = F::one() - val;
            eq_eval_idx_s_vec[s][1] = val;
        }

        let mut polys = stream_poly.next_shard();
        let num_polys = polys.len();
        for round in 0..split_at {
            let mask = (1 << round) - 1;
            let mut accumulator = vec![F::zero(); degree + 1];
            for shard_idx in 0..num_shards {
                let base_poly_idx = shard_length * shard_idx;
                (_, polys) = rayon::join(
                    || {
                        let chunk_size = 1 << (round + 1);
                        let no_of_chunks = shard_length / chunk_size;
                        let acc = (0..no_of_chunks)
                            .into_par_iter()
                            .fold(
                                || vec![F::zero(); degree + 1],
                                |mut acc, chunk_iter| {
                                    let base_chunk_idx = chunk_iter * chunk_size;
                                    let witness_eval = (0..chunk_size).fold(
                                        (|| vec![vec![F::zero(); degree + 1]; num_polys])(),
                                        |mut acc, idx_in_chunk| {
                                            let idx_in_shard = base_chunk_idx + idx_in_chunk;
                                            let idx_in_poly = base_poly_idx + idx_in_shard;
                                            let bit = (idx_in_poly >> round) & 1;
                                            let eval_1 = evals_1[idx_in_poly & mask];
                                            for i in 0..num_polys {
                                                for j in 0..=degree {
                                                    acc[i][j] += polys[i].get_coeff(idx_in_shard)
                                                        * eval_1.mul_01_optimized(
                                                            eq_eval_idx_s_vec[j][bit],
                                                        );
                                                }
                                            }
                                            acc
                                        },
                                    );
                                    acc.iter_mut().enumerate().for_each(|(deg_iter, acc)| {
                                        *acc += comb_func(
                                            &(0..num_polys)
                                                .map(|poly_iter| witness_eval[poly_iter][deg_iter])
                                                .collect::<Vec<F>>(),
                                        )
                                    });
                                    acc
                                },
                            )
                            .reduce(
                                || vec![F::zero(); degree + 1],
                                |mut acc, evals| {
                                    acc.iter_mut()
                                        .zip(evals.iter())
                                        .for_each(|(acc, eval)| *acc += *eval);
                                    acc
                                },
                            );
                        accumulator
                            .iter_mut()
                            .zip(acc.iter())
                            .for_each(|(acc, eval)| *acc += *eval);
                    },
                    || stream_poly.next_shard(),
                );
            }

            let univariate_poly = UniPoly::from_evals(&accumulator);
            let compressed_poly = univariate_poly.compress();
            compressed_poly.append_to_transcript(transcript);

            let r_i = transcript.challenge_scalar();
            r.push(r_i);
            compressed_polys.push(compressed_poly);

            update_evals(&mut evals_1, r_i, 1 << round);
        }

        //Bind Polynomials
        let chunk_size = evals_1.len();
        assert!(
            shard_length >= chunk_size,
            "shard length {} must be greater than equal to chunk size {}",
            shard_length,
            chunk_size
        );

        let no_of_chunks = shard_length / chunk_size;
        let mut bind_shards = vec![Vec::with_capacity(num_shards * no_of_chunks); num_polys];
        for shard_idx in 0..num_shards {
            (_, polys) = rayon::join(
                || {
                    (0..no_of_chunks).for_each(|chunk_iter| {
                        let start_idx = chunk_iter * chunk_size;
                        let end_idx = start_idx + chunk_size;
                        let dot_products = into_optimal_iter!(start_idx..end_idx)
                            .zip(optimal_iter!(evals_1))
                            .fold(
                                || vec![F::zero(); num_polys],
                                |acc, (coeff_at, eval)| {
                                    (0..num_polys)
                                        .map(|idx| acc[idx] + polys[idx].get_coeff(coeff_at) * eval)
                                        .collect::<Vec<F>>()
                                },
                            )
                            .reduce(
                                || vec![F::zero(); num_polys],
                                |acc, sum| {
                                    (0..num_polys)
                                        .map(|idx| acc[idx] + sum[idx])
                                        .collect::<Vec<F>>()
                                },
                            );
                        bind_shards
                            .iter_mut()
                            .enumerate()
                            .for_each(|(idx, poly)| poly.push(dot_products[idx]));
                    });
                },
                || {
                    let mut polys = Vec::with_capacity(num_polys);
                    if shard_idx != num_shards - 1 {
                        polys = stream_poly.next_shard();
                    }
                    polys
                },
            );
        }

        let mut bind_polys = bind_shards
            .into_par_iter()
            .map(|poly| MultilinearPolynomial::from(poly))
            .collect::<Vec<MultilinearPolynomial<F>>>();

        let second_half_claim = (0..1 << (num_rounds - split_at))
            .into_par_iter()
            .map(|iter| {
                let params: Vec<F> = bind_polys.iter().map(|poly| poly.get_coeff(iter)).collect();
                comb_func(&params)
            })
            .reduce(|| F::zero(), |acc, eval| acc + eval);

        let (mut second_half_proof, mut second_half_r, shift_sumcheck_claims) =
            SumcheckInstanceProof::prove_arbitrary(
                &second_half_claim,
                num_rounds - split_at,
                &mut bind_polys,
                comb_func,
                2,
                BindingOrder::LowToHigh,
                transcript,
            );
        compressed_polys.append(&mut second_half_proof.compressed_polys);
        r.append(&mut second_half_r);

        (
            SumcheckInstanceProof::new(compressed_polys),
            r,
            shift_sumcheck_claims,
        )
    }
}
#[inline]
pub fn eq_plus_one_shards<F: JoltField>(
    step_shard: usize,
    shard_length: usize,
    eq_rx_step: &SplitEqPolynomial<F>,
    num_x1_bits: usize,
    x1_bitmask: usize,
) -> MultilinearPolynomial<F> {
    if step_shard == 0 {
        let mut evals: Vec<F> = (0..shard_length - 1)
            .map(|idx| {
                let poly_idx = step_shard + idx;
                let x1 = poly_idx & x1_bitmask;
                let x2 = poly_idx >> num_x1_bits;
                eq_rx_step.E1[x1] * eq_rx_step.E2[x2]
            })
            .collect();
        evals.insert(0, F::zero());
        MultilinearPolynomial::from(evals)
    } else {
        let evals = (0..shard_length)
            .map(|idx| {
                let poly_idx = step_shard + idx - 1;
                let x1 = poly_idx & x1_bitmask;
                let x2 = poly_idx >> num_x1_bits;
                eq_rx_step.E1[x1] * eq_rx_step.E2[x2]
            })
            .collect::<Vec<F>>();
        MultilinearPolynomial::from(evals)
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct SumcheckInstanceProof<F: JoltField, ProofTranscript: Transcript> {
    pub compressed_polys: Vec<CompressedUniPoly<F>>,
    _marker: PhantomData<ProofTranscript>,
}

impl<F: JoltField, ProofTranscript: Transcript> SumcheckInstanceProof<F, ProofTranscript> {
    pub fn new(
        compressed_polys: Vec<CompressedUniPoly<F>>,
    ) -> SumcheckInstanceProof<F, ProofTranscript> {
        SumcheckInstanceProof {
            compressed_polys,
            _marker: PhantomData,
        }
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
    pub fn verify(
        &self,
        claim: F,
        num_rounds: usize,
        degree_bound: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<(F, Vec<F>), ProofVerifyError> {
        let mut e = claim;
        let mut r: Vec<F> = Vec::new();

        // verify that there is a univariate polynomial for each round
        assert_eq!(self.compressed_polys.len(), num_rounds);
        for i in 0..self.compressed_polys.len() {
            // verify degree bound
            if self.compressed_polys[i].degree() != degree_bound {
                return Err(ProofVerifyError::InvalidInputLength(
                    degree_bound,
                    self.compressed_polys[i].degree(),
                ));
            }

            // append the prover's message to the transcript
            self.compressed_polys[i].append_to_transcript(transcript);

            //derive the verifier's challenge for the next round
            let r_i = transcript.challenge_scalar();
            r.push(r_i);

            // evaluate the claimed degree-ell polynomial at r_i using the hint
            e = self.compressed_polys[i].eval_from_hint(&e, &r_i);
        }

        Ok((e, r))
    }
}

/// Helper function to encapsulate the common subroutine for sumcheck with eq poly factor:
/// - Compute the linear factor E_i(X) from the current eq-poly
/// - Reconstruct the cubic polynomial s_i(X) = E_i(X) * t_i(X) for the i-th round
/// - Compress the cubic polynomial
/// - Append the compressed polynomial to the transcript
/// - Derive the challenge for the next round
/// - Bind the cubic polynomial to the challenge
/// - Update the claim as the evaluation of the cubic polynomial at the challenge
///
/// Returns the derived challenge
#[inline]
pub fn process_eq_sumcheck_round<F: JoltField, ProofTranscript: Transcript>(
    quadratic_evals: (F, F), // (t_i(0), t_i(infty))
    eq_poly: &mut GruenSplitEqPolynomial<F>,
    polys: &mut Vec<CompressedUniPoly<F>>,
    r: &mut Vec<F>,
    claim: &mut F,
    transcript: &mut ProofTranscript,
) -> F {
    let scalar_times_w_i = eq_poly.current_scalar * eq_poly.w[eq_poly.current_index - 1];

    let cubic_poly = UniPoly::from_linear_times_quadratic_with_hint(
        // The coefficients of `eq(w[(n - i)..], r[..i]) * eq(w[n - i - 1], X)`
        [
            eq_poly.current_scalar - scalar_times_w_i,
            scalar_times_w_i + scalar_times_w_i - eq_poly.current_scalar,
        ],
        quadratic_evals.0,
        quadratic_evals.1,
        *claim,
    );

    // Compress and add to transcript
    let compressed_poly = cubic_poly.compress();
    compressed_poly.append_to_transcript(transcript);

    // Derive challenge
    let r_i: F = transcript.challenge_scalar();
    r.push(r_i);
    polys.push(compressed_poly);

    // Evaluate for next round's claim
    *claim = cubic_poly.evaluate(&r_i);

    // Bind eq_poly for next round
    eq_poly.bind(r_i);

    r_i
}
