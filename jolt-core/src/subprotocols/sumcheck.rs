#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use std::marker::PhantomData;

use ark_serialize::*;
use num::zero;
use rayon::prelude::*;

use crate::field::JoltField;
use crate::jolt::instruction::JoltInstructionSet;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::{EqPolynomial, StreamingEqPolynomial};
use crate::poly::multilinear_polynomial::{
    BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
};
use crate::poly::spartan_interleaved_poly::SpartanInterleavedPolynomial;
use crate::poly::split_eq_poly::SplitEqPolynomial;
use crate::poly::unipoly::{CompressedUniPoly, UniPoly};
use crate::r1cs::spartan::{AzBzCz, BindZRyVarOracle};
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;
use crate::utils::mul_0_optimized;
use crate::utils::streaming::Oracle;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::{AppendToTranscript, Transcript};

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
    #[tracing::instrument(skip_all, name = "Sumcheck.prove")]
    pub fn prove_arbitrary<Func>(
        claim: &F,
        num_rounds: usize,
        polys: &mut Vec<MultilinearPolynomial<F>>,
        comb_func: Func,
        combined_degree: usize,
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
            let total_evals = 1 << num_rounds;
            let mut sum = F::zero();
            for i in 0..total_evals {
                let params: Vec<F> = polys.iter().map(|poly| poly.get_coeff(i)).collect();
                sum += comb_func(&params);
            }
            // println!("Claim: {}, Sum: {}", claim, sum);
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
                            poly.sumcheck_evals(
                                poly_term_i,
                                combined_degree,
                                BindingOrder::HighToLow,
                            )
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
                .for_each(|poly| poly.bind(r_j, BindingOrder::HighToLow));
            previous_claim = univariate_poly.evaluate(&r_j);
            compressed_polys.push(compressed_poly);
        }

        let final_evals = polys
            .iter()
            .map(|poly| poly.final_sumcheck_claim())
            .collect();

        (SumcheckInstanceProof::new(compressed_polys), r, final_evals)
    }

    #[tracing::instrument(skip_all, name = "Spartan2::sumcheck::prove_spartan_cubic")]
    pub fn prove_spartan_cubic(
        num_rounds: usize,
        eq_poly: &mut SplitEqPolynomial<F>,
        az_bz_cz_poly: &mut SpartanInterleavedPolynomial<F>,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, [F; 3]) {
        let mut r: Vec<F> = Vec::new();
        let mut polys: Vec<CompressedUniPoly<F>> = Vec::new();
        let mut claim = F::zero();

        for round in 0..num_rounds {
            if round == 0 {
                az_bz_cz_poly
                    .first_sumcheck_round(eq_poly, transcript, &mut r, &mut polys, &mut claim);
            } else {
                az_bz_cz_poly
                    .subsequent_sumcheck_round(eq_poly, transcript, &mut r, &mut polys, &mut claim);
            }
        }
        // println!("Non streaming prover complressed polys = {:?}", polys);
        (
            SumcheckInstanceProof::new(polys),
            r,
            az_bz_cz_poly.final_sumcheck_evals(),
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

    pub fn stream_prove_cubic<O>(
        num_shards: usize,
        num_rounds: usize,
        interleaved_az_bz_cz: &mut O,
        shard_length: usize,
        padded_num_constraints: usize,
        tau: Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, Vec<F>)
    where
        O: Oracle<Item = AzBzCz>,
    {
        let mut eq_tau = SplitEqPolynomial::new(&tau);
        let mut r: Vec<F> = Vec::new();
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::new();
        let mut uni_polys: Vec<UniPoly<F>> = Vec::new();
        let mut witness_eval_for_final_eval = vec![vec![F::zero(); 2]; 4];
        let mut final_eval = vec![F::zero(); 3];
        let eq_shard_len = shard_length * padded_num_constraints;

        for round in 0..num_rounds {
            let mut accumulator = vec![F::zero(); 4];
            if eq_tau.E1_len == 1 {
                let eq_evals: Vec<[F; 4]> = eq_tau.E2[..eq_tau.E2_len]
                    .par_chunks(2)
                    .map(|eq_chunk| {
                        let eval_point_0 = eq_chunk[0];
                        let m_eq = eq_chunk[1] - eq_chunk[0];
                        let eval_point_2 = eq_chunk[1] + m_eq;
                        let eval_point_3 = eval_point_2 + m_eq;
                        [eval_point_0, eq_chunk[1], eval_point_2, eval_point_3]
                    })
                    .collect();
                let mut current_window_ub = 1 << (round + 1);

                let mut witness_eval = vec![vec![F::zero(); 4]; 4];
                let mut eq_poly = StreamingEqPolynomial::new(r.clone(), num_rounds, None, true);

                for _ in 0..num_shards {
                    let az_bz_cz_shard = interleaved_az_bz_cz
                        .next_shard(shard_length)
                        .interleaved_az_bz_cz;
                    let blocks = az_bz_cz_shard.chunk_by(|a, b| a.0 / 3 == b.0 / 3);

                    let mut prev_idx = 0;
                    let eq_shard = eq_poly.next_shard(eq_shard_len);
                    for block in blocks {
                        let idx = block[0].0 / 3;
                        if idx >= current_window_ub {
                            witness_eval[0] = eq_evals[prev_idx / (1 << (round + 1))].to_vec();
                            for s in 0..=3 {
                                let eval = witness_eval[0][s]
                                    * (witness_eval[1][s] * witness_eval[2][s]
                                        - witness_eval[3][s]);
                                accumulator[s] += eval;
                            }
                            witness_eval = vec![vec![F::zero(); 4]; 4];
                            current_window_ub =
                                idx - (idx % (1 << (round + 1))) + (1 << (round + 1));
                        }

                        let mut az_eval = 0;
                        let mut bz_eval = 0;
                        let mut cz_eval = 0;

                        for b in block {
                            if b.0 % 3 == 0 {
                                az_eval = b.1;
                            }
                            if b.0 % 3 == 1 {
                                bz_eval = b.1;
                            }
                            if b.0 % 3 == 2 {
                                cz_eval = b.1;
                            }
                        }

                        let bit = (idx >> round) & 1;

                        for s in 0..=3 {
                            let val = F::from_u64(s as u64);
                            let eq_eval_idx_s = if bit == 0 { F::one() - val } else { val };
                            witness_eval[1][s] += eq_shard[idx % eq_shard_len]
                                * eq_eval_idx_s
                                * F::from_i128(az_eval);
                            witness_eval[2][s] += eq_shard[idx % eq_shard_len]
                                * eq_eval_idx_s
                                * F::from_i128(bz_eval);
                            witness_eval[3][s] += eq_shard[idx % eq_shard_len]
                                * eq_eval_idx_s
                                * F::from_i128(cz_eval);
                        }
                        prev_idx = idx
                    }

                    let idx = interleaved_az_bz_cz.get_step() * padded_num_constraints;
                    if round == num_rounds - 1 && idx == (1 << num_rounds) {
                        for k in 0..=3 {
                            witness_eval_for_final_eval[k][0] = witness_eval[k][0];
                            witness_eval_for_final_eval[k][1] = witness_eval[k][1];
                        }
                    }

                    if idx >= current_window_ub {
                        witness_eval[0] =
                            eq_evals[(current_window_ub - 1) / (1 << (round + 1))].to_vec();
                        for s in 0..=3 {
                            let eval = witness_eval[0][s]
                                * (witness_eval[1][s] * witness_eval[2][s] - witness_eval[3][s]);
                            accumulator[s] += eval;
                        }
                        witness_eval = vec![vec![F::zero(); 4]; 4];
                        current_window_ub = idx - (idx % (1 << (round + 1))) + (1 << (round + 1));
                    }
                }
            } else {
                let num_x1_bits = eq_tau.E1_len.log_2();
                let x1_bitmask = (1 << (num_x1_bits + round)) - 1;
                let E1_evals: Vec<_> = eq_tau.E1[..eq_tau.E1_len]
                    .par_chunks(2)
                    .map(|E1_chunk| {
                        let eval_point_0 = E1_chunk[0];
                        let m_eq = E1_chunk[1] - E1_chunk[0];
                        let eval_point_2 = E1_chunk[1] + m_eq;
                        let eval_point_3 = eval_point_2 + m_eq;
                        [eval_point_0, E1_chunk[1], eval_point_2, eval_point_3]
                    })
                    .collect();
                let mut current_window_ub = 1 << (round + 1);

                let mut witness_eval = vec![vec![F::zero(); 4]; 4];
                let mut eq_poly = StreamingEqPolynomial::new(r.clone(), num_rounds, None, true);

                for _ in 0..num_shards {
                    let az_bz_cz_shard = interleaved_az_bz_cz
                        .next_shard(shard_length)
                        .interleaved_az_bz_cz;
                    let blocks = az_bz_cz_shard.chunk_by(|a, b| a.0 / 3 == b.0 / 3);
                    let mut prev_idx = 0;
                    let eq_shard = eq_poly.next_shard(eq_shard_len);
                    for block in blocks {
                        let idx = block[0].0 / 3;
                        if idx >= current_window_ub {
                            let x1 = (prev_idx & x1_bitmask) >> (round + 1);
                            let x2 = (prev_idx >> (num_x1_bits + round));
                            for s in 0..=3 {
                                witness_eval[0][s] = (E1_evals[x1][s] * eq_tau.E2[x2]);
                                let eval = witness_eval[0][s]
                                    * (witness_eval[1][s] * witness_eval[2][s]
                                        - witness_eval[3][s]);
                                accumulator[s] += eval;
                            }
                            witness_eval = vec![vec![F::zero(); 4]; 4];
                            current_window_ub =
                                idx - (idx % (1 << (round + 1))) + (1 << (round + 1));
                        }

                        let mut az_eval = 0;
                        let mut bz_eval = 0;
                        let mut cz_eval = 0;

                        for b in block {
                            if b.0 % 3 == 0 {
                                az_eval = b.1;
                            }
                            if b.0 % 3 == 1 {
                                bz_eval = b.1;
                            }
                            if b.0 % 3 == 2 {
                                cz_eval = b.1;
                            }
                        }

                        let bit = (idx >> round) & 1;

                        for s in 0..=3 {
                            let val = F::from_u64(s as u64);
                            let eq_eval_idx_s = if bit == 0 { F::one() - val } else { val };
                            witness_eval[1][s] += eq_shard[idx % eq_shard_len]
                                * eq_eval_idx_s
                                * F::from_i128(az_eval);
                            witness_eval[2][s] += eq_shard[idx % eq_shard_len]
                                * eq_eval_idx_s
                                * F::from_i128(bz_eval);
                            witness_eval[3][s] += eq_shard[idx % eq_shard_len]
                                * eq_eval_idx_s
                                * F::from_i128(cz_eval);
                        }
                        prev_idx = idx
                    }

                    let idx = interleaved_az_bz_cz.get_step() * padded_num_constraints;
                    if round == num_rounds - 1 && idx == (1 << num_rounds) {
                        for k in 0..=3 {
                            witness_eval_for_final_eval[k][0] = witness_eval[k][0];
                            witness_eval_for_final_eval[k][1] = witness_eval[k][1];
                        }
                    }

                    if idx >= current_window_ub {
                        for s in 0..=3 {
                            let x1 = ((current_window_ub - 1) & x1_bitmask) >> (round + 1);
                            let x2 = ((current_window_ub - 1) >> (num_x1_bits + round));

                            witness_eval[0][s] = (E1_evals[x1][s] * eq_tau.E2[x2]);
                            let eval = witness_eval[0][s]
                                * (witness_eval[1][s] * witness_eval[2][s] - witness_eval[3][s]);
                            accumulator[s] += eval;
                        }
                        witness_eval = vec![vec![F::zero(); 4]; 4];
                        current_window_ub = idx - (idx % (1 << (round + 1))) + (1 << (round + 1));
                    }
                }
            }
            let univariate_poly = UniPoly::from_evals(&accumulator);
            let compressed_poly = univariate_poly.compress();
            compressed_poly.append_to_transcript(transcript);

            let r_i = transcript.challenge_scalar();
            r.push(r_i);
            compressed_polys.push(compressed_poly);
            uni_polys.push(univariate_poly);

            // Bind polynomials
            eq_tau.bind(r_i);

            interleaved_az_bz_cz.reset();
        }

        #[cfg(test)]
        {
            let mut e = F::zero();
            for i in 0..uni_polys.len() {
                // check if G_k(0) + G_k(1) = e
                assert_eq!(
                    uni_polys[i].eval_at_zero() + uni_polys[i].eval_at_one(),
                    e,
                    "failed at round {i}"
                );
                // evaluate the claimed degree-ell polynomial at r_i using the hint
                e = uni_polys[i].evaluate(&r[i]);
            }
        }

        final_eval = (1..=3)
            .map(|i| {
                (F::one() - r[num_rounds - 1]) * witness_eval_for_final_eval[i][0]
                    + r[num_rounds - 1] * witness_eval_for_final_eval[i][1]
            })
            .collect();

        (SumcheckInstanceProof::new(compressed_polys), r, final_eval)
    }

    pub fn stream_prove_arbitrary<'a, Func1, Func2, I: JoltInstructionSet>(
        num_rounds: usize,
        stream_polys: &mut Stream<F, I>,
        extract_poly_fn: Func1,
        comb_fn: Func2,
        degree: usize,
        shard_length: usize,
        num_polys: usize,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, Vec<F>)
    where
        Func1: Fn(&OracleItem<F>) -> Vec<MultilinearPolynomial<F>> + std::marker::Sync,
        Func2: Fn(&[F]) -> F + std::marker::Sync,
    {
        let mut r: Vec<F> = Vec::new();

        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::new();
        let mut final_eval = vec![F::zero(); num_polys];

        let mut witness_eval_for_final_eval = vec![vec![F::zero(); 2]; num_polys];
        let num_shards = (1 << num_rounds) / shard_length;
        for i in 0..num_rounds {
            let mut accumulator = vec![F::zero(); degree + 1];

            let mut witness_eval = vec![vec![F::zero(); degree + 1]; num_polys];
            let mut eq_poly = StreamingEqPolynomial::new(r.clone(), num_rounds, None, true);
            for shard in 0..num_shards {
                let shards = stream_polys.next_shard(shard_length);

                let polys = extract_poly_fn(&shards);
                let eq_shard = eq_poly.next_shard(shard_length);
                for j in 0..shard_length {
                    let idx = shard_length * shard + j;

                    let mut eq_eval_idx_s_vec = vec![F::one(); degree + 1];

                    let bit = (idx >> i) & 1;

                    for s in 0..=degree {
                        let val = F::from_u64(s as u64);
                        eq_eval_idx_s_vec[s] = if bit == 0 { F::one() - val } else { val };
                    }

                    for k in 0..num_polys {
                        for s in 0..=degree {
                            witness_eval[k][s] +=
                                eq_shard[j] * eq_eval_idx_s_vec[s] * polys[k].get_coeff(j);
                        }
                    }

                    if i == num_rounds - 1 && idx == (1 << num_rounds) - 1 {
                        for k in 0..num_polys {
                            witness_eval_for_final_eval[k][0] = witness_eval[k][0];
                            witness_eval_for_final_eval[k][1] = witness_eval[k][1];
                        }
                    }

                    if (idx + 1) % (1 << (i + 1)) == 0 {
                        for s in 0..=degree {
                            let eval = comb_fn(
                                &(0..num_polys)
                                    .map(|k| witness_eval[k][s])
                                    .collect::<Vec<F>>(),
                            );
                            accumulator[s] += eval;
                        }
                        witness_eval = vec![vec![F::zero(); degree + 1]; num_polys];
                    }
                }
            }

            let univariate_poly = UniPoly::from_evals(&accumulator);
            let compressed_poly = univariate_poly.compress();
            compressed_poly.append_to_transcript(transcript);

            let r_i = transcript.challenge_scalar();
            r.push(r_i);
            compressed_polys.push(compressed_poly);

            if i == num_rounds - 1 {
                final_eval = (0..num_polys)
                    .map(|i| {
                        (F::one() - r_i) * witness_eval_for_final_eval[i][0]
                            + r_i * witness_eval_for_final_eval[i][1]
                    })
                    .collect();
            }

            stream_polys.reset();
        }

        (SumcheckInstanceProof::new(compressed_polys), r, final_eval)
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
pub enum Stream<'a, F: JoltField, I: JoltInstructionSet> {
    SumCheck(StreamSumCheck<'a, F>),
    SpartanSumCheck((BindZRyVarOracle<'a, F, I>, StreamingEqPolynomial<F>)),
}
pub enum OracleItem<F: JoltField> {
    SumCheck(SumCheckPolys<MultilinearPolynomial<F>>),
    SpartanSumCheck(Vec<MultilinearPolynomial<F>>),
}
impl<'a, F: JoltField, I: JoltInstructionSet> Oracle for Stream<'a, F, I> {
    type Item = OracleItem<F>;

    fn next_shard(&mut self, shard_len: usize) -> Self::Item {
        let shard = match self {
            Stream::SumCheck(inner) => {
                let shard = inner.next_shard(shard_len);
                OracleItem::SumCheck(shard)
            }
            Stream::SpartanSumCheck(inner) => {
                let shard1 = inner.0.next_shard(shard_len);
                let shard2 = MultilinearPolynomial::from(inner.1.next_shard(shard_len));
                OracleItem::SpartanSumCheck([shard1, shard2].to_vec())
            }
        };
        shard
    }

    fn reset(&mut self) {
        match self {
            Stream::SumCheck(inner) => {
                inner.reset();
            }
            Stream::SpartanSumCheck(inner) => {
                inner.0.reset();
                inner.1.reset()
            }
        }
    }
}
struct StreamTrace<'a> {
    pub(crate) length: usize,
    pub(crate) counter: usize,
    pub(crate) trace: &'a Vec<u64>,
}
impl<'a> StreamTrace<'a> {
    pub fn new(trace: &'a Vec<u64>) -> Self {
        Self {
            length: trace.len(),
            counter: 0,
            trace,
        }
    }
}
impl<'a> Oracle for StreamTrace<'a> {
    type Item = &'a [u64];

    fn next_shard(&mut self, shard_length: usize) -> Self::Item {
        let shard_start = self.counter;
        self.counter += shard_length;
        &self.trace[shard_start..self.counter]
    }

    fn reset(&mut self) {
        if self.counter == self.length {
            self.counter = 0;
        } else {
            panic!(
                "Can't reset, trace not exhausted. couter {}, length {}",
                self.counter, self.length
            );
        }
    }
}
#[derive(Clone)]
pub struct SumCheckPolys<T> {
    pub poly1: T,
    pub poly2: T,
}

pub struct StreamSumCheck<'a, F: JoltField> {
    pub trace_oracle: StreamTrace<'a>,
    pub func: Box<dyn (Fn(&[u64]) -> SumCheckPolys<MultilinearPolynomial<F>>) + 'a>,
}
impl<'a, F: JoltField> StreamSumCheck<'a, F> {
    pub fn new(trace: &'a Vec<u64>) -> Self {
        let trace_oracle = StreamTrace::new(trace);
        let stream_poly = |shard: &[u64]| {
            let (poly1, poly2): (Vec<F>, Vec<F>) = shard
                .into_iter()
                .map(|value| (F::from_u64(*value), F::from_u64(2 * value)))
                .collect();
            SumCheckPolys {
                poly1: MultilinearPolynomial::from(poly1),
                poly2: MultilinearPolynomial::from(poly2),
            }
        };

        Self {
            trace_oracle,
            func: Box::new(stream_poly),
        }
    }
}

impl<'a, F: JoltField> Oracle for StreamSumCheck<'a, F> {
    type Item = SumCheckPolys<MultilinearPolynomial<F>>;
    fn next_shard(&mut self, shard_length: usize) -> Self::Item {
        (self.func)(self.trace_oracle.next_shard(shard_length))
    }
    fn reset(&mut self) {
        self.trace_oracle.reset()
    }
}

mod test {
    use crate::field::JoltField;
    use crate::jolt::vm::rv32i_vm::RV32I;
    use crate::poly::multilinear_polynomial::MultilinearPolynomial;
    use crate::subprotocols::sumcheck::{
        OracleItem, Stream, StreamSumCheck, SumCheckPolys, SumcheckInstanceProof,
    };
    use crate::utils::streaming::Oracle;
    use crate::utils::transcript::KeccakTranscript;

    #[test]
    fn test_stream_prove_arbitrary() {
        use crate::utils::transcript::Transcript;
        use ark_bn254::Fr;

        let num_vars = 10;
        let num_polys = 2;
        let trace: Vec<u64> = (0..1 << num_vars).map(|elem: u64| elem).collect();
        let mut stream_sum_check_polys = StreamSumCheck::new(&trace);

        let extract_poly_fn = |stream_data: &OracleItem<Fr>| -> Vec<MultilinearPolynomial<Fr>> {
            match stream_data {
                OracleItem::SumCheck(stream) => {
                    [stream.poly1.clone(), stream.poly2.clone()].to_vec()
                }
                _ => vec![],
            }
        };

        let comb_func = |poly_evals: &[Fr]| -> Fr {
            assert_eq!(poly_evals.len(), 2);
            &poly_evals[0] * &poly_evals[1] * &poly_evals[0] + &poly_evals[1]
        };

        let shard_length = 1 << 5;
        let mut transcript = <KeccakTranscript as Transcript>::new(b"test");
        let claim: Fr = (0..1 << num_vars)
            .map(|idx| comb_func(&[Fr::from_u64(idx), Fr::from_u64(2 * idx)]))
            .sum();
        let degree = 3;
        let (proof, _r, final_evals) = SumcheckInstanceProof::stream_prove_arbitrary::<_, _, RV32I>(
            num_vars,
            &mut Stream::SumCheck(stream_sum_check_polys),
            extract_poly_fn,
            comb_func,
            degree,
            shard_length,
            num_polys,
            &mut transcript,
        );
        let mut transcript = <KeccakTranscript as Transcript>::new(b"test");
        let (e_verify, _) = proof
            .verify(claim, num_vars, degree, &mut transcript)
            .unwrap();
        //
        let res = comb_func(&final_evals);
        assert_eq!(res, e_verify, "Final assertion failed");
    }
}
