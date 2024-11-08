#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::field::JoltField;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::split_eq_poly::SplitEqPolynomial;
use crate::poly::unipoly::{CompressedUniPoly, UniPoly};
use crate::r1cs::special_polys::{SparsePolynomial, SparseTripleIterator};
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;
use crate::utils::mul_0_optimized;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::{AppendToTranscript, Transcript};
use ark_serialize::*;
use rayon::prelude::*;
use std::marker::PhantomData;

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
        _claim: &F,
        num_rounds: usize,
        polys: &mut Vec<DensePolynomial<F>>,
        comb_func: Func,
        combined_degree: usize,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, Vec<F>)
    where
        Func: Fn(&[F]) -> F + std::marker::Sync,
    {
        let mut r: Vec<F> = Vec::new();
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        for _round in 0..num_rounds {
            // Vector storing evaluations of combined polynomials g(x) = P_0(x) * ... P_{num_polys} (x)
            // for points {0, ..., |g(x)|}
            let mut eval_points = vec![F::zero(); combined_degree + 1];

            let mle_half = polys[0].len() / 2;

            let accum: Vec<Vec<F>> = (0..mle_half)
                .into_par_iter()
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
                    // D_n(index, 0) = D_{n-1}[LOW]
                    // D_n(index, 1) = D_{n-1}[HIGH]
                    // D_n(index, 2) = D_{n-1}[HIGH] + (D_{n-1}[HIGH] - D_{n-1}[LOW])
                    // D_n(index, 3) = D_{n-1}[HIGH] + (D_{n-1}[HIGH] - D_{n-1}[LOW]) + (D_{n-1}[HIGH] - D_{n-1}[LOW])
                    // ...
                    let mut existing_term = params_one;
                    for eval_i in 2..(combined_degree + 1) {
                        let mut poly_evals = vec![F::zero(); polys.len()];
                        for poly_i in 0..polys.len() {
                            let poly = &polys[poly_i];
                            poly_evals[poly_i] = existing_term[poly_i]
                                + poly[mle_half + poly_term_i]
                                - poly[poly_term_i];
                        }

                        accum[eval_i] += comb_func(&poly_evals);
                        existing_term = poly_evals;
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

            let round_uni_poly = UniPoly::from_evals(&eval_points);
            let round_compressed_poly = round_uni_poly.compress();

            // append the prover's message to the transcript
            round_compressed_poly.append_to_transcript(transcript);
            let r_j = transcript.challenge_scalar();
            r.push(r_j);

            // bound all tables to the verifier's challenege
            polys
                .par_iter_mut()
                .for_each(|poly| poly.bound_poly_var_top(&r_j));
            compressed_polys.push(round_compressed_poly);
        }

        let final_evals = polys.iter().map(|poly| poly[0]).collect();

        (SumcheckInstanceProof::new(compressed_polys), r, final_evals)
    }

    #[inline]
    #[tracing::instrument(
        skip_all,
        name = "Spartan2::sumcheck::compute_eval_points_spartan_cubic"
    )]
    /// Binds from the bottom rather than the top.
    pub fn compute_eval_points_spartan_cubic(
        poly_eq: &SplitEqPolynomial<F>,
        poly_A: &SparsePolynomial<F>,
        poly_B: &SparsePolynomial<F>,
        poly_C: &SparsePolynomial<F>,
    ) -> (F, F, F) {
        let comb_func = |eq: &F, az: &F, bz: &F, cz: &F| -> F {
            // Below is an optimized form of: eq * (Az * Bz - Cz)
            if az.is_zero() || bz.is_zero() {
                if cz.is_zero() {
                    F::zero()
                } else {
                    *eq * (-(*cz))
                }
            } else {
                let inner = *az * *bz - *cz;
                if inner.is_zero() {
                    F::zero()
                } else {
                    *eq * inner
                }
            }
        };

        // num_threads * 8 enables better work stealing
        let mut iterators =
            SparseTripleIterator::chunks(poly_A, poly_B, poly_C, rayon::current_num_threads() * 16);

        // We use the Dao-Thaler optimization for the EQ polynomial, so there are two cases we
        // must handle. For details, refer to Section 2.2 of https://eprint.iacr.org/2024/1210.pdf
        if poly_eq.E1_len == 1 {
            let eq_evals: Vec<(F, F, F)> = poly_eq.E2[..poly_eq.E2_len]
                .par_chunks(2)
                .map(|eq_chunk| {
                    let eval_point_0 = eq_chunk[0];
                    let m_eq = eq_chunk[1] - eq_chunk[0];
                    let eval_point_2 = eq_chunk[1] + m_eq;
                    let eval_point_3 = eval_point_2 + m_eq;
                    (eval_point_0, eval_point_2, eval_point_3)
                })
                .collect();

            iterators
                .par_iter_mut()
                .map(|iterator| {
                    let span = tracing::span!(tracing::Level::DEBUG, "eval_par_inner");
                    let _enter = span.enter();
                    let mut eval_point_0 = F::zero();
                    let mut eval_point_2 = F::zero();
                    let mut eval_point_3 = F::zero();
                    while iterator.has_next() {
                        let (dense_index, a_low, a_high, b_low, b_high, c_low, c_high) =
                            iterator.next_pairs();
                        assert!(dense_index % 2 == 0);
                        let eq_evals = eq_evals[dense_index / 2];

                        // eval 0: bound_func is A(low)
                        eval_point_0 += comb_func(&eq_evals.0, &a_low, &b_low, &c_low);

                        let m_A = a_high - a_low;
                        let m_B = b_high - b_low;
                        let m_C = c_high - c_low;

                        // eval 2
                        let poly_A_bound_point = a_high + m_A;
                        let poly_B_bound_point = b_high + m_B;
                        let poly_C_bound_point = c_high + m_C;
                        eval_point_2 += comb_func(
                            &eq_evals.1,
                            &poly_A_bound_point,
                            &poly_B_bound_point,
                            &poly_C_bound_point,
                        );

                        // eval 3
                        let poly_A_bound_point = poly_A_bound_point + m_A;
                        let poly_B_bound_point = poly_B_bound_point + m_B;
                        let poly_C_bound_point = poly_C_bound_point + m_C;
                        eval_point_3 += comb_func(
                            &eq_evals.2,
                            &poly_A_bound_point,
                            &poly_B_bound_point,
                            &poly_C_bound_point,
                        );
                    }
                    (eval_point_0, eval_point_2, eval_point_3)
                })
                .reduce(
                    || (F::zero(), F::zero(), F::zero()),
                    |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
                )
        } else {
            // We start by computing the E1 evals:
            // (1 - j) * E1[0, x1] + j * E1[1, x1]
            let E1_evals: Vec<_> = poly_eq.E1[..poly_eq.E1_len]
                .par_chunks(2)
                .map(|E1_chunk| {
                    let eval_point_0 = E1_chunk[0];
                    let m_eq = E1_chunk[1] - E1_chunk[0];
                    let eval_point_2 = E1_chunk[1] + m_eq;
                    let eval_point_3 = eval_point_2 + m_eq;
                    (eval_point_0, eval_point_2, eval_point_3)
                })
                .collect();

            let num_x1_bits = poly_eq.E1_len.log_2() - 1;
            let x1_bitmask = (1 << num_x1_bits) - 1;

            iterators
                .par_iter_mut()
                .map(|iterator| {
                    let span = tracing::span!(tracing::Level::DEBUG, "eval_par_inner");
                    let _enter = span.enter();
                    let mut eval_point_0 = F::zero();
                    let mut eval_point_2 = F::zero();
                    let mut eval_point_3 = F::zero();

                    let mut inner_sums = (F::zero(), F::zero(), F::zero());
                    let mut prev_x2 = 0;

                    while iterator.has_next() {
                        let (dense_index, a_low, a_high, b_low, b_high, c_low, c_high) =
                            iterator.next_pairs();
                        assert!(dense_index % 2 == 0);

                        let x1 = (dense_index / 2) & x1_bitmask;
                        let E1_evals = E1_evals[x1];
                        let x2 = (dense_index / 2) >> num_x1_bits;

                        if x2 != prev_x2 {
                            eval_point_0 += poly_eq.E2[prev_x2] * inner_sums.0;
                            eval_point_2 += poly_eq.E2[prev_x2] * inner_sums.1;
                            eval_point_3 += poly_eq.E2[prev_x2] * inner_sums.2;

                            inner_sums = (F::zero(), F::zero(), F::zero());
                            prev_x2 = x2;
                        }

                        // eval 0: bound_func is A(low)
                        inner_sums.0 += comb_func(&E1_evals.0, &a_low, &b_low, &c_low);

                        let m_A = a_high - a_low;
                        let m_B = b_high - b_low;
                        let m_C = c_high - c_low;

                        // eval 2
                        let poly_A_bound_point = a_high + m_A;
                        let poly_B_bound_point = b_high + m_B;
                        let poly_C_bound_point = c_high + m_C;
                        inner_sums.1 += comb_func(
                            &E1_evals.1,
                            &poly_A_bound_point,
                            &poly_B_bound_point,
                            &poly_C_bound_point,
                        );

                        // eval 3
                        let poly_A_bound_point = poly_A_bound_point + m_A;
                        let poly_B_bound_point = poly_B_bound_point + m_B;
                        let poly_C_bound_point = poly_C_bound_point + m_C;
                        inner_sums.2 += comb_func(
                            &E1_evals.2,
                            &poly_A_bound_point,
                            &poly_B_bound_point,
                            &poly_C_bound_point,
                        );
                    }

                    eval_point_0 += poly_eq.E2[prev_x2] * inner_sums.0;
                    eval_point_2 += poly_eq.E2[prev_x2] * inner_sums.1;
                    eval_point_3 += poly_eq.E2[prev_x2] * inner_sums.2;

                    (eval_point_0, eval_point_2, eval_point_3)
                })
                .reduce(
                    || (F::zero(), F::zero(), F::zero()),
                    |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
                )
        }
    }

    #[tracing::instrument(skip_all, name = "Spartan2::sumcheck::prove_spartan_cubic")]
    pub fn prove_spartan_cubic(
        claim: &F,
        num_rounds: usize,
        poly_eq: &mut SplitEqPolynomial<F>,
        poly_A: &mut SparsePolynomial<F>,
        poly_B: &mut SparsePolynomial<F>,
        poly_C: &mut SparsePolynomial<F>,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, Vec<F>) {
        let mut r: Vec<F> = Vec::new();
        let mut polys: Vec<CompressedUniPoly<F>> = Vec::new();
        let mut claim_per_round = *claim;

        for _ in 0..num_rounds {
            let poly = {
                // Make an iterator returning the contributions to the evaluations
                let (eval_point_0, eval_point_2, eval_point_3) =
                    Self::compute_eval_points_spartan_cubic(poly_eq, poly_A, poly_B, poly_C);

                let evals = [
                    eval_point_0,
                    claim_per_round - eval_point_0,
                    eval_point_2,
                    eval_point_3,
                ];

                UniPoly::from_evals(&evals)
            };

            let compressed_poly = poly.compress();

            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);

            //derive the verifier's challenge for the next round
            let r_i = transcript.challenge_scalar();
            r.push(r_i);
            polys.push(compressed_poly);

            // Set up next round
            claim_per_round = poly.evaluate(&r_i);

            // bound all tables to the verifier's challenege
            poly_eq.bind(r_i);
            poly_A.bound_poly_var_bot_par(&r_i);
            poly_B.bound_poly_var_bot_par(&r_i);
            poly_C.bound_poly_var_bot_par(&r_i);
        }

        (
            SumcheckInstanceProof::new(polys),
            r,
            vec![
                poly_A.final_eval(),
                poly_B.final_eval(),
                poly_C.final_eval(),
            ],
        )
    }

    #[tracing::instrument(skip_all)]
    // A specialized sumcheck implementation with the 0th round unrolled from the rest of the
    // `for` loop. This allows us to pass in `witness_polynomials` by reference instead of
    // passing them in as a single `DensePolynomial`, which would require an expensive
    // concatenation. We defer the actual instantation of a `DensePolynomial` to the end of the
    // 0th round.
    pub fn prove_spartan_quadratic(
        claim: &F,
        num_rounds: usize,
        poly_A: &mut DensePolynomial<F>,
        witness_polynomials: &[&DensePolynomial<F>],
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, Vec<F>) {
        let mut r: Vec<F> = Vec::with_capacity(num_rounds);
        let mut polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
        let mut claim_per_round = *claim;

        /*          Round 0 START         */

        let len = poly_A.len() / 2;
        let trace_len = witness_polynomials[0].len();
        witness_polynomials
            .iter()
            .for_each(|poly| debug_assert_eq!(poly.len(), trace_len));

        // We don't materialize the full, flattened witness vector, but this closure
        // simulates it
        let witness_value = |index: usize| {
            if (index / trace_len) >= witness_polynomials.len() {
                F::zero()
            } else {
                witness_polynomials[index / trace_len][index % trace_len]
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
                &(F::from_u64(2).unwrap() - witness_value(0)),
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
