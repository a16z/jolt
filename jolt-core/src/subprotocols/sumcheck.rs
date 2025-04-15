#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use std::marker::PhantomData;

use ark_serialize::*;
use itertools::Itertools;
use rayon::prelude::*;

use crate::field::JoltField;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::multilinear_polynomial::{
    BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
};
use crate::poly::spartan_interleaved_poly::SpartanInterleavedPolynomial;
use crate::poly::split_eq_poly::SplitEqPolynomial;
use crate::poly::unipoly::{CompressedUniPoly, UniPoly};
use crate::utils::errors::ProofVerifyError;
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

    pub fn stream_prove_arbitrary<O, Func1, Func2>(
        num_rounds: usize,
        stream_polys: &mut O,
        extract_poly_fn: Func1,
        comb_fn: Func2,
        degree: usize,
        shard_length: usize,
        num_polys: usize,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, Vec<F>)
    where
        O: Oracle,
        Func1: Fn(&O::Item) -> Vec<MultilinearPolynomial<F>> + std::marker::Sync,
        Func2: Fn(&[F]) -> F + std::marker::Sync,
    {
        let mut r: Vec<F> = Vec::new();
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::new();
        let mut witness_eval_for_final_eval = vec![vec![F::zero(); 2]; num_polys];

        let mut evals_1 = vec![F::zero(); 1 << (num_rounds - 1)];
        let mut evals_2 = vec![F::zero(); 1 << (num_rounds - 1)];
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

        for round in 0..num_rounds / 2 {
            let mut accumulator = vec![F::zero(); degree + 1];
            let mut witness_eval = vec![vec![F::zero(); degree + 1]; num_polys];

            for shard_idx in 0..num_shards {
                let shards = stream_polys.next_shard(shard_length);
                let polys = extract_poly_fn(&shards);

                for idx_in_shard in 0..shard_length {
                    let idx_in_poly = shard_length * shard_idx + idx_in_shard;
                    let mut eq_eval_idx_s_vec = vec![F::one(); degree + 1];
                    let bit = (idx_in_poly >> round) & 1;

                    for s in 0..=degree {
                        let val = F::from_u64(s as u64);
                        eq_eval_idx_s_vec[s] = if bit == 0 { F::one() - val } else { val };
                    }

                    for k in 0..num_polys {
                        for s in 0..=degree {
                            witness_eval[k][s] += evals_1[idx_in_poly % (1 << round)]
                                * eq_eval_idx_s_vec[s]
                                * polys[k].get_coeff(idx_in_shard);
                        }
                    }

                    if (idx_in_poly + 1) % (1 << (round + 1)) == 0 {
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

            update_evals(&mut evals_1, r_i, 1 << round);

            stream_polys.reset();
        }

        for round in num_rounds / 2..num_rounds {
            let mut accumulator = vec![F::zero(); degree + 1];
            let mut witness_eval = vec![vec![F::zero(); degree + 1]; num_polys];
            let mut int_witness_eval = vec![vec![F::zero(); degree + 1]; num_polys];
            for shard_idx in 0..num_shards {
                let shards = stream_polys.next_shard(shard_length);
                let polys = extract_poly_fn(&shards);

                for idx_in_shard in 0..shard_length {
                    let idx_in_poly = shard_length * shard_idx + idx_in_shard;
                    let mut eq_eval_idx_s_vec = vec![F::one(); degree + 1];

                    let bit = (idx_in_poly >> round) & 1;
                    for s in 0..=degree {
                        let val = F::from_u64(s as u64);
                        eq_eval_idx_s_vec[s] = if bit == 0 { F::one() - val } else { val };
                    }

                    for k in 0..num_polys {
                        for s in 0..=degree {
                            int_witness_eval[k][s] += evals_1
                                [idx_in_poly % (1 << (num_rounds / 2))]
                                * eq_eval_idx_s_vec[s]
                                * polys[k].get_coeff(idx_in_shard);
                        }
                    }

                    if (idx_in_poly + 1) % (1 << (num_rounds / 2)) == 0 {
                        let temp = idx_in_poly >> (num_rounds / 2);
                        for k in 0..num_polys {
                            for s in 0..=degree {
                                witness_eval[k][s] += evals_2
                                    [temp % (1 << (round - num_rounds / 2))]
                                    * int_witness_eval[k][s];
                            }
                        }
                        int_witness_eval = vec![vec![F::zero(); degree + 1]; num_polys];
                    }

                    if (idx_in_poly + 1) % (1 << (round + 1)) == 0 {
                        for s in 0..=degree {
                            let eval = comb_fn(
                                &(0..num_polys)
                                    .map(|k| witness_eval[k][s])
                                    .collect::<Vec<F>>(),
                            );
                            accumulator[s] += eval;
                        }
                        if round == num_rounds - 1 {
                            for k in 0..num_polys {
                                witness_eval_for_final_eval[k][0] = witness_eval[k][0];
                                witness_eval_for_final_eval[k][1] = witness_eval[k][1];
                            }
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

            update_evals(&mut evals_2, r_i, 1 << (round - num_rounds / 2));

            stream_polys.reset();
        }

        let final_eval = (0..num_polys)
            .map(|i| {
                (F::one() - r[num_rounds - 1]) * witness_eval_for_final_eval[i][0]
                    + r[num_rounds - 1] * witness_eval_for_final_eval[i][1]
            })
            .collect();

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

mod test {
    use crate::field::JoltField;
    use crate::poly::multilinear_polynomial::MultilinearPolynomial;
    use crate::subprotocols::sumcheck::SumcheckInstanceProof;
    use crate::utils::streaming::Oracle;
    use crate::utils::transcript::KeccakTranscript;
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

        fn peek(&mut self) -> Option<Self::Item> {
            Some(&self.trace[self.counter..self.counter + 1])
        }

        fn get_length(&self) -> usize {
            self.length
        }

        fn get_step(&self) -> usize {
            self.counter
        }
    }
    #[derive(Clone)]
    struct SumCheckPolys<T> {
        pub poly1: T,
        pub poly2: T,
    }

    struct StreamSumCheck<'a, F: JoltField> {
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

        fn peek(&mut self) -> Option<Self::Item> {
            Some((self.func)(self.trace_oracle.peek().unwrap()))
        }

        fn get_length(&self) -> usize {
            self.trace_oracle.get_length()
        }

        fn get_step(&self) -> usize {
            self.trace_oracle.get_step()
        }
    }

    #[test]
    fn test_stream_prove_arbitrary() {
        use crate::utils::transcript::Transcript;
        use ark_bn254::Fr;

        let num_vars = 20;
        let num_polys = 2;
        let trace: Vec<u64> = (0..1 << num_vars).map(|elem: u64| elem).collect();
        let mut stream_sum_check_polys = StreamSumCheck::new(&trace);

        let extract_poly_fn = |
            stream_data: &SumCheckPolys<MultilinearPolynomial<Fr>>
        | -> Vec<MultilinearPolynomial<Fr>> {
            [stream_data.poly1.clone(), stream_data.poly2.clone()].to_vec()
        };

        let comb_func = |poly_evals: &[Fr]| -> Fr {
            assert_eq!(poly_evals.len(), 2);
            &poly_evals[0] * &poly_evals[1] * &poly_evals[0] + &poly_evals[1]
        };

        let shard_length = 1024;
        let mut transcript = <KeccakTranscript as Transcript>::new(b"test");
        let claim: Fr = (0..1 << num_vars)
            .map(|idx| comb_func(&[Fr::from_u64(idx), Fr::from_u64(2 * idx)]))
            .sum();

        let degree = 3;
        let (proof, _r, final_evals) = SumcheckInstanceProof::stream_prove_arbitrary(
            num_vars,
            &mut stream_sum_check_polys,
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

        let res = comb_func(&final_evals);
        assert_eq!(res, e_verify, "Final assertion failed");
    }
}
