#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::field::JoltField;
use crate::poly::unipoly::{CompressedUniPoly, UniPoly};
use crate::r1cs::spartan::IndexablePoly;
use crate::utils::errors::ProofVerifyError;
use crate::utils::mul_0_optimized;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::{AppendToTranscript, ProofTranscript};
use allocative::Allocative;
use ark_serialize::*;
use itertools::multizip;
use rayon::prelude::*;
use tracing::trace_span;

#[derive(Debug, Clone, PartialEq, Allocative)]
pub enum CubicSumcheckType {
    // eq * A * B
    Prod,

    // eq * A * B, optimized for high probability (A, B) = 1
    ProdOnes,

    // eq *(A * flags + (1 - flags))
    Flags,
}

#[derive(Debug, Clone, PartialEq, Allocative)]
pub struct CubicSumcheckParams<F: JoltField> {
    poly_As: Vec<DensePolynomial<F>>,
    poly_Bs: Vec<DensePolynomial<F>>,

    poly_eq: DensePolynomial<F>,

    pub num_rounds: usize,

    pub sumcheck_type: CubicSumcheckType,
}

impl<F: JoltField> CubicSumcheckParams<F> {
    pub fn new_prod(
        poly_lefts: Vec<DensePolynomial<F>>,
        poly_rights: Vec<DensePolynomial<F>>,
        poly_eq: DensePolynomial<F>,
        num_rounds: usize,
    ) -> Self {
        debug_assert_eq!(poly_lefts.len(), poly_rights.len());
        debug_assert_eq!(poly_lefts[0].len(), poly_rights[0].len());
        debug_assert_eq!(poly_lefts[0].len(), poly_eq.len());

        CubicSumcheckParams {
            poly_As: poly_lefts,
            poly_Bs: poly_rights,
            poly_eq,
            num_rounds,
            sumcheck_type: CubicSumcheckType::Prod,
        }
    }

    pub fn new_prod_ones(
        poly_lefts: Vec<DensePolynomial<F>>,
        poly_rights: Vec<DensePolynomial<F>>,
        poly_eq: DensePolynomial<F>,
        num_rounds: usize,
    ) -> Self {
        debug_assert_eq!(poly_lefts.len(), poly_rights.len());
        debug_assert_eq!(poly_lefts[0].len(), poly_rights[0].len());
        debug_assert_eq!(poly_lefts[0].len(), poly_eq.len());

        CubicSumcheckParams {
            poly_As: poly_lefts,
            poly_Bs: poly_rights,
            poly_eq,
            num_rounds,
            sumcheck_type: CubicSumcheckType::ProdOnes,
        }
    }

    pub fn new_flags(
        poly_leaves: Vec<DensePolynomial<F>>,
        poly_flags: Vec<DensePolynomial<F>>,
        poly_eq: DensePolynomial<F>,
        num_rounds: usize,
    ) -> Self {
        debug_assert_eq!(poly_leaves[0].len(), poly_flags[0].len());
        debug_assert_eq!(poly_leaves[0].len(), poly_eq.len());

        CubicSumcheckParams {
            poly_As: poly_leaves,
            poly_Bs: poly_flags,
            poly_eq,
            num_rounds,
            sumcheck_type: CubicSumcheckType::Flags,
        }
    }

    #[inline]
    pub fn combine_prod(l: &F, r: &F, eq: &F) -> F {
        if *l == F::one() && *r == F::one() {
            *eq
        } else if *l == F::one() {
            *r * eq
        } else if *r == F::one() {
            *l * eq
        } else {
            *l * r * eq
        }
    }

    #[inline]
    pub fn combine_flags(h: &F, flag: &F, eq: &F) -> F {
        if *flag == F::zero() {
            *eq
        } else if *flag == F::one() {
            *eq * *h
        } else {
            *eq * (*flag * h + (F::one() + flag.neg()))
        }
    }

    pub fn get_final_evals(&self) -> (Vec<F>, Vec<F>, F) {
        debug_assert_eq!(self.poly_As[0].len(), 1);
        debug_assert_eq!(self.poly_Bs[0].len(), 1);
        debug_assert_eq!(self.poly_eq.len(), 1);

        let poly_A_final: Vec<F> = (0..self.poly_As.len())
            .map(|i| self.poly_As[i][0])
            .collect();

        let poly_B_final: Vec<F> = (0..self.poly_Bs.len())
            .map(|i| self.poly_Bs[i][0])
            .collect();

        let poly_eq_final = self.poly_eq[0];

        (poly_A_final, poly_B_final, poly_eq_final)
    }
}

impl<F: JoltField> SumcheckInstanceProof<F> {
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

            // append the prover's message to the transcript
            round_uni_poly.append_to_transcript(b"poly", transcript);
            let r_j = transcript.challenge_scalar(b"challenge_nextround");
            r.push(r_j);

            // bound all tables to the verifier's challenege
            polys
                .par_iter_mut()
                .for_each(|poly| poly.bound_poly_var_top(&r_j));
            compressed_polys.push(round_uni_poly.compress());
        }

        let final_evals = polys.iter().map(|poly| poly[0]).collect();

        (SumcheckInstanceProof::new(compressed_polys), r, final_evals)
    }

    #[tracing::instrument(skip_all, name = "Sumcheck.prove_batched")]
    pub fn prove_cubic_batched(
        claim: &F,
        params: CubicSumcheckParams<F>,
        coeffs: &[F],
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, (Vec<F>, Vec<F>, F)) {
        match params.sumcheck_type {
            CubicSumcheckType::Prod => {
                Self::prove_cubic_batched_prod(claim, params, coeffs, transcript)
            }
            CubicSumcheckType::ProdOnes => {
                Self::prove_cubic_batched_prod_ones(claim, params, coeffs, transcript)
            }
            CubicSumcheckType::Flags => {
                Self::prove_cubic_batched_flags(claim, params, coeffs, transcript)
            }
        }
    }

    #[tracing::instrument(skip_all, name = "Sumcheck.prove_cubic_batched_prod")]
    pub fn prove_cubic_batched_prod(
        claim: &F,
        params: CubicSumcheckParams<F>,
        coeffs: &[F],
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, (Vec<F>, Vec<F>, F)) {
        assert_eq!(params.poly_As.len(), params.poly_Bs.len());
        assert_eq!(params.poly_As.len(), coeffs.len());

        let mut params = params;

        let mut e = *claim;
        let mut r: Vec<F> = Vec::new();
        let mut cubic_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        for _j in 0..params.num_rounds {
            let len = params.poly_As[0].len() / 2;
            let eq = &params.poly_eq;

            let _span = trace_span!("eval_loop");
            let _enter = _span.enter();
            let evals = (0..len)
                .into_par_iter()
                .map(|low_index| {
                    let high_index = low_index + len;

                    let eq_evals = {
                        let eval_point_0 = eq[low_index];
                        let m_eq = eq[high_index] - eq[low_index];
                        let eval_point_2 = eq[high_index] + m_eq;
                        let eval_point_3 = eval_point_2 + m_eq;
                        (eval_point_0, eval_point_2, eval_point_3)
                    };

                    let mut evals = (F::zero(), F::zero(), F::zero());

                    for (coeff, poly_A, poly_B) in
                        multizip((coeffs, &params.poly_As, &params.poly_Bs))
                    {
                        // We want to compute:
                        //     evals.0 += coeff * poly_A[low_index] * poly_B[low_index]
                        //     evals.1 += coeff * (2 * poly_A[high_index] - poly_A[low_index]) * (2 * poly_B[high_index] - poly_B[low_index])
                        //     evals.0 += coeff * (3 * poly_A[high_index] - 2 * poly_A[low_index]) * (3 * poly_B[high_index] - 2 * poly_B[low_index])
                        // which naively requires 3 multiplications by `coeff`.
                        // By computing these values `A_low` and `A_high`, we only use 2 multiplications by `coeff`.
                        let A_low = *coeff * poly_A[low_index];
                        let A_high = *coeff * poly_A[high_index];

                        let m_a = A_high - A_low;
                        let m_b = poly_B[high_index] - poly_B[low_index];

                        let point_2_A = A_high + m_a;
                        let point_3_A = point_2_A + m_a;

                        let point_2_B = poly_B[high_index] + m_b;
                        let point_3_B = point_2_B + m_b;

                        evals.0 += A_low * poly_B[low_index];
                        evals.1 += point_2_A * point_2_B;
                        evals.2 += point_3_A * point_3_B;
                    }

                    evals.0 *= eq_evals.0;
                    evals.1 *= eq_evals.1;
                    evals.2 *= eq_evals.2;
                    evals
                })
                .reduce(
                    || (F::zero(), F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                );
            drop(_enter);
            drop(_span);

            let evals = [evals.0, e - evals.0, evals.1, evals.2];
            let poly = UniPoly::from_evals(&evals);

            // append the prover's message to the transcript
            poly.append_to_transcript(b"poly", transcript);

            //derive the verifier's challenge for the next round
            let r_j = transcript.challenge_scalar(b"challenge_nextround");
            r.push(r_j);

            // bound all tables to the verifier's challenege
            let _span = trace_span!("binding");
            let _enter = _span.enter();

            let poly_iter = params
                .poly_As
                .par_iter_mut()
                .chain(params.poly_Bs.par_iter_mut());

            rayon::join(
                || poly_iter.for_each(|poly| poly.bound_poly_var_top(&r_j)),
                || params.poly_eq.bound_poly_var_top(&r_j),
            );

            drop(_enter);
            drop(_span);

            e = poly.evaluate(&r_j);
            cubic_polys.push(poly.compress());
        }

        let claims_prod = params.get_final_evals();

        drop_in_background_thread(params);

        (SumcheckInstanceProof::new(cubic_polys), r, claims_prod)
    }

    #[tracing::instrument(skip_all, name = "Sumcheck.prove_cubic_batched_prod_ones")]
    pub fn prove_cubic_batched_prod_ones(
        claim: &F,
        params: CubicSumcheckParams<F>,
        coeffs: &[F],
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, (Vec<F>, Vec<F>, F)) {
        let mut params = params;

        let mut e = *claim;
        let mut r: Vec<F> = Vec::new();
        let mut cubic_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        for _j in 0..params.num_rounds {
            let len = params.poly_As[0].len() / 2;
            let eq = &params.poly_eq;
            let eq_evals: Vec<(F, F, F)> = (0..len)
                .into_par_iter()
                .map(|i| {
                    let low = i;
                    let high = len + i;

                    let eval_point_0 = eq[low];
                    let m_eq = eq[high] - eq[low];
                    let eval_point_2 = eq[high] + m_eq;
                    let eval_point_3 = eval_point_2 + m_eq;
                    (eval_point_0, eval_point_2, eval_point_3)
                })
                .collect();

            let _span = trace_span!("eval_loop");
            let _enter = _span.enter();
            let evals: Vec<(F, F, F)> = (0..params.poly_As.len())
                .into_par_iter()
                .with_max_len(4)
                .map(|batch_index| {
                    let poly_A = &params.poly_As[batch_index];
                    let poly_B = &params.poly_Bs[batch_index];
                    let len = poly_A.len() / 2;

                    // In the case of a flagged tree, the majority of the leaves will be 1s, optimize for this case.
                    let (eval_point_0, eval_point_2, eval_point_3) = (0..len)
                        .map(|mle_index| {
                            let low = mle_index;
                            let high = len + mle_index;

                            // Optimized version of the product for the high probability that A[low], A[high], B[low], B[high] == 1

                            let a_low_one = poly_A[low].is_one();
                            let a_high_one = poly_A[high].is_one();
                            let b_low_one = poly_B[low].is_one();
                            let b_high_one = poly_B[high].is_one();

                            let eval_point_0: F = if a_low_one && b_low_one {
                                eq_evals[low].0
                            }
                            else if a_low_one {
                                poly_B[low] * eq_evals[low].0
                            }
                            else if b_low_one {
                                poly_A[low] * eq_evals[low].0
                            }
                            else {
                                poly_A[low] * poly_B[low] * eq_evals[low].0
                            };

                            let m_a_zero = a_low_one && a_high_one;
                            let m_b_zero = b_low_one && b_high_one;

                            let (eval_point_2, eval_point_3) = if m_a_zero && m_b_zero {
                                (eq_evals[low].1, eq_evals[low].2)
                            }
                            else if m_a_zero {
                                let m_b = poly_B[high] - poly_B[low];
                                let point_2_B = poly_B[high] + m_b;
                                let point_3_B = point_2_B + m_b;

                                let eval_point_2 = eq_evals[low].1 * point_2_B;
                                let eval_point_3 = eq_evals[low].2 * point_3_B;
                                (eval_point_2, eval_point_3)
                            } else if m_b_zero {
                                let m_a = poly_A[high] - poly_A[low];
                                let point_2_A = poly_A[high] + m_a;
                                let point_3_A = point_2_A + m_a;

                                let eval_point_2 = eq_evals[low].1 * point_2_A;
                                let eval_point_3 = eq_evals[low].2 * point_3_A;
                                (eval_point_2, eval_point_3)
                            }
                            else {
                                let m_a = poly_A[high] - poly_A[low];
                                let m_b = poly_B[high] - poly_B[low];

                                let point_2_A = poly_A[high] + m_a;
                                let point_3_A = point_2_A + m_a;

                                let point_2_B = poly_B[high] + m_b;
                                let point_3_B = point_2_B + m_b;

                                let eval_point_2 = eq_evals[low].1 * point_2_A * point_2_B;
                                let eval_point_3 = eq_evals[low].2 * point_3_A * point_3_B;
                                (eval_point_2, eval_point_3)
                            };

                            (eval_point_0, eval_point_2, eval_point_3)
                        })
                        // For parallel
                        // .reduce(
                        //     || (F::zero(), F::zero(), F::zero()),
                        //     |(sum_0, sum_2, sum_3), (a, b, c)| (sum_0 + a, sum_2 + b, sum_3 + c),
                        // );
                        // For normal
                        .fold(
                            (F::zero(), F::zero(), F::zero()),
                            |(sum_0, sum_2, sum_3), (a, b, c)| (sum_0 + a, sum_2 + b, sum_3 + c),
                        );

                    (eval_point_0, eval_point_2, eval_point_3)
                })
                .collect();
            drop(_enter);
            drop(_span);

            let evals_combined_0 = (0..evals.len()).map(|i| evals[i].0 * coeffs[i]).sum();
            let evals_combined_2 = (0..evals.len()).map(|i| evals[i].1 * coeffs[i]).sum();
            let evals_combined_3 = (0..evals.len()).map(|i| evals[i].2 * coeffs[i]).sum();

            let evals = [
                evals_combined_0,
                e - evals_combined_0,
                evals_combined_2,
                evals_combined_3,
            ];
            let poly = UniPoly::from_evals(&evals);

            // append the prover's message to the transcript
            poly.append_to_transcript(b"poly", transcript);

            //derive the verifier's challenge for the next round
            let r_j = transcript.challenge_scalar(b"challenge_nextround");
            r.push(r_j);

            // bound all tables to the verifier's challenege
            let _span = trace_span!("binding (ones)");
            let _enter = _span.enter();

            let poly_iter = params
                .poly_As
                .par_iter_mut()
                .chain(params.poly_Bs.par_iter_mut());

            rayon::join(
                || poly_iter.for_each(|poly| poly.bound_poly_var_top_many_ones(&r_j)),
                || params.poly_eq.bound_poly_var_top(&r_j),
            );

            drop(_enter);
            drop(_span);

            e = poly.evaluate(&r_j);
            cubic_polys.push(poly.compress());
        }

        let claims_prod = params.get_final_evals();

        drop_in_background_thread(params);

        (SumcheckInstanceProof::new(cubic_polys), r, claims_prod)
    }

    #[tracing::instrument(skip_all, name = "SumcheckInstanceProof::compute_cubic_evals_flags")]
    fn compute_cubic_evals_flags(
        flags: &DensePolynomial<F>,
        leaves: &DensePolynomial<F>,
        eq_evals: &Vec<(F, F, F)>,
        len: usize,
    ) -> (F, F, F) {
        let (flags_low, flags_high) = flags.split_evals(len);
        let (leaves_low, leaves_high) = leaves.split_evals(len);

        let mut evals = (F::zero(), F::zero(), F::zero());
        for (&flag_low, &flag_high, &leaf_low, &leaf_high, eq_eval) in
            multizip((flags_low, flags_high, leaves_low, leaves_high, eq_evals))
        {
            let m_eq: F = flag_high - flag_low;
            let (flag_eval_point_2, flag_eval_point_3) = if m_eq.is_zero() {
                (flag_high, flag_high)
            } else {
                let eval_point_2 = flag_high + m_eq;
                let eval_point_3 = eval_point_2 + m_eq;
                (eval_point_2, eval_point_3)
            };

            let flag_eval = (flag_low, flag_eval_point_2, flag_eval_point_3);

            if flag_eval.0.is_zero() {
                evals.0 += eq_eval.0
            } else if flag_eval.0.is_one() {
                evals.0 += eq_eval.0 * leaf_low
            } else {
                evals.0 += eq_eval.0 * (flag_eval.0 * leaf_low + (F::one() - flag_eval.0))
            };

            let opt_poly_2_res: Option<(F, F)> = if flag_eval.1.is_zero() {
                evals.1 += eq_eval.1;
                None
            } else if flag_eval.1.is_one() {
                let poly_m = leaf_high - leaf_low;
                let poly_2 = leaf_high + poly_m;
                evals.1 += eq_eval.1 * poly_2;
                Some((poly_2, poly_m))
            } else {
                let poly_m = leaf_high - leaf_low;
                let poly_2 = leaf_high + poly_m;
                evals.1 += eq_eval.1 * (flag_eval.1 * poly_2 + (F::one() - flag_eval.1));
                Some((poly_2, poly_m))
            };

            if let Some((poly_2, poly_m)) = opt_poly_2_res {
                if flag_eval.2.is_zero() {
                    evals.2 += eq_eval.2; // TODO(sragss): Path may never happen
                } else if flag_eval.2.is_one() {
                    let poly_3 = poly_2 + poly_m;
                    evals.2 += eq_eval.2 * poly_3;
                } else {
                    let poly_3 = poly_2 + poly_m;
                    evals.2 += eq_eval.2 * (flag_eval.2 * poly_3 + (F::one() - flag_eval.2));
                }
            } else {
                evals.2 += eq_eval.2;
            };

            // Above is just a more complicated form of the following, optimizing for 0 / 1 flags.
            // let poly_m = poly_eval[high] - poly_eval[low];
            // let poly_2 = poly_eval[high] + poly_m;
            // let poly_3 = poly_2 + poly_m;

            // let eval_0 += params.combine(&poly_eval[low], &flag_eval.0, &eq_eval.0);
            // let eval_2 += params.combine(&poly_2, &flag_eval.1, &eq_eval.1);
            // let eval_3 += params.combine(&poly_3, &flag_eval.2, &eq_eval.2);
        }
        evals
    }

    #[tracing::instrument(skip_all, name = "Sumcheck.prove_batched_special_fork_flags")]
    pub fn prove_cubic_batched_flags(
        claim: &F,
        params: CubicSumcheckParams<F>,
        coeffs: &[F],
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, (Vec<F>, Vec<F>, F)) {
        let mut params = params;

        let mut e = *claim;
        let mut r: Vec<F> = Vec::new();
        let mut cubic_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        let mut eq_evals: Vec<(F, F, F)> = Vec::with_capacity(params.poly_As[0].len() / 2);

        for _j in 0..params.num_rounds {
            let len = params.poly_As[0].len() / 2;
            let eq_span = trace_span!("eq_evals");
            let _eq_enter = eq_span.enter();
            (0..len)
                .into_par_iter()
                .map(|i| {
                    let low = i;
                    let high = len + i;

                    let eq = &params.poly_eq;

                    let eval_point_0 = eq[low];
                    let m_eq = eq[high] - eq[low];
                    let eval_point_2 = eq[high] + m_eq;
                    let eval_point_3 = eval_point_2 + m_eq;
                    (eval_point_0, eval_point_2, eval_point_3)
                })
                .collect_into_vec(&mut eq_evals);
            drop(_eq_enter);
            drop(eq_span);

            let _span = trace_span!("eval_loop");
            let _enter = _span.enter();
            let evals: Vec<(F, F, F)> = params
                .poly_Bs
                .par_iter()
                .enumerate()
                .flat_map(|(memory_index, memory_flag_poly)| {
                    let read_leaves = &params.poly_As[2 * memory_index];
                    let write_leaves = &params.poly_As[2 * memory_index + 1];

                    let (read_evals, write_evals) = rayon::join(
                        || {
                            Self::compute_cubic_evals_flags(
                                memory_flag_poly,
                                read_leaves,
                                &eq_evals,
                                len,
                            )
                        },
                        || {
                            Self::compute_cubic_evals_flags(
                                memory_flag_poly,
                                write_leaves,
                                &eq_evals,
                                len,
                            )
                        },
                    );

                    [read_evals, write_evals]
                })
                .collect();
            drop(_enter);
            drop(_span);

            let evals_combined_0 = (0..evals.len()).map(|i| evals[i].0 * coeffs[i]).sum();
            let evals_combined_2 = (0..evals.len()).map(|i| evals[i].1 * coeffs[i]).sum();
            let evals_combined_3 = (0..evals.len()).map(|i| evals[i].2 * coeffs[i]).sum();

            let cubic_evals = [
                evals_combined_0,
                e - evals_combined_0,
                evals_combined_2,
                evals_combined_3,
            ];
            let poly = UniPoly::from_evals(&cubic_evals);

            // append the prover's message to the transcript
            poly.append_to_transcript(b"poly", transcript);

            //derive the verifier's challenge for the next round
            let r_j = transcript.challenge_scalar(b"challenge_nextround");
            r.push(r_j);

            let poly_As_span = trace_span!("Bind leaves");
            let _poly_As_enter = poly_As_span.enter();
            params
                .poly_As
                .par_iter_mut()
                .for_each(|poly| poly.bound_poly_var_top(&r_j));
            drop(_poly_As_enter);
            drop(poly_As_span);

            let poly_other_span = trace_span!("Bind EQ and flags");
            let _poly_other_enter = poly_other_span.enter();
            rayon::join(
                || params.poly_eq.bound_poly_var_top(&r_j),
                || {
                    params
                        .poly_Bs
                        .par_iter_mut()
                        .for_each(|poly| poly.bound_poly_var_top_many_ones(&r_j))
                },
            );
            drop(_poly_other_enter);
            drop(poly_other_span);

            e = poly.evaluate(&r_j);
            cubic_polys.push(poly.compress());
        }

        let leaves_claims: Vec<F> = (0..params.poly_As.len())
            .map(|i| params.poly_As[i][0])
            .collect();

        let flags_claims: Vec<F> = (0..params.poly_As.len())
            .map(|i| params.poly_Bs[i / 2][0])
            .collect();

        let poly_eq_final = params.poly_eq[0];

        let claims_prod = (leaves_claims, flags_claims, poly_eq_final);

        drop_in_background_thread(params);

        (SumcheckInstanceProof::new(cubic_polys), r, claims_prod)
    }

    #[inline]
    #[tracing::instrument(
        skip_all,
        name = "Spartan2::sumcheck::compute_eval_points_spartan_cubic"
    )]
    pub fn compute_eval_points_spartan_cubic<Func>(
        poly_A: &DensePolynomial<F>,
        poly_B: &DensePolynomial<F>,
        poly_C: &DensePolynomial<F>,
        poly_D: &DensePolynomial<F>,
        comb_func: &Func,
    ) -> (F, F, F)
    where
        Func: Fn(&F, &F, &F, &F) -> F + Sync,
    {
        let len = poly_A.len() / 2;
        (0..len)
            .into_par_iter()
            .map(|i| {
                // eval 0: bound_func is A(low)
                let eval_point_0 = comb_func(&poly_A[i], &poly_B[i], &poly_C[i], &poly_D[i]);

                let m_A = poly_A[len + i] - poly_A[i];
                let m_B = poly_B[len + i] - poly_B[i];
                let m_C = poly_C[len + i] - poly_C[i];
                let m_D = poly_D[len + i] - poly_D[i];

                // eval 2: bound_func is -A(low) + 2*A(high)
                let poly_A_bound_point = poly_A[len + i] + m_A;
                let poly_B_bound_point = poly_B[len + i] + m_B;
                let poly_C_bound_point = poly_C[len + i] + m_C;
                let poly_D_bound_point = poly_D[len + i] + m_D;
                let eval_point_2 = comb_func(
                    &poly_A_bound_point,
                    &poly_B_bound_point,
                    &poly_C_bound_point,
                    &poly_D_bound_point,
                );

                // eval 3: bound_func is -2A(low) + 3A(high); computed incrementally with bound_func applied to eval(2)
                let poly_A_bound_point = poly_A_bound_point + m_A;
                let poly_B_bound_point = poly_B_bound_point + m_B;
                let poly_C_bound_point = poly_C_bound_point + m_C;
                let poly_D_bound_point = poly_D_bound_point + m_D;
                let eval_point_3 = comb_func(
                    &poly_A_bound_point,
                    &poly_B_bound_point,
                    &poly_C_bound_point,
                    &poly_D_bound_point,
                );
                (eval_point_0, eval_point_2, eval_point_3)
            })
            .reduce(
                || (F::zero(), F::zero(), F::zero()),
                |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
            )
    }

    #[tracing::instrument(skip_all, name = "Spartan2::sumcheck::prove_spartan_cubic")]
    pub fn prove_spartan_cubic<Func>(
        claim: &F,
        num_rounds: usize,
        poly_A: &mut DensePolynomial<F>,
        poly_B: &mut DensePolynomial<F>,
        poly_C: &mut DensePolynomial<F>,
        poly_D: &mut DensePolynomial<F>,
        comb_func: Func,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, Vec<F>)
    where
        Func: Fn(&F, &F, &F, &F) -> F + Sync,
    {
        let mut r: Vec<F> = Vec::new();
        let mut polys: Vec<CompressedUniPoly<F>> = Vec::new();
        let mut claim_per_round = *claim;

        for _ in 0..num_rounds {
            let poly = {
                // Make an iterator returning the contributions to the evaluations
                let (eval_point_0, eval_point_2, eval_point_3) =
                    Self::compute_eval_points_spartan_cubic(
                        poly_A, poly_B, poly_C, poly_D, &comb_func,
                    );

                let evals = [
                    eval_point_0,
                    claim_per_round - eval_point_0,
                    eval_point_2,
                    eval_point_3,
                ];
                UniPoly::from_evals(&evals)
            };

            // append the prover's message to the transcript
            poly.append_to_transcript(b"poly", transcript);

            //derive the verifier's challenge for the next round
            let r_i = transcript.challenge_scalar(b"challenge_nextround");
            r.push(r_i);
            polys.push(poly.compress());

            // Set up next round
            claim_per_round = poly.evaluate(&r_i);

            // bound all tables to the verifier's challenege
            rayon::join(
                || poly_A.bound_poly_var_top_par(&r_i),
                || {
                    rayon::join(
                        || poly_B.bound_poly_var_top_zero_optimized(&r_i),
                        || {
                            rayon::join(
                                || poly_C.bound_poly_var_top_zero_optimized(&r_i),
                                || poly_D.bound_poly_var_top_zero_optimized(&r_i),
                            )
                        },
                    )
                },
            );
        }

        (
            SumcheckInstanceProof::new(polys),
            r,
            vec![poly_A[0], poly_B[0], poly_C[0], poly_D[0]],
        )
    }

    #[tracing::instrument(skip_all, name = "Spartan2::sumcheck::prove_spartan_quadratic")]
    // A fork of `prove_quad` with the 0th round unrolled from the rest of the
    // for loop. This allows us to pass in `W` and `X` as references instead of
    // passing them in as a single `MultilinearPolynomial`, which would require
    // an expensive concatenation. We defer the actual instantation of a
    // `MultilinearPolynomial` to the end of the 0th round.
    pub fn prove_spartan_quadratic<P: IndexablePoly<F>>(
        claim: &F,
        num_rounds: usize,
        poly_A: &mut DensePolynomial<F>,
        W: &P,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, Vec<F>) {
        let mut r: Vec<F> = Vec::with_capacity(num_rounds);
        let mut polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
        let mut claim_per_round = *claim;

        /*          Round 0 START         */

        let len = poly_A.len() / 2;
        assert_eq!(len, W.len());

        let poly = {
            // eval_point_0 = \sum_i A[i] * B[i]
            // where B[i] = W[i] for i in 0..len
            let eval_point_0: F = (0..len)
                .into_par_iter()
                .map(|i| {
                    if poly_A[i].is_zero() || W[i].is_zero() {
                        F::zero()
                    } else {
                        poly_A[i] * W[i]
                    }
                })
                .sum();
            // eval_point_2 = \sum_i (2 * A[len + i] - A[i]) * (2 * B[len + i] - B[i])
            // where B[i] = W[i] for i in 0..len, B[len] = 1, and B[i] = 0 for i > len
            let mut eval_point_2: F = (1..len)
                .into_par_iter()
                .map(|i| {
                    if W[i].is_zero() {
                        F::zero()
                    } else {
                        let poly_A_bound_point = poly_A[len + i] + poly_A[len + i] - poly_A[i];
                        let poly_B_bound_point = -W[i];
                        mul_0_optimized(&poly_A_bound_point, &poly_B_bound_point)
                    }
                })
                .sum();
            eval_point_2 += mul_0_optimized(
                &(poly_A[len] + poly_A[len] - poly_A[0]),
                &(F::from_u64(2).unwrap() - W[0]),
            );

            let evals = [eval_point_0, claim_per_round - eval_point_0, eval_point_2];
            UniPoly::from_evals(&evals)
        };

        // append the prover's message to the transcript
        poly.append_to_transcript(b"poly", transcript);

        //derive the verifier's challenge for the next round
        let r_i: F = transcript.challenge_scalar(b"challenge_nextround");
        r.push(r_i);
        polys.push(poly.compress());

        // Set up next round
        claim_per_round = poly.evaluate(&r_i);

        // bound all tables to the verifier's challenge
        let (_, mut poly_B) = rayon::join(
            || poly_A.bound_poly_var_top_zero_optimized(&r_i),
            || {
                // Simulates `poly_B.bound_poly_var_top(&r_i)`
                // We need to do this because we don't actually have
                // a `MultilinearPolynomial` instance for `poly_B` yet,
                // only the constituents of its (Lagrange basis) coefficients
                // `W` and `X`.
                let zero = F::zero();
                let one = [F::one()];
                let W_iter = (0..W.len()).into_par_iter().map(move |i| &W[i]);
                let Z_iter = W_iter
                    .chain(one.par_iter())
                    .chain(rayon::iter::repeatn(&zero, len));
                let left_iter = Z_iter.clone().take(len);
                let right_iter = Z_iter.skip(len).take(len);
                let B = left_iter
                    .zip(right_iter)
                    .map(|(a, b)| if *a == *b { *a } else { *a + r_i * (*b - *a) })
                    .collect();
                DensePolynomial::new(B)
            },
        );

        /*          Round 0 END          */

        for _ in 1..num_rounds {
            let poly = {
                let (eval_point_0, eval_point_2) =
                    Self::compute_eval_points_spartan_quadratic(poly_A, &poly_B);

                let evals = [eval_point_0, claim_per_round - eval_point_0, eval_point_2];
                UniPoly::from_evals(&evals)
            };

            // append the prover's message to the transcript
            poly.append_to_transcript(b"poly", transcript);

            //derive the verifier's challenge for the next round
            let r_i: F = transcript.challenge_scalar(b"challenge_nextround");

            r.push(r_i);
            polys.push(poly.compress());

            // Set up next round
            claim_per_round = poly.evaluate(&r_i);

            // bound all tables to the verifier's challenege
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
pub struct SumcheckInstanceProof<F: JoltField> {
    compressed_polys: Vec<CompressedUniPoly<F>>,
}

impl<F: JoltField> SumcheckInstanceProof<F> {
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
            poly.append_to_transcript(b"poly", transcript);

            //derive the verifier's challenge for the next round
            let r_i = transcript.challenge_scalar(b"challenge_nextround");

            r.push(r_i);

            // evaluate the claimed degree-ell polynomial at r_i
            e = poly.evaluate(&r_i);
        }

        Ok((e, r))
    }
}

pub mod bench {
    use super::*;
    use crate::poly::dense_mlpoly::DensePolynomial;
    use crate::poly::eq_poly::EqPolynomial;
    use crate::subprotocols::sumcheck::{CubicSumcheckParams, SumcheckInstanceProof};
    use crate::utils::index_to_field_bitvector;
    use ark_bn254::Fr;
    use ark_std::{rand::Rng, test_rng, UniformRand};
    use criterion::black_box;

    pub fn sumcheck_bench(
        group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    ) {
        let num_vars = 16;
        let mut rng = test_rng();

        // PLAIN
        let r1 = vec![Fr::rand(&mut rng); num_vars];
        let r2 = vec![Fr::rand(&mut rng); num_vars];
        let r3 = vec![Fr::rand(&mut rng); num_vars];
        let eq1 = DensePolynomial::new(EqPolynomial::new(r1).evals());
        let left = DensePolynomial::new(EqPolynomial::new(r2).evals());
        let right = DensePolynomial::new(EqPolynomial::new(r3).evals());
        let params = CubicSumcheckParams::new_prod(
            vec![eq1.clone()],
            vec![left.clone()],
            right.clone(),
            num_vars,
        );

        let mut claim = Fr::zero();
        for i in 0..num_vars {
            let eval1 = eq1.evaluate(&index_to_field_bitvector(i, num_vars));
            let eval2 = left.evaluate(&index_to_field_bitvector(i, num_vars));
            let eval3 = right.evaluate(&index_to_field_bitvector(i, num_vars));

            claim += eval1 * eval2 * eval3;
        }

        let coeffs = vec![Fr::one()];

        group.bench_function("sumcheck unbatched 2^16", |b| {
            b.iter(|| {
                let mut transcript = ProofTranscript::new(b"test_transcript");
                let params = black_box(params.clone());
                let (_proof, _r, _evals) = SumcheckInstanceProof::prove_cubic_batched(
                    &claim,
                    params,
                    &coeffs,
                    &mut transcript,
                );
            })
        });

        // FLAGGED
        let num_leaves = 1 << num_vars;
        let mut vals1 = vec![Fr::rand(&mut rng); num_leaves];
        let mut vals2 = vec![Fr::rand(&mut rng); num_leaves];
        // Set approximately half of the leaves to 1
        for _ in 0..num_vars / 2 {
            let rand_index = rng.gen_range(0..num_leaves);
            vals1[rand_index] = Fr::one();
            vals2[rand_index] = Fr::one();
        }
        let poly_a = DensePolynomial::new(vals1);
        let poly_b = DensePolynomial::new(vals2);
        let r = vec![Fr::rand(&mut rng); num_vars];
        let eq = DensePolynomial::new(EqPolynomial::new(r).evals());
        let params = CubicSumcheckParams::new_prod(
            vec![poly_a.clone()],
            vec![poly_b.clone()],
            eq.clone(),
            num_vars,
        );

        let mut claim = Fr::zero();
        for i in 0..num_vars {
            let eval1 = poly_a.evaluate(&index_to_field_bitvector(i, num_vars));
            let eval2 = poly_b.evaluate(&index_to_field_bitvector(i, num_vars));
            let eval3 = eq.evaluate(&index_to_field_bitvector(i, num_vars));

            claim += eval1 * eval2 * eval3;
        }

        let coeffs = vec![Fr::one()];

        group.bench_function("sumcheck unbatched (ones) 2^16", |b| {
            b.iter(|| {
                let mut transcript = ProofTranscript::new(b"test_transcript");
                let params = black_box(params.clone());
                let (_proof, _r, _evals) = SumcheckInstanceProof::prove_cubic_batched(
                    &claim,
                    params,
                    &coeffs,
                    &mut transcript,
                );
            })
        });

        // BATCHED
        let batch_size = 10;
        let num_vars = 14;

        let r_eq = vec![Fr::rand(&mut rng); num_vars];
        let eq = DensePolynomial::new(EqPolynomial::new(r_eq).evals());

        let mut poly_as = Vec::with_capacity(batch_size);
        let mut poly_bs = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let ra = vec![Fr::rand(&mut rng); num_vars];
            let rb = vec![Fr::rand(&mut rng); num_vars];
            let a = DensePolynomial::new(EqPolynomial::new(ra).evals());
            let b = DensePolynomial::new(EqPolynomial::new(rb).evals());
            poly_as.push(a);
            poly_bs.push(b);
        }
        let params =
            CubicSumcheckParams::new_prod(poly_as.clone(), poly_bs.clone(), eq.clone(), num_vars);
        let coeffs = vec![Fr::rand(&mut rng); batch_size];

        let mut joint_claim = Fr::zero();
        for batch_i in 0..batch_size {
            let mut claim = Fr::zero();
            for var_i in 0..num_vars {
                let eval_a = poly_as[batch_i].evaluate(&index_to_field_bitvector(var_i, num_vars));
                let eval_b = poly_bs[batch_i].evaluate(&index_to_field_bitvector(var_i, num_vars));
                let eval_eq = eq.evaluate(&index_to_field_bitvector(var_i, num_vars));

                claim += eval_a * eval_b * eval_eq;
            }
            joint_claim += coeffs[batch_i] * claim;
        }

        group.bench_function("sumcheck 10xbatched 2^14", |b| {
            b.iter(|| {
                let mut transcript = ProofTranscript::new(b"test_transcript");
                let params = black_box(params.clone());
                let (_proof, _r, _evals) = SumcheckInstanceProof::prove_cubic_batched(
                    &joint_claim,
                    params,
                    &coeffs,
                    &mut transcript,
                );
            })
        });
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::poly::eq_poly::EqPolynomial;
    use ark_bn254::Fr;

    #[test]
    fn flags_special_trivial() {
        let factorial =
            DensePolynomial::new(vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)]);
        let flags = DensePolynomial::new(vec![Fr::one(), Fr::one(), Fr::one(), Fr::one()]);
        let r = vec![Fr::one(), Fr::one()];
        let eq = DensePolynomial::new(EqPolynomial::new(r.clone()).evals());
        let num_rounds = 2;

        let claim = Fr::from(4); // r points eq to the 1,1 eval
        let coeffs = vec![Fr::one(), Fr::zero()];

        let comb_func = |h: &Fr, f: &Fr, eq: &Fr| eq * &(h * f + Fr::one() - f);

        let cubic_sumcheck_params = CubicSumcheckParams::new_flags(
            vec![factorial.clone(), factorial.clone()],
            vec![flags.clone()],
            eq.clone(),
            num_rounds,
        );

        let mut transcript = ProofTranscript::new(b"test_transcript");
        let (proof, prove_randomness, _evals) = SumcheckInstanceProof::prove_cubic_batched(
            &claim,
            cubic_sumcheck_params,
            &coeffs,
            &mut transcript,
        );

        let mut transcript = ProofTranscript::new(b"test_transcript");
        let verify_result = proof.verify(claim, 2, 3, &mut transcript);
        assert!(verify_result.is_ok());

        let (verify_evaluation, verify_randomness) = verify_result.unwrap();
        assert_eq!(prove_randomness, verify_randomness);

        let factorial_eval = factorial.evaluate(prove_randomness.as_slice());
        let flag_eval = flags.evaluate(prove_randomness.as_slice());
        let eq_eval = eq.evaluate(prove_randomness.as_slice());
        let oracle_query = comb_func(&factorial_eval, &flag_eval, &eq_eval);
        assert_eq!(verify_evaluation, oracle_query);
    }

    #[test]
    fn flags_special_non_trivial() {
        // H(r_0, r_1, r_2) = sum_{x \in {0,1} ^ 3}{ eq(r, x) \cdot [flags(x) * h(x) + 1 - flags(x)]}
        // Inside the boolean hypercube H(i) = flags(i) * h(i) + 1 - flags(i)
        // Which means if flags(i) = 1, H(i) = h(i)
        //             if flags(i) = 0, H(i) = 1
        // In reality we'll perform sumcheck to transform H(r_0, r_1, r_2) to evaluations of eq(r_0, r_1, r_2, r_3, r_4, r_5), h(r_3, r_4, r_5), flags(r_3, r_4, r_5)
        // where (r_3, r_4, r_5) are generated over the course of sumcheck.
        // The verifier can check this by computing eq(...), h(...), flags(...) on their own.

        let h = DensePolynomial::new(vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)]);
        let flags = DensePolynomial::new(vec![Fr::one(), Fr::zero(), Fr::one(), Fr::one()]);
        let r = vec![Fr::from(100), Fr::from(200)];
        let eq = DensePolynomial::new(EqPolynomial::new(r.clone()).evals());
        let num_rounds = 2;

        let mut claim = Fr::zero();
        let num_evals = 4;
        let num_vars = 2;
        for i in 0..num_evals {
            use crate::utils::index_to_field_bitvector;

            let h_eval = h.evaluate(&index_to_field_bitvector(i, num_vars));
            let flag_eval = flags.evaluate(&index_to_field_bitvector(i, num_vars));
            let eq_eval = eq.evaluate(&index_to_field_bitvector(i, num_vars));

            claim += eq_eval * (flag_eval * h_eval + Fr::one() - flag_eval);
        }

        let coeffs = vec![Fr::one(), Fr::zero()]; // TODO(sragss): Idk how to make this work in the case of non-one coefficients.

        let comb_func = |h: &Fr, f: &Fr, eq: &Fr| eq * &(h * f + Fr::one() - f);

        let cubic_sumcheck_params = CubicSumcheckParams::new_flags(
            vec![h.clone(), h.clone()],
            vec![flags.clone()],
            eq.clone(),
            num_rounds,
        );

        let mut transcript = ProofTranscript::new(b"test_transcript");
        let (proof, prove_randomness, prove_evals) = SumcheckInstanceProof::prove_cubic_batched(
            &claim,
            cubic_sumcheck_params,
            &coeffs,
            &mut transcript,
        );

        // Prover eval: unwrap and combine
        let (leaf_eval, flag_eval, eq_eval) = prove_evals;
        assert_eq!(leaf_eval.len(), 2);
        assert_eq!(flag_eval.len(), 2);
        let leaf_eval = leaf_eval[0];
        let flag_eval = flag_eval[0];
        let prove_fingerprint_eval = flag_eval * leaf_eval + Fr::one() - flag_eval;
        let prove_eval = eq_eval * (flag_eval * leaf_eval + Fr::one() - flag_eval);

        let mut transcript = ProofTranscript::new(b"test_transcript");
        let verify_result = proof.verify(claim, 2, 3, &mut transcript);
        assert!(verify_result.is_ok());

        let (verify_evaluation, verify_randomness) = verify_result.unwrap();
        assert_eq!(prove_randomness, verify_randomness);
        assert_eq!(verify_evaluation, prove_eval);

        let h_eval = h.evaluate(prove_randomness.as_slice());
        let flag_eval = flags.evaluate(prove_randomness.as_slice());
        let eq_eval = eq.evaluate(prove_randomness.as_slice());
        let oracle_query = comb_func(&h_eval, &flag_eval, &eq_eval);
        assert_eq!(verify_evaluation, oracle_query);

        let fingerprint_oracle_query = flag_eval * h_eval + Fr::one() - flag_eval;
        assert_eq!(prove_fingerprint_eval, fingerprint_oracle_query);
    }
}
