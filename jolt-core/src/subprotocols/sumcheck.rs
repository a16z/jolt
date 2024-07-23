#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::field::JoltField;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::unipoly::{CompressedUniPoly, UniPoly};
use crate::r1cs::spartan::IndexablePoly;
use crate::utils::errors::ProofVerifyError;
use crate::utils::mul_0_optimized;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::{AppendToTranscript, ProofTranscript};
use crate::field::OptimizedMul;
use ark_serialize::*;
use rayon::prelude::*;

/// Batched cubic sumcheck used in grand products
pub trait BatchedCubicSumcheck<F: JoltField>: Sync {
    fn num_rounds(&self) -> usize;
    fn bind(&mut self, eq_poly: &mut DensePolynomial<F>, r: &F);
    fn compute_cubic(
        &self,
        coeffs: &[F],
        eq_poly: &DensePolynomial<F>,
        previous_round_claim: F,
    ) -> UniPoly<F>;
    fn final_claims(&self) -> (Vec<F>, Vec<F>);

    #[tracing::instrument(skip_all, name = "BatchedCubicSumcheck::prove_sumcheck")]
    fn prove_sumcheck(
        &mut self,
        claim: &F,
        coeffs: &[F],
        eq_poly: &mut DensePolynomial<F>,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F>, Vec<F>, (Vec<F>, Vec<F>)) {
        debug_assert_eq!(eq_poly.get_num_vars(), self.num_rounds());

        let mut previous_claim = *claim;
        let mut r: Vec<F> = Vec::new();
        let mut cubic_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        for _round in 0..self.num_rounds() {
            let cubic_poly = self.compute_cubic(coeffs, eq_poly, previous_claim);
            // append the prover's message to the transcript
            cubic_poly.append_to_transcript(transcript);
            //derive the verifier's challenge for the next round
            let r_j = transcript.challenge_scalar();

            r.push(r_j);
            // bind polynomials to verifier's challenge
            self.bind(eq_poly, &r_j);

            previous_claim = cubic_poly.evaluate(&r_j);
            cubic_polys.push(cubic_poly.compress());
        }

        debug_assert_eq!(eq_poly.len(), 1);

        (
            SumcheckInstanceProof::new(cubic_polys),
            r,
            self.final_claims(),
        )
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
            let span = tracing::span!(tracing::Level::TRACE, "round");
            let _enter = span.enter();
            // Vector storing evaluations of combined polynomials g(x) = P_0(x) * ... P_{num_polys} (x)
            // for points {0, ..., |g(x)|}
            // let mut eval_points = vec![F::zero(); combined_degree + 1];

            let mle_half = polys[0].len() / 2;

            let eval_points: Vec<F> = (0..mle_half)
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
                    accum[0] = comb_func(&params_zero);

                    // TODO(#28): Can be computed from prev_round_claim - eval_point_0
                    let params_one: Vec<F> = polys
                        .iter()
                        .map(|poly| poly[mle_half + poly_term_i])
                        .collect();
                    accum[1] = comb_func(&params_one);
                    // println!("params_one {params_one:?}");

                    // D_n(index, r) = D_{n-1}[half + index] + r * (D_{n-1}[half + index] - D_{n-1}[index])
                    // D_n(index, 0) = D_{n-1}[LOW]
                    // D_n(index, 1) = D_{n-1}[HIGH]
                    // D_n(index, 2) = D_{n-1}[HIGH] + (D_{n-1}[HIGH] - D_{n-1}[LOW])
                    //      D_n(index, 2) = (1 - 2) * D[LOW] + 2 * D[HIGH] = 2 * D[HIGH] - D[LOW]
                    // D_n(index, 3) = D_{n-1}[HIGH] + (D_{n-1}[HIGH] - D_{n-1}[LOW]) + (D_{n-1}[HIGH] - D_{n-1}[LOW])
                    //      D_n(index, 2) = (1 - 3) * D[LOW] + 3 * D[HIGH] = D[HIGH] + 2 * [ D[HIGH] - D[LOW] ]
                    // ...

                    // D_n(index, r) = D_{n-1}[half + index] + r * (D_{n-1}[half + index] - D_{n-1}[index])
                    // D_n(index, 0) = D_{n-1}[LOW]
                    // D_n(index, 1) = D_{n-1}[HIGH]
                    // z_0 = Binius((1,0))
                    // z_1 = Binius((1,1))
                    // D_n(index, z_0) = (1 + z_0) * D[LOW] + z_0 * D[HIGH]
                    // D_n(index, z_1) =
                    // let mut existing_term = params_one;
                    for eval_i in 2..(combined_degree + 1) {
                        let mut poly_evals = vec![F::zero(); polys.len()];
                        for poly_i in 0..polys.len() {
                            // let poly = &polys[poly_i];
                            // poly_evals[poly_i] = existing_term[poly_i]
                            //     + poly[mle_half + poly_term_i]
                            //     - poly[poly_term_i];
                            let poly = &polys[poly_i];
                            let r = F::from_u64(eval_i as u64).unwrap();
                            poly_evals[poly_i] =  poly[poly_term_i].mul_01_optimized(F::one() - r) + mul_0_optimized(&poly[mle_half + poly_term_i], &r);
                        }

                        accum[eval_i] = comb_func(&poly_evals);
                        // existing_term = poly_evals;
                    }
                    accum
                }).reduce(
                    || vec![F::zero(); combined_degree + 1],
                    |mut accum, item| {
                        for (i, val) in item.iter().enumerate() {
                            accum[i] += *val;
                        }
                        accum
                    }
                );

            // eval_points
            //     .par_iter_mut()
            //     .enumerate()
            //     .for_each(|(poly_i, eval_point)| {
            //         *eval_point = accum
            //             .par_iter()
            //             .take(mle_half)
            //             .map(|mle| mle[poly_i])
            //             .sum::<F>();
            //     });


            // println!("evals: {eval_points:?}");
            let round_uni_poly = UniPoly::from_evals(&eval_points);
            // println!(
                // "[{_round}] eval(0) + eval(1) = {:?}",
                // eval_points[0] + eval_points[1]
            // );

            // append the prover's message to the transcript
            // println!("[{_round}] appending to transcript: {round_uni_poly:?}");
            round_uni_poly.append_to_transcript(transcript);
            let r_j = transcript.challenge_scalar();
            r.push(r_j);

            // bound all tables to the verifier's challenege
            polys
                .par_iter_mut()
                .for_each(|poly| poly.bound_poly_var_top_zero_optimized(&r_j));
            compressed_polys.push(round_uni_poly.compress());
        }

        polys
            .iter()
            .for_each(|poly| debug_assert_eq!(poly.len(), 1));
        let final_evals = polys.iter().map(|poly| poly[0]).collect();

        (SumcheckInstanceProof::new(compressed_polys), r, final_evals)
    }

    #[tracing::instrument(skip_all, name = "Sumcheck.prove_special")]
    pub fn prove_special(
        _claim: &F,
        num_rounds: usize,
        poly_l: &mut DensePolynomial<F>,
        poly_r: &mut DensePolynomial<F>,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, Vec<F>)
    {
        let num_eval_points = 3;

        let mut r: Vec<F> = Vec::new();
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        let two = F::from_u64(2).unwrap();
        let lhs_point = F::one() - two;

        for _round in 0..num_rounds {
            let span = tracing::span!(tracing::Level::TRACE, "round");
            let _enter = span.enter();

            let mle_half = poly_l.len() / 2;

            let eval_points: (F, F, F) = (0..mle_half)
                .into_par_iter()
                .map(|poly_term_i| {
                    let eval_0 = poly_l[poly_term_i].mul_01_optimized(poly_r[poly_term_i]);
                    let eval_1 = poly_l[mle_half + poly_term_i].mul_01_optimized(poly_r[mle_half + poly_term_i]);
                    let eval_left = poly_l[poly_term_i].mul_01_optimized(lhs_point) + mul_0_optimized(&poly_l[mle_half + poly_term_i], &two);
                    let eval_right= poly_r[poly_term_i].mul_01_optimized(lhs_point) + mul_0_optimized(&poly_r[mle_half + poly_term_i], &two);
                    let eval_2 = eval_left.mul_01_optimized(eval_right);
                    (eval_0, eval_1, eval_2)
                }).reduce(
                    || (F::zero(), F::zero(), F::zero()),
                    |mut accum, item| {
                        accum.0 += item.0;
                        accum.1 += item.1;
                        accum.2 += item.2;
                        accum
                    }
                );

            // println!("evals: {eval_points:?}");
            let round_uni_poly = UniPoly::from_evals(&vec![eval_points.0, eval_points.1, eval_points.2]);
            round_uni_poly.append_to_transcript(transcript);
            let r_j = transcript.challenge_scalar();
            r.push(r_j);

            // bound all tables to the verifier's challenege
            rayon::join(
                || poly_l.bound_poly_var_top_zero_optimized(&r_j),
                || poly_r.bound_poly_var_top_zero_optimized(&r_j),
            );
            compressed_polys.push(round_uni_poly.compress());
        }

        assert_eq!(poly_l.len(), 1);
        assert_eq!(poly_r.len(), 1);
        let final_evals = vec![poly_l[0], poly_r[0]];

        (SumcheckInstanceProof::new(compressed_polys), r, final_evals)
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
        let two = F::from_u64(2 as u64).unwrap();
        let one_plus_two = F::one() + two;
        let three = F::from_u64(3 as u64).unwrap();
        let one_plus_three = F::one() + three;
        let len = poly_A.len() / 2;
        (0..len)
            .into_par_iter()
            .map(|i| {
                // eval 0: bound_func is A(low)
                let eval_point_0 = comb_func(&poly_A[i], &poly_B[i], &poly_C[i], &poly_D[i]);

                let low = i;
                let high = len + i;

                let eval_point_2 = comb_func(
                    &(one_plus_two.mul_01_optimized(poly_A[low]) + two.mul_01_optimized(poly_A[high])),
                    &(one_plus_two.mul_01_optimized(poly_B[low]) + two.mul_01_optimized(poly_B[high])),
                    &(one_plus_two.mul_01_optimized(poly_C[low]) + two.mul_01_optimized(poly_C[high])),
                    &(one_plus_two.mul_01_optimized(poly_D[low]) + two.mul_01_optimized(poly_D[high])),
                );
                let eval_point_3 = comb_func(
                    &(one_plus_three.mul_01_optimized(poly_A[low]) + three.mul_01_optimized(poly_A[high])),
                    &(one_plus_three.mul_01_optimized(poly_B[low]) + three.mul_01_optimized(poly_B[high])),
                    &(one_plus_three.mul_01_optimized(poly_C[low]) + three.mul_01_optimized(poly_C[high])),
                    &(one_plus_three.mul_01_optimized(poly_D[low]) + three.mul_01_optimized(poly_D[high])),
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
                // println!("EVALS: {evals:?}");
                UniPoly::from_evals(&evals)
            };

            // append the prover's message to the transcript
            poly.append_to_transcript(transcript);

            //derive the verifier's challenge for the next round
            let r_i = transcript.challenge_scalar();
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

    // Sam plan
    // Flatten W
    // Evaluate {0,1,2} normally

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
        const NEW: bool = true;

        if NEW {
            let len = poly_A.len() / 2;
            let W_iter = (0..W.len()).into_par_iter().map(move |i| &W[i]);
            let zero = F::zero();
            let one = [F::one()];
            let Z_iter = W_iter
                .chain(one.par_iter())
                .chain(rayon::iter::repeatn(&zero, len - 1));
            let flat_z: Vec<F> = Z_iter.cloned().collect();
            let mut poly_z = DensePolynomial::new(flat_z);
            assert_eq!(poly_A.len(), poly_z.len());

            let comb_func = |inputs: &[F]| -> F {
                debug_assert_eq!(inputs.len(), 2);
                inputs[0].mul_01_optimized(inputs[1])
            };
            // let mut polys_vec = vec![poly_A.clone(), poly_z];
            // SumcheckInstanceProof::prove_arbitrary(
            //     claim,
            //     num_rounds,
            //     &mut polys_vec,
            //     comb_func,
            //     2,
            //     transcript,
            // )
            SumcheckInstanceProof::prove_special(
                claim,
                num_rounds,
                // &mut polys_vec,
                &mut poly_A.clone(),
                &mut poly_z,
                // comb_func,
                // 2,
                transcript,
            )
        } else {
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
                // println!("evals: {evals:?}");
                UniPoly::from_evals(&evals)
            };

            // append the prover's message to the transcript
            // println!("[0] appending to transcript: {poly:?}");
            poly.append_to_transcript(transcript);

            //derive the verifier's challenge for the next round
            let r_i: F = transcript.challenge_scalar();
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

            for i in 1..num_rounds {
                let poly = {
                    let (eval_point_0, eval_point_2) =
                        Self::compute_eval_points_spartan_quadratic(poly_A, &poly_B);

                    let evals = [eval_point_0, claim_per_round - eval_point_0, eval_point_2];
                    UniPoly::from_evals(&evals)
                };

                // append the prover's message to the transcript
                // println!("[{i}] appending to transcript: {poly:?}");
                poly.append_to_transcript(transcript);

                //derive the verifier's challenge for the next round
                let r_i: F = transcript.challenge_scalar();

                r.push(r_i);
                polys.push(poly.compress());

                // Set up next round
                claim_per_round = poly.evaluate(&r_i);

                // bound all tables to the verifier's challenege
                rayon::join(
                    || poly_A.bound_poly_var_top_zero_optimized(&r_i),
                    || poly_B.bound_poly_var_top_zero_optimized(&r_i),
                );

                if i == num_rounds - 1 {
                    assert_eq!(poly.evaluate(&r_i), poly_A[0] * poly_B[0]);
                }
            }

            let evals = vec![poly_A[0], poly_B[0]];
            drop_in_background_thread(poly_B);

            (SumcheckInstanceProof::new(polys), r, evals)
        }
    }

    #[tracing::instrument(skip_all, name = "Spartan2::sumcheck::prove_spartan_quadratic")]
    // A fork of `prove_quad` with the 0th round unrolled from the rest of the
    // for loop. This allows us to pass in `W` and `X` as references instead of
    // passing them in as a single `MultilinearPolynomial`, which would require
    // an expensive concatenation. We defer the actual instantation of a
    // `MultilinearPolynomial` to the end of the 0th round.
    pub fn prove_spartan_quadratic_old<P: IndexablePoly<F>>(
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
        poly.append_to_transcript(transcript);

        //derive the verifier's challenge for the next round
        let r_i: F = transcript.challenge_scalar();
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

        for i in 1..num_rounds {
            let poly = {
                let (eval_point_0, eval_point_2) =
                    Self::compute_eval_points_spartan_quadratic(poly_A, &poly_B);

                let evals = [eval_point_0, claim_per_round - eval_point_0, eval_point_2];
                UniPoly::from_evals(&evals)
            };

            // append the prover's message to the transcript
            poly.append_to_transcript(transcript);

            //derive the verifier's challenge for the next round
            let r_i: F = transcript.challenge_scalar();

            r.push(r_i);
            polys.push(poly.compress());

            // Set up next round
            claim_per_round = poly.evaluate(&r_i);

            // bound all tables to the verifier's challenege
            rayon::join(
                || poly_A.bound_poly_var_top_zero_optimized(&r_i),
                || poly_B.bound_poly_var_top_zero_optimized(&r_i),
            );

            if i == num_rounds - 1 {
                assert_eq!(poly.evaluate(&r_i), poly_A[0] * poly_B[0]);
            }
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
            // println!("[{i}] hint = eval(0) + eval(1) = {e:?}");
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
            // println!("[{i}] appending to transcript: {poly:?}");
            poly.append_to_transcript(transcript);

            //derive the verifier's challenge for the next round
            let r_i = transcript.challenge_scalar();

            r.push(r_i);

            // evaluate the claimed degree-ell polynomial at r_i
            e = poly.evaluate(&r_i);
        }

        Ok((e, r))
    }
}
