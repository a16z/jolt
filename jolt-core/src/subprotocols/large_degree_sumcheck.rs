use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::field::MontU128;
use crate::{
    field::{JoltField, OptimizedMul},
    poly::{multilinear_polynomial::MultilinearPolynomial, unipoly::UniPoly},
    subprotocols::{
        karatsuba::{coeff_kara_16, coeff_kara_32, coeff_kara_4, coeff_kara_8},
        toom::{eval_toom16, eval_toom4, eval_toom8, FieldMulSmall},
    },
    utils::math::Math,
};

pub fn compute_eq_mle_product_univariate<F: JoltField>(
    mle_product_coeffs: Vec<F>,
    round: usize,
    r_cycle: &[MontU128],
) -> UniPoly<F> {
    let mut univariate_evals: Vec<F> = Vec::with_capacity(mle_product_coeffs.len() + 2);

    // Recall that the eq polynomial is rc + (1 - r)(1 - c), which has constant term 1 - r and slope (2r - 1)
    let eq_coeffs = [
        F::one() - r_cycle[round],
        r_cycle[round] + r_cycle[round] - F::one(),
    ];

    // Constant term
    univariate_evals.push(eq_coeffs[0] * mle_product_coeffs[0]);

    // Middle terms
    let mul_by_evals_0 = mle_product_coeffs[1..]
        .par_iter()
        .map(|x| *x * eq_coeffs[0])
        .collect::<Vec<_>>();
    let mul_by_evals_1 = mle_product_coeffs[..mle_product_coeffs.len() - 1]
        .par_iter()
        .map(|x| *x * eq_coeffs[1])
        .collect::<Vec<_>>();

    univariate_evals.extend(
        (0..mle_product_coeffs.len() - 1)
            .into_par_iter()
            .map(|i| mul_by_evals_0[i] + mul_by_evals_1[i])
            .collect::<Vec<_>>(),
    );

    // Last term
    univariate_evals.push(*mle_product_coeffs.last().unwrap() * eq_coeffs[1]);

    UniPoly {
        coeffs: univariate_evals,
    }
}

#[inline(always)]
pub fn compute_mle_product_coeffs_toom<
    F: FieldMulSmall,
    const D: usize,
    const D_PLUS_ONE: usize,
>(
    mle_vec: &[MultilinearPolynomial<F>],
    round: usize,
    log_T: usize,
    factor: &F,
    E_table: &[Vec<F>],
) -> Vec<F> {
    let evals = (0..(log_T - round - 1).pow2())
        .into_par_iter()
        .map(|j| {
            let j_factor = if round < log_T - 1 {
                factor.mul_1_optimized(E_table[round][j])
            } else {
                *factor
            };

            // let span = tracing::span!(tracing::Level::INFO, "Initialize left and right arrays");
            // let _guard = span.enter();
            let polys = (0..D)
                .map(|i| {
                    if i == 0 {
                        (
                            j_factor.mul_1_optimized(mle_vec[i].get_bound_coeff(j)),
                            j_factor.mul_1_optimized(
                                mle_vec[i].get_bound_coeff(j + mle_vec[i].len() / 2),
                            ),
                        )
                    } else {
                        (
                            mle_vec[i].get_bound_coeff(j),
                            mle_vec[i].get_bound_coeff(j + mle_vec[i].len() / 2),
                        )
                    }
                })
                .collect::<Vec<_>>();

            // drop(_guard);
            // drop(span);

            // let span = tracing::span!(tracing::Level::INFO, "Karatsuba step");
            // let _guard = span.enter();

            let res: [F; D_PLUS_ONE] = match D {
                16 => eval_toom16(polys[..16].try_into().unwrap())[..]
                    .try_into()
                    .unwrap(),
                8 => eval_toom8(polys[..8].try_into().unwrap())[..]
                    .try_into()
                    .unwrap(),
                4 => eval_toom4(polys[..4].try_into().unwrap())[..]
                    .try_into()
                    .unwrap(),
                _ => panic!("Unsupported number of polynomials, got {D} and expected 16, 8, or 4"),
            };

            // drop(_guard);
            // drop(span);

            res
        })
        .reduce(
            || [F::zero(); D_PLUS_ONE],
            |mut running, new| {
                for i in 0..D_PLUS_ONE {
                    running[i] += new[i];
                }
                running
            },
        );

    let univariate_poly = UniPoly::from_evals_toom(&evals);
    univariate_poly.coeffs
}

pub fn compute_mle_product_coeffs_katatsuba<
    F: JoltField,
    const D: usize,
    const D_PLUS_ONE: usize,
>(
    mle_vec: &[MultilinearPolynomial<F>],
    round: usize,
    log_T: usize,
    factor: &F,
    E_table: &[Vec<F>],
) -> Vec<F> {
    let coeffs = (0..(log_T - round - 1).pow2())
        .into_par_iter()
        .map(|j| {
            let j_factor = if round < log_T - 1 {
                factor.mul_1_optimized(E_table[round][j])
            } else {
                *factor
            };

            // let span = tracing::span!(tracing::Level::INFO, "Initialize left and right arrays");
            // let _guard = span.enter();

            let left: [F; D] = core::array::from_fn(|i| {
                // Optimization
                if i < 2 {
                    if i == 0 {
                        return mle_vec[0].get_bound_coeff(j) * j_factor;
                    }

                    if i == 1 {
                        return (mle_vec[0].get_bound_coeff(j + mle_vec[0].len() / 2)
                            - mle_vec[0].get_bound_coeff(j))
                            * j_factor;
                    }
                }

                if i % 2 == 0 {
                    mle_vec[i / 2].get_bound_coeff(j)
                } else {
                    mle_vec[i / 2].get_bound_coeff(j + mle_vec[i / 2].len() / 2)
                        - mle_vec[i / 2].get_bound_coeff(j)
                }
            });

            let right: [F; D] = core::array::from_fn(|i| {
                if i % 2 == 0 {
                    mle_vec[(D + i) / 2].get_bound_coeff(j)
                } else {
                    mle_vec[(D + i) / 2].get_bound_coeff(j + mle_vec[(D + i) / 2].len() / 2)
                        - mle_vec[(D + i) / 2].get_bound_coeff(j)
                }
            });

            // drop(_guard);
            // drop(span);

            // let span = tracing::span!(tracing::Level::INFO, "Karatsuba step");
            // let _guard = span.enter();

            let res: [F; D_PLUS_ONE] = match D {
                32 => coeff_kara_32(
                    &left[..32].try_into().unwrap(),
                    &right[..32].try_into().unwrap(),
                )[..]
                    .try_into()
                    .unwrap(),
                16 => coeff_kara_16(
                    &left[..16].try_into().unwrap(),
                    &right[..16].try_into().unwrap(),
                )[..]
                    .try_into()
                    .unwrap(),
                8 => coeff_kara_8(
                    &left[..8].try_into().unwrap(),
                    &right[..8].try_into().unwrap(),
                )[..]
                    .try_into()
                    .unwrap(),
                4 => coeff_kara_4(
                    &left[..4].try_into().unwrap(),
                    &right[..4].try_into().unwrap(),
                )[..]
                    .try_into()
                    .unwrap(),
                _ => unimplemented!(),
            };

            // drop(_guard);
            // drop(span);

            res
        })
        .reduce(
            || [F::zero(); D_PLUS_ONE],
            |mut running, new| {
                for i in 0..D_PLUS_ONE {
                    running[i] += new[i];
                }
                running
            },
        );

    Vec::from(coeffs)
}
