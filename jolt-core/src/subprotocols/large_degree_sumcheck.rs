use std::iter::zip;

use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

use crate::{
    field::JoltField,
    poly::{multilinear_polynomial::MultilinearPolynomial, unipoly::UniPoly},
    utils::math::Math,
};

pub fn compute_eq_mle_product_univariate<F: JoltField>(
    mle_product_coeffs: Vec<F>,
    round: usize,
    r_cycle: &[F],
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

/// Evaluates `g(X) = sum_j eq((0, j), r) * prod_i mle_i(X, j)` for `X` in `[1, 2, ..., deg(g) - 1, ∞]`.
pub fn compute_mle_product_sum<F: JoltField>(
    mle_vec: &[MultilinearPolynomial<F>],
    round: usize,
    log_T: usize,
    E_table: &[Vec<F>],
) -> Vec<F> {
    let sum_n_terms = (log_T - round - 1).pow2();

    let chunk_size = 1024;

    let evals = (0..sum_n_terms)
        .into_par_iter()
        .chunks(chunk_size)
        .map(|j_chunk| {
            let mut sums = vec![F::zero(); mle_vec.len()];
            let mut mle_eval_pair_vec = vec![(F::zero(), F::zero()); mle_vec.len()];

            for j in j_chunk {
                for (i, mle) in mle_vec.iter().enumerate() {
                    let mle_eval_at_0_j = mle.get_bound_coeff(j);
                    let mle_eval_at_1_j = mle.get_bound_coeff(j + sum_n_terms);
                    mle_eval_pair_vec[i] = (mle_eval_at_0_j, mle_eval_at_1_j);
                }

                let eq_eval_at_0_j = E_table[round][j];
                mle_eval_pair_vec[0].0 *= eq_eval_at_0_j;
                mle_eval_pair_vec[0].1 *= eq_eval_at_0_j;

                product_eval_univariate(&mle_eval_pair_vec, &mut sums);
            }

            sums
        })
        .reduce(
            || vec![F::zero(); mle_vec.len()],
            |sums_a, sums_b| zip(sums_a, sums_b).map(|(a, b)| a + b).collect(),
        );

    evals
}

/// Computes the product of `D` linear polynomials on `U_D = [1, 2, ..., D - 1, ∞]`.
///
/// The evaluations on `U_D` are accumulated into `sums`.
///
/// Inputs:
/// - `pairs[j] = (p_j(0), p_j(1))`
/// - `sums`: accumulator with layout `[1, 2, ..., D - 1, ∞]`
fn product_eval_univariate<F: JoltField>(pairs: &[(F, F)], sums: &mut [F]) {
    match pairs.len() {
        4 => eval_inter4_final_accumulate(pairs.try_into().unwrap(), sums),
        8 => eval_inter8_final_accumulate(pairs.try_into().unwrap(), sums),
        16 => eval_inter16_final_accumulate(pairs.try_into().unwrap(), sums),
        _ => unimplemented!(),
    }
}

// d = 2: seed points [1, 2, ∞] from two linear factors
#[inline]
fn eval_inter2<F: JoltField>((p0, p1): (F, F), (q0, q1): (F, F)) -> (F, F, F) {
    let p_inf = p1 - p0;
    let p2 = p_inf + p1;
    let q_inf = q1 - q0;
    let q2 = q_inf + q1;
    let r1 = p1 * q1;
    let r2 = p2 * q2;
    let r_inf = p_inf * q_inf;
    (r1, r2, r_inf)
}

fn eval_inter4<F: JoltField>(p: [(F, F); 4]) -> (F, F, F, F, F) {
    let (a1, a2, a_inf) = eval_inter2(p[0], p[1]);
    let a3 = ex2(&[a1, a2], &a_inf);
    let a4 = ex2(&[a2, a3], &a_inf);
    let (b1, b2, b_inf) = eval_inter2(p[2], p[3]);
    let b3 = ex2(&[b1, b2], &b_inf);
    let b4 = ex2(&[b2, b3], &b_inf);
    (a1 * b1, a2 * b2, a3 * b3, a4 * b4, a_inf * b_inf)
}

fn eval_inter4_final_accumulate<F: JoltField>(p: &[(F, F); 4], sums: &mut [F]) {
    let (a1, a2, a_inf) = eval_inter2(p[0], p[1]);
    let a3 = ex2(&[a1, a2], &a_inf);
    let (b1, b2, b_inf) = eval_inter2(p[2], p[3]);
    let b3 = ex2(&[b1, b2], &b_inf);
    sums[0] += a1 * b1; // 1
    sums[1] += a2 * b2; // 2
    sums[2] += a3 * b3; // 3
    sums[3] += a_inf * b_inf; // ∞
}

fn eval_inter8<F: JoltField>(p: [(F, F); 8]) -> [F; 9] {
    #[inline]
    fn batch_helper<F: JoltField>(f0: F, f1: F, f2: F, f3: F, f_inf: F) -> (F, F, F, F) {
        let f_inf6 = mul6(f_inf);
        let (f4, f5) = ex4_2(&[f0, f1, f2, f3], &f_inf6);
        let (f6, f7) = ex4_2(&[f2, f3, f4, f5], &f_inf6);
        (f4, f5, f6, f7)
    }
    let (a1, a2, a3, a4, a_inf) = eval_inter4(unsafe { *(p[0..4].as_ptr() as *const [(F, F); 4]) });
    let (a5, a6, a7, a8) = batch_helper(a1, a2, a3, a4, a_inf);
    let (b1, b2, b3, b4, b_inf) = eval_inter4(unsafe { *(p[4..8].as_ptr() as *const [(F, F); 4]) });
    let (b5, b6, b7, b8) = batch_helper(b1, b2, b3, b4, b_inf);
    [
        a1 * b1,
        a2 * b2,
        a3 * b3,
        a4 * b4,
        a5 * b5,
        a6 * b6,
        a7 * b7,
        a8 * b8,
        a_inf * b_inf,
    ]
}

fn eval_inter8_final_accumulate<F: JoltField>(p: &[(F, F); 8], sums: &mut [F]) {
    // A direct port of the logic from eval_inter8_final, but accumulating.
    #[inline]
    fn batch_helper<F: JoltField>(f0: F, f1: F, f2: F, f3: F, f_inf: F) -> (F, F, F) {
        let f_inf6 = mul6(f_inf);
        let (f4, f5) = ex4_2(&[f0, f1, f2, f3], &f_inf6);
        let f6 = ex4(&[f2, f3, f4, f5], &f_inf6);
        (f4, f5, f6)
    }
    let (a1, a2, a3, a4, a_inf) = eval_inter4(unsafe { *(p[0..4].as_ptr() as *const [(F, F); 4]) });
    let (a5, a6, a7) = batch_helper(a1, a2, a3, a4, a_inf);
    let (b1, b2, b3, b4, b_inf) = eval_inter4(unsafe { *(p[4..8].as_ptr() as *const [(F, F); 4]) });
    let (b5, b6, b7) = batch_helper(b1, b2, b3, b4, b_inf);

    sums[0] += a1 * b1;
    sums[1] += a2 * b2;
    sums[2] += a3 * b3;
    sums[3] += a4 * b4;
    sums[4] += a5 * b5;
    sums[5] += a6 * b6;
    sums[6] += a7 * b7;
    sums[7] += a_inf * b_inf;
}

#[inline(always)]
fn eval_inter16_final_accumulate<F: JoltField>(p: &[(F, F); 16], sums: &mut [F]) {
    #[inline]
    fn batch_values<F: JoltField>(vals: &[F; 9]) -> [F; 16] {
        let mut f = [F::zero(); 16]; // f[1..15, inf]
        f[..8].copy_from_slice(&vals[..8]);
        f[15] = vals[8];
        let f_inf40320 = vals[8].mul_u64(40320);
        for i in 0..7 {
            f[i] = ex8(&f[i..i + 8].try_into().unwrap(), f_inf40320);
        }
        f
    }
    let a = eval_inter8(unsafe { *(p[0..8].as_ptr() as *const [(F, F); 8]) });
    let b = eval_inter8(unsafe { *(p[8..16].as_ptr() as *const [(F, F); 8]) });
    let mut av = batch_values(&a);
    let bv = batch_values(&b);
    // Include all entries [1..15, inf]
    for i in 0..16 {
        av[i] *= bv[i];
        sums[i] += av[i];
    }
}

#[inline(always)]
fn ex2<F: JoltField>(f: &[F; 2], f_inf: &F) -> F {
    dbl(f[1] + f_inf) - f[0]
}

fn ex4<F: JoltField>(f: &[F; 4], f_inf6: &F) -> F {
    // Natural-grid coeffs for target x+4: [1, -4, 6, -4] and 4!*a4 = 24*a4.
    let mut t = *f_inf6;
    t += f[3];
    t -= f[2];
    t += f[1];
    dbl_assign(&mut t);
    t -= f[2];
    dbl_assign(&mut t);
    t -= f[0];
    t
}

#[inline(always)]
fn ex4_2<F: JoltField>(f: &[F; 4], f_inf6: &F) -> (F, F) {
    let f3m2 = f[3] - f[2];
    let mut f4 = *f_inf6;
    f4 += f3m2;
    f4 += f[1];
    dbl_assign(&mut f4);
    f4 -= f[2];
    dbl_assign(&mut f4);
    f4 -= f[0];

    let mut f5 = f4 - f3m2 + f_inf6;
    dbl_assign(&mut f5);
    f5 -= f[3];
    dbl_assign(&mut f5);
    f5 -= f[1];

    (f4, f5)
}

#[inline(always)]
fn ex8<F: JoltField>(f: &[F; 8], f_inf40320: F) -> F {
    F::linear_combination_i64(
        &[(f[1] + f[7], 8), (f[3] + f[5], 56)],
        &[(f[2] + f[6], 28), (f[4], 70)],
        &[f_inf40320], // positive add terms
        &[f[0]],       // negative add terms
    )
}

#[inline]
fn dbl<F: JoltField>(x: F) -> F {
    x + x
}

#[inline]
fn mul6<F: JoltField>(x: F) -> F {
    x.mul_u64(6)
}

// In-place helpers to reduce temporaries
#[inline(always)]
fn dbl_assign<F: JoltField>(x: &mut F) {
    *x += *x;
}

#[inline(always)]
fn tpl_assign<F: JoltField>(x: &mut F) {
    // 3*x = 2*x + x using the efficient doubling
    let y = *x;
    dbl_assign(x);
    *x += y;
}

mod tests {
    // TODO.
}
