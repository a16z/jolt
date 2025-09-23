use std::iter::zip;

use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial, multilinear_polynomial::MultilinearPolynomial, unipoly::UniPoly,
    },
};

/// Computes the univariate polynomial `g(X) = sum_j correction_factor * eq((X, j), r) * prod_i mle_i(X, j)`.
///
/// Inputs:
/// - `claim` should equal `g(0) + g(1)`.
/// - `eq_evals[j]` should store `eq(j, r[1..])`.
/// - `r0` should equal `r[0]`.
pub fn compute_mles_product_sum<F: JoltField>(
    mles: &[MultilinearPolynomial<F>],
    claim: F,
    r0: F,
    eq_evals: &[F],
    correction_factor: F,
    log_sum_n_terms: u32,
) -> UniPoly<F> {
    let sum_n_terms = 1 << log_sum_n_terms;

    let chunk_size = 512;

    // Evaluate sum_j eq(j, r[1..]) * prod_i mle_i(X, j) at [1, 2, ..., |mles| - 1, inf].
    let mut sum_evals = (0..sum_n_terms)
        .into_par_iter()
        .chunks(chunk_size)
        .map(|j_chunk| {
            let mut sums = vec![F::zero(); mles.len()];
            let mut mle_eval_pair_vec = vec![(F::zero(), F::zero()); mles.len()];

            for j in j_chunk {
                for (i, mle) in mles.iter().enumerate() {
                    let mle_eval_at_0_j = mle.get_bound_coeff(j);
                    let mle_eval_at_1_j = mle.get_bound_coeff(j + sum_n_terms);
                    mle_eval_pair_vec[i] = (mle_eval_at_0_j, mle_eval_at_1_j);
                }

                let eq_eval_at_j_r = eq_evals[j];
                mle_eval_pair_vec[0].0 *= eq_eval_at_j_r;
                mle_eval_pair_vec[0].1 *= eq_eval_at_j_r;

                product_eval_univariate(&mle_eval_pair_vec, &mut sums);
            }

            sums
        })
        .reduce(
            || vec![F::zero(); mles.len()],
            |sums_a, sums_b| zip(sums_a, sums_b).map(|(a, b)| a + b).collect(),
        );

    // Apply correction factor.
    sum_evals
        .iter_mut()
        .for_each(|eval| *eval *= correction_factor);

    let eq_eval_at_0 = EqPolynomial::mle(&[F::zero()], &[r0]);
    let eq_eval_at_1 = EqPolynomial::mle(&[F::one()], &[r0]);

    // Obtain the eval at 0 from the claim.
    let eval_at_1 = sum_evals[0];
    let eval_at_0 = (claim - eq_eval_at_1 * eval_at_1) / eq_eval_at_0;

    // Interpolate the intermediate polynomial.
    let toom_evals = [&[eval_at_0], &*sum_evals].concat();
    let tmp_coeffs = UniPoly::from_evals_toom(&toom_evals).coeffs;

    // Add in the missing eq(X, r[0]) factor.
    // Note eq(X, r[0]) = (1 - r[0]) + (2r[0] - 1)X.
    let constant_coeff = F::one() - r0;
    let x_coeff = r0 + r0 - F::one();
    let mut coeffs = vec![F::zero(); tmp_coeffs.len() + 1];
    for (i, coeff) in tmp_coeffs.into_iter().enumerate() {
        coeffs[i] += coeff * constant_coeff;
        coeffs[i + 1] += coeff * x_coeff;
    }

    UniPoly::from_coeff(coeffs)
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
        2 => eval_inter2_final_accumulate(pairs.try_into().unwrap(), sums),
        4 => eval_inter4_final_accumulate(pairs.try_into().unwrap(), sums),
        8 => eval_inter8_final_accumulate(pairs.try_into().unwrap(), sums),
        16 => eval_inter16_final_accumulate(pairs.try_into().unwrap(), sums),
        _ => unimplemented!(),
    }
}

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

pub fn eval_inter2_final_accumulate<F: JoltField>(pairs: &[(F, F); 2], sums: &mut [F]) {
    sums[0] += pairs[0].1 * pairs[1].1; // 1
    sums[1] += (pairs[0].1 - pairs[0].0) * (pairs[1].1 - pairs[1].0); // ∞
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
        let f_inf6 = f_inf.mul_u64(6);
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
    #[inline]
    fn batch_helper<F: JoltField>(f0: F, f1: F, f2: F, f3: F, f_inf: F) -> (F, F, F) {
        let f_inf6 = f_inf.mul_u64(6);
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
    fn batch_helper<F: JoltField>(vals: &[F; 9]) -> [F; 16] {
        let mut f = [F::zero(); 16]; // f[1, ..., 15, inf]
        f[..8].copy_from_slice(&vals[..8]);
        f[15] = vals[8];
        let f_inf40320 = vals[8].mul_u64(40320);
        for i in 0..7 {
            f[8 + i] = ex8(&f[i..i + 8].try_into().unwrap(), f_inf40320);
        }
        f
    }
    let a = eval_inter8(unsafe { *(p[0..8].as_ptr() as *const [(F, F); 8]) });
    let mut av = batch_helper(&a);
    let b = eval_inter8(unsafe { *(p[8..16].as_ptr() as *const [(F, F); 8]) });
    let bv = batch_helper(&b);
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
    let a1 = f[1] + f[7];
    let a2 = f[3] + f[5];
    let n1 = f[2] + f[6];
    let n2 = f[4];
    let n3 = f[0];
    F::linear_combination_i64(
        &[(a1, 8), (a2, 56)],
        &[(n1, 28), (n2, 70)],
        &[f_inf40320], // positive add terms
        &[n3],         // negative add terms
    )
}

#[inline]
fn dbl<F: JoltField>(x: F) -> F {
    x + x
}

fn dbl_assign<F: JoltField>(x: &mut F) {
    *x += *x;
}

#[cfg(test)]
mod tests {
    use std::array::from_fn;

    use ark_bn254::Fr;
    use dory::curve::test_rng;
    use rand::{rngs::StdRng, Rng};

    use crate::{
        poly::{
            dense_mlpoly::DensePolynomial,
            eq_poly::EqPolynomial,
            multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        },
        subprotocols::mles_product_sum::compute_mles_product_sum,
    };

    #[test]
    fn test_compute_mles_product_sum_with_2_mles() {
        const N_MLE: usize = 2;
        let rng = &mut test_rng();
        let r: &[Fr; 1] = &rng.gen();
        let mles: [_; N_MLE] = from_fn(|_| random_mle(1, rng));
        let correction_factor = rng.gen();
        let claim = correction_factor * gen_product_mle(&mles).evaluate(r);
        let r_prime: &[Fr; 1] = &rng.gen();
        let mle_challenge_product = mles.iter().map(|mle| mle.evaluate(r_prime)).product::<Fr>();
        let eval = correction_factor * EqPolynomial::mle(r_prime, r) * mle_challenge_product;
        let log_sum_n_terms = 0;

        let sum_poly = compute_mles_product_sum(
            &mles,
            claim,
            r[0],
            &EqPolynomial::evals(&[]),
            correction_factor,
            log_sum_n_terms,
        );

        assert_eq!(eval, sum_poly.evaluate(&r_prime[0]));
    }

    #[test]
    fn test_compute_mles_product_sum_with_4_mles() {
        const N_MLE: usize = 4;
        let rng = &mut test_rng();
        let r: &[Fr; 1] = &rng.gen();
        let mles: [_; N_MLE] = from_fn(|_| random_mle(1, rng));
        let correction_factor = rng.gen();
        let claim = correction_factor * gen_product_mle(&mles).evaluate(r);
        let r_prime: &[Fr; 1] = &rng.gen();
        let mle_challenge_product = mles.iter().map(|mle| mle.evaluate(r_prime)).product::<Fr>();
        let eval = correction_factor * EqPolynomial::mle(r_prime, r) * mle_challenge_product;
        let log_sum_n_terms = 0;

        let sum_poly = compute_mles_product_sum(
            &mles,
            claim,
            r[0],
            &EqPolynomial::evals(&[]),
            correction_factor,
            log_sum_n_terms,
        );

        assert_eq!(eval, sum_poly.evaluate(&r_prime[0]));
    }

    #[test]
    fn test_compute_mles_product_sum_with_8_mles() {
        const N_MLE: usize = 8;
        let rng = &mut test_rng();
        let r: &[Fr; 1] = &rng.gen();
        let mles: [_; N_MLE] = from_fn(|_| random_mle(1, rng));
        let correction_factor = rng.gen();
        let claim = correction_factor * gen_product_mle(&mles).evaluate(r);
        let r_prime: &[Fr; 1] = &rng.gen();
        let mle_challenge_product = mles.iter().map(|mle| mle.evaluate(r_prime)).product::<Fr>();
        let eval = correction_factor * EqPolynomial::mle(r_prime, r) * mle_challenge_product;
        let log_sum_n_terms = 0;

        let sum_poly = compute_mles_product_sum(
            &mles,
            claim,
            r[0],
            &EqPolynomial::evals(&[]),
            correction_factor,
            log_sum_n_terms,
        );

        assert_eq!(eval, sum_poly.evaluate(&r_prime[0]));
    }

    #[test]
    fn test_compute_mles_product_sum_with_16_mles() {
        const N_MLE: usize = 16;
        let rng = &mut test_rng();
        let r: &[Fr; 1] = &rng.gen();
        let mles: [_; N_MLE] = from_fn(|_| random_mle(1, rng));
        let correction_factor = rng.gen();
        let claim = correction_factor * gen_product_mle(&mles).evaluate(r);
        let r_prime: &[Fr; 1] = &rng.gen();
        let mle_challenge_product = mles.iter().map(|mle| mle.evaluate(r_prime)).product::<Fr>();
        let eval = correction_factor * EqPolynomial::mle(r_prime, r) * mle_challenge_product;
        let log_sum_n_terms = 0;

        let sum_poly = compute_mles_product_sum(
            &mles,
            claim,
            r[0],
            &EqPolynomial::evals(&[]),
            correction_factor,
            log_sum_n_terms,
        );

        assert_eq!(eval, sum_poly.evaluate(&r_prime[0]));
    }

    fn random_mle(n_vars: usize, rng: &mut StdRng) -> MultilinearPolynomial<Fr> {
        MultilinearPolynomial::LargeScalars(DensePolynomial::random(n_vars, rng))
    }

    /// Generates MLE `p(x) = sum_j eq(j, x) * prod_i mle_i(j)`.
    fn gen_product_mle(mles: &[MultilinearPolynomial<Fr>]) -> MultilinearPolynomial<Fr> {
        let n_vars = mles[0].get_num_vars();
        assert!(mles.iter().all(|mle| mle.get_num_vars() == n_vars));
        let res = (0..1 << n_vars)
            .map(|i| mles.iter().map(|mle| mle.get_bound_coeff(i)).product())
            .collect::<Vec<Fr>>();
        res.into()
    }
}
