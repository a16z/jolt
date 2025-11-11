use num_traits::Zero;
use std::iter::zip;

use rayon::prelude::*;

use crate::{
    field::{JoltField, MulU64WithCarry},
    poly::{
        eq_poly::EqPolynomial, ra_poly::RaPolynomial, split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    utils::math::Math,
};

/// Computes the univariate polynomial `g(X) = sum_j eq((r', X, j), r) * prod_i mle_i(X, j)`.
///
/// Note `claim` should equal `g(0) + g(1)`.
pub fn compute_mles_product_sum<F: JoltField>(
    mles: &[RaPolynomial<u8, F>],
    claim: F,
    eq_poly: &GruenSplitEqPolynomial<F>,
) -> UniPoly<F> {
    // Split Eq poly optimization using GruenSplitEqPolynomial.
    // See https://eprint.iacr.org/2025/1117.pdf section 5.2.

    // Get the eq polynomial evaluations from the split structure
    // Note: With LowToHigh binding, E_out corresponds to the first half (outer loop)
    // and E_in corresponds to the second half (inner loop)
    let num_x_out = eq_poly.E_out_current_len();
    let num_x_in = eq_poly.E_in_current_len();

    // Get the scaling factor that accumulates eq evaluations for already-bound variables
    let current_scalar = eq_poly.get_current_scalar();

    // Evaluate g(X) / eq(X, r[round]) at [1, 2, ..., |mles| - 1, inf].
    let sum_evals: Vec<F> = if num_x_in == 1 {
        // E_in is fully bound - simplified computation
        let eq_in_eval = eq_poly.E_in_current()[0] * current_scalar;
        let eq_out_evals = eq_poly.E_out_current();

        (0..num_x_out)
            .into_par_iter()
            .map(|j_out| {
                let mut partial_evals = vec![F::zero(); mles.len()];
                let mut mle_eval_pairs = vec![(F::zero(), F::zero()); mles.len()];

                for (i, mle) in mles.iter().enumerate() {
                    let mle_eval_at_0_j = mle.get_bound_coeff(2 * j_out);
                    let mle_eval_at_1_j = mle.get_bound_coeff(2 * j_out + 1);
                    mle_eval_pairs[i] = (mle_eval_at_0_j, mle_eval_at_1_j);
                }

                mle_eval_pairs[0].0 *= eq_in_eval;
                mle_eval_pairs[0].1 *= eq_in_eval;
                product_eval_univariate_accumulate(&mle_eval_pairs, &mut partial_evals);

                partial_evals
                    .into_iter()
                    .map(|v| {
                        let result = v * eq_out_evals[j_out];
                        let unreduced = *result.as_unreduced_ref();
                        unreduced
                    })
                    .collect::<Vec<_>>()
            })
            .fold_with(
                vec![F::Unreduced::<5>::zero(); mles.len()],
                |running, new: Vec<F::Unreduced<4>>| {
                    zip(running, new).map(|(a, b)| a + b).collect()
                },
            )
            .reduce(
                || vec![F::Unreduced::zero(); mles.len()],
                |running, new| zip(running, new).map(|(a, b)| a + b).collect(),
            )
            .into_iter()
            .map(F::from_barrett_reduce)
            .collect()
    } else {
        // General case with both E_in and E_out
        let num_x_in_bits = num_x_in.log_2();
        let eq_in_evals = eq_poly.E_in_current();
        let eq_out_evals = eq_poly.E_out_current();

        (0..num_x_out)
            .into_par_iter()
            .map(|j_out| {
                let mut partial_evals = vec![F::zero(); mles.len()];
                let mut mle_eval_pairs = vec![(F::zero(), F::zero()); mles.len()];

                for j_in in 0..num_x_in {
                    let j = (j_out << num_x_in_bits) | j_in;

                    for (i, mle) in mles.iter().enumerate() {
                        let mle_eval_at_0_j = mle.get_bound_coeff(2 * j);
                        let mle_eval_at_1_j = mle.get_bound_coeff(2 * j + 1);
                        mle_eval_pairs[i] = (mle_eval_at_0_j, mle_eval_at_1_j);
                    }

                    mle_eval_pairs[0].0 *= eq_in_evals[j_in] * current_scalar;
                    mle_eval_pairs[0].1 *= eq_in_evals[j_in] * current_scalar;
                    product_eval_univariate_accumulate(&mle_eval_pairs, &mut partial_evals);
                }

                partial_evals
                    .into_iter()
                    .map(|v| {
                        let result = v * eq_out_evals[j_out];
                        let unreduced = *result.as_unreduced_ref();
                        unreduced
                    })
                    .collect::<Vec<_>>()
            })
            .fold_with(
                vec![F::Unreduced::<5>::zero(); mles.len()],
                |running, new: Vec<F::Unreduced<4>>| {
                    zip(running, new).map(|(a, b)| a + b).collect()
                },
            )
            .reduce(
                || vec![F::Unreduced::zero(); mles.len()],
                |running, new| zip(running, new).map(|(a, b)| a + b).collect(),
            )
            .into_iter()
            .map(F::from_barrett_reduce)
            .collect()
    };

    // Get r[round] from the eq polynomial
    let r_round = eq_poly.get_current_w();
    let eq_eval_at_0 = EqPolynomial::mle(&[F::zero()], &[r_round]);
    let eq_eval_at_1 = EqPolynomial::mle(&[F::one()], &[r_round]);

    // Obtain the eval at 0 from the claim.
    let eval_at_1 = sum_evals[0];
    let eval_at_0 = (claim - eq_eval_at_1 * eval_at_1) / eq_eval_at_0;

    // Interpolate the intermediate polynomial.
    let toom_evals = [&[eval_at_0], &*sum_evals].concat();
    let tmp_coeffs = UniPoly::from_evals_toom(&toom_evals).coeffs;

    // Add in the missing eq(X, r[round]) factor.
    // Note eq(X, r[round]) = (1 - r[round]) + (2r[round] - 1)X.
    let constant_coeff = F::one() - r_round;
    let x_coeff = r_round + r_round - F::one();
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
fn product_eval_univariate_accumulate<F: JoltField>(pairs: &[(F, F)], sums: &mut [F]) {
    match pairs.len() {
        2 => eval_inter2_final_op(pairs.try_into().unwrap(), sums, F::add_assign),
        4 => eval_inter4_final_op(pairs.try_into().unwrap(), sums, F::add_assign),
        8 => eval_inter8_final_op(pairs.try_into().unwrap(), sums, F::add_assign),
        16 => eval_inter16_final_op(pairs.try_into().unwrap(), sums, F::add_assign),
        _ => unimplemented!(),
    }
}

/// Computes the product of `D` linear polynomials on `U_D = [1, 2, ..., D - 1, ∞]`.
///
/// The evaluations on `U_D` are assigned to `evals`.
///
/// Inputs:
/// - `pairs[j] = (p_j(0), p_j(1))`
/// - `evals`: output slice with layout `[1, 2, ..., D - 1, ∞]`
pub fn product_eval_univariate_assign<F: JoltField>(pairs: &[(F, F)], evals: &mut [F]) {
    match pairs.len() {
        2 => eval_inter2_final_op(pairs.try_into().unwrap(), evals, assign),
        3 => eval_inter3_final_op(pairs.try_into().unwrap(), evals, assign),
        4 => eval_inter4_final_op(pairs.try_into().unwrap(), evals, assign),
        8 => eval_inter8_final_op(pairs.try_into().unwrap(), evals, assign),
        16 => eval_inter16_final_op(pairs.try_into().unwrap(), evals, assign),
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

fn eval_inter2_final_op<F: JoltField>(p: &[(F, F); 2], outputs: &mut [F], op: impl Fn(&mut F, F)) {
    op(&mut outputs[0], p[0].1 * p[1].1); // 1
    op(&mut outputs[1], (p[0].1 - p[0].0) * (p[1].1 - p[1].0)); // ∞
}

fn eval_inter3_final_op<F: JoltField>(
    pairs: &[(F, F); 3],
    outputs: &mut [F],
    op: impl Fn(&mut F, F),
) {
    let (a1, a2, a_inf) = eval_inter2(pairs[0], pairs[1]);
    let (b0, b1) = pairs[2];
    let b_inf = b1 - b0;
    let b2 = b1 + b_inf;
    op(&mut outputs[0], a1 * b1);
    op(&mut outputs[1], a2 * b2);
    op(&mut outputs[2], a_inf * b_inf);
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

fn eval_inter4_final_op<F: JoltField>(p: &[(F, F); 4], outputs: &mut [F], op: impl Fn(&mut F, F)) {
    let (a1, a2, a_inf) = eval_inter2(p[0], p[1]);
    let a3 = ex2(&[a1, a2], &a_inf);
    let (b1, b2, b_inf) = eval_inter2(p[2], p[3]);
    let b3 = ex2(&[b1, b2], &b_inf);
    op(&mut outputs[0], a1 * b1); // 1
    op(&mut outputs[1], a2 * b2); // 2
    op(&mut outputs[2], a3 * b3); // 3
    op(&mut outputs[3], a_inf * b_inf); // ∞
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

fn eval_inter8_final_op<F: JoltField>(p: &[(F, F); 8], outputs: &mut [F], op: impl Fn(&mut F, F)) {
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

    op(&mut outputs[0], a1 * b1);
    op(&mut outputs[1], a2 * b2);
    op(&mut outputs[2], a3 * b3);
    op(&mut outputs[3], a4 * b4);
    op(&mut outputs[4], a5 * b5);
    op(&mut outputs[5], a6 * b6);
    op(&mut outputs[6], a7 * b7);
    op(&mut outputs[7], a_inf * b_inf);
}

fn eval_inter16_final_op<F: JoltField>(
    p: &[(F, F); 16],
    outputs: &mut [F],
    op: impl Fn(&mut F, F),
) {
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
        op(&mut outputs[i], av[i]);
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
    // P(9) from f[i]=P(i+1): 8(f[1]+f[7]) + 56(f[3]+f[5]) - 28(f[2]+f[6]) - 70 f[4] - f[0] + f_inf40320
    let a1: F::Unreduced<4> = *f[1].as_unreduced_ref() + f[7].as_unreduced_ref();
    let mut pos_acc: F::Unreduced<5> = a1.mul_u64_w_carry::<5>(8);
    let a2: F::Unreduced<4> = *f[3].as_unreduced_ref() + f[5].as_unreduced_ref();
    pos_acc += a2.mul_u64_w_carry::<5>(56);
    pos_acc += f_inf40320.mul_u64_unreduced(1);

    let n1: F::Unreduced<4> = *f[2].as_unreduced_ref() + f[6].as_unreduced_ref();
    let mut neg_acc: F::Unreduced<5> = n1.mul_u64_w_carry::<5>(28);
    neg_acc += f[4].as_unreduced_ref().mul_u64_w_carry::<5>(70);
    neg_acc += f[0].mul_u64_unreduced(1);

    let reduced_pos = F::from_barrett_reduce(pos_acc);
    let reduced_neg = F::from_barrett_reduce(neg_acc);

    reduced_pos - reduced_neg
}

#[inline]
fn dbl<F: JoltField>(x: F) -> F {
    x + x
}

fn dbl_assign<F: JoltField>(x: &mut F) {
    *x += *x;
}

fn assign<T: Sized>(dst: &mut T, src: T) {
    *dst = src;
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_std::{test_rng, UniformRand};
    use std::array::from_fn;

    use crate::{
        field::JoltField,
        poly::{
            dense_mlpoly::DensePolynomial,
            eq_poly::EqPolynomial,
            multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialEvaluation},
            ra_poly::RaPolynomial,
            split_eq_poly::GruenSplitEqPolynomial,
        },
        subprotocols::mles_product_sum::compute_mles_product_sum,
    };

    #[test]
    fn test_compute_mles_product_sum_with_2_mles() {
        const N_MLE: usize = 2;
        let mut rng = &mut test_rng();
        let r_whole = [<Fr as JoltField>::Challenge::rand(&mut rng)];
        let r: &[<Fr as JoltField>::Challenge; 1] = &r_whole;
        let mles: [_; N_MLE] = from_fn(|_| random_mle(1, rng));
        let claim = gen_product_mle(&mles).evaluate(r);

        let r_whole = [<Fr as JoltField>::Challenge::rand(&mut rng)];
        let challenge: &[<Fr as JoltField>::Challenge; 1] = &r_whole;
        let mle_challenge_product = mles.iter().map(|p| p.evaluate(challenge)).product::<Fr>();
        let eval = EqPolynomial::mle(challenge, r) * mle_challenge_product;
        let mles = mles.map(RaPolynomial::RoundN);

        let eq_poly = GruenSplitEqPolynomial::new(r, BindingOrder::LowToHigh);
        let sum_poly = compute_mles_product_sum(&mles, claim, &eq_poly);

        assert_eq!(eval, sum_poly.evaluate(&challenge[0]));
    }

    #[test]
    fn test_compute_mles_product_sum_with_4_mles() {
        const N_MLE: usize = 4;
        let mut rng = &mut test_rng();
        let r_whole = [<Fr as JoltField>::Challenge::random(&mut rng)];
        let r: &[<Fr as JoltField>::Challenge; 1] = &r_whole;
        let mles: [_; N_MLE] = from_fn(|_| random_mle(1, rng));
        let claim = gen_product_mle(&mles).evaluate(r);
        let r_whole = [<Fr as JoltField>::Challenge::rand(&mut rng)];
        let challenge: &[<Fr as JoltField>::Challenge; 1] = &r_whole;
        let mle_challenge_product = mles.iter().map(|p| p.evaluate(challenge)).product::<Fr>();
        let eval = EqPolynomial::mle(challenge, r) * mle_challenge_product;
        let mles = mles.map(RaPolynomial::RoundN);

        let eq_poly = GruenSplitEqPolynomial::new(r, BindingOrder::LowToHigh);
        let sum_poly = compute_mles_product_sum(&mles, claim, &eq_poly);

        assert_eq!(eval, sum_poly.evaluate(&challenge[0]));
    }

    #[test]
    fn test_compute_mles_product_sum_with_8_mles() {
        const N_MLE: usize = 8;
        let mut rng = &mut test_rng();
        let r_whole = [<Fr as JoltField>::Challenge::random(&mut rng)];
        let r: &[<Fr as JoltField>::Challenge; 1] = &r_whole;
        let mles: [_; N_MLE] = from_fn(|_| random_mle(1, rng));
        let claim = gen_product_mle(&mles).evaluate(r);
        let r_whole = [<Fr as JoltField>::Challenge::rand(&mut rng)];
        let challenge: &[<Fr as JoltField>::Challenge; 1] = &r_whole;
        let mle_challenge_product = mles.iter().map(|p| p.evaluate(challenge)).product::<Fr>();
        let eval = EqPolynomial::mle(challenge, r) * mle_challenge_product;
        let mles = mles.map(RaPolynomial::RoundN);

        let eq_poly = GruenSplitEqPolynomial::new(r, BindingOrder::LowToHigh);
        let sum_poly = compute_mles_product_sum(&mles, claim, &eq_poly);

        assert_eq!(eval, sum_poly.evaluate(&challenge[0]));
    }

    #[test]
    fn test_compute_mles_product_sum_with_16_mles() {
        const N_MLE: usize = 16;
        let mut rng = &mut test_rng();
        let r_whole = [<Fr as JoltField>::Challenge::random(&mut rng)];
        let r: &[<Fr as JoltField>::Challenge; 1] = &r_whole;
        let mles: [_; N_MLE] = from_fn(|_| random_mle(1, rng));
        let claim = gen_product_mle(&mles).evaluate(r);
        let r_whole = [<Fr as JoltField>::Challenge::random(&mut rng)];
        let challenge: &[<Fr as JoltField>::Challenge; 1] = &r_whole;
        let mle_challenge_product = mles.iter().map(|p| p.evaluate(challenge)).product::<Fr>();
        let eval = EqPolynomial::mle(challenge, r) * mle_challenge_product;
        let mles = mles.map(RaPolynomial::RoundN);

        let eq_poly = GruenSplitEqPolynomial::new(r, BindingOrder::LowToHigh);
        let sum_poly = compute_mles_product_sum(&mles, claim, &eq_poly);

        assert_eq!(eval, sum_poly.evaluate(&challenge[0]));
    }

    fn random_mle(n_vars: usize, rng: &mut impl rand::Rng) -> MultilinearPolynomial<Fr> {
        let values: Vec<Fr> = (0..(1 << n_vars)).map(|_| Fr::random(rng)).collect();
        MultilinearPolynomial::LargeScalars(DensePolynomial::new(values))
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
