use crate::{
    field::{JoltField, MulU64WithCarry},
    poly::{
        eq_poly::EqPolynomial, ra_poly::RaPolynomial, split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
};
use num_traits::Zero;

/// Computes the univariate polynomial `g(X) = sum_j eq((r', X, j), r) * prod_i mle_i(X, j)`.
///
/// Note `claim` should equal `g(0) + g(1)`.
pub fn compute_mles_product_sum<F: JoltField>(
    mles: &[RaPolynomial<u8, F>],
    claim: F,
    eq_poly: &GruenSplitEqPolynomial<F>,
) -> UniPoly<F> {
    // Evaluate g(X) / eq(X, r[round]) at [1, 2, ..., |mles| - 1, ∞] using split-eq fold.
    let d = mles.len();
    let current_scalar = eq_poly.get_current_scalar();
    let sum_evals: Vec<F> = eq_poly
        .par_fold_out_in(
            || vec![F::Unreduced::<9>::zero(); d],
            |inner, g, _x_in, e_in| {
                // Build per-g pairs [(p0, p1); D]
                let mut pairs: Vec<(F, F)> = Vec::with_capacity(d);
                for mle in mles.iter() {
                    let p0 = mle.get_bound_coeff(2 * g);
                    let p1 = mle.get_bound_coeff(2 * g + 1);
                    pairs.push((p0, p1));
                }
                // Compute endpoints on U_D into a small Vec<F>
                let mut endpoints = vec![F::zero(); d];
                product_eval_univariate_assign(&pairs, &mut endpoints);
                // Accumulate with unreduced arithmetic
                for k in 0..d {
                    inner[k] += e_in.mul_unreduced::<9>(endpoints[k]);
                }
            },
            |_x_out, e_out, inner| {
                // Reduce inner lanes, scale by e_out (unreduced), return outer acc vector
                let mut out = vec![F::Unreduced::<9>::zero(); d];
                for k in 0..d {
                    let reduced_k = F::from_montgomery_reduce::<9>(inner[k]);
                    out[k] = e_out.mul_unreduced::<9>(reduced_k);
                }
                out
            },
            |mut a, b| {
                for k in 0..d {
                    a[k] += b[k];
                }
                a
            },
        )
        .into_iter()
        .map(|x| F::from_montgomery_reduce::<9>(x) * current_scalar)
        .collect();

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

#[inline]
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

#[inline]
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

#[inline]
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

#[inline(always)]
fn dbl<F: JoltField>(x: F) -> F {
    x + x
}

#[inline(always)]
fn dbl_assign<F: JoltField>(x: &mut F) {
    *x += *x;
}

#[inline(always)]
fn assign<T: Sized>(dst: &mut T, src: T) {
    *dst = src;
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_std::UniformRand;
    use dory::curve::test_rng;
    use rand::rngs::StdRng;
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
