use ark_bn254::Fr;
use ark_std::test_rng;
use criterion::{criterion_group, criterion_main, Criterion};
use jolt_core::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial},
        ra_poly::RaPolynomial,
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::mles_product_sum::compute_mles_product_sum,
    utils::math::Math,
};
use num_traits::Zero;
use rayon::prelude::*;

fn product_eval_univariate_naive_accumulate<F: JoltField>(pairs: &[(F, F)], sums: &mut [F]) {
    let d = pairs.len();
    debug_assert_eq!(sums.len(), d);
    if d == 0 {
        return;
    }
    // Memoize p(1)=p1, then p(2)=p(1)+pinf, p(3)=p(2)+pinf, ...
    let mut cur_vals = Vec::with_capacity(d);
    let mut pinfs = Vec::with_capacity(d);
    for &(p0, p1) in pairs.iter() {
        let pinf = p1 - p0;
        cur_vals.push(p1);
        pinfs.push(pinf);
    }
    // Evaluate at x = 1..(d-1)
    for sums_slot in sums.iter_mut().take(d - 1) {
        let mut acc = F::one();
        for v in cur_vals.iter() {
            acc *= *v;
        }
        *sums_slot += acc;
        // advance all to next x
        for (cur_val, pinf) in cur_vals.iter_mut().zip(pinfs.iter()) {
            *cur_val += *pinf;
        }
    }
    // Evaluate at infinity (product of leading coefficients)
    let mut acc_inf = F::one();
    for pinf in pinfs.iter() {
        acc_inf *= *pinf;
    }
    sums[d - 1] += acc_inf;
}

fn compute_mles_product_sum_naive<F: JoltField>(
    mles: &[RaPolynomial<u8, F>],
    claim: F,
    eq_poly: &GruenSplitEqPolynomial<F>,
) -> UniPoly<F> {
    let num_x_out = eq_poly.E_out_current_len();
    let num_x_in = eq_poly.E_in_current_len();
    let current_scalar = eq_poly.get_current_scalar();

    let sum_evals: Vec<F> = if num_x_in == 1 {
        let eq_in_eval = eq_poly.E_in_current()[0] * current_scalar;
        let eq_out_evals = eq_poly.E_out_current();

        (0..num_x_out)
            .into_par_iter()
            .map(|j_out| {
                // partial_evals layout: [1, 2, ..., D-1, ∞]
                let mut partial_evals = vec![F::zero(); mles.len()];
                let mut mle_eval_pairs = vec![(F::zero(), F::zero()); mles.len()];

                for (i, mle) in mles.iter().enumerate() {
                    let v0 = mle.get_bound_coeff(2 * j_out);
                    let v1 = mle.get_bound_coeff(2 * j_out + 1);
                    mle_eval_pairs[i] = (v0, v1);
                }
                // incorporate eq_in * current_scalar as a common factor by scaling one pair
                mle_eval_pairs[0].0 *= eq_in_eval;
                mle_eval_pairs[0].1 *= eq_in_eval;

                product_eval_univariate_naive_accumulate(&mle_eval_pairs, &mut partial_evals);

                partial_evals
                    .into_iter()
                    .map(|v| {
                        let result = v * eq_out_evals[j_out];
                        *result.as_unreduced_ref()
                    })
                    .collect::<Vec<_>>()
            })
            .fold_with(
                vec![F::Unreduced::<5>::zero(); mles.len()],
                |running, new: Vec<F::Unreduced<4>>| {
                    running.into_iter().zip(new).map(|(a, b)| a + b).collect()
                },
            )
            .reduce(
                || vec![F::Unreduced::zero(); mles.len()],
                |running, new| running.into_iter().zip(new).map(|(a, b)| a + b).collect(),
            )
            .into_iter()
            .map(F::from_barrett_reduce)
            .collect()
    } else {
        let num_x_in_bits = num_x_in.log_2();
        let eq_in_evals = eq_poly.E_in_current();
        let eq_out_evals = eq_poly.E_out_current();

        (0..num_x_out)
            .into_par_iter()
            .map(|j_out| {
                let mut partial_evals = vec![F::zero(); mles.len()];
                let mut mle_eval_pairs = vec![(F::zero(), F::zero()); mles.len()];

                for (j_in, eq_in_eval) in eq_in_evals.iter().take(num_x_in).enumerate() {
                    let j = (j_out << num_x_in_bits) | j_in;

                    for (i, mle) in mles.iter().enumerate() {
                        let v0 = mle.get_bound_coeff(2 * j);
                        let v1 = mle.get_bound_coeff(2 * j + 1);
                        mle_eval_pairs[i] = (v0, v1);
                    }

                    let scale = *eq_in_eval * current_scalar;
                    mle_eval_pairs[0].0 *= scale;
                    mle_eval_pairs[0].1 *= scale;
                    product_eval_univariate_naive_accumulate(&mle_eval_pairs, &mut partial_evals);
                }

                partial_evals
                    .into_iter()
                    .map(|v| {
                        let result = v * eq_out_evals[j_out];
                        *result.as_unreduced_ref()
                    })
                    .collect::<Vec<_>>()
            })
            .fold_with(
                vec![F::Unreduced::<5>::zero(); mles.len()],
                |running, new: Vec<F::Unreduced<4>>| {
                    running.into_iter().zip(new).map(|(a, b)| a + b).collect()
                },
            )
            .reduce(
                || vec![F::Unreduced::zero(); mles.len()],
                |running, new| running.into_iter().zip(new).map(|(a, b)| a + b).collect(),
            )
            .into_iter()
            .map(F::from_barrett_reduce)
            .collect()
    };

    let r_round = eq_poly.get_current_w();
    let eq_eval_at_0 = EqPolynomial::mle(&[F::zero()], &[r_round]);
    let eq_eval_at_1 = EqPolynomial::mle(&[F::one()], &[r_round]);

    let eval_at_1 = sum_evals[0];
    let eval_at_0 = (claim - eq_eval_at_1 * eval_at_1) / eq_eval_at_0;

    let toom_evals = [&[eval_at_0], &*sum_evals].concat();
    let tmp_coeffs = UniPoly::from_evals_toom(&toom_evals).coeffs;

    // Multiply by eq(X, r_round) = (1 - r_round) + (2r_round - 1)X
    let constant_coeff = F::one() - r_round;
    let x_coeff = r_round + r_round - F::one();
    let mut coeffs = vec![F::zero(); tmp_coeffs.len() + 1];
    for (i, coeff) in tmp_coeffs.into_iter().enumerate() {
        coeffs[i] += coeff * constant_coeff;
        coeffs[i + 1] += coeff * x_coeff;
    }

    UniPoly::from_coeff(coeffs)
}

fn bench_mles_product_sum(c: &mut Criterion, n_mle: usize) {
    let rng = &mut test_rng();
    let mle_n_vars = 14;
    let random_mle: MultilinearPolynomial<Fr> = vec![Fr::random(rng); 1 << mle_n_vars].into();
    let mles = vec![RaPolynomial::RoundN(random_mle); n_mle];
    let r = vec![<Fr as JoltField>::Challenge::random(rng); mle_n_vars];
    let claim = Fr::random(rng);
    let eq_poly = GruenSplitEqPolynomial::new(&r, BindingOrder::LowToHigh);

    let mut group = c.benchmark_group(format!("Product of {n_mle} MLEs sum"));
    group.bench_function("optimized", |b| {
        b.iter(|| compute_mles_product_sum(&mles, claim, &eq_poly))
    });
    group.bench_function("naive", |b| {
        b.iter(|| compute_mles_product_sum_naive(&mles, claim, &eq_poly))
    });
    group.finish();
}

fn mles_product_sum_benches(c: &mut Criterion) {
    bench_mles_product_sum(c, 4);
    bench_mles_product_sum(c, 8);
    bench_mles_product_sum(c, 16);
    bench_mles_product_sum(c, 32);
}

criterion_group!(benches, mles_product_sum_benches);
criterion_main!(benches);
