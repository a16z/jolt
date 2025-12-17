use ark_bn254::Fr;
use ark_std::test_rng;
use criterion::{criterion_group, criterion_main, Criterion};
use jolt_core::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial},
        ra_poly::RaPolynomial,
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::mles_product_sum::{
        compute_mles_product_sum, eval_linear_prod_naive_assign, finish_mles_product_sum_from_evals,
    },
};

fn compute_mles_product_sum_naive<F: JoltField>(
    mles: &[RaPolynomial<u8, F>],
    claim: F,
    eq_poly: &GruenSplitEqPolynomial<F>,
) -> UniPoly<F> {
    /// Per-`x_out` accumulator used inside `par_fold_out_in` for the naive path.
    ///
    /// - `lanes[k]` accumulates the contribution for output lane `k`.
    /// - `pairs` and `evals` are scratch buffers reused across all `g` values
    ///   for a given `x_out`, to avoid repeated allocations and zeroing.
    struct NaiveInnerAcc<F: JoltField> {
        lanes: Vec<F>,
        pairs: Vec<(F, F)>,
        evals: Vec<F>,
    }

    let d = mles.len();
    let current_scalar = eq_poly.get_current_scalar();

    // Naive implementation using the same split-eq parallel structure as the
    // optimized version, but with the O(d^2) product evaluator.
    //
    // We fold over the current split-eq weights:
    //   Σ_{x_out} E_out[x_out] · Σ_{x_in} E_in[x_in] · P_g(x)
    // where `g = group_index(x_out, x_in)` and `P_g` is the product over MLEs.
    let sum_evals: Vec<F> = eq_poly
        .par_fold_out_in(
            // Per-`x_out` accumulator: one lane per output point plus scratch
            // buffers for the naive product evaluation.
            || NaiveInnerAcc {
                lanes: vec![F::zero(); d],
                pairs: vec![(F::zero(), F::zero()); d],
                evals: vec![F::zero(); d],
            },
            |inner, g, _x_in, e_in| {
                // Build per-g pairs [(p0, p1); d] in-place.
                for (i, mle) in mles.iter().enumerate() {
                    let v0 = mle.get_bound_coeff(2 * g);
                    let v1 = mle.get_bound_coeff(2 * g + 1);
                    inner.pairs[i] = (v0, v1);
                }

                // Evaluate the product on the grid using the naive O(d^2) kernel,
                // then accumulate with the inner eq weight `e_in`.
                eval_linear_prod_naive_assign(&inner.pairs, &mut inner.evals);
                for k in 0..d {
                    inner.lanes[k] += e_in * inner.evals[k];
                }
            },
            |_x_out, e_out, mut inner| {
                // Scale by the outer eq weight, reusing the `lanes` allocation
                // as the outer accumulator.
                for v in &mut inner.lanes {
                    *v *= e_out;
                }
                inner.lanes
            },
            |mut a, b| {
                // Merge accumulators across `x_out`.
                for k in 0..d {
                    a[k] += b[k];
                }
                a
            },
        )
        .into_iter()
        .map(|x| x * current_scalar)
        .collect();

    finish_mles_product_sum_from_evals(&sum_evals, claim, eq_poly)
}

fn bench_mles_product_sum(c: &mut Criterion, n_mle: usize) {
    let rng = &mut test_rng();
    let mle_n_vars = 14;
    let random_mle: MultilinearPolynomial<Fr> =
        vec![<Fr as JoltField>::random(rng); 1 << mle_n_vars].into();
    let mles = vec![RaPolynomial::RoundN(random_mle); n_mle];
    let r = vec![<Fr as JoltField>::Challenge::random(rng); mle_n_vars];
    let claim = <Fr as JoltField>::random(rng);
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
    bench_mles_product_sum(c, 2);
    bench_mles_product_sum(c, 3);
    bench_mles_product_sum(c, 4);
    bench_mles_product_sum(c, 8);
    bench_mles_product_sum(c, 16);
    bench_mles_product_sum(c, 32);
}

criterion_group!(benches, mles_product_sum_benches);
criterion_main!(benches);
