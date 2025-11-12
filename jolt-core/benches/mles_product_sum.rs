use ark_bn254::Fr;
use ark_std::test_rng;
use criterion::{criterion_group, criterion_main, Criterion};
use jolt_core::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial},
        ra_poly::RaPolynomial,
        split_eq_poly::GruenSplitEqPolynomial,
    },
    subprotocols::mles_product_sum::compute_mles_product_sum,
};

fn bench_mles_product_sum(c: &mut Criterion, n_mle: usize) {
    let rng = &mut test_rng();
    let mle_n_vars = 14;
    let random_mle: MultilinearPolynomial<Fr> =
        vec![<Fr as JoltField>::random(rng); 1 << mle_n_vars].into();
    let mles = vec![RaPolynomial::RoundN(random_mle); n_mle];
    let r = vec![<Fr as JoltField>::Challenge::random(rng); mle_n_vars];
    let claim = <Fr as JoltField>::random(rng);
    let eq_poly = GruenSplitEqPolynomial::new(&r, BindingOrder::LowToHigh);

    c.bench_function(&format!("Product of {n_mle} MLEs sum"), |b| {
        b.iter(|| compute_mles_product_sum(&mles, claim, &eq_poly))
    });
}

fn mles_product_sum_benches(c: &mut Criterion) {
    bench_mles_product_sum(c, 4);
    bench_mles_product_sum(c, 8);
    bench_mles_product_sum(c, 16);
}

criterion_group!(benches, mles_product_sum_benches);
criterion_main!(benches);
