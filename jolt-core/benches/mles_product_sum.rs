use ark_bn254::Fr;
use ark_std::test_rng;
use criterion::{criterion_group, criterion_main, Criterion};
use jolt_core::{
    field::JoltField, poly::multilinear_polynomial::MultilinearPolynomial,
    subprotocols::mles_product_sum::compute_mles_product_sum,
};

fn bench_mles_product_sum(c: &mut Criterion, n_mle: usize) {
    let rng = &mut test_rng();
    let mle_n_vars = 14;
    let random_mle: MultilinearPolynomial<Fr> = vec![Fr::random(rng); 1 << mle_n_vars].into();
    let mles = vec![random_mle; n_mle];
    let log_sum_n_terms = mle_n_vars - 1;
    let eq_evals = vec![Fr::random(rng); 1 << log_sum_n_terms];
    let r0 = Fr::random(rng);
    let correction_factor = Fr::random(rng);
    let claim = Fr::random(rng);

    c.bench_function(&format!("Product of {n_mle} MLEs sum"), |b| {
        b.iter(|| {
            compute_mles_product_sum(
                &mles,
                claim,
                r0,
                &eq_evals,
                correction_factor,
                log_sum_n_terms,
            )
        })
    });
}

fn mles_product_sum_benches(c: &mut Criterion) {
    bench_mles_product_sum(c, 4);
    bench_mles_product_sum(c, 8);
    bench_mles_product_sum(c, 16);
}

criterion_group!(benches, mles_product_sum_benches);
criterion_main!(benches);
