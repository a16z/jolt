use ark_curve25519::{Fr, EdwardsProjective};
use ark_std::{test_rng, UniformRand, Zero, One};
use criterion::{black_box, criterion_group, criterion_main, Criterion, SamplingMode, Throughput};
use liblasso::poly::dense_mlpoly::DensePolynomial;
use liblasso::poly::eq_poly::EqPolynomial;
use liblasso::subprotocols::sumcheck::{CubicSumcheckParams, SumcheckInstanceProof};
use merlin::Transcript;

fn criterion_config(c: &mut Criterion) -> criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> {
    let mut group = c.benchmark_group("sumcheck");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);
    group.throughput(Throughput::Elements(1));
    group
}

fn sumcheck_bench(c: &mut Criterion) {
    let mut group = criterion_config(c);

    let num_vars = 16;
    let mut rng = test_rng();

    let r1 = vec![Fr::rand(&mut rng); num_vars];
    let r2 = vec![Fr::rand(&mut rng); num_vars];
    let r3 = vec![Fr::rand(&mut rng); num_vars];
    let eq1 = DensePolynomial::new(EqPolynomial::new(r1).evals());
    let eq2 = DensePolynomial::new(EqPolynomial::new(r2).evals());
    let eq3 = DensePolynomial::new(EqPolynomial::new(r3).evals());
    let params = CubicSumcheckParams::new_prod(vec![eq1.clone()], vec![eq2.clone()], eq3.clone(), num_vars);

    let mut claim = Fr::zero();
    for i in 0..num_vars{
        use liblasso::utils::index_to_field_bitvector;
    
        let eval1 = eq1.evaluate(&index_to_field_bitvector(i, num_vars));
        let eval2 = eq2.evaluate(&index_to_field_bitvector(i, num_vars));
        let eval3 = eq3.evaluate(&index_to_field_bitvector(i, num_vars));
    
        claim += eval1 * eval2 * eval3;
    }

    let coeffs = vec![Fr::one()];

    group.bench_function("sumcheck unbatched 2^16", |b| {


        b.iter(|| {
            let mut transcript = Transcript::new(b"test_transcript");
            let params = black_box(params.clone());
            let (proof, r, evals)  = SumcheckInstanceProof::prove_cubic_batched_special::<EdwardsProjective>(
                &claim,
                params,
                &coeffs,
                &mut transcript
            );
        })
    });
    group.finish();
}

fn sumcheck(poly_vars: usize) {

}

criterion_group!(benches, sumcheck_bench);
criterion_main!(benches);