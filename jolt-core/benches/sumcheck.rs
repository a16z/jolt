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

    let batch_size = 10;
    let num_vars = 14;

    let r_eq = vec![Fr::rand(&mut rng); num_vars];
    let eq = DensePolynomial::new(EqPolynomial::new(r_eq).evals());

    let mut poly_as = Vec::with_capacity(batch_size);
    let mut poly_bs = Vec::with_capacity(batch_size);
    for _ in 0..batch_size {
        let ra = vec![Fr::rand(&mut rng); num_vars];
        let rb = vec![Fr::rand(&mut rng); num_vars];
        let a = DensePolynomial::new(EqPolynomial::new(ra).evals());
        let b = DensePolynomial::new(EqPolynomial::new(rb).evals());
        poly_as.push(a);
        poly_bs.push(b);
    }
    let params = CubicSumcheckParams::new_prod(poly_as.clone(), poly_bs.clone(), eq.clone(), num_vars);
    let coeffs = vec![Fr::rand(&mut rng); batch_size];

    let mut joint_claim = Fr::zero();
    for batch_i in 0..batch_size {
        let mut claim = Fr::zero();
        for var_i in 0..num_vars{
            use liblasso::utils::index_to_field_bitvector;
        
            let eval_a = poly_as[batch_i].evaluate(&index_to_field_bitvector(var_i, num_vars));
            let eval_b = poly_bs[batch_i].evaluate(&index_to_field_bitvector(var_i, num_vars));
            let eval_eq = eq.evaluate(&index_to_field_bitvector(var_i, num_vars));
        
            claim += eval_a * eval_b * eval_eq;
        }
        joint_claim += coeffs[batch_i] * claim;
    }

    group.bench_function("sumcheck 10xbatched 2^14", |b| {
        b.iter(|| {
            let mut transcript = Transcript::new(b"test_transcript");
            let params = black_box(params.clone());
            let (proof, r, evals)  = SumcheckInstanceProof::prove_cubic_batched_special::<EdwardsProjective>(
                &joint_claim,
                params,
                &coeffs,
                &mut transcript
            );
        })
    });
    group.finish();
}

criterion_group!(benches, sumcheck_bench);
criterion_main!(benches);