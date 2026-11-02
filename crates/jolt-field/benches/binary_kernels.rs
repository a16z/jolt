use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use jolt_field::{
    Accumulator, ExtField, Field, NaiveAccumulator, WithAccumulator, F128, F192, F64,
};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::hint::black_box;

fn bench_field<F: Field>(c: &mut Criterion, name: &str, a: F, b: F) {
    let mut group = c.benchmark_group(name);
    let _ = group.bench_function("mul", |bencher| {
        bencher.iter(|| black_box(black_box(a) * black_box(b)));
    });
    let _ = group.bench_function("square", |bencher| {
        bencher.iter(|| black_box(black_box(a).square()));
    });
    let mut rng = ChaCha20Rng::seed_from_u64(0x6d75_6c5f_736c_6963);
    let pairs: Vec<_> = (0..1024)
        .map(|_| (F::random(&mut rng), F::random(&mut rng)))
        .collect();
    let mut output = vec![F::zero(); pairs.len()];
    let _ = group.throughput(Throughput::Elements(1024));
    let _ = group.bench_function("mul_slice", |bencher| {
        bencher.iter(|| {
            for (dest, &(a, b)) in output.iter_mut().zip(black_box(&pairs)) {
                *dest = a * b;
            }
            let _ = black_box(&output);
        });
    });
    group.finish();
}

fn bench_accumulators<F: Field + WithAccumulator>(c: &mut Criterion, name: &str) {
    let mut rng = ChaCha20Rng::seed_from_u64(0x6163_6375_6d75_6c61);
    let pairs: Vec<_> = (0..1024)
        .map(|_| (F::random(&mut rng), F::random(&mut rng)))
        .collect();
    let mut group = c.benchmark_group(format!("{name}/accumulator"));
    let _ = group.throughput(Throughput::Elements(1024));
    let _ = group.bench_function("deferred", |bencher| {
        bencher.iter(|| {
            let mut acc = F::Accumulator::default();
            for &(a, b) in black_box(&pairs) {
                acc.fmadd(black_box(a), black_box(b));
            }
            black_box(acc.reduce())
        });
    });
    let _ = group.bench_function("naive", |bencher| {
        bencher.iter(|| {
            let mut acc = NaiveAccumulator::<F>::default();
            for &(a, b) in black_box(&pairs) {
                acc.fmadd(black_box(a), black_box(b));
            }
            black_box(acc.reduce())
        });
    });
    group.finish();
}

fn binary_kernels(c: &mut Criterion) {
    let a64 = F64::from_raw(0xfedc_ba98_7654_3210);
    let b64 = F64::from_raw(0x89ab_cdef_0123_4567);
    let a128 = F128::from_raw(0x8123_4567_89ab_cdef_fedc_ba98_7654_3210);
    let b128 = F128::from_raw(0xf0e1_d2c3_b4a5_9687_89ab_cdef_0123_4567);
    let a192 = [
        a64,
        F64::from_raw(0x8123_4567_89ab_cdef),
        F64::from_raw(0xc39a_5f06_7d28_b4e1),
    ];
    let b192 = [
        b64,
        F64::from_raw(0xf0e1_d2c3_b4a5_9687),
        F64::from_raw(0xb7d2_468a_1357_9cef),
    ];

    bench_accumulators::<F64>(c, "F64");
    bench_accumulators::<F128>(c, "F128");
    bench_accumulators::<F192>(c, "F192");
    bench_field(c, "F64", a64, b64);
    bench_field(c, "F128", a128, b128);
    bench_field(
        c,
        "F192",
        F192::from_base_fn(|i| a192[i]),
        F192::from_base_fn(|i| b192[i]),
    );
}

criterion_group!(benches, binary_kernels);
criterion_main!(benches);
