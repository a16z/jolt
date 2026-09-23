use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use jolt_field::{Field, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::hint::black_box;

const SAMPLES: usize = 1 << 12;

fn bench_field<F: Field>(c: &mut Criterion, name: &str) {
    let mut rng = ChaCha20Rng::seed_from_u64(0x4a4f_4c54);
    let values: Vec<(F, F)> = (0..SAMPLES)
        .map(|_| (F::random(&mut rng), F::random(&mut rng)))
        .collect();
    let mut group = c.benchmark_group("field arithmetic");

    let _ = group.bench_with_input(BenchmarkId::new("add", name), &values, |b, values| {
        b.iter(|| {
            for &(a, d) in values {
                let _ = black_box(a + d);
            }
        });
    });
    let _ = group.bench_with_input(BenchmarkId::new("sub", name), &values, |b, values| {
        b.iter(|| {
            for &(a, d) in values {
                let _ = black_box(a - d);
            }
        });
    });
    let _ = group.bench_with_input(BenchmarkId::new("mul", name), &values, |b, values| {
        b.iter(|| {
            for &(a, d) in values {
                let _ = black_box(a * d);
            }
        });
    });
    let _ = group.bench_with_input(BenchmarkId::new("square", name), &values, |b, values| {
        b.iter(|| {
            for &(a, _) in values {
                let _ = black_box(a.square());
            }
        });
    });
    group.finish();
}

fn field_arithmetic(c: &mut Criterion) {
    bench_field::<Prime32Offset99>(c, "fp32_c99");
    bench_field::<Prime64Offset59>(c, "fp64_c59");
    bench_field::<Prime128OffsetA7F7>(c, "fp128_a7f7");
}

criterion_group!(benches, field_arithmetic);
criterion_main!(benches);
