use criterion::{criterion_group, criterion_main, Criterion};
use jolt_field::{ExtField, Ring, F128, F192, F64};
use std::hint::black_box;

fn bench_field<F: Ring>(c: &mut Criterion, name: &str, a: F, b: F) {
    let mut group = c.benchmark_group(name);
    let _ = group.bench_function("mul", |bencher| {
        bencher.iter(|| black_box(black_box(a) * black_box(b)));
    });
    let _ = group.bench_function("square", |bencher| {
        bencher.iter(|| black_box(black_box(a).square()));
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
