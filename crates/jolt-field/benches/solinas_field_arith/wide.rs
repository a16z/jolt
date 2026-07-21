use criterion::{black_box, Criterion};
use jolt_field::{CanonicalField, Prime128Offset275};
use rand::{rngs::StdRng, RngCore, SeedableRng};

use super::data::rand_u128;

pub(crate) fn bench_wide_ops(c: &mut Criterion) {
    type F = Prime128Offset275;

    let mut rng = StdRng::seed_from_u64(0x01de_be0c_0001);
    let a = F::from_canonical_u128_reduced(rand_u128(&mut rng));
    let b = F::from_canonical_u128_reduced(rand_u128(&mut rng));
    let b_u64 = rng.next_u64();

    let mut group = c.benchmark_group("field_arith/wide/prime128_offset275");

    group.bench_function("mul_wide_u64_only", |bench| {
        bench.iter(|| black_box(black_box(a).mul_wide_u64(black_box(b_u64))))
    });

    group.bench_function("mul_wide_only", |bench| {
        bench.iter(|| black_box(black_box(a).mul_wide(black_box(b))))
    });

    let limbs3 = [rng.next_u64(), rng.next_u64(), rng.next_u64()];
    let limbs4 = [
        rng.next_u64(),
        rng.next_u64(),
        rng.next_u64(),
        rng.next_u64(),
    ];

    group.bench_function("mul_wide_limbs_3_to_5_only", |bench| {
        bench.iter(|| black_box(black_box(a).mul_wide_limbs::<3, 5>(black_box(limbs3))))
    });
    group.bench_function("mul_wide_limbs_3_to_4_only", |bench| {
        bench.iter(|| black_box(black_box(a).mul_wide_limbs::<3, 4>(black_box(limbs3))))
    });
    group.bench_function("mul_wide_limbs_4_to_5_only", |bench| {
        bench.iter(|| black_box(black_box(a).mul_wide_limbs::<4, 5>(black_box(limbs4))))
    });
    group.bench_function("mul_wide_limbs_4_to_4_only", |bench| {
        bench.iter(|| black_box(black_box(a).mul_wide_limbs::<4, 4>(black_box(limbs4))))
    });

    group.bench_function("full_mul_u64_reduce", |bench| {
        bench.iter(|| black_box(black_box(a) * F::from_u64(black_box(b_u64))))
    });

    group.bench_function("full_mul_reduce", |bench| {
        bench.iter(|| black_box(black_box(a) * black_box(b)))
    });

    let wide3 = a.mul_wide_u64(b_u64);
    let wide4 = a.mul_wide(b);
    let wide5 = {
        let mut l = [0u64; 5];
        l[..3].copy_from_slice(&wide3);
        l[4] = rng.next_u64() & 0xFF;
        l
    };

    group.bench_function("solinas_reduce_3_limbs", |bench| {
        bench.iter(|| black_box(F::solinas_reduce(black_box(&wide3))))
    });

    group.bench_function("solinas_reduce_4_limbs", |bench| {
        bench.iter(|| black_box(F::solinas_reduce(black_box(&wide4))))
    });

    group.bench_function("solinas_reduce_5_limbs", |bench| {
        bench.iter(|| black_box(F::solinas_reduce(black_box(&wide5))))
    });

    group.bench_function("mul_wide_u64_roundtrip", |bench| {
        bench.iter(|| {
            let x = black_box(a);
            let y = black_box(b_u64);
            black_box(F::solinas_reduce(&x.mul_wide_u64(y)))
        })
    });

    group.bench_function("mul_wide_roundtrip", |bench| {
        bench.iter(|| {
            let x = black_box(a);
            let y = black_box(b);
            black_box(F::solinas_reduce(&x.mul_wide(y)))
        })
    });

    group.bench_function("mul_wide_limbs_3_to_5_roundtrip", |bench| {
        bench.iter(|| {
            let x = black_box(a);
            let m = black_box(limbs3);
            black_box(F::solinas_reduce(&x.mul_wide_limbs::<3, 5>(m)))
        })
    });
    group.bench_function("mul_wide_limbs_3_to_4_roundtrip", |bench| {
        bench.iter(|| {
            let x = black_box(a);
            let m = black_box(limbs3);
            black_box(F::solinas_reduce(&x.mul_wide_limbs::<3, 4>(m)))
        })
    });
    group.bench_function("mul_wide_limbs_4_to_5_roundtrip", |bench| {
        bench.iter(|| {
            let x = black_box(a);
            let m = black_box(limbs4);
            black_box(F::solinas_reduce(&x.mul_wide_limbs::<4, 5>(m)))
        })
    });
    group.bench_function("mul_wide_limbs_4_to_4_roundtrip", |bench| {
        bench.iter(|| {
            let x = black_box(a);
            let m = black_box(limbs4);
            black_box(F::solinas_reduce(&x.mul_wide_limbs::<4, 4>(m)))
        })
    });

    group.finish();
}
