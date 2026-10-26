//! Arithmetic benchmark coverage for the 63-bit carry-preserving scalar and
//! packed kernels introduced by the Akita #427 port, and for quadratic
//! extension multiplication over the full-word prime `2^64 - 59`.
//!
//! Run with:
//! `cargo bench -p jolt-field --no-default-features --features solinas --bench fp64_kernels`

#[cfg(feature = "solinas")]
mod harness {
    use criterion::Criterion;
    use jolt_field::{Ext2, Field, FpExt2, NegOneNr, Packed, Prime64Offset59, Ring, WithPacking};
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use std::hint::black_box;

    type F63 = jolt_field::Fp64<{ (1u64 << 63) - 259 }>;
    type PF63 = <F63 as WithPacking>::Packing;
    type E63 = Ext2<F63>;
    type PE63 = <E63 as WithPacking>::Packing;
    type F64 = Prime64Offset59;
    type E64 = Ext2<F64>;
    type PE64 = <E64 as WithPacking>::Packing;
    type E64NegOne = FpExt2<F64, NegOneNr>;

    pub(crate) fn bench(c: &mut Criterion) {
        let mut rng = ChaCha20Rng::seed_from_u64(0xA427_0063);
        let a = F63::random(&mut rng);
        let b = F63::random(&mut rng);
        let pa = PF63::from_fn(|_| F63::random(&mut rng));
        let pb = PF63::from_fn(|_| F63::random(&mut rng));
        let ea = E63::random(&mut rng);
        let eb = E63::random(&mut rng);
        let pea = PE63::from_fn(|_| E63::random(&mut rng));
        let peb = PE63::from_fn(|_| E63::random(&mut rng));

        let mut group = c.benchmark_group("fp64_prime63_offset259");
        let _ = group.bench_function("scalar_mul", |bencher| {
            bencher.iter(|| black_box(a) * black_box(b));
        });
        let _ = group.bench_function("packed_mul", |bencher| {
            bencher.iter(|| black_box(pa) * black_box(pb));
        });
        let _ = group.bench_function("ext2_scalar_mul", |bencher| {
            bencher.iter(|| black_box(ea) * black_box(eb));
        });
        let _ = group.bench_function("ext2_packed_mul", |bencher| {
            bencher.iter(|| black_box(pea) * black_box(peb));
        });
        group.finish();

        let ea = E64::random(&mut rng);
        let eb = E64::random(&mut rng);
        let na = E64NegOne::random(&mut rng);
        let nb = E64NegOne::random(&mut rng);
        let pea = PE64::from_fn(|_| E64::random(&mut rng));
        let peb = PE64::from_fn(|_| E64::random(&mut rng));
        let chain: Vec<E64> = (0..1024).map(|_| E64::random(&mut rng)).collect();

        let mut group = c.benchmark_group("fp64_prime64_offset59");
        let _ = group.bench_function("ext2_scalar_mul", |bencher| {
            bencher.iter(|| black_box(ea) * black_box(eb));
        });
        let _ = group.bench_function("ext2_scalar_square", |bencher| {
            bencher.iter(|| black_box(ea).square());
        });
        let _ = group.bench_function("ext2_neg_one_scalar_mul", |bencher| {
            bencher.iter(|| black_box(na) * black_box(nb));
        });
        let _ = group.bench_function("ext2_packed_mul", |bencher| {
            bencher.iter(|| black_box(pea) * black_box(peb));
        });
        let _ = group.bench_function("ext2_dependent_mul_1024", |bencher| {
            bencher.iter(|| {
                chain
                    .iter()
                    .fold(black_box(ea), |accumulator, &x| accumulator * x)
            });
        });
        group.finish();
    }
}

#[cfg(feature = "solinas")]
criterion::criterion_group!(benches, harness::bench);
#[cfg(feature = "solinas")]
criterion::criterion_main!(benches);

#[cfg(not(feature = "solinas"))]
fn main() {}
