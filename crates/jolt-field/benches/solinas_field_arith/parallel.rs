#[cfg(feature = "parallel")]
use std::env;
#[cfg(feature = "parallel")]
use std::thread;

#[cfg(feature = "parallel")]
use criterion::{black_box, Criterion, Throughput};
#[cfg(feature = "parallel")]
use jolt_field::packed::{PackedField, PackedValue};
#[cfg(feature = "parallel")]
use jolt_field::{
    CanonicalField, Prime128Offset275, Prime31Offset19, Prime64Offset59, RandomSampling,
};
#[cfg(feature = "parallel")]
use rand::{rngs::StdRng, SeedableRng};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "parallel")]
use rayon::ThreadPoolBuilder;

#[cfg(feature = "parallel")]
use super::cases::*;
#[cfg(feature = "parallel")]
use super::data::rand_u128;
#[cfg(feature = "parallel")]
use super::params::env_usize;

#[cfg(feature = "parallel")]
pub(crate) fn bench_parallel_throughput(c: &mut Criterion) {
    let profile = env::var("AKITA_BENCH_PAR_PROFILE").unwrap_or_else(|_| "dev".to_string());
    let default_n = match profile.as_str() {
        "scale" | "large" => 1 << 20,
        "xlarge" => 1 << 22,
        _ => 1 << 15,
    };
    let n = env_usize("AKITA_BENCH_PAR_N", default_n);
    let default_chunk = match profile.as_str() {
        "scale" | "large" => 1 << 14,
        "xlarge" => 1 << 15,
        _ => 1 << 12,
    };
    let chunk = env_usize("AKITA_BENCH_PAR_CHUNK", default_chunk);
    let threads = env_usize(
        "AKITA_BENCH_PAR_THREADS",
        thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(1),
    );

    assert!(threads > 0, "AKITA_BENCH_PAR_THREADS must be > 0");
    assert!(n > 0, "AKITA_BENCH_PAR_N must be > 0");
    assert!(chunk > 0, "AKITA_BENCH_PAR_CHUNK must be > 0");

    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("build benchmark rayon pool");

    let mut rng = StdRng::seed_from_u64(0x7061_7261_0001);
    let lhs31: Vec<Prime31Offset19> = (0..n).map(|_| RandomSampling::random(&mut rng)).collect();
    let rhs31: Vec<Prime31Offset19> = (0..n).map(|_| RandomSampling::random(&mut rng)).collect();
    let lhs64: Vec<Prime64Offset59> = (0..n).map(|_| RandomSampling::random(&mut rng)).collect();
    let rhs64: Vec<Prime64Offset59> = (0..n).map(|_| RandomSampling::random(&mut rng)).collect();
    let lhs128: Vec<Prime128Offset275> = (0..n)
        .map(|_| Prime128Offset275::from_canonical_u128_reduced(rand_u128(&mut rng)))
        .collect();
    let rhs128: Vec<Prime128Offset275> = (0..n)
        .map(|_| Prime128Offset275::from_canonical_u128_reduced(rand_u128(&mut rng)))
        .collect();

    let lhs31_p = P31O19::pack_slice(&lhs31);
    let rhs31_p = P31O19::pack_slice(&rhs31);
    let lhs64_p = P64O59::pack_slice(&lhs64);
    let rhs64_p = P64O59::pack_slice(&rhs64);
    let lhs128_p = P128O275::pack_slice(&lhs128);
    let rhs128_p = P128O275::pack_slice(&rhs128);

    let mut out31 = vec![Prime31Offset19::zero(); n];
    let mut out64 = vec![Prime64Offset59::zero(); n];
    let mut out128 = vec![F128::zero(); n];
    let mut out31_p = vec![P31O19::broadcast(Prime31Offset19::zero()); lhs31_p.len()];
    let mut out64_p = vec![P64O59::broadcast(Prime64Offset59::zero()); lhs64_p.len()];
    let mut out128_p = vec![P128O275::broadcast(F128::zero()); lhs128_p.len()];

    let mut group = c.benchmark_group(format!(
        "field_arith/parallel/{profile}/n{n}/chunk{chunk}/threads{threads}"
    ));
    group.throughput(Throughput::Elements(n as u64));

    bench_scalar_parallel(
        &mut group,
        &pool,
        PRIME31_OFFSET19,
        &lhs31,
        &rhs31,
        &mut out31,
        chunk,
    );
    bench_scalar_parallel(
        &mut group,
        &pool,
        PRIME64_OFFSET59,
        &lhs64,
        &rhs64,
        &mut out64,
        chunk,
    );
    bench_scalar_parallel(
        &mut group,
        &pool,
        PRIME128_OFFSET275,
        &lhs128,
        &rhs128,
        &mut out128,
        chunk,
    );
    bench_packed_parallel(
        &mut group,
        &pool,
        PRIME31_OFFSET19,
        &lhs31_p,
        &rhs31_p,
        &mut out31_p,
        (chunk / P31O19::WIDTH).max(1),
    );
    bench_packed_parallel(
        &mut group,
        &pool,
        PRIME64_OFFSET59,
        &lhs64_p,
        &rhs64_p,
        &mut out64_p,
        (chunk / P64O59::WIDTH).max(1),
    );
    bench_packed_parallel(
        &mut group,
        &pool,
        PRIME128_OFFSET275,
        &lhs128_p,
        &rhs128_p,
        &mut out128_p,
        (chunk / P128O275::WIDTH).max(1),
    );

    group.finish();
}

#[cfg(feature = "parallel")]
fn bench_scalar_parallel<F>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    pool: &rayon::ThreadPool,
    label: &str,
    lhs: &[F],
    rhs: &[F],
    out: &mut [F],
    chunk: usize,
) where
    F: jolt_field::FieldCore + Send + Sync,
{
    group.bench_function(format!("{label}_mul_seq"), |b| {
        b.iter(|| {
            let a = black_box(lhs);
            let b_v = black_box(rhs);
            for i in 0..out.len() {
                out[i] = a[i] * b_v[i];
            }
            black_box(out[0])
        })
    });

    group.bench_function(format!("{label}_mul_par_chunked"), |b| {
        b.iter(|| {
            let a = black_box(lhs);
            let b_v = black_box(rhs);
            pool.install(|| {
                out.par_chunks_mut(chunk)
                    .enumerate()
                    .for_each(|(chunk_idx, out_chunk)| {
                        let start = chunk_idx * chunk;
                        for (j, dst) in out_chunk.iter_mut().enumerate() {
                            let idx = start + j;
                            *dst = a[idx] * b_v[idx];
                        }
                    });
            });
            black_box(out[0])
        })
    });
}

#[cfg(feature = "parallel")]
fn bench_packed_parallel<PF>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    pool: &rayon::ThreadPool,
    label: &str,
    lhs: &[PF],
    rhs: &[PF],
    out: &mut [PF],
    chunk: usize,
) where
    PF: PackedField + Copy + Send + Sync,
{
    group.bench_function(format!("{label}_packed_mul_seq"), |b| {
        b.iter(|| {
            let a = black_box(lhs);
            let b_v = black_box(rhs);
            for i in 0..out.len() {
                out[i] = a[i] * b_v[i];
            }
            black_box(out[0].extract(0))
        })
    });

    group.bench_function(format!("{label}_packed_mul_par_chunked"), |b| {
        b.iter(|| {
            let a = black_box(lhs);
            let b_v = black_box(rhs);
            pool.install(|| {
                out.par_chunks_mut(chunk)
                    .enumerate()
                    .for_each(|(chunk_idx, out_chunk)| {
                        let start = chunk_idx * chunk;
                        for (j, dst) in out_chunk.iter_mut().enumerate() {
                            let idx = start + j;
                            *dst = a[idx] * b_v[idx];
                        }
                    });
            });
            black_box(out[0].extract(0))
        })
    });
}

#[cfg(not(feature = "parallel"))]
pub(crate) fn bench_parallel_throughput(_: &mut criterion::Criterion) {}
