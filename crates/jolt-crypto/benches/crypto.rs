#![expect(unused_results, clippy::unwrap_used)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use jolt_crypto::{
    compress_gt, decompress_gt, Bn254, Bn254G1, Bn254G1Affine, Bn254G2, JoltGroup, PairingGroup,
    Pedersen, PedersenSetup, VectorCommitment,
};
use jolt_field::{Field, Fr, Ring};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

fn bench_g1_scalar_mul(c: &mut Criterion) {
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let g = Bn254::random_g1(&mut rng);
    let s = Fr::random(&mut rng);

    c.bench_function("g1_scalar_mul", |b| {
        b.iter(|| g.scalar_mul(&s));
    });
}

fn bench_g2_scalar_mul(c: &mut Criterion) {
    let mut rng = ChaCha20Rng::seed_from_u64(1);
    let g = Bn254::g2_generator();
    let s = Fr::random(&mut rng);

    c.bench_function("g2_scalar_mul", |b| {
        b.iter(|| g.scalar_mul(&s));
    });
}

fn bench_g1_add(c: &mut Criterion) {
    let mut rng = ChaCha20Rng::seed_from_u64(2);
    let a = Bn254::random_g1(&mut rng);
    let b = Bn254::random_g1(&mut rng);

    c.bench_function("g1_add", |b_| {
        b_.iter(|| a + b);
    });
}

fn bench_g1_double(c: &mut Criterion) {
    let mut rng = ChaCha20Rng::seed_from_u64(3);
    let a = Bn254::random_g1(&mut rng);

    c.bench_function("g1_double", |b| {
        b.iter(|| a.double());
    });
}

fn bench_g1_msm(c: &mut Criterion) {
    let mut group = c.benchmark_group("g1_msm");

    for size in [4, 16, 64, 256, 1024] {
        let mut rng = ChaCha20Rng::seed_from_u64(10);
        let bases: Vec<Bn254G1> = (0..size).map(|_| Bn254::random_g1(&mut rng)).collect();
        let scalars: Vec<Fr> = (0..size).map(|_| Fr::random(&mut rng)).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| Bn254G1::msm(&bases, &scalars));
        });
    }
    group.finish();
}

/// Distinct affine points `[i]G`, i = 1..=n, cheap to generate at 2^20.
fn sequential_affine_bases(n: usize) -> Vec<Bn254G1Affine> {
    let g = Bn254::g1_generator();
    let mut acc = g;
    let mut points = Vec::with_capacity(n);
    for _ in 0..n {
        points.push(acc);
        acc += g;
    }
    Bn254::g1_to_affine(&points)
}

fn bench_g1_msm_small(c: &mut Criterion) {
    use jolt_crypto::ec::bn254::bit_columns::g1_bit_columns_msm;

    let mut group = c.benchmark_group("g1_msm_small");
    group.sample_size(10);
    for log_n in [17, 20] {
        let n = 1usize << log_n;
        let mut rng = ChaCha20Rng::seed_from_u64(log_n as u64);
        let bases = sequential_affine_bases(n);
        let full: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let u16s: Vec<u16> = (0..n).map(|_| rng.next_u32() as u16).collect();
        let u8s: Vec<u8> = (0..n).map(|_| rng.next_u32() as u8).collect();
        let bits: Vec<u8> = (0..n).map(|_| (rng.next_u32() & 1) as u8).collect();

        group.bench_with_input(BenchmarkId::new("full_width", log_n), &n, |b, _| {
            b.iter(|| Bn254::g1_affine_msm(&bases, &full));
        });
        group.bench_with_input(BenchmarkId::new("u16", log_n), &n, |b, _| {
            b.iter(|| Bn254::g1_affine_msm_small(&bases, &u16s));
        });
        group.bench_with_input(BenchmarkId::new("u8", log_n), &n, |b, _| {
            b.iter(|| Bn254::g1_affine_msm_small(&bases, &u8s));
        });
        group.bench_with_input(BenchmarkId::new("bit_column", log_n), &n, |b, _| {
            b.iter(|| g1_bit_columns_msm(&bases, &[&bits]));
        });
    }
    group.finish();
}

fn bench_g2_msm(c: &mut Criterion) {
    let mut group = c.benchmark_group("g2_msm");

    for size in [4, 16, 64, 256] {
        let mut rng = ChaCha20Rng::seed_from_u64(20);
        let g = Bn254::g2_generator();
        let bases: Vec<Bn254G2> = (0..size)
            .map(|i| g.scalar_mul(&Fr::from_u64(i as u64 + 1)))
            .collect();
        let scalars: Vec<Fr> = (0..size).map(|_| Fr::random(&mut rng)).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| Bn254G2::msm(&bases, &scalars));
        });
    }
    group.finish();
}

fn bench_pairing(c: &mut Criterion) {
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();

    c.bench_function("pairing", |b| {
        b.iter(|| Bn254::pairing(&g1, &g2));
    });
}

fn bench_multi_pairing(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_pairing");

    for size in [2, 4, 8, 16] {
        let mut rng = ChaCha20Rng::seed_from_u64(30);
        let g1s: Vec<Bn254G1> = (0..size).map(|_| Bn254::random_g1(&mut rng)).collect();
        let g2 = Bn254::g2_generator();
        let g2s: Vec<Bn254G2> = (0..size)
            .map(|i| g2.scalar_mul(&Fr::from_u64(i as u64 + 1)))
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| Bn254::multi_pairing(&g1s, &g2s));
        });
    }
    group.finish();
}

fn bench_pedersen_commit(c: &mut Criterion) {
    let mut group = c.benchmark_group("pedersen_commit");

    for size in [4, 16, 64, 256, 1024] {
        let mut rng = ChaCha20Rng::seed_from_u64(40);
        let gens: Vec<Bn254G1> = (0..size).map(|_| Bn254::random_g1(&mut rng)).collect();
        let blinding_gen = Bn254::random_g1(&mut rng);
        let setup = PedersenSetup::new(gens, blinding_gen);
        let values: Vec<Fr> = (0..size).map(|_| Fr::random(&mut rng)).collect();
        let blinding = Fr::random(&mut rng);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| Pedersen::<Bn254G1>::commit(&setup, &values, &blinding));
        });
    }
    group.finish();
}

fn bench_gt_scalar_mul(c: &mut Criterion) {
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();
    let gt = Bn254::pairing(&g1, &g2);
    let mut rng = ChaCha20Rng::seed_from_u64(50);
    let s = Fr::random(&mut rng);

    c.bench_function("gt_scalar_mul", |b| {
        b.iter(|| gt.scalar_mul(&s));
    });
}

fn bench_gt_decompression(c: &mut Criterion) {
    let gt = Bn254::pairing(&Bn254::g1_generator(), &Bn254::g2_generator());
    let compressed = compress_gt(&gt);
    c.bench_function("gt_decompress", |b| {
        b.iter(|| decompress_gt(&compressed).unwrap());
    });

    let compressed_set = vec![compressed; 110];
    c.bench_function("gt_decompress_110", |b| {
        b.iter(|| {
            for compressed in &compressed_set {
                decompress_gt(compressed).unwrap();
            }
        });
    });
}

fn bench_g1_serde(c: &mut Criterion) {
    let mut rng = ChaCha20Rng::seed_from_u64(60);
    let g = Bn254::random_g1(&mut rng);
    let config = bincode::config::standard();
    let bytes = bincode::serde::encode_to_vec(g, config).unwrap();

    c.bench_function("g1_serialize_bincode", |b| {
        b.iter(|| bincode::serde::encode_to_vec(g, config).unwrap());
    });

    c.bench_function("g1_deserialize_bincode", |b| {
        b.iter(|| {
            bincode::serde::decode_from_slice::<Bn254G1, _>(&bytes, config).unwrap();
        });
    });
}

criterion_group!(
    benches,
    bench_g1_scalar_mul,
    bench_g2_scalar_mul,
    bench_g1_add,
    bench_g1_double,
    bench_g1_msm,
    bench_g1_msm_small,
    bench_g2_msm,
    bench_pairing,
    bench_multi_pairing,
    bench_pedersen_commit,
    bench_gt_scalar_mul,
    bench_gt_decompression,
    bench_g1_serde,
);
criterion_main!(benches);
