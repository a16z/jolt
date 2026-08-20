#![expect(clippy::print_stdout, clippy::expect_used, reason = "bench harness")]

use std::time::Instant;

use ark_ff::UniformRand;
use jolt_kernels::cuda::shared_context;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

const PAIRS: usize = 8192;
const COLUMNS: usize = 42;
const SAMPLES: usize = 3;

fn g1_words(point: &ark_bn254::G1Projective) -> Vec<u64> {
    let mut out = Vec::with_capacity(12);
    for value in [&point.x, &point.y, &point.z] {
        out.extend_from_slice(&value.0 .0);
    }
    out
}

fn g2_words(point: &ark_bn254::G2Projective) -> Vec<u64> {
    let mut out = Vec::with_capacity(24);
    for value in [&point.x, &point.y, &point.z] {
        out.extend_from_slice(&value.c0.0 .0);
        out.extend_from_slice(&value.c1.0 .0);
    }
    out
}

fn main() {
    let Some(context) = shared_context() else {
        println!("no CUDA device");
        return;
    };
    let mut rng = ChaCha20Rng::seed_from_u64(20_260_820);
    let g1: Vec<u64> = (0..PAIRS)
        .flat_map(|_| g1_words(&ark_bn254::G1Projective::rand(&mut rng)))
        .collect();
    let g2: Vec<u64> = (0..PAIRS)
        .flat_map(|_| g2_words(&ark_bn254::G2Projective::rand(&mut rng)))
        .collect();
    let device_g1 = context.upload_raw_u64(&g1).expect("upload g1");
    let device_g2 = context.upload_raw_u64(&g2).expect("upload g2");

    let mut plain = f64::MAX;
    for _ in 0..SAMPLES {
        let now = Instant::now();
        let _ = context
            .multi_miller_resident(&device_g1, 0, &device_g2, 0, PAIRS)
            .expect("plain miller");
        plain = plain.min(now.elapsed().as_secs_f64() * 1e3);
    }

    let now = Instant::now();
    let prepared = context
        .prepare_g2_lines(&device_g2, 0, PAIRS)
        .expect("prepare");
    let prepare_ms = now.elapsed().as_secs_f64() * 1e3;

    let mut with_table = f64::MAX;
    for _ in 0..SAMPLES {
        let now = Instant::now();
        let _ = context
            .multi_miller_prepared(&device_g1, 0, &prepared, PAIRS)
            .expect("prepared miller");
        with_table = with_table.min(now.elapsed().as_secs_f64() * 1e3);
    }

    println!("pairs={PAIRS}");
    println!("  plain miller      {plain:>8.2} ms");
    println!("  prepare (once)    {prepare_ms:>8.2} ms");
    println!(
        "  prepared miller   {with_table:>8.2} ms   {:.2}x",
        plain / with_table
    );
    println!(
        "  tier-2 projection over {COLUMNS} columns: plain {:.0} ms vs prepared {:.0} ms",
        plain * COLUMNS as f64,
        prepare_ms + with_table * COLUMNS as f64
    );
}
