#![expect(clippy::print_stdout, clippy::expect_used, reason = "bench harness")]

use std::time::Instant;

use ark_ff::UniformRand;
use jolt_kernels::cuda::{context_for, device_count, enter_device};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

const SAMPLES: usize = 3;

const GEOMETRIES: [(usize, usize); 15] = [
    (21, 16_384),
    (42, 8_192),
    (168, 2_048),
    (640, 512),
    (1_344, 256),
    (2_688, 128),
    (20, 16_384),
    (10, 16_384),
    (7, 16_384),
    (5, 16_384),
    (4, 16_384),
    (3, 16_384),
    (2, 16_384),
    (1, 16_384),
    (1, 1_024),
];

fn main() {
    let devices = device_count();
    if devices == 0 {
        println!("no CUDA device");
        return;
    }
    let mut rng = ChaCha20Rng::seed_from_u64(20_260_825);
    let pairs = GEOMETRIES
        .iter()
        .map(|&(lanes, count)| lanes * count)
        .max()
        .unwrap_or(0);
    let mut g1 = Vec::with_capacity(pairs * 12);
    let mut g2 = Vec::with_capacity(pairs * 24);
    for _ in 0..pairs {
        let point = ark_bn254::G1Projective::rand(&mut rng);
        for coordinate in [&point.x, &point.y, &point.z] {
            g1.extend_from_slice(&coordinate.0 .0);
        }
        let other = ark_bn254::G2Projective::rand(&mut rng);
        for coordinate in [&other.x, &other.y, &other.z] {
            for part in [&coordinate.c0, &coordinate.c1] {
                g2.extend_from_slice(&part.0 .0);
            }
        }
    }

    println!(
        "{:>6}  {:>6}  {:>7}  {:>10}  {:>10}  {:>12}",
        "device", "lanes", "pairs", "total", "ms", "pairs/ms"
    );
    for device in 0..devices {
        let _guard = enter_device(device);
        let Some(context) = context_for(device) else {
            continue;
        };
        let device_g1 = context.upload_raw_u64(&g1).expect("upload g1");
        let device_g2 = context.upload_raw_u64(&g2).expect("upload g2");
        for (lanes, count) in GEOMETRIES {
            let segments: Vec<(usize, usize)> = (0..lanes)
                .map(|lane| (lane * count, lane * count))
                .collect();
            let _ = context
                .multi_miller_batch(&device_g1, &device_g2, &segments, count)
                .expect("warm miller");
            let mut best = f64::MAX;
            for _ in 0..SAMPLES {
                let now = Instant::now();
                let _ = context
                    .multi_miller_batch(&device_g1, &device_g2, &segments, count)
                    .expect("miller");
                best = best.min(now.elapsed().as_secs_f64() * 1e3);
            }
            println!(
                "{device:>6}  {lanes:>6}  {count:>7}  {:>10}  {best:>10.2}  {:>12.1}",
                lanes * count,
                (lanes * count) as f64 / best,
            );
        }
    }
}
