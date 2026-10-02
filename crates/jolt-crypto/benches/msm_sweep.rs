//! Manual width/size/thread sweep with one warmup and process CPU accounting.
//!
//! Build before taking the measurement mutex:
//! `cargo bench -p jolt-crypto --bench msm_sweep --no-run`.
//! With `/tmp/wrapper-gate.lock` held and idle load below the campaign threshold:
//! `MSM_CLASS=u16 MSM_LOGS=18,20,22,23 MSM_THREADS=10 MSM_REPEATS=3
//! cargo bench -p jolt-crypto --bench msm_sweep`.
//! Classes: `full`, `u16`, `u32`, `bit`. Bit rates use physical points.

#![expect(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::print_stdout,
    reason = "manual benchmark validates its inputs and prints measurements"
)]

use std::hint::black_box;
use std::process::{self, Command};
use std::time::Instant;

use jolt_crypto::ec::bn254::bit_columns::g1_bit_columns_msm;
use jolt_crypto::{Bn254, PairingGroup};
use jolt_field::{Field, Fr};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};
use rayon::{ThreadPool, ThreadPoolBuilder};

fn main() {
    let class = std::env::var("MSM_CLASS").expect("MSM_CLASS");
    let logs = std::env::var("MSM_LOGS")
        .unwrap_or_else(|_| "12,13,14,15,16,17,18,19,20,21,22,23".to_owned())
        .split(',')
        .map(|value| value.parse::<usize>().expect("integer log size"))
        .collect::<Vec<_>>();
    let repeats = std::env::var("MSM_REPEATS")
        .map_or(2, |value| value.parse().expect("integer repeat count"));
    let threads = std::env::var("MSM_THREADS")
        .map_or(10, |value| value.parse().expect("integer thread count"));
    let max_n = 1usize << logs.iter().copied().max().expect("at least one size");

    let generator = Bn254::g1_generator();
    let mut point = generator;
    let mut projective = Vec::with_capacity(max_n);
    for _ in 0..max_n {
        projective.push(point);
        point += generator;
    }
    let bases = Bn254::g1_to_affine(&projective);
    drop(projective);

    let mut rng = ChaCha20Rng::seed_from_u64(0x5045_5246_3504);
    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool");

    match class.as_str() {
        "full" => {
            let scalars = (0..max_n).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();
            sweep(&pool, &logs, repeats, threads, &class, |n| {
                let _ = black_box(Bn254::g1_affine_msm(&bases[..n], &scalars[..n]));
            });
        }
        "u32" => {
            let scalars = (0..max_n).map(|_| rng.next_u32()).collect::<Vec<_>>();
            sweep(&pool, &logs, repeats, threads, &class, |n| {
                let _ = black_box(Bn254::g1_affine_msm_small(&bases[..n], &scalars[..n]));
            });
        }
        "u16" => {
            let scalars = (0..max_n)
                .map(|_| rng.next_u32() as u16)
                .collect::<Vec<_>>();
            sweep(&pool, &logs, repeats, threads, &class, |n| {
                let _ = black_box(Bn254::g1_affine_msm_small(&bases[..n], &scalars[..n]));
            });
        }
        "bit" => {
            let scalars = (0..max_n)
                .map(|_| (rng.next_u32() & 1) as u8)
                .collect::<Vec<_>>();
            sweep(&pool, &logs, repeats, threads, &class, |n| {
                let _ = black_box(g1_bit_columns_msm(&bases[..n], &[&scalars[..n]]));
            });
        }
        _ => panic!("unknown MSM_CLASS {class}"),
    }
}

fn sweep(
    pool: &ThreadPool,
    logs: &[usize],
    repeats: usize,
    threads: usize,
    class: &str,
    mut run: impl FnMut(usize) + Send,
) {
    for &log_n in logs {
        let n = 1usize << log_n;
        pool.install(|| run(n));
        let mut wall_seconds = Vec::with_capacity(repeats);
        let cpu_start = process_cpu_seconds();
        let all_started = Instant::now();
        for _ in 0..repeats {
            let started = Instant::now();
            pool.install(|| run(n));
            wall_seconds.push(started.elapsed().as_secs_f64());
        }
        let all_wall = all_started.elapsed().as_secs_f64();
        let cpu_seconds = process_cpu_seconds() - cpu_start;
        let samples = wall_seconds.clone();
        wall_seconds.sort_by(f64::total_cmp);
        let min = wall_seconds[0];
        let median = wall_seconds[wall_seconds.len() / 2];
        println!(
            "class={class} log_n={log_n} threads={threads} samples={samples:?} min_s={min:.6} median_s={median:.6} min_us_per_point={:.6} median_us_per_point={:.6} cpu_s={cpu_seconds:.3} wall_s={all_wall:.3} cpu_per_wall={:.3}",
            min * 1e6 / n as f64,
            median * 1e6 / n as f64,
            cpu_seconds / all_wall,
        );
    }
}

fn process_cpu_seconds() -> f64 {
    let output = Command::new("ps")
        .args(["-o", "time=", "-p"])
        .arg(process::id().to_string())
        .output()
        .expect("process CPU time");
    let value = String::from_utf8(output.stdout).expect("process CPU time is UTF-8");
    let value = value.trim();
    let (days, clock) = value.split_once('-').map_or((0, value), |(days, clock)| {
        (days.parse::<u64>().expect("CPU days"), clock)
    });
    let parts = clock
        .split(':')
        .map(|part| part.parse::<f64>().expect("CPU clock component"))
        .collect::<Vec<_>>();
    let (hours, minutes, seconds) = match parts.as_slice() {
        [minutes, seconds] => (0.0, *minutes, *seconds),
        [hours, minutes, seconds] => (*hours, *minutes, *seconds),
        _ => panic!("unexpected process CPU time: {value}"),
    };
    days as f64 * 86_400.0 + hours * 3_600.0 + minutes * 60.0 + seconds
}
