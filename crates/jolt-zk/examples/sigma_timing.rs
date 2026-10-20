//! Per-proof and per-attempt time of the Whiteout Σ-protocol prover on a
//! scaled synthetic instance (A16 of `specs/whiteout-sigma.md`; a
//! measurement tool, not a CI test).
//!
//! ```text
//! cargo run --release -p jolt-zk --features whiteout-diagnostics --example sigma_timing -- \
//!     [witness scale, default 512] [proofs, default 40]
//! ```
//!
//! The instance is the test fixture over `R_256` with its witness `scale`
//! times longer (`1,536 · scale` coefficients, a third of them public). Each
//! proof includes the release self-checks and leaves out the response proof
//! (W5). A least-squares fit of time against attempt count separates the
//! per-proof cost from the per-attempt cost.

#![expect(clippy::print_stdout, reason = "measurement tool reports to stdout")]
#![expect(clippy::expect_used, reason = "a measurement tool on a valid fixture")]

use std::time::Instant;

use jolt_zk::whiteout::diagnostics::AttemptHarness;

fn main() {
    let mut args = std::env::args().skip(1);
    let scale: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(512);
    let proofs: u64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(40);
    let start = Instant::now();
    let harness = AttemptHarness::<256>::scaled(1, scale).expect("instance");
    println!(
        "s: {} coefficients ({} Gaussian); setup and lift {:.2} s; {} threads",
        harness.len(),
        harness
            .gaussian_ranges()
            .iter()
            .map(|range| range.len())
            .sum::<usize>(),
        start.elapsed().as_secs_f64(),
        rayon::current_num_threads()
    );
    let samples: Vec<(f64, f64)> = (0..proofs)
        .map(|seed| {
            let start = Instant::now();
            let attempts = harness.prove(seed).expect("proof");
            (attempts as f64, start.elapsed().as_secs_f64())
        })
        .collect();
    let n = samples.len() as f64;
    let (mean_a, mean_t) = samples
        .iter()
        .fold((0.0, 0.0), |(a, t), &(x, y)| (a + x / n, t + y / n));
    let (sxy, sxx) = samples.iter().fold((0.0, 0.0), |(sxy, sxx), &(x, y)| {
        (
            sxy + (x - mean_a) * (y - mean_t),
            sxx + (x - mean_a).powi(2),
        )
    });
    let per_attempt = sxy / sxx;
    println!(
        "{proofs} proofs: mean {mean_a:.2} attempts, {mean_t:.3} s per proof; \
         fit {per_attempt:.3} s per attempt + {:.3} s per proof",
        mean_t - per_attempt * mean_a
    );
}
