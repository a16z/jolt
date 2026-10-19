//! dudect-style timing check of `jolt_crypto::sample` (M5a acceptance; an
//! intentional diagnostic, not a CI test).
//!
//! ```text
//! cargo run --release -p jolt-crypto --example sample_dudect -- [measurements]
//! ```
//!
//! Two Welch t-tests, each on raw timings and on timings cropped at the
//! pooled 90th percentile (dudect's outlier cropping):
//!
//! - `GaussianSampler::sample` at the headline width `k = 16,080,468`, split
//!   by the secret output: `|z| < σ` against `|z| ≥ σ`. Isochrony means the
//!   whole call, retries included, is independent of the returned value.
//! - `JointRejection::accept` on 4,096 coordinates, a fixed `(z, v)` against
//!   fresh random `(z, v)`, interleaved at random.
//!
//! `|t| < 4.5` is dudect's pass threshold. Run it on an idle machine.

#![expect(clippy::print_stdout, reason = "diagnostic tool reports to stdout")]
#![expect(
    clippy::indexing_slicing,
    reason = "indices are bounded by construction"
)]

use std::hint::black_box;
use std::time::Instant;

use jolt_crypto::sample::{BlockParams, GaussianSampler, GaussianWidth, JointRejection};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

const HEADLINE_K: u64 = 16_080_468;
const BOX_BOUND: u64 = 93_120_000;
const SHIFT_BOUND: u64 = 1 << 13;
const LEN: usize = 4096;

/// Welch's t over two samples of timings.
fn welch_t(a: &[f64], b: &[f64]) -> f64 {
    let moments = |x: &[f64]| {
        let n = x.len() as f64;
        let mean = x.iter().sum::<f64>() / n;
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (mean, var, n)
    };
    let (ma, va, na) = moments(a);
    let (mb, vb, nb) = moments(b);
    (ma - mb) / (va / na + vb / nb).sqrt()
}

/// Prints raw and cropped t, cropping both classes at the pooled 90th
/// percentile.
fn report(name: &str, classes: &[Vec<f64>; 2]) {
    let mut pooled: Vec<f64> = classes.iter().flatten().copied().collect();
    pooled.sort_by(f64::total_cmp);
    let cut = pooled[pooled.len() * 9 / 10];
    let crop = |x: &[f64]| x.iter().copied().filter(|&t| t <= cut).collect::<Vec<_>>();
    let raw = welch_t(&classes[0], &classes[1]);
    let cropped = welch_t(&crop(&classes[0]), &crop(&classes[1]));
    let verdict = if raw.abs().max(cropped.abs()) < 4.5 {
        "PASS"
    } else {
        "LEAK?"
    };
    println!(
        "{name}: n = {} / {}, t = {raw:.2}, cropped t = {cropped:.2} -> {verdict}",
        classes[0].len(),
        classes[1].len()
    );
}

fn main() {
    let measurements: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(200_000);
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    let mut rng = ChaCha20Rng::seed_from_u64(seed);

    let Ok(width) = GaussianWidth::from_multiplier(HEADLINE_K) else {
        return;
    };
    let sampler = GaussianSampler::new(width);
    let sigma = width.sigma();
    let mut classes = [Vec::new(), Vec::new()];
    for _ in 0..measurements {
        let start = Instant::now();
        let z = black_box(sampler.sample(&mut rng));
        let elapsed = start.elapsed().as_nanos() as f64;
        classes[usize::from((z as f64).abs() >= sigma)].push(elapsed);
    }
    let mean = classes.iter().flatten().sum::<f64>() / measurements as f64;
    println!("sampler: {mean:.0} ns/sample (single thread, including timer overhead)");
    report("sampler, |z| < σ vs |z| ≥ σ", &classes);

    let block = BlockParams {
        width,
        box_bound: BOX_BOUND,
        shift_bound: SHIFT_BOUND,
        len: LEN,
    };
    let Ok(rule) = JointRejection::new(vec![block], 16, 137) else {
        return;
    };
    let draw = |rng: &mut ChaCha20Rng, bound: u64| -> Vec<i64> {
        (0..LEN)
            .map(|_| (rng.next_u64() % (2 * bound + 1)) as i64 - bound as i64)
            .collect()
    };
    let fixed = (draw(&mut rng, BOX_BOUND), draw(&mut rng, SHIFT_BOUND));
    let mut classes = [Vec::new(), Vec::new()];
    for _ in 0..measurements / 10 {
        let class = (rng.next_u32() & 1) as usize;
        let random;
        let (z, v) = if class == 0 {
            (&fixed.0, &fixed.1)
        } else {
            random = (draw(&mut rng, BOX_BOUND), draw(&mut rng, SHIFT_BOUND));
            (&random.0, &random.1)
        };
        let start = Instant::now();
        let accepted = black_box(rule.accept(black_box(z), black_box(v), &mut rng));
        classes[class].push(start.elapsed().as_nanos() as f64);
        if accepted.is_err() {
            return;
        }
    }
    report("rejection, fixed vs random (z, v)", &classes);
}
