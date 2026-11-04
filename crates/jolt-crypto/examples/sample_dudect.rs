//! dudect-style timing check of `jolt_crypto::sample` (an intentional
//! diagnostic, not a CI test).
//!
//! ```text
//! cargo run --release -p jolt-crypto --features sample --example sample_dudect -- \
//!     [measurements per test, default 10^7] [seeds, default 3]
//! ```
//!
//! Each test is a Welch t-test, repeated for every seed, on the raw timings
//! and on the timings cropped at the 50th, 75th, 90th, 95th and 99th
//! percentiles of the pooled distribution (dudect's crops, which remove the
//! long tail of interrupts and migrations). A test passes when every one of
//! its statistics has `|t| < 4.5`, dudect's threshold. Run it on an idle
//! machine.
//!
//! - **Positive control.** The headline sampler followed by a deliberate
//!   secret-dependent delay (a short busy loop when `|z| ≥ σ`), split like
//!   the sampler test below. It must be detected (`|t| ≥ 4.5`); if it is not,
//!   the run is too noisy to vouch for the other tests and counts as a
//!   failure.
//! - **Sampler.** `GaussianSampler::sample`, one call per measurement, split
//!   by the secret output. At the headline width `k = 16,080,468`: `|z| < σ`
//!   vs `|z| ≥ σ`, CDT index `x = ⌊|z|/k⌋ = 0` vs `x ≥ 1`, and the sign. At
//!   `k = 1`, where `z = 0` is common: `z = 0` vs `z ≠ 0`. Isochrony means
//!   the whole call, retries included, is independent of the returned value.
//! - **Parallel fill.** `GaussianSampler::fill_parallel` of two chunks
//!   (`2^15` outputs) at the production width `k = 58,642,511,897`, one call
//!   per measurement, `measurements / 1000` calls, split by the secret
//!   output: more or fewer than the expected share of `|z| ≥ σ`, and the sign
//!   of the sum. The work split is public, so the whole call must be
//!   independent of its outputs.
//! - **Rejection.** `JointRejection::accept` on 64 coordinates, a fixed
//!   `(z, v)` vs fresh random `(z, v)`, classes interleaved at random. All
//!   inputs of a chunk are generated before it is timed, and both classes
//!   read their inputs from the same buffer layout.

#![expect(clippy::print_stdout, reason = "diagnostic tool reports to stdout")]
#![expect(clippy::print_stderr, reason = "diagnostic tool reports errors")]
#![expect(
    clippy::indexing_slicing,
    reason = "indices are bounded by construction"
)]

use std::hint::black_box;
use std::process::ExitCode;
use std::time::Instant;

use jolt_crypto::sample::{
    BlockParams, GaussianSampler, GaussianWidth, JointRejection, SampleError,
};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

const HEADLINE_K: u64 = 16_080_468;
const PRODUCTION_K: u64 = 58_642_511_897;
const FILL_LEN: usize = 1 << 15;
const BOX_BOUND: u64 = 93_120_000;
const SHIFT_BOUND: u64 = 1 << 13;
const LEN: usize = 64;
const CHUNK: usize = 10_000;
const THRESHOLD: f64 = 4.5;
/// dudect's crop percentiles.
const CROPS: [f64; 5] = [0.5, 0.75, 0.9, 0.95, 0.99];
/// Busy-loop iterations of the positive control's injected delay.
const CONTROL_SPINS: usize = 32;

/// One test's timings, kept whole so that percentile crops can be applied.
#[derive(Default)]
struct Timings {
    samples: Vec<(usize, f64)>,
}

impl Timings {
    fn push(&mut self, class: usize, x: f64) {
        self.samples.push((class, x));
    }

    fn count(&self, class: usize) -> usize {
        self.samples.iter().filter(|(c, _)| *c == class).count()
    }

    /// Welch's t over the samples at or below `cutoff`.
    fn welch(&self, cutoff: f64) -> f64 {
        let (mut n, mut mean, mut m2) = ([0.0f64; 2], [0.0f64; 2], [0.0f64; 2]);
        for &(class, x) in self.samples.iter().filter(|(_, x)| *x <= cutoff) {
            n[class] += 1.0;
            let delta = x - mean[class];
            mean[class] += delta / n[class];
            m2[class] += delta * (x - mean[class]);
        }
        let var = |c: usize| m2[c] / (n[c] - 1.0) / n[c];
        (mean[0] - mean[1]) / (var(0) + var(1)).sqrt()
    }

    /// The raw statistic, then one per crop percentile.
    fn t_stats(&self) -> Vec<f64> {
        let mut sorted: Vec<f64> = self.samples.iter().map(|(_, x)| *x).collect();
        sorted.sort_unstable_by(f64::total_cmp);
        std::iter::once(f64::INFINITY)
            .chain(
                CROPS
                    .iter()
                    .map(|p| sorted[((sorted.len() - 1) as f64 * p) as usize]),
            )
            .map(|cutoff| self.welch(cutoff))
            .collect()
    }
}

fn report(name: &str, test: &Timings, failures: &mut usize) {
    let stats = test.t_stats();
    let worst = stats.iter().fold(0.0f64, |m, t| m.max(t.abs()));
    let pass = worst < THRESHOLD;
    if !pass {
        *failures += 1;
    }
    println!(
        "  {name}: n = {} / {}, raw t = {:.2}, max |t| over crops = {worst:.2} {}",
        test.count(0),
        test.count(1),
        stats[0],
        if pass { "PASS" } else { "LEAK?" }
    );
}

/// A test that must be detected: counts a failure when it is not.
fn report_control(name: &str, test: &Timings, failures: &mut usize) {
    let worst = test.t_stats().iter().fold(0.0f64, |m, t| m.max(t.abs()));
    let detected = worst >= THRESHOLD;
    if !detected {
        *failures += 1;
    }
    println!(
        "  {name}: max |t| = {worst:.2} {}",
        if detected {
            "DETECTED"
        } else {
            "MISSED (run too noisy)"
        }
    );
}

fn run_control(
    measurements: usize,
    rng: &mut ChaCha20Rng,
    failures: &mut usize,
) -> Result<(), SampleError> {
    let width = GaussianWidth::from_multiplier(HEADLINE_K)?;
    let sampler = GaussianSampler::new(width);
    let sigma = width.sigma();
    let mut test = Timings::default();
    for _ in 0..measurements {
        let start = Instant::now();
        let z = black_box(sampler.sample(rng));
        let wide = (z as f64).abs() >= sigma;
        if wide {
            let mut spin = 0usize;
            for i in 0..CONTROL_SPINS {
                spin = black_box(spin.wrapping_add(i));
            }
        }
        test.push(usize::from(wide), start.elapsed().as_nanos() as f64);
    }
    report_control(
        "positive control: sampler + delay when |z| ≥ σ",
        &test,
        failures,
    );
    Ok(())
}

fn run_sampler(
    measurements: usize,
    rng: &mut ChaCha20Rng,
    failures: &mut usize,
) -> Result<(), SampleError> {
    let width = GaussianWidth::from_multiplier(HEADLINE_K)?;
    let sampler = GaussianSampler::new(width);
    let sigma = width.sigma();
    let (mut magnitude, mut cdt, mut sign) =
        (Timings::default(), Timings::default(), Timings::default());
    let mut total = 0.0;
    for _ in 0..measurements {
        let start = Instant::now();
        let z = black_box(sampler.sample(rng));
        let elapsed = start.elapsed().as_nanos() as f64;
        total += elapsed;
        magnitude.push(usize::from((z as f64).abs() >= sigma), elapsed);
        cdt.push(usize::from(z.unsigned_abs() >= HEADLINE_K), elapsed);
        sign.push(usize::from(z < 0), elapsed);
    }
    println!(
        "  sampler: {:.0} ns/sample (one thread, including timer overhead)",
        total / measurements as f64
    );
    report("sampler |z| < σ vs |z| ≥ σ", &magnitude, failures);
    report("sampler x = 0 vs x ≥ 1", &cdt, failures);
    report("sampler sign", &sign, failures);

    let small = GaussianSampler::new(GaussianWidth::from_multiplier(1)?);
    let mut zero = Timings::default();
    for _ in 0..measurements {
        let start = Instant::now();
        let z = black_box(small.sample(rng));
        zero.push(usize::from(z == 0), start.elapsed().as_nanos() as f64);
    }
    report("sampler (k = 1) z = 0 vs z ≠ 0", &zero, failures);
    Ok(())
}

fn run_parallel_fill(
    measurements: usize,
    rng: &mut ChaCha20Rng,
    failures: &mut usize,
) -> Result<(), SampleError> {
    let width = GaussianWidth::from_multiplier(PRODUCTION_K)?;
    let sampler = GaussianSampler::new(width);
    let sigma = width.sigma();
    // Pr[|z| ≥ σ] for a Gaussian of this width.
    let expected = 0.3173 * FILL_LEN as f64;
    let (mut tail, mut sign) = (Timings::default(), Timings::default());
    let mut out = vec![0i64; FILL_LEN];
    let mut total = 0.0;
    let calls = (measurements / 1000).max(100);
    for _ in 0..calls {
        let start = Instant::now();
        sampler.fill_parallel(rng, black_box(&mut out));
        let elapsed = start.elapsed().as_nanos() as f64;
        total += elapsed;
        let wide = out.iter().filter(|&&z| (z as f64).abs() >= sigma).count();
        tail.push(usize::from(wide as f64 >= expected), elapsed);
        sign.push(usize::from(out.iter().sum::<i64>() < 0), elapsed);
    }
    println!(
        "  parallel fill: {:.0} ns/sample wall ({} threads)",
        total / (calls * FILL_LEN) as f64,
        rayon::current_num_threads()
    );
    report(
        "parallel fill: |z| ≥ σ share above vs below",
        &tail,
        failures,
    );
    report("parallel fill: sign of the sum", &sign, failures);
    Ok(())
}

fn run_rejection(
    measurements: usize,
    rng: &mut ChaCha20Rng,
    failures: &mut usize,
) -> Result<(), SampleError> {
    let block = BlockParams {
        width: GaussianWidth::from_multiplier(HEADLINE_K)?,
        box_bound: BOX_BOUND,
        shift_bound: SHIFT_BOUND,
        len: LEN,
    };
    let rule = JointRejection::new(vec![block], 16, 137)?;
    let fill = |rng: &mut ChaCha20Rng, out: &mut [i64], bound: u64| {
        for slot in out {
            *slot = (rng.next_u64() % (2 * bound + 1)) as i64 - bound as i64;
        }
    };
    let (mut fixed_z, mut fixed_v) = (vec![0i64; LEN], vec![0i64; LEN]);
    fill(rng, &mut fixed_z, BOX_BOUND);
    fill(rng, &mut fixed_v, SHIFT_BOUND);

    let mut test = Timings::default();
    let mut accepts = 0u64;
    let (mut zs, mut vs) = (vec![0i64; CHUNK * LEN], vec![0i64; CHUNK * LEN]);
    let mut classes = vec![0usize; CHUNK];
    for _ in 0..measurements.div_ceil(CHUNK) {
        for (i, class) in classes.iter_mut().enumerate() {
            *class = (rng.next_u32() & 1) as usize;
            let range = i * LEN..(i + 1) * LEN;
            if *class == 0 {
                zs[range.clone()].copy_from_slice(&fixed_z);
                vs[range].copy_from_slice(&fixed_v);
            } else {
                fill(rng, &mut zs[range.clone()], BOX_BOUND);
                fill(rng, &mut vs[range], SHIFT_BOUND);
            }
        }
        for (i, &class) in classes.iter().enumerate() {
            let range = i * LEN..(i + 1) * LEN;
            let (z, v) = (&zs[range.clone()], &vs[range]);
            let start = Instant::now();
            let accepted = black_box(rule.accept(black_box(z), black_box(v), rng));
            test.push(class, start.elapsed().as_nanos() as f64);
            accepts += u64::from(accepted?.unwrap_u8());
        }
    }
    println!(
        "  rejection: {:.3} accepted",
        accepts as f64 / test.samples.len() as f64
    );
    report("rejection fixed vs random (z, v)", &test, failures);
    Ok(())
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let measurements: usize = args
        .next()
        .and_then(|a| a.parse().ok())
        .unwrap_or(10_000_000);
    let seeds: u64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(3);
    let mut failures = 0;
    for seed in 0..seeds {
        println!("seed {seed}:");
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let result = run_control(measurements, &mut rng, &mut failures)
            .and_then(|()| run_sampler(measurements, &mut rng, &mut failures))
            .and_then(|()| run_parallel_fill(measurements, &mut rng, &mut failures))
            .and_then(|()| run_rejection(measurements, &mut rng, &mut failures));
        if let Err(error) = result {
            eprintln!("sample_dudect: {error}");
            return ExitCode::FAILURE;
        }
    }
    println!("{failures} failure(s): leaks at |t| ≥ {THRESHOLD} or missed controls");
    if failures == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
