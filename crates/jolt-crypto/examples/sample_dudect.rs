//! dudect-style timing check of `jolt_crypto::sample` (M5a acceptance; an
//! intentional diagnostic, not a CI test).
//!
//! ```text
//! cargo run --release -p jolt-crypto --features sample --example sample_dudect -- \
//!     [measurements per test, default 10^7] [seeds, default 3]
//! ```
//!
//! Each test is a Welch t-test on raw timings, repeated for every seed;
//! `|t| < 4.5` is dudect's pass threshold. Run it on an idle machine.
//!
//! - **Sampler.** `GaussianSampler::sample`, one call per measurement, split
//!   by the secret output. At the headline width `k = 16,080,468`: `|z| < σ`
//!   vs `|z| ≥ σ`, CDT index `x = ⌊|z|/k⌋ = 0` vs `x ≥ 1`, and the sign. At
//!   `k = 1`, where `z = 0` is common: `z = 0` vs `z ≠ 0`. Isochrony means
//!   the whole call, retries included, is independent of the returned value.
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
const BOX_BOUND: u64 = 93_120_000;
const SHIFT_BOUND: u64 = 1 << 13;
const LEN: usize = 64;
const CHUNK: usize = 10_000;
const THRESHOLD: f64 = 4.5;

/// Welch's t over two samples of timings, by running moments.
#[derive(Default)]
struct Welch {
    n: [f64; 2],
    mean: [f64; 2],
    m2: [f64; 2],
}

impl Welch {
    fn push(&mut self, class: usize, x: f64) {
        self.n[class] += 1.0;
        let delta = x - self.mean[class];
        self.mean[class] += delta / self.n[class];
        self.m2[class] += delta * (x - self.mean[class]);
    }

    fn t(&self) -> f64 {
        let var = |c: usize| self.m2[c] / (self.n[c] - 1.0) / self.n[c];
        (self.mean[0] - self.mean[1]) / (var(0) + var(1)).sqrt()
    }
}

fn report(name: &str, test: &Welch, failures: &mut usize) {
    let t = test.t();
    let pass = t.abs() < THRESHOLD;
    if !pass {
        *failures += 1;
    }
    println!(
        "  {name}: n = {:.0} / {:.0}, t = {t:.2} {}",
        test.n[0],
        test.n[1],
        if pass { "PASS" } else { "LEAK?" }
    );
}

fn run_sampler(
    measurements: usize,
    rng: &mut ChaCha20Rng,
    failures: &mut usize,
) -> Result<(), SampleError> {
    let width = GaussianWidth::from_multiplier(HEADLINE_K)?;
    let sampler = GaussianSampler::new(width);
    let sigma = width.sigma();
    let (mut magnitude, mut cdt, mut sign) = (Welch::default(), Welch::default(), Welch::default());
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
    let mut zero = Welch::default();
    for _ in 0..measurements {
        let start = Instant::now();
        let z = black_box(small.sample(rng));
        zero.push(usize::from(z == 0), start.elapsed().as_nanos() as f64);
    }
    report("sampler (k = 1) z = 0 vs z ≠ 0", &zero, failures);
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

    let mut test = Welch::default();
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
        accepts as f64 / (test.n[0] + test.n[1])
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
        let result = run_sampler(measurements, &mut rng, &mut failures)
            .and_then(|()| run_rejection(measurements, &mut rng, &mut failures));
        if let Err(error) = result {
            eprintln!("sample_dudect: {error}");
            return ExitCode::FAILURE;
        }
    }
    println!("{failures} test(s) above |t| = {THRESHOLD}");
    if failures == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
