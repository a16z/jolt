//! dudect-style timing check of the Σ-protocol attempt's secret-dependent
//! steps (W4 acceptance; an intentional diagnostic, not a CI test).
//!
//! ```text
//! cargo run --release -p jolt-zk --features whiteout-diagnostics --example sigma_dudect -- \
//!     [measurements per test, default 10^5] [seeds, default 2] [t_D measurements on seed 0, default = first]
//! ```
//!
//! Each test is a Welch t-test on raw timings, fixed class vs random class,
//! interleaved at random, every input of a chunk written into preallocated
//! slots before the chunk is timed. As in dudect, the statistic is also computed on measurements
//! cropped at ten percentiles (taken from a first batch); a test passes if
//! every `|t| < 4.5`. Fixed classes for masks: all 0, all +1, all `+B`, all
//! `−B` (`B` the prover's tail cut `14k`, about `2^{39.5}`, which takes the
//! Ajtai map's two-limb path), one repeated bell-shaped mask, and fresh
//! magnitudes with every sign positive, against uniform masks in
//! `[−B, B]`. The constant and sign classes catch degenerate data, such as
//! an all-zero limb, which the M4 can time differently even under DIT (P7
//! in §2.8 of `specs/whiteout-sigma.md`). For `V`: zero and `−1`; for `c`:
//! zero, all `+2` and all `−2`.
//!
//! Steps, at `D = 64` and at the stride case `D = 256`: the linear parts of y
//! (the lift, witness-claim sums and dense forms), `Q_μ(0, y)`, the polar
//! form, `N·y` with its gadget block, `L·y`, `t_D = A_D G⁻¹(V)`, the attempt
//! fork seed, and the `z`/`v` assembly. The constant-time challenge sampler
//! is tested conditioned on its candidate count, split by the challenge it
//! returns. The Gaussian sampler and the rejection rule are covered by
//! jolt-crypto's `sample_dudect`.
//!
//! Every step runs as the prover's attempts do: on the DIT pool of
//! `whiteout::diagnostics::with_dit` (A11).

#![expect(clippy::print_stdout, reason = "diagnostic tool reports to stdout")]
#![expect(
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "a diagnostic tool on fixed, valid shapes"
)]

use std::hint::black_box;
use std::ops::Range;
use std::process::ExitCode;
use std::time::Instant;

use jolt_algebra::{CyclotomicRing, Prime128OffsetA7F7, Ring};
use jolt_crypto::challenge::sample_challenge_ct_counted;
use jolt_zk::whiteout::diagnostics::{with_dit, AttemptHarness, RING};
use jolt_zk::whiteout::DenseChallenge;
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

type F = Prime128OffsetA7F7;
const CHUNK: usize = 1_000;
const THRESHOLD: f64 = 4.5;
const CROPS: usize = 10;

/// Welch's t over two samples of timings, by running moments.
#[derive(Clone, Default)]
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

/// The raw test and tests cropped at dudect's percentiles
/// `1 − 0.5^{10(k+1)/CROPS}`, thresholds from the first chunk.
#[derive(Default)]
struct Cropped {
    raw: Welch,
    thresholds: Vec<f64>,
    crops: Vec<Welch>,
}

impl Cropped {
    fn push(&mut self, class: usize, x: f64) {
        self.raw.push(class, x);
        for (threshold, test) in self.thresholds.iter().zip(&mut self.crops) {
            if x < *threshold {
                test.push(class, x);
            }
        }
    }

    fn set_thresholds(&mut self, first: &mut [f64]) {
        first.sort_by(f64::total_cmp);
        self.thresholds = (0..CROPS)
            .map(|k| {
                let q = 1.0 - 0.5f64.powf(10.0 * (k as f64 + 1.0) / CROPS as f64);
                first[((first.len() - 1) as f64 * q) as usize]
            })
            .collect();
        self.crops = vec![Welch::default(); CROPS];
    }

    /// The largest `|t|` over the raw and cropped tests, with its test
    /// (`None` for the raw one, else the crop index).
    fn max_t(&self) -> (f64, Option<usize>) {
        std::iter::once((None, &self.raw))
            .chain(
                self.crops
                    .iter()
                    .enumerate()
                    .map(|(k, test)| (Some(k), test)),
            )
            .filter(|(_, test)| test.n[0] > 100.0 && test.n[1] > 100.0)
            .map(|(k, test)| (test.t().abs(), k))
            .fold(
                (0.0, None),
                |best, next| if next.0 > best.0 { next } else { best },
            )
    }
}

fn report(name: &str, test: &Cropped, failures: &mut usize) {
    let (t, crop) = test.max_t();
    let pass = t < THRESHOLD;
    if !pass {
        *failures += 1;
    }
    println!(
        "  {name}: n = {:.0} / {:.0}, mean {:.0} / {:.0} ns, raw |t| = {:.2}, max |t| = {t:.2} ({}) {}",
        test.raw.n[0],
        test.raw.n[1],
        test.raw.mean[0],
        test.raw.mean[1],
        test.raw.t().abs(),
        crop.map_or("raw".to_owned(), |k| format!("crop {k}")),
        if pass { "PASS" } else { "LEAK?" }
    );
}

/// Times `run` on a fixed input vs fresh inputs. The inputs live in
/// `CHUNK` slots allocated once, so both classes are read from the same
/// addresses; `fill` rewrites a slot in place for its class (`true` for
/// fixed), drawing the same randomness for both. Fresh allocations per input
/// gave both classes different memory placement, and crop-only false
/// positives of up to |t| = 15 for two inputs with identical value
/// distributions (A11).
fn fixed_vs_random<T>(
    measurements: usize,
    rng: &mut ChaCha20Rng,
    slot: impl Fn() -> T,
    mut fill: impl FnMut(&mut T, bool, &mut ChaCha20Rng),
    mut run: impl FnMut(&T),
) -> Cropped {
    let mut test = Cropped::default();
    let mut first = Vec::new();
    let mut slots: Vec<(usize, T)> = (0..CHUNK).map(|_| (0, slot())).collect();
    for chunk in 0..measurements.div_ceil(CHUNK) {
        for (class, input) in &mut slots {
            *class = (rng.next_u32() & 1) as usize;
            fill(input, *class == 0, rng);
        }
        for (class, input) in &slots {
            let start = Instant::now();
            run(black_box(input));
            let elapsed = start.elapsed().as_nanos() as f64;
            if chunk == 0 {
                first.push(elapsed);
            }
            test.push(*class, elapsed);
        }
        if chunk == 0 {
            test.set_thresholds(&mut first);
        }
    }
    test
}

/// A mask's fixed class.
#[derive(Clone, Copy)]
enum MaskClass {
    /// Every Gaussian coordinate equal to this value.
    Constant(i64),
    /// One bell-shaped mask (a sum of four uniforms), drawn once and
    /// repeated.
    Bell,
    /// Fresh uniform magnitudes, every sign positive.
    NonNegative,
}

/// Writes a mask into `y`: the Gaussian ranges get `fixed`'s values if
/// given, else uniform values in `[−B, B]`; public coordinates stay zero. A
/// random value is drawn either way and the fixed one selected without a
/// branch, so both classes leave the same state behind. `bell` is the
/// [`MaskClass::Bell`] mask, laid out like `y`.
fn fill_mask(
    y: &mut [i64],
    ranges: &[Range<usize>],
    bound: i64,
    fixed: Option<MaskClass>,
    bell: &[i64],
    rng: &mut ChaCha20Rng,
) {
    let keep_random = -i64::from(fixed.is_none());
    for range in ranges {
        for (slot, &bell) in y[range.clone()].iter_mut().zip(&bell[range.clone()]) {
            let random = (rng.next_u64() % (2 * bound as u64 + 1)) as i64 - bound;
            let fixed = match fixed {
                Some(MaskClass::Constant(value)) => value,
                Some(MaskClass::Bell) => bell,
                Some(MaskClass::NonNegative) => random.abs(),
                None => 0,
            };
            *slot = (random & black_box(keep_random)) | (fixed & !black_box(keep_random));
        }
    }
}

/// A bell-shaped mask within `[−B, B]` on the Gaussian ranges.
fn bell_mask(len: usize, ranges: &[Range<usize>], bound: i64, rng: &mut ChaCha20Rng) -> Vec<i64> {
    let quarter = bound / 4;
    let mut y = vec![0i64; len];
    for range in ranges {
        for slot in &mut y[range.clone()] {
            *slot = (0..4)
                .map(|_| (rng.next_u64() % (2 * quarter as u64 + 1)) as i64 - quarter)
                .sum();
        }
    }
    y
}

fn run_dimension<const D: usize>(
    seed: u64,
    measurements: usize,
    commit_measurements: usize,
    failures: &mut usize,
) {
    println!(" D = {D}:");
    let harness = AttemptHarness::<D>::new(seed).expect("harness");
    let ranges = harness.gaussian_ranges();
    let len = harness.len();
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let bound = harness.tail_bound().expect("tail bound");
    let mask_bound = bound as i64;
    harness
        .steps(|steps| {
            let bell = bell_mask(len, &ranges, mask_bound, &mut rng);
            for (label, class) in [
                ("0", MaskClass::Constant(0)),
                ("+1", MaskClass::Constant(1)),
                ("+B", MaskClass::Constant(mask_bound)),
                ("−B", MaskClass::Constant(-mask_bound)),
                ("bell", MaskClass::Bell),
                ("≥ 0", MaskClass::NonNegative),
            ] {
                let slot = || vec![0i64; len];
                let fill = |y: &mut Vec<i64>, fixed: bool, rng: &mut ChaCha20Rng| {
                    fill_mask(y, &ranges, mask_bound, fixed.then_some(class), &bell, rng);
                };
                let test = fixed_vs_random(measurements, &mut rng, slot, fill, |y| {
                    let _ = black_box(steps.linear_parts(y));
                });
                report(
                    &format!("linear parts of y, fixed {label}"),
                    &test,
                    failures,
                );
                let test = fixed_vs_random(measurements, &mut rng, slot, fill, |y| {
                    let _ = black_box(steps.ajtai(y, bound));
                });
                report(&format!("N·y with gadget, fixed {label}"), &test, failures);
                let test = fixed_vs_random(measurements, &mut rng, slot, fill, |y| {
                    let _ = black_box(steps.l_row(y, bound));
                });
                report(&format!("L·y, fixed {label}"), &test, failures);
            }
            let new_mask = |fixed: Option<MaskClass>, rng: &mut ChaCha20Rng| {
                let mut y = vec![0i64; len];
                fill_mask(&mut y, &ranges, mask_bound, fixed, &bell, rng);
                y
            };
            // Q_μ(0, y) and the polar form, on precomputed parts.
            let parts: Vec<_> = (0..64)
                .map(|_| steps.linear_parts(&new_mask(None, &mut rng)))
                .collect::<Result<_, _>>()?;
            // As many separate copies of the fixed class, so both classes
            // cycle through the same amount of memory.
            let fixed_parts: Vec<_> = (0..64)
                .map(|_| {
                    steps.linear_parts(&new_mask(Some(MaskClass::Constant(mask_bound)), &mut rng))
                })
                .collect::<Result<_, _>>()?;
            let mut test = Cropped::default();
            let mut first = Vec::new();
            for index in 0..measurements {
                let class = (rng.next_u32() & 1) as usize;
                let input = if class == 0 {
                    &fixed_parts[index % 64]
                } else {
                    &parts[index % 64]
                };
                let start = Instant::now();
                let _ = black_box(steps.quadratic_at_zero(black_box(input)));
                let _ = black_box(steps.polar(black_box(input)));
                let elapsed = start.elapsed().as_nanos() as f64;
                if index < CHUNK {
                    first.push(elapsed);
                    if index + 1 == CHUNK {
                        test.set_thresholds(&mut first);
                    }
                }
                test.push(class, elapsed);
            }
            report("Q_μ(0,y) and polar form, fixed +B", &test, failures);

            let v_rings = harness.v_rings();
            let zero_v = || vec![CyclotomicRing::<F, RING>::zero(); v_rings];
            for (label, value) in [("0", F::from_u64(0)), ("−1", -F::from_u64(1))] {
                let fill =
                    |v: &mut Vec<CyclotomicRing<F, RING>>, fixed: bool, rng: &mut ChaCha20Rng| {
                        for ring in v.iter_mut() {
                            let random = CyclotomicRing::random(rng);
                            *ring = if fixed {
                                CyclotomicRing::from_coefficients([value; RING])
                            } else {
                                random
                            };
                        }
                    };
                let test = fixed_vs_random(commit_measurements, &mut rng, zero_v, fill, |v| {
                    let _ = black_box(steps.commit_v(v));
                });
                report(&format!("t_D = A_D G⁻¹(V), fixed {label}"), &test, failures);
            }

            let random_v: Vec<_> = (0..v_rings)
                .map(|_| CyclotomicRing::random(&mut rng))
                .collect();
            let t_d = steps.commit_v(&random_v)?;
            let test = fixed_vs_random(
                measurements,
                &mut rng,
                || [0u8; 32],
                |nonce, fixed, rng| {
                    rng.fill_bytes(nonce);
                    if fixed {
                        *nonce = [0; 32];
                    }
                },
                |nonce| {
                    let _ = black_box(steps.seed(nonce, &t_d));
                },
            );
            report("attempt fork seed, fixed zero nonce", &test, failures);

            let y = new_mask(None, &mut rng);
            let gaussian: usize = ranges.iter().map(|range| range.len()).sum();
            let (mut z, mut v) = (vec![0i64; gaussian], vec![0i64; gaussian]);
            let random_c = |rng: &mut ChaCha20Rng| -> DenseChallenge {
                std::array::from_fn(|_| (rng.next_u32() % 5) as i8 - 2)
            };
            for (label, value) in [("0", 0i8), ("+2", 2), ("−2", -2)] {
                let test = fixed_vs_random(
                    measurements,
                    &mut rng,
                    || [0; 64],
                    |c, fixed, rng| {
                        let random = random_c(rng);
                        *c = if fixed { [value; 64] } else { random };
                    },
                    |c| {
                        let _ = black_box(steps.assemble(c, &y, &mut z, &mut v));
                    },
                );
                report(&format!("z/v assembly, fixed c = {label}"), &test, failures);
            }
            Ok(())
        })
        .expect("steps");
}

fn run_sampler(seed: u64, measurements: usize, failures: &mut usize) {
    // Per candidate count, the accepted challenge's support splits the
    // timings: 0 in the support or not.
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut tests: Vec<Cropped> = (0..4).map(|_| Cropped::default()).collect();
    let mut first: Vec<Vec<f64>> = vec![Vec::new(); 4];
    for _ in 0..measurements {
        let mut sample_seed = [0u8; 32];
        rng.fill_bytes(&mut sample_seed);
        let start = Instant::now();
        let (challenge, count) =
            black_box(sample_challenge_ct_counted(black_box(&sample_seed))).expect("challenge");
        let elapsed = start.elapsed().as_nanos() as f64;
        if let Some(index) = count.checked_sub(1).filter(|&index| index < 4) {
            let class = usize::from(challenge.positions.contains(&0));
            if first[index].len() < CHUNK {
                first[index].push(elapsed);
                if first[index].len() == CHUNK {
                    tests[index].set_thresholds(&mut first[index]);
                }
            }
            tests[index].push(class, elapsed);
        }
    }
    for (index, test) in tests.iter().enumerate() {
        report(
            &format!(
                "challenge sampling, {} candidate(s), 0 ∈ support vs not",
                index + 1
            ),
            test,
            failures,
        );
    }
}

fn main() -> ExitCode {
    println!(
        "DIT: supported {}, set on the attempt pool {}",
        jolt_dit::is_supported(),
        with_dit(jolt_dit::is_enabled).expect("DIT pool")
    );
    let mut args = std::env::args().skip(1);
    let measurements: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(100_000);
    let seeds: u64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(2);
    let commit_measurements: usize = args
        .next()
        .and_then(|a| a.parse().ok())
        .unwrap_or(measurements);
    let mut failures = 0;
    for seed in 0..seeds {
        println!("seed {seed}:");
        let commit = if seed == 0 {
            commit_measurements
        } else {
            measurements
        };
        run_dimension::<64>(seed, measurements, commit, &mut failures);
        run_dimension::<256>(seed, measurements / 4, commit / 4, &mut failures);
        with_dit(|| run_sampler(seed, measurements, &mut failures)).expect("DIT pool");
    }
    println!("{failures} test(s) above |t| = {THRESHOLD}");
    if failures == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
