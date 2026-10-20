//! dudect-style timing check of the Σ-protocol attempt's secret-dependent
//! steps (W4 acceptance; an intentional diagnostic, not a CI test).
//!
//! ```text
//! cargo run --release -p jolt-zk --features whiteout-prover --example sigma_dudect -- \
//!     [measurements per test, default 10^5] [seeds, default 3]
//! ```
//!
//! Each test is a Welch t-test on raw timings, repeated for every seed;
//! `|t| < 4.5` is dudect's pass threshold. Run it on an idle machine. The
//! Gaussian sampler and the rejection rule are covered by jolt-crypto's
//! `sample_dudect`; this tool covers what the attempt adds around them:
//!
//! - **Ajtai map** (`N·y`): `PreparedMatrix::apply_centered` on a fixed vs a
//!   fresh Gaussian-range mask.
//! - **Commitment to V** (`t_D`): `commit`, decomposition included, on a
//!   fixed vs a fresh field vector.
//! - **Challenge view** (the lift of y into F): `SegmentTable::challenge_view`
//!   on a fixed vs a fresh mask.
//! - **Shift** (`c·s`): `challenge_times_i64` with a fixed vs a fresh
//!   challenge, `s` fixed.
//! - **Challenge sampling**: `sample_challenge_ct` on fresh seeds, split by
//!   the accepted challenge (whether position 0 is in its support). The
//!   number of discarded candidates varies with the seed by design and is
//!   independent of the accepted value, so the split tests that the whole
//!   call is independent of the challenge it returns.
//!
//! Classes are interleaved at random, and every input of a chunk is
//! generated before the chunk is timed.

#![expect(clippy::print_stdout, reason = "diagnostic tool reports to stdout")]
#![expect(clippy::print_stderr, reason = "diagnostic tool reports errors")]
#![expect(
    clippy::expect_used,
    reason = "a diagnostic tool on fixed, valid shapes"
)]

use std::hint::black_box;
use std::process::ExitCode;
use std::time::Instant;

use jolt_algebra::{CyclotomicRing, Prime128OffsetA7F7};
use jolt_crypto::challenge::sample_challenge_ct;
use jolt_crypto::lattice::{
    commit, derive_public_matrix_prefix, AjtaiLevel, GadgetDigits, MatrixShape, MatrixView,
    PreparedMatrix, PublicMatrixSeed, CHALLENGE_RING_DIM,
};
use jolt_zk::whiteout::{
    challenge_times_i64, DenseChallenge, Segment, SegmentMask, SegmentRole, SegmentShape,
    SegmentTable,
};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

type F = Prime128OffsetA7F7;
const D: usize = 64;
/// A Gaussian-range bound: the headline tail cut is about 2^28.
const MASK_BOUND: u64 = 1 << 28;
const CHUNK: usize = 1_000;
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
        "  {name}: n = {:.0} / {:.0}, mean {:.0} / {:.0} ns, t = {t:.2} {}",
        test.n[0],
        test.n[1],
        test.mean[0],
        test.mean[1],
        if pass { "PASS" } else { "LEAK?" }
    );
}

fn centered(rng: &mut ChaCha20Rng, len: usize, bound: u64) -> Vec<i64> {
    (0..len)
        .map(|_| (rng.next_u64() % (2 * bound + 1)) as i64 - bound as i64)
        .collect()
}

/// Times `run` on fixed vs fresh inputs from `fresh`, classes interleaved
/// at random, inputs generated before each chunk is timed.
fn fixed_vs_random<T: Clone>(
    measurements: usize,
    rng: &mut ChaCha20Rng,
    mut fresh: impl FnMut(&mut ChaCha20Rng) -> T,
    mut run: impl FnMut(&T),
) -> Welch {
    let fixed = fresh(rng);
    let mut test = Welch::default();
    for _ in 0..measurements.div_ceil(CHUNK) {
        let inputs: Vec<(usize, T)> = (0..CHUNK)
            .map(|_| {
                let class = (rng.next_u32() & 1) as usize;
                (
                    class,
                    if class == 0 {
                        fixed.clone()
                    } else {
                        fresh(rng)
                    },
                )
            })
            .collect();
        for (class, input) in &inputs {
            let start = Instant::now();
            run(black_box(input));
            test.push(*class, start.elapsed().as_nanos() as f64);
        }
    }
    test
}

fn prepared(rows: usize, width: usize, ring_dim: usize) -> PreparedMatrix<F, D> {
    let shape = MatrixShape {
        rows,
        width,
        ring_dim,
    };
    let data = derive_public_matrix_prefix(
        shape.num_field_elements().expect("shape"),
        &PublicMatrixSeed::shake256_paged_v1([7; 32]),
    )
    .expect("stream");
    PreparedMatrix::new(&MatrixView::new(shape, &data).expect("view")).expect("matrix")
}

fn run(seed: u64, measurements: usize, failures: &mut usize) {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);

    let ajtai = prepared(3, 16, D);
    let test = fixed_vs_random(
        measurements,
        &mut rng,
        |rng| centered(rng, 16 * D, MASK_BOUND),
        |y| {
            let _ = black_box(ajtai.apply_centered(y, MASK_BOUND as u32));
        },
    );
    report("Ajtai map N·y, fixed vs random y", &test, failures);

    let level = AjtaiLevel {
        matrix: MatrixShape {
            rows: 1,
            width: 4 * 32,
            ring_dim: D,
        },
        digits: GadgetDigits {
            log_basis: 4,
            levels: 32,
        },
        message_rings: 4,
        randomness_columns: 0,
    };
    let v_matrix = prepared(1, 4 * 32, D);
    let test = fixed_vs_random(
        measurements,
        &mut rng,
        |rng| {
            (0..4)
                .map(|_| CyclotomicRing::<F, D>::random(rng))
                .collect::<Vec<_>>()
        },
        |v| {
            let _ = black_box(commit(&level, &v_matrix, v, &[]));
        },
    );
    report(
        "commitment t_D = A_D G⁻¹(V), fixed vs random V",
        &test,
        failures,
    );

    let table = SegmentTable::new(vec![Segment {
        role: SegmentRole::Witness,
        len: 16,
        native_dim: 4 * CHALLENGE_RING_DIM,
        log_basis: 4,
        shape: SegmentShape::Dense,
        mask: SegmentMask::Gaussian,
    }])
    .expect("table");
    let test = fixed_vs_random(
        measurements,
        &mut rng,
        |rng| centered(rng, table.num_coefficients(), MASK_BOUND),
        |y| {
            let _ = black_box(table.challenge_view::<F>(y));
        },
    );
    report("challenge view of y, fixed vs random y", &test, failures);

    let s = centered(&mut rng, 16 * CHALLENGE_RING_DIM, 8);
    let test = fixed_vs_random(
        measurements,
        &mut rng,
        |rng| -> DenseChallenge {
            let mut seed = [0u8; 32];
            rng.fill_bytes(&mut seed);
            sample_challenge_ct(&seed)
                .and_then(|c| c.to_dense_ct())
                .expect("challenge")
        },
        |c| {
            let _ = black_box(challenge_times_i64(c, CHALLENGE_RING_DIM, &s));
        },
    );
    report("shift c·s, fixed vs random c", &test, failures);

    let mut test = Welch::default();
    let mut seeds = vec![[0u8; 32]; CHUNK];
    for _ in 0..measurements.div_ceil(CHUNK) {
        for seed in &mut seeds {
            rng.fill_bytes(seed);
        }
        for seed in &seeds {
            let start = Instant::now();
            let challenge = black_box(sample_challenge_ct(black_box(seed))).expect("challenge");
            let elapsed = start.elapsed().as_nanos() as f64;
            test.push(usize::from(challenge.positions.contains(&0)), elapsed);
        }
    }
    report("challenge sampling, 0 ∈ support vs not", &test, failures);
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let measurements: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(100_000);
    let seeds: u64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(3);
    let mut failures = 0;
    for seed in 0..seeds {
        println!("seed {seed}:");
        run(seed, measurements, &mut failures);
    }
    println!("{failures} test(s) above |t| = {THRESHOLD}");
    if failures == 0 {
        ExitCode::SUCCESS
    } else {
        eprintln!("sigma_dudect: timing differences found");
        ExitCode::FAILURE
    }
}
