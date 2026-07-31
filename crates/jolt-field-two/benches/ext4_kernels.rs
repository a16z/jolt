//! Bench gate for the `Fp32` degree-4 ext-mul kernels (checkpoint 6
//! acceptance): the generic coefficient-formula schedule (what the crate
//! ships as the `PseudoMersenne` hook default) vs a local port of the
//! baseline's fused u128-accumulation `Fp32` override, on batched degree-4
//! muls and squares over `Prime32Offset99`. (The original jolt-field
//! baseline, which shipped the fused override, timed identically to the
//! local port while both crates coexisted.)
//!
//! Outcome recorded in specs/jolt-field-rebuild.md: the fused port LOST on aarch64/Apple M4
//! (generic ≈ 2.5x faster on mul, ≈ 1.85x on square; the port reproduces
//! the baseline override's timing exactly), so the override was dropped
//! and the crate keeps the generic defaults. This harness stays as the
//! reproducible evidence; rerun it before reintroducing an override.
//!
//! Run: `cargo bench -p jolt-field-two --features solinas --bench ext4_kernels`

#![expect(clippy::print_stdout, reason = "bench harness: stdout is the report")]

use jolt_field_two as two;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::hint::black_box;
use std::time::Instant;

use two::{CanonicalEncoding, Field, Ring};

type Fp = two::Prime32Offset99;
type E4 = two::FpExt4<Fp>;

const N: usize = 1 << 12;
const REPS: usize = 100;
const TRIALS: usize = 7;

/// Widening product of canonical `Fp32` values, exact in `u64`
/// (`a·b < P² < 2^64`), widened to `u128` for column accumulation.
#[inline(always)]
fn product(a: Fp, b: Fp) -> u128 {
    ((a.to_limbs() as u64) * (b.to_limbs() as u64)) as u128
}

const P: u32 = 4_294_967_197; // 2^32 − 99

/// Port of the baseline's fused `Fp32` degree-4 multiply: accumulate the
/// raw products of each output coefficient in a `u128`, reduce once.
///
/// Bounds (every term `< P² < 2^64`, sums evaluated left to right):
/// `c0 ≤ 7·P² < 2^67`; `c1 ≤ 6·P²`; `c2` has a `P²` bias ≥ the single
/// subtrahend `p33`; `c3` has a `2·P²` bias ≥ `p23 + p32`. No `u128` wrap,
/// biases are multiples of `P`, so results equal the generic schedule's.
#[inline(always)]
fn fused_mul(a: [Fp; 4], b: [Fp; 4]) -> [Fp; 4] {
    let [a0, a1, a2, a3] = a;
    let [b0, b1, b2, b3] = b;
    let msq = (P as u128) * (P as u128);
    [
        Fp::from_u128_reduced(
            product(a0, b0) + 2 * (product(a1, b1) + product(a2, b2) + product(a3, b3)),
        ),
        Fp::from_u128_reduced(
            product(a0, b1)
                + product(a1, b0)
                + product(a1, b2)
                + product(a2, b1)
                + product(a2, b3)
                + product(a3, b2),
        ),
        Fp::from_u128_reduced(
            product(a0, b2)
                + product(a2, b0)
                + product(a1, b1)
                + product(a1, b3)
                + product(a3, b1)
                + msq
                - product(a3, b3),
        ),
        Fp::from_u128_reduced(
            product(a0, b3) + product(a3, b0) + product(a1, b2) + product(a2, b1) + 2 * msq
                - product(a2, b3)
                - product(a3, b2),
        ),
    ]
}

/// Port of the baseline's fused `Fp32` degree-4 squaring (10 products);
/// same bound structure as [`fused_mul`], every column `< 8·P² < 2^67`.
#[inline(always)]
fn fused_square(a: [Fp; 4]) -> [Fp; 4] {
    let [a0, a1, a2, a3] = a;
    let msq = (P as u128) * (P as u128);
    let a0_square = product(a0, a0);
    let a1_square = product(a1, a1);
    let a2_square = product(a2, a2);
    let a3_square = product(a3, a3);
    let a0a1 = product(a0, a1);
    let a0a2 = product(a0, a2);
    let a0a3 = product(a0, a3);
    let a1a2 = product(a1, a2);
    let a1a3 = product(a1, a3);
    let a2a3 = product(a2, a3);
    [
        Fp::from_u128_reduced(a0_square + 2 * (a1_square + a2_square + a3_square)),
        Fp::from_u128_reduced(2 * (a0a1 + a1a2 + a2a3)),
        Fp::from_u128_reduced(2 * a0a2 + a1_square + 2 * a1a3 + msq - a3_square),
        Fp::from_u128_reduced(2 * (a0a3 + a1a2 + msq - a2a3)),
    ]
}

fn measure<T: Copy, R: Copy>(inputs: &[T], mut op: impl FnMut(T) -> R) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..TRIALS {
        let start = Instant::now();
        for _ in 0..REPS {
            for &x in inputs {
                let _ = black_box(op(x));
            }
        }
        let ns = start.elapsed().as_nanos() as f64 / (REPS * inputs.len()) as f64;
        best = best.min(ns);
    }
    best
}

fn main() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xE4B_E4B);
    let pairs: Vec<(E4, E4)> = (0..N)
        .map(|_| (E4::random(&mut rng), E4::random(&mut rng)))
        .collect();
    // Sanity: the fused port agrees with the wired generic path.
    for (a, b) in pairs.iter().take(64) {
        assert_eq!((*a * *b).coeffs, fused_mul(a.coeffs, b.coeffs));
        assert_eq!(Ring::square(a).coeffs, fused_square(a.coeffs));
    }

    let generic_mul_ns = measure(&pairs, |(a, b)| (a * b).coeffs[0]);
    let fused_mul_ns = measure(&pairs, |(a, b)| fused_mul(a.coeffs, b.coeffs)[0]);

    let generic_sq_ns = measure(&pairs, |(a, _)| Ring::square(&a).coeffs[0]);
    let fused_sq_ns = measure(&pairs, |(a, _)| fused_square(a.coeffs)[0]);

    println!("ext4 over Prime32Offset99, {N} elements x {REPS} reps, best of {TRIALS}");
    println!("  mul    generic default (wired): {generic_mul_ns:7.2} ns/op");
    println!("  mul    fused port (dropped)   : {fused_mul_ns:7.2} ns/op");
    println!(
        "  mul    fused/generic           : {:.2}x",
        fused_mul_ns / generic_mul_ns
    );
    println!("  square generic default (wired): {generic_sq_ns:7.2} ns/op");
    println!("  square fused port (dropped)   : {fused_sq_ns:7.2} ns/op");
    println!(
        "  square fused/generic           : {:.2}x",
        fused_sq_ns / generic_sq_ns
    );
}
