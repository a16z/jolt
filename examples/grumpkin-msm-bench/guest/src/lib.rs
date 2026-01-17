#![cfg_attr(feature = "guest", no_std)]
//! Grumpkin MSM benchmark guest program.
//!
//! This benchmarks multi-scalar multiplication (MSM) on the Grumpkin curve
//! using the inline-accelerated field operations.

use core::hint::black_box;
use jolt::{end_cycle_tracking, start_cycle_tracking};
use jolt_inlines_grumpkin::GrumpkinPoint;

/// MSM size - adjust this constant to change benchmark size.
/// Recommended values: 16 (quick test), 256, 1024, 2048
const MSM_SIZE: usize = 2048;

/// Simple double-and-add scalar multiplication.
/// Returns scalar * point.
#[inline(never)]
fn scalar_mul(scalar: &[u64; 4], point: &GrumpkinPoint) -> GrumpkinPoint {
    let mut result = GrumpkinPoint::infinity();

    // Process each limb from most significant to least significant
    for limb_idx in (0..4).rev() {
        let limb = scalar[limb_idx];
        // Process each bit in the limb
        for bit_idx in (0..64).rev() {
            let bit = (limb >> bit_idx) & 1;
            if bit == 1 {
                result = result.double_and_add(point);
            } else {
                result = result.double();
            }
        }
    }

    result
}

/// Simple MSM implementation: sum of scalar_i * point_i
/// Uses the existing GrumpkinPoint operations which leverage division inlines.
#[inline(never)]
fn msm(scalars: &[[u64; 4]], points: &[GrumpkinPoint]) -> GrumpkinPoint {
    assert_eq!(scalars.len(), points.len());

    let mut result = GrumpkinPoint::infinity();
    for (scalar, point) in scalars.iter().zip(points.iter()) {
        let term = scalar_mul(scalar, point);
        result = result.add(&term);
    }
    result
}

/// Generate deterministic test scalars using a simple LCG.
fn generate_scalars(seed: u64, count: usize) -> [[u64; 4]; MSM_SIZE] {
    let mut scalars = [[0u64; 4]; MSM_SIZE];
    let (a, c) = (6364136223846793005u64, 1442695040888963407u64);
    let mut state = seed;

    for scalar in scalars.iter_mut().take(count.min(MSM_SIZE)) {
        for limb in scalar.iter_mut() {
            state = state.wrapping_mul(a).wrapping_add(c);
            *limb = state;
        }
    }
    scalars
}

/// Generate deterministic test points by scalar multiplication of generator.
fn generate_points(seed: u64, count: usize) -> [GrumpkinPoint; MSM_SIZE] {
    // Initialize with infinity - we'll replace these
    let mut points: [GrumpkinPoint; MSM_SIZE] = core::array::from_fn(|_| GrumpkinPoint::infinity());
    let generator = GrumpkinPoint::generator();

    let (a, c) = (6364136223846793005u64, 1442695040888963407u64);
    let mut state = seed;

    for point in points.iter_mut().take(count.min(MSM_SIZE)) {
        // Generate a small scalar for point generation (just use one limb)
        state = state.wrapping_mul(a).wrapping_add(c);
        let small_scalar = [state, 0, 0, 0];
        *point = scalar_mul(&small_scalar, &generator);
    }
    points
}

#[jolt::provable(
    max_output_size = 1024,
    memory_size = 67108864,    // 64 MB
    stack_size = 16777216,     // 16 MB
    max_trace_length = 500000000
)]
fn grumpkin_msm_bench(seed: u64) -> [u64; 8] {
    // ========== Individual operation benchmarks ==========

    let g = GrumpkinPoint::generator();
    let g2 = g.double();

    // Benchmark: Point addition
    start_cycle_tracking("point_add");
    let _r = black_box(black_box(&g).add(black_box(&g2)));
    end_cycle_tracking("point_add");

    // Benchmark: Point doubling
    start_cycle_tracking("point_double");
    let _r = black_box(black_box(&g).double());
    end_cycle_tracking("point_double");

    // Benchmark: Double-and-add (2P + Q)
    start_cycle_tracking("point_double_and_add");
    let _r = black_box(black_box(&g).double_and_add(black_box(&g2)));
    end_cycle_tracking("point_double_and_add");

    // Benchmark: Field division (Fq) - uses inline
    let a = g.x();
    let b = g2.x();
    start_cycle_tracking("fq_div");
    let _r = black_box(black_box(&a).div(black_box(&b)));
    end_cycle_tracking("fq_div");

    // Benchmark: Field multiplication (Fq)
    start_cycle_tracking("fq_mul");
    let _r = black_box(black_box(&a).mul(black_box(&b)));
    end_cycle_tracking("fq_mul");

    // Benchmark: Field squaring (Fq)
    start_cycle_tracking("fq_square");
    let _r = black_box(black_box(&a).square());
    end_cycle_tracking("fq_square");

    // ========== Single scalar multiplication ==========

    // Use a 256-bit scalar
    let test_scalar: [u64; 4] = [
        0x123456789ABCDEF0,
        0xFEDCBA9876543210,
        0x1111111111111111,
        0x2222222222222222,
    ];

    start_cycle_tracking("scalar_mul_256bit");
    let _r = black_box(scalar_mul(black_box(&test_scalar), black_box(&g)));
    end_cycle_tracking("scalar_mul_256bit");

    // ========== MSM benchmark ==========

    // Generate test data
    start_cycle_tracking("msm_setup");
    let scalars = generate_scalars(seed, MSM_SIZE);
    let points = generate_points(seed.wrapping_add(1), MSM_SIZE);
    end_cycle_tracking("msm_setup");

    // Run MSM
    start_cycle_tracking("msm");
    let result = black_box(msm(black_box(&scalars), black_box(&points)));
    end_cycle_tracking("msm");

    // Return result to prevent optimization
    result.to_u64_arr()
}
