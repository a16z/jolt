#![cfg_attr(feature = "guest", no_std)]
//! Grumpkin MSM benchmark guest program.
//!
//! This benchmarks multi-scalar multiplication (MSM) on the Grumpkin curve
//! using the inline-accelerated field operations.

use core::hint::black_box;
use jolt::{end_cycle_tracking, start_cycle_tracking};
use jolt_inlines_grumpkin::{GrumpkinFr, GrumpkinPoint, UnwrapOrSpoilProof};

/// MSM size - adjust this constant to change benchmark size.
/// Recommended values: 16 (quick test), 256, 1024, 2048
const MSM_SIZE: usize = 1024;

// Pippenger parameters for baseline (256-bit scalars).
// Keep window <= 16 to fit in a u16.
const SCALAR_BITS: usize = 256;
const BASELINE_WINDOW: usize = 12;
const BASELINE_BUCKETS: usize = 1 << BASELINE_WINDOW;
const BASELINE_WINDOWS: usize = SCALAR_BITS.div_ceil(BASELINE_WINDOW);

// Pippenger parameters for GLV (128-bit scalars).
const GLV_SCALAR_BITS: usize = 128;
const GLV_WINDOW: usize = 8;
const GLV_BUCKETS: usize = 1 << GLV_WINDOW;
const GLV_WINDOWS: usize = GLV_SCALAR_BITS.div_ceil(GLV_WINDOW);

const RUN_BASELINE: bool = false;
const RUN_GLV: bool = false;
const RUN_PIPPENGER: bool = false;
const RUN_GLV_PIPPENGER: bool = false;
const RUN_PIPPENGER_ONLY: bool = false;
const RUN_GLV_PIPPENGER_ONLY: bool = true;

const FR_MODULUS_LIMBS: [u64; 4] = [
    4332616871279656263,
    10917124144477883021,
    13281191951274694749,
    3486998266802970665,
];

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

/// Two-scalar Straus/Shamir multiplication for 128-bit scalars.
#[inline(never)]
fn scalar_mul_2x128(k1: u128, k2: u128, p: &GrumpkinPoint, q: &GrumpkinPoint) -> GrumpkinPoint {
    let table = [GrumpkinPoint::infinity(), p.clone(), q.clone(), p.add(q)];
    let mut res = GrumpkinPoint::infinity();

    for i in (0..128).rev() {
        let mut idx = 0usize;
        if ((k1 >> i) & 1) == 1 {
            idx |= 1;
        }
        if ((k2 >> i) & 1) == 1 {
            idx |= 2;
        }
        if idx == 0 {
            res = res.double();
        } else {
            res = res.double_and_add(&table[idx]);
        }
    }

    res
}

#[inline(always)]
fn pippenger_window_value(scalar: &[u64; 4], window: usize) -> u16 {
    let offset = window * BASELINE_WINDOW;
    let limb = offset / 64;
    let bit = offset % 64;
    if limb >= 4 {
        return 0;
    }
    let mut val = scalar[limb] >> bit;
    if bit + BASELINE_WINDOW > 64 && limb + 1 < 4 {
        val |= scalar[limb + 1] << (64 - bit);
    }
    let mask = (1u64 << BASELINE_WINDOW) - 1;
    (val & mask) as u16
}

#[inline(always)]
fn pippenger_window_value_128(scalar: u128, window: usize) -> u16 {
    let offset = window * GLV_WINDOW;
    if offset >= GLV_SCALAR_BITS {
        return 0;
    }
    let val = scalar >> offset;
    let mask = (1u128 << GLV_WINDOW) - 1;
    (val & mask) as u16
}

#[inline(always)]
fn scalar_to_fr(scalar: &[u64; 4]) -> GrumpkinFr {
    GrumpkinFr::from_u64_arr(scalar).unwrap_or_spoil_proof()
}

#[inline(always)]
fn is_ge_modulus(x: &[u64; 4]) -> bool {
    let m = FR_MODULUS_LIMBS;
    if x[3] > m[3] {
        return true;
    }
    if x[3] < m[3] {
        return false;
    }
    if x[2] > m[2] {
        return true;
    }
    if x[2] < m[2] {
        return false;
    }
    if x[1] > m[1] {
        return true;
    }
    if x[1] < m[1] {
        return false;
    }
    x[0] >= m[0]
}

#[inline(always)]
fn sub_modulus(x: [u64; 4]) -> [u64; 4] {
    let m = FR_MODULUS_LIMBS;
    let mut out = [0u64; 4];
    let mut borrow = 0u128;
    for i in 0..4 {
        let xi = x[i] as u128;
        let mi = m[i] as u128 + borrow;
        if xi >= mi {
            out[i] = (xi - mi) as u64;
            borrow = 0;
        } else {
            out[i] = ((1u128 << 64) + xi - mi) as u64;
            borrow = 1;
        }
    }
    out
}

#[inline(always)]
fn reduce_scalar(mut scalar: [u64; 4]) -> [u64; 4] {
    while is_ge_modulus(&scalar) {
        scalar = sub_modulus(scalar);
    }
    scalar
}

/// GLV-based scalar multiplication: k = k1 + k2 * lambda, so
/// k*P = k1*P + k2*phi(P) where phi is the curve endomorphism.
#[inline(never)]
fn scalar_mul_glv(scalar: &[u64; 4], point: &GrumpkinPoint) -> GrumpkinPoint {
    let k = scalar_to_fr(scalar);
    let decomp = GrumpkinPoint::decompose_scalar(&k);

    let mut p = point.clone();
    let mut q = point.endomorphism();
    if decomp[0].0 {
        p = p.neg();
    }
    if decomp[1].0 {
        q = q.neg();
    }

    scalar_mul_2x128(decomp[0].1, decomp[1].1, &p, &q)
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

/// GLV MSM: sum of scalar_i * point_i using GLV decomposition per scalar.
#[inline(never)]
fn msm_glv(scalars: &[[u64; 4]], points: &[GrumpkinPoint]) -> GrumpkinPoint {
    assert_eq!(scalars.len(), points.len());

    let mut result = GrumpkinPoint::infinity();
    for (scalar, point) in scalars.iter().zip(points.iter()) {
        let term = scalar_mul_glv(scalar, point);
        result = result.add(&term);
    }
    result
}

/// Pippenger MSM (unsigned windowed, affine buckets).
#[inline(never)]
fn msm_pippenger(scalars: &[[u64; 4]], points: &[GrumpkinPoint]) -> GrumpkinPoint {
    assert_eq!(scalars.len(), points.len());

    let mut result = GrumpkinPoint::infinity();
    let mut started = false;

    for window in (0..BASELINE_WINDOWS).rev() {
        let mut buckets: [GrumpkinPoint; BASELINE_BUCKETS] =
            core::array::from_fn(|_| GrumpkinPoint::infinity());

        for (scalar, point) in scalars.iter().zip(points.iter()) {
            let idx = pippenger_window_value(scalar, window) as usize;
            if idx != 0 {
                buckets[idx] = buckets[idx].add(point);
            }
        }

        let mut running = GrumpkinPoint::infinity();
        let mut window_sum = GrumpkinPoint::infinity();
        for i in (1..BASELINE_BUCKETS).rev() {
            running = running.add(&buckets[i]);
            window_sum = window_sum.add(&running);
        }

        if started {
            for _ in 0..BASELINE_WINDOW {
                result = result.double();
            }
        } else if !window_sum.is_infinity() {
            started = true;
        }

        if started {
            result = result.add(&window_sum);
        }
    }

    result
}

/// Pippenger MSM using GLV decomposition (unsigned windowed, affine buckets).
#[inline(never)]
fn msm_pippenger_glv(scalars: &[[u64; 4]], points: &[GrumpkinPoint]) -> GrumpkinPoint {
    assert_eq!(scalars.len(), points.len());

    let n = points.len();
    let mut k1s = [0u128; MSM_SIZE];
    let mut k2s = [0u128; MSM_SIZE];
    let mut p1s: [GrumpkinPoint; MSM_SIZE] = core::array::from_fn(|_| GrumpkinPoint::infinity());
    let mut p2s: [GrumpkinPoint; MSM_SIZE] = core::array::from_fn(|_| GrumpkinPoint::infinity());

    for (i, (scalar, point)) in scalars.iter().zip(points.iter()).enumerate() {
        let k = scalar_to_fr(scalar);
        let decomp = GrumpkinPoint::decompose_scalar(&k);

        let mut p = point.clone();
        let mut q = point.endomorphism();
        if decomp[0].0 {
            p = p.neg();
        }
        if decomp[1].0 {
            q = q.neg();
        }

        k1s[i] = decomp[0].1;
        k2s[i] = decomp[1].1;
        p1s[i] = p;
        p2s[i] = q;
    }

    let mut result = GrumpkinPoint::infinity();
    let mut started = false;

    for window in (0..GLV_WINDOWS).rev() {
        let mut buckets: [GrumpkinPoint; GLV_BUCKETS] =
            core::array::from_fn(|_| GrumpkinPoint::infinity());

        for i in 0..n {
            let idx1 = pippenger_window_value_128(k1s[i], window) as usize;
            if idx1 != 0 {
                buckets[idx1] = buckets[idx1].add(&p1s[i]);
            }
            let idx2 = pippenger_window_value_128(k2s[i], window) as usize;
            if idx2 != 0 {
                buckets[idx2] = buckets[idx2].add(&p2s[i]);
            }
        }

        let mut running = GrumpkinPoint::infinity();
        let mut window_sum = GrumpkinPoint::infinity();
        for i in (1..GLV_BUCKETS).rev() {
            running = running.add(&buckets[i]);
            window_sum = window_sum.add(&running);
        }

        if started {
            for _ in 0..GLV_WINDOW {
                result = result.double();
            }
        } else if !window_sum.is_infinity() {
            started = true;
        }

        if started {
            result = result.add(&window_sum);
        }
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
        *scalar = reduce_scalar(*scalar);
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
    let test_scalar = reduce_scalar([
        0x123456789ABCDEF0,
        0xFEDCBA9876543210,
        0x1111111111111111,
        0x2222222222222222,
    ]);

    start_cycle_tracking("scalar_mul_256bit");
    let _r = black_box(scalar_mul(black_box(&test_scalar), black_box(&g)));
    end_cycle_tracking("scalar_mul_256bit");

    start_cycle_tracking("scalar_mul_glv_2x128");
    let _r = black_box(scalar_mul_glv(black_box(&test_scalar), black_box(&g)));
    end_cycle_tracking("scalar_mul_glv_2x128");

    // ========== MSM benchmark ==========

    // Generate test data
    start_cycle_tracking("msm_setup");
    let scalars = generate_scalars(seed, MSM_SIZE);
    let points = generate_points(seed.wrapping_add(1), MSM_SIZE);
    end_cycle_tracking("msm_setup");

    if RUN_PIPPENGER_ONLY {
        start_cycle_tracking("msm_pippenger");
        let result = black_box(msm_pippenger(black_box(&scalars), black_box(&points)));
        end_cycle_tracking("msm_pippenger");
        return result.to_u64_arr();
    }

    if RUN_GLV_PIPPENGER_ONLY {
        start_cycle_tracking("msm_glv_pippenger");
        let result = black_box(msm_pippenger_glv(black_box(&scalars), black_box(&points)));
        end_cycle_tracking("msm_glv_pippenger");
        return result.to_u64_arr();
    }

    let mut result = GrumpkinPoint::infinity();
    if RUN_BASELINE {
        // Run MSM
        start_cycle_tracking("msm");
        result = black_box(msm(black_box(&scalars), black_box(&points)));
        end_cycle_tracking("msm");
    }

    if RUN_GLV {
        start_cycle_tracking("msm_glv");
        let _glv_result = black_box(msm_glv(black_box(&scalars), black_box(&points)));
        end_cycle_tracking("msm_glv");
    }

    if RUN_PIPPENGER {
        start_cycle_tracking("msm_pippenger");
        let pippenger_result = black_box(msm_pippenger(black_box(&scalars), black_box(&points)));
        end_cycle_tracking("msm_pippenger");
        if !RUN_BASELINE {
            result = pippenger_result;
        }
    }

    if RUN_GLV_PIPPENGER {
        start_cycle_tracking("msm_glv_pippenger");
        let glv_pippenger_result =
            black_box(msm_pippenger_glv(black_box(&scalars), black_box(&points)));
        end_cycle_tracking("msm_glv_pippenger");
        if !RUN_BASELINE && !RUN_PIPPENGER {
            result = glv_pippenger_result;
        }
    }

    // Return result to prevent optimization
    result.to_u64_arr()
}
