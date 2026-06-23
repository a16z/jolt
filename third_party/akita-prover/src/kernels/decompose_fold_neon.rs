//! AArch64 NEON kernel for sparse-multiply-accumulate in the decompose-fold
//! pipeline.
//!
//! Called from [`crate::backend::poly_helpers::sparse_mul_acc`] when NEON is available and
//! challenge coefficients have magnitude ≤ 2.  Rotates an i8 digit plane by
//! each challenge position and accumulates into an i32 accumulator using
//! widening add/sub (`SADDW` / `SSUBW`).

use std::arch::aarch64::*;

/// NEON sparse-multiply-accumulate.
///
/// For each challenge term `(pos, coeff)`, rotates the `digit_plane` by `pos`
/// positions in the negacyclic ring (X^D + 1) and adds or subtracts the
/// widened i8 values into the i32 `acc`. Small magnitudes like `+/-2` reuse
/// the unit add/sub kernel multiple times so two-magnitude families
/// (e.g. exact-shell challenges with `max_abs_coeff <= 2`) stay on the NEON
/// fast path.
///
/// # Safety
///
/// - `digit_plane` must point to at least `d` valid i8 values.
/// - `acc` must point to at least `d` valid i32 values.
/// - `d` must be a multiple of 16.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn sparse_mul_acc_neon(
    digit_plane: *const i8,
    acc: *mut i32,
    d: usize,
    positions: &[u32],
    coeffs: &[i8],
) {
    debug_assert!(d.is_multiple_of(16));

    for (&pos, &coeff) in positions.iter().zip(coeffs.iter()) {
        let p = pos as usize;
        let split = d - p;

        match coeff {
            1 => acc_rotated_add(digit_plane, acc, d, p, split),
            -1 => acc_rotated_sub(digit_plane, acc, d, p, split),
            2 => {
                acc_rotated_add(digit_plane, acc, d, p, split);
                acc_rotated_add(digit_plane, acc, d, p, split);
            }
            -2 => {
                acc_rotated_sub(digit_plane, acc, d, p, split);
                acc_rotated_sub(digit_plane, acc, d, p, split);
            }
            _ => {
                for _ in 0..coeff.unsigned_abs() {
                    if coeff > 0 {
                        acc_rotated_add(digit_plane, acc, d, p, split);
                    } else {
                        acc_rotated_sub(digit_plane, acc, d, p, split);
                    }
                }
            }
        }
    }
}

/// Add rotated digit plane: acc[i+p] += digits[i] for i in [0, split),
/// acc[i-split] -= digits[i] for i in [split, D) (negacyclic wrap).
#[inline(always)]
unsafe fn acc_rotated_add(digits: *const i8, acc: *mut i32, d: usize, p: usize, split: usize) {
    // First segment: digits[0..split] -> acc[p..D], ADD
    acc_segment_add(digits, acc.add(p), split);
    // Second segment: digits[split..D] -> acc[0..p], SUB (negacyclic)
    if p > 0 {
        acc_segment_sub(digits.add(split), acc, p);
    }
    let _ = d;
}

/// Sub rotated digit plane: acc[i+p] -= digits[i] for i in [0, split),
/// acc[i-split] += digits[i] for i in [split, D) (negacyclic wrap).
#[inline(always)]
unsafe fn acc_rotated_sub(digits: *const i8, acc: *mut i32, d: usize, p: usize, split: usize) {
    // First segment: digits[0..split] -> acc[p..D], SUB
    acc_segment_sub(digits, acc.add(p), split);
    // Second segment: digits[split..D] -> acc[0..p], ADD (negacyclic)
    if p > 0 {
        acc_segment_add(digits.add(split), acc, p);
    }
    let _ = d;
}

/// Widen i8 source values to i32 and ADD into accumulator.
/// Handles arbitrary length (processes 16 at a time, then remainder).
#[inline(always)]
unsafe fn acc_segment_add(src: *const i8, dst: *mut i32, len: usize) {
    let chunks = len / 16;
    let rem = len % 16;

    for i in 0..chunks {
        let offset = i * 16;
        let v = vld1q_s8(src.add(offset));

        let lo8 = vget_low_s8(v);
        let hi8 = vget_high_s8(v);
        let lo16 = vmovl_s8(lo8);
        let hi16 = vmovl_s8(hi8);

        let s0 = vmovl_s16(vget_low_s16(lo16));
        let s1 = vmovl_s16(vget_high_s16(lo16));
        let s2 = vmovl_s16(vget_low_s16(hi16));
        let s3 = vmovl_s16(vget_high_s16(hi16));

        let d0 = vld1q_s32(dst.add(offset));
        let d1 = vld1q_s32(dst.add(offset + 4));
        let d2 = vld1q_s32(dst.add(offset + 8));
        let d3 = vld1q_s32(dst.add(offset + 12));

        vst1q_s32(dst.add(offset), vaddq_s32(d0, s0));
        vst1q_s32(dst.add(offset + 4), vaddq_s32(d1, s1));
        vst1q_s32(dst.add(offset + 8), vaddq_s32(d2, s2));
        vst1q_s32(dst.add(offset + 12), vaddq_s32(d3, s3));
    }

    let base = chunks * 16;
    for i in 0..rem {
        let val = *src.add(base + i) as i32;
        *dst.add(base + i) += val;
    }
}

/// Widen i8 source values to i32 and SUB from accumulator.
/// Handles arbitrary length (processes 16 at a time, then remainder).
#[inline(always)]
unsafe fn acc_segment_sub(src: *const i8, dst: *mut i32, len: usize) {
    let chunks = len / 16;
    let rem = len % 16;

    for i in 0..chunks {
        let offset = i * 16;
        let v = vld1q_s8(src.add(offset));

        let lo8 = vget_low_s8(v);
        let hi8 = vget_high_s8(v);
        let lo16 = vmovl_s8(lo8);
        let hi16 = vmovl_s8(hi8);

        let s0 = vmovl_s16(vget_low_s16(lo16));
        let s1 = vmovl_s16(vget_high_s16(lo16));
        let s2 = vmovl_s16(vget_low_s16(hi16));
        let s3 = vmovl_s16(vget_high_s16(hi16));

        let d0 = vld1q_s32(dst.add(offset));
        let d1 = vld1q_s32(dst.add(offset + 4));
        let d2 = vld1q_s32(dst.add(offset + 8));
        let d3 = vld1q_s32(dst.add(offset + 12));

        vst1q_s32(dst.add(offset), vsubq_s32(d0, s0));
        vst1q_s32(dst.add(offset + 4), vsubq_s32(d1, s1));
        vst1q_s32(dst.add(offset + 8), vsubq_s32(d2, s2));
        vst1q_s32(dst.add(offset + 12), vsubq_s32(d3, s3));
    }

    let base = chunks * 16;
    for i in 0..rem {
        let val = *src.add(base + i) as i32;
        *dst.add(base + i) -= val;
    }
}
