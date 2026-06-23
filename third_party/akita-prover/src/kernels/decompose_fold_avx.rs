//! x86_64 AVX2 kernel for sparse-multiply-accumulate in the decompose-fold
//! pipeline.
//!
//! Called from [`crate::backend::poly_helpers::sparse_mul_acc`] when AVX2 is
//! available and challenge coefficients have magnitude ≤ 2. Mirrors
//! [`super::decompose_fold_neon`]: rotates an `i8` digit plane by each
//! challenge position and accumulates into an `i32` accumulator using
//! widening add/sub (`vpmovsxbd` then `vpaddd`/`vpsubd`).

use std::arch::x86_64::*;

/// AVX2 sparse multiply-accumulate.
///
/// For each challenge term `(pos, coeff)`, rotates the `digit_plane` by `pos`
/// positions in the negacyclic ring (X^D + 1) and adds or subtracts the
/// widened `i8` values into the `i32` `acc`. Small magnitudes like `±2` reuse
/// the unit add/sub kernel multiple times so two-magnitude families
/// (e.g. exact-shell challenges with `max_abs_coeff <= 2`) stay on the SIMD
/// fast path.
///
/// # Safety
///
/// - `digit_plane` must point to at least `d` valid `i8` values.
/// - `acc` must point to at least `d` valid `i32` values.
/// - `d` must be a multiple of 16.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn sparse_mul_acc_avx(
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

/// `acc[i+p] += digits[i]` for `i in [0, split)`,
/// `acc[i-split] -= digits[i]` for `i in [split, D)` (negacyclic wrap).
//
// `d` is unused here (the segment lengths `split` and `p` already cover the
// whole `[0, D)` range), but the parameter is kept for API symmetry with the
// NEON kernel in [`super::decompose_fold_neon`].
#[inline(always)]
unsafe fn acc_rotated_add(digits: *const i8, acc: *mut i32, d: usize, p: usize, split: usize) {
    acc_segment_add(digits, acc.add(p), split);
    if p > 0 {
        acc_segment_sub(digits.add(split), acc, p);
    }
    let _ = d;
}

/// `acc[i+p] -= digits[i]` for `i in [0, split)`,
/// `acc[i-split] += digits[i]` for `i in [split, D)` (negacyclic wrap).
//
// `d` is unused; see `acc_rotated_add` for the NEON-parity rationale.
#[inline(always)]
unsafe fn acc_rotated_sub(digits: *const i8, acc: *mut i32, d: usize, p: usize, split: usize) {
    acc_segment_sub(digits, acc.add(p), split);
    if p > 0 {
        acc_segment_add(digits.add(split), acc, p);
    }
    let _ = d;
}

/// Widen `i8` source values to `i32` and add into accumulator.
/// Processes 16 elements per outer iteration (two 8-wide `__m256i` chunks),
/// then handles any remainder scalar.
#[inline(always)]
unsafe fn acc_segment_add(src: *const i8, dst: *mut i32, len: usize) {
    let chunks = len / 16;
    let rem = len % 16;

    for i in 0..chunks {
        let offset = i * 16;
        // Load 16 i8s.
        let v = _mm_loadu_si128(src.add(offset).cast());
        // Widen bytes 0..7 to 8 i32 lanes (signed).
        let lo = _mm256_cvtepi8_epi32(v);
        // Move bytes 8..15 to position 0..7 then widen.
        let v_hi = _mm_srli_si128::<8>(v);
        let hi = _mm256_cvtepi8_epi32(v_hi);

        let d0 = _mm256_loadu_si256(dst.add(offset).cast());
        let d1 = _mm256_loadu_si256(dst.add(offset + 8).cast());

        _mm256_storeu_si256(dst.add(offset).cast(), _mm256_add_epi32(d0, lo));
        _mm256_storeu_si256(dst.add(offset + 8).cast(), _mm256_add_epi32(d1, hi));
    }

    let base = chunks * 16;
    for i in 0..rem {
        let val = *src.add(base + i) as i32;
        *dst.add(base + i) += val;
    }
}

/// Widen `i8` source values to `i32` and subtract from accumulator.
#[inline(always)]
unsafe fn acc_segment_sub(src: *const i8, dst: *mut i32, len: usize) {
    let chunks = len / 16;
    let rem = len % 16;

    for i in 0..chunks {
        let offset = i * 16;
        let v = _mm_loadu_si128(src.add(offset).cast());
        let lo = _mm256_cvtepi8_epi32(v);
        let v_hi = _mm_srli_si128::<8>(v);
        let hi = _mm256_cvtepi8_epi32(v_hi);

        let d0 = _mm256_loadu_si256(dst.add(offset).cast());
        let d1 = _mm256_loadu_si256(dst.add(offset + 8).cast());

        _mm256_storeu_si256(dst.add(offset).cast(), _mm256_sub_epi32(d0, lo));
        _mm256_storeu_si256(dst.add(offset + 8).cast(), _mm256_sub_epi32(d1, hi));
    }

    let base = chunks * 16;
    for i in 0..rem {
        let val = *src.add(base + i) as i32;
        *dst.add(base + i) -= val;
    }
}
