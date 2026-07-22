//! AVX2 packed backends for Fp32, Fp64, Fp128.
//!
//! Techniques adapted from plonky2 (Goldilocks) and plonky3 (Mersenne-31).

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "ported AVX2 kernels retain their audited intrinsic-level invariants"
)]

use super::PackedField;
use crate::ext::FpExt2Config;
use crate::FieldCore;
use crate::{Fp128, Fp32, Fp64};
use core::arch::x86_64::*;
use core::fmt;
use core::mem::transmute;
use core::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

/// Duplicate high 32 bits of each 64-bit lane into the low 32 bits.
/// Uses the float `movehdup` instruction which runs on port 5 (doesn't compete
/// with multiply on ports 0/1).
#[inline(always)]
unsafe fn movehdup_epi32(x: __m256i) -> __m256i {
    _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(x)))
}

#[inline(always)]
unsafe fn moveldup_epi32(x: __m256i) -> __m256i {
    _mm256_castps_si256(_mm256_moveldup_ps(_mm256_castsi256_ps(x)))
}

/// 64×64→128 schoolbook multiply using 32×32→64 partial products.
/// Returns (hi, lo) representing the 128-bit product.
#[inline]
unsafe fn mul64_64_256(x: __m256i, y: __m256i) -> (__m256i, __m256i) {
    let x_hi = movehdup_epi32(x);
    let y_hi = movehdup_epi32(y);

    let mul_ll = _mm256_mul_epu32(x, y);
    let mul_lh = _mm256_mul_epu32(x, y_hi);
    let mul_hl = _mm256_mul_epu32(x_hi, y);
    let mul_hh = _mm256_mul_epu32(x_hi, y_hi);

    let mul_ll_hi = _mm256_srli_epi64::<32>(mul_ll);
    let t0 = _mm256_add_epi64(mul_hl, mul_ll_hi);
    let mask32 = _mm256_set1_epi64x(0xFFFF_FFFF_i64);
    let t0_lo = _mm256_and_si256(t0, mask32);
    let t0_hi = _mm256_srli_epi64::<32>(t0);
    let t1 = _mm256_add_epi64(mul_lh, t0_lo);
    let t2 = _mm256_add_epi64(mul_hh, t0_hi);
    let t1_hi = _mm256_srli_epi64::<32>(t1);
    let res_hi = _mm256_add_epi64(t2, t1_hi);

    let t1_lo = moveldup_epi32(t1);
    let res_lo = _mm256_blend_epi32::<0b10101010>(mul_ll, t1_lo);

    (res_hi, res_lo)
}

mod fp128;
mod fp32;
mod fp64;
pub(crate) use fp128::*;
pub(crate) use fp32::*;
pub(crate) use fp64::*;
