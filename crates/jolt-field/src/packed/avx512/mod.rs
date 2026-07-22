//! AVX-512 packed backends for Fp32, Fp64, Fp128.
//!
//! Requires AVX-512F + AVX-512DQ. Uses native unsigned comparisons and mask
//! registers for branchless conditionals.

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "ported AVX-512 kernels retain their audited intrinsic-level invariants"
)]

use super::{PackedField};
use crate::ext::FpExt2Config;
use crate::FieldCore;
use crate::{Fp128, Fp32, Fp64};
use core::arch::x86_64::*;
use core::fmt;
use core::mem::transmute;
use core::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

#[inline(always)]
unsafe fn movehdup_epi32_512(x: __m512i) -> __m512i {
    _mm512_castps_si512(_mm512_movehdup_ps(_mm512_castsi512_ps(x)))
}

#[inline(always)]
unsafe fn moveldup_epi32_512(x: __m512i) -> __m512i {
    _mm512_castps_si512(_mm512_moveldup_ps(_mm512_castsi512_ps(x)))
}

/// 64×64→128 schoolbook multiply using 32×32→64 partial products.
/// Returns (hi, lo) representing the 128-bit product.
/// Adapted from plonky3's Goldilocks AVX-512 backend.
#[inline]
unsafe fn mul64_64_512(x: __m512i, y: __m512i) -> (__m512i, __m512i) {
    let x_hi = movehdup_epi32_512(x);
    let y_hi = movehdup_epi32_512(y);

    let mul_ll = _mm512_mul_epu32(x, y);
    let mul_lh = _mm512_mul_epu32(x, y_hi);
    let mul_hl = _mm512_mul_epu32(x_hi, y);
    let mul_hh = _mm512_mul_epu32(x_hi, y_hi);

    let mul_ll_hi = _mm512_srli_epi64::<32>(mul_ll);
    let t0 = _mm512_add_epi64(mul_hl, mul_ll_hi);
    let mask32 = _mm512_set1_epi64(0xFFFF_FFFF_i64);
    let t0_lo = _mm512_and_si512(t0, mask32);
    let t0_hi = _mm512_srli_epi64::<32>(t0);
    let t1 = _mm512_add_epi64(mul_lh, t0_lo);
    let t2 = _mm512_add_epi64(mul_hh, t0_hi);
    let t1_hi = _mm512_srli_epi64::<32>(t1);
    let res_hi = _mm512_add_epi64(t2, t1_hi);

    let t1_lo = moveldup_epi32_512(t1);
    let res_lo = _mm512_mask_blend_epi32(0b0101_0101_0101_0101, t1_lo, mul_ll);

    (res_hi, res_lo)
}

mod fp128;
mod fp32;
mod fp64;
pub(crate) use fp128::*;
pub(crate) use fp32::*;
pub(crate) use fp64::*;
