//! AArch64 NEON packed backends for Fp32, Fp64, Fp128.

#![expect(
    clippy::undocumented_unsafe_blocks,
    reason = "ported NEON kernels retain their audited intrinsic-level invariants"
)]

use super::{PackedField, PackedValue};
use crate::ext::FpExt2Config;
use crate::FieldCore;
use crate::{Fp128, Fp32, Fp64};
use core::arch::aarch64::{
    uint32x2_t, uint32x4_t, uint64x2_t, vaddq_u32, vaddq_u64, vandq_u32, vandq_u64, vbslq_u64,
    vcltq_u32, vcltq_u64, vcombine_u32, vdup_n_u32, vdupq_n_s64, vdupq_n_u32, vdupq_n_u64,
    vget_low_u32, vminq_u32, vmlsq_u32, vmovn_u64, vmull_high_u32, vmull_u32, vmulq_u32, vorrq_u64,
    vqdmulhq_s32, vreinterpretq_s32_u32, vreinterpretq_u32_s32, vshlq_n_u64, vshlq_u64,
    vshrq_n_u32, vshrq_n_u64, vsubq_u32, vsubq_u64,
};
use core::fmt;
use core::mem::transmute;
use core::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

#[inline(always)]
fn to_vec(x: [u64; 2]) -> uint64x2_t {
    unsafe { transmute::<[u64; 2], uint64x2_t>(x) }
}

#[inline(always)]
fn from_vec(v: uint64x2_t) -> [u64; 2] {
    unsafe { transmute::<uint64x2_t, [u64; 2]>(v) }
}

#[inline(always)]
fn mask_to_bit(mask: uint64x2_t) -> uint64x2_t {
    unsafe { vandq_u64(mask, vdupq_n_u64(1)) }
}

mod fp128;
mod fp32;
mod fp64;
pub(crate) use fp128::*;
pub(crate) use fp32::*;
pub(crate) use fp64::*;
