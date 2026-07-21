//! Wide unreduced field accumulators for carry-free signed addition.
//!
//! Each type splits a canonical field element into 16-bit limbs stored in
//! `i32` slots.  Addition and negation are element-wise i32 ops — no carry
//! propagation, no modular reduction.  Reduction back to canonical form
//! happens once after accumulation via
//! [`reduce`](crate::unreduced::Fp128x8i32::reduce).
//!
//! The i32 overflow budget is `i32::MAX / u16::MAX ≈ 32,769` signed
//! additions before any limb can overflow.

#![cfg_attr(
    target_arch = "aarch64",
    expect(
        clippy::undocumented_unsafe_blocks,
        reason = "ported NEON accumulator operations retain their audited lane invariants"
    )
)]

use std::ops::{Add, AddAssign, Neg, Sub, SubAssign};

use crate::{AdditiveGroup, CanonicalField, FieldCore};

use super::prime::{Fp128, Fp32, Fp64};

mod accum;
mod native_algebra;
pub use accum::*;

/// Wide unreduced accumulator for `Fp32`: 2 × i32 limbs (16-bit data each).
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct Fp32x2i32(pub [i32; 2]);

impl Fp32x2i32 {
    /// Additive identity accumulator.
    pub const ZERO: Self = Self([0; 2]);

    /// Returns the zero accumulator.
    #[inline]
    pub fn zero() -> Self {
        Self::ZERO
    }
}

impl<const P: u32> From<Fp32<P>> for Fp32x2i32 {
    #[inline]
    fn from(x: Fp32<P>) -> Self {
        let v = x.0;
        Self([(v & 0xFFFF) as i32, (v >> 16) as i32])
    }
}

impl Fp32x2i32 {
    /// Multiply every limb by a small signed scalar.
    ///
    /// Safe when `|small| * max_limb_magnitude` fits in i32. After `From`,
    /// limbs are in `[0, 0xFFFF]`, so `|small| ≤ 32_767` is safe for a single
    /// product.  For accumulation of `k` scaled values, require
    /// `k * |small| * 0xFFFF < i32::MAX`, i.e. roughly `k * |small| < 32_768`.
    #[inline]
    pub fn scale_i32(self, small: i32) -> Self {
        Self([self.0[0] * small, self.0[1] * small])
    }

    /// Reduce back to canonical `Fp32<P>`.
    ///
    /// Carry-propagates the i32 limbs into a signed value, normalizes to
    /// `[0, p)`, and returns the canonical field element.
    #[inline]
    pub fn reduce<const P: u32>(self) -> Fp32<P> {
        let [l0, l1] = self.0;
        // Carry-propagate: value = l0 + l1 * 2^16
        let wide = l0 as i64 + (l1 as i64) * (1i64 << 16);
        // Normalize to [0, p)
        let p = P as i64;
        let normalized = ((wide % p) + p) % p;
        Fp32::from_canonical_u32(normalized as u32)
    }
}

impl Add for Fp32x2i32 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1]])
    }
}

impl AddAssign for Fp32x2i32 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0[0] += rhs.0[0];
        self.0[1] += rhs.0[1];
    }
}

impl Sub for Fp32x2i32 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([self.0[0] - rhs.0[0], self.0[1] - rhs.0[1]])
    }
}

impl SubAssign for Fp32x2i32 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0[0] -= rhs.0[0];
        self.0[1] -= rhs.0[1];
    }
}

impl Neg for Fp32x2i32 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self([-self.0[0], -self.0[1]])
    }
}

/// Wide unreduced accumulator for `Fp64`: 4 × i32 limbs (16-bit data each).
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct Fp64x4i32(pub [i32; 4]);

impl Fp64x4i32 {
    /// Additive identity accumulator.
    pub const ZERO: Self = Self([0; 4]);

    /// Returns the zero accumulator.
    #[inline]
    pub fn zero() -> Self {
        Self::ZERO
    }
}

impl<const P: u64> From<Fp64<P>> for Fp64x4i32 {
    #[inline]
    fn from(x: Fp64<P>) -> Self {
        let v = x.0;
        Self([
            (v & 0xFFFF) as i32,
            ((v >> 16) & 0xFFFF) as i32,
            ((v >> 32) & 0xFFFF) as i32,
            ((v >> 48) & 0xFFFF) as i32,
        ])
    }
}

impl Fp64x4i32 {
    /// Multiply every limb by a small signed scalar. See [`Fp32x2i32::scale_i32`].
    #[inline]
    pub fn scale_i32(self, small: i32) -> Self {
        Self([
            self.0[0] * small,
            self.0[1] * small,
            self.0[2] * small,
            self.0[3] * small,
        ])
    }

    /// Reduce back to canonical `Fp64<P>`.
    #[inline]
    pub fn reduce<const P: u64>(self) -> Fp64<P> {
        let [l0, l1, l2, l3] = self.0;
        // Carry-propagate: value = l0 + l1*2^16 + l2*2^32 + l3*2^48
        let wide = l0 as i128
            + (l1 as i128) * (1i128 << 16)
            + (l2 as i128) * (1i128 << 32)
            + (l3 as i128) * (1i128 << 48);
        let p = P as i128;
        let normalized = ((wide % p) + p) % p;
        Fp64::<P>::from_canonical_u64(normalized as u64)
    }
}

#[cfg(target_arch = "aarch64")]
impl Add for Fp64x4i32 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        unsafe {
            use std::arch::aarch64::*;
            let a = vld1q_s32(self.0.as_ptr());
            let b = vld1q_s32(rhs.0.as_ptr());
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), vaddq_s32(a, b));
            Self(out)
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl AddAssign for Fp64x4i32 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

#[cfg(target_arch = "aarch64")]
impl Sub for Fp64x4i32 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        unsafe {
            use std::arch::aarch64::*;
            let a = vld1q_s32(self.0.as_ptr());
            let b = vld1q_s32(rhs.0.as_ptr());
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), vsubq_s32(a, b));
            Self(out)
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl SubAssign for Fp64x4i32 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

#[cfg(target_arch = "aarch64")]
impl Neg for Fp64x4i32 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        unsafe {
            use std::arch::aarch64::*;
            let a = vld1q_s32(self.0.as_ptr());
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), vnegq_s32(a));
            Self(out)
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl Add for Fp64x4i32 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
            self.0[3] + rhs.0[3],
        ])
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl AddAssign for Fp64x4i32 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0[0] += rhs.0[0];
        self.0[1] += rhs.0[1];
        self.0[2] += rhs.0[2];
        self.0[3] += rhs.0[3];
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl Sub for Fp64x4i32 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
            self.0[3] - rhs.0[3],
        ])
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl SubAssign for Fp64x4i32 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0[0] -= rhs.0[0];
        self.0[1] -= rhs.0[1];
        self.0[2] -= rhs.0[2];
        self.0[3] -= rhs.0[3];
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl Neg for Fp64x4i32 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self([-self.0[0], -self.0[1], -self.0[2], -self.0[3]])
    }
}

/// Wide unreduced accumulator for `Fp128`: 8 × i32 limbs (16-bit data each).
///
/// On AVX2, one element fits a single 256-bit YMM register.  On NEON, it
/// spans two 128-bit Q registers.  All arithmetic is carry-free element-wise
/// i32 operations.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct Fp128x8i32(pub [i32; 8]);

impl Fp128x8i32 {
    /// Additive identity accumulator.
    pub const ZERO: Self = Self([0; 8]);

    /// Returns the zero accumulator.
    #[inline]
    pub fn zero() -> Self {
        Self::ZERO
    }
}

impl<const P: u128> From<Fp128<P>> for Fp128x8i32 {
    #[inline]
    fn from(x: Fp128<P>) -> Self {
        let lo = x.0[0];
        let hi = x.0[1];
        Self([
            (lo & 0xFFFF) as i32,
            ((lo >> 16) & 0xFFFF) as i32,
            ((lo >> 32) & 0xFFFF) as i32,
            ((lo >> 48) & 0xFFFF) as i32,
            (hi & 0xFFFF) as i32,
            ((hi >> 16) & 0xFFFF) as i32,
            ((hi >> 32) & 0xFFFF) as i32,
            ((hi >> 48) & 0xFFFF) as i32,
        ])
    }
}

impl Fp128x8i32 {
    /// Multiply every limb by a small signed scalar. See [`Fp32x2i32::scale_i32`].
    #[inline]
    pub fn scale_i32(self, small: i32) -> Self {
        Self([
            self.0[0] * small,
            self.0[1] * small,
            self.0[2] * small,
            self.0[3] * small,
            self.0[4] * small,
            self.0[5] * small,
            self.0[6] * small,
            self.0[7] * small,
        ])
    }

    /// Reduce back to canonical `Fp128<P>`.
    ///
    /// Carry-propagates the 8 × i32 limbs into unsigned u64 limbs, then
    /// applies Solinas reduction.
    #[inline]
    pub fn reduce<const P: u128>(self) -> Fp128<P> {
        let limbs = self.0;

        // Carry-propagate from low to high, accumulating into i64 slots.
        // Each i32 limb can be in [-32769*65535, 32769*65535] ≈ ±2^31.
        // After propagation, each 16-bit "digit" is in [0, 65535] and we
        // may have a signed residual in the top that overflows 128 bits.
        let mut carry: i64 = 0;
        let mut digits = [0u16; 8];
        for i in 0..8 {
            let v = limbs[i] as i64 + carry;
            // Arithmetic right-shift to propagate sign correctly
            digits[i] = (v & 0xFFFF) as u16;
            carry = v >> 16;
        }

        // Reassemble into u64 limbs
        let lo = digits[0] as u64
            | (digits[1] as u64) << 16
            | (digits[2] as u64) << 32
            | (digits[3] as u64) << 48;
        let hi = digits[4] as u64
            | (digits[5] as u64) << 16
            | (digits[6] as u64) << 32
            | (digits[7] as u64) << 48;

        // p = 2^128 - c, so 2^128 ≡ c (mod p).
        // value = lo + hi*2^64 + carry*2^128 ≡ lo + hi*2^64 + carry*c (mod p).
        let c = Fp128::<P>::C_LO;
        match carry.cmp(&0) {
            std::cmp::Ordering::Equal => {
                Fp128::<P>::from_canonical_u128_reduced(lo as u128 | (hi as u128) << 64)
            }
            std::cmp::Ordering::Greater => Fp128::<P>::solinas_reduce(&[lo, hi, carry as u64]),
            std::cmp::Ordering::Less => {
                // carry < 0: value = base - |carry|*c.
                let neg_carry = (-carry) as u64;
                let sub = neg_carry as u128 * c as u128;
                let base = lo as u128 | (hi as u128) << 64;
                if base >= sub {
                    Fp128::<P>::from_canonical_u128_reduced(base - sub)
                } else {
                    let diff = sub - base;
                    Fp128::<P>::from_canonical_u128_reduced(P - diff)
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl Add for Fp128x8i32 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        unsafe {
            use std::arch::aarch64::*;
            let a0 = vld1q_s32(self.0.as_ptr());
            let a1 = vld1q_s32(self.0.as_ptr().add(4));
            let b0 = vld1q_s32(rhs.0.as_ptr());
            let b1 = vld1q_s32(rhs.0.as_ptr().add(4));
            let mut out = [0i32; 8];
            vst1q_s32(out.as_mut_ptr(), vaddq_s32(a0, b0));
            vst1q_s32(out.as_mut_ptr().add(4), vaddq_s32(a1, b1));
            Self(out)
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl AddAssign for Fp128x8i32 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

#[cfg(target_arch = "aarch64")]
impl Sub for Fp128x8i32 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        unsafe {
            use std::arch::aarch64::*;
            let a0 = vld1q_s32(self.0.as_ptr());
            let a1 = vld1q_s32(self.0.as_ptr().add(4));
            let b0 = vld1q_s32(rhs.0.as_ptr());
            let b1 = vld1q_s32(rhs.0.as_ptr().add(4));
            let mut out = [0i32; 8];
            vst1q_s32(out.as_mut_ptr(), vsubq_s32(a0, b0));
            vst1q_s32(out.as_mut_ptr().add(4), vsubq_s32(a1, b1));
            Self(out)
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl SubAssign for Fp128x8i32 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

#[cfg(target_arch = "aarch64")]
impl Neg for Fp128x8i32 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        unsafe {
            use std::arch::aarch64::*;
            let a0 = vld1q_s32(self.0.as_ptr());
            let a1 = vld1q_s32(self.0.as_ptr().add(4));
            let mut out = [0i32; 8];
            vst1q_s32(out.as_mut_ptr(), vnegq_s32(a0));
            vst1q_s32(out.as_mut_ptr().add(4), vnegq_s32(a1));
            Self(out)
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl Add for Fp128x8i32 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
            self.0[3] + rhs.0[3],
            self.0[4] + rhs.0[4],
            self.0[5] + rhs.0[5],
            self.0[6] + rhs.0[6],
            self.0[7] + rhs.0[7],
        ])
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl AddAssign for Fp128x8i32 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0[0] += rhs.0[0];
        self.0[1] += rhs.0[1];
        self.0[2] += rhs.0[2];
        self.0[3] += rhs.0[3];
        self.0[4] += rhs.0[4];
        self.0[5] += rhs.0[5];
        self.0[6] += rhs.0[6];
        self.0[7] += rhs.0[7];
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl Sub for Fp128x8i32 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
            self.0[3] - rhs.0[3],
            self.0[4] - rhs.0[4],
            self.0[5] - rhs.0[5],
            self.0[6] - rhs.0[6],
            self.0[7] - rhs.0[7],
        ])
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl SubAssign for Fp128x8i32 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0[0] -= rhs.0[0];
        self.0[1] -= rhs.0[1];
        self.0[2] -= rhs.0[2];
        self.0[3] -= rhs.0[3];
        self.0[4] -= rhs.0[4];
        self.0[5] -= rhs.0[5];
        self.0[6] -= rhs.0[6];
        self.0[7] -= rhs.0[7];
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl Neg for Fp128x8i32 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self([
            -self.0[0], -self.0[1], -self.0[2], -self.0[3], -self.0[4], -self.0[5], -self.0[6],
            -self.0[7],
        ])
    }
}

/// Reduce a wide unreduced accumulator back to a canonical field element.
pub trait ReduceTo<F> {
    /// Carry-propagate and reduce to a canonical field element.
    fn reduce(self) -> F;
}

impl<const P: u32> ReduceTo<Fp32<P>> for Fp32x2i32 {
    #[inline]
    fn reduce(self) -> Fp32<P> {
        Fp32x2i32::reduce::<P>(self)
    }
}

impl<const P: u64> ReduceTo<Fp64<P>> for Fp64x4i32 {
    #[inline]
    fn reduce(self) -> Fp64<P> {
        Fp64x4i32::reduce::<P>(self)
    }
}

impl<const P: u128> ReduceTo<Fp128<P>> for Fp128x8i32 {
    #[inline]
    fn reduce(self) -> Fp128<P> {
        Fp128x8i32::reduce::<P>(self)
    }
}

/// Precomputed fold context for `FpExt4<Fp32<P>>`.
///
/// Stores a 4×4 multiplication matrix derived from the challenge `r`,
/// enabling fold via 4 scalar multiply-accumulates per coefficient
/// instead of the general 22-product ring multiplication.
#[derive(Debug, Clone, Copy)]
pub struct FoldMatrixFp32(pub(crate) [[u32; 4]; 4]);

/// Precomputed fold context for `FpExt2<Fp64<P>, C>`.
///
/// Stores the 2×2 "multiply by the challenge `r`" matrix in the `[1, u]`
/// basis (`u² = NR`) as canonical `u64` limbs. Folding then uses two
/// base-field products per output coordinate with a single delayed
/// reduction, instead of the generic per-element Karatsuba multiply that
/// reduces three times.
#[derive(Debug, Clone, Copy)]
pub struct FoldMatrixFp64(pub(crate) [[u64; 2]; 2]);

/// Per-element fold optimization trait.
///
/// Allows field types to precompute a fold context from challenge `r`
/// (e.g. a multiplication matrix) and apply it per-element. The loop
/// structure and parallelism live in the caller (`fold_evals_in_place`).
pub trait HasOptimizedFold: FieldCore {
    /// Precomputed context for folding by a fixed challenge `r`.
    type FoldCtx: Copy + Send + Sync;

    /// Build the fold context from challenge `r`.
    fn precompute_fold(r: Self) -> Self::FoldCtx;

    /// Fold one element pair: `even + r*(odd - even)`.
    fn fold_one(ctx: &Self::FoldCtx, even: Self, odd: Self) -> Self;
}

/// Multi-level unreduced multiplication hierarchy.
///
/// Provides `field × u64` and `field × field` widening multiplies that return
/// accumulator types supporting carry-free addition. Reduction back to a
/// canonical field element happens once after accumulation.
pub trait HasUnreducedOps: FieldCore {
    /// Accumulator for `self × u64` products (narrower than full product).
    type MulU64Accum: AdditiveGroup;
    /// Accumulator for `self × self` products.
    type ProductAccum: AdditiveGroup;

    /// Whether delayed reduction over `ProductAccum` is exact relative to
    /// per-term `Mul` for the small product batches used by inner products.
    ///
    /// When `true`, `reduce_product_accum(sum_i mul_to_product_accum(a_i, b_i))`
    /// equals `sum_i a_i * b_i` for batch sizes within the accumulator's
    /// non-wrapping headroom. The conservative default is `false`; a field opts
    /// in only once its accumulator is proven exact (see `FpExt4<Fp32>`
    /// and `FpExt2<Fp64>`). Fields that leave it `false` keep the per-term reduce
    /// path, so callers that must stay byte-identical to `Mul` are unaffected.
    const DELAYED_PRODUCT_SUM_IS_EXACT: bool = false;

    /// Widening `self × small` with no reduction.
    fn mul_u64_unreduced(self, small: u64) -> Self::MulU64Accum;
    /// Widening `self × other` with no reduction.
    fn mul_to_product_accum(self, other: Self) -> Self::ProductAccum;

    /// Reduce a narrow-mul accumulator to a canonical field element.
    fn reduce_mul_u64_accum(accum: Self::MulU64Accum) -> Self;
    /// Reduce a full-product accumulator to a canonical field element.
    fn reduce_product_accum(accum: Self::ProductAccum) -> Self;
}

macro_rules! impl_default_optimized_fold {
    ($base:ident<$p:ident: $pty:ty>) => {
        impl<const $p: $pty> HasOptimizedFold for $base<$p> {
            type FoldCtx = Self;
            #[inline]
            fn precompute_fold(r: Self) -> Self {
                r
            }
            #[inline]
            fn fold_one(r: &Self, even: Self, odd: Self) -> Self {
                even + *r * (odd - even)
            }
        }
    };
}

impl_default_optimized_fold!(Fp64<P: u64>);
impl_default_optimized_fold!(Fp32<P: u32>);
impl_default_optimized_fold!(Fp128<P: u128>);

impl<const P: u64> HasUnreducedOps for Fp64<P> {
    type MulU64Accum = Fp64ProductAccum;
    type ProductAccum = Fp64ProductAccum;

    #[inline]
    fn mul_u64_unreduced(self, small: u64) -> Fp64ProductAccum {
        let wide = self.mul_wide_u64(small);
        Fp64ProductAccum([wide & u64::MAX as u128, wide >> 64])
    }

    #[inline]
    fn mul_to_product_accum(self, other: Self) -> Fp64ProductAccum {
        let wide = self.mul_wide(other);
        Fp64ProductAccum([wide & u64::MAX as u128, wide >> 64])
    }

    #[inline]
    fn reduce_mul_u64_accum(accum: Fp64ProductAccum) -> Self {
        accum.reduce::<P>()
    }

    #[inline]
    fn reduce_product_accum(accum: Fp64ProductAccum) -> Self {
        accum.reduce::<P>()
    }
}

impl<const P: u32> HasUnreducedOps for Fp32<P> {
    type MulU64Accum = Fp32ProductAccum;
    type ProductAccum = Fp32ProductAccum;

    #[inline]
    fn mul_u64_unreduced(self, small: u64) -> Fp32ProductAccum {
        let wide = (self.to_limbs() as u128) * (small as u128);
        Fp32ProductAccum([wide & u64::MAX as u128, wide >> 64])
    }

    #[inline]
    fn mul_to_product_accum(self, other: Self) -> Fp32ProductAccum {
        Fp32ProductAccum([self.mul_wide(other) as u128, 0])
    }

    #[inline]
    fn reduce_mul_u64_accum(accum: Fp32ProductAccum) -> Self {
        accum.reduce::<P>()
    }

    #[inline]
    fn reduce_product_accum(accum: Fp32ProductAccum) -> Self {
        accum.reduce::<P>()
    }
}

impl<const P: u128> HasUnreducedOps for Fp128<P> {
    type MulU64Accum = Fp128MulU64Accum;
    type ProductAccum = Fp128ProductAccum;

    #[inline]
    fn mul_u64_unreduced(self, small: u64) -> Fp128MulU64Accum {
        let [lo, mid, hi] = self.mul_wide_u64(small);
        Fp128MulU64Accum([lo as u128, mid as u128, hi as u128])
    }

    #[inline]
    fn mul_to_product_accum(self, other: Self) -> Fp128ProductAccum {
        let [r0, r1, r2, r3] = self.mul_wide(other);
        Fp128ProductAccum([r0 as u128, r1 as u128, r2 as u128, r3 as u128])
    }

    #[inline]
    fn reduce_mul_u64_accum(accum: Fp128MulU64Accum) -> Self {
        accum.reduce::<P>()
    }

    #[inline]
    fn reduce_product_accum(accum: Fp128ProductAccum) -> Self {
        accum.reduce::<P>()
    }
}

/// Element-wise scaling of a wide accumulator by a small signed integer.
pub trait ScaleI32 {
    /// Scale each element by `small`.
    fn scale_i32(self, small: i32) -> Self;
}

impl ScaleI32 for Fp32x2i32 {
    #[inline]
    fn scale_i32(self, small: i32) -> Self {
        self.scale_i32(small)
    }
}

impl ScaleI32 for Fp64x4i32 {
    #[inline]
    fn scale_i32(self, small: i32) -> Self {
        self.scale_i32(small)
    }
}

impl ScaleI32 for Fp128x8i32 {
    #[inline]
    fn scale_i32(self, small: i32) -> Self {
        self.scale_i32(small)
    }
}

/// Associates a field type with its wide unreduced accumulator.
pub trait HasWide: FieldCore {
    /// The wide accumulator type.
    type Wide: AdditiveGroup + From<Self> + ReduceTo<Self> + ScaleI32;

    /// Convert `self` to wide form and scale every limb by `small`.
    ///
    /// Equivalent to `Self::Wide::from(self).scale_i32(small)` but avoids
    /// the trait-method ambiguity at call sites.
    #[inline]
    fn mul_small_to_wide(self, small: i32) -> Self::Wide {
        Self::Wide::from(self).scale_i32(small)
    }
}

impl<const P: u32> HasWide for Fp32<P> {
    type Wide = Fp32x2i32;
}

impl<const P: u64> HasWide for Fp64<P> {
    type Wide = Fp64x4i32;
}

impl<const P: u128> HasWide for Fp128<P> {
    type Wide = Fp128x8i32;
}

#[cfg(test)]
mod tests;
