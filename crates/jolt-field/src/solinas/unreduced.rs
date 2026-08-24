//! Deferred-reduction backend: `i32`-lane wide accumulators, `u128`-slot
//! product accumulators, challenge-fold matrices, and the [`Unreduced`],
//! [`Fold`], and [`MulBaseUnreduced`] impls for every Solinas field and
//! extension.
//!
//! # Accumulator semantics and headroom
//!
//! Product accumulators are `[u128; N]` with **wrapping** per-slot ops,
//! i.e. the group `(Z/2^128)^N`. Reduction reads each slot as a plain
//! integer, so a sum reduces exactly iff the *final* integer value of every
//! slot lies in `[0, 2^128)` — intermediate dips below zero cancel exactly
//! under wrapping, and no runtime check enforces the bound. The per-type
//! headroom (worst-case per-term slot contribution, hence how many fmadds
//! fit) is derived in each accumulation formula's comment.
//!
//! Wide accumulators are `[i32; N]` (16 data bits per lane) with
//! **non-wrapping** ops: lane overflow panics in debug builds, which is the
//! only runtime enforcement of the lane headroom. Splitting a canonical
//! element gives lanes in `[0, 2^16)`, so at least
//! `⌊(2^31 − 1) / (2^16 − 1)⌋ = 32768` same-sign accumulations (or
//! `k` accumulations scaled by `s` with `k·|s|·(2^16 − 1) < 2^31`) fit
//! before any lane can overflow.
//!
//! The baseline's NEON intrinsic Add/Sub/Neg lane paths are dropped: LLVM
//! auto-vectorizes the element-wise `[i32; N]` code to the identical
//! `add.4s`/`sub.4s`/`neg.4s` (and `mul.4s` for scaling) instructions at
//! opt-level 3 (see specs/jolt-field-rebuild.md dropped-specialization evidence).

use super::{Fp128, Fp32, Fp64, FpExt2, FpExt4, FpExt8};
use crate::{
    CanonicalEncoding, Ext2Config, ExtField, Fold, MulBaseUnreduced, PseudoMersenne, Ring,
    Unreduced, WithCommitAccumulator,
};

/// Splits a canonical value into 16-bit digits stored one per `i32` lane.
#[inline(always)]
fn split16<const N: usize>(v: u128) -> [i32; N] {
    std::array::from_fn(|i| ((v >> (16 * i)) & 0xFFFF) as i32)
}

/// `Σᵢ laneᵢ · 2^{16i}` as a signed integer. Max magnitude for ≤ 4 lanes:
/// `4 · 2^31 · 2^48 = 2^81`, far inside `i128` (the 8-lane type does not
/// use this — its top lane alone would need 2^{112+31} bits).
#[inline(always)]
fn recombine16(lanes: &[i32]) -> i128 {
    lanes
        .iter()
        .enumerate()
        .map(|(i, &lane)| (lane as i128) << (16 * i))
        .sum()
}

macro_rules! wide_lanes {
    ($($(#[$doc:meta])* $name:ident: $n:literal;)*) => {$(
        $(#[$doc])*
        #[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        #[repr(C)]
        pub struct $name(pub [i32; $n]);

        $crate::impl_group_ops!(impl[] $name {
            add(a, b): $name(std::array::from_fn(|i| a.0[i] + b.0[i])),
            sub(a, b): $name(std::array::from_fn(|i| a.0[i] - b.0[i])),
            neg(a): $name(std::array::from_fn(|i| -a.0[i])),
            zero: $name([0; $n]),
        });

        impl $name {
            /// Multiplies every lane by a small signed scalar.
            ///
            /// Safe when `|small| · max_lane_magnitude < 2^31`; for lanes
            /// fresh from a canonical split (`< 2^16`) any `|small| ≤ 32768`
            /// fits a single product.
            #[inline]
            pub fn scale_i32(self, small: i32) -> Self {
                Self(self.0.map(|lane| lane * small))
            }
        }
    )*};
}

wide_lanes! {
    /// Wide unreduced accumulator for [`Fp32`]: 2 × `i32` lanes.
    Fp32x2i32: 2;
    /// Wide unreduced accumulator for [`Fp64`]: 4 × `i32` lanes.
    Fp64x4i32: 4;
    /// Wide unreduced accumulator for [`Fp128`]: 8 × `i32` lanes (one
    /// 256-bit vector register on AVX2, two 128-bit on NEON).
    Fp128x8i32: 8;
}

const MAX_WIDE_LANE_ACCUMULATIONS: usize = (i32::MAX as usize) / (u16::MAX as usize);

impl<const P: u32> WithCommitAccumulator for Fp32<P> {
    const MAX_COMMIT_ACCUMULATIONS: usize = MAX_WIDE_LANE_ACCUMULATIONS;
}

impl<const P: u64> WithCommitAccumulator for Fp64<P> {
    const MAX_COMMIT_ACCUMULATIONS: usize = MAX_WIDE_LANE_ACCUMULATIONS;
}

impl<const P: u128> WithCommitAccumulator for Fp128<P> {
    const MAX_COMMIT_ACCUMULATIONS: usize = MAX_WIDE_LANE_ACCUMULATIONS;
}

macro_rules! product_accum {
    ($($(#[$doc:meta])* $name:ident: $n:literal;)*) => {$(
        $(#[$doc])*
        #[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $name(pub [u128; $n]);

        $crate::impl_group_ops!(impl[] $name {
            add(a, b): $name(std::array::from_fn(|i| a.0[i].wrapping_add(b.0[i]))),
            sub(a, b): $name(std::array::from_fn(|i| a.0[i].wrapping_sub(b.0[i]))),
            neg(a): $name(std::array::from_fn(|i| a.0[i].wrapping_neg())),
            zero: $name([0; $n]),
        });
    )*};
}

product_accum! {
    /// Accumulator for `Fp32 × Fp32` and `Fp32 × u64` products.
    ///
    /// Slot semantics: `value = s0 + s1·2^64`. Per term: an `Fp32 × Fp32`
    /// product (`< 2^64`) lands whole in `s0`; an `Fp32 × u64` product
    /// (`< 2^96`) is split at bit 64 (`s0 += lo64 < 2^64`,
    /// `s1 += hi < 2^32`). Headroom: `2^128 / 2^64 = 2^64` terms.
    Fp32ProductAccum: 2;
    /// Accumulator for `Fp64 × Fp64` and `Fp64 × u64` products.
    ///
    /// Slot semantics: `value = s0 + s1·2^64`; each `< 2^128` product is
    /// split at bit 64, so both slots grow by `< 2^64` per term. Headroom:
    /// `2^64` terms.
    Fp64ProductAccum: 2;
    /// Accumulator for `Fp128 × u64` products (3 result limbs of
    /// `mul_wide_u64`, one per slot). Each slot grows by `< 2^64` per term;
    /// headroom `2^64 − 1` terms (the reduction's carry chain needs
    /// `sᵢ + carry < 2^128`, see `Fp128::reduce_small_product`).
    Fp128MulU64Accum: 3;
    /// Accumulator for `Fp128 × Fp128` products (4 result limbs of
    /// `mul_wide`, one per slot). Headroom `2^64 − 1` terms, as for
    /// [`Fp128MulU64Accum`].
    Fp128ProductAccum: 4;
    /// Accumulator for `FpExt4<Fp32>` products with delayed reduction: one
    /// slot per ring-subfield coefficient. The φ(X) ring reduction is fused
    /// into the accumulation formulas (`fp_ext4_mul_to_accum_fp32`); only
    /// the per-coefficient Solinas reduction is deferred.
    ///
    /// Headroom: each term contributes at most `7·P² < 7·2^64 < 2^67` per
    /// slot (slot 0: `p00 + 2(p11 + p22 + p33) ≤ 7(P−1)²`; the biased slots
    /// stay ≤ `7P²`), so at least `2^128 / 2^67 = 2^61` terms fit. (The
    /// baseline documented `7·P² ≈ 2^65` and `2^63` accumulations — both
    /// off; the safe bound is `2^61`.)
    FpExt4Fp32ProductAccum: 4;
    /// Accumulator for `FpExt2<Fp64>` products with delayed reduction:
    /// slots `[c0_lo, c0_hi, c1_lo, c1_hi]`, each coefficient a
    /// base-2^64 limb pair reduced like [`Fp64ProductAccum`].
    ///
    /// Headroom: per term `*_lo` grows by `< 2^64` and `*_hi` by `< 3·2^64`
    /// (the limb split keeps a carry of ≤ 2 in bits ≥ 64, see
    /// `fp_ext2_mul_to_accum_fp64`), so at least `2^128 / 3·2^64 > 2^62`
    /// terms fit.
    FpExt2Fp64ProductAccum: 4;
}

/// Lifts of a canonical element into its accumulator/lane shapes.
macro_rules! impl_from {
    ($(impl[$($g:tt)*] $src:ty => $dst:ty { $x:ident => $body:expr })*) => {$(
        impl<$($g)*> From<$src> for $dst {
            #[inline]
            fn from($x: $src) -> Self {
                $body
            }
        }
    )*};
}

impl_from! {
    impl[const P: u32] Fp32<P> => Fp32x2i32 { x => Self(split16(x.to_limbs() as u128)) }
    impl[const P: u64] Fp64<P> => Fp64x4i32 { x => Self(split16(x.to_limbs() as u128)) }
    impl[const P: u128] Fp128<P> => Fp128x8i32 { x => Self(split16(x.to_canonical_u128())) }
    impl[const P: u32] Fp32<P> => Fp32ProductAccum { x => Self([x.to_limbs() as u128, 0]) }
    impl[const P: u64] Fp64<P> => Fp64ProductAccum { x => Self([x.to_limbs() as u128, 0]) }
    impl[const P: u128] Fp128<P> => Fp128MulU64Accum {
        x => { let [lo, hi] = x.to_limbs(); Self([lo as u128, hi as u128, 0]) }
    }
    impl[const P: u128] Fp128<P> => Fp128ProductAccum {
        x => { let [lo, hi] = x.to_limbs(); Self([lo as u128, hi as u128, 0, 0]) }
    }
}

/// Pair accumulator for quadratic extensions: two base accumulators,
/// component-wise.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccumPair<A>(pub A, pub A);

crate::impl_group_ops!(impl[A: crate::AdditiveGroup + PartialEq] AccumPair<A> {
    add(a, b): AccumPair(a.0 + b.0, a.1 + b.1),
    sub(a, b): AccumPair(a.0 - b.0, a.1 - b.1),
    neg(a): AccumPair(-a.0, -a.1),
    zero: AccumPair(A::zero(), A::zero()),
});

impl<const P: u32> Unreduced for Fp32<P> {
    type Product = Fp32ProductAccum;
    type SmallProduct = Fp32ProductAccum;
    type Wide = Fp32x2i32;

    #[inline]
    fn mul_unreduced(self, other: Self) -> Fp32ProductAccum {
        Fp32ProductAccum([self.mul_wide(other) as u128, 0])
    }

    #[inline]
    fn mul_u64_unreduced(self, small: u64) -> Fp32ProductAccum {
        let wide = (self.to_limbs() as u128) * (small as u128);
        Fp32ProductAccum([wide as u64 as u128, wide >> 64])
    }

    #[inline]
    fn scale_wide(self, small: i32) -> Fp32x2i32 {
        Fp32x2i32::from(self).scale_i32(small)
    }

    /// `s0 + s1·2^64 (mod p)`, exact for any slot values (each slot is
    /// reduced independently, then recombined in the field).
    #[inline]
    fn reduce_product(accum: Fp32ProductAccum) -> Self {
        let [s0, s1] = accum.0;
        let shift64 = Self::from_u128_reduced(1u128 << 64);
        Self::from_u128_reduced(s0) + Self::from_u128_reduced(s1) * shift64
    }

    #[inline]
    fn reduce_small_product(accum: Fp32ProductAccum) -> Self {
        Self::reduce_product(accum)
    }

    #[inline]
    fn reduce_wide(wide: Fp32x2i32) -> Self {
        Self::from_i128(recombine16(&wide.0))
    }
}

impl<const P: u64> Unreduced for Fp64<P> {
    type Product = Fp64ProductAccum;
    type SmallProduct = Fp64ProductAccum;
    type Wide = Fp64x4i32;

    #[inline]
    fn mul_unreduced(self, other: Self) -> Fp64ProductAccum {
        let wide = self.mul_wide(other);
        Fp64ProductAccum([wide as u64 as u128, wide >> 64])
    }

    #[inline]
    fn mul_u64_unreduced(self, small: u64) -> Fp64ProductAccum {
        let wide = self.mul_wide_u64(small);
        Fp64ProductAccum([wide as u64 as u128, wide >> 64])
    }

    #[inline]
    fn scale_wide(self, small: i32) -> Fp64x4i32 {
        Fp64x4i32::from(self).scale_i32(small)
    }

    /// `s0 + s1·2^64 (mod p)`, exact for any slot values.
    #[inline]
    fn reduce_product(accum: Fp64ProductAccum) -> Self {
        let [s0, s1] = accum.0;
        Self::solinas_reduce(s0) + Self::solinas_reduce(s1) * Self::solinas_reduce(1u128 << 64)
    }

    #[inline]
    fn reduce_small_product(accum: Fp64ProductAccum) -> Self {
        Self::reduce_product(accum)
    }

    #[inline]
    fn reduce_wide(wide: Fp64x4i32) -> Self {
        Self::from_i128(recombine16(&wide.0))
    }
}

impl<const P: u128> Unreduced for Fp128<P> {
    type Product = Fp128ProductAccum;
    type SmallProduct = Fp128MulU64Accum;
    type Wide = Fp128x8i32;

    #[inline]
    fn mul_unreduced(self, other: Self) -> Fp128ProductAccum {
        let [r0, r1, r2, r3] = self.mul_wide(other);
        Fp128ProductAccum([r0 as u128, r1 as u128, r2 as u128, r3 as u128])
    }

    #[inline]
    fn mul_u64_unreduced(self, small: u64) -> Fp128MulU64Accum {
        let [lo, mid, hi] = self.mul_wide_u64(small);
        Fp128MulU64Accum([lo as u128, mid as u128, hi as u128])
    }

    #[inline]
    fn scale_wide(self, small: i32) -> Fp128x8i32 {
        Fp128x8i32::from(self).scale_i32(small)
    }

    /// Carry-propagates the slot sums into base-2^64 limbs, then Solinas
    /// reduction. With `k ≤ 2^64 − 1` terms each slot is `≤ k(2^64 − 1)`
    /// and each carry `< k`, so `sᵢ + carry ≤ k·2^64 < 2^128` never
    /// overflows (debug-checked by the non-wrapping `+`).
    #[inline]
    fn reduce_product(accum: Fp128ProductAccum) -> Self {
        let [s0, s1, s2, s3] = accum.0;
        let t1 = s1 + (s0 >> 64);
        let t2 = s2 + (t1 >> 64);
        let t3 = s3 + (t2 >> 64);
        Self::solinas_reduce(&[
            s0 as u64,
            t1 as u64,
            t2 as u64,
            t3 as u64,
            (t3 >> 64) as u64,
        ])
    }

    /// Same carry chain and headroom as
    /// [`reduce_product`](Self::reduce_product), one limb shorter.
    #[inline]
    fn reduce_small_product(accum: Fp128MulU64Accum) -> Self {
        let [s0, s1, s2] = accum.0;
        let t1 = s1 + (s0 >> 64);
        let t2 = s2 + (t1 >> 64);
        Self::solinas_reduce(&[s0 as u64, t1 as u64, t2 as u64, (t2 >> 64) as u64])
    }

    /// Carry-propagates the signed lanes into 16-bit digits plus a signed
    /// top carry, then reduces `digits + carry·2^128 ≡ digits + carry·C`.
    ///
    /// With lanes in `(−2^31, 2^31)` the running carry stays within
    /// `±2^15 − 1` after each step (`|v| < 2^31 + 2^15`), so
    /// `|carry|·C < 2^16 · 2^32 = 2^48 < p` and both sign branches stay
    /// canonical.
    #[inline]
    fn reduce_wide(wide: Fp128x8i32) -> Self {
        let mut carry: i64 = 0;
        let mut digits = [0u64; 8];
        for (digit, &lane) in digits.iter_mut().zip(wide.0.iter()) {
            let v = lane as i64 + carry;
            *digit = (v & 0xFFFF) as u64;
            carry = v >> 16;
        }
        let lo = digits[0] | digits[1] << 16 | digits[2] << 32 | digits[3] << 48;
        let hi = digits[4] | digits[5] << 16 | digits[6] << 32 | digits[7] << 48;
        if carry >= 0 {
            Self::solinas_reduce(&[lo, hi, carry as u64])
        } else {
            let base = lo as u128 | (hi as u128) << 64;
            let sub = (-carry) as u128 * Self::C;
            Self::from_u128_reduced(if base >= sub {
                base - sub
            } else {
                P - (sub - base)
            })
        }
    }
}

/// Widening `FpExt4<Fp32>` multiply into one `u128` slot per coefficient:
/// the deg-4 schedule (`crate::schedules::ext4_mul_coeffs`) over raw
/// products, with subtracted terms biased by `P² ≡ 0 (mod p)` so every
/// per-term slot contribution is non-negative.
///
/// Per-term slot bounds (`pᵢⱼ ≤ (P−1)²`): slot 0 `≤ 7(P−1)²`; slot 1
/// `≤ 6(P−1)²`; slot 2 `≤ 5(P−1)² + P²` (the `−p33` cannot underflow the
/// `+P²` bias); slot 3 `≤ 4(P−1)² + 2P²`. All `≤ 7P² < 2^67`, giving the
/// `2^61`-term headroom documented on [`FpExt4Fp32ProductAccum`].
/// Subtractions are evaluated after every addition (left-to-right), so no
/// intermediate dips below zero either.
#[inline(always)]
pub(super) fn fp_ext4_mul_to_accum_fp32<const P: u32>(
    a: [Fp32<P>; 4],
    b: [Fp32<P>; 4],
) -> FpExt4Fp32ProductAccum {
    #[inline(always)]
    fn product<const P: u32>(a: Fp32<P>, b: Fp32<P>) -> u128 {
        (a.to_limbs() as u128) * (b.to_limbs() as u128)
    }
    let [a0, a1, a2, a3] = a;
    let [b0, b1, b2, b3] = b;
    let p_sq = (P as u128) * (P as u128);
    FpExt4Fp32ProductAccum([
        product(a0, b0) + 2 * (product(a1, b1) + product(a2, b2) + product(a3, b3)),
        product(a0, b1)
            + product(a1, b0)
            + product(a1, b2)
            + product(a2, b1)
            + product(a2, b3)
            + product(a3, b2),
        product(a0, b2)
            + product(a2, b0)
            + product(a1, b1)
            + product(a1, b3)
            + product(a3, b1)
            + p_sq
            - product(a3, b3),
        product(a0, b3) + product(a3, b0) + product(a1, b2) + product(a2, b1) + 2 * p_sq
            - product(a2, b3)
            - product(a3, b2),
    ])
}

/// Widening `FpExt4<Fp32>` square into one `u128` slot per coefficient.
/// This is the ten-product specialization of [`fp_ext4_mul_to_accum_fp32`].
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(super) fn fp_ext4_square_to_accum_fp32<const P: u32>(
    a: [Fp32<P>; 4],
) -> FpExt4Fp32ProductAccum {
    #[inline(always)]
    fn product<const P: u32>(a: Fp32<P>, b: Fp32<P>) -> u128 {
        (a.to_limbs() as u128) * (b.to_limbs() as u128)
    }
    let [a0, a1, a2, a3] = a;
    let p_sq = (P as u128) * (P as u128);
    let a0_sq = product(a0, a0);
    let a1_sq = product(a1, a1);
    let a2_sq = product(a2, a2);
    let a3_sq = product(a3, a3);
    let a0a1 = product(a0, a1);
    let a0a2 = product(a0, a2);
    let a0a3 = product(a0, a3);
    let a1a2 = product(a1, a2);
    let a1a3 = product(a1, a3);
    let a2a3 = product(a2, a3);
    FpExt4Fp32ProductAccum([
        a0_sq + 2 * (a1_sq + a2_sq + a3_sq),
        2 * (a0a1 + a1a2 + a2a3),
        2 * a0a2 + a1_sq + 2 * a1a3 + p_sq - a3_sq,
        2 * (a0a3 + a1a2 + p_sq - a2a3),
    ])
}

impl<const P: u32> Unreduced for FpExt4<Fp32<P>> {
    type Product = FpExt4Fp32ProductAccum;
    type SmallProduct = Self;
    type Wide = Self;

    // `fp_ext4_mul_to_accum_fp32` keeps every per-term slot contribution
    // non-negative and `< 2^67`, so a summed batch within the documented
    // headroom reduces to exactly the per-term product sum.
    const SUM_IS_EXACT: bool = true;

    #[inline]
    fn mul_unreduced(self, other: Self) -> FpExt4Fp32ProductAccum {
        fp_ext4_mul_to_accum_fp32(self.coeffs, other.coeffs)
    }

    #[inline]
    fn mul_u64_unreduced(self, small: u64) -> Self {
        self.mul_base(Fp32::from_u64(small))
    }

    #[inline]
    fn scale_wide(self, small: i32) -> Self {
        self.mul_base(Fp32::from_i64(small as i64))
    }

    #[inline]
    fn reduce_product(accum: FpExt4Fp32ProductAccum) -> Self {
        Self::new(accum.0.map(Fp32::from_u128_reduced))
    }

    #[inline]
    fn reduce_small_product(accum: Self) -> Self {
        accum
    }

    #[inline]
    fn reduce_wide(wide: Self) -> Self {
        wide
    }
}

impl<const P: u32> MulBaseUnreduced<Fp32<P>> for FpExt4<Fp32<P>> {
    /// `E × F` scales each coordinate into its own slot; each product is
    /// `< P² < 2^64`, so batches inherit the accumulator headroom.
    #[inline]
    fn mul_base_unreduced(self, x: Fp32<P>) -> FpExt4Fp32ProductAccum {
        let x = x.to_limbs() as u128;
        FpExt4Fp32ProductAccum(self.coeffs.map(|c| (c.to_limbs() as u128) * x))
    }
}

/// Splits `value = lo128 + hi_carry·2^128` into base-2^64 limbs
/// `[bits 0..64, bits 64..]` for a limb-pair slot.
///
/// The high limb carries `hi_carry` (≤ 2 here) in bits ≥ 64 and the
/// reduction reconstructs `lo + hi·2^64` exactly, so the full (> 128-bit)
/// coefficient survives without the mod-2^128 wrap a single `u128`
/// intermediate would incur — the wrap is **not** congruent mod `p` and
/// was the baseline's historical Fp64 bug pattern.
#[inline(always)]
fn fp64_accum_limbs(lo128: u128, hi_carry: u128) -> [u128; 2] {
    [lo128 as u64 as u128, (lo128 >> 64) | (hi_carry << 64)]
}

/// Widening `FpExt2<Fp64>` multiply with delayed reduction and explicit
/// carry tracking.
///
/// Coefficient bounds (`pᵢⱼ ≤ (P−1)² < 2^128`):
/// - `NR = −1`: `c0 = p00 + P² − p11 ∈ [0, 2P²) ⊂ [0, 2^129)`. The add
///   carry and sub borrow satisfy `carry ≥ borrow` (a borrow with no carry
///   would need `p00 + P² < p11 < P²`, impossible), so
///   `hi_carry = carry − borrow ∈ {0, 1}`.
/// - `NR = 2`: `c0 = p00 + 2·p11 < 3P² < 2^130`, carry ∈ {0, 1, 2}.
/// - `c1 = p01 + p10 < 2P² < 2^129`, carry ∈ {0, 1}.
///
/// The specialized carry analysis applies to these two non-residues. Other
/// configurations reduce each product first and lift its canonical
/// coordinates into the same exact accumulator representation.
#[inline(always)]
fn fp_ext2_mul_to_accum_fp64<const P: u64, C: Ext2Config<Fp64<P>>>(
    a: [Fp64<P>; 2],
    b: [Fp64<P>; 2],
) -> FpExt2Fp64ProductAccum {
    let subtract_p11 = match C::NON_RESIDUE_KIND {
        crate::Ext2NonResidueKind::Generic => {
            return fp_ext2_reduced_product_accum::<P, C>(a, b);
        }
        crate::Ext2NonResidueKind::NegOne => true,
        crate::Ext2NonResidueKind::Two => false,
    };
    let p00 = a[0].mul_wide(b[0]);
    let p11 = a[1].mul_wide(b[1]);
    let p01 = a[0].mul_wide(b[1]);
    let p10 = a[1].mul_wide(b[0]);

    let [c0_lo, c0_hi] = if subtract_p11 {
        // c0 = p00 + P² − p11 (the P² bias keeps it non-negative and is
        // invisible mod p).
        let modulus_sq = (P as u128) * (P as u128);
        let (sum, carry_add) = p00.overflowing_add(modulus_sq);
        let (diff, borrow) = sum.overflowing_sub(p11);
        let hi_carry = (carry_add as u128) - (borrow as u128);
        fp64_accum_limbs(diff, hi_carry)
    } else {
        // c0 = p00 + 2·p11.
        let (sum1, carry1) = p00.overflowing_add(p11);
        let (sum2, carry2) = sum1.overflowing_add(p11);
        fp64_accum_limbs(sum2, (carry1 as u128) + (carry2 as u128))
    };
    let (c1_sum, c1_carry) = p01.overflowing_add(p10);
    let [c1_lo, c1_hi] = fp64_accum_limbs(c1_sum, c1_carry as u128);

    FpExt2Fp64ProductAccum([c0_lo, c0_hi, c1_lo, c1_hi])
}

#[inline(always)]
fn fp_ext2_reduced_product_accum<const P: u64, C: Ext2Config<Fp64<P>>>(
    a: [Fp64<P>; 2],
    b: [Fp64<P>; 2],
) -> FpExt2Fp64ProductAccum {
    // An arbitrary non-residue can make NR*p11 wider than the two-limb
    // coefficient accumulator. Reduce this uncommon configuration first,
    // then lift the canonical coordinates into the exact split-limb sum.
    let product = FpExt2::<Fp64<P>, C>::new(a[0], a[1]) * FpExt2::new(b[0], b[1]);
    FpExt2Fp64ProductAccum([
        product.coeffs[0].0 as u128,
        0,
        product.coeffs[1].0 as u128,
        0,
    ])
}

impl<const P: u64, C: Ext2Config<Fp64<P>>> Unreduced for FpExt2<Fp64<P>, C> {
    type Product = FpExt2Fp64ProductAccum;
    type SmallProduct = AccumPair<Fp64ProductAccum>;
    type Wide = Self;

    // `fp_ext2_mul_to_accum_fp64` keeps the full > 128-bit coefficients via
    // carry-aware base-2^64 limbs, so summing a batch and reducing once
    // equals the per-term `Mul` sum within the documented headroom.
    const SUM_IS_EXACT: bool = true;

    #[inline]
    fn mul_unreduced(self, other: Self) -> FpExt2Fp64ProductAccum {
        fp_ext2_mul_to_accum_fp64::<P, C>(self.coeffs, other.coeffs)
    }

    #[inline]
    fn mul_u64_unreduced(self, small: u64) -> Self::SmallProduct {
        AccumPair(
            self.coeffs[0].mul_u64_unreduced(small),
            self.coeffs[1].mul_u64_unreduced(small),
        )
    }

    #[inline]
    fn scale_wide(self, small: i32) -> Self {
        self.mul_base(Fp64::from_i64(small as i64))
    }

    #[inline]
    fn reduce_product(accum: FpExt2Fp64ProductAccum) -> Self {
        let [c0_lo, c0_hi, c1_lo, c1_hi] = accum.0;
        Self::new(
            Fp64::reduce_product(Fp64ProductAccum([c0_lo, c0_hi])),
            Fp64::reduce_product(Fp64ProductAccum([c1_lo, c1_hi])),
        )
    }

    #[inline]
    fn reduce_small_product(accum: Self::SmallProduct) -> Self {
        Self::new(
            Fp64::reduce_small_product(accum.0),
            Fp64::reduce_small_product(accum.1),
        )
    }

    #[inline]
    fn reduce_wide(wide: Self) -> Self {
        wide
    }
}

impl<const P: u64, C: Ext2Config<Fp64<P>>> MulBaseUnreduced<Fp64<P>> for FpExt2<Fp64<P>, C> {}

/// Identity-shape [`Unreduced`] for extension variants without a dedicated
/// accumulator: every "unreduced" op reduces immediately (`Product = Self`),
/// which is trivially exact per term — `SUM_IS_EXACT` keeps its
/// conservative `false` so callers do not switch to batched reduction.
macro_rules! unreduced_identity {
    (impl[$($g:tt)*] $ty:ty, base: $base:ty) => {
        impl<$($g)*> Unreduced for $ty {
            type Product = Self;
            type SmallProduct = Self;
            type Wide = Self;

            #[inline]
            fn mul_unreduced(self, other: Self) -> Self {
                self * other
            }
            #[inline]
            fn mul_u64_unreduced(self, small: u64) -> Self {
                self.mul_base(<$base>::from_u64(small))
            }
            #[inline]
            fn scale_wide(self, small: i32) -> Self {
                self.mul_base(<$base>::from_i64(small as i64))
            }
            #[inline]
            fn reduce_product(accum: Self) -> Self {
                accum
            }
            #[inline]
            fn reduce_small_product(accum: Self) -> Self {
                accum
            }
            #[inline]
            fn reduce_wide(wide: Self) -> Self {
                wide
            }
        }

        impl<$($g)*> MulBaseUnreduced<$base> for $ty {}
    };
}

unreduced_identity!(impl[const P: u32, C: Ext2Config<Fp32<P>>] FpExt2<Fp32<P>, C>, base: Fp32<P>);
unreduced_identity!(impl[const P: u128, C: Ext2Config<Fp128<P>>] FpExt2<Fp128<P>, C>, base: Fp128<P>);
unreduced_identity!(impl[const P: u64] FpExt4<Fp64<P>>, base: Fp64<P>);
unreduced_identity!(impl[const P: u128] FpExt4<Fp128<P>>, base: Fp128<P>);
unreduced_identity!(impl[F: PseudoMersenne] FpExt8<F>, base: F);

/// Default [`Fold`]: no precomputation, one generic multiply per pair.
macro_rules! fold_default {
    (impl[$($g:tt)*] $ty:ty) => {
        impl<$($g)*> Fold for $ty {
            type Ctx = Self;

            #[inline]
            fn precompute(r: Self) -> Self {
                r
            }
            #[inline]
            fn fold_one(r: &Self, even: Self, odd: Self) -> Self {
                even + *r * (odd - even)
            }
        }
    };
}

// Base fields and the extension variants without a specialized fold
// matrix. (A blanket `impl<F: PseudoMersenne> Fold for F` would be
// rejected by coherence against the `FpExt2<_, C>` impls: `C` is an open
// type parameter, so the compiler cannot rule out a downstream
// `PseudoMersenne` impl for a quadratic extension.)
fold_default!(impl[const P: u32] Fp32<P>);
fold_default!(impl[const P: u64] Fp64<P>);
fold_default!(impl[const P: u128] Fp128<P>);
fold_default!(impl[const P: u32, C: Ext2Config<Fp32<P>>] FpExt2<Fp32<P>, C>);
fold_default!(impl[const P: u128, C: Ext2Config<Fp128<P>>] FpExt2<Fp128<P>, C>);
fold_default!(impl[const P: u64] FpExt4<Fp64<P>>);
fold_default!(impl[const P: u128] FpExt4<Fp128<P>>);
fold_default!(impl[F: PseudoMersenne] FpExt8<F>);

/// Precomputed fold context for `FpExt4<Fp32>`: the 4×4 multiply-by-`r`
/// matrix in the `[1, e1, e2, e3]` basis, as canonical `u32` residues.
#[derive(Debug, Clone, Copy)]
pub struct FoldMatrixFp32(pub(crate) [[u32; 4]; 4]);

impl<const P: u32> Fold for FpExt4<Fp32<P>> {
    type Ctx = FoldMatrixFp32;

    /// Columns of the deg-4 schedule (`ext4_mul_coeffs`) with `a = r`:
    /// signed schedule entries like `r1 − r3` are baked as canonical
    /// residues, so the fold is 16 unsigned multiply-adds.
    #[inline]
    fn precompute(r: Self) -> FoldMatrixFp32 {
        let [r0, r1, r2, r3] = r.coeffs;
        let two = Fp32::<P>::from_u64(2);
        let lim = |x: Fp32<P>| x.to_limbs();
        FoldMatrixFp32([
            [lim(r0), lim(two * r1), lim(two * r2), lim(two * r3)],
            [lim(r1), lim(r0 + r2), lim(r1 + r3), lim(r2)],
            [lim(r2), lim(r1 + r3), lim(r0), lim(r1 - r3)],
            [lim(r3), lim(r2), lim(r1 - r3), lim(r0 - r2)],
        ])
    }

    /// `even + r·(odd − even)` via 4 base multiply-adds per coefficient
    /// instead of the 22-product generic multiply. For `P < 2^31` each
    /// product is `< 2^62` and a row sum of 4 fits `u64`; otherwise
    /// products are `< 2^64` and the row sum (`< 2^66`) uses `u128`.
    #[inline]
    fn fold_one(ctx: &FoldMatrixFp32, even: Self, odd: Self) -> Self {
        let m = &ctx.0;
        let d: [u32; 4] = std::array::from_fn(|j| (odd.coeffs[j] - even.coeffs[j]).to_limbs());
        let folded: [Fp32<P>; 4] = if P < (1u32 << 31) {
            std::array::from_fn(|row| {
                let acc = (0..4)
                    .map(|j| (m[row][j] as u64) * (d[j] as u64))
                    .sum::<u64>();
                Fp32::from_u64(acc) + even.coeffs[row]
            })
        } else {
            std::array::from_fn(|row| {
                let acc = (0..4)
                    .map(|j| (m[row][j] as u128) * (d[j] as u128))
                    .sum::<u128>();
                Fp32::from_u128_reduced(acc) + even.coeffs[row]
            })
        };
        Self::new(folded)
    }
}

/// Reduces the integer sum `w0 + w1` of two products of canonical `Fp64`
/// residues (each `< 2^{2·BITS}`).
///
/// - Sub-word primes (`BITS < 64`): the sum is `< 2^127`, reduce directly.
/// - Full-word primes: the sum reaches `< 2^129`; fold bits 64.. with
///   `2^64 ≡ C` and the overflow bit with `2^128 ≡ C²` (both need
///   `BITS = 64`). The folded value is `< 2^64 + 2^96 + 2^64 < 2^97`.
#[inline(always)]
fn fp64_reduce_sum_of_two_products<const P: u64>(w0: u128, w1: u128) -> Fp64<P> {
    if Fp64::<P>::BITS < 64 {
        Fp64::solinas_reduce(w0 + w1)
    } else {
        let (s, carry) = w0.overflowing_add(w1);
        let c = Fp64::<P>::C as u128;
        Fp64::solinas_reduce(
            (s as u64 as u128) + ((s >> 64) as u64 as u128) * c + (carry as u128) * c * c,
        )
    }
}

/// Precomputed fold context for `FpExt2<Fp64>`: the 2×2 multiply-by-`r`
/// matrix `[[r0, NR·r1], [r1, r0]]` in the `[1, u]` basis (`u² = NR`), as
/// canonical `u64` residues.
#[derive(Debug, Clone, Copy)]
pub struct FoldMatrixFp64(pub(crate) [[u64; 2]; 2]);

impl<const P: u64, C: Ext2Config<Fp64<P>>> Fold for FpExt2<Fp64<P>, C> {
    type Ctx = FoldMatrixFp64;

    #[inline]
    fn precompute(r: Self) -> FoldMatrixFp64 {
        let [r0, r1] = r.coeffs;
        let nr_r1 = C::mul_non_residue(r1, |base| base);
        FoldMatrixFp64([
            [r0.to_limbs(), nr_r1.to_limbs()],
            [r1.to_limbs(), r0.to_limbs()],
        ])
    }

    /// `even + r·(odd − even)`: each output coordinate is two `u64 × u64`
    /// products with one delayed reduction
    /// ([`fp64_reduce_sum_of_two_products`]) — schoolbook with 2 reductions
    /// versus the generic Karatsuba's 3. Canonical, hence byte-identical to
    /// the generic fold.
    #[inline]
    fn fold_one(ctx: &FoldMatrixFp64, even: Self, odd: Self) -> Self {
        let m = &ctx.0;
        let d0 = (odd.coeffs[0] - even.coeffs[0]).to_limbs() as u128;
        let d1 = (odd.coeffs[1] - even.coeffs[1]).to_limbs() as u128;
        let c0 = fp64_reduce_sum_of_two_products((m[0][0] as u128) * d0, (m[0][1] as u128) * d1);
        let c1 = fp64_reduce_sum_of_two_products((m[1][0] as u128) * d0, (m[1][1] as u128) * d1);
        Self::new(even.coeffs[0] + c0, even.coeffs[1] + c1)
    }
}
