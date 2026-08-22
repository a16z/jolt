//! Shared packed Solinas algebra for the word-sized fields, written once
//! against the [`SimdWord`] vocabulary and instantiated per ISA through the
//! marker type parameter `I` — the one source of truth for the packed
//! fold/canonicalize structure across NEON, AVX2, and AVX-512.
//!
//! [`PackedFp32`] is the u32-lane engine (widen to 64-bit products, two or
//! three Solinas folds, fused deferred-reduction dot products for the
//! degree-4 extension kernels); [`PackedFp64`] is the u64-lane engine
//! (128-bit products folded through `2^BITS ≡ C`). The fold constants are
//! taken from the scalar field types, so the `C(C+1) < P` precondition is
//! asserted in exactly one place per width (`word.rs`).

#![cfg(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "x86_64", target_feature = "avx2")
))]

use super::simd::SimdWord;
use crate::solinas::{Fp32, Fp64};
use crate::{Ext2NonResidueKind, Packed};

pub(crate) use super::fp128::PackedFp128;

/// Stamps `Clone`/`Copy` and the operator matrix from `add_raw`/`sub_raw`/
/// `mul_raw` inherent methods. Shared by all three packed engines.
macro_rules! impl_packed_arith {
    (impl[$($g:tt)*] $ty:ty) => {
        impl<$($g)*> Clone for $ty {
            #[inline(always)]
            fn clone(&self) -> Self {
                *self
            }
        }
        impl<$($g)*> Copy for $ty {}
        impl<$($g)*> ::core::ops::Add for $ty {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                Self::add_raw(self, rhs)
            }
        }
        impl<$($g)*> ::core::ops::Sub for $ty {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                Self::sub_raw(self, rhs)
            }
        }
        impl<$($g)*> ::core::ops::Mul for $ty {
            type Output = Self;
            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                Self::mul_raw(self, rhs)
            }
        }
        impl<$($g)*> ::core::ops::AddAssign for $ty {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }
        impl<$($g)*> ::core::ops::SubAssign for $ty {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }
        impl<$($g)*> ::core::ops::MulAssign for $ty {
            #[inline(always)]
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }
    };
}
pub(crate) use impl_packed_arith;

/// `c·v` on 64-bit lanes for a compile-time-constant offset `c < 2^32`:
/// shift/add when `c = 2^a ± 1`, otherwise the ISA's small multiply.
/// Callers guarantee the exact product fits 64 bits.
#[inline(always)]
fn mul_by_offset<I: SimdWord>(v: I::V64, c: u64) -> I::V64 {
    if c == 1 {
        v
    } else if (c - 1).is_power_of_two() {
        I::add64(I::shl64(v, (c - 1).trailing_zeros()), v)
    } else if (c + 1).is_power_of_two() {
        I::sub64(I::shl64(v, (c + 1).trailing_zeros()), v)
    } else {
        I::mul_small(v, c)
    }
}

/// Packed `Fp32` lanes over ISA `I` (`I::W32` lanes).
#[repr(transparent)]
pub struct PackedFp32<const P: u32, I: SimdWord>(I::V32);

impl<const P: u32, I: SimdWord> PackedFp32<P, I> {
    const BITS: u32 = Fp32::<P>::BITS;
    const C: u32 = Fp32::<P>::C;
    const MASK: u64 = if Self::BITS == 32 {
        u32::MAX as u64
    } else {
        (1u64 << Self::BITS) - 1
    };
    /// Whether two Solinas folds bring a sum of up to four `(P−1)²`
    /// products into `[0, 2P)` for the final canonicalize step.
    const TWO_FOLD_OK: bool = {
        let c = Self::C as u64;
        4 * c * c + 3 * c <= (1u64 << Self::BITS)
    };

    #[inline(always)]
    fn add_raw(a: Self, b: Self) -> Self {
        let t = I::add32(a.0, b.0);
        let t = if Self::BITS == 32 {
            // The carry out of u32 is 2^32 ≡ C; fold it before canonicalizing.
            I::select32(I::lt_u32(t, a.0), I::add32(t, I::splat32(Self::C)), t)
        } else {
            t
        };
        Self(I::min_u32(t, I::sub32(t, I::splat32(P))))
    }

    #[inline(always)]
    fn sub_raw(a: Self, b: Self) -> Self {
        let t = I::sub32(a.0, b.0);
        if Self::BITS == 32 {
            // A wrap adds 2^32 ≡ C, so subtract C where a < b.
            Self(I::select32(
                I::lt_u32(a.0, b.0),
                I::sub32(t, I::splat32(Self::C)),
                t,
            ))
        } else {
            Self(I::min_u32(t, I::add32(t, I::splat32(P))))
        }
    }

    #[inline(always)]
    fn mul_raw(a: Self, b: Self) -> Self {
        if Self::BITS == 31 {
            // ISAs with a 32-bit high-multiply reduce 31-bit primes without
            // ever widening to 64-bit lanes.
            if let Some(r) = I::mul_pm31(a.0, b.0, P, Self::C) {
                return Self(r);
            }
        }
        Self(Self::reduce(I::widen_mul(a.0, b.0)))
    }

    /// One Solinas fold: `(v & MASK) + C·(v >> BITS)`.
    #[inline(always)]
    fn fold(v: I::V64) -> I::V64 {
        I::add64(
            I::and64(v, I::splat64(Self::MASK)),
            mul_by_offset::<I>(I::shr64(v, Self::BITS), Self::C as u64),
        )
    }

    /// Two/three-fold reduction of widened products (or sums of up to four
    /// products, pre-folded when `BITS == 32`) to canonical 32-bit lanes.
    #[inline(always)]
    fn reduce(x: [I::V64; 2]) -> I::V32 {
        let f = x.map(|v| Self::fold(Self::fold(v)));
        let f = if Self::TWO_FOLD_OK {
            f
        } else {
            f.map(Self::fold)
        };
        if Self::BITS == 32 {
            // The two-fold residue can exceed 2^32 (up to 2^32 + C²), so
            // canonicalize on the 64-bit lanes before packing.
            let p = I::splat64(u64::from(P));
            I::narrow_pack(f.map(|v| I::select64(I::lt_u64(v, p), v, I::sub64(v, p))))
        } else {
            let packed = I::narrow_pack(f);
            I::min_u32(packed, I::sub32(packed, I::splat32(P)))
        }
    }

    /// `Σ a_i·b_i` with one deferred end-reduction (`N ≤ 4`). For
    /// `BITS ≤ 31` the raw 64-bit products sum without overflow; for
    /// `BITS == 32` each product is pre-folded once (`< (C+1)·2^32`) so four
    /// terms still sum carry-free.
    #[inline(always)]
    fn dot<const N: usize>(a: [Self; N], b: [Self; N]) -> Self {
        let term = |i: usize| {
            let p = I::widen_mul(a[i].0, b[i].0);
            if Self::BITS == 32 {
                p.map(Self::fold)
            } else {
                p
            }
        };
        let mut sums = term(0);
        for i in 1..N {
            let t = term(i);
            sums = [I::add64(sums[0], t[0]), I::add64(sums[1], t[1])];
        }
        Self(Self::reduce(sums))
    }
}

impl_packed_arith!(impl[const P: u32, I: SimdWord] PackedFp32<P, I>);

impl<const P: u32, I: SimdWord> Packed for PackedFp32<P, I> {
    type Scalar = Fp32<P>;
    const WIDTH: usize = I::W32;

    #[inline]
    fn from_fn(mut f: impl FnMut(usize) -> Fp32<P>) -> Self {
        Self(I::v32_from_fn(|i| f(i).0))
    }

    #[inline]
    fn extract(&self, lane: usize) -> Fp32<P> {
        debug_assert!(lane < I::W32);
        Fp32(I::v32_lane(self.0, lane))
    }

    #[inline]
    fn broadcast(value: Fp32<P>) -> Self {
        Self(I::splat32(value.0))
    }

    /// Fused kernel: each output coefficient is one deferred-reduction dot
    /// product instead of six independently reduced multiplies.
    #[inline(always)]
    fn ext4_mul(a: [Self; 4], b: [Self; 4]) -> [Self; 4] {
        let [b0, b1, b2, b3] = b;
        [
            Self::dot(a, [b0, b1 + b1, b2 + b2, b3 + b3]),
            Self::dot(a, [b1, b0 + b2, b1 + b3, b2]),
            Self::dot(a, [b2, b1 + b3, b0, b1 - b3]),
            Self::dot(a, [b3, b2, b1 - b3, b0 - b2]),
        ]
    }

    /// Fused kernel: squaring via three- and four-term dot products.
    #[inline(always)]
    fn ext4_square(a: [Self; 4]) -> [Self; 4] {
        let [a0, a1, a2, a3] = a;
        let zero = Self::broadcast(Fp32(0));
        [
            Self::dot(a, [a0, a1 + a1, a2 + a2, a3 + a3]),
            Self::dot([a0, a1, a2], [a1 + a1, a2 + a2, a3 + a3]),
            Self::dot([a0, a1, a1, a3], [a2 + a2, a1, a3 + a3, zero - a3]),
            Self::dot([a0, a1, a2], [a3 + a3, a2 + a2, zero - (a3 + a3)]),
        ]
    }
}

/// Packed `Fp64` lanes over ISA `I` (`I::W64` lanes).
#[repr(transparent)]
pub struct PackedFp64<const P: u64, I: SimdWord>(I::V64);

impl<const P: u64, I: SimdWord> PackedFp64<P, I> {
    const BITS: u32 = Fp64::<P>::BITS;
    // The scalar invariant C(C+1) < P < 2^64 implies C < 2^32, which the
    // `mul_small`-based reduction below relies on.
    const C: u64 = Fp64::<P>::C;
    const MASK: u64 = if Self::BITS == 64 {
        u64::MAX
    } else {
        (1u64 << Self::BITS) - 1
    };
    /// Whether two folds and one subtraction reduce a sum of three products.
    const EXT2_TWO_FUSION_SAFE: bool =
        Self::BITS < 64 && 3 * (Self::C as u128) * (Self::C as u128 + 1) < P as u128;

    /// Reduces a scalar sum of up to three products for lane-wise backends.
    #[inline(always)]
    fn reduce_three_product_sum(lo: u64, hi: u64) -> u64 {
        debug_assert!(Self::EXT2_TWO_FUSION_SAFE);
        Fp64::<P>::reduce_sub_word_wide(lo, hi, hi >> Self::BITS)
    }

    /// Adds lane-wise 128-bit values represented as `[lo, hi]`.
    #[inline(always)]
    fn add128(a: [I::V64; 2], b: [I::V64; 2]) -> [I::V64; 2] {
        let lo = I::add64(a[0], b[0]);
        let carry = I::select64(I::lt_u64(lo, a[0]), I::splat64(1), I::splat64(0));
        [lo, I::add64(I::add64(a[1], b[1]), carry)]
    }

    #[inline(always)]
    fn add_raw(a: Self, b: Self) -> Self {
        let p = I::splat64(P);
        let s = I::add64(a.0, b.0);
        if Self::BITS < 64 {
            // a + b < 2P < 2^64: no wrap possible.
            Self(I::select64(I::lt_u64(s, p), s, I::sub64(s, p)))
        } else {
            let no_wrap = I::select64(I::lt_u64(s, p), s, I::sub64(s, p));
            // On wrap the true sum is s + 2^64 ≡ s + C, already canonical
            // (s < 2P − 2^64 = P − C).
            Self(I::select64(
                I::lt_u64(s, a.0),
                I::add64(s, I::splat64(Self::C)),
                no_wrap,
            ))
        }
    }

    #[inline(always)]
    fn sub_raw(a: Self, b: Self) -> Self {
        let d = I::sub64(a.0, b.0);
        // On wrap, +P ≡ −C (mod 2^64) restores the canonical value for any
        // BITS, including 64.
        Self(I::select64(
            I::lt_u64(a.0, b.0),
            I::add64(d, I::splat64(P)),
            d,
        ))
    }

    #[inline(always)]
    fn mul_raw(a: Self, b: Self) -> Self {
        if I::FP64_MUL_BY_LANES {
            Self(I::v64_from_fn(|i| {
                let x = I::v64_lane(a.0, i);
                let y = I::v64_lane(b.0, i);
                if Self::BITS == 63 {
                    if let Some(reduced) = I::mul_pm63(x, y, P, Self::C) {
                        return reduced;
                    }
                }
                (Fp64::<P>(x) * Fp64::<P>(y)).0
            }))
        } else {
            let [lo, hi] = I::mul64_wide(a.0, b.0);
            Self(Self::reduce128(lo, hi))
        }
    }

    /// Solinas reduction of per-lane 128-bit products `hi·2^64 + lo`.
    #[inline(always)]
    fn reduce128(lo: I::V64, hi: I::V64) -> I::V64 {
        let p = I::splat64(P);
        if Self::BITS < 64 && Fp64::<P>::FOLD_IN_U64 {
            let mask = I::splat64(Self::MASK);
            let hi_k = I::add64(I::shr64(lo, Self::BITS), I::shl64(hi, 64 - Self::BITS));
            let f1 = I::add64(I::and64(lo, mask), mul_by_offset::<I>(hi_k, Self::C));
            let f2 = I::add64(
                I::and64(f1, mask),
                mul_by_offset::<I>(I::shr64(f1, Self::BITS), Self::C),
            );
            I::select64(I::lt_u64(f2, p), f2, I::sub64(f2, p))
        } else if Self::BITS < 64 {
            Self::reduce128_sub_word_wide(lo, hi)
        } else {
            // BITS == 64: hi·2^64 ≡ C·hi, a 96-bit product. Fold its carry
            // limb once more; the final +C correction cannot cascade
            // (r < C·2^32 after a wrap) and one subtract canonicalizes
            // (everything is < 2^64 = P + C < 2P).
            let [cl, ch] = I::mul_small_wide(hi, Self::C);
            let s = I::add64(lo, cl);
            let k1 = I::select64(I::lt_u64(s, lo), I::splat64(1), I::splat64(0));
            let m = I::mul_small(I::add64(ch, k1), Self::C);
            let r = I::add64(s, m);
            let r = I::select64(I::lt_u64(r, s), I::add64(r, I::splat64(Self::C)), r);
            I::select64(I::lt_u64(r, p), r, I::sub64(r, p))
        }
    }

    /// Two-fold sub-word reduction that retains the carry out of the first
    /// `C * (product >> BITS)` fold. It also accepts sums of up to three
    /// products when [`Self::EXT2_TWO_FUSION_SAFE`] holds.
    #[inline(always)]
    fn reduce128_sub_word_wide(lo: I::V64, hi: I::V64) -> I::V64 {
        let mask = I::splat64(Self::MASK);
        let high = I::or64(I::shr64(lo, Self::BITS), I::shl64(hi, 64 - Self::BITS));
        let high_overflow = I::shr64(hi, Self::BITS);
        let [c_high_lo, c_high_hi] = I::mul_small_wide(high, Self::C);
        let c_high_hi = I::add64(c_high_hi, I::mul_small(high_overflow, Self::C));

        let low_bits = I::and64(lo, mask);
        let fold1_lo = I::add64(low_bits, c_high_lo);
        let carry = I::select64(I::lt_u64(fold1_lo, low_bits), I::splat64(1), I::splat64(0));
        let fold1_hi = I::add64(c_high_hi, carry);
        let fold1_high = I::or64(
            I::shr64(fold1_lo, Self::BITS),
            I::shl64(fold1_hi, 64 - Self::BITS),
        );
        let fold2 = I::add64(I::and64(fold1_lo, mask), I::mul_small(fold1_high, Self::C));
        let p = I::splat64(P);
        I::select64(I::lt_u64(fold2, p), fold2, I::sub64(fold2, p))
    }
}

impl_packed_arith!(impl[const P: u64, I: SimdWord] PackedFp64<P, I>);

impl<const P: u64, I: SimdWord> Packed for PackedFp64<P, I> {
    type Scalar = Fp64<P>;
    const WIDTH: usize = I::W64;

    #[inline]
    fn from_fn(mut f: impl FnMut(usize) -> Fp64<P>) -> Self {
        Self(I::v64_from_fn(|i| f(i).0))
    }

    #[inline]
    fn extract(&self, lane: usize) -> Fp64<P> {
        debug_assert!(lane < I::W64);
        Fp64(I::v64_lane(self.0, lane))
    }

    #[inline]
    fn broadcast(value: Fp64<P>) -> Self {
        Self(I::splat64(value.0))
    }

    #[inline(always)]
    fn ext2_mul<C: crate::Ext2Config<Self::Scalar>>(
        a0: Self,
        a1: Self,
        b0: Self,
        b1: Self,
    ) -> (Self, Self) {
        if C::NON_RESIDUE_KIND == Ext2NonResidueKind::Two && Self::EXT2_TWO_FUSION_SAFE {
            if I::FP64_MUL_BY_LANES {
                let c0 = I::v64_from_fn(|lane| {
                    let a0 = I::v64_lane(a0.0, lane) as u128;
                    let a1 = I::v64_lane(a1.0, lane) as u128;
                    let b0 = I::v64_lane(b0.0, lane) as u128;
                    let b1 = I::v64_lane(b1.0, lane) as u128;
                    let z = a0 * b0 + 2 * a1 * b1;
                    Self::reduce_three_product_sum(z as u64, (z >> 64) as u64)
                });
                let c1 = I::v64_from_fn(|lane| {
                    let a0 = I::v64_lane(a0.0, lane) as u128;
                    let a1 = I::v64_lane(a1.0, lane) as u128;
                    let b0 = I::v64_lane(b0.0, lane) as u128;
                    let b1 = I::v64_lane(b1.0, lane) as u128;
                    let z = a0 * b1 + a1 * b0;
                    Self::reduce_three_product_sum(z as u64, (z >> 64) as u64)
                });
                return (Self(c0), Self(c1));
            }

            let p00 = I::mul64_wide(a0.0, b0.0);
            let p11 = I::mul64_wide(a1.0, b1.0);
            let p01 = I::mul64_wide(a0.0, b1.0);
            let p10 = I::mul64_wide(a1.0, b0.0);
            let z0 = Self::add128(Self::add128(p00, p11), p11);
            let z1 = Self::add128(p01, p10);
            return (
                Self(Self::reduce128_sub_word_wide(z0[0], z0[1])),
                Self(Self::reduce128_sub_word_wide(z1[0], z1[1])),
            );
        }

        let v0 = a0 * b0;
        let v1 = a1 * b1;
        let cross = (a0 + a1) * (b0 + b1);
        (
            v0 + C::mul_non_residue(v1, Self::broadcast),
            cross - v0 - v1,
        )
    }
}
