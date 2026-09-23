//! Packed two-limb field: `I::W64` [`Fp128`] lanes in SoA layout
//! (`lo`/`hi` limb vectors), shared across ISAs like the word engines.
//!
//! Add/sub vectorize the 128-bit carry chains with fused reduction;
//! multiplication goes lane-by-lane through the scalar kernel (which is the
//! AArch64 inline-asm multiply on that target) — no ISA in the baseline had
//! a vectorized 128-bit multiply either.

#![cfg(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "x86_64", target_feature = "avx2")
))]

use super::engine::impl_packed_arith;
use super::simd::SimdWord;
use crate::solinas::Fp128;
use crate::Packed;

/// Packed `Fp128` lanes over ISA `I`: `lo[i]`/`hi[i]` are lane `i`'s limbs.
pub struct PackedFp128<const P: u128, I: SimdWord> {
    lo: I::V64,
    hi: I::V64,
}

impl<const P: u128, I: SimdWord> PackedFp128<P, I> {
    const C_LO: u64 = Fp128::<P>::C_LO;
    const P_LO: u64 = P as u64;
    const P_HI: u64 = (P >> 64) as u64;

    /// Carry-chain add with fused reduction (see the scalar
    /// [`Fp128`] `add_raw` for the case analysis): compute `s = a + b`
    /// tracking the 128-bit wrap, then `t = s + C ≡ s − p (mod 2^128)` and
    /// select `t` where the sum wrapped or `t` carried (`s ≥ p`).
    #[inline(always)]
    fn add_raw(a: Self, b: Self) -> Self {
        let s_lo = I::add64(a.lo, b.lo);
        let carry_lo = I::lt_u64(s_lo, a.lo);
        let h1 = I::add64(a.hi, b.hi);
        let wrap1 = I::lt_u64(h1, a.hi);
        // Subtracting an all-ones carry mask adds one.
        let s_hi = I::sub64(h1, carry_lo);
        let wrap2 = I::lt_u64(s_hi, h1);
        let wrapped = I::or64(wrap1, wrap2);

        let t_lo = I::add64(s_lo, I::splat64(Self::C_LO));
        let carry_c = I::lt_u64(t_lo, s_lo);
        let t_hi = I::sub64(s_hi, carry_c);
        let carried = I::lt_u64(t_hi, s_hi);

        let use_t = I::or64(wrapped, carried);
        Self {
            lo: I::select64(use_t, t_lo, s_lo),
            hi: I::select64(use_t, t_hi, s_hi),
        }
    }

    /// Subtract with borrow-conditional modulus add-back.
    #[inline(always)]
    fn sub_raw(a: Self, b: Self) -> Self {
        let one = I::splat64(1);
        let d_lo = I::sub64(a.lo, b.lo);
        let borrow_lo = I::and64(I::lt_u64(a.lo, b.lo), one);
        let h1 = I::sub64(a.hi, b.hi);
        let bw1 = I::lt_u64(a.hi, b.hi);
        let d_hi = I::sub64(h1, borrow_lo);
        let bw2 = I::lt_u64(h1, borrow_lo);
        let borrowed = I::or64(bw1, bw2);

        let corr_lo = I::add64(d_lo, I::splat64(Self::P_LO));
        let carry = I::and64(I::lt_u64(corr_lo, d_lo), one);
        let corr_hi = I::add64(I::add64(d_hi, I::splat64(Self::P_HI)), carry);
        Self {
            lo: I::select64(borrowed, corr_lo, d_lo),
            hi: I::select64(borrowed, corr_hi, d_hi),
        }
    }

    /// Lane-by-lane scalar multiply (inline-asm kernel on AArch64).
    #[inline(always)]
    fn mul_raw(a: Self, b: Self) -> Self {
        let mut lo = [0u64; 8];
        let mut hi = [0u64; 8];
        for (i, (l, h)) in lo.iter_mut().zip(hi.iter_mut()).enumerate().take(I::W64) {
            let x = Fp128::<P>([I::v64_lane(a.lo, i), I::v64_lane(a.hi, i)]);
            let y = Fp128::<P>([I::v64_lane(b.lo, i), I::v64_lane(b.hi, i)]);
            let r = (x * y).0;
            (*l, *h) = (r[0], r[1]);
        }
        Self {
            lo: I::v64_from_fn(|i| lo[i]),
            hi: I::v64_from_fn(|i| hi[i]),
        }
    }
}

impl_packed_arith!(impl[const P: u128, I: SimdWord] PackedFp128<P, I>);

impl<const P: u128, I: SimdWord> Packed for PackedFp128<P, I> {
    type Scalar = Fp128<P>;
    const WIDTH: usize = I::W64;

    #[inline]
    fn from_fn(mut f: impl FnMut(usize) -> Fp128<P>) -> Self {
        let mut lo = [0u64; 8];
        let mut hi = [0u64; 8];
        for (i, (l, h)) in lo.iter_mut().zip(hi.iter_mut()).enumerate().take(I::W64) {
            let v = f(i).0;
            (*l, *h) = (v[0], v[1]);
        }
        Self {
            lo: I::v64_from_fn(|i| lo[i]),
            hi: I::v64_from_fn(|i| hi[i]),
        }
    }

    #[inline]
    fn extract(&self, lane: usize) -> Fp128<P> {
        debug_assert!(lane < I::W64);
        Fp128([I::v64_lane(self.lo, lane), I::v64_lane(self.hi, lane)])
    }

    #[inline]
    fn broadcast(value: Fp128<P>) -> Self {
        Self {
            lo: I::splat64(value.0[0]),
            hi: I::splat64(value.0[1]),
        }
    }
}
