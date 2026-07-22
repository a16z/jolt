use super::*;

/// Number of packed `Fp128` lanes in this backend.
pub(crate) const FP128_WIDTH: usize = 2;

/// True SoA layout for two packed `Fp128` lanes.
///
/// `lo = [lane0.lo, lane1.lo]`
/// `hi = [lane0.hi, lane1.hi]`
#[derive(Clone, Copy)]
pub struct PackedFp128Neon<const P: u128> {
    lo: [u64; 2],
    hi: [u64; 2],
}
#[inline(always)]
const fn modulus_lo<const P: u128>() -> u64 {
    P as u64
}

#[inline(always)]
const fn modulus_hi<const P: u128>() -> u64 {
    (P >> 64) as u64
}

use crate::prime::util::{is_pow2_u64, log2_pow2_u64};
impl<const P: u128> PackedFp128Neon<P> {
    const C: u128 = {
        let c = 0u128.wrapping_sub(P);
        assert!(P != 0, "modulus must be nonzero");
        assert!(P & 1 == 1, "modulus must be odd");
        assert!(c < (1u128 << 64), "P must be 2^128 - c with c < 2^64");
        assert!(
            c * (c + 1) < P,
            "C(C+1) < P required for fused canonicalize"
        );
        c
    };
    const C_LO: u64 = Self::C as u64;
    const C_SHIFT_KIND: i8 = {
        let c = Self::C_LO;
        if c > 1 && is_pow2_u64(c - 1) {
            1
        } else if c == u64::MAX || is_pow2_u64(c + 1) {
            -1
        } else {
            0
        }
    };
    const C_SHIFT: u32 = {
        let c = Self::C_LO;
        if Self::C_SHIFT_KIND == 1 {
            log2_pow2_u64(c - 1)
        } else if Self::C_SHIFT_KIND == -1 {
            if c == u64::MAX {
                64
            } else {
                log2_pow2_u64(c + 1)
            }
        } else {
            0
        }
    };

    #[inline(always)]
    fn mul_wide_u64(a: u64, b: u64) -> (u64, u64) {
        let prod = (a as u128) * (b as u128);
        (prod as u64, (prod >> 64) as u64)
    }

    #[inline(always)]
    fn mul_c_wide(x: u64) -> (u64, u64) {
        if Self::C_SHIFT_KIND == 1 {
            let v = ((x as u128) << Self::C_SHIFT) + x as u128;
            (v as u64, (v >> 64) as u64)
        } else if Self::C_SHIFT_KIND == -1 {
            let v = ((x as u128) << Self::C_SHIFT) - x as u128;
            (v as u64, (v >> 64) as u64)
        } else {
            Self::mul_wide_u64(Self::C_LO, x)
        }
    }

    #[inline(always)]
    fn fold2_canonicalize(t0: u64, t1: u64, t2: u64) -> (u64, u64) {
        let (ct2_lo, ct2_hi) = Self::mul_c_wide(t2);

        let (s0, carry0) = t0.overflowing_add(ct2_lo);
        let (s1a, carry1a) = t1.overflowing_add(ct2_hi);
        let (s1, carry1b) = s1a.overflowing_add(carry0 as u64);
        let overflow = carry1a | carry1b;

        let (r0, carry2) = s0.overflowing_add(Self::C_LO);
        let (r1, carry3) = s1.overflowing_add(carry2 as u64);

        if overflow | carry3 {
            (r0, r1)
        } else {
            (s0, s1)
        }
    }

    #[inline(always)]
    fn mul_raw_lane(a0: u64, a1: u64, b0: u64, b1: u64) -> (u64, u64) {
        let (p00_lo, p00_hi) = Self::mul_wide_u64(a0, b0);
        let (p01_lo, p01_hi) = Self::mul_wide_u64(a0, b1);
        let (p10_lo, p10_hi) = Self::mul_wide_u64(a1, b0);
        let (p11_lo, p11_hi) = Self::mul_wide_u64(a1, b1);

        let row1 = p00_hi as u128 + p01_lo as u128 + p10_lo as u128;
        let r0 = p00_lo;
        let r1 = row1 as u64;
        let carry1 = (row1 >> 64) as u64;

        let row2 = p01_hi as u128 + p10_hi as u128 + p11_lo as u128 + carry1 as u128;
        let r2 = row2 as u64;
        let carry2 = (row2 >> 64) as u64;

        let row3 = p11_hi as u128 + carry2 as u128;
        let r3 = row3 as u64;
        debug_assert_eq!(row3 >> 64, 0);

        let (cr2_lo, cr2_hi) = Self::mul_c_wide(r2);
        let (cr3_lo, cr3_hi) = Self::mul_c_wide(r3);

        let t0_sum = r0 as u128 + cr2_lo as u128;
        let t0 = t0_sum as u64;
        let carryf = (t0_sum >> 64) as u64;

        let t1_sum = r1 as u128 + cr2_hi as u128 + cr3_lo as u128 + carryf as u128;
        let t1 = t1_sum as u64;

        let t2_sum = cr3_hi as u128 + (t1_sum >> 64);
        let t2 = t2_sum as u64;
        debug_assert_eq!(t2_sum >> 64, 0);

        Self::fold2_canonicalize(t0, t1, t2)
    }
}

impl<const P: u128> Default for PackedFp128Neon<P> {
    #[inline]
    fn default() -> Self {
        Self::broadcast(Fp128::zero())
    }
}

impl<const P: u128> fmt::Debug for PackedFp128Neon<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PackedFp128Neon")
            .field(&[self.extract(0), self.extract(1)])
            .finish()
    }
}

impl<const P: u128> PartialEq for PackedFp128Neon<P> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.extract(0) == other.extract(0) && self.extract(1) == other.extract(1)
    }
}

impl<const P: u128> Eq for PackedFp128Neon<P> {}

impl<const P: u128> Add for PackedFp128Neon<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let lo_a = to_vec(self.lo);
        let hi_a = to_vec(self.hi);
        let lo_b = to_vec(rhs.lo);
        let hi_b = to_vec(rhs.hi);

        let (out_lo, out_hi) = unsafe {
            let c_vec = vdupq_n_u64(Self::C_LO);

            // s = a + b (128-bit, two lanes).
            // Carry propagation uses raw comparison masks with sub: subtracting
            // a lane of all-1s is equivalent to adding 1 in wrapping arithmetic.
            let sum_lo = vaddq_u64(lo_a, lo_b);
            let carry_lo = vcltq_u64(sum_lo, lo_a);

            let hi_tmp = vaddq_u64(hi_a, hi_b);
            let carry_hi1 = vcltq_u64(hi_tmp, hi_a);
            let sum_hi = vsubq_u64(hi_tmp, carry_lo);
            let carry_hi2 = vcltq_u64(sum_hi, hi_tmp);
            let overflow = vorrq_u64(carry_hi1, carry_hi2);

            // t = s + C.  Since p = 2^128 - C, this is s - p (mod 2^128).
            // If s + C >= 2^128 then s >= p, so the reduced value t is correct.
            let t_lo = vaddq_u64(sum_lo, c_vec);
            let carry_c = vcltq_u64(t_lo, sum_lo);
            let t_hi = vsubq_u64(sum_hi, carry_c);
            let carry_t = vcltq_u64(t_hi, sum_hi);

            let use_reduced = vorrq_u64(overflow, carry_t);
            let out_lo = vbslq_u64(use_reduced, t_lo, sum_lo);
            let out_hi = vbslq_u64(use_reduced, t_hi, sum_hi);
            (out_lo, out_hi)
        };

        Self {
            lo: from_vec(out_lo),
            hi: from_vec(out_hi),
        }
    }
}

impl<const P: u128> Sub for PackedFp128Neon<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let lo_a = to_vec(self.lo);
        let hi_a = to_vec(self.hi);
        let lo_b = to_vec(rhs.lo);
        let hi_b = to_vec(rhs.hi);

        let (out_lo, out_hi) = unsafe {
            let p_lo = vdupq_n_u64(modulus_lo::<P>());
            let p_hi = vdupq_n_u64(modulus_hi::<P>());

            let diff_lo = vsubq_u64(lo_a, lo_b);
            let borrow_lo = mask_to_bit(vcltq_u64(lo_a, lo_b));

            let diff_hi_tmp = vsubq_u64(hi_a, hi_b);
            let borrow_hi1 = vcltq_u64(hi_a, hi_b);
            let diff_hi = vsubq_u64(diff_hi_tmp, borrow_lo);
            let borrow_hi2 = vcltq_u64(diff_hi_tmp, borrow_lo);
            let borrow_128 = vorrq_u64(borrow_hi1, borrow_hi2);

            let corr_lo = vaddq_u64(diff_lo, p_lo);
            let carry_lo = mask_to_bit(vcltq_u64(corr_lo, diff_lo));

            let corr_hi_tmp = vaddq_u64(diff_hi, p_hi);
            let corr_hi = vaddq_u64(corr_hi_tmp, carry_lo);

            let out_lo = vbslq_u64(borrow_128, corr_lo, diff_lo);
            let out_hi = vbslq_u64(borrow_128, corr_hi, diff_hi);
            (out_lo, out_hi)
        };

        Self {
            lo: from_vec(out_lo),
            hi: from_vec(out_hi),
        }
    }
}

impl<const P: u128> Mul for PackedFp128Neon<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let (o0_lo, o0_hi) = Self::mul_raw_lane(self.lo[0], self.hi[0], rhs.lo[0], rhs.hi[0]);
        let (o1_lo, o1_hi) = Self::mul_raw_lane(self.lo[1], self.hi[1], rhs.lo[1], rhs.hi[1]);

        Self {
            lo: [o0_lo, o1_lo],
            hi: [o0_hi, o1_hi],
        }
    }
}

impl<const P: u128> AddAssign for PackedFp128Neon<P> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u128> SubAssign for PackedFp128Neon<P> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u128> MulAssign for PackedFp128Neon<P> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const P: u128> PackedField for PackedFp128Neon<P> {
    const WIDTH: usize = FP128_WIDTH;

    #[inline]
    fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize) -> Self::Scalar,
    {
        let x0 = f(0);
        let x1 = f(1);
        Self {
            lo: [x0.0[0], x1.0[0]],
            hi: [x0.0[1], x1.0[1]],
        }
    }

    #[inline]
    fn extract(&self, lane: usize) -> Self::Scalar {
        debug_assert!(lane < FP128_WIDTH);
        Fp128([self.lo[lane], self.hi[lane]])
    }

    type Scalar = Fp128<P>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self::from_fn(|_| value)
    }
}
