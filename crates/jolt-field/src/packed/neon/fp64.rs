use super::*;

/// Number of packed `Fp64` lanes.
pub(crate) const FP64_WIDTH: usize = 2;

/// NEON packed `Fp64` backend: 2 lanes in `uint64x2_t`.
#[derive(Clone, Copy)]
pub struct PackedFp64Neon<const P: u64> {
    vals: [u64; 2],
}

impl<const P: u64> PackedFp64Neon<P> {
    const BITS: u32 = 64 - P.leading_zeros();

    const C_LO: u64 = {
        let c = if Self::BITS == 64 {
            0u64.wrapping_sub(P)
        } else {
            (1u64 << Self::BITS) - P
        };
        assert!(P != 0, "modulus must be nonzero");
        assert!(P & 1 == 1, "modulus must be odd");
        c
    };

    const MASK64: u64 = if Self::BITS < 64 {
        (1u64 << Self::BITS) - 1
    } else {
        u64::MAX
    };

    const MASK_U128: u128 = if Self::BITS == 64 {
        u64::MAX as u128
    } else {
        (1u128 << Self::BITS) - 1
    };

    const FOLD_IN_U64: bool =
        Self::BITS < 64 && (Self::C_LO as u128) < (1u128 << (64 - Self::BITS));

    #[inline(always)]
    fn mul_c_narrow(x: u64) -> u64 {
        Self::C_LO.wrapping_mul(x)
    }

    #[inline(always)]
    fn reduce_product(x: u128) -> u64 {
        if Self::FOLD_IN_U64 {
            let lo = x as u64;
            let hi = (x >> 64) as u64;
            let high = (lo >> Self::BITS) | (hi << (64 - Self::BITS));
            let f1 = (lo & Self::MASK64).wrapping_add(Self::mul_c_narrow(high));
            let f2 = (f1 & Self::MASK64).wrapping_add(Self::mul_c_narrow(f1 >> Self::BITS));
            let reduced = f2.wrapping_sub(P);
            let borrow = reduced >> 63;
            reduced.wrapping_add(borrow.wrapping_neg() & P)
        } else {
            let f1 =
                (x & Self::MASK_U128) + (Self::C_LO as u128) * ((x >> Self::BITS) as u64 as u128);
            let f2 =
                (f1 & Self::MASK_U128) + (Self::C_LO as u128) * ((f1 >> Self::BITS) as u64 as u128);
            let reduced = f2.wrapping_sub(P as u128);
            let borrow = reduced >> 127;
            reduced.wrapping_add(borrow.wrapping_neg() & (P as u128)) as u64
        }
    }
}

impl<const P: u64> Default for PackedFp64Neon<P> {
    #[inline]
    fn default() -> Self {
        Self { vals: [0; 2] }
    }
}

impl<const P: u64> fmt::Debug for PackedFp64Neon<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PackedFp64Neon").field(&self.vals).finish()
    }
}

impl<const P: u64> PartialEq for PackedFp64Neon<P> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.vals == other.vals
    }
}

impl<const P: u64> Eq for PackedFp64Neon<P> {}

impl<const P: u64> Add for PackedFp64Neon<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let a = to_vec(self.vals);
        let b = to_vec(rhs.vals);
        let result = unsafe {
            let p = vdupq_n_u64(P);
            if Self::BITS == 64 {
                let s = vaddq_u64(a, b);
                let overflow = vcltq_u64(s, a);
                let folded = vaddq_u64(s, vandq_u64(overflow, vdupq_n_u64(Self::C_LO)));
                let reduced = vsubq_u64(folded, p);
                let borrow = vcltq_u64(folded, p);
                vbslq_u64(borrow, folded, reduced)
            } else if Self::BITS <= 62 {
                let s = vaddq_u64(a, b);
                let r = vsubq_u64(s, p);
                let borrow = vcltq_u64(s, p);
                vbslq_u64(borrow, s, r)
            } else {
                let s = vaddq_u64(a, b);
                let overflow = vcltq_u64(s, a);
                let c = vdupq_n_u64(Self::C_LO);
                let s_plus_c = vaddq_u64(s, c);
                let s_minus_p = vsubq_u64(s, p);
                let borrow = vcltq_u64(s, p);
                let no_of = vbslq_u64(borrow, s, s_minus_p);
                vbslq_u64(overflow, s_plus_c, no_of)
            }
        };
        Self {
            vals: from_vec(result),
        }
    }
}

impl<const P: u64> Sub for PackedFp64Neon<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let a = to_vec(self.vals);
        let b = to_vec(rhs.vals);
        let result = unsafe {
            let d = vsubq_u64(a, b);
            let underflow = vcltq_u64(a, b);
            if Self::BITS == 64 {
                vsubq_u64(d, vandq_u64(underflow, vdupq_n_u64(Self::C_LO)))
            } else {
                vbslq_u64(underflow, vaddq_u64(d, vdupq_n_u64(P)), d)
            }
        };
        Self {
            vals: from_vec(result),
        }
    }
}

impl<const P: u64> Mul for PackedFp64Neon<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let x0 = (self.vals[0] as u128) * (rhs.vals[0] as u128);
        let x1 = (self.vals[1] as u128) * (rhs.vals[1] as u128);
        let r0 = Self::reduce_product(x0);
        let r1 = Self::reduce_product(x1);
        Self { vals: [r0, r1] }
    }
}

impl<const P: u64> AddAssign for PackedFp64Neon<P> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u64> SubAssign for PackedFp64Neon<P> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u64> MulAssign for PackedFp64Neon<P> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const P: u64> PackedField for PackedFp64Neon<P> {
    const WIDTH: usize = FP64_WIDTH;

    #[inline]
    fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize) -> Self::Scalar,
    {
        Self {
            vals: [f(0).0, f(1).0],
        }
    }

    #[inline]
    fn extract(&self, lane: usize) -> Self::Scalar {
        debug_assert!(lane < FP64_WIDTH);
        Fp64(self.vals[lane])
    }

    type Scalar = Fp64<P>;

    #[inline]
    fn broadcast(value: Self::Scalar) -> Self {
        Self { vals: [value.0; 2] }
    }

    #[inline(always)]
    fn fp_ext2_mul<C>(a0: Self, a1: Self, b0: Self, b1: Self) -> (Self, Self)
    where
        C: FpExt2Config<Self::Scalar>,
    {
        let v0 = a0 * b0;
        let v1 = a1 * b1;
        let cross = (a0 + a1) * (b0 + b1);
        (
            v0 + C::mul_non_residue(v1, Self::broadcast),
            cross - v0 - v1,
        )
    }
}
