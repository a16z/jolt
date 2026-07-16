use super::*;

impl<const P: u128> Add for Fp128<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(Self::add_raw(self.0, rhs.0))
    }
}

impl<const P: u128> Sub for Fp128<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(Self::sub_raw(self.0, rhs.0))
    }
}

impl<const P: u128> Mul for Fp128<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(Self::mul_raw(self.0, rhs.0))
    }
}

impl<const P: u128> Neg for Fp128<P> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self(Self::sub_raw(pack(0, 0), self.0))
    }
}

impl<const P: u128> AddAssign for Fp128<P> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u128> SubAssign for Fp128<P> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u128> MulAssign for Fp128<P> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<'a, const P: u128> Add<&'a Self> for Fp128<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: &'a Self) -> Self::Output {
        self + *rhs
    }
}

impl<'a, const P: u128> Sub<&'a Self> for Fp128<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: &'a Self) -> Self::Output {
        self - *rhs
    }
}

impl<'a, const P: u128> Mul<&'a Self> for Fp128<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: &'a Self) -> Self::Output {
        self * *rhs
    }
}

impl<const P: u128> Invertible for Fp128<P> {
    #[inline(always)]
    fn inverse(&self) -> Option<Self> {
        let inv = self.inv_or_zero();
        if self.is_zero() {
            None
        } else {
            Some(inv)
        }
    }

    #[inline(always)]
    fn inv_or_zero(self) -> Self {
        let candidate = self.pow_u128(P.wrapping_sub(2));
        let v = to_u128(self.0);
        let nz = ((v | v.wrapping_neg()) >> 127) & 1;
        let mask = 0u128.wrapping_sub(nz);
        let masked = to_u128(candidate.0) & mask;
        Self(from_u128(masked))
    }
}

impl<const P: u128> HalvingField for Fp128<P> {
    #[inline]
    fn half(self) -> Self {
        let x = to_u128(self.0);
        let half = (x >> 1) + (x & 1) * ((P >> 1) + 1);
        Self(from_u128(half))
    }
}

impl<const P: u128> RandomSampling for Fp128<P> {
    #[inline(always)]
    fn random<R: RngCore>(rng: &mut R) -> Self {
        loop {
            let lo = rng.next_u64();
            let hi = rng.next_u64();
            let x = lo as u128 | (hi as u128) << 64;
            if x < P {
                return Self(pack(lo, hi));
            }
        }
    }
}

impl<const P: u128> FromPrimitiveInt for Fp128<P> {
    #[inline(always)]
    fn from_u64(val: u64) -> Self {
        // For Fp128 pseudo-Mersenne primes, p = 2^128 - c with c < 2^64.
        // Therefore any u64 is always canonical (< p), so this can be a
        // direct limb construction with no reduction path.
        Self::from_u64(val)
    }

    #[inline(always)]
    fn from_i64(val: i64) -> Self {
        Self::from_i64(val)
    }

    #[inline(always)]
    fn from_u128(val: u128) -> Self {
        Self::from_canonical_u128_reduced(val)
    }

    #[inline(always)]
    fn from_i128(val: i128) -> Self {
        if val >= 0 {
            Self::from_u128(val as u128)
        } else {
            -Self::from_u128(val.unsigned_abs())
        }
    }
}

impl<const P: u128> BalancedDigitLookup for Fp128<P> {
    fn digit_lut(log_basis: u32) -> [Self; 64] {
        Self::digit_lut(log_basis)
    }
}

impl<const P: u128> CanonicalField for Fp128<P> {
    fn to_canonical_u128(self) -> u128 {
        to_u128(self.0)
    }

    fn modulus_bits() -> u32 {
        u128::BITS - P.leading_zeros()
    }

    fn from_canonical_u128_checked(val: u128) -> Option<Self> {
        if val < P {
            Some(Self(from_u128(val)))
        } else {
            None
        }
    }

    fn from_canonical_u128_reduced(val: u128) -> Self {
        let (sub, borrow) = val.overflowing_sub(P);
        Self(from_u128(if borrow { val } else { sub }))
    }
}

impl<const P: u128> PseudoMersenneField for Fp128<P> {
    const MODULUS_BITS: u32 = 128;
    const MODULUS_OFFSET: u128 = Self::C;
}
