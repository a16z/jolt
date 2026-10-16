use super::inverse;
use crate::{CanonicalBytes, CanonicalEncoding, Field, NaiveAccumulator, Ring, WithAccumulator};
#[cfg(feature = "allocative")]
use allocative::Allocative;
use rand_core::RngCore;
use std::fmt::{Display, Formatter, Result as FmtResult};

/// `F_2[x]/(x^8 + x^4 + x^3 + x + 1)`, with bit i the coefficient of x^i.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "allocative", derive(Allocative))]
pub struct F8(u8);

impl F8 {
    /// Embeds polynomial coefficients, distinct from the integer ring map.
    pub const fn from_raw(value: u8) -> Self {
        Self(value)
    }

    /// Returns the polynomial coefficient vector in low-bit-first order.
    pub const fn to_raw(self) -> u8 {
        self.0
    }

    fn multiply(self, rhs: Self) -> Self {
        let mut a = self.0;
        let mut b = rhs.0;
        let mut product = 0;
        for _ in 0..8 {
            if b & 1 != 0 {
                product ^= a;
            }
            let carry = a >> 7;
            a = (a << 1) ^ (carry * 0x1b);
            b >>= 1;
        }
        Self(product)
    }

    fn add_coefficients(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}

impl Display for F8 {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{:02x}", self.0)
    }
}

crate::impl_ring_ops!(impl[] F8 {
    add(a, b): a.add_coefficients(b),
    sub(a, b): a.add_coefficients(b),
    mul(a, b): a.multiply(b),
    neg(a): a,
    zero: F8(0),
    one: F8(1),
});

impl Ring for F8 {
    #[inline]
    fn square(&self) -> Self {
        self.multiply(*self)
    }

    fn from_u64(v: u64) -> Self {
        Self((v & 1) as u8)
    }

    fn from_i64(v: i64) -> Self {
        Self::from_u64(v as u64)
    }

    fn from_u128(v: u128) -> Self {
        Self::from_u64(v as u64)
    }

    fn from_i128(v: i128) -> Self {
        Self::from_u64(v as u64)
    }
}

impl Field for F8 {
    fn inverse(&self) -> Option<Self> {
        inverse(*self)
    }

    fn random<R: RngCore>(rng: &mut R) -> Self {
        let mut bytes = [0; 1];
        rng.fill_bytes(&mut bytes);
        Self(u8::from_le_bytes(bytes))
    }
}

impl CanonicalBytes for F8 {
    const NUM_BYTES: usize = 1;

    fn to_bytes_le(&self, out: &mut [u8]) {
        out.copy_from_slice(&self.0.to_le_bytes());
    }
}

impl CanonicalEncoding for F8 {
    const MODULUS_BITS: u32 = 9;

    fn from_bytes_le_reduced(bytes: &[u8]) -> Self {
        let mut word = [0; 1];
        for (dest, src) in word.iter_mut().zip(bytes) {
            *dest = *src;
        }
        Self(u8::from_le_bytes(word))
    }

    fn from_bytes_le_checked(bytes: &[u8]) -> Option<Self> {
        Some(Self(u8::from_le_bytes(bytes.try_into().ok()?)))
    }

    fn to_u128_checked(&self) -> Option<u128> {
        Some(self.0 as u128)
    }

    fn from_u128_checked(v: u128) -> Option<Self> {
        Some(Self(u8::try_from(v).ok()?))
    }

    fn from_u128_reduced(v: u128) -> Self {
        Self(v as u8)
    }

    fn num_bits(&self) -> u32 {
        u8::BITS - self.0.leading_zeros()
    }

    fn from_scalar_challenge_bytes(bytes: &[u8]) -> Self {
        Self::from_bytes_le_reduced(bytes)
    }
}

impl WithAccumulator for F8 {
    type Accumulator = NaiveAccumulator<Self>;
    type SmallScalarAccumulator = NaiveAccumulator<Self>;
    type SignedProductAccumulator = NaiveAccumulator<Self>;
}

crate::impl_serde_bytes!(impl[] F8, 1);
