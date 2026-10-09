use super::accumulator::F64Accumulator;
use super::{arithmetic, inverse};
use crate::{CanonicalBytes, CanonicalEncoding, Field, Ring, WithAccumulator};
#[cfg(feature = "allocative")]
use allocative::Allocative;
use rand_core::RngCore;
use std::fmt::{Display, Formatter, Result as FmtResult};

/// `F_2[x]/(x^64 + x^4 + x^3 + x + 1)`, with bit i the coefficient of x^i.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "allocative", derive(Allocative))]
pub struct F64(u64);

impl F64 {
    /// Embeds polynomial coefficients, distinct from the integer ring map.
    pub const fn from_raw(value: u64) -> Self {
        Self(value)
    }

    /// Returns the polynomial coefficient vector in low-bit-first order.
    pub const fn to_raw(self) -> u64 {
        self.0
    }

    fn add_coefficients(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}

impl Display for F64 {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{:x}", self.0)
    }
}

crate::impl_ring_ops!(impl[] F64 {
    add(a, b): a.add_coefficients(b),
    sub(a, b): a.add_coefficients(b),
    mul(a, b): F64(arithmetic::multiply64(a.0, b.0)),
    neg(a): a,
    zero: F64(0),
    one: F64(1),
});

impl Ring for F64 {
    #[inline]
    fn square(&self) -> Self {
        Self(arithmetic::square64(self.0))
    }

    fn from_u64(v: u64) -> Self {
        Self(v & 1)
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

impl Field for F64 {
    fn inverse(&self) -> Option<Self> {
        inverse(*self)
    }

    fn random<R: RngCore>(rng: &mut R) -> Self {
        let mut bytes = [0; 8];
        rng.fill_bytes(&mut bytes);
        Self(u64::from_le_bytes(bytes))
    }
}

impl CanonicalBytes for F64 {
    const NUM_BYTES: usize = 8;

    fn to_bytes_le(&self, out: &mut [u8]) {
        out.copy_from_slice(&self.0.to_le_bytes());
    }
}

impl CanonicalEncoding for F64 {
    const MODULUS_BITS: u32 = 65;

    fn from_bytes_le_reduced(bytes: &[u8]) -> Self {
        let mut word = [0; 8];
        for (dest, src) in word.iter_mut().zip(bytes) {
            *dest = *src;
        }
        Self(u64::from_le_bytes(word))
    }

    fn from_bytes_le_checked(bytes: &[u8]) -> Option<Self> {
        Some(Self(u64::from_le_bytes(bytes.try_into().ok()?)))
    }

    fn to_u128_checked(&self) -> Option<u128> {
        Some(self.0 as u128)
    }

    fn from_u128_checked(v: u128) -> Option<Self> {
        Some(Self(u64::try_from(v).ok()?))
    }

    fn from_u128_reduced(v: u128) -> Self {
        Self(v as u64)
    }

    fn num_bits(&self) -> u32 {
        u64::BITS - self.0.leading_zeros()
    }

    fn from_scalar_challenge_bytes(bytes: &[u8]) -> Self {
        Self::from_bytes_le_reduced(bytes)
    }
}

impl WithAccumulator for F64 {
    type Accumulator = F64Accumulator;
    type SmallScalarAccumulator = F64Accumulator;
    type SignedProductAccumulator = F64Accumulator;
}

crate::impl_serde_bytes!(impl[] F64, 8);
