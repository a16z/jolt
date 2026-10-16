use super::accumulator::F128Accumulator;
use super::{arithmetic, inverse};
use crate::{CanonicalBytes, CanonicalEncoding, Field, Ring, WithAccumulator};
#[cfg(feature = "allocative")]
use allocative::Allocative;
use rand_core::RngCore;
use std::fmt::{Display, Formatter, Result as FmtResult};

/// `F_2[x]/(x^128 + x^7 + x^2 + x + 1)`, with bit i the coefficient of x^i.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "allocative", derive(Allocative))]
pub struct F128(u128);

impl F128 {
    /// Embeds polynomial coefficients, distinct from the integer ring map.
    pub const fn from_raw(value: u128) -> Self {
        Self(value)
    }

    /// Returns the polynomial coefficient vector in low-bit-first order.
    pub const fn to_raw(self) -> u128 {
        self.0
    }

    /// Multiplies by the polynomial `x`, represented by `from_raw(2)`.
    /// This is distinct from the integer ring map, which retains only parity.
    #[inline]
    pub const fn mul_x(self) -> Self {
        Self((self.0 << 1) ^ if self.0 >> 127 != 0 { 0x87 } else { 0 })
    }

    /// Multiplies by the degree-below-64 polynomial whose coefficients are
    /// the bits of `word`, rather than the parity scalar of `Ring::mul_u64`.
    /// On carry-less backends, this uses two carry-less multiplications for the
    /// product and one to reduce the 64 overflow bits (degree below 64 times
    /// the degree-seven modulus tail cannot overflow 128 bits again), versus
    /// five on x86-64 or six on aarch64 for the general product.
    #[inline]
    pub fn mul_word(self, word: u64) -> Self {
        Self(arithmetic::multiply128_word(self.0, word))
    }

    fn add_coefficients(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}

impl Display for F128 {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{:x}", self.0)
    }
}

crate::impl_ring_ops!(impl[] F128 {
    add(a, b): a.add_coefficients(b),
    sub(a, b): a.add_coefficients(b),
    mul(a, b): F128(arithmetic::multiply128(a.0, b.0)),
    neg(a): a,
    zero: F128(0),
    one: F128(1),
});

impl Ring for F128 {
    #[inline]
    fn square(&self) -> Self {
        Self(arithmetic::square128(self.0))
    }

    fn from_u64(v: u64) -> Self {
        Self((v & 1) as u128)
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

impl Field for F128 {
    fn inverse(&self) -> Option<Self> {
        inverse(*self)
    }

    fn random<R: RngCore>(rng: &mut R) -> Self {
        let mut bytes = [0; 16];
        rng.fill_bytes(&mut bytes);
        Self(u128::from_le_bytes(bytes))
    }
}

impl CanonicalBytes for F128 {
    const NUM_BYTES: usize = 16;

    fn to_bytes_le(&self, out: &mut [u8]) {
        out.copy_from_slice(&self.0.to_le_bytes());
    }
}

impl CanonicalEncoding for F128 {
    const MODULUS_BITS: u32 = 129;

    fn from_bytes_le_reduced(bytes: &[u8]) -> Self {
        let mut word = [0; 16];
        for (dest, src) in word.iter_mut().zip(bytes) {
            *dest = *src;
        }
        Self(u128::from_le_bytes(word))
    }

    fn from_bytes_le_checked(bytes: &[u8]) -> Option<Self> {
        Some(Self(u128::from_le_bytes(bytes.try_into().ok()?)))
    }

    fn to_u128_checked(&self) -> Option<u128> {
        Some(self.0)
    }

    fn from_u128_checked(v: u128) -> Option<Self> {
        Some(Self(v))
    }

    fn from_u128_reduced(v: u128) -> Self {
        Self(v)
    }

    fn num_bits(&self) -> u32 {
        u128::BITS - self.0.leading_zeros()
    }

    fn from_scalar_challenge_bytes(bytes: &[u8]) -> Self {
        Self::from_bytes_le_reduced(bytes)
    }
}

impl WithAccumulator for F128 {
    type Accumulator = F128Accumulator;
    type SmallScalarAccumulator = F128Accumulator;
    type SignedProductAccumulator = F128Accumulator;
}

crate::impl_serde_bytes!(impl[] F128, 16);
