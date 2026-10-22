use super::accumulator::F192Accumulator;
use super::{arithmetic, inverse, F64};
use crate::{CanonicalBytes, CanonicalEncoding, ExtField, Field, One, Ring, WithAccumulator, Zero};
#[cfg(feature = "allocative")]
use allocative::Allocative;
use rand_core::RngCore;
use std::fmt::{Display, Formatter, Result as FmtResult};

/// `F64[y]/(y^3 + y + 1)`, with coefficients in ascending degree order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "allocative", derive(Allocative))]
pub struct F192([F64; 3]);

impl F192 {
    fn add_coefficients(self, rhs: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] + rhs.0[i]))
    }
}

impl Display for F192 {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let [c0, c1, c2] = self.0;
        write!(
            f,
            "{:016x}{:016x}{:016x}",
            c2.to_raw(),
            c1.to_raw(),
            c0.to_raw()
        )
    }
}

crate::impl_ring_ops!(impl[] F192 {
    add(a, b): a.add_coefficients(b),
    sub(a, b): a.add_coefficients(b),
    mul(a, b): F192(arithmetic::multiply192(a.0.map(F64::to_raw), b.0.map(F64::to_raw)).map(F64::from_raw)),
    neg(a): a,
    zero: F192([F64::zero(); 3]),
    one: F192::lift_base(F64::one()),
});

impl Ring for F192 {
    #[inline]
    fn square(&self) -> Self {
        Self(arithmetic::square192(self.0.map(F64::to_raw)).map(F64::from_raw))
    }

    fn from_u64(v: u64) -> Self {
        Self::lift_base(F64::from_u64(v))
    }

    fn from_i64(v: i64) -> Self {
        Self::lift_base(F64::from_i64(v))
    }

    fn from_u128(v: u128) -> Self {
        Self::lift_base(F64::from_u128(v))
    }

    fn from_i128(v: i128) -> Self {
        Self::lift_base(F64::from_i128(v))
    }
}

impl Field for F192 {
    fn inverse(&self) -> Option<Self> {
        inverse(*self)
    }

    fn random<R: RngCore>(rng: &mut R) -> Self {
        let mut bytes = [0; 24];
        rng.fill_bytes(&mut bytes);
        Self::from_bytes_le_reduced(&bytes)
    }
}

impl ExtField<F64> for F192 {
    const DEGREE: usize = 3;

    fn lift_base(x: F64) -> Self {
        Self([x, F64::zero(), F64::zero()])
    }

    fn mul_base(self, x: F64) -> Self {
        Self(self.0.map(|c| c * x))
    }

    fn from_base_fn<G: FnMut(usize) -> F64>(f: G) -> Self {
        Self(std::array::from_fn(f))
    }

    fn base_coefficient(&self, index: usize) -> F64 {
        self.0[index]
    }

    fn frobenius_pow(self, power: usize) -> Self {
        let mut result = self;
        for _ in 0..power % Self::DEGREE {
            let [c0, c1, c2] = result.0;
            result = Self([c0, c2, c1 + c2]);
        }
        result
    }
}

impl CanonicalBytes for F192 {
    const NUM_BYTES: usize = 24;

    fn to_bytes_le(&self, out: &mut [u8]) {
        assert_eq!(out.len(), Self::NUM_BYTES);
        for (coefficient, bytes) in self.0.iter().zip(out.chunks_exact_mut(F64::NUM_BYTES)) {
            coefficient.to_bytes_le(bytes);
        }
    }
}

impl CanonicalEncoding for F192 {
    const MODULUS_BITS: u32 = 193;

    fn from_bytes_le_reduced(bytes: &[u8]) -> Self {
        let mut coefficients = [F64::zero(); 3];
        for (coefficient, bytes) in coefficients.iter_mut().zip(bytes.chunks(F64::NUM_BYTES)) {
            *coefficient = F64::from_bytes_le_reduced(bytes);
        }
        Self(coefficients)
    }

    fn from_bytes_le_checked(bytes: &[u8]) -> Option<Self> {
        (bytes.len() == Self::NUM_BYTES).then(|| Self::from_bytes_le_reduced(bytes))
    }

    fn to_u128_checked(&self) -> Option<u128> {
        let [c0, c1, c2] = self.0;
        if c1.is_zero() && c2.is_zero() {
            c0.to_u128_checked()
        } else {
            None
        }
    }

    fn from_u128_checked(v: u128) -> Option<Self> {
        F64::from_u128_checked(v).map(Self::lift_base)
    }

    fn from_u128_reduced(v: u128) -> Self {
        Self::lift_base(F64::from_u128_reduced(v))
    }

    fn num_bits(&self) -> u32 {
        self.0
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, c)| (!c.is_zero()).then(|| i as u32 * 64 + c.num_bits()))
            .unwrap_or(0)
    }

    fn from_scalar_challenge_bytes(bytes: &[u8]) -> Self {
        Self::from_bytes_le_reduced(bytes)
    }
}

impl WithAccumulator for F192 {
    type Accumulator = F192Accumulator;
    type SmallScalarAccumulator = F192Accumulator;
    type SignedProductAccumulator = F192Accumulator;
}

crate::impl_serde_bytes!(impl[] F192, 24);
