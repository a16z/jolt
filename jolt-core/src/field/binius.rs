use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use binius_field::{BinaryField128b, BinaryField128bPolyval};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use super::JoltField;

impl BiniusConstructable for BinaryField128b {
    fn new(n: u64) -> Self {
        Self::new(n as u128)
    }
}

impl BiniusConstructable for BinaryField128bPolyval {
    fn new(n: u64) -> Self {
        Self::new(n as u128)
    }
}

impl BiniusSpecific for BinaryField128b {}
impl BiniusSpecific for BinaryField128bPolyval {}

/// Trait for BiniusField functionality specific to each impl.
pub trait BiniusSpecific: binius_field::TowerField + BiniusConstructable + bytemuck::Pod {}

pub trait BiniusConstructable {
    fn new(n: u64) -> Self;
}

#[derive(Default, Debug, Copy, Clone, Eq, PartialEq)]
pub struct BiniusField<F: BiniusSpecific>(F);

/// Wrapper for all generic BiniusField functionality.
impl<F: BiniusSpecific> JoltField for BiniusField<F> {
    const NUM_BYTES: usize = 16;

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        Self(F::random(rng))
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    fn is_one(&self) -> bool {
        self.0 == Self::one().0
    }

    fn zero() -> Self {
        Self(F::ZERO)
    }

    fn one() -> Self {
        Self(F::ONE)
    }

    fn from_u64(n: u64) -> Option<Self> {
        Some(Self(F::new(n)))
    }

    fn from_i64(val: i64) -> Self {
        if val > 0 {
            <Self as JoltField>::from_u64(val as u64).unwrap()
        } else {
            <Self as JoltField>::from_u64(-val as u64).unwrap()
        }
    }

    fn square(&self) -> Self {
        Self(self.0.square())
    }

    fn inverse(&self) -> Option<Self> {
        self.0.invert().map(Self)
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), Self::NUM_BYTES);

        let field_element = bytemuck::try_from_bytes::<F>(bytes).unwrap();
        Self(field_element.to_owned())
    }
}

impl<F: BiniusSpecific> Neg for BiniusField<F> {
    type Output = Self;

    fn neg(self) -> Self {
        BiniusField(-self.0)
    }
}

impl<F: BiniusSpecific> Add for BiniusField<F> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        BiniusField(self.0 + other.0)
    }
}

impl<F: BiniusSpecific> Sub for BiniusField<F> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        BiniusField(self.0 - other.0)
    }
}

impl<F: BiniusSpecific> Mul for BiniusField<F> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        BiniusField(self.0 * other.0)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<F: BiniusSpecific> Div for BiniusField<F> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self(self.0 * other.0.invert().unwrap())
    }
}

impl<F: BiniusSpecific> AddAssign for BiniusField<F> {
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
    }
}

impl<F: BiniusSpecific> SubAssign for BiniusField<F> {
    fn sub_assign(&mut self, other: Self) {
        self.0 -= other.0;
    }
}

impl<F: BiniusSpecific> MulAssign for BiniusField<F> {
    fn mul_assign(&mut self, other: Self) {
        self.0 *= other.0;
    }
}

impl<F: BiniusSpecific> core::iter::Sum for BiniusField<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| BiniusField(acc.0 + x.0))
    }
}

impl<'a, F: BiniusSpecific> core::iter::Sum<&'a Self> for BiniusField<F> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| BiniusField(acc.0 + x.0))
    }
}

impl<F: BiniusSpecific> core::iter::Product for BiniusField<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| BiniusField(acc.0 * x.0))
    }
}

impl<'a, F: BiniusSpecific> core::iter::Product<&'a Self> for BiniusField<F> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| BiniusField(acc.0 * x.0))
    }
}

impl<F: BiniusSpecific> CanonicalSerialize for BiniusField<F> {
    fn serialize_with_mode<W: ark_std::io::Write>(
        &self,
        mut writer: W,
        _compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        let bytes = bytemuck::bytes_of(&self.0);
        writer.write_all(bytes)?;
        Ok(())
    }

    fn serialized_size(&self, _compress: ark_serialize::Compress) -> usize {
        Self::NUM_BYTES
    }
}

impl<F: BiniusSpecific> CanonicalDeserialize for BiniusField<F> {
    // Required method
    fn deserialize_with_mode<R: std::io::prelude::Read>(
        _reader: R,
        _compress: ark_serialize::Compress,
        _validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        todo!()
    }
}

impl<F: BiniusSpecific> ark_serialize::Valid for BiniusField<F> {
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        todo!()
    }
}
