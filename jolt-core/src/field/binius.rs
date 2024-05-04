use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use super::JoltField;

#[derive(Default, Debug, Copy, Clone, Eq, PartialEq)]
pub struct BiniusField<F: binius_field::TowerField>(F);

impl<F: binius_field::TowerField> BiniusField<F> {}

impl<F: binius_field::TowerField> JoltField for BiniusField<F> {
    const NUM_BYTES: usize = 16;

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        Self(F::random(rng))
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero().into()
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
        todo!()
    }

    fn square(&self) -> Self {
        Self(self.0.square())
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        todo!()
    }
}

impl<F: binius_field::TowerField> Neg for BiniusField<F> {
    type Output = Self;

    fn neg(self) -> Self {
        BiniusField(self.0.neg())
    }
}

impl<F: binius_field::TowerField> Add for BiniusField<F> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        BiniusField(self.0.add(other.0))
    }
}

impl<F: binius_field::TowerField> Sub for BiniusField<F> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        BiniusField(self.0.sub(other.0))
    }
}

impl<F: binius_field::TowerField> Mul for BiniusField<F> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        BiniusField(self.0.mul(other.0))
    }
}

impl<F: binius_field::TowerField> Div for BiniusField<F> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        todo!()
        // BiniusField(self.0.div(other.0))
    }
}

impl<F: binius_field::TowerField> AddAssign for BiniusField<F> {
    fn add_assign(&mut self, other: Self) {
        self.0.add_assign(other.0);
    }
}

impl<F: binius_field::TowerField> SubAssign for BiniusField<F> {
    fn sub_assign(&mut self, other: Self) {
        self.0.sub_assign(other.0);
    }
}

impl<F: binius_field::TowerField> MulAssign for BiniusField<F> {
    fn mul_assign(&mut self, other: Self) {
        self.0.mul_assign(other.0);
    }
}

impl<F: binius_field::TowerField> core::iter::Sum for BiniusField<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| BiniusField(acc.0.add(x.0)))
    }
}

impl<'a, F: binius_field::TowerField> core::iter::Sum<&'a Self> for BiniusField<F> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| BiniusField(acc.0.add(x.0)))
    }
}

impl<F: binius_field::TowerField> core::iter::Product for BiniusField<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| BiniusField(acc.0.mul(x.0)))
    }
}

impl<'a, F: binius_field::TowerField> core::iter::Product<&'a Self> for BiniusField<F> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| BiniusField(acc.0.mul(x.0)))
    }
}

impl<F: binius_field::TowerField> CanonicalSerialize for BiniusField<F> {
    fn serialize_with_mode<W: std::io::prelude::Write>(
        &self,
        writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        todo!()
    }

    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        todo!()
    }
}

impl<F: binius_field::TowerField> CanonicalDeserialize for BiniusField<F> {
    // Required method
    fn deserialize_with_mode<R: std::io::prelude::Read>(
        reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        todo!()
    }
}

impl<F: binius_field::TowerField> ark_serialize::Valid for BiniusField<F> {
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        todo!()
    }
}
