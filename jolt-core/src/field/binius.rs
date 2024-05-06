use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use binius_field::{BinaryField128b, BinaryField128bPolyval};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use super::JoltField;

// TODO(sragss): Likely don't need both Pod and BiniusConstructable. Most of the binius types
// already implement pod: https://gitlab.com/search?search=pod&nav_source=navbar&project_id=52331778&group_id=78402358&search_code=true&repository_ref=main


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

impl BiniusGeneric for BinaryField128b {}
impl BiniusGeneric for BinaryField128bPolyval {}

pub trait BiniusGeneric: binius_field::TowerField + BiniusConstructable + bytemuck::Pod {}

pub trait BiniusConstructable {
    fn new(n: u64) -> Self;
}

#[derive(Default, Debug, Copy, Clone, Eq, PartialEq)]
pub struct BiniusField<F: BiniusGeneric>(F);

impl<F: BiniusGeneric> JoltField for BiniusField<F> {
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
        // let field_element = bytemuck::cast::<u64, F>(n);
        // Some(Self(field_element))
        Some(Self(F::new(n)))
    }

    fn square(&self) -> Self {
        Self(self.0.square())
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), Self::NUM_BYTES);

        let field_element = bytemuck::try_from_bytes::<F>(bytes).unwrap();
        Self(field_element.to_owned())
    }
}

impl<F: BiniusGeneric> Neg for BiniusField<F> {
    type Output = Self;

    fn neg(self) -> Self {
        BiniusField(self.0.neg())
    }
}

impl<F: BiniusGeneric> Add for BiniusField<F> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        BiniusField(self.0.add(other.0))
    }
}

impl<F: BiniusGeneric> Sub for BiniusField<F> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        BiniusField(self.0.sub(other.0))
    }
}

impl<F: BiniusGeneric> Mul for BiniusField<F> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        BiniusField(self.0.mul(other.0))
    }
}

impl<F: BiniusGeneric> Div for BiniusField<F> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        todo!()
        // BiniusField(self.0.div(other.0))
    }
}

impl<F: BiniusGeneric> AddAssign for BiniusField<F> {
    fn add_assign(&mut self, other: Self) {
        self.0.add_assign(other.0);
    }
}

impl<F: BiniusGeneric> SubAssign for BiniusField<F> {
    fn sub_assign(&mut self, other: Self) {
        self.0.sub_assign(other.0);
    }
}

impl<F: BiniusGeneric> MulAssign for BiniusField<F> {
    fn mul_assign(&mut self, other: Self) {
        self.0.mul_assign(other.0);
    }
}

impl<F: BiniusGeneric> core::iter::Sum for BiniusField<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| BiniusField(acc.0.add(x.0)))
    }
}

impl<'a, F: BiniusGeneric> core::iter::Sum<&'a Self> for BiniusField<F> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| BiniusField(acc.0.add(x.0)))
    }
}

impl<F: BiniusGeneric> core::iter::Product for BiniusField<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| BiniusField(acc.0.mul(x.0)))
    }
}

impl<'a, F: BiniusGeneric> core::iter::Product<&'a Self> for BiniusField<F> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| BiniusField(acc.0.mul(x.0)))
    }
}

impl<F: BiniusGeneric> CanonicalSerialize for BiniusField<F> {
    fn serialize_with_mode<W: ark_std::io::Write>(
        &self,
        mut writer: W,
        _compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
            let bytes = bytemuck::bytes_of(&self.0);
            writer.write_all(&bytes)?;
            Ok(())
    }

    fn serialized_size(&self, _compress: ark_serialize::Compress) -> usize {
        Self::NUM_BYTES
    }
}

impl<F: BiniusGeneric> CanonicalDeserialize for BiniusField<F> {
    // Required method
    fn deserialize_with_mode<R: std::io::prelude::Read>(
        reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        todo!()
    }
}

impl<F: BiniusGeneric> ark_serialize::Valid for BiniusField<F> {
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        todo!()
    }
}
