//! Implements `dory_pcs` traits for [`Fr`], allowing jolt-dory to use
//! jolt field types directly without arkworks-to-jolt conversions.

use crate::arkworks::bn254::Fr;
use crate::Field;

use ark_serialize::CanonicalDeserialize;
use ark_serialize::CanonicalSerialize;
use dory::primitives::arithmetic;
use dory::primitives::serialization::{
    Compress, DoryDeserialize, DorySerialize, SerializationError, Valid, Validate,
};
use rand_core::OsRng;
use std::io::{Read, Write};

type InnerFr = ark_bn254::Fr;

// Serialization bridge
//
// dory-pcs serialization enums mirror arkworks's, so we convert between them.

#[inline(always)]
fn to_ark_compress(c: Compress) -> ark_serialize::Compress {
    match c {
        Compress::Yes => ark_serialize::Compress::Yes,
        Compress::No => ark_serialize::Compress::No,
    }
}

#[inline(always)]
fn to_ark_validate(v: Validate) -> ark_serialize::Validate {
    match v {
        Validate::Yes => ark_serialize::Validate::Yes,
        Validate::No => ark_serialize::Validate::No,
    }
}

fn ark_err_to_dory(e: ark_serialize::SerializationError) -> SerializationError {
    match e {
        ark_serialize::SerializationError::IoError(io) => SerializationError::IoError(io),
        ark_serialize::SerializationError::InvalidData => {
            SerializationError::InvalidData("arkworks: invalid data".into())
        }
        ark_serialize::SerializationError::UnexpectedFlags => SerializationError::UnexpectedData,
        ark_serialize::SerializationError::NotEnoughSpace => {
            SerializationError::InvalidData("arkworks: not enough space".into())
        }
    }
}

impl Valid for Fr {
    fn check(&self) -> Result<(), SerializationError> {
        ark_serialize::Valid::check(&self.0).map_err(ark_err_to_dory)
    }
}

impl DorySerialize for Fr {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0
            .serialize_with_mode(writer, to_ark_compress(compress))
            .map_err(ark_err_to_dory)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.0.serialized_size(to_ark_compress(compress))
    }
}

impl DoryDeserialize for Fr {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        InnerFr::deserialize_with_mode(reader, to_ark_compress(compress), to_ark_validate(validate))
            .map(Fr)
            .map_err(ark_err_to_dory)
    }
}

// dory::Field

impl arithmetic::Field for Fr {
    #[inline(always)]
    fn zero() -> Self {
        Field::from_u64(0)
    }

    #[inline(always)]
    fn one() -> Self {
        Field::from_u64(1)
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        *self == <Self as arithmetic::Field>::zero()
    }

    #[inline(always)]
    fn add(&self, rhs: &Self) -> Self {
        *self + *rhs
    }

    #[inline(always)]
    fn sub(&self, rhs: &Self) -> Self {
        *self - *rhs
    }

    #[inline(always)]
    fn mul(&self, rhs: &Self) -> Self {
        *self * *rhs
    }

    #[inline(always)]
    fn inv(self) -> Option<Self> {
        Field::inverse(&self)
    }

    #[inline]
    fn random() -> Self {
        Field::random(&mut OsRng)
    }

    #[inline(always)]
    fn from_u64(val: u64) -> Self {
        Field::from_u64(val)
    }

    #[inline(always)]
    fn from_i64(val: i64) -> Self {
        Field::from_i64(val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dory::primitives::arithmetic::Field as DoryField;

    #[test]
    fn field_zero_one() {
        let zero = <Fr as DoryField>::zero();
        let one = <Fr as DoryField>::one();
        assert!(DoryField::is_zero(&zero));
        assert!(!DoryField::is_zero(&one));
        assert_eq!(DoryField::add(&zero, &one), one);
        assert_eq!(DoryField::mul(&zero, &one), zero);
    }

    #[test]
    fn field_add_sub_mul() {
        let a = <Fr as DoryField>::from_u64(7);
        let b = <Fr as DoryField>::from_u64(11);
        assert_eq!(DoryField::add(&a, &b), <Fr as DoryField>::from_u64(18));
        assert_eq!(DoryField::sub(&b, &a), <Fr as DoryField>::from_u64(4));
        assert_eq!(DoryField::mul(&a, &b), <Fr as DoryField>::from_u64(77));
    }

    #[test]
    fn field_inv() {
        let a = <Fr as DoryField>::from_u64(42);
        let inv_a = DoryField::inv(a).expect("nonzero element must have inverse");
        assert_eq!(DoryField::mul(&a, &inv_a), <Fr as DoryField>::one());

        let zero = <Fr as DoryField>::zero();
        assert!(DoryField::inv(zero).is_none());
    }

    #[test]
    fn field_from_i64_negative() {
        let neg_one = <Fr as DoryField>::from_i64(-1);
        let one = <Fr as DoryField>::one();
        assert_eq!(DoryField::add(&neg_one, &one), <Fr as DoryField>::zero());
    }

    #[test]
    fn field_serialization_roundtrip() {
        let val = <Fr as DoryField>::from_u64(123456789);
        let mut buf = Vec::new();
        DorySerialize::serialize_compressed(&val, &mut buf).unwrap();
        let recovered: Fr = DoryDeserialize::deserialize_compressed(&buf[..]).unwrap();
        assert_eq!(val, recovered);
    }

    #[test]
    fn field_random_nonzero() {
        // Statistical: 100 random elements should not all be zero
        let vals: Vec<Fr> = (0..100).map(|_| <Fr as DoryField>::random()).collect();
        assert!(vals.iter().any(|v| !DoryField::is_zero(v)));
    }
}
