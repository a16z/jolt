use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid,
};
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

use super::{compact_polynomial::CompactPolynomial, dense_mlpoly::DensePolynomial};
use crate::field::JoltField;

#[repr(u8)]
#[derive(Clone, Debug, EnumIter)]
pub enum MultilinearPolynomial<F: JoltField> {
    LargeScalars(DensePolynomial<F>),
    U8Scalars(CompactPolynomial<u8, F>),
    U16Scalars(CompactPolynomial<u16, F>),
    U32Scalars(CompactPolynomial<u32, F>),
    U64Scalars(CompactPolynomial<u64, F>),
}

impl<F: JoltField> Default for MultilinearPolynomial<F> {
    fn default() -> Self {
        Self::LargeScalars(DensePolynomial::default())
    }
}

impl<F: JoltField> MultilinearPolynomial<F> {
    pub fn len(&self) -> usize {
        todo!()
    }

    pub fn get_num_vars(&self) -> usize {
        todo!()
    }
}

impl<F: JoltField> From<Vec<F>> for MultilinearPolynomial<F> {
    fn from(coeffs: Vec<F>) -> Self {
        let poly = DensePolynomial::new(coeffs);
        Self::LargeScalars(poly)
    }
}

impl<F: JoltField> From<Vec<u8>> for MultilinearPolynomial<F> {
    fn from(coeffs: Vec<u8>) -> Self {
        let poly = CompactPolynomial::from_coeffs(coeffs);
        Self::U8Scalars(poly)
    }
}

impl<F: JoltField> From<Vec<u16>> for MultilinearPolynomial<F> {
    fn from(coeffs: Vec<u16>) -> Self {
        let poly = CompactPolynomial::from_coeffs(coeffs);
        Self::U16Scalars(poly)
    }
}

impl<F: JoltField> From<Vec<u32>> for MultilinearPolynomial<F> {
    fn from(coeffs: Vec<u32>) -> Self {
        let poly = CompactPolynomial::from_coeffs(coeffs);
        Self::U32Scalars(poly)
    }
}

impl<F: JoltField> From<Vec<u64>> for MultilinearPolynomial<F> {
    fn from(coeffs: Vec<u64>) -> Self {
        let poly = CompactPolynomial::from_coeffs(coeffs);
        Self::U64Scalars(poly)
    }
}

impl<'a, F: JoltField> TryFrom<&'a MultilinearPolynomial<F>> for &'a DensePolynomial<F> {
    type Error = (); // TODO(moodlezoup)

    fn try_from(poly: &'a MultilinearPolynomial<F>) -> Result<Self, Self::Error> {
        match poly {
            MultilinearPolynomial::LargeScalars(poly) => Ok(poly),
            MultilinearPolynomial::U8Scalars(_) => Err(()),
            MultilinearPolynomial::U16Scalars(_) => Err(()),
            MultilinearPolynomial::U32Scalars(_) => Err(()),
            MultilinearPolynomial::U64Scalars(_) => Err(()),
        }
    }
}

impl<'a, F: JoltField> TryFrom<&'a MultilinearPolynomial<F>> for &'a CompactPolynomial<u8, F> {
    type Error = (); // TODO(moodlezoup)

    fn try_from(poly: &'a MultilinearPolynomial<F>) -> Result<Self, Self::Error> {
        match poly {
            MultilinearPolynomial::LargeScalars(_) => Err(()),
            MultilinearPolynomial::U8Scalars(poly) => Ok(poly),
            MultilinearPolynomial::U16Scalars(_) => Err(()),
            MultilinearPolynomial::U32Scalars(_) => Err(()),
            MultilinearPolynomial::U64Scalars(_) => Err(()),
        }
    }
}

impl<'a, F: JoltField> TryFrom<&'a MultilinearPolynomial<F>> for &'a CompactPolynomial<u16, F> {
    type Error = (); // TODO(moodlezoup)

    fn try_from(poly: &'a MultilinearPolynomial<F>) -> Result<Self, Self::Error> {
        match poly {
            MultilinearPolynomial::LargeScalars(_) => Err(()),
            MultilinearPolynomial::U8Scalars(_) => Err(()),
            MultilinearPolynomial::U16Scalars(poly) => Ok(poly),
            MultilinearPolynomial::U32Scalars(_) => Err(()),
            MultilinearPolynomial::U64Scalars(_) => Err(()),
        }
    }
}

impl<'a, F: JoltField> TryFrom<&'a MultilinearPolynomial<F>> for &'a CompactPolynomial<u32, F> {
    type Error = (); // TODO(moodlezoup)

    fn try_from(poly: &'a MultilinearPolynomial<F>) -> Result<Self, Self::Error> {
        match poly {
            MultilinearPolynomial::LargeScalars(_) => Err(()),
            MultilinearPolynomial::U8Scalars(_) => Err(()),
            MultilinearPolynomial::U16Scalars(_) => Err(()),
            MultilinearPolynomial::U32Scalars(poly) => Ok(poly),
            MultilinearPolynomial::U64Scalars(_) => Err(()),
        }
    }
}

impl<'a, F: JoltField> TryFrom<&'a MultilinearPolynomial<F>> for &'a CompactPolynomial<u64, F> {
    type Error = (); // TODO(moodlezoup)

    fn try_from(poly: &'a MultilinearPolynomial<F>) -> Result<Self, Self::Error> {
        match poly {
            MultilinearPolynomial::LargeScalars(_) => Err(()),
            MultilinearPolynomial::U8Scalars(_) => Err(()),
            MultilinearPolynomial::U16Scalars(_) => Err(()),
            MultilinearPolynomial::U32Scalars(_) => Err(()),
            MultilinearPolynomial::U64Scalars(poly) => Ok(poly),
        }
    }
}

impl<F: JoltField> CanonicalSerialize for MultilinearPolynomial<F> {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        let discriminant = unsafe { *(self as *const Self as *const u8) };
        discriminant.serialize_with_mode(&mut writer, compress)?;
        match self {
            MultilinearPolynomial::LargeScalars(poly) => poly.serialize_with_mode(writer, compress),
            MultilinearPolynomial::U8Scalars(poly) => poly.serialize_with_mode(writer, compress),
            MultilinearPolynomial::U16Scalars(poly) => poly.serialize_with_mode(writer, compress),
            MultilinearPolynomial::U32Scalars(poly) => poly.serialize_with_mode(writer, compress),
            MultilinearPolynomial::U64Scalars(poly) => poly.serialize_with_mode(writer, compress),
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let discriminant = unsafe { *(self as *const Self as *const u8) };
        let poly_size = match self {
            MultilinearPolynomial::LargeScalars(poly) => poly.serialized_size(compress),
            MultilinearPolynomial::U8Scalars(poly) => poly.serialized_size(compress),
            MultilinearPolynomial::U16Scalars(poly) => poly.serialized_size(compress),
            MultilinearPolynomial::U32Scalars(poly) => poly.serialized_size(compress),
            MultilinearPolynomial::U64Scalars(poly) => poly.serialized_size(compress),
        };

        poly_size + discriminant.serialized_size(compress)
    }
}

impl<F: JoltField> CanonicalDeserialize for MultilinearPolynomial<F> {
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, SerializationError> {
        let discriminant: u8 = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        let variants: Vec<_> = Self::iter().collect();
        let deserialized_poly: Self = match &variants[discriminant as usize] {
            MultilinearPolynomial::LargeScalars(_) => MultilinearPolynomial::LargeScalars(
                CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?,
            ),
            MultilinearPolynomial::U8Scalars(_) => MultilinearPolynomial::U8Scalars(
                CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?,
            ),
            MultilinearPolynomial::U16Scalars(_) => MultilinearPolynomial::U16Scalars(
                CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?,
            ),
            MultilinearPolynomial::U32Scalars(_) => MultilinearPolynomial::U32Scalars(
                CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?,
            ),
            MultilinearPolynomial::U64Scalars(_) => MultilinearPolynomial::U64Scalars(
                CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?,
            ),
        };
        Ok(deserialized_poly)
    }
}

impl<F: JoltField> Valid for MultilinearPolynomial<F> {
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            MultilinearPolynomial::LargeScalars(poly) => poly.check(),
            MultilinearPolynomial::U8Scalars(poly) => poly.check(),
            MultilinearPolynomial::U16Scalars(poly) => poly.check(),
            MultilinearPolynomial::U32Scalars(poly) => poly.check(),
            MultilinearPolynomial::U64Scalars(poly) => poly.check(),
        }
    }
}

pub trait PolynomialBinding<F: JoltField> {
    fn bind(&mut self, r: F);
    fn bind_parallel(&mut self, r: F);
}

pub trait PolynomialEvaluation<F: JoltField> {
    fn evaluate(&self, r: &[F]) -> F;
    fn evaluate_with_chis(&self, chis: &[F]) -> F;
}

impl<F: JoltField> PolynomialBinding<F> for MultilinearPolynomial<F> {
    fn bind(&mut self, r: F) {
        todo!()
    }

    fn bind_parallel(&mut self, r: F) {
        todo!()
    }
}

impl<F: JoltField> PolynomialEvaluation<F> for MultilinearPolynomial<F> {
    fn evaluate(&self, r: &[F]) -> F {
        todo!()
    }

    fn evaluate_with_chis(&self, chis: &[F]) -> F {
        todo!()
    }
}
