use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid,
};
use rayon::prelude::*;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

use super::{
    compact_polynomial::CompactPolynomial, dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial,
};
use crate::{
    field::{JoltField, OptimizedMul},
    utils::thread::unsafe_allocate_zero_vec,
};

#[repr(u8)]
#[derive(Clone, Debug, EnumIter, PartialEq)]
pub enum MultilinearPolynomial<F: JoltField> {
    LargeScalars(DensePolynomial<F>),
    U8Scalars(CompactPolynomial<u8, F>),
    U16Scalars(CompactPolynomial<u16, F>),
    U32Scalars(CompactPolynomial<u32, F>),
    U64Scalars(CompactPolynomial<u64, F>),
}

pub enum BindingOrder {
    LowToHigh,
    HighToLow,
}

impl<F: JoltField> Default for MultilinearPolynomial<F> {
    fn default() -> Self {
        Self::LargeScalars(DensePolynomial::default())
    }
}

impl<F: JoltField> MultilinearPolynomial<F> {
    pub fn len(&self) -> usize {
        match self {
            MultilinearPolynomial::LargeScalars(poly) => poly.len(),
            MultilinearPolynomial::U8Scalars(poly) => poly.len(),
            MultilinearPolynomial::U16Scalars(poly) => poly.len(),
            MultilinearPolynomial::U32Scalars(poly) => poly.len(),
            MultilinearPolynomial::U64Scalars(poly) => poly.len(),
        }
    }

    pub fn get_num_vars(&self) -> usize {
        match self {
            MultilinearPolynomial::LargeScalars(poly) => poly.get_num_vars(),
            MultilinearPolynomial::U8Scalars(poly) => poly.get_num_vars(),
            MultilinearPolynomial::U16Scalars(poly) => poly.get_num_vars(),
            MultilinearPolynomial::U32Scalars(poly) => poly.get_num_vars(),
            MultilinearPolynomial::U64Scalars(poly) => poly.get_num_vars(),
        }
    }

    pub fn linear_combination(polynomials: &[&Self], coefficients: &[F]) -> Self {
        let max_length = polynomials.iter().map(|poly| poly.len()).max().unwrap();
        let num_chunks = rayon::current_num_threads()
            .next_power_of_two()
            .min(max_length);
        let chunk_size = (max_length / num_chunks).max(1);

        let lc_coeffs: Vec<F> = (0..num_chunks)
            .into_par_iter()
            .flat_map_iter(|chunk_index| {
                let index = chunk_index * chunk_size;
                let mut chunk = unsafe_allocate_zero_vec::<F>(chunk_size);

                for (coeff, poly) in coefficients.iter().zip(polynomials.iter()) {
                    let poly_len = poly.len();
                    if index >= poly_len {
                        continue;
                    }

                    match poly {
                        MultilinearPolynomial::LargeScalars(poly) => {
                            let poly_evals = &poly.evals_ref()[index..];
                            for (rlc, poly_eval) in chunk.iter_mut().zip(poly_evals.iter()) {
                                *rlc += poly_eval.mul_01_optimized(*coeff);
                            }
                        }
                        MultilinearPolynomial::U8Scalars(poly) => {
                            let coeff_r2 = F::montgomery_r2().unwrap_or(F::one()) * coeff;
                            for (rlc, poly_eval) in chunk.iter_mut().zip(poly.coeffs.iter()) {
                                *rlc += coeff_r2.mul_u64_unchecked(*poly_eval as u64);
                            }
                        }
                        MultilinearPolynomial::U16Scalars(poly) => {
                            let coeff_r2 = F::montgomery_r2().unwrap_or(F::one()) * coeff;
                            for (rlc, poly_eval) in chunk.iter_mut().zip(poly.coeffs.iter()) {
                                *rlc += coeff_r2.mul_u64_unchecked(*poly_eval as u64);
                            }
                        }
                        MultilinearPolynomial::U32Scalars(poly) => {
                            let coeff_r2 = F::montgomery_r2().unwrap_or(F::one()) * coeff;
                            for (rlc, poly_eval) in chunk.iter_mut().zip(poly.coeffs.iter()) {
                                *rlc += coeff_r2.mul_u64_unchecked(*poly_eval as u64);
                            }
                        }
                        MultilinearPolynomial::U64Scalars(poly) => {
                            let coeff_r2 = F::montgomery_r2().unwrap_or(F::one()) * coeff;
                            for (rlc, poly_eval) in chunk.iter_mut().zip(poly.coeffs.iter()) {
                                *rlc += coeff_r2.mul_u64_unchecked(*poly_eval);
                            }
                        }
                    }
                }
                chunk
            })
            .collect();

        MultilinearPolynomial::from(lc_coeffs)
    }

    pub fn get_coeff(&self, index: usize) -> F {
        match self {
            MultilinearPolynomial::LargeScalars(poly) => poly[index],
            MultilinearPolynomial::U8Scalars(poly) => F::from_u8(poly.coeffs[index]),
            MultilinearPolynomial::U16Scalars(poly) => F::from_u16(poly.coeffs[index]),
            MultilinearPolynomial::U32Scalars(poly) => F::from_u32(poly.coeffs[index]),
            MultilinearPolynomial::U64Scalars(poly) => F::from_u64(poly.coeffs[index]),
        }
    }

    #[inline]
    pub fn get_bound_coeff(&self, index: usize) -> F {
        match self {
            MultilinearPolynomial::LargeScalars(poly) => poly[index],
            MultilinearPolynomial::U8Scalars(poly) => {
                if poly.is_bound() {
                    poly.bound_coeffs[index]
                } else {
                    F::from_u8(poly.coeffs[index])
                }
            }
            MultilinearPolynomial::U16Scalars(poly) => {
                if poly.is_bound() {
                    poly.bound_coeffs[index]
                } else {
                    F::from_u16(poly.coeffs[index])
                }
            }
            MultilinearPolynomial::U32Scalars(poly) => {
                if poly.is_bound() {
                    poly.bound_coeffs[index]
                } else {
                    F::from_u32(poly.coeffs[index])
                }
            }
            MultilinearPolynomial::U64Scalars(poly) => {
                if poly.is_bound() {
                    poly.bound_coeffs[index]
                } else {
                    F::from_u64(poly.coeffs[index])
                }
            }
        }
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
    fn is_bound(&self) -> bool;
    fn bind(&mut self, r: F, order: BindingOrder);
    fn bind_parallel(&mut self, r: F, order: BindingOrder);
    fn final_sumcheck_claim(&self) -> F;
}

pub trait PolynomialEvaluation<F: JoltField> {
    fn evaluate(&self, r: &[F]) -> F;
    fn evaluate_with_chis(&self, chis: &[F]) -> F;
    fn sumcheck_evals(&self, index: usize, degree: usize, order: BindingOrder) -> Vec<F>;
}

impl<F: JoltField> PolynomialBinding<F> for MultilinearPolynomial<F> {
    fn is_bound(&self) -> bool {
        match self {
            MultilinearPolynomial::LargeScalars(poly) => poly.is_bound(),
            MultilinearPolynomial::U8Scalars(poly) => poly.is_bound(),
            MultilinearPolynomial::U16Scalars(poly) => poly.is_bound(),
            MultilinearPolynomial::U32Scalars(poly) => poly.is_bound(),
            MultilinearPolynomial::U64Scalars(poly) => poly.is_bound(),
        }
    }

    #[tracing::instrument(skip_all, name = "MultilinearPolynomial::bind")]
    fn bind(&mut self, r: F, order: BindingOrder) {
        match self {
            MultilinearPolynomial::LargeScalars(poly) => match order {
                BindingOrder::LowToHigh => poly.bound_poly_var_bot(&r),
                BindingOrder::HighToLow => poly.bound_poly_var_top(&r),
            },
            MultilinearPolynomial::U8Scalars(poly) => poly.bind(r, order),
            MultilinearPolynomial::U16Scalars(poly) => poly.bind(r, order),
            MultilinearPolynomial::U32Scalars(poly) => poly.bind(r, order),
            MultilinearPolynomial::U64Scalars(poly) => poly.bind(r, order),
        }
    }

    #[tracing::instrument(skip_all, name = "MultilinearPolynomial::bind_parallel")]
    fn bind_parallel(&mut self, r: F, order: BindingOrder) {
        todo!()
    }

    fn final_sumcheck_claim(&self) -> F {
        match self {
            MultilinearPolynomial::LargeScalars(poly) => {
                assert_eq!(poly.len(), 1);
                poly.Z[0]
            }
            MultilinearPolynomial::U8Scalars(poly) => poly.final_sumcheck_claim(),
            MultilinearPolynomial::U16Scalars(poly) => poly.final_sumcheck_claim(),
            MultilinearPolynomial::U32Scalars(poly) => poly.final_sumcheck_claim(),
            MultilinearPolynomial::U64Scalars(poly) => poly.final_sumcheck_claim(),
        }
    }
}

impl<F: JoltField> PolynomialEvaluation<F> for MultilinearPolynomial<F> {
    fn evaluate(&self, r: &[F]) -> F {
        match self {
            MultilinearPolynomial::LargeScalars(poly) => poly.evaluate(r),
            MultilinearPolynomial::U8Scalars(poly) => {
                let chis = EqPolynomial::evals_with_r2(&r);
                assert_eq!(chis.len(), poly.coeffs.len());
                chis.par_iter()
                    .zip(poly.coeffs.par_iter())
                    .map(|(a_i, b_i)| a_i.mul_u64_unchecked(*b_i as u64))
                    .sum()
            }
            MultilinearPolynomial::U16Scalars(poly) => {
                let chis = EqPolynomial::evals_with_r2(&r);
                assert_eq!(chis.len(), poly.coeffs.len());
                chis.par_iter()
                    .zip(poly.coeffs.par_iter())
                    .map(|(a_i, b_i)| a_i.mul_u64_unchecked(*b_i as u64))
                    .sum()
            }
            MultilinearPolynomial::U32Scalars(poly) => {
                let chis = EqPolynomial::evals_with_r2(&r);
                assert_eq!(chis.len(), poly.coeffs.len());
                chis.par_iter()
                    .zip(poly.coeffs.par_iter())
                    .map(|(a_i, b_i)| a_i.mul_u64_unchecked(*b_i as u64))
                    .sum()
            }
            MultilinearPolynomial::U64Scalars(poly) => {
                let chis = EqPolynomial::evals_with_r2(&r);
                assert_eq!(chis.len(), poly.coeffs.len());
                chis.par_iter()
                    .zip(poly.coeffs.par_iter())
                    .map(|(a_i, b_i)| a_i.mul_u64_unchecked(*b_i))
                    .sum()
            }
        }
    }

    // TODO(moodlezoup): This is suboptimal for CompactPolynomials because get_coeff
    // requires a field multiplication (to convert the coefficient into Montgomery
    // form). Avoid this as much as possible.
    #[tracing::instrument(skip_all, name = "MultilinearPolynomial::evaluate_with_chis")]
    fn evaluate_with_chis(&self, chis: &[F]) -> F {
        // assert_eq!(chis.len(), self.len());
        chis.par_iter()
            .enumerate()
            .map(|(i, x)| *x * self.get_coeff(i))
            .sum()
    }

    // TODO(moodlezoup): tinyvec?
    // TODO(moodlezoup): Return Vec of length degree
    #[inline]
    fn sumcheck_evals(&self, index: usize, degree: usize, order: BindingOrder) -> Vec<F> {
        debug_assert!(degree > 0);
        debug_assert!(index < self.len() / 2);
        match order {
            BindingOrder::HighToLow => {
                let mut evals = vec![F::zero(); degree + 1];
                evals[0] = self.get_bound_coeff(index);
                evals[1] = self.get_bound_coeff(index + self.len() / 2);
                let m = evals[1] - evals[0];
                for i in 1..degree {
                    evals[i + 1] = evals[i] + m;
                }
                evals
            }
            BindingOrder::LowToHigh => {
                let mut evals = vec![F::zero(); degree + 1];
                evals[0] = self.get_bound_coeff(2 * index);
                evals[1] = self.get_bound_coeff(2 * index + 1);
                let m = evals[1] - evals[0];
                for i in 1..degree {
                    evals[i + 1] = evals[i] + m;
                }
                evals
            }
        }
    }
}
