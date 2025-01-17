use crate::utils::math::Math;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use rayon::prelude::*;
use strum_macros::EnumIter;

use super::{
    compact_polynomial::{CompactPolynomial, SmallScalar},
    dense_mlpoly::DensePolynomial,
    eq_poly::EqPolynomial,
};
use crate::{
    field::{JoltField, OptimizedMul},
    utils::thread::unsafe_allocate_zero_vec,
};

/// Wrapper enum for the various multilinear polynomial types used in Jolt
#[repr(u8)]
#[derive(Clone, Debug, EnumIter, PartialEq)]
pub enum MultilinearPolynomial<F: JoltField> {
    LargeScalars(DensePolynomial<F>),
    U8Scalars(CompactPolynomial<u8, F>),
    U16Scalars(CompactPolynomial<u16, F>),
    U32Scalars(CompactPolynomial<u32, F>),
    U64Scalars(CompactPolynomial<u64, F>),
    I64Scalars(CompactPolynomial<i64, F>),
}

/// The order in which polynomial variables are bound in sumcheck
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
    /// The length of the polynomial before it was bound
    pub fn original_len(&self) -> usize {
        match self {
            MultilinearPolynomial::LargeScalars(poly) => poly.Z.len(),
            MultilinearPolynomial::U8Scalars(poly) => poly.coeffs.len(),
            MultilinearPolynomial::U16Scalars(poly) => poly.coeffs.len(),
            MultilinearPolynomial::U32Scalars(poly) => poly.coeffs.len(),
            MultilinearPolynomial::U64Scalars(poly) => poly.coeffs.len(),
            MultilinearPolynomial::I64Scalars(poly) => poly.coeffs.len(),
        }
    }

    /// The current length of the polynomial
    pub fn len(&self) -> usize {
        match self {
            MultilinearPolynomial::LargeScalars(poly) => poly.len(),
            MultilinearPolynomial::U8Scalars(poly) => poly.len(),
            MultilinearPolynomial::U16Scalars(poly) => poly.len(),
            MultilinearPolynomial::U32Scalars(poly) => poly.len(),
            MultilinearPolynomial::U64Scalars(poly) => poly.len(),
            MultilinearPolynomial::I64Scalars(poly) => poly.len(),
        }
    }

    pub fn get_num_vars(&self) -> usize {
        match self {
            MultilinearPolynomial::LargeScalars(poly) => poly.get_num_vars(),
            MultilinearPolynomial::U8Scalars(poly) => poly.get_num_vars(),
            MultilinearPolynomial::U16Scalars(poly) => poly.get_num_vars(),
            MultilinearPolynomial::U32Scalars(poly) => poly.get_num_vars(),
            MultilinearPolynomial::U64Scalars(poly) => poly.get_num_vars(),
            MultilinearPolynomial::I64Scalars(poly) => poly.get_num_vars(),
        }
    }

    /// The maximum number of bits occupied by one of the polynomial's coefficients.
    #[tracing::instrument(skip_all)]
    pub fn max_num_bits(&self) -> u32 {
        match self {
            MultilinearPolynomial::LargeScalars(poly) => poly
                .evals_ref()
                .par_iter()
                .map(|s| s.num_bits())
                .max()
                .unwrap(),
            MultilinearPolynomial::U8Scalars(poly) => {
                (*poly.coeffs.iter().max().unwrap() as usize).num_bits() as u32
            }
            MultilinearPolynomial::U16Scalars(poly) => {
                (*poly.coeffs.iter().max().unwrap() as usize).num_bits() as u32
            }
            MultilinearPolynomial::U32Scalars(poly) => {
                (*poly.coeffs.iter().max().unwrap() as usize).num_bits() as u32
            }
            MultilinearPolynomial::U64Scalars(poly) => {
                (*poly.coeffs.iter().max().unwrap() as usize).num_bits() as u32
            }
            MultilinearPolynomial::I64Scalars(_) => {
                // HACK(moodlezoup): i64 coefficients are converted into full-width field
                // elements before computing the MSM
                F::NUM_BYTES as u32 * 8
            }
        }
    }

    pub fn linear_combination(polynomials: &[&Self], coefficients: &[F]) -> Self {
        debug_assert_eq!(polynomials.len(), coefficients.len());

        let max_length = polynomials
            .iter()
            .map(|poly| poly.original_len())
            .max()
            .unwrap();
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
                    let poly_len = poly.original_len();
                    if index >= poly_len {
                        continue;
                    }

                    match poly {
                        MultilinearPolynomial::LargeScalars(poly) => {
                            debug_assert!(!poly.is_bound());
                            let poly_evals = &poly.evals_ref()[index..];
                            for (rlc, poly_eval) in chunk.iter_mut().zip(poly_evals.iter()) {
                                *rlc += poly_eval.mul_01_optimized(*coeff);
                            }
                        }
                        MultilinearPolynomial::U8Scalars(poly) => {
                            let poly_evals = &poly.coeffs[index..];
                            let coeff_r2 = F::montgomery_r2().unwrap_or(F::one()) * coeff;
                            for (rlc, poly_eval) in chunk.iter_mut().zip(poly_evals.iter()) {
                                *rlc += poly_eval.field_mul(coeff_r2);
                            }
                        }
                        MultilinearPolynomial::U16Scalars(poly) => {
                            let poly_evals = &poly.coeffs[index..];
                            let coeff_r2 = F::montgomery_r2().unwrap_or(F::one()) * coeff;
                            for (rlc, poly_eval) in chunk.iter_mut().zip(poly_evals.iter()) {
                                *rlc += poly_eval.field_mul(coeff_r2);
                            }
                        }
                        MultilinearPolynomial::U32Scalars(poly) => {
                            let poly_evals = &poly.coeffs[index..];
                            let coeff_r2 = F::montgomery_r2().unwrap_or(F::one()) * coeff;
                            for (rlc, poly_eval) in chunk.iter_mut().zip(poly_evals.iter()) {
                                *rlc += poly_eval.field_mul(coeff_r2);
                            }
                        }
                        MultilinearPolynomial::U64Scalars(poly) => {
                            let poly_evals = &poly.coeffs[index..];
                            let coeff_r2 = F::montgomery_r2().unwrap_or(F::one()) * coeff;
                            for (rlc, poly_eval) in chunk.iter_mut().zip(poly_evals.iter()) {
                                *rlc += poly_eval.field_mul(coeff_r2);
                            }
                        }
                        MultilinearPolynomial::I64Scalars(poly) => {
                            let poly_evals = &poly.coeffs[index..];
                            let coeff_r2 = F::montgomery_r2().unwrap_or(F::one()) * coeff;
                            for (rlc, poly_eval) in chunk.iter_mut().zip(poly_evals.iter()) {
                                *rlc += poly_eval.field_mul(coeff_r2);
                            }
                        }
                    }
                }
                chunk
            })
            .collect();

        MultilinearPolynomial::from(lc_coeffs)
    }

    /// Gets the polynomial coefficient at the given `index`
    pub fn get_coeff(&self, index: usize) -> F {
        match self {
            MultilinearPolynomial::LargeScalars(poly) => poly[index],
            MultilinearPolynomial::U8Scalars(poly) => F::from_u8(poly.coeffs[index]),
            MultilinearPolynomial::U16Scalars(poly) => F::from_u16(poly.coeffs[index]),
            MultilinearPolynomial::U32Scalars(poly) => F::from_u32(poly.coeffs[index]),
            MultilinearPolynomial::U64Scalars(poly) => F::from_u64(poly.coeffs[index]),
            MultilinearPolynomial::I64Scalars(poly) => F::from_i64(poly.coeffs[index]),
        }
    }

    /// Gets the polynomial coefficient at the given `index`, as an `i64`.
    /// Panics if the polynomial is a large-scalar polynomial.
    pub fn get_coeff_i64(&self, index: usize) -> i64 {
        match self {
            MultilinearPolynomial::LargeScalars(_) => {
                panic!("Unexpected large-scalar polynomial")
            }
            MultilinearPolynomial::U8Scalars(poly) => i64::from(poly.coeffs[index]),
            MultilinearPolynomial::U16Scalars(poly) => i64::from(poly.coeffs[index]),
            MultilinearPolynomial::U32Scalars(poly) => i64::from(poly.coeffs[index]),
            MultilinearPolynomial::U64Scalars(poly) => i64::try_from(poly.coeffs[index]).unwrap(),
            MultilinearPolynomial::I64Scalars(poly) => poly.coeffs[index],
        }
    }

    /// Gets the polynomial coefficient at the given `index`, as an `i128`.
    /// Panics if the polynomial is a large-scalar polynomial.
    pub fn get_coeff_i128(&self, index: usize) -> i128 {
        match self {
            MultilinearPolynomial::LargeScalars(_) => {
                panic!("Unexpected large-scalar polynomial")
            }
            MultilinearPolynomial::U8Scalars(poly) => i128::from(poly.coeffs[index]),
            MultilinearPolynomial::U16Scalars(poly) => i128::from(poly.coeffs[index]),
            MultilinearPolynomial::U32Scalars(poly) => i128::from(poly.coeffs[index]),
            MultilinearPolynomial::U64Scalars(poly) => i128::from(poly.coeffs[index]),
            MultilinearPolynomial::I64Scalars(poly) => i128::from(poly.coeffs[index]),
        }
    }

    /// Gets the polynomial coefficient at the given `index`. The polynomial may have
    /// been bound over the course of sumcheck.
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
            MultilinearPolynomial::I64Scalars(poly) => {
                if poly.is_bound() {
                    poly.bound_coeffs[index]
                } else {
                    F::from_i64(poly.coeffs[index])
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

impl<F: JoltField> From<Vec<i64>> for MultilinearPolynomial<F> {
    fn from(coeffs: Vec<i64>) -> Self {
        let poly = CompactPolynomial::from_coeffs(coeffs);
        Self::I64Scalars(poly)
    }
}

impl<'a, F: JoltField> TryFrom<&'a MultilinearPolynomial<F>> for &'a DensePolynomial<F> {
    type Error = (); // TODO(moodlezoup)

    fn try_from(poly: &'a MultilinearPolynomial<F>) -> Result<Self, Self::Error> {
        match poly {
            MultilinearPolynomial::LargeScalars(poly) => Ok(poly),
            _ => Err(()),
        }
    }
}

impl<'a, F: JoltField> TryFrom<&'a MultilinearPolynomial<F>> for &'a CompactPolynomial<u8, F> {
    type Error = (); // TODO(moodlezoup)

    fn try_from(poly: &'a MultilinearPolynomial<F>) -> Result<Self, Self::Error> {
        match poly {
            MultilinearPolynomial::U8Scalars(poly) => Ok(poly),
            _ => Err(()),
        }
    }
}

impl<'a, F: JoltField> TryFrom<&'a MultilinearPolynomial<F>> for &'a CompactPolynomial<u16, F> {
    type Error = (); // TODO(moodlezoup)

    fn try_from(poly: &'a MultilinearPolynomial<F>) -> Result<Self, Self::Error> {
        match poly {
            MultilinearPolynomial::U16Scalars(poly) => Ok(poly),
            _ => Err(()),
        }
    }
}

impl<'a, F: JoltField> TryFrom<&'a MultilinearPolynomial<F>> for &'a CompactPolynomial<u32, F> {
    type Error = (); // TODO(moodlezoup)

    fn try_from(poly: &'a MultilinearPolynomial<F>) -> Result<Self, Self::Error> {
        match poly {
            MultilinearPolynomial::U32Scalars(poly) => Ok(poly),
            _ => Err(()),
        }
    }
}

impl<'a, F: JoltField> TryFrom<&'a MultilinearPolynomial<F>> for &'a CompactPolynomial<u64, F> {
    type Error = (); // TODO(moodlezoup)

    fn try_from(poly: &'a MultilinearPolynomial<F>) -> Result<Self, Self::Error> {
        match poly {
            MultilinearPolynomial::U64Scalars(poly) => Ok(poly),
            _ => Err(()),
        }
    }
}

impl<'a, F: JoltField> TryFrom<&'a MultilinearPolynomial<F>> for &'a CompactPolynomial<i64, F> {
    type Error = (); // TODO(moodlezoup)

    fn try_from(poly: &'a MultilinearPolynomial<F>) -> Result<Self, Self::Error> {
        match poly {
            MultilinearPolynomial::I64Scalars(poly) => Ok(poly),
            _ => Err(()),
        }
    }
}

impl<F: JoltField> CanonicalSerialize for MultilinearPolynomial<F> {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        _writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        unimplemented!("Unused; needed to satisfy trait bounds for StructuredPolynomialData")
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        unimplemented!("Unused; needed to satisfy trait bounds for StructuredPolynomialData")
    }
}

impl<F: JoltField> CanonicalDeserialize for MultilinearPolynomial<F> {
    fn deserialize_with_mode<R: std::io::Read>(
        _reader: R,
        _compress: Compress,
        _validate: Validate,
    ) -> Result<Self, SerializationError> {
        unimplemented!("Unused; needed to satisfy trait bounds for StructuredPolynomialData")
    }
}

impl<F: JoltField> Valid for MultilinearPolynomial<F> {
    fn check(&self) -> Result<(), SerializationError> {
        unimplemented!("Unused; needed to satisfy trait bounds for StructuredPolynomialData")
    }
}

pub trait PolynomialBinding<F: JoltField> {
    /// Returns whether or not the polynomial has been bound (in a sumcheck)
    fn is_bound(&self) -> bool;
    /// Binds the polynomial to a random field element `r`.
    fn bind(&mut self, r: F, order: BindingOrder);
    /// Returns the final sumcheck claim about the polynomial.
    fn final_sumcheck_claim(&self) -> F;
}

pub trait PolynomialEvaluation<F: JoltField> {
    /// Returns the final sumcheck claim about the polynomial.
    fn evaluate(&self, r: &[F]) -> F;
    /// Evaluates a batch of polynomials on the same point `r`.
    /// Returns: (evals, EQ table)
    /// where EQ table is EQ(x, r) for x \in {0, 1}^|r|. This is used for
    /// batched opening proofs (see opening_proof.rs)
    fn batch_evaluate(polys: &[&Self], r: &[F]) -> (Vec<F>, Vec<F>);
    /// Computes this polynomial's contribution to the computation of a prover
    /// sumcheck message (i.e. a univariate polynomial of the given `degree`).
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
            MultilinearPolynomial::I64Scalars(poly) => poly.is_bound(),
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
            MultilinearPolynomial::I64Scalars(poly) => poly.bind(r, order),
        }
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
            MultilinearPolynomial::I64Scalars(poly) => poly.final_sumcheck_claim(),
        }
    }
}

impl<F: JoltField> PolynomialEvaluation<F> for MultilinearPolynomial<F> {
    #[tracing::instrument(skip_all, name = "MultilinearPolynomial::evaluate")]
    fn evaluate(&self, r: &[F]) -> F {
        match self {
            MultilinearPolynomial::LargeScalars(poly) => poly.evaluate(r),
            MultilinearPolynomial::U8Scalars(poly) => {
                let chis = EqPolynomial::evals_with_r2(r);
                assert_eq!(chis.len(), poly.coeffs.len());
                chis.par_iter()
                    .zip(poly.coeffs.par_iter())
                    .map(|(a_i, b_i)| b_i.field_mul(*a_i))
                    .sum()
            }
            MultilinearPolynomial::U16Scalars(poly) => {
                let chis = EqPolynomial::evals_with_r2(r);
                assert_eq!(chis.len(), poly.coeffs.len());
                chis.par_iter()
                    .zip(poly.coeffs.par_iter())
                    .map(|(a_i, b_i)| b_i.field_mul(*a_i))
                    .sum()
            }
            MultilinearPolynomial::U32Scalars(poly) => {
                let chis = EqPolynomial::evals_with_r2(r);
                assert_eq!(chis.len(), poly.coeffs.len());
                chis.par_iter()
                    .zip(poly.coeffs.par_iter())
                    .map(|(a_i, b_i)| b_i.field_mul(*a_i))
                    .sum()
            }
            MultilinearPolynomial::U64Scalars(poly) => {
                let chis = EqPolynomial::evals_with_r2(r);
                assert_eq!(chis.len(), poly.coeffs.len());
                chis.par_iter()
                    .zip(poly.coeffs.par_iter())
                    .map(|(a_i, b_i)| b_i.field_mul(*a_i))
                    .sum()
            }
            MultilinearPolynomial::I64Scalars(poly) => {
                let chis = EqPolynomial::evals_with_r2(r);
                assert_eq!(chis.len(), poly.coeffs.len());
                chis.par_iter()
                    .zip(poly.coeffs.par_iter())
                    .map(|(a_i, b_i)| b_i.field_mul(*a_i))
                    .sum()
            }
        }
    }

    #[tracing::instrument(skip_all, name = "MultilinearPolynomial::batch_evaluate")]
    fn batch_evaluate(polys: &[&Self], r: &[F]) -> (Vec<F>, Vec<F>) {
        let eq = EqPolynomial::evals(r);

        if polys
            .iter()
            .any(|poly| !matches!(poly, MultilinearPolynomial::LargeScalars(_)))
        {
            // If any of the polynomials contain non-Montgomery form coefficients,
            // we need to compute the R^2-adjusted EQ table.
            let eq_r2 = EqPolynomial::evals_with_r2(r);
            let evals: Vec<F> = polys
                .into_par_iter()
                .map(|&poly| match poly {
                    MultilinearPolynomial::LargeScalars(poly) => {
                        poly.evaluate_at_chi_low_optimized(&eq)
                    }
                    MultilinearPolynomial::U8Scalars(poly) => eq_r2
                        .par_iter()
                        .zip(poly.coeffs.par_iter())
                        .map(|(chi, coeff)| coeff.field_mul(*chi))
                        .sum(),
                    MultilinearPolynomial::U16Scalars(poly) => eq_r2
                        .par_iter()
                        .zip(poly.coeffs.par_iter())
                        .map(|(chi, coeff)| coeff.field_mul(*chi))
                        .sum(),
                    MultilinearPolynomial::U32Scalars(poly) => eq_r2
                        .par_iter()
                        .zip(poly.coeffs.par_iter())
                        .map(|(chi, coeff)| coeff.field_mul(*chi))
                        .sum(),
                    MultilinearPolynomial::U64Scalars(poly) => eq_r2
                        .par_iter()
                        .zip(poly.coeffs.par_iter())
                        .map(|(chi, coeff)| coeff.field_mul(*chi))
                        .sum(),
                    MultilinearPolynomial::I64Scalars(poly) => eq_r2
                        .par_iter()
                        .zip(poly.coeffs.par_iter())
                        .map(|(chi, coeff)| coeff.field_mul(*chi))
                        .sum(),
                })
                .collect();
            (evals, eq)
        } else {
            let evals: Vec<F> = polys
                .into_par_iter()
                .map(|&poly| {
                    let poly: &DensePolynomial<F> = poly.try_into().unwrap();
                    poly.evaluate_at_chi_low_optimized(&eq)
                })
                .collect();
            (evals, eq)
        }
    }

    #[inline]
    fn sumcheck_evals(&self, index: usize, degree: usize, order: BindingOrder) -> Vec<F> {
        debug_assert!(degree > 0);
        debug_assert!(index < self.len() / 2);
        match order {
            BindingOrder::HighToLow => {
                let mut evals = vec![F::zero(); degree];
                evals[0] = self.get_bound_coeff(index);
                if degree == 1 {
                    return evals;
                }
                let mut eval = self.get_bound_coeff(index + self.len() / 2);
                let m = eval - evals[0];
                for i in 1..degree {
                    eval += m;
                    evals[i] = eval;
                }
                evals
            }
            BindingOrder::LowToHigh => {
                let mut evals = vec![F::zero(); degree];
                evals[0] = self.get_bound_coeff(2 * index);
                if degree == 1 {
                    return evals;
                }
                let mut eval = self.get_bound_coeff(2 * index + 1);
                let m = eval - evals[0];
                for i in 1..degree {
                    eval += m;
                    evals[i] = eval;
                }
                evals
            }
        }
    }
}
