use crate::{
    poly::{one_hot_polynomial::OneHotPolynomial, rlc_polynomial::RLCPolynomial},
    utils::compute_dotproduct,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Valid};
use num_traits::MulAdd;
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
    RLC(RLCPolynomial<F>),
    OneHot(OneHotPolynomial<F>),
}

impl<F: JoltField> Valid for MultilinearPolynomial<F> {
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        unimplemented!("Only here to satisfy trait bounds")
    }
}

impl<F: JoltField> CanonicalDeserialize for MultilinearPolynomial<F> {
    fn deserialize_with_mode<R: std::io::Read>(
        _reader: R,
        _compress: ark_serialize::Compress,
        _validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        unimplemented!("Only here to satisfy trait bounds")
    }
}

impl<F: JoltField> CanonicalSerialize for MultilinearPolynomial<F> {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        _writer: W,
        _compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        unimplemented!("Only here to satisfy trait bounds")
    }

    fn serialized_size(&self, _compress: ark_serialize::Compress) -> usize {
        unimplemented!("Only here to satisfy trait bounds")
    }
}

/// The order in which polynomial variables are bound in sumcheck
#[derive(Clone, Copy, Debug, PartialEq)]
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
            _ => unimplemented!("Unexpected MultilinearPolynomial variant"),
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
            _ => unimplemented!("Unexpected MultilinearPolynomial variant"),
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
            MultilinearPolynomial::OneHot(poly) => poly.get_num_vars(),
            _ => unimplemented!("Unexpected MultilinearPolynomial variant"),
        }
    }

    #[tracing::instrument(skip_all)]
    pub fn linear_combination(polynomials: &[&Self], coefficients: &[F]) -> Self {
        debug_assert_eq!(polynomials.len(), coefficients.len());

        // If there's at least one sparse polynomial in `polynomials`, the linear
        // combination will be represented by an `RLCPolynomial`. Otherwise, it will
        // be represented by a `DensePolynomial`.
        if polynomials
            .iter()
            .any(|poly| matches!(poly, MultilinearPolynomial::OneHot(_)))
        {
            let mut result = RLCPolynomial::<F>::new();
            for (coeff, polynomial) in coefficients.iter().zip(polynomials.iter()) {
                result = match polynomial {
                    MultilinearPolynomial::LargeScalars(poly) => poly.mul_add(*coeff, result),
                    MultilinearPolynomial::U8Scalars(poly) => poly.mul_add(*coeff, result),
                    MultilinearPolynomial::U16Scalars(poly) => poly.mul_add(*coeff, result),
                    MultilinearPolynomial::U32Scalars(poly) => poly.mul_add(*coeff, result),
                    MultilinearPolynomial::U64Scalars(poly) => poly.mul_add(*coeff, result),
                    MultilinearPolynomial::I64Scalars(poly) => poly.mul_add(*coeff, result),
                    MultilinearPolynomial::OneHot(poly) => poly.mul_add(*coeff, result),
                    _ => unimplemented!("Unexpected polynomial type"),
                };
            }
            return MultilinearPolynomial::RLC(result);
        }

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
                            for (rlc, poly_eval) in chunk.iter_mut().zip(poly_evals.iter()) {
                                *rlc += poly_eval.field_mul(*coeff);
                            }
                        }
                        MultilinearPolynomial::U16Scalars(poly) => {
                            let poly_evals = &poly.coeffs[index..];
                            for (rlc, poly_eval) in chunk.iter_mut().zip(poly_evals.iter()) {
                                *rlc += poly_eval.field_mul(*coeff);
                            }
                        }
                        MultilinearPolynomial::U32Scalars(poly) => {
                            let poly_evals = &poly.coeffs[index..];
                            for (rlc, poly_eval) in chunk.iter_mut().zip(poly_evals.iter()) {
                                *rlc += poly_eval.field_mul(*coeff);
                            }
                        }
                        MultilinearPolynomial::U64Scalars(poly) => {
                            let poly_evals = &poly.coeffs[index..];
                            for (rlc, poly_eval) in chunk.iter_mut().zip(poly_evals.iter()) {
                                *rlc += poly_eval.field_mul(*coeff);
                            }
                        }
                        MultilinearPolynomial::I64Scalars(poly) => {
                            let poly_evals = &poly.coeffs[index..];
                            for (rlc, poly_eval) in chunk.iter_mut().zip(poly_evals.iter()) {
                                *rlc += poly_eval.field_mul(*coeff);
                            }
                        }
                        _ => unimplemented!("Unexpected MultilinearPolynomial variant"),
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
            _ => unimplemented!("Unexpected MultilinearPolynomial variant"),
        }
    }

    /// Gets the polynomial coefficient at the given `index`, as an `i64`.
    /// Panics if the polynomial is a large-scalar polynomial.
    pub fn get_coeff_i64(&self, index: usize) -> i64 {
        match self {
            MultilinearPolynomial::U8Scalars(poly) => i64::from(poly.coeffs[index]),
            MultilinearPolynomial::U16Scalars(poly) => i64::from(poly.coeffs[index]),
            MultilinearPolynomial::U32Scalars(poly) => i64::from(poly.coeffs[index]),
            MultilinearPolynomial::U64Scalars(poly) => i64::try_from(poly.coeffs[index]).unwrap(),
            MultilinearPolynomial::I64Scalars(poly) => poly.coeffs[index],
            _ => unimplemented!("Unexpected MultilinearPolynomial variant"),
        }
    }

    /// Gets the polynomial coefficient at the given `index`, as an `i128`.
    /// Panics if the polynomial is a large-scalar polynomial.
    pub fn get_coeff_i128(&self, index: usize) -> i128 {
        match self {
            MultilinearPolynomial::U8Scalars(poly) => i128::from(poly.coeffs[index]),
            MultilinearPolynomial::U16Scalars(poly) => i128::from(poly.coeffs[index]),
            MultilinearPolynomial::U32Scalars(poly) => i128::from(poly.coeffs[index]),
            MultilinearPolynomial::U64Scalars(poly) => i128::from(poly.coeffs[index]),
            MultilinearPolynomial::I64Scalars(poly) => i128::from(poly.coeffs[index]),
            _ => unimplemented!("Unexpected MultilinearPolynomial variant"),
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
            _ => unimplemented!("Unexpected MultilinearPolynomial variant"),
        }
    }

    // This is the old polynomial evaluation code that uses
    // the dot product with langrange bases as the algorithm
    // This might be eventually removed from the code base
    pub fn evaluate_dot_product(&self, r: &[F]) -> F {
        match self {
            MultilinearPolynomial::LargeScalars(poly) => poly.evaluate(r),
            MultilinearPolynomial::RLC(_) => {
                unimplemented!("Unexpected RLC polynomial")
            }
            _ => {
                let chis = EqPolynomial::evals(r);
                self.dot_product(&chis)
            }
        }
    }

    /// Computes the dot product of the polynomial's coefficients and a vector
    /// of field elements.
    pub fn dot_product(&self, other: &[F]) -> F {
        match self {
            MultilinearPolynomial::LargeScalars(poly) => compute_dotproduct(&poly.Z, other),
            MultilinearPolynomial::U8Scalars(poly) => poly
                .coeffs
                .par_iter()
                .zip_eq(other.par_iter())
                .map(|(a, b)| a.field_mul(*b))
                .sum(),
            MultilinearPolynomial::U16Scalars(poly) => poly
                .coeffs
                .par_iter()
                .zip_eq(other.par_iter())
                .map(|(a, b)| a.field_mul(*b))
                .sum(),
            MultilinearPolynomial::U32Scalars(poly) => poly
                .coeffs
                .par_iter()
                .zip_eq(other.par_iter())
                .map(|(a, b)| a.field_mul(*b))
                .sum(),
            MultilinearPolynomial::U64Scalars(poly) => poly
                .coeffs
                .par_iter()
                .zip_eq(other.par_iter())
                .map(|(a, b)| a.field_mul(*b))
                .sum(),
            MultilinearPolynomial::I64Scalars(poly) => poly
                .coeffs
                .par_iter()
                .zip_eq(other.par_iter())
                .map(|(a, b)| a.field_mul(*b))
                .sum(),
            _ => unimplemented!("Unexpected MultilinearPolynomial variant"),
        }
    }

    #[inline]
    pub fn sumcheck_evals_array<const DEGREE: usize>(
        &self,
        index: usize,
        order: BindingOrder,
    ) -> [F; DEGREE] {
        debug_assert!(DEGREE > 0);
        debug_assert!(index < self.len() / 2);

        let mut evals = [F::zero(); DEGREE];
        match order {
            BindingOrder::HighToLow => {
                evals[0] = self.get_bound_coeff(index);
                if DEGREE == 1 {
                    return evals;
                }
                let mut eval = self.get_bound_coeff(index + self.len() / 2);
                let m = eval - evals[0];
                for i in 1..DEGREE {
                    eval += m;
                    evals[i] = eval;
                }
            }
            BindingOrder::LowToHigh => {
                evals[0] = self.get_bound_coeff(2 * index);
                if DEGREE == 1 {
                    return evals;
                }
                let mut eval = self.get_bound_coeff(2 * index + 1);
                let m = eval - evals[0];
                for i in 1..DEGREE {
                    eval += m;
                    evals[i] = eval;
                }
            }
        };
        evals
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

pub trait PolynomialBinding<F: JoltField> {
    /// Returns whether or not the polynomial has been bound (in a sumcheck)
    fn is_bound(&self) -> bool;
    /// Binds the polynomial to a random field element `r`.
    fn bind(&mut self, r: F, order: BindingOrder);
    /// Binds the polynomial to a random field element `r`, parallelizing
    /// by coefficient.
    fn bind_parallel(&mut self, r: F, order: BindingOrder);
    /// Returns the final sumcheck claim about the polynomial.
    fn final_sumcheck_claim(&self) -> F;
}

pub trait PolynomialEvaluation<F: JoltField> {
    /// Returns the final sumcheck claim about the polynomial.
    /// This uses the algorithm in Lemma 4.3 in Thaler, Proofs and
    /// Arguments -- the inside out processing
    fn evaluate(&self, r: &[F]) -> F;

    /// Evaluates a batch of polynomials on the same point `r`.
    /// Returns: (evals, EQ table)
    /// where EQ table is EQ(x, r) for x \in {0, 1}^|r|. This is used for
    /// batched opening proofs (see opening_proof.rs)
    fn batch_evaluate(polys: &[&Self], r: &[F]) -> Vec<F>
    where
        Self: Sized;
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
            _ => unimplemented!("Unexpected MultilinearPolynomial variant"),
        }
    }

    #[tracing::instrument(skip_all, name = "MultilinearPolynomial::bind")]
    fn bind(&mut self, r: F, order: BindingOrder) {
        match self {
            MultilinearPolynomial::LargeScalars(poly) => poly.bind(r, order),
            MultilinearPolynomial::U8Scalars(poly) => poly.bind(r, order),
            MultilinearPolynomial::U16Scalars(poly) => poly.bind(r, order),
            MultilinearPolynomial::U32Scalars(poly) => poly.bind(r, order),
            MultilinearPolynomial::U64Scalars(poly) => poly.bind(r, order),
            MultilinearPolynomial::I64Scalars(poly) => poly.bind(r, order),
            _ => unimplemented!("Unexpected MultilinearPolynomial variant"),
        }
    }

    #[tracing::instrument(skip_all, name = "MultilinearPolynomial::bind_parallel")]
    fn bind_parallel(&mut self, r: F, order: BindingOrder) {
        match self {
            MultilinearPolynomial::LargeScalars(poly) => poly.bind_parallel(r, order),
            MultilinearPolynomial::U8Scalars(poly) => poly.bind_parallel(r, order),
            MultilinearPolynomial::U16Scalars(poly) => poly.bind_parallel(r, order),
            MultilinearPolynomial::U32Scalars(poly) => poly.bind_parallel(r, order),
            MultilinearPolynomial::U64Scalars(poly) => poly.bind_parallel(r, order),
            MultilinearPolynomial::I64Scalars(poly) => poly.bind_parallel(r, order),
            _ => unimplemented!("Unexpected MultilinearPolynomial variant"),
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
            _ => unimplemented!("Unexpected MultilinearPolynomial variant"),
        }
    }
}

impl<F: JoltField> PolynomialEvaluation<F> for MultilinearPolynomial<F> {
    #[tracing::instrument(skip_all, name = "MultilinearPolynomial::evaluate")]
    fn evaluate(&self, r: &[F]) -> F {
        match self {
            MultilinearPolynomial::LargeScalars(poly) => {
                let m = r.len() / 2;
                let (r2, r1) = r.split_at(m);
                let (eq_one, eq_two) =
                    rayon::join(|| EqPolynomial::evals(r2), || EqPolynomial::evals(r1));

                poly.split_eq_evaluate(r, &eq_one, &eq_two)
            }
            MultilinearPolynomial::U8Scalars(poly) => {
                let m = r.len() / 2;
                let (r2, r1) = r.split_at(m);
                let (eq_one, eq_two) =
                    rayon::join(|| EqPolynomial::evals(r2), || EqPolynomial::evals(r1));

                poly.split_eq_evaluate(r, &eq_one, &eq_two)
            }
            MultilinearPolynomial::U16Scalars(poly) => {
                let m = r.len() / 2;
                let (r2, r1) = r.split_at(m);
                let (eq_one, eq_two) =
                    rayon::join(|| EqPolynomial::evals(r2), || EqPolynomial::evals(r1));

                poly.split_eq_evaluate(r, &eq_one, &eq_two)
            }
            MultilinearPolynomial::U32Scalars(poly) => {
                let m = r.len() / 2;
                let (r2, r1) = r.split_at(m);
                let (eq_one, eq_two) =
                    rayon::join(|| EqPolynomial::evals(r2), || EqPolynomial::evals(r1));

                poly.split_eq_evaluate(r, &eq_one, &eq_two)
            }
            MultilinearPolynomial::U64Scalars(poly) => {
                let m = r.len() / 2;
                let (r2, r1) = r.split_at(m);
                let (eq_one, eq_two) =
                    rayon::join(|| EqPolynomial::evals(r2), || EqPolynomial::evals(r1));

                poly.split_eq_evaluate(r, &eq_one, &eq_two)
            }
            MultilinearPolynomial::I64Scalars(poly) => {
                let m = r.len() / 2;
                let (r2, r1) = r.split_at(m);
                let (eq_one, eq_two) =
                    rayon::join(|| EqPolynomial::evals(r2), || EqPolynomial::evals(r1));

                poly.split_eq_evaluate(r, &eq_one, &eq_two)
            }
            MultilinearPolynomial::OneHot(poly) => poly.evaluate(r),
            _ => unimplemented!("Unsupported MultilinearPolynomial variant"),
        }
    }

    #[tracing::instrument(skip_all, name = "MultilinearPolynomial::batch_evaluate")]
    fn batch_evaluate(polys: &[&Self], r: &[F]) -> Vec<F> {
        let num_polys = polys.len();
        let m = r.len() / 2;
        let (r2, r1) = r.split_at(m);
        let (eq_one, eq_two) = rayon::join(|| EqPolynomial::evals(r2), || EqPolynomial::evals(r1));

        let evals = (0..eq_one.len())
            .into_par_iter()
            .map(|x1| {
                let eq1_val = eq_one[x1];
                // computing agg[x1]
                let inner_sums = (0..eq_two.len())
                    .into_par_iter()
                    .filter_map(|x2| {
                        let eq2_val = eq_two[x2];
                        let idx = x1 * eq_two.len() + x2;
                        let partial: Vec<F> = polys
                            .iter()
                            .map(|poly| match poly {
                                MultilinearPolynomial::LargeScalars(poly) => {
                                    let z = poly.Z[idx];
                                    OptimizedMul::mul_01_optimized(eq2_val, z)
                                }
                                MultilinearPolynomial::U8Scalars(poly) => {
                                    let z = poly.coeffs[idx];
                                    z.field_mul(eq2_val)
                                }
                                MultilinearPolynomial::U16Scalars(poly) => {
                                    let z = poly.coeffs[idx];
                                    z.field_mul(eq2_val)
                                }
                                MultilinearPolynomial::U32Scalars(poly) => {
                                    let z = poly.coeffs[idx];
                                    z.field_mul(eq2_val)
                                }
                                MultilinearPolynomial::U64Scalars(poly) => {
                                    let z = poly.coeffs[idx];
                                    z.field_mul(eq2_val)
                                }
                                MultilinearPolynomial::I64Scalars(poly) => {
                                    let z = poly.coeffs[idx];
                                    z.field_mul(eq2_val)
                                }
                                _ => unimplemented!(),
                            })
                            .collect();

                        Some(partial)
                    })
                    .reduce(
                        || vec![F::zero(); num_polys],
                        |mut acc, item| {
                            for i in 0..num_polys {
                                acc[i] += item[i];
                            }
                            acc
                        },
                    );
                // now inner_sums[i] = eq1[x1]*\sum_{x_2} eq2[x2]*f(x1||x2)
                inner_sums
                    .into_iter()
                    .map(|s| OptimizedMul::mul_01_optimized(eq1_val, s))
                    .collect::<Vec<_>>()
            })
            .reduce(
                || vec![F::zero(); num_polys],
                |mut acc, item| {
                    for i in 0..num_polys {
                        acc[i] += item[i];
                    }
                    acc
                },
            );
        evals
    }

    #[inline]
    fn sumcheck_evals(&self, index: usize, degree: usize, order: BindingOrder) -> Vec<F> {
        debug_assert!(degree > 0);
        debug_assert!(index < self.len() / 2);

        let mut evals = vec![F::zero(); degree];
        match order {
            BindingOrder::HighToLow => {
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
            }
            BindingOrder::LowToHigh => {
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
            }
        };
        evals
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use rand_chacha::ChaCha20Rng;
    use rand_core::{RngCore, SeedableRng};

    fn random_poly(max_num_bits: usize, len: usize) -> MultilinearPolynomial<Fr> {
        let mut rng = ChaCha20Rng::seed_from_u64(len as u64);
        match max_num_bits {
            0 => MultilinearPolynomial::from(vec![0u8; len]),
            1..=8 => MultilinearPolynomial::from(
                (0..len)
                    .map(|_| {
                        let mask = if max_num_bits == 8 {
                            u8::MAX
                        } else {
                            (1u8 << max_num_bits) - 1
                        };
                        (rng.next_u32() & (mask as u32)) as u8
                    })
                    .collect::<Vec<_>>(),
            ),
            9..=16 => MultilinearPolynomial::from(
                (0..len)
                    .map(|_| {
                        let mask = if max_num_bits == 16 {
                            u16::MAX
                        } else {
                            (1u16 << max_num_bits) - 1
                        };
                        (rng.next_u32() & (mask as u32)) as u16
                    })
                    .collect::<Vec<_>>(),
            ),
            17..=32 => MultilinearPolynomial::from(
                (0..len)
                    .map(|_| {
                        let mask = if max_num_bits == 32 {
                            u32::MAX
                        } else {
                            (1u32 << max_num_bits) - 1
                        };
                        (rng.next_u64() & (mask as u64)) as u32
                    })
                    .collect::<Vec<_>>(),
            ),
            33..=64 => MultilinearPolynomial::from(
                (0..len)
                    .map(|_| {
                        let mask = if max_num_bits == 64 {
                            u64::MAX
                        } else {
                            (1u64 << max_num_bits) - 1
                        };
                        rng.next_u64() & mask
                    })
                    .collect::<Vec<_>>(),
            ),
            _ => MultilinearPolynomial::from(
                (0..len).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>(),
            ),
        }
    }

    #[test]
    fn test_poly_to_field_elements() {
        let small_value_lookup_tables = <Fr as JoltField>::compute_lookup_tables();
        <Fr as JoltField>::initialize_lookup_tables(small_value_lookup_tables);

        let max_num_bits = [
            vec![8; 100],
            vec![16; 100],
            vec![32; 100],
            vec![64; 100],
            vec![256; 300],
        ]
        .concat();

        for &max_num_bits in max_num_bits.iter() {
            let len = 1 << 2;
            let poly = random_poly(max_num_bits, len);
            let field_elements: Vec<Fr> = match poly {
                MultilinearPolynomial::U8Scalars(poly) => poly.coeffs_as_field_elements(),
                MultilinearPolynomial::U16Scalars(poly) => poly.coeffs_as_field_elements(),
                MultilinearPolynomial::U32Scalars(poly) => poly.coeffs_as_field_elements(),
                MultilinearPolynomial::U64Scalars(poly) => poly.coeffs_as_field_elements(),
                MultilinearPolynomial::LargeScalars(poly) => poly.evals(),
                _ => {
                    panic!("Unexpected MultilinearPolynomial variant");
                }
            };
            assert_eq!(field_elements.len(), len);
        }
    }
}
