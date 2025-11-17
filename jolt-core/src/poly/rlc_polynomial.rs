use crate::field::JoltField;
use crate::msm::VariableBaseMSM;
use crate::poly::commitment::dory::{DoryGlobals, JoltFieldWrapper, JoltGroupWrapper};
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::utils::small_scalar::SmallScalar;
use crate::utils::thread::unsafe_allocate_zero_vec;
use allocative::Allocative;
use ark_bn254::{Fr, G1Projective};
use ark_ec::CurveGroup;
use rayon::prelude::*;
use std::sync::Arc;
use tracing::trace_span;

/// `RLCPolynomial` represents a multilinear polynomial comprised of a
/// random linear combination of multiple polynomials, potentially with
/// different sizes.
#[derive(Default, Clone, Debug, PartialEq, Allocative)]
pub struct RLCPolynomial<F: JoltField> {
    /// Random linear combination of dense (i.e. length T) polynomials.
    pub dense_rlc: Vec<F>,
    /// Random linear combination of one-hot polynomials (length T x K
    /// for some K). Instead of pre-emptively combining these polynomials,
    /// as we do for `dense_rlc`, we store a vector of (coefficient, polynomial)
    /// pairs and lazily handle the linear combination in `commit_rows`
    /// and `vector_matrix_product`.
    pub one_hot_rlc: Vec<(F, Arc<MultilinearPolynomial<F>>)>,
}

impl<F: JoltField> RLCPolynomial<F> {
    pub fn new() -> Self {
        Self {
            dense_rlc: unsafe_allocate_zero_vec(DoryGlobals::get_T()),
            one_hot_rlc: vec![],
        }
    }

    #[tracing::instrument(skip_all)]
    pub fn linear_combination(
        polynomials: Vec<Arc<MultilinearPolynomial<F>>>,
        coefficients: &[F],
    ) -> Self {
        debug_assert_eq!(polynomials.len(), coefficients.len());

        let mut result = RLCPolynomial::<F>::new();
        let dense_indices: Vec<usize> = polynomials
            .iter()
            .enumerate()
            .filter(|(_, p)| !matches!(p.as_ref(), MultilinearPolynomial::OneHot(_)))
            .map(|(i, _)| i)
            .collect();

        if !dense_indices.is_empty() {
            let dense_len = result.dense_rlc.len();

            result.dense_rlc = (0..dense_len)
                .into_par_iter()
                .map(|i| {
                    let mut acc = F::zero();
                    for &poly_idx in &dense_indices {
                        let poly = polynomials[poly_idx].as_ref();
                        let coeff = coefficients[poly_idx];

                        if i < poly.original_len() {
                            match poly {
                                MultilinearPolynomial::U8Scalars(p) => {
                                    acc += p.coeffs[i].field_mul(coeff);
                                }
                                MultilinearPolynomial::U16Scalars(p) => {
                                    acc += p.coeffs[i].field_mul(coeff);
                                }
                                MultilinearPolynomial::U32Scalars(p) => {
                                    acc += p.coeffs[i].field_mul(coeff);
                                }
                                MultilinearPolynomial::U64Scalars(p) => {
                                    acc += p.coeffs[i].field_mul(coeff);
                                }
                                MultilinearPolynomial::I64Scalars(p) => {
                                    acc += p.coeffs[i].field_mul(coeff);
                                }
                                MultilinearPolynomial::U128Scalars(p) => {
                                    acc += p.coeffs[i].field_mul(coeff);
                                }
                                MultilinearPolynomial::I128Scalars(p) => {
                                    acc += p.coeffs[i].field_mul(coeff);
                                }
                                MultilinearPolynomial::S128Scalars(p) => {
                                    acc += p.coeffs[i].field_mul(coeff);
                                }
                                MultilinearPolynomial::LargeScalars(p) => {
                                    acc += p.Z[i] * coeff;
                                }
                                _ => unreachable!(),
                            }
                        }
                    }
                    acc
                })
                .collect();
        }
        for (i, poly) in polynomials.into_iter().enumerate() {
            if matches!(poly.as_ref(), MultilinearPolynomial::OneHot(_)) {
                result.one_hot_rlc.push((coefficients[i], poly));
            }
        }

        result
    }

    /// Commits to the rows of `RLCPolynomial`, viewing its coefficients
    /// as a matrix (used in Dory).
    /// We do so by computing the row commitments for the individual
    /// polynomials comprising the linear combination, and taking the
    /// linear combination of the resulting commitments.
    // TODO(moodlezoup): we should be able to cache the row commitments
    // for each underlying polynomial and take a linear combination of those
    #[tracing::instrument(skip_all, name = "RLCPolynomial::commit_rows")]
    pub fn commit_rows<G: CurveGroup<ScalarField = F> + VariableBaseMSM>(
        &self,
        bases: &[G::Affine],
    ) -> Vec<JoltGroupWrapper<G>> {
        let num_rows = DoryGlobals::get_max_num_rows();
        tracing::debug!("Committing to RLC polynomial with {num_rows} rows");
        let row_len = DoryGlobals::get_num_columns();

        let mut row_commitments = vec![JoltGroupWrapper(G::zero()); num_rows];

        // Compute the row commitments for dense submatrix
        self.dense_rlc
            .par_chunks(row_len)
            .zip(row_commitments.par_iter_mut())
            .for_each(|(dense_row, commitment)| {
                let msm_result: G =
                    VariableBaseMSM::msm_field_elements(&bases[..dense_row.len()], dense_row)
                        .unwrap();
                *commitment = JoltGroupWrapper(commitment.0 + msm_result)
            });

        // Compute the row commitments for one-hot polynomials
        for (coeff, poly) in self.one_hot_rlc.iter() {
            let mut new_row_commitments: Vec<JoltGroupWrapper<G>> = match poly.as_ref() {
                MultilinearPolynomial::OneHot(one_hot) => one_hot.commit_rows(bases),
                _ => panic!("Expected OneHot polynomial in one_hot_rlc"),
            };

            // TODO(moodlezoup): Avoid resize
            new_row_commitments.resize(num_rows, JoltGroupWrapper(G::zero()));

            let updated_row_commitments: &mut [G1Projective] = unsafe {
                std::slice::from_raw_parts_mut(
                    new_row_commitments.as_mut_ptr() as *mut G1Projective,
                    new_row_commitments.len(),
                )
            };

            let current_row_commitments: &[G1Projective] = unsafe {
                std::slice::from_raw_parts(
                    row_commitments.as_ptr() as *const G1Projective,
                    row_commitments.len(),
                )
            };

            let coeff_fr = unsafe { *(&raw const *coeff as *const Fr) };

            let _span = trace_span!("vector_scalar_mul_add_gamma_g1_online");
            let _enter = _span.enter();

            // Scales the row commitments for the current polynomial by
            // its coefficient
            jolt_optimizations::vector_scalar_mul_add_gamma_g1_online(
                updated_row_commitments,
                coeff_fr,
                current_row_commitments,
            );

            let _ = std::mem::replace(&mut row_commitments, new_row_commitments);
        }

        row_commitments
    }

    /// Computes a vector-matrix product, viewing the coefficients of the
    /// polynomial as a matrix (used in Dory).
    /// We do so by computing the vector-matrix product for the individual
    /// polynomials comprising the linear combination, and taking the
    /// linear combination of the resulting products.
    #[tracing::instrument(skip_all, name = "RLCPolynomial::vector_matrix_product")]
    pub fn vector_matrix_product(
        &self,
        left_vec: &[JoltFieldWrapper<F>],
    ) -> Vec<JoltFieldWrapper<F>> {
        let left_vec: &[F] =
            unsafe { std::slice::from_raw_parts(left_vec.as_ptr() as *const F, left_vec.len()) };
        let num_columns = DoryGlobals::get_num_columns();

        // Compute the vector-matrix product for dense submatrix
        // TODO(moodlezoup): better parallelism
        let mut result: Vec<_> = (0..num_columns)
            .into_par_iter()
            .map(|col_index| {
                JoltFieldWrapper(
                    self.dense_rlc
                        .iter()
                        .skip(col_index)
                        .step_by(num_columns)
                        .zip(left_vec.iter())
                        .map(|(&a, &b)| -> F { a * b })
                        .sum::<F>(),
                )
            })
            .collect();
        let result_slice: &mut [F] =
            unsafe { std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut F, result.len()) };

        // Compute the vector-matrix product for one-hot polynomials
        for (coeff, poly) in self.one_hot_rlc.iter() {
            match poly.as_ref() {
                MultilinearPolynomial::OneHot(one_hot) => {
                    one_hot.vector_matrix_product(left_vec, *coeff, result_slice);
                }
                _ => panic!("Expected OneHot polynomial in one_hot_rlc"),
            }
        }

        result
    }
}

type OneHotRow = Vec<Option<u8>>;

#[derive(Debug, Clone)]
pub struct StreamingOneHotRLCPolynomial<F, I> {
    polynomials: Vec<I>,
    coefficients: Vec<F>,
}

impl<F: JoltField, I: Iterator<Item = OneHotRow>> Iterator for StreamingOneHotRLCPolynomial<F, I> {
    type Item = Vec<(OneHotRow, F)>;

    fn next(&mut self) -> Option<Self::Item> {
        self.polynomials
            .iter_mut()
            .map(Iterator::next)
            .zip(&self.coefficients)
            .map(|(opt_row, coeff)| opt_row.map(|row| (row, *coeff)))
            .collect()
    }
}

impl<F: JoltField, I: Iterator<Item = OneHotRow>> StreamingOneHotRLCPolynomial<F, I> {
    pub fn linear_combination(polynomials: Vec<I>, coefficients: Vec<F>) -> Self {
        debug_assert_eq!(polynomials.len(), coefficients.len());

        Self {
            polynomials,
            coefficients,
        }
    }

    pub fn vector_matrix_product(
        self,
        left_vec: &[JoltFieldWrapper<F>],
    ) -> Vec<JoltFieldWrapper<F>> {
        debug_assert_eq!(left_vec.len(), DoryGlobals::get_max_num_rows());
        let left_vec: &[F] =
            unsafe { std::slice::from_raw_parts(left_vec.as_ptr() as *const F, left_vec.len()) };

        let T = DoryGlobals::get_T();
        let num_cols = DoryGlobals::get_num_columns();
        // XXX Is this always true?
        let chunk_len = num_cols;

        // TODO: Use par_iter; think about better ways to organize iteration
        let mut result = vec![F::zero(); num_cols];
        if T >= num_cols {
            let rows_per_k = T / num_cols;

            self.into_iter()
                .enumerate()
                .for_each(|(chunk_index, poly_chunks)| {
                    for (chunk, coeff) in poly_chunks {
                        for (col_index, dest) in result.iter_mut().enumerate() {
                            if let Some(k) = chunk[col_index] {
                                let row_index = k as usize * rows_per_k + chunk_index;
                                *dest += coeff * left_vec[row_index];
                            }
                        }
                    }
                });
        } else {
            self.into_iter()
                .enumerate()
                .for_each(|(chunk_index, poly_chunks)| {
                    for (chunk, coeff) in poly_chunks {
                        for (t, k) in chunk.into_iter().enumerate() {
                            let t = chunk_index * chunk_len + t;
                            if let Some(k) = k {
                                let global_index = k as u128 * T as u128 + t as u128;
                                let col_index = (global_index % num_cols as u128) as usize;
                                let row_index = (global_index / num_cols as u128) as usize;
                                result[col_index] += coeff * left_vec[row_index];
                            }
                        }
                    }
                })
        }

        result.into_iter().map(JoltFieldWrapper).collect()
    }
}

type DenseRow<F> = Vec<F>;

#[derive(Debug, Clone)]
pub struct StreamingDenseRLCPolynomial<F, I> {
    polynomials: Vec<I>,
    coefficients: Vec<F>,
}

impl<F: JoltField, I: Iterator<Item = DenseRow<F>>> Iterator for StreamingDenseRLCPolynomial<F, I> {
    type Item = DenseRow<F>;

    fn next(&mut self) -> Option<Self::Item> {
        let num_cols = DoryGlobals::get_num_columns();
        self.polynomials
            .iter_mut()
            .map(Iterator::next)
            .zip(&self.coefficients)
            .fold(
                Some(vec![F::zero(); num_cols]),
                |opt_acc, (opt_row, coefficient)| match (opt_acc, opt_row) {
                    (Some(mut acc), Some(row)) => {
                        for col in 0..row.len() {
                            acc[col] += *coefficient * row[col];
                        }

                        Some(acc)
                    }
                    _ => None,
                },
            )
    }
}

impl<F: JoltField, I: Iterator<Item = DenseRow<F>>> StreamingDenseRLCPolynomial<F, I> {
    pub fn linear_combination(polynomials: Vec<I>, coefficients: Vec<F>) -> Self {
        debug_assert_eq!(polynomials.len(), coefficients.len());

        Self {
            polynomials,
            coefficients,
        }
    }

    pub fn vector_matrix_product(
        self,
        left_vec: &[JoltFieldWrapper<F>],
    ) -> Vec<JoltFieldWrapper<F>> {
        debug_assert_eq!(left_vec.len(), DoryGlobals::get_max_num_rows());
        let left_vec: &[F] =
            unsafe { std::slice::from_raw_parts(left_vec.as_ptr() as *const F, left_vec.len()) };

        let num_cols = DoryGlobals::get_num_columns();
        self.into_iter()
            .zip(left_vec)
            .fold(vec![F::zero(); num_cols], |mut acc, (row, coefficient)| {
                for col in 0..num_cols {
                    acc[col] += *coefficient * row[col];
                }

                acc
            })
            .into_iter()
            .map(JoltFieldWrapper)
            .collect()
    }
}

#[cfg(test)]
mod test {
    use crate::poly::one_hot_polynomial::OneHotPolynomial;

    use super::*;

    enum StreamingMultilinearPolynomial<F> {
        Dense(Vec<DenseRow<F>>),
        OneHot(Vec<OneHotRow>),
    }

    impl<F: JoltField> From<&MultilinearPolynomial<F>> for StreamingMultilinearPolynomial<F> {
        fn from(value: &MultilinearPolynomial<F>) -> Self {
            let chunk_size = DoryGlobals::get_num_columns();

            match value {
                MultilinearPolynomial::LargeScalars(poly) => {
                    let chunks = poly
                        .Z
                        .chunks(chunk_size)
                        .map(|chunk| chunk.to_owned())
                        .collect::<Vec<_>>();
                    Self::Dense(chunks)
                }
                MultilinearPolynomial::OneHot(poly) => {
                    let chunks = poly
                        .nonzero_indices
                        .chunks(chunk_size)
                        .map(|chunk| chunk.to_owned())
                        .collect::<Vec<_>>();
                    Self::OneHot(chunks)
                }
                _ => unimplemented!(),
            }
        }
    }

    impl<F: JoltField> From<StreamingMultilinearPolynomial<F>> for Vec<DenseRow<F>> {
        fn from(value: StreamingMultilinearPolynomial<F>) -> Self {
            match value {
                StreamingMultilinearPolynomial::Dense(poly) => poly,
                _ => panic!("not a dense polynomial"),
            }
        }
    }

    impl<F: JoltField> From<StreamingMultilinearPolynomial<F>> for Vec<OneHotRow> {
        fn from(value: StreamingMultilinearPolynomial<F>) -> Self {
            match value {
                StreamingMultilinearPolynomial::OneHot(poly) => poly,
                _ => panic!("not a one-hot polynomial"),
            }
        }
    }

    fn random_multilinear_polynomial<F: JoltField>(
        rng: &mut impl rand_core::RngCore,
    ) -> MultilinearPolynomial<F> {
        let T = DoryGlobals::get_T();
        let K = (DoryGlobals::get_num_columns() * DoryGlobals::get_max_num_rows()) / T;

        if rng.next_u64() % 2 == 1 {
            {
                let indices: Vec<Option<u8>> = (0..T)
                    .map(|_| (rng.next_u64() % 2 == 1).then(|| (rng.next_u64() % (K as u64)) as u8))
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(indices, K))
            }
        } else {
            {
                let coeffs: Vec<F> = (0..T).map(|_| F::random(rng)).collect();
                MultilinearPolynomial::from(coeffs)
            }
        }
    }

    fn streaming_vector_matrix_product_k_t(K: usize, T: usize) {
        type F = ark_bn254::Fr;

        let _guard = DoryGlobals::initialize(K, T);
        let mut rng = rand_core::OsRng;

        let num_polys = 22;
        let num_rows = DoryGlobals::get_max_num_rows();

        let polynomials: Vec<MultilinearPolynomial<F>> = (0..num_polys)
            .map(|_| random_multilinear_polynomial(&mut rng))
            .collect();
        let coefficients: Vec<F> = (0..num_polys).map(|_| F::random(&mut rng)).collect();
        let left_vector: Vec<_> = (0..num_rows)
            .map(|_| JoltFieldWrapper(F::random(&mut rng)))
            .collect();

        let streaming_product: Vec<JoltFieldWrapper<F>> = {
            // Separate out dense and one-hot polynomials with their respective coefficients
            let (dense_polys_and_coeffs, one_hot_polys_and_coeffs) = polynomials
                .iter()
                .map(StreamingMultilinearPolynomial::from)
                .zip(coefficients.clone())
                .partition::<Vec<_>, _>(|(poly, _)| match poly {
                    StreamingMultilinearPolynomial::Dense(_) => true,
                    StreamingMultilinearPolynomial::OneHot(_) => false,
                });
            let (dense_polys, dense_coeffs): (Vec<_>, Vec<F>) = dense_polys_and_coeffs
                .into_iter()
                .map(|(poly, coeff)| (<Vec<DenseRow<F>>>::from(poly).into_iter(), coeff))
                .unzip();
            let (one_hot_polys, one_hot_coeffs): (Vec<_>, Vec<F>) = one_hot_polys_and_coeffs
                .into_iter()
                .map(|(poly, coeff)| (<Vec<OneHotRow>>::from(poly).into_iter(), coeff))
                .unzip();

            let dense_prod =
                StreamingDenseRLCPolynomial::linear_combination(dense_polys, dense_coeffs)
                    .vector_matrix_product(&left_vector);
            let dense_prod: &[F] = unsafe {
                std::slice::from_raw_parts(dense_prod.as_ptr() as *const F, dense_prod.len())
            };
            let one_hot_prod =
                StreamingOneHotRLCPolynomial::linear_combination(one_hot_polys, one_hot_coeffs)
                    .vector_matrix_product(&left_vector);
            let one_hot_prod: &[F] = unsafe {
                std::slice::from_raw_parts(one_hot_prod.as_ptr() as *const F, one_hot_prod.len())
            };
            dense_prod
                .iter()
                .zip(one_hot_prod)
                .map(|(a, b)| JoltFieldWrapper(a + b))
                .collect()
        };

        let non_streaming_product: Vec<JoltFieldWrapper<F>> = {
            let polynomials: Vec<Arc<MultilinearPolynomial<F>>> =
                polynomials.into_iter().map(|poly| Arc::new(poly)).collect();
            RLCPolynomial::linear_combination(polynomials, &coefficients)
                .vector_matrix_product(&left_vector)
        };

        assert_eq!(streaming_product, non_streaming_product,)
    }

    #[test]
    fn streaming_vector_matrix_product() {
        streaming_vector_matrix_product_k_t(1, 256);
        streaming_vector_matrix_product_k_t(256, 64);
        streaming_vector_matrix_product_k_t(256, 4096);
    }
}
