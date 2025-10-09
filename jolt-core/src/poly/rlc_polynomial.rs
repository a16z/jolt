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
    /// In practice, this function is only used for testing; we compute the
    /// row commitments for an RLCPolynomial by combining the row commitments
    /// of the constituent polynomials.
    #[tracing::instrument(skip_all, name = "RLCPolynomial::commit_rows")]
    pub fn commit_rows<G: CurveGroup<ScalarField = F> + VariableBaseMSM>(
        &self,
        bases: &[G::Affine],
    ) -> Vec<JoltGroupWrapper<G>> {
        let num_rows = DoryGlobals::get_dimension();
        tracing::debug!("Committing to RLC polynomial with {num_rows} rows");
        let row_len = DoryGlobals::get_dimension();

        let T = DoryGlobals::get_T();
        assert!(T > num_rows, "T = {T}, why are you doing this",);
        let cycles_per_row = T / num_rows;

        let mut row_commitments = vec![JoltGroupWrapper(G::zero()); num_rows];

        let dense_poly_bases: Vec<_> = bases
            .iter()
            .take(row_len)
            .step_by(row_len / cycles_per_row)
            .copied()
            .collect();

        // // Compute the row commitments for dense submatrix
        // self.dense_rlc
        //     .par_chunks(cycles_per_row)
        //     .zip(row_commitments.par_iter_mut())
        //     .for_each(|(dense_row, commitment)| {
        //         let msm_result: G =
        //             VariableBaseMSM::msm_field_elements(&dense_poly_bases, dense_row).unwrap();
        //         *commitment = JoltGroupWrapper(msm_result)
        //     });

        // Compute the row commitments for one-hot polynomials
        for (coeff, poly) in self.one_hot_rlc.iter() {
            let mut new_row_commitments: Vec<JoltGroupWrapper<G>> = match poly.as_ref() {
                MultilinearPolynomial::OneHot(one_hot) => one_hot.commit_rows(bases),
                _ => panic!("Expected OneHot polynomial in one_hot_rlc"),
            };

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

        let num_rows = DoryGlobals::get_dimension();
        let row_len = DoryGlobals::get_dimension();
        let T = DoryGlobals::get_T();
        assert!(T > num_rows, "T = {T}, why are you doing this",);
        let cycles_per_row = T / num_rows;

        let mut result: Vec<JoltFieldWrapper<F>> = vec![JoltFieldWrapper(F::zero()); row_len]; // row_len == # columns

        // Compute the vector-matrix product for dense submatrix
        result
            .par_iter_mut()
            .step_by(row_len / cycles_per_row)
            .enumerate()
            .for_each(|(offset, dot_product_result)| {
                (*dot_product_result).0 = self
                    .dense_rlc
                    .par_iter()
                    .skip(offset)
                    .step_by(cycles_per_row)
                    .zip(left_vec.par_iter())
                    .map(|(&a, &b)| -> F { a * b })
                    .sum::<F>();
            });

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
