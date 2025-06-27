use crate::field::JoltField;
use crate::msm::VariableBaseMSM;
use crate::poly::commitment::dory::{get_T, get_num_columns, JoltFieldWrapper, JoltGroupWrapper};
use crate::poly::compact_polynomial::{CompactPolynomial, SmallScalar};
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::inc_polynomial::IncPolynomial;
use crate::poly::one_hot_polynomial::OneHotPolynomial;
use crate::utils::thread::unsafe_allocate_zero_vec;
use ark_bn254::{Fr, G1Projective};
use ark_ec::CurveGroup;
use num_traits::MulAdd;
use rayon::prelude::*;
use tracing::trace_span;

#[derive(Default, Clone, Debug, PartialEq)]
pub struct RLCPolynomial<F: JoltField> {
    pub num_rows: usize,
    /// Length-T vector of (dense) coefficients (i.e. k=0)
    pub dense_rlc: Vec<F>,
    /// Random linear combiation of one-hot polynomials, represented
    /// by a vector of (coefficient, one-hot polynomial) pairs
    one_hot_rlc: Vec<(F, OneHotPolynomial<F>)>,
    /// Random linear combiation of Inc polynomials, represented
    /// by a vector of (coefficient, Inc polynomial) pairs
    inc_rlc: Vec<(F, IncPolynomial<F>)>,
    num_variables_bound: usize,
}

impl<F: JoltField> RLCPolynomial<F> {
    pub fn new(num_rows: usize) -> Self {
        Self {
            num_rows,
            dense_rlc: unsafe_allocate_zero_vec(get_T()),
            one_hot_rlc: vec![],
            inc_rlc: vec![],
            num_variables_bound: 0,
        }
    }

    // TODO(moodlezoup): we should be able to cache the row commitments
    // for each underlying polynomial and take a linear combination of those
    #[tracing::instrument(skip_all, name = "RLCPolynomial::commit_rows")]
    pub fn commit_rows<G: CurveGroup<ScalarField = F> + VariableBaseMSM>(
        &self,
        bases: &[G::Affine],
    ) -> Vec<JoltGroupWrapper<G>> {
        let num_rows = self.num_rows;
        println!("# rows = {num_rows}");
        let row_len = get_num_columns();

        let mut row_commitments = vec![JoltGroupWrapper(G::zero()); num_rows];

        self.dense_rlc
            .par_chunks(row_len)
            .zip(row_commitments.par_iter_mut())
            .for_each(|(dense_row, commitment)| {
                let msm_result: G =
                    VariableBaseMSM::msm_field_elements(bases, None, dense_row, None, false)
                        .unwrap();
                *commitment = JoltGroupWrapper(commitment.0 + msm_result)
            });

        for (coeff, poly) in self.one_hot_rlc.iter() {
            let mut new_row_commitments: Vec<JoltGroupWrapper<G>> = poly.commit_rows(bases);

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

            // Use `jolt-optimizations`: v[i] = scalar * v[i] + gamma[i]
            jolt_optimizations::vector_scalar_mul_add_gamma_g1_online(
                updated_row_commitments,
                coeff_fr,
                current_row_commitments,
            );

            let _ = std::mem::replace(&mut row_commitments, new_row_commitments);
        }

        for (coeff, poly) in self.inc_rlc.iter() {
            let mut new_row_commitments: Vec<JoltGroupWrapper<G>> = poly.commit_rows(bases);

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

            // Use `jolt-optimizations`: v[i] = scalar * v[i] + gamma[i]
            jolt_optimizations::vector_scalar_mul_add_gamma_g1_online(
                updated_row_commitments,
                coeff_fr,
                current_row_commitments,
            );

            let _ = std::mem::replace(&mut row_commitments, new_row_commitments);
        }

        row_commitments
    }

    #[tracing::instrument(skip_all, name = "RLCPolynomial::vector_matrix_product")]
    pub fn vector_matrix_product(
        &self,
        left_vec: &[JoltFieldWrapper<F>],
    ) -> Vec<JoltFieldWrapper<F>> {
        let left_vec: &[F] =
            unsafe { std::slice::from_raw_parts(left_vec.as_ptr() as *const F, left_vec.len()) };
        let num_columns = get_num_columns();

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

        for (coeff, poly) in self.one_hot_rlc.iter() {
            // TODO(moodlezoup): Pass result by mutable reference to
            // poly.vector_matrix_product
            result
                .par_iter_mut()
                .zip(poly.vector_matrix_product(left_vec).into_par_iter())
                .for_each(|(result, new)| {
                    result.0 += new * coeff;
                });
        }

        for (coeff, poly) in self.inc_rlc.iter() {
            // TODO(moodlezoup): Pass result by mutable reference to
            // poly.vector_matrix_product
            result
                .par_iter_mut()
                .zip(poly.vector_matrix_product(left_vec).into_par_iter())
                .for_each(|(result, new)| {
                    result.0 += new * coeff;
                });
        }

        result
    }
}

impl<F: JoltField> MulAdd<F, RLCPolynomial<F>> for &OneHotPolynomial<F> {
    type Output = RLCPolynomial<F>;

    fn mul_add(self, a: F, mut b: RLCPolynomial<F>) -> RLCPolynomial<F> {
        b.one_hot_rlc.push((a, self.clone())); // TODO(moodlezoup): avoid clone
        b
    }
}

impl<F: JoltField> MulAdd<F, RLCPolynomial<F>> for &IncPolynomial<F> {
    type Output = RLCPolynomial<F>;

    fn mul_add(self, a: F, mut b: RLCPolynomial<F>) -> RLCPolynomial<F> {
        b.inc_rlc.push((a, self.clone())); // TODO(moodlezoup): avoid clone
        b
    }
}

impl<T: SmallScalar, F: JoltField> MulAdd<F, RLCPolynomial<F>> for &CompactPolynomial<T, F> {
    type Output = RLCPolynomial<F>;

    fn mul_add(self, a: F, mut b: RLCPolynomial<F>) -> RLCPolynomial<F> {
        b.dense_rlc
            .par_iter_mut()
            .zip_eq(self.coeffs.par_iter())
            .for_each(|(acc, new)| {
                *acc += new.field_mul(a);
            });
        b
    }
}

impl<F: JoltField> MulAdd<F, RLCPolynomial<F>> for &DensePolynomial<F> {
    type Output = RLCPolynomial<F>;

    fn mul_add(self, a: F, mut b: RLCPolynomial<F>) -> RLCPolynomial<F> {
        b.dense_rlc
            .par_iter_mut()
            .zip_eq(self.Z.par_iter())
            .for_each(|(acc, new)| {
                *acc += a * new;
            });
        b
    }
}
