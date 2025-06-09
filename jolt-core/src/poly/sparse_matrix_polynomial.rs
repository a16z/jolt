use std::ops::Index;

use super::multilinear_polynomial::{BindingOrder, PolynomialBinding};
use crate::poly::compact_polynomial::{CompactPolynomial, SmallScalar};
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::one_hot_polynomial::OneHotPolynomial;
use crate::utils::compute_dotproduct;
use crate::utils::math::Math;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::{field::JoltField, utils};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use num_integer::Integer;
use num_traits::MulAdd;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use std::cmp::Ordering;

#[derive(Default, Clone, Debug, PartialEq)]
pub struct SparseMatrixPolynomial<F: JoltField> {
    pub num_rows: usize,
    dense_submatrix: Vec<F>,
    /// Each group is a vector of (row_index, col_index, coeff) tuples.
    /// There is no guarantee on the ordering of tuples within a given group.
    pub row_groups: Vec<Vec<(usize, usize, F)>>,
}

static ROWS_PER_GROUP: OnceCell<usize> = OnceCell::new();
static ROW_LENGTH: OnceCell<usize> = OnceCell::new();
static T: OnceCell<usize> = OnceCell::new();

impl<F: JoltField> SparseMatrixPolynomial<F> {
    pub fn initialize(K: usize, trace_length: usize) {
        let num_vars = K.log_2() + trace_length.log_2();
        let num_rows = num_vars + 1 / 2;
        let num_cols = num_vars - num_rows;
        let num_groups = rayon::current_num_threads() * 16;
        let rows_per_group = std::cmp::max(num_rows / num_groups, 1);
        ROWS_PER_GROUP.set(rows_per_group);
        ROW_LENGTH.set(num_cols);
        T.set(trace_length);
    }

    fn get_rows_per_group() -> usize {
        ROWS_PER_GROUP
            .get()
            .cloned()
            .expect("ROWS_PER_GROUP is uninitialized")
    }

    fn get_row_len() -> usize {
        ROW_LENGTH
            .get()
            .cloned()
            .expect("ROW_LENGTH is uninitialized")
    }

    fn get_trace_len() -> usize {
        T.get().cloned().expect("T is uninitialized")
    }

    fn num_dense_rows(&self) -> usize {
        self.dense_submatrix.len() / Self::get_row_len()
    }

    pub fn new(num_rows: usize) -> Self {
        debug_assert!(num_rows.is_power_of_two());

        let num_groups = std::cmp::max(num_rows / Self::get_rows_per_group(), 1);

        Self {
            num_rows,
            dense_submatrix: unsafe_allocate_zero_vec(Self::get_trace_len()),
            // TODO(moodlezoup): init w/ zeros?
            row_groups: vec![vec![]; num_groups],
        }
    }

    pub fn matrix_vector_product(&self, r_vec: Vec<F>) -> Vec<F> {
        let row_length = Self::get_row_len();
        assert_eq!(r_vec.len(), row_length);
        let num_rows_per_group = Self::get_rows_per_group();
        // TODO(moodlezoup): preallocate result vector to avoid flat_map
        let mut sparse_matrix_vector_product: Vec<F> = self
            .row_groups
            .par_iter()
            .flat_map(|group| {
                let mut dot_products = unsafe_allocate_zero_vec(num_rows_per_group);
                for (row_index, col_index, coeff) in group.iter() {
                    dot_products[row_index % num_rows_per_group] += r_vec[*col_index] * coeff;
                }
                dot_products
            })
            .collect();

        sparse_matrix_vector_product[..self.num_dense_rows()]
            .par_iter_mut()
            .zip(self.dense_submatrix.par_chunks(row_length))
            .for_each(|(result, dense_row)| *result += compute_dotproduct(dense_row, &r_vec));

        sparse_matrix_vector_product
    }

    pub fn evaluate(&self, r: &[F]) -> F {
        let (r_left, r_right) = r.split_at(self.num_rows.log_2());
        let (eq_left, eq_right) = rayon::join(
            || EqPolynomial::evals(r_left),
            || EqPolynomial::evals(r_right),
        );

        // Compute evaluation as a vector-matrix-vector product
        self.matrix_vector_product(eq_right)
            .into_par_iter()
            .zip_eq(eq_left.into_par_iter())
            .map(|(x, y)| x * y)
            .sum()
    }
}

impl<F: JoltField> PolynomialBinding<F> for SparseMatrixPolynomial<F> {
    fn is_bound(&self) -> bool {
        todo!()
    }

    fn bind(&mut self, r: F, order: BindingOrder) {
        todo!()
    }

    fn bind_parallel(&mut self, r: F, order: BindingOrder) {
        todo!()
    }

    fn final_sumcheck_claim(&self) -> F {
        todo!()
    }
}

impl<F: JoltField> MulAdd<F, SparseMatrixPolynomial<F>> for &OneHotPolynomial<F> {
    type Output = SparseMatrixPolynomial<F>;

    fn mul_add(self, a: F, mut b: SparseMatrixPolynomial<F>) -> SparseMatrixPolynomial<F> {
        todo!();
        // if self.nonzero_coeffs.is_empty() {
        //     // All non-zero coefficients are implicitly 1
        //     for index in self.nonzero_indices.iter() {}
        //     assert_eq!(self.num_cols, b.num_cols, "num_cols (row size)");
        // } else {
        // }

        b
    }
}

impl<F: JoltField> MulAdd<F, SparseMatrixPolynomial<F>> for &SparseMatrixPolynomial<F> {
    type Output = SparseMatrixPolynomial<F>;

    fn mul_add(self, a: F, mut b: SparseMatrixPolynomial<F>) -> SparseMatrixPolynomial<F> {
        assert_eq!(
            self.dense_submatrix.len(),
            b.dense_submatrix.len(),
            "dense submatrix size"
        );

        b.dense_submatrix
            .par_iter_mut()
            .zip(self.dense_submatrix.par_iter())
            .for_each(|(acc, new)| {
                *acc += a * new;
            });

        // Potentially end up with more than one tuple per (row_index, col_index)
        // Seems like that would be OK for vector-matrix product and evaluation,
        // but might make binding/univariate poly computation more annoying?
        b.row_groups
            .par_iter_mut()
            .zip(self.row_groups.par_iter())
            .for_each(|(acc, new)| {
                acc.extend_from_slice(new);
            });
        // TODO(moodlezoup): Parallelize
        if self.row_groups.len() > b.row_groups.len() {
            for group in self.row_groups[b.row_groups.len()..].iter() {
                b.row_groups.push(group.clone());
            }
        }

        b
    }
}

impl<T: SmallScalar, F: JoltField> MulAdd<F, SparseMatrixPolynomial<F>>
    for &CompactPolynomial<T, F>
{
    type Output = SparseMatrixPolynomial<F>;

    fn mul_add(self, a: F, mut b: SparseMatrixPolynomial<F>) -> SparseMatrixPolynomial<F> {
        b.dense_submatrix
            .par_iter_mut()
            .zip(self.coeffs.par_iter())
            .for_each(|(acc, new)| {
                *acc += new.field_mul(a);
            });
        b
    }
}

impl<F: JoltField> MulAdd<F, SparseMatrixPolynomial<F>> for &DensePolynomial<F> {
    type Output = SparseMatrixPolynomial<F>;

    fn mul_add(self, a: F, mut b: SparseMatrixPolynomial<F>) -> SparseMatrixPolynomial<F> {
        b.dense_submatrix
            .par_iter_mut()
            .zip(self.Z.par_iter())
            .for_each(|(acc, new)| {
                *acc += a * new;
            });
        b
    }
}
