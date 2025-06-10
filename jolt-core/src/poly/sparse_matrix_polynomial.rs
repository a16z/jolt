use std::ops::Index;

use super::multilinear_polynomial::{BindingOrder, PolynomialBinding};
use crate::poly::compact_polynomial::{CompactPolynomial, SmallScalar};
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
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
    dense_row: Vec<F>,
    /// Each group is a vector of (row_index, col_index, coeff) tuples.
    /// There is no guarantee on the ordering of tuples within a given group.
    pub row_groups: Vec<Vec<(usize, usize, F)>>,
}

static ROWS_PER_GROUP: OnceCell<usize> = OnceCell::new();
static GLOBAL_K: OnceCell<usize> = OnceCell::new();
static GLOBAL_T: OnceCell<usize> = OnceCell::new();

fn get_rows_per_group() -> usize {
    ROWS_PER_GROUP
        .get()
        .cloned()
        .expect("ROWS_PER_GROUP is uninitialized")
}

fn get_T() -> usize {
    GLOBAL_T.get().cloned().expect("T is uninitialized")
}

fn get_K() -> usize {
    GLOBAL_K.get().cloned().expect("K is uninitialized")
}

impl<F: JoltField> SparseMatrixPolynomial<F> {
    pub fn initialize(K: usize, T: usize) {
        let num_groups = rayon::current_num_threads() * 64;
        let rows_per_group = std::cmp::max(K / num_groups, 1);
        let _ = ROWS_PER_GROUP.set(rows_per_group);
        let _ = GLOBAL_K.set(K);
        let _ = GLOBAL_T.set(T);
    }

    pub fn new(num_rows: usize) -> Self {
        debug_assert!(num_rows.is_power_of_two());

        let num_groups = std::cmp::max(num_rows / get_rows_per_group(), 1);

        Self {
            num_rows,
            dense_row: unsafe_allocate_zero_vec(get_T()),
            row_groups: vec![vec![]; num_groups],
        }
    }

    pub fn matrix_vector_product(&self, r_vec: Vec<F>) -> Vec<F> {
        let row_length = get_T();
        assert_eq!(r_vec.len(), row_length);
        let num_rows_per_group = get_rows_per_group();
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

        sparse_matrix_vector_product[0] += compute_dotproduct(&self.dense_row, &r_vec);

        sparse_matrix_vector_product
    }

    pub fn evaluate(&self, r: &[F]) -> F {
        let row_length = get_T();
        let (r_left, r_right) = r.split_at(r.len() - row_length);
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
        assert!(b.row_groups.len() >= self.row_groups.len());
        // Potentially end up with more than one tuple per (row_index, col_index)
        // Seems like that would be OK for vector-matrix product and evaluation,
        // but might make binding/univariate poly computation more annoying?
        b.row_groups
            .par_iter_mut()
            .zip(self.row_groups.par_iter())
            .for_each(|(acc, new)| {
                acc.extend(new.iter().map(|(row, col, coeff)| (*row, *col, a * coeff)));
            });

        b
    }
}

impl<F: JoltField> MulAdd<F, SparseMatrixPolynomial<F>> for &SparseMatrixPolynomial<F> {
    type Output = SparseMatrixPolynomial<F>;

    fn mul_add(self, a: F, mut b: SparseMatrixPolynomial<F>) -> SparseMatrixPolynomial<F> {
        assert_eq!(
            self.dense_row.len(),
            b.dense_row.len(),
            "dense submatrix size"
        );

        b.dense_row
            .par_iter_mut()
            .zip(self.dense_row.par_iter())
            .for_each(|(acc, new)| {
                *acc += a * new;
            });

        assert!(b.row_groups.len() >= self.row_groups.len());
        // Potentially end up with more than one tuple per (row_index, col_index)
        // Seems like that would be OK for vector-matrix product and evaluation,
        // but might make binding/univariate poly computation more annoying?
        b.row_groups
            .par_iter_mut()
            .zip(self.row_groups.par_iter())
            .for_each(|(acc, new)| {
                acc.extend(new.iter().map(|(row, col, coeff)| (*row, *col, a * coeff)));
            });

        b
    }
}

impl<T: SmallScalar, F: JoltField> MulAdd<F, SparseMatrixPolynomial<F>>
    for &CompactPolynomial<T, F>
{
    type Output = SparseMatrixPolynomial<F>;

    fn mul_add(self, a: F, mut b: SparseMatrixPolynomial<F>) -> SparseMatrixPolynomial<F> {
        b.dense_row
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
        b.dense_row
            .par_iter_mut()
            .zip(self.Z.par_iter())
            .for_each(|(acc, new)| {
                *acc += a * new;
            });
        b
    }
}

#[derive(Default, Clone, Debug, PartialEq)]
pub struct OneHotPolynomial<F: JoltField> {
    pub num_rows: usize,
    pub nonzero_indices: Vec<usize>,
    pub nonzero_coeffs: Vec<F>,
    row_groups: Vec<Vec<(usize, usize, F)>>,
}

impl<F: JoltField> OneHotPolynomial<F> {
    pub fn from_increments(increments: Vec<(usize, i64)>) -> Self {
        todo!()
    }

    pub fn from_indices(indices: Vec<usize>) -> Self {
        todo!()
    }

    fn evaluate(&self, r: &[F]) -> F {
        debug_assert_eq!(self.nonzero_indices.len(), get_T());

        let row_length = get_T();
        let (r_left, r_right) = r.split_at(r.len() - row_length);
        let (eq_left, eq_right) = rayon::join(
            || EqPolynomial::evals(r_left),
            || EqPolynomial::evals(r_right),
        );

        if self.nonzero_coeffs.is_empty() {
            // All nonzero coefficients are implicitly 1
            self.nonzero_indices
                .par_iter()
                .enumerate()
                .map(|(t, k)| eq_left[*k] * eq_right[t])
                .sum()
        } else {
            debug_assert_eq!(self.nonzero_indices.len(), self.nonzero_coeffs.len());
            self.nonzero_indices
                .par_iter()
                .zip(self.nonzero_coeffs.par_iter())
                .enumerate()
                .map(|(t, (k, coeff))| eq_left[*k] * coeff * eq_right[t])
                .sum()
        }
    }
}

impl<F: JoltField> PolynomialBinding<F> for OneHotPolynomial<F> {
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
