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
    dense_submatrix: Vec<F>,
    /// Each group is a vector of (row_index, col_index, coeff) tuples.
    /// There is no guarantee on the ordering of tuples within a given group.
    pub column_groups: Vec<Vec<(usize, usize, F)>>,
}

static GLOBAL_K: OnceCell<usize> = OnceCell::new();
static GLOBAL_T: OnceCell<usize> = OnceCell::new();
static MAX_NUM_ROWS: OnceCell<usize> = OnceCell::new();
static NUM_COLUMNS: OnceCell<usize> = OnceCell::new();
static COLUMNS_PER_GROUP: OnceCell<usize> = OnceCell::new();

fn get_columns_per_group() -> usize {
    COLUMNS_PER_GROUP
        .get()
        .cloned()
        .expect("COLUMNS_PER_GROUP is uninitialized")
}

fn get_max_num_rows() -> usize {
    MAX_NUM_ROWS
        .get()
        .cloned()
        .expect("MAX_NUM_ROWS is uninitialized")
}

fn get_num_columns() -> usize {
    NUM_COLUMNS
        .get()
        .cloned()
        .expect("NUM_COLUMNS is uninitialized")
}

fn get_T() -> usize {
    GLOBAL_T.get().cloned().expect("T is uninitialized")
}

fn get_K() -> usize {
    GLOBAL_K.get().cloned().expect("K is uninitialized")
}

impl<F: JoltField> SparseMatrixPolynomial<F> {
    pub fn initialize(K: usize, T: usize) {
        debug_assert!(T >= K);
        let matrix_size = K as u128 * T as u128;
        let num_rows = matrix_size.isqrt();
        let num_columns = matrix_size / num_rows;
        let num_groups = rayon::current_num_threads() * 64;
        let columns_per_group = std::cmp::max(num_columns as usize / num_groups, 1);
        let _ = GLOBAL_K.set(K);
        let _ = GLOBAL_T.set(T);
        let _ = MAX_NUM_ROWS.set(num_rows as usize);
        let _ = NUM_COLUMNS.set(num_columns as usize);
        let _ = COLUMNS_PER_GROUP.set(columns_per_group);
    }

    pub fn new(num_rows: usize) -> Self {
        debug_assert!(num_rows.is_power_of_two());
        let num_groups = std::cmp::max(num_rows / get_columns_per_group(), 1);

        Self {
            num_rows,
            dense_submatrix: unsafe_allocate_zero_vec(get_T()),
            column_groups: vec![vec![]; num_groups],
        }
    }

    pub fn vector_matrix_product(&self, l_vec: &[F]) -> Vec<F> {
        let column_height = get_max_num_rows();
        debug_assert_eq!(column_height, l_vec.len());

        let num_cols_per_group = get_columns_per_group();
        // TODO(moodlezoup): Avoid flat_map
        let mut sparse_product: Vec<F> = self
            .column_groups
            .par_iter()
            .flat_map(|group| {
                let mut dot_products = unsafe_allocate_zero_vec(num_cols_per_group);
                for (row_index, col_index, coeff) in group.iter() {
                    dot_products[col_index % num_cols_per_group] += l_vec[*row_index] * coeff;
                }
                dot_products
            })
            .collect();

        let K = get_K();
        let row_length = get_num_columns();
        sparse_product
            .par_iter_mut()
            .enumerate()
            .for_each(|(column_index, dest)| {
                *dest += l_vec
                    .iter()
                    .step_by(K)
                    .enumerate()
                    .map(|(i, l_entry)| {
                        *l_entry * self.dense_submatrix[i * row_length + column_index]
                    })
                    .sum();
            });

        sparse_product
    }

    pub fn evaluate(&self, r: &[F]) -> F {
        let row_length = get_num_columns();
        let column_height = get_max_num_rows();
        assert_eq!(row_length.log_2() + column_height.log_2(), r.len());

        let (r_left, r_right) = r.split_at(column_height.log_2());
        let (eq_left, eq_right) = rayon::join(
            || EqPolynomial::evals(r_left),
            || EqPolynomial::evals(r_right),
        );

        // Compute evaluation as a vector-matrix-vector product
        self.vector_matrix_product(&eq_left)
            .into_par_iter()
            .zip_eq(eq_right.into_par_iter())
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
        assert!(b.column_groups.len() >= self.column_groups.len());
        // Potentially end up with more than one tuple per (row_index, col_index)
        b.column_groups
            .par_iter_mut()
            .zip(self.column_groups.par_iter())
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

        assert!(b.column_groups.len() >= self.column_groups.len());

        // Potentially end up with more than one tuple per (row_index, col_index)
        b.column_groups
            .par_iter_mut()
            .zip(self.column_groups.par_iter())
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

#[derive(Default, Clone, Debug, PartialEq)]
pub struct OneHotPolynomial<F: JoltField> {
    pub K: usize,
    pub nonzero_indices: Vec<usize>,
    pub nonzero_coeffs: Vec<i64>,
    column_groups: Vec<Vec<(usize, usize, F)>>,
}

impl<F: JoltField> OneHotPolynomial<F> {
    pub fn from_increments(increments: Vec<(usize, i64)>) -> Self {
        let T = get_T();
        let global_K = get_K();
        debug_assert_eq!(T, increments.len());

        let num_columns = get_num_columns();
        let column_height = get_max_num_rows();
        let num_groups = num_columns / get_columns_per_group();
        let cycles_per_column = column_height / global_K;

        let column_groups: Vec<_> = increments
            .par_iter()
            .enumerate()
            .chunks(T / num_groups)
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|(t, (k, coeff))| {
                        let column_index = t / cycles_per_column;
                        let row_index = (t % cycles_per_column) * global_K + *k;
                        (row_index, column_index, F::from_i64(*coeff))
                    })
                    .collect()
            })
            .collect();

        let (nonzero_indices, nonzero_coeffs) = increments.into_iter().unzip();
        Self {
            K: global_K,
            nonzero_indices,
            nonzero_coeffs,
            column_groups,
        }
    }

    pub fn from_indices(indices: Vec<usize>, K: usize) -> Self {
        let T = get_T();
        let global_K = get_K();
        debug_assert_eq!(T, indices.len());

        let num_columns = get_num_columns();
        let column_height = get_max_num_rows();
        let num_groups = num_columns / get_columns_per_group();
        let cycles_per_column = column_height / global_K;

        let column_groups: Vec<_> = indices
            .par_iter()
            .enumerate()
            .chunks(T / num_groups)
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|(t, k)| {
                        let column_index = t / cycles_per_column;
                        let row_index = (t % cycles_per_column) * global_K + *k;
                        (row_index, column_index, F::one())
                    })
                    .collect()
            })
            .collect();

        Self {
            K: global_K,
            nonzero_indices: indices,
            nonzero_coeffs: vec![],
            column_groups,
        }
    }

    fn evaluate(&self, r: &[F]) -> F {
        debug_assert_eq!(self.nonzero_indices.len(), get_T());
        assert_eq!(self.K.log_2() + get_num_columns().log_2(), r.len());

        let (r_left, r_right) = r.split_at(self.K.log_2());
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
                .map(|(t, (k, coeff))| eq_left[*k] * coeff.field_mul(eq_right[t]))
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
