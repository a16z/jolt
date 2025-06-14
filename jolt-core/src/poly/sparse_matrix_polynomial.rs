use super::multilinear_polynomial::{BindingOrder, PolynomialBinding};
use crate::field::JoltField;
use crate::poly::compact_polynomial::{CompactPolynomial, SmallScalar};
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::utils::compute_dotproduct;
use crate::utils::math::Math;
use crate::utils::thread::unsafe_allocate_zero_vec;
use num_traits::MulAdd;
use once_cell::sync::OnceCell;
use rayon::prelude::*;

#[derive(Default, Clone, Debug, PartialEq)]
pub struct SparseMatrixPolynomial<F: JoltField> {
    pub num_rows: usize,
    dense_submatrix: Vec<F>,
    /// 2d vector of (k, t, coeff) tuples.
    /// There is no guarantee on the ordering of tuples within a given vector.
    pub sparse_coeffs: Vec<Vec<(usize, usize, F)>>,
}

static GLOBAL_K: OnceCell<usize> = OnceCell::new();
static GLOBAL_T: OnceCell<usize> = OnceCell::new();
static MAX_NUM_ROWS: OnceCell<usize> = OnceCell::new();
static NUM_COLUMNS: OnceCell<usize> = OnceCell::new();
static CYCLES_PER_GROUP: OnceCell<usize> = OnceCell::new();

pub fn get_max_num_vars() -> usize {
    get_K().log_2() + get_T().log_2()
}

fn get_cycles_per_group() -> usize {
    CYCLES_PER_GROUP
        .get()
        .cloned()
        .expect("CYCLES_PER_GROUP is uninitialized")
}

fn get_max_num_rows() -> usize {
    MAX_NUM_ROWS
        .get()
        .cloned()
        .expect("MAX_NUM_ROWS is uninitialized")
}

pub fn get_num_columns() -> usize {
    NUM_COLUMNS
        .get()
        .cloned()
        .expect("NUM_COLUMNS is uninitialized")
}

fn get_T() -> usize {
    GLOBAL_T.get().cloned().expect("GLOBAL_T is uninitialized")
}

pub fn get_K() -> usize {
    GLOBAL_K.get().cloned().expect("GLOBAL_K is uninitialized")
}

impl<F: JoltField> SparseMatrixPolynomial<F> {
    pub fn initialize(K: usize, T: usize) {
        // debug_assert!(T >= K);
        let matrix_size = K as u128 * T as u128;
        let num_columns = matrix_size.isqrt().next_power_of_two();
        let num_rows = matrix_size / num_columns;
        println!("# rows: {num_rows}");
        println!("# cols: {num_columns}");
        let num_groups = rayon::current_num_threads() * 64;
        let cycles_per_group = std::cmp::max(T / num_groups, 1);
        let _ = GLOBAL_K.set(K);
        let _ = GLOBAL_T.set(T);
        let _ = MAX_NUM_ROWS.set(num_rows as usize);
        let _ = NUM_COLUMNS.set(num_columns as usize);
        let _ = CYCLES_PER_GROUP.set(cycles_per_group);
    }

    pub fn new(num_rows: usize) -> Self {
        let num_groups = get_T() / get_cycles_per_group();
        Self {
            num_rows,
            dense_submatrix: unsafe_allocate_zero_vec(get_T()),
            sparse_coeffs: vec![vec![]; num_groups],
        }
    }

    pub fn vector_matrix_product(&self, l_vec: &[F]) -> Vec<F> {
        let column_height = get_max_num_rows();
        debug_assert_eq!(column_height, l_vec.len());

        let K = get_K();
        let T = get_T();

        let num_cycles_per_group = get_cycles_per_group();
        let num_groups = T / num_cycles_per_group;
        let num_columns = get_num_columns();
        let num_columns_per_group = num_columns / num_groups;

        // TODO(moodlezoup): Avoid flat_map
        let _sparse_product: Vec<F> = self
            .sparse_coeffs
            .par_iter()
            .flat_map(|group| {
                let mut dot_products = unsafe_allocate_zero_vec(num_columns_per_group);
                for (k, t, coeff) in group.iter() {
                    let global_index = *t as u128 * K as u128 + *k as u128;
                    let column_index = global_index / column_height as u128;
                    let row_index = global_index % column_height as u128;
                    dot_products[column_index as usize] += l_vec[row_index as usize] * coeff;
                }
                dot_products
            })
            .collect();

        todo!();
    }

    pub fn evaluate(&self, r: &[F]) -> F {
        let K = self.num_rows;
        assert_eq!(K.log_2() + get_T().log_2(), r.len());

        let (r_address, r_cycle) = r.split_at(K.log_2());
        let (eq_r_address, eq_r_cycle) = rayon::join(
            || EqPolynomial::evals(r_address),
            || EqPolynomial::evals(r_cycle),
        );

        let sparse_eval: F = self
            .sparse_coeffs
            .par_iter()
            .map(|group| {
                group
                    .iter()
                    .map(|(k, t, coeff)| eq_r_address[*k] * eq_r_cycle[*t] * coeff)
                    .sum::<F>()
            })
            .sum();
        let dense_eval = compute_dotproduct(&self.dense_submatrix, &eq_r_cycle);
        sparse_eval + dense_eval
    }
}

impl<F: JoltField> PolynomialBinding<F> for SparseMatrixPolynomial<F> {
    fn is_bound(&self) -> bool {
        todo!()
    }

    fn bind(&mut self, _r: F, _order: BindingOrder) {
        todo!()
    }

    fn bind_parallel(&mut self, _r: F, _order: BindingOrder) {
        todo!()
    }

    fn final_sumcheck_claim(&self) -> F {
        todo!()
    }
}

impl<F: JoltField> MulAdd<F, SparseMatrixPolynomial<F>> for &OneHotPolynomial {
    type Output = SparseMatrixPolynomial<F>;

    fn mul_add(self, a: F, mut b: SparseMatrixPolynomial<F>) -> SparseMatrixPolynomial<F> {
        let cycles_per_group = get_cycles_per_group();
        // Potentially end up with more than one tuple per (row_index, col_index)
        b.sparse_coeffs
            .par_iter_mut()
            .zip(
                self.nonzero_indices
                    .par_iter()
                    .enumerate()
                    .chunks(cycles_per_group),
            )
            .for_each(|(acc, new)| {
                acc.extend(new.iter().map(|(t, &k)| (k, *t, a)));
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

        assert!(b.sparse_coeffs.len() >= self.sparse_coeffs.len());

        // Potentially end up with more than one tuple per (k, t)
        b.sparse_coeffs
            .par_iter_mut()
            .zip(self.sparse_coeffs.par_iter())
            .for_each(|(acc, new)| {
                acc.extend(new.iter().map(|(k, t, coeff)| (*k, *t, a * coeff)));
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
pub struct OneHotPolynomial {
    pub K: usize,
    pub nonzero_indices: Vec<usize>,
    // pub nonzero_coeffs: Vec<i64>,
}

impl OneHotPolynomial {
    pub fn num_rows(&self) -> usize {
        let T = self.nonzero_indices.len() as u128;
        let row_length = get_num_columns() as u128;
        (T * self.K as u128 / row_length) as usize
    }

    // pub fn from_increments(increments: Vec<(usize, i64)>) -> Self {
    //     let T = get_T();
    //     let global_K = get_K();
    //     debug_assert_eq!(T, increments.len());

    //     let num_columns = get_num_columns();
    //     let column_height = get_max_num_rows();
    //     let num_groups = num_columns / get_cycles_per_group();
    //     let cycles_per_column = column_height / global_K;

    //     // let column_groups: Vec<_> = increments
    //     //     .par_iter()
    //     //     .enumerate()
    //     //     .chunks(T / num_groups)
    //     //     .map(|chunk| {
    //     //         chunk
    //     //             .iter()
    //     //             .map(|(t, (k, coeff))| {
    //     //                 let column_index = t / cycles_per_column;
    //     //                 let row_index = (t % cycles_per_column) * global_K + *k;
    //     //                 (row_index, column_index, F::from_i64(*coeff))
    //     //             })
    //     //             .collect()
    //     //     })
    //     //     .collect();

    //     let (nonzero_indices, nonzero_coeffs) = increments.into_iter().unzip();
    //     Self {
    //         K: global_K,
    //         nonzero_indices,
    //         nonzero_coeffs,
    //     }
    // }

    pub fn from_indices(indices: Vec<usize>, K: usize) -> Self {
        debug_assert_eq!(get_T(), indices.len());

        // let column_groups: Vec<_> = indices
        //     .par_iter()
        //     .enumerate()
        //     .chunks(global_T / num_groups)
        //     .map(|chunk| {
        //         chunk
        //             .iter()
        //             .map(|(t, k)| {
        //                 let column_index = t / cycles_per_column;
        //                 let row_index = (t % cycles_per_column) * global_K + *k;
        //                 (row_index, column_index, F::one())
        //             })
        //             .collect()
        //     })
        //     .collect();

        Self {
            K,
            nonzero_indices: indices,
            // nonzero_coeffs: vec![],
        }
    }

    fn _evaluate<F: JoltField>(&self, r: &[F]) -> F {
        let T = get_T();
        debug_assert_eq!(self.nonzero_indices.len(), T);
        assert_eq!(self.K.log_2() + T.log_2(), r.len());

        let (r_left, r_right) = r.split_at(self.K.log_2());
        let (eq_left, eq_right) = rayon::join(
            || EqPolynomial::evals(r_left),
            || EqPolynomial::evals(r_right),
        );

        self.nonzero_indices
            .par_iter()
            .enumerate()
            .map(|(t, k)| eq_left[*k] * eq_right[t])
            .sum()

        // if self.nonzero_coeffs.is_empty() {
        //     // All nonzero coefficients are implicitly 1
        //     self.nonzero_indices
        //         .par_iter()
        //         .enumerate()
        //         .map(|(t, k)| eq_left[*k] * eq_right[t])
        //         .sum()
        // } else {
        //     debug_assert_eq!(self.nonzero_indices.len(), self.nonzero_coeffs.len());
        //     self.nonzero_indices
        //         .par_iter()
        //         .zip(self.nonzero_coeffs.par_iter())
        //         .enumerate()
        //         .map(|(t, (k, coeff))| eq_left[*k] * coeff.field_mul(eq_right[t]))
        //         .sum()
        // }
    }
}

impl<F: JoltField> PolynomialBinding<F> for OneHotPolynomial {
    fn is_bound(&self) -> bool {
        todo!()
    }

    fn bind(&mut self, _r: F, _order: BindingOrder) {
        todo!()
    }

    fn bind_parallel(&mut self, _r: F, _order: BindingOrder) {
        todo!()
    }

    fn final_sumcheck_claim(&self) -> F {
        todo!()
    }
}
