//! One-hot multilinear polynomial — sparse representation where each row has
//! at most one nonzero entry with value 1.
//!
//! Used for Jolt's RA (random access) lookup index polynomials, where each
//! cycle selects exactly one of `k` possible values. Storing the hot index
//! per row instead of a dense `T × k` evaluation table reduces memory by
//! a factor of `k` and enables ~254× faster commitment via generator lookup
//! instead of full MSM.

use jolt_field::Field;

use crate::multilinear::MultilinearPoly;

/// Sparse multilinear polynomial where each row has at most one nonzero
/// entry, and that entry is always `F::one()`.
///
/// The polynomial represents a `(T × k)` evaluation table where `T` is the
/// number of rows (cycles) and `k` is the number of columns. Row `i` has
/// value 1 at column `indices[i]` and 0 elsewhere. `None` means the entire
/// row is zero.
///
/// `num_vars = log2(T * k)` where `T * k` must be a power of two.
#[derive(Clone, Debug)]
pub struct OneHotPolynomial {
    k: usize,
    indices: Vec<Option<u8>>,
    num_vars: usize,
}

impl OneHotPolynomial {
    /// Creates a one-hot polynomial from column indices.
    ///
    /// # Panics
    ///
    /// Panics if `k * indices.len()` is not a power of two.
    pub fn new(k: usize, indices: Vec<Option<u8>>) -> Self {
        assert!(
            k <= u8::MAX as usize + 1,
            "k exceeds u8 index range ({k} > 256)"
        );
        let total = k * indices.len();
        assert!(
            total.is_power_of_two(),
            "k * num_rows must be a power of two, got {total}"
        );
        let num_vars = total.trailing_zeros() as usize;
        Self {
            k,
            indices,
            num_vars,
        }
    }

    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }

    #[inline]
    pub fn indices(&self) -> &[Option<u8>] {
        &self.indices
    }

    #[inline]
    pub fn num_rows(&self) -> usize {
        self.indices.len()
    }

    /// Number of variables $n$. The polynomial has $2^n$ evaluations.
    ///
    /// Inherent method avoids trait disambiguation since [`MultilinearPoly`]
    /// is generic over `F`.
    #[inline]
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<F: Field> MultilinearPoly<F> for OneHotPolynomial {
    #[inline]
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn evaluate(&self, point: &[F]) -> F {
        assert_eq!(point.len(), self.num_vars);
        let eq_evals = crate::EqPolynomial::new(point.to_vec()).evaluations();
        let mut result = F::zero();
        for (row, &opt_col) in self.indices.iter().enumerate() {
            if let Some(col) = opt_col {
                result += eq_evals[row * self.k + col as usize];
            }
        }
        result
    }

    fn for_each_row(&self, sigma: usize, f: &mut dyn FnMut(usize, &[F])) {
        let num_cols = 1usize << sigma;
        let total_len = 1usize << self.num_vars;
        let num_rows = total_len / num_cols;

        // Pre-index nonzero entries by matrix row.
        let mut row_hot_cols: Vec<Vec<usize>> = vec![Vec::new(); num_rows];
        for (cycle, &opt_col) in self.indices.iter().enumerate() {
            if let Some(col) = opt_col {
                let flat = cycle * self.k + col as usize;
                row_hot_cols[flat / num_cols].push(flat % num_cols);
            }
        }

        let mut buf = crate::thread::unsafe_allocate_zero_vec(num_cols);
        for (row_idx, cols) in row_hot_cols.into_iter().enumerate() {
            buf.fill(F::zero());
            for c in cols {
                buf[c] = F::one();
            }
            f(row_idx, &buf);
        }
    }

    /// O(T) sparse fold — accumulates `left[row]` into `result[col]` only at
    /// nonzero positions, avoiding the O(T × K) dense iteration.
    fn fold_rows(&self, left: &[F], sigma: usize) -> Vec<F> {
        let num_cols = 1usize << sigma;
        let mut result = crate::thread::unsafe_allocate_zero_vec(num_cols);
        for (cycle, &opt_col) in self.indices.iter().enumerate() {
            if let Some(col) = opt_col {
                let flat = cycle * self.k + col as usize;
                result[flat % num_cols] += left[flat / num_cols];
            }
        }
        result
    }

    #[inline]
    fn is_sparse(&self) -> bool {
        true
    }

    fn for_each_nonzero(&self, f: &mut dyn FnMut(usize, F)) {
        for (cycle, &opt_col) in self.indices.iter().enumerate() {
            if let Some(col) = opt_col {
                f(cycle * self.k + col as usize, F::one());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Polynomial;
    use jolt_field::Fr;
    use num_traits::Zero;
    use rand_chacha::ChaCha20Rng;
    use rand_core::{RngCore, SeedableRng};

    fn make_one_hot(k: usize, indices: &[Option<u8>]) -> OneHotPolynomial {
        OneHotPolynomial::new(k, indices.to_vec())
    }

    fn to_dense<F: Field>(oh: &OneHotPolynomial) -> Polynomial<F> {
        let total = 1usize << oh.num_vars;
        let mut table = vec![F::zero(); total];
        for (row, &opt_col) in oh.indices.iter().enumerate() {
            if let Some(col) = opt_col {
                table[row * oh.k + col as usize] = F::one();
            }
        }
        Polynomial::new(table)
    }

    #[test]
    fn evaluate_matches_dense() {
        let mut rng = ChaCha20Rng::seed_from_u64(1);
        let k = 4;
        let indices: Vec<Option<u8>> = (0..4)
            .map(|_| Some((rng.next_u32() % k as u32) as u8))
            .collect();
        let oh = make_one_hot(k, &indices);
        let dense: Polynomial<Fr> = to_dense(&oh);
        let nv = oh.num_vars();

        for _ in 0..5 {
            let point: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();
            assert_eq!(oh.evaluate(&point), dense.evaluate(&point));
        }
    }

    #[test]
    fn fold_rows_matches_dense() {
        let mut rng = ChaCha20Rng::seed_from_u64(2);
        let k = 8;
        let n_rows = 8;
        let indices: Vec<Option<u8>> = (0..n_rows)
            .map(|i| {
                if i % 3 == 0 {
                    None
                } else {
                    Some((rng.next_u32() % k as u32) as u8)
                }
            })
            .collect();
        let oh = make_one_hot(k, &indices);
        let dense: Polynomial<Fr> = to_dense(&oh);

        let sigma = 3;
        let num_rows_matrix = (1usize << oh.num_vars()) >> sigma;
        let left: Vec<Fr> = (0..num_rows_matrix).map(|_| Fr::random(&mut rng)).collect();

        assert_eq!(oh.fold_rows(&left, sigma), dense.fold_rows(&left, sigma));
    }

    #[test]
    fn for_each_row_matches_dense() {
        let k = 4;
        let indices = vec![Some(0), Some(3), None, Some(1)];
        let oh = make_one_hot(k, &indices);
        let dense: Polynomial<Fr> = to_dense(&oh);

        let sigma = 2;
        let mut oh_rows: Vec<Vec<Fr>> = Vec::new();
        oh.for_each_row(sigma, &mut |_, row: &[Fr]| oh_rows.push(row.to_vec()));

        let mut dense_rows: Vec<Vec<Fr>> = Vec::new();
        dense.for_each_row(sigma, &mut |_, row: &[Fr]| dense_rows.push(row.to_vec()));

        assert_eq!(oh_rows, dense_rows);
    }

    #[test]
    fn for_each_nonzero_yields_correct_entries() {
        let k = 4;
        let indices = vec![Some(2), None, Some(0), Some(3)];
        let oh = make_one_hot(k, &indices);

        let mut entries = Vec::new();
        oh.for_each_nonzero(&mut |idx, val: Fr| entries.push((idx, val)));

        assert_eq!(entries.len(), 3);
        // cycle 0, col 2; cycle 2, col 0; cycle 3, col 3
        assert_eq!(entries[0].0, 2);
        assert_eq!(entries[1].0, 2 * 4);
        assert_eq!(entries[2].0, 3 * 4 + 3);
        assert!(entries.iter().all(|(_, v)| *v == Fr::from_u64(1)));
    }

    #[test]
    fn is_sparse_returns_true() {
        let oh = make_one_hot(4, &[Some(0), Some(1), Some(2), Some(3)]);
        assert!(MultilinearPoly::<Fr>::is_sparse(&oh));
    }

    #[test]
    fn all_none_evaluates_to_zero() {
        let oh = make_one_hot(4, &[None, None, None, None]);
        let mut rng = ChaCha20Rng::seed_from_u64(10);
        let nv = <OneHotPolynomial as MultilinearPoly<Fr>>::num_vars(&oh);
        let point: Vec<Fr> = (0..nv).map(|_| Fr::random(&mut rng)).collect();
        let eval: Fr = oh.evaluate(&point);
        assert!(eval.is_zero());
    }
}
