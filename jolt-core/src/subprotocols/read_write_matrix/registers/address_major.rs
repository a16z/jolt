//! Address-major sparse matrix representation for register read/write checking.
//!
//! This module provides the address-major (column-major) variant for registers,
//! used when binding address variables. It mirrors the structure of
//! [`super::ReadWriteMatrixAddressMajor`] but with three RA coefficient vectors
//! instead of one.
//!
//! # Struct-of-Arrays Layout with Option<F>
//!
//! Uses SoA layout for cache efficiency during address binding. Each field is
//! stored in a separate vector:
//! - `rows`: cycle indices
//! - `cols`: register indices  
//! - `vals`: Val coefficients
//! - `rs1_ras`, `rs2_ras`, `rd_was`: Optional RA coefficients as `Vec<Option<F>>`
//!
//! # Why Option<F> instead of F::zero()
//!
//! We MUST use `Option<F>` because `F::zero()` as a sentinel would INVALIDATE the sumcheck:
//! - `None` means "no access of this type"
//! - `Some(F::zero())` is a valid bound coefficient that could arise during binding
//!
//! # Conversion from Cycle-Major
//!
//! Typically constructed by converting from [`super::RegisterMatrixCycleMajor`]
//! after binding some/all cycle variables.

use super::cycle_major::RegisterMatrixCycleMajor;
use crate::field::{JoltField, OptimizedMul};
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use allocative::Allocative;
use num::Integer;
use rayon::prelude::*;

use common::constants::REGISTER_COUNT;

#[allow(dead_code)]
const K: usize = REGISTER_COUNT as usize;
#[allow(dead_code)]
const LOG_K: usize = REGISTER_COUNT.ilog2() as usize;

/// Address-major (column-major) sparse matrix for register read/write checking.
///
/// Entries are sorted by `(col, row)` where col = register index, row = cycle index.
/// Uses Struct-of-Arrays layout with dense `val_init` and `val_final` arrays.
#[derive(Allocative)]
pub struct RegisterMatrixAddressMajor<F: JoltField> {
    // Sparse entry data (SoA layout)
    /// Row (cycle) indices for each entry.
    pub rows: Vec<usize>,
    /// Column (register) indices for each entry.
    pub cols: Vec<u8>,
    /// Val coefficients for each entry.
    pub vals: Vec<F>,
    /// rs1 RA coefficients. None if rs1 didn't access this (row, col).
    pub rs1_ras: Vec<Option<F>>,
    /// rs2 RA coefficients. None if rs2 didn't access this (row, col).
    pub rs2_ras: Vec<Option<F>>,
    /// rd WA coefficients. None if rd didn't write this (row, col).
    pub rd_was: Vec<Option<F>>,

    // Dense arrays for boundary values
    /// Initial value of each register (before any cycles).
    /// After binding address variables, this becomes eq-weighted.
    pub val_init: MultilinearPolynomial<F>,
    /// Final value of each register (after all cycles).
    /// Size K (halves each address binding round).
    pub val_final: MultilinearPolynomial<F>,

    /// Number of row bits remaining (cycle bits).
    pub(crate) num_row_bits: usize,
    /// Number of column bits remaining (address bits).
    pub(crate) num_col_bits: usize,
}

impl<F: JoltField> RegisterMatrixAddressMajor<F> {
    /// Number of sparse entries.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the matrix is empty.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Current number of row bits.
    pub fn num_row_bits(&self) -> usize {
        self.num_row_bits
    }

    /// Current number of column bits.
    pub fn num_col_bits(&self) -> usize {
        self.num_col_bits
    }

    /// Get the "next value" for entry at index `i`.
    ///
    /// If entry `i+1` is in the same column, returns `vals[i+1]`.
    /// Otherwise returns `val_final[cols[i]]` (value persists after last access).
    #[inline]
    pub fn get_next_val(&self, i: usize) -> F {
        if i + 1 < self.len() && self.cols[i + 1] == self.cols[i] {
            self.vals[i + 1]
        } else {
            self.val_final.get_bound_coeff(self.cols[i] as usize)
        }
    }

    /// Bind one address (column) variable using random challenge `r`.
    ///
    /// Merges entries at adjacent columns (2k and 2k+1) with the same row.
    /// After binding, `num_col_bits()` decreases by 1.
    ///
    /// Uses the **checkpoint pattern** from RAM: tracks the running value in each column
    /// to correctly compute implicit values when only one column has an entry.
    #[tracing::instrument(skip_all, name = "RegisterMatrixAddressMajor::bind")]
    pub fn bind(&mut self, r: F::Challenge) {
        // Group entries by column/2 (adjacent column pairs)
        // Since entries are sorted by (col, row), we can process column pairs
        // NOTE: We process entries BEFORE binding val_init/val_final because
        // we need the original (unbound) checkpoint values.

        let mut new_rows = Vec::with_capacity(self.rows.len());
        let mut new_cols = Vec::with_capacity(self.rows.len());
        let mut new_vals = Vec::with_capacity(self.rows.len());
        let mut new_rs1_ras: Vec<Option<F>> = Vec::with_capacity(self.rows.len());
        let mut new_rs2_ras: Vec<Option<F>> = Vec::with_capacity(self.rows.len());
        let mut new_rd_was: Vec<Option<F>> = Vec::with_capacity(self.rows.len());

        let mut i = 0;
        while i < self.rows.len() {
            let col_pair = self.cols[i] / 2;
            let even_col_idx = (2 * col_pair) as usize;

            // Find all entries in this column pair
            let pair_start = i;
            while i < self.rows.len() && self.cols[i] / 2 == col_pair {
                i += 1;
            }
            let pair_end = i;

            // Split into even and odd columns within this range
            let pivot = (pair_start..pair_end)
                .find(|&idx| self.cols[idx].is_odd())
                .unwrap_or(pair_end);

            let even_range = pair_start..pivot;
            let odd_range = pivot..pair_end;

            // Initialize checkpoints from val_init (BEFORE any binding)
            let mut even_checkpoint = self.val_init.get_bound_coeff(even_col_idx);
            let mut odd_checkpoint = self.val_init.get_bound_coeff(even_col_idx + 1);

            // Two-pointer merge by row with checkpoint tracking
            let mut ei = even_range.start;
            let mut oi = odd_range.start;

            while ei < even_range.end && oi < odd_range.end {
                let row_e = self.rows[ei];
                let row_o = self.rows[oi];

                if row_e == row_o {
                    // Both columns have this row - merge
                    let new_val = self.vals[ei] + r.mul_0_optimized(self.vals[oi] - self.vals[ei]);
                    new_rows.push(row_e);
                    new_cols.push(col_pair);
                    new_vals.push(new_val);
                    new_rs1_ras.push(Self::bind_optional_ra(
                        self.rs1_ras[ei],
                        self.rs1_ras[oi],
                        r,
                    ));
                    new_rs2_ras.push(Self::bind_optional_ra(
                        self.rs2_ras[ei],
                        self.rs2_ras[oi],
                        r,
                    ));
                    new_rd_was.push(Self::bind_optional_ra(self.rd_was[ei], self.rd_was[oi], r));

                    // Update checkpoints
                    even_checkpoint = self.get_next_val(ei);
                    odd_checkpoint = self.get_next_val(oi);
                    ei += 1;
                    oi += 1;
                } else if row_e < row_o {
                    // Only even column has this row - use odd_checkpoint for implicit odd value
                    let val_even = self.vals[ei];
                    let new_val = val_even + r.mul_0_optimized(odd_checkpoint - val_even);
                    new_rows.push(row_e);
                    new_cols.push(col_pair);
                    new_vals.push(new_val);
                    new_rs1_ras.push(self.rs1_ras[ei].map(|ra| (F::one() - r).mul_1_optimized(ra)));
                    new_rs2_ras.push(self.rs2_ras[ei].map(|ra| (F::one() - r).mul_1_optimized(ra)));
                    new_rd_was.push(self.rd_was[ei].map(|wa| (F::one() - r).mul_1_optimized(wa)));

                    // Update even checkpoint
                    even_checkpoint = self.get_next_val(ei);
                    ei += 1;
                } else {
                    // Only odd column has this row - use even_checkpoint for implicit even value
                    let val_odd = self.vals[oi];
                    let new_val = even_checkpoint + r.mul_0_optimized(val_odd - even_checkpoint);
                    new_rows.push(row_o);
                    new_cols.push(col_pair);
                    new_vals.push(new_val);
                    new_rs1_ras.push(self.rs1_ras[oi].map(|ra| r.mul_1_optimized(ra)));
                    new_rs2_ras.push(self.rs2_ras[oi].map(|ra| r.mul_1_optimized(ra)));
                    new_rd_was.push(self.rd_was[oi].map(|wa| r.mul_1_optimized(wa)));

                    // Update odd checkpoint
                    odd_checkpoint = self.get_next_val(oi);
                    oi += 1;
                }
            }

            // Remaining even-only entries
            while ei < even_range.end {
                let row_e = self.rows[ei];
                let val_even = self.vals[ei];
                let new_val = val_even + r.mul_0_optimized(odd_checkpoint - val_even);
                new_rows.push(row_e);
                new_cols.push(col_pair);
                new_vals.push(new_val);
                new_rs1_ras.push(self.rs1_ras[ei].map(|ra| (F::one() - r).mul_1_optimized(ra)));
                new_rs2_ras.push(self.rs2_ras[ei].map(|ra| (F::one() - r).mul_1_optimized(ra)));
                new_rd_was.push(self.rd_was[ei].map(|wa| (F::one() - r).mul_1_optimized(wa)));
                ei += 1;
            }

            // Remaining odd-only entries
            while oi < odd_range.end {
                let row_o = self.rows[oi];
                let val_odd = self.vals[oi];
                let new_val = even_checkpoint + r.mul_0_optimized(val_odd - even_checkpoint);
                new_rows.push(row_o);
                new_cols.push(col_pair);
                new_vals.push(new_val);
                new_rs1_ras.push(self.rs1_ras[oi].map(|ra| r.mul_1_optimized(ra)));
                new_rs2_ras.push(self.rs2_ras[oi].map(|ra| r.mul_1_optimized(ra)));
                new_rd_was.push(self.rd_was[oi].map(|wa| r.mul_1_optimized(wa)));
                oi += 1;
            }
        }

        // Now bind val_init and val_final AFTER processing entries
        self.val_init.bind_parallel(r, BindingOrder::LowToHigh);
        self.val_final.bind_parallel(r, BindingOrder::LowToHigh);

        self.rows = new_rows;
        self.cols = new_cols;
        self.vals = new_vals;
        self.rs1_ras = new_rs1_ras;
        self.rs2_ras = new_rs2_ras;
        self.rd_was = new_rd_was;
        self.num_col_bits -= 1;
    }

    /// Bind optional RA coefficients.
    #[inline]
    fn bind_optional_ra(even: Option<F>, odd: Option<F>, r: F::Challenge) -> Option<F> {
        match (even, odd) {
            (Some(e), Some(o)) => Some(e + r.mul_0_optimized(o - e)),
            (Some(e), None) => Some((F::one() - r).mul_1_optimized(e)),
            (None, Some(o)) => Some(r.mul_1_optimized(o)),
            (None, None) => None,
        }
    }

    /// Materialize into dense polynomial vectors.
    ///
    /// Polynomial indexing: `index = k * T' + j` where:
    /// - `k` is the address (register) index (high-order bits)
    /// - `j` is the cycle index (low-order bits)
    pub fn materialize(
        self,
    ) -> (
        MultilinearPolynomial<F>,
        MultilinearPolynomial<F>,
        MultilinearPolynomial<F>,
        MultilinearPolynomial<F>,
    ) {
        let k_size = 1 << self.num_col_bits;
        let t_size = 1 << self.num_row_bits;
        let total_size = k_size * t_size;

        let mut rs1_ra = vec![F::zero(); total_size];
        let mut rs2_ra = vec![F::zero(); total_size];
        let mut rd_wa = vec![F::zero(); total_size];
        let mut val = vec![F::zero(); total_size];

        // Initialize val with val_init values for each register
        // Index layout: k * t_size + j (address bits high, cycle bits low)
        for k in 0..k_size {
            let init_val = self.val_init.get_bound_coeff(k);
            for j in 0..t_size {
                val[k * t_size + j] = init_val;
            }
        }

        // Fill in sparse entries
        // Note: entries are in address-major order (sorted by col, row)
        for i in 0..self.rows.len() {
            let row = self.rows[i];
            let col = self.cols[i] as usize;
            // Index = k * t_size + j (address bits high, cycle bits low)
            let idx = col * t_size + row;

            if let Some(ra) = self.rs1_ras[i] {
                rs1_ra[idx] = ra;
            }
            if let Some(ra) = self.rs2_ras[i] {
                rs2_ra[idx] = ra;
            }
            if let Some(wa) = self.rd_was[i] {
                rd_wa[idx] = wa;
            }
            val[idx] = self.vals[i];
        }

        (
            MultilinearPolynomial::from(rs1_ra),
            MultilinearPolynomial::from(rs2_ra),
            MultilinearPolynomial::from(rd_wa),
            MultilinearPolynomial::from(val),
        )
    }
}

/// Convert from cycle-major to address-major representation.
impl<F: JoltField> From<RegisterMatrixCycleMajor<F>> for RegisterMatrixAddressMajor<F> {
    #[tracing::instrument(skip_all, name = "RegisterMatrixAddressMajor::from")]
    fn from(cycle_major: RegisterMatrixCycleMajor<F>) -> Self {
        let mut entries = cycle_major.entries;
        let val_init = cycle_major.val_init;
        let num_row_bits = cycle_major.num_row_bits;
        let num_col_bits = cycle_major.num_col_bits;

        // Sort by (col, row) - address-major order
        entries.par_sort_by(|a, b| a.col.cmp(&b.col).then(a.row.cmp(&b.row)));

        // Build SoA arrays
        let rows: Vec<usize> = entries.iter().map(|e| e.row).collect();
        let cols: Vec<u8> = entries.iter().map(|e| e.col).collect();
        let vals: Vec<F> = entries.iter().map(|e| e.val_coeff).collect();

        // Keep Option<F> for RAs - do NOT convert to F::zero()!
        let rs1_ras: Vec<Option<F>> = entries.iter().map(|e| e.rs1_ra).collect();
        let rs2_ras: Vec<Option<F>> = entries.iter().map(|e| e.rs2_ra).collect();
        let rd_was: Vec<Option<F>> = entries.iter().map(|e| e.rd_wa).collect();

        // Compute val_final for each column
        let k_size = 1 << num_col_bits;
        let mut val_final_vec = vec![F::zero(); k_size];

        // val_final[k] = value of register k after all cycles
        // This is the last entry's next_val for each column, or val_init if no entries
        for k in 0..k_size {
            val_final_vec[k] = val_init.get_bound_coeff(k);
        }

        // Find last entry for each column and use its implied final value
        // Since sorted by (col, row), we can find the last entry per column
        let mut i = 0;
        while i < entries.len() {
            let col = entries[i].col as usize;
            // Find last entry for this column
            let mut last_idx = i;
            while i < entries.len() && entries[i].col as usize == col {
                last_idx = i;
                i += 1;
            }
            // The final value is the last entry's next_val
            val_final_vec[col] = F::from_u64(entries[last_idx].next_val);
        }

        Self {
            rows,
            cols,
            vals,
            rs1_ras,
            rs2_ras,
            rd_was,
            val_init,
            val_final: val_final_vec.into(),
            num_row_bits,
            num_col_bits,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::JoltField;
    use ark_bn254::Fr;
    use ark_std::{One, Zero};

    type F = Fr;

    /// Helper to create a RegisterMatrixCycleMajor for testing conversion.
    fn make_cycle_major_for_conversion() -> RegisterMatrixCycleMajor<F> {
        use crate::subprotocols::read_write_matrix::RegisterEntry;

        // Create entries sorted by (row, col)
        let entries = vec![
            RegisterEntry {
                row: 0,
                col: 2,
                prev_val: 0,
                next_val: 100,
                val_coeff: F::from(100u64),
                rs1_ra: Some(F::one()),
                rs2_ra: None,
                rd_wa: None,
            },
            RegisterEntry {
                row: 0,
                col: 5,
                prev_val: 0,
                next_val: 200,
                val_coeff: F::from(200u64),
                rs1_ra: None,
                rs2_ra: Some(F::one()),
                rd_wa: None,
            },
            RegisterEntry {
                row: 1,
                col: 2,
                prev_val: 100,
                next_val: 150,
                val_coeff: F::from(150u64),
                rs1_ra: None,
                rs2_ra: None,
                rd_wa: Some(F::one()),
            },
        ];

        RegisterMatrixCycleMajor {
            entries,
            val_init: vec![F::zero(); K].into(),
            num_row_bits: 1, // 2 rows
            num_col_bits: LOG_K,
        }
    }

    #[test]
    fn test_conversion_from_cycle_major() {
        let cycle_major = make_cycle_major_for_conversion();
        let address_major: RegisterMatrixAddressMajor<F> = cycle_major.into();

        // After conversion, entries should be sorted by (col, row)
        // Original entries: (0,2), (0,5), (1,2)
        // Sorted by (col, row): (0,2), (1,2), (0,5)
        assert_eq!(address_major.len(), 3);

        // First entry: col=2, row=0
        assert_eq!(address_major.cols[0], 2);
        assert_eq!(address_major.rows[0], 0);
        assert_eq!(address_major.rs1_ras[0], Some(F::one()));

        // Second entry: col=2, row=1
        assert_eq!(address_major.cols[1], 2);
        assert_eq!(address_major.rows[1], 1);
        assert_eq!(address_major.rd_was[1], Some(F::one()));

        // Third entry: col=5, row=0
        assert_eq!(address_major.cols[2], 5);
        assert_eq!(address_major.rows[2], 0);
        assert_eq!(address_major.rs2_ras[2], Some(F::one()));
    }

    #[test]
    fn test_get_next_val_same_column() {
        let cycle_major = make_cycle_major_for_conversion();
        let address_major: RegisterMatrixAddressMajor<F> = cycle_major.into();

        // Entry 0 (col=2, row=0) should have next_val = entry 1's val (same column)
        let next_val_0 = address_major.get_next_val(0);
        assert_eq!(next_val_0, address_major.vals[1]);
    }

    #[test]
    fn test_get_next_val_different_column() {
        let cycle_major = make_cycle_major_for_conversion();
        let address_major: RegisterMatrixAddressMajor<F> = cycle_major.into();

        // Entry 1 (col=2, row=1) is last in column 2, should return val_final[2]
        let next_val_1 = address_major.get_next_val(1);
        assert_eq!(next_val_1, address_major.val_final.get_bound_coeff(2));
    }

    #[test]
    fn test_bind_optional_ra_function() {
        // Test the helper function independently using the proper challenge type
        let even = Some(F::from(10u64));
        let odd = Some(F::from(20u64));
        let r: <F as JoltField>::Challenge = 3u128.into();

        let result = RegisterMatrixAddressMajor::<F>::bind_optional_ra(even, odd, r);
        // e + r*(o - e) = 10 + r*(20-10)
        let expected = F::from(10u64) + r * (F::from(20u64) - F::from(10u64));
        assert_eq!(result, Some(expected));

        // Even only: (1-r)*e
        let result_even = RegisterMatrixAddressMajor::<F>::bind_optional_ra(even, None, r);
        let expected_even = (F::one() - r) * F::from(10u64);
        assert_eq!(result_even, Some(expected_even));

        // Odd only: r*o
        let result_odd = RegisterMatrixAddressMajor::<F>::bind_optional_ra(None, odd, r);
        let expected_odd = r * F::from(20u64);
        assert_eq!(result_odd, Some(expected_odd));

        // Both None
        let result_none = RegisterMatrixAddressMajor::<F>::bind_optional_ra(None, None, r);
        assert_eq!(result_none, None);
    }

    #[test]
    fn test_bind_reduces_col_bits() {
        // Create a matrix and verify binding reduces column bits
        let cycle_major = make_cycle_major_for_conversion();
        let mut address_major: RegisterMatrixAddressMajor<F> = cycle_major.into();

        assert_eq!(address_major.num_col_bits(), LOG_K);

        let r: <F as JoltField>::Challenge = 5u128.into();
        address_major.bind(r);

        assert_eq!(address_major.num_col_bits(), LOG_K - 1);
        // val_init and val_final should also be halved
        assert_eq!(address_major.val_init.len(), K / 2);
        assert_eq!(address_major.val_final.len(), K / 2);
    }

    #[test]
    fn test_materialize() {
        // Test materialization produces correct dense polynomials
        let cycle_major = make_cycle_major_for_conversion();
        let address_major: RegisterMatrixAddressMajor<F> = cycle_major.into();

        let (rs1_ra, rs2_ra, rd_wa, val) = address_major.materialize();

        let k_size = 1 << LOG_K;
        let t_size = 2; // 2 rows

        assert_eq!(rs1_ra.len(), k_size * t_size);
        assert_eq!(rs2_ra.len(), k_size * t_size);
        assert_eq!(rd_wa.len(), k_size * t_size);
        assert_eq!(val.len(), k_size * t_size);

        // Check specific entries
        // Entry at (row=0, col=2): idx = 2*2 + 0 = 4
        assert_eq!(rs1_ra.get_bound_coeff(4), F::one());

        // Entry at (row=0, col=5): idx = 5*2 + 0 = 10
        assert_eq!(rs2_ra.get_bound_coeff(10), F::one());

        // Entry at (row=1, col=2): idx = 2*2 + 1 = 5
        assert_eq!(rd_wa.get_bound_coeff(5), F::one());
    }
}
