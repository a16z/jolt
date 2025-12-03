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

use super::cycle_major_registers::RegisterMatrixCycleMajor;
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
    #[tracing::instrument(skip_all, name = "RegisterMatrixAddressMajor::bind")]
    pub fn bind(&mut self, r: F::Challenge) {
        // Update val_init and val_final
        self.val_init.bind_parallel(r, BindingOrder::LowToHigh);
        self.val_final.bind_parallel(r, BindingOrder::LowToHigh);

        // Group entries by column/2 (adjacent column pairs)
        // Since entries are sorted by (col, row), we can process column pairs
        
        let mut new_rows = Vec::with_capacity(self.rows.len());
        let mut new_cols = Vec::with_capacity(self.rows.len());
        let mut new_vals = Vec::with_capacity(self.rows.len());
        let mut new_rs1_ras: Vec<Option<F>> = Vec::with_capacity(self.rows.len());
        let mut new_rs2_ras: Vec<Option<F>> = Vec::with_capacity(self.rows.len());
        let mut new_rd_was: Vec<Option<F>> = Vec::with_capacity(self.rows.len());

        let mut i = 0;
        while i < self.rows.len() {
            let col_pair = self.cols[i] / 2;
            
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

            // Two-pointer merge by row
            let mut ei = even_range.start;
            let mut oi = odd_range.start;

            while ei < even_range.end || oi < odd_range.end {
                let even_row = (ei < even_range.end).then(|| self.rows[ei]);
                let odd_row = (oi < odd_range.end).then(|| self.rows[oi]);

                match (even_row, odd_row) {
                    (Some(er), Some(or)) if er == or => {
                        // Both columns have this row - merge
                        Self::bind_entry_pair(
                            Some(ei), Some(oi), r, col_pair,
                            &self.rows, &self.cols, &self.vals,
                            &self.rs1_ras, &self.rs2_ras, &self.rd_was,
                            &self.val_init,
                            &mut new_rows, &mut new_cols, &mut new_vals,
                            &mut new_rs1_ras, &mut new_rs2_ras, &mut new_rd_was,
                        );
                        ei += 1;
                        oi += 1;
                    }
                    (Some(er), Some(or)) if er < or => {
                        // Only even column has this row
                        Self::bind_entry_pair(
                            Some(ei), None, r, col_pair,
                            &self.rows, &self.cols, &self.vals,
                            &self.rs1_ras, &self.rs2_ras, &self.rd_was,
                            &self.val_init,
                            &mut new_rows, &mut new_cols, &mut new_vals,
                            &mut new_rs1_ras, &mut new_rs2_ras, &mut new_rd_was,
                        );
                        ei += 1;
                    }
                    (Some(_), Some(_)) => {
                        // Only odd column has this row
                        Self::bind_entry_pair(
                            None, Some(oi), r, col_pair,
                            &self.rows, &self.cols, &self.vals,
                            &self.rs1_ras, &self.rs2_ras, &self.rd_was,
                            &self.val_init,
                            &mut new_rows, &mut new_cols, &mut new_vals,
                            &mut new_rs1_ras, &mut new_rs2_ras, &mut new_rd_was,
                        );
                        oi += 1;
                    }
                    (Some(_), None) => {
                        Self::bind_entry_pair(
                            Some(ei), None, r, col_pair,
                            &self.rows, &self.cols, &self.vals,
                            &self.rs1_ras, &self.rs2_ras, &self.rd_was,
                            &self.val_init,
                            &mut new_rows, &mut new_cols, &mut new_vals,
                            &mut new_rs1_ras, &mut new_rs2_ras, &mut new_rd_was,
                        );
                        ei += 1;
                    }
                    (None, Some(_)) => {
                        Self::bind_entry_pair(
                            None, Some(oi), r, col_pair,
                            &self.rows, &self.cols, &self.vals,
                            &self.rs1_ras, &self.rs2_ras, &self.rd_was,
                            &self.val_init,
                            &mut new_rows, &mut new_cols, &mut new_vals,
                            &mut new_rs1_ras, &mut new_rs2_ras, &mut new_rd_was,
                        );
                        oi += 1;
                    }
                    (None, None) => break,
                }
            }
        }

        self.rows = new_rows;
        self.cols = new_cols;
        self.vals = new_vals;
        self.rs1_ras = new_rs1_ras;
        self.rs2_ras = new_rs2_ras;
        self.rd_was = new_rd_was;
        self.num_col_bits -= 1;
    }

    /// Bind a pair of entries at the same row from adjacent columns.
    #[allow(clippy::too_many_arguments)]
    fn bind_entry_pair(
        even_idx: Option<usize>,
        odd_idx: Option<usize>,
        r: F::Challenge,
        new_col: u8,
        rows: &[usize],
        cols: &[u8],
        vals: &[F],
        rs1_ras: &[Option<F>],
        rs2_ras: &[Option<F>],
        rd_was: &[Option<F>],
        val_init: &MultilinearPolynomial<F>,
        out_rows: &mut Vec<usize>,
        out_cols: &mut Vec<u8>,
        out_vals: &mut Vec<F>,
        out_rs1_ras: &mut Vec<Option<F>>,
        out_rs2_ras: &mut Vec<Option<F>>,
        out_rd_was: &mut Vec<Option<F>>,
    ) {
        match (even_idx, odd_idx) {
            (Some(ei), Some(oi)) => {
                // Both columns have this row
                let row = rows[ei];
                let even_val = vals[ei];
                let odd_val = vals[oi];
                let new_val = even_val + r.mul_0_optimized(odd_val - even_val);

                out_rows.push(row);
                out_cols.push(new_col);
                out_vals.push(new_val);
                out_rs1_ras.push(Self::bind_optional_ra(rs1_ras[ei], rs1_ras[oi], r));
                out_rs2_ras.push(Self::bind_optional_ra(rs2_ras[ei], rs2_ras[oi], r));
                out_rd_was.push(Self::bind_optional_ra(rd_was[ei], rd_was[oi], r));
            }
            (Some(ei), None) => {
                // Only even column has this row
                // Implicit odd entry: val = val_init[odd_col], all RAs = None
                let row = rows[ei];
                let even_col = cols[ei];
                let odd_col = even_col + 1;
                let even_val = vals[ei];
                let implicit_odd_val = val_init.get_bound_coeff(odd_col as usize);
                let new_val = even_val + r.mul_0_optimized(implicit_odd_val - even_val);

                out_rows.push(row);
                out_cols.push(new_col);
                out_vals.push(new_val);
                out_rs1_ras.push(rs1_ras[ei].map(|ra| (F::one() - r).mul_1_optimized(ra)));
                out_rs2_ras.push(rs2_ras[ei].map(|ra| (F::one() - r).mul_1_optimized(ra)));
                out_rd_was.push(rd_was[ei].map(|wa| (F::one() - r).mul_1_optimized(wa)));
            }
            (None, Some(oi)) => {
                // Only odd column has this row
                // Implicit even entry: val = val_init[even_col], all RAs = None
                let row = rows[oi];
                let odd_col = cols[oi];
                let even_col = odd_col - 1;
                let odd_val = vals[oi];
                let implicit_even_val = val_init.get_bound_coeff(even_col as usize);
                let new_val = implicit_even_val + r.mul_0_optimized(odd_val - implicit_even_val);

                out_rows.push(row);
                out_cols.push(new_col);
                out_vals.push(new_val);
                out_rs1_ras.push(rs1_ras[oi].map(|ra| r.mul_1_optimized(ra)));
                out_rs2_ras.push(rs2_ras[oi].map(|ra| r.mul_1_optimized(ra)));
                out_rd_was.push(rd_was[oi].map(|wa| r.mul_1_optimized(wa)));
            }
            (None, None) => {
                panic!("bind_entry_pair called with both indices None")
            }
        }
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
    pub fn materialize(self) -> (MultilinearPolynomial<F>, MultilinearPolynomial<F>, MultilinearPolynomial<F>, MultilinearPolynomial<F>) {
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
        entries.par_sort_by(|a, b| {
            a.col.cmp(&b.col).then(a.row.cmp(&b.row))
        });

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
    // TODO: Add tests for conversion and address binding
}
