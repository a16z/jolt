//! Cycle-major sparse matrix representation for register read/write checking.
//!
//! This module provides a specialized implementation for registers that differs from
//! the RAM implementation ([`super::ReadWriteMatrixCycleMajor`]) in key ways:
//!
//! - **Multiple access types per entry**: Each entry can have up to 3 RA coefficients
//!   (rs1_ra, rs2_ra, rd_wa) stored as Options, allowing a single entry to represent
//!   multiple accesses to the same register in the same cycle.
//!
//! - **Shared Val**: When rs1, rs2, and rd all access the same register in a cycle,
//!   they share a single `val_coeff`, ensuring consistency during binding.
//!
//! - **Simpler binding**: No need for two-level merge by (col, access_type) since
//!   all access types are bundled in one entry.

use crate::field::{JoltField, OptimizedMul};
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::subprotocols::read_write_matrix::merge_utils::{bind_optional, linear_interpolate};
use crate::utils::math::Math;
use allocative::Allocative;
use num::Integer;
use rayon::prelude::*;
use tracer::instruction::Cycle;

use common::constants::REGISTER_COUNT;

const K: usize = REGISTER_COUNT as usize;
const LOG_K: usize = REGISTER_COUNT.ilog2() as usize;

/// A single entry in the register cycle-major sparse matrix.
///
/// Unlike the RAM version which has one `ra_coeff` per entry, this struct
/// bundles all three access types (rs1, rs2, rd) into optional RA coefficients.
/// This allows entries at the same `(row, col)` with different access types to
/// share a single `val_coeff`.
///
/// # Important: Option<F> vs F::zero()
///
/// We use `Option<F>` for RA coefficients rather than storing `F::zero()` because:
/// - `None` means "no access of this type occurred"
/// - `Some(F::zero())` could arise from binding and would be semantically different
/// - Using F::zero() as a sentinel value would INVALIDATE the sumcheck
#[derive(Allocative, Debug, Clone, Copy)]
pub struct RegisterEntry<F: JoltField> {
    /// The row (cycle) index.
    pub row: usize,
    /// The column (register) index, 0-127.
    pub col: u8,
    /// Value of this register before any access at this cycle.
    /// Used when an implicit even entry is needed during binding.
    pub(crate) prev_val: u64,
    /// Value of this register after all accesses at this cycle.
    /// Used when an implicit odd entry is needed during binding.
    pub(crate) next_val: u64,
    /// The bound Val coefficient for this register at this cycle.
    pub val_coeff: F,
    /// RA coefficient for rs1 access. None if rs1 didn't access this register at this cycle.
    pub rs1_ra: Option<F>,
    /// RA coefficient for rs2 access. None if rs2 didn't access this register at this cycle.
    pub rs2_ra: Option<F>,
    /// WA coefficient for rd access. None if rd didn't write this register at this cycle.
    pub rd_wa: Option<F>,
}

impl<F: JoltField> RegisterEntry<F> {
    /// Convert the raw `next_val` to a field element.
    #[inline]
    pub fn next_val_as_field(&self) -> F {
        F::from_u64(self.next_val)
    }

    /// Convert the raw `prev_val` to a field element.
    #[inline]
    pub fn prev_val_as_field(&self) -> F {
        F::from_u64(self.prev_val)
    }
}

/// Cycle-major sparse matrix for register read/write checking.
///
/// Entries are sorted by `(row, col)` where row = cycle index, col = register index.
/// Each entry represents all accesses to a particular register at a particular cycle.
#[derive(Allocative)]
pub struct RegisterMatrixCycleMajor<F: JoltField> {
    /// The sparse entries, sorted by (row, col).
    pub entries: Vec<RegisterEntry<F>>,
    /// Initial value of each register (all zeros for registers).
    /// Used for implicit entry computation during binding.
    pub(crate) val_init: MultilinearPolynomial<F>,
    /// Number of row bits remaining (log2 of current row dimension).
    pub(crate) num_row_bits: usize,
    /// Number of column bits (always LOG_K for registers).
    pub(crate) num_col_bits: usize,
}

impl<F: JoltField> RegisterMatrixCycleMajor<F> {
    /// Construct a RegisterMatrixCycleMajor from an execution trace.
    ///
    /// For each cycle, creates entries for each unique register accessed by rs1, rs2, or rd.
    /// When multiple access types hit the same register, they're combined into one entry.
    #[tracing::instrument(skip_all, name = "RegisterMatrixCycleMajor::from_trace")]
    pub fn from_trace(trace: &[Cycle]) -> Self {
        let t_size = trace.len();

        // Build entries: for each cycle, identify unique registers and their access types.
        // We iterate cycles in order, and within each cycle we will sort registers by index,
        // so the resulting entries vector is already sorted by (row, col) and does not need
        // any global sort.
        let entries: Vec<RegisterEntry<F>> = trace
            .par_iter()
            .enumerate()
            .flat_map_iter(|(j, cycle)| {
                let (rs1_reg, rs1_val) = cycle.rs1_read();
                let (rs2_reg, rs2_val) = cycle.rs2_read();
                let (rd_reg, rd_pre_val, rd_post_val) = cycle.rd_write();

                // Collect unique registers accessed in this cycle
                // Format: (reg, rs1_ra, rs2_ra, rd_wa, prev_val, next_val)
                type RegAccess<F> = (u8, Option<F>, Option<F>, Option<F>, u64, u64);
                let mut regs: Vec<RegAccess<F>> = Vec::with_capacity(3);

                // Helper to add or merge access for a register within this cycle
                let mut add_access =
                    |reg: u8, is_rs1: bool, is_rs2: bool, is_rd: bool, val: u64, next: u64| {
                        if let Some(entry) = regs.iter_mut().find(|(r, _, _, _, _, _)| *r == reg) {
                            if is_rs1 {
                                entry.1 = Some(F::one());
                            }
                            if is_rs2 {
                                entry.2 = Some(F::one());
                            }
                            if is_rd {
                                entry.3 = Some(F::one());
                            }
                            // Update next_val if this is a write: post-write value at this cycle
                            if is_rd {
                                entry.5 = next;
                            }
                        } else {
                            regs.push((
                                reg,
                                if is_rs1 { Some(F::one()) } else { None },
                                if is_rs2 { Some(F::one()) } else { None },
                                if is_rd { Some(F::one()) } else { None },
                                val,  // prev_val (value before this cycle)
                                next, // next_val (value after this cycle)
                            ));
                        }
                    };

                // Add rs1 access (read-only: prev == next == value at this cycle)
                add_access(rs1_reg, true, false, false, rs1_val, rs1_val);
                // Add rs2 access
                add_access(rs2_reg, false, true, false, rs2_val, rs2_val);
                // Add rd access (write)
                add_access(rd_reg, false, false, true, rd_pre_val, rd_post_val);

                // Ensure registers are in increasing order for this cycle so that the
                // global entries vector is sorted by (row, col) without a global sort.
                regs.sort_unstable_by(|(reg_a, _, _, _, _, _), (reg_b, _, _, _, _, _)| {
                    reg_a.cmp(reg_b)
                });

                regs.into_iter()
                    .map(
                        move |(reg, rs1_ra, rs2_ra, rd_wa, prev, next)| RegisterEntry {
                            row: j,
                            col: reg,
                            prev_val: prev,
                            next_val: next,
                            // val_coeff stores the value *before* any access at this cycle.
                            val_coeff: F::from_u64(prev),
                            rs1_ra,
                            rs2_ra,
                            rd_wa,
                        },
                    )
            })
            .collect();

        // Initial values for all registers (all zeros)
        let val_init: Vec<F> = vec![F::zero(); K];

        Self {
            entries,
            val_init: val_init.into(),
            num_row_bits: t_size.log_2(),
            num_col_bits: LOG_K,
        }
    }

    /// Bind a cycle variable using the random challenge `r`.
    ///
    /// This merges entries at adjacent rows (2k and 2k+1) with the same column.
    /// Val coefficients are interpolated, and each RA coefficient binds independently.
    #[tracing::instrument(skip_all, name = "RegisterMatrixCycleMajor::bind")]
    pub fn bind(&mut self, r: F::Challenge) {
        // Group entries by row/2 (adjacent row pairs) and bind them sequentially.
        // We preallocate a single output buffer and reuse it for all row pairs to
        // avoid many small allocations.
        let mut bound_entries: Vec<RegisterEntry<F>> = Vec::with_capacity(self.entries.len());

        for pair in self.entries.chunk_by(|a, b| a.row / 2 == b.row / 2) {
            let pivot = pair.partition_point(|e| e.row.is_even());
            let (even_row, odd_row) = pair.split_at(pivot);
            Self::bind_rows(even_row, odd_row, r, &mut bound_entries);
        }

        self.entries = bound_entries;
        self.num_row_bits -= 1;
    }

    /// Bind adjacent rows and append results to output.
    ///
    /// Uses `TwoPointerMerge` to iterate over column-aligned entry pairs.
    fn bind_rows(
        even: &[RegisterEntry<F>],
        odd: &[RegisterEntry<F>],
        r: F::Challenge,
        out: &mut Vec<RegisterEntry<F>>,
    ) {
        let new_row = even
            .first()
            .map(|e| e.row / 2)
            .or_else(|| odd.first().map(|o| o.row / 2))
            .unwrap_or(0);

        // Manual two-pointer merge by column. Both `even` and `odd` are individually
        // sorted by `col`, so we can avoid constructing a `TwoPointerMerge` iterator
        // for each row pair, which significantly reduces overhead when Phase 1
        // binds all cycle variables.
        let mut ei = 0;
        let mut oi = 0;

        while ei < even.len() || oi < odd.len() {
            if ei < even.len() && oi < odd.len() {
                let e = &even[ei];
                let o = &odd[oi];
                if e.col == o.col {
                    out.push(Self::bind_entry_both(e, o, r, new_row));
                    ei += 1;
                    oi += 1;
                } else if e.col < o.col {
                    out.push(Self::bind_entry_even_only(e, r, new_row));
                    ei += 1;
                } else {
                    out.push(Self::bind_entry_odd_only(o, r, new_row));
                    oi += 1;
                }
            } else if ei < even.len() {
                let e = &even[ei];
                out.push(Self::bind_entry_even_only(e, r, new_row));
                ei += 1;
            } else {
                let o = &odd[oi];
                out.push(Self::bind_entry_odd_only(o, r, new_row));
                oi += 1;
            }
        }
    }

    /// Bind two entries at the same column (both explicit).
    #[inline]
    fn bind_entry_both(
        e: &RegisterEntry<F>,
        o: &RegisterEntry<F>,
        r: F::Challenge,
        new_row: usize,
    ) -> RegisterEntry<F> {
        debug_assert_eq!(e.col, o.col);
        RegisterEntry {
            row: new_row,
            col: e.col,
            prev_val: e.prev_val,
            next_val: o.next_val,
            val_coeff: linear_interpolate(e.val_coeff, o.val_coeff, r),
            rs1_ra: bind_optional(e.rs1_ra, o.rs1_ra, r),
            rs2_ra: bind_optional(e.rs2_ra, o.rs2_ra, r),
            rd_wa: bind_optional(e.rd_wa, o.rd_wa, r),
        }
    }

    /// Bind an even entry with implicit odd (val = next_val, all RAs = None).
    #[inline]
    fn bind_entry_even_only(
        e: &RegisterEntry<F>,
        r: F::Challenge,
        new_row: usize,
    ) -> RegisterEntry<F> {
        let implicit_odd_val = F::from_u64(e.next_val);
        let one_minus_r = F::one() - r;
        RegisterEntry {
            row: new_row,
            col: e.col,
            prev_val: e.prev_val,
            next_val: e.next_val,
            val_coeff: linear_interpolate(e.val_coeff, implicit_odd_val, r),
            rs1_ra: e.rs1_ra.map(|ra| one_minus_r.mul_1_optimized(ra)),
            rs2_ra: e.rs2_ra.map(|ra| one_minus_r.mul_1_optimized(ra)),
            rd_wa: e.rd_wa.map(|ra| one_minus_r.mul_1_optimized(ra)),
        }
    }

    /// Bind an odd entry with implicit even (val = prev_val, all RAs = None).
    #[inline]
    fn bind_entry_odd_only(
        o: &RegisterEntry<F>,
        r: F::Challenge,
        new_row: usize,
    ) -> RegisterEntry<F> {
        let implicit_even_val = F::from_u64(o.prev_val);
        let r_f: F = F::one() * r;
        RegisterEntry {
            row: new_row,
            col: o.col,
            prev_val: o.prev_val,
            next_val: o.next_val,
            val_coeff: linear_interpolate(implicit_even_val, o.val_coeff, r),
            rs1_ra: o.rs1_ra.map(|ra| r_f.mul_1_optimized(ra)),
            rs2_ra: o.rs2_ra.map(|ra| r_f.mul_1_optimized(ra)),
            rd_wa: o.rd_wa.map(|ra| r_f.mul_1_optimized(ra)),
        }
    }

    /// Number of sparse entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the matrix is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Current number of row bits.
    pub fn num_row_bits(&self) -> usize {
        self.num_row_bits
    }

    /// Current number of column bits.
    pub fn num_col_bits(&self) -> usize {
        self.num_col_bits
    }

    /// Convert to address-major representation.
    pub fn into_address_major(self) -> super::RegisterMatrixAddressMajor<F> {
        super::RegisterMatrixAddressMajor::from(self)
    }

    /// Materialize the sparse matrix into dense polynomial vectors.
    ///
    /// Creates four dense vectors of size `2^(num_row_bits + num_col_bits)`:
    /// - rs1_ra: rs1 read access indicator
    /// - rs2_ra: rs2 read access indicator  
    /// - rd_wa: rd write access indicator
    /// - val: register values
    ///
    /// Polynomial indexing: `index = k * T' + j` where:
    /// - `k` is the address (register) index (high-order bits)
    /// - `j` is the cycle index (low-order bits)
    ///
    /// This layout enables LowToHigh binding to bind cycle variables first.
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
        for entry in &self.entries {
            // Index = k * t_size + j (address bits high, cycle bits low)
            let idx = (entry.col as usize) * t_size + entry.row;

            if let Some(ra) = entry.rs1_ra {
                rs1_ra[idx] = ra;
            }
            if let Some(ra) = entry.rs2_ra {
                rs2_ra[idx] = ra;
            }
            if let Some(wa) = entry.rd_wa {
                rd_wa[idx] = wa;
            }
            val[idx] = entry.val_coeff;
        }

        (
            MultilinearPolynomial::from(rs1_ra),
            MultilinearPolynomial::from(rs2_ra),
            MultilinearPolynomial::from(rd_wa),
            MultilinearPolynomial::from(val),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::JoltField;
    use ark_bn254::Fr;
    use ark_std::{One, Zero};

    type F = Fr;

    /// Helper to create a test entry with specified values.
    fn make_entry(
        row: usize,
        col: u8,
        prev_val: u64,
        next_val: u64,
        val_coeff: F,
        rs1_ra: Option<F>,
        rs2_ra: Option<F>,
        rd_wa: Option<F>,
    ) -> RegisterEntry<F> {
        RegisterEntry {
            row,
            col,
            prev_val,
            next_val,
            val_coeff,
            rs1_ra,
            rs2_ra,
            rd_wa,
        }
    }

    #[test]
    fn test_bind_entry_both() {
        // Test binding when both even and odd entries exist at the same column
        let even = make_entry(
            0,
            5,
            10,
            20,
            F::from(10u64),
            Some(F::one()),
            None,
            Some(F::one()),
        );
        let odd = make_entry(1, 5, 20, 30, F::from(20u64), None, Some(F::one()), None);
        let r: <F as JoltField>::Challenge = 2u128.into();

        let result = RegisterMatrixCycleMajor::bind_entry_both(&even, &odd, r, 0);

        assert_eq!(result.row, 0);
        assert_eq!(result.col, 5);
        assert_eq!(result.prev_val, 10);
        assert_eq!(result.next_val, 30);
        // val_coeff: e + r*(o - e) = 10 + r*(20-10)
        let expected_val = F::from(10u64) + r * (F::from(20u64) - F::from(10u64));
        assert_eq!(result.val_coeff, expected_val);
    }

    #[test]
    fn test_bind_entry_even_only() {
        // Test binding when only even entry exists
        let even = make_entry(0, 5, 10, 20, F::from(10u64), Some(F::one()), None, None);
        let r: <F as JoltField>::Challenge = 2u128.into();

        let result = RegisterMatrixCycleMajor::bind_entry_even_only(&even, r, 0);

        assert_eq!(result.row, 0);
        assert_eq!(result.col, 5);
        // implicit odd val = next_val = 20
        // val_coeff: e + r*(implicit_odd - e) = 10 + r*(20-10)
        let expected_val = F::from(10u64) + r * (F::from(20u64) - F::from(10u64));
        assert_eq!(result.val_coeff, expected_val);
        // rs1_ra: (1-r)*1
        assert!(result.rs1_ra.is_some());
    }

    #[test]
    fn test_bind_entry_odd_only() {
        // Test binding when only odd entry exists
        let odd = make_entry(1, 5, 10, 20, F::from(20u64), None, Some(F::one()), None);
        let r: <F as JoltField>::Challenge = 2u128.into();

        let result = RegisterMatrixCycleMajor::bind_entry_odd_only(&odd, r, 0);

        assert_eq!(result.row, 0);
        assert_eq!(result.col, 5);
        // implicit even val = prev_val = 10
        // val_coeff: implicit_even + r*(o - implicit_even) = 10 + r*(20-10)
        let expected_val = F::from(10u64) + r * (F::from(20u64) - F::from(10u64));
        assert_eq!(result.val_coeff, expected_val);
        // rs2_ra: r*1
        let expected_rs2 = r * F::one();
        assert_eq!(result.rs2_ra, Some(expected_rs2));
    }

    #[test]
    fn test_materialize_simple() {
        // Create a simple matrix with known entries and verify materialization
        let entries = vec![
            make_entry(0, 1, 0, 100, F::from(100u64), Some(F::one()), None, None),
            make_entry(1, 2, 0, 200, F::from(200u64), None, Some(F::one()), None),
        ];

        let matrix = RegisterMatrixCycleMajor {
            entries,
            val_init: vec![F::zero(); K].into(),
            num_row_bits: 1, // 2 rows
            num_col_bits: LOG_K,
        };

        let (rs1_ra, rs2_ra, rd_wa, val) = matrix.materialize();

        // Index = k * t_size + j (address bits high, cycle bits low)
        // t_size = 2
        // For entry at (row=0, col=1): idx = 1*2 + 0 = 2
        assert_eq!(rs1_ra.get_bound_coeff(2), F::one());
        assert_eq!(val.get_bound_coeff(2), F::from(100u64));

        // For entry at (row=1, col=2): idx = 2*2 + 1 = 5
        assert_eq!(rs2_ra.get_bound_coeff(5), F::one());
        assert_eq!(val.get_bound_coeff(5), F::from(200u64));

        // rd_wa should be zero for these entries
        assert_eq!(rd_wa.get_bound_coeff(2), F::zero());
        assert_eq!(rd_wa.get_bound_coeff(5), F::zero());
    }

    #[test]
    fn test_bind_reduces_row_bits() {
        // Create a matrix with 4 rows (2 bits) and verify binding reduces to 2 rows (1 bit)
        let entries = vec![
            make_entry(0, 1, 0, 10, F::from(10u64), Some(F::one()), None, None),
            make_entry(2, 1, 10, 20, F::from(20u64), None, Some(F::one()), None),
        ];

        let mut matrix = RegisterMatrixCycleMajor {
            entries,
            val_init: vec![F::zero(); K].into(),
            num_row_bits: 2, // 4 rows
            num_col_bits: LOG_K,
        };

        assert_eq!(matrix.num_row_bits(), 2);

        let r: <F as JoltField>::Challenge = 3u128.into();
        matrix.bind(r);

        assert_eq!(matrix.num_row_bits(), 1);
        // After binding, rows 0,1 -> row 0 and rows 2,3 -> row 1
        // Original entries at rows 0 and 2 become entries at rows 0 and 1
        assert_eq!(matrix.entries.len(), 2);
        assert_eq!(matrix.entries[0].row, 0);
        assert_eq!(matrix.entries[1].row, 1);
    }
}
