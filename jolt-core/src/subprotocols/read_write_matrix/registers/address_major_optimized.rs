//! Memory-optimized address-major sparse matrix for registers.
//!
//! This implementation is optimized for the **address-first** binding configuration
//! (phase1 = 0, phase2 = LOG_K), where we bind all address variables before any cycle variables.
//!
//! # Memory Optimization: Compact Register Indices
//!
//! Instead of storing `Option<F>` (~33 bytes) per entry for each of `rs1_ra`, `rs2_ra`, `rd_wa`,
//! we store a single `u8` per entry indicating which register was accessed:
//!
//! ```text
//! Before optimization:
//!   rs1_ra[i] = Some(F::one()) if rs1 accessed register k, None otherwise
//!
//! After optimization:
//!   rs1_ras[i] = k (register index 0-127) if rs1 accessed register k
//!   rs1_ras[i] = u8::MAX if no rs1 access
//!   + single ExpandingTable A that tracks eq(k, r) for all registers
//! ```
//!
//! # How It Works
//!
//! The `ExpandingTable` starts with K=128 entries, one per register. As we bind address
//! variables, it tracks the eq polynomial evaluations:
//! - Initially: A[k] = 1 for all k
//! - After binding r₀: A[k/2] = (1-r₀) * A_old[2k] + r₀ * A_old[2k+1]
//! - After all LOG_K rounds: A[0] = eq(r, original_register)
//!
//! To get the RA coefficient for entry i:
//! - If rs1_ras[i] == u8::MAX: return 0 (no access)
//! - Else: return A[rs1_ras[i]] (the eq-weighted coefficient)
//!
//! # Memory Savings
//!
//! - Before: 3 × Option<F> = 3 × 33 bytes = 99 bytes per entry
//! - After: 3 × u8 = 3 bytes per entry + small ExpandingTable
//! - **~97% reduction** in ra/wa storage!
//!
//! # Limitations
//!
//! This optimization only works when:
//! - We start with address-first binding (phase1 = 0)
//! - Initial RA coefficients are 0 or 1 (not arbitrary field elements)

use allocative::Allocative;
use common::constants::REGISTER_COUNT;
use num::Integer;
use rayon::prelude::*;
use tracer::instruction::Cycle;

use crate::field::JoltField;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::subprotocols::read_write_matrix::merge_utils::{
    linear_interpolate, MergeItem, TwoPointerMerge,
};
use crate::utils::expanding_table::ExpandingTable;
use crate::utils::thread::unsafe_allocate_zero_vec;

use super::cycle_major::RegisterMatrixCycleMajor;

const K: usize = REGISTER_COUNT as usize;
const LOG_K: usize = REGISTER_COUNT.ilog2() as usize;

/// Sentinel value indicating "no access" for register indices.
/// Valid register indices are 0-127, so 255 is safe to use as sentinel.
const NO_ACCESS: u8 = u8::MAX;

/// Chunk type returned when merging a single column pair in the optimized matrix.
type OptimizedColumnPairChunk<F> = (Vec<usize>, Vec<u8>, Vec<F>, Vec<u8>, Vec<u8>, Vec<u8>);

/// Memory-optimized address-major sparse matrix for registers.
///
/// Uses compact `u8` register indices instead of `Option<F>` for RA coefficients,
/// with an ExpandingTable that grows as we bind address variables.
///
/// # Key Insight: rs1_ras stores ORIGINAL register index
///
/// ```text
/// Entry i accessed register k originally.
/// rs1_ras[i] = k (NEVER changes during binding)
///
/// As we bind address variables r₀, r₁, ...:
///   - ExpandingTable A grows: 1 → 2 → 4 → ... → K
///   - After all LOG_K rounds: A[k] = eq(k, r_address)
///   - get_rs1_ra(i) = A[rs1_ras[i]]
/// ```
#[derive(Allocative, Debug, Clone)]
pub struct RegisterMatrixAddressMajorOptimized<F: JoltField> {
    /// Row (cycle) indices for each sparse entry.
    pub rows: Vec<usize>,
    /// Column (register) indices for each sparse entry.
    /// NOTE: This shrinks as we merge column pairs (128 → 64 → ... → 1).
    pub cols: Vec<u8>,
    /// Val coefficients for each entry.
    pub vals: Vec<F>,
    /// rs1 RA: ORIGINAL register index that rs1 accessed.
    /// `NO_ACCESS` (u8::MAX) if rs1 didn't access this entry.
    /// NOTE: This NEVER changes during binding - we always store the original k.
    pub rs1_ras: Vec<u8>,
    /// rs2 RA: ORIGINAL register index that rs2 accessed.
    pub rs2_ras: Vec<u8>,
    /// rd WA: ORIGINAL register index that rd wrote.
    pub rd_was: Vec<u8>,
    /// ExpandingTable tracking eq(x, r) as we bind address variables.
    /// Grows from 1 entry to K=128 entries.
    /// After all LOG_K rounds: A[k] = eq(k, r_address).
    pub eq_table: ExpandingTable<F>,
    /// Initial value of each register (before any cycles).
    pub val_init: MultilinearPolynomial<F>,
    /// Final value of each register (after all cycles).
    pub val_final: MultilinearPolynomial<F>,
    /// Number of row bits remaining (cycle bits).
    pub(crate) num_row_bits: usize,
    /// Number of column bits remaining (address bits).
    pub(crate) num_col_bits: usize,
}

impl<F: JoltField> Default for RegisterMatrixAddressMajorOptimized<F> {
    fn default() -> Self {
        // Initialize ExpandingTable with capacity K, starts at length 1 with value 1
        let mut eq_table = ExpandingTable::new(K, BindingOrder::LowToHigh);
        eq_table.reset(F::one());
        Self {
            rows: Vec::new(),
            cols: Vec::new(),
            vals: Vec::new(),
            rs1_ras: Vec::new(),
            rs2_ras: Vec::new(),
            rd_was: Vec::new(),
            eq_table,
            val_init: MultilinearPolynomial::default(),
            val_final: MultilinearPolynomial::default(),
            num_row_bits: 0,
            num_col_bits: LOG_K,
        }
    }
}

impl<F: JoltField> RegisterMatrixAddressMajorOptimized<F> {
    /// Create an optimized address-major register matrix **directly from the trace**.
    ///
    /// This is the primary constructor for the **address-first** configuration
    /// (phase1 = 0, phase2 = LOG_K). It mirrors the semantics of:
    ///
    /// - [`RegisterMatrixCycleMajor::from_trace`] followed by
    /// - [`RegisterMatrixAddressMajorOptimized::from`],
    ///
    /// but avoids materializing the intermediate cycle-major representation.
    ///
    /// # Algorithm
    ///
    /// 1. For each cycle, build entries for each unique register touched by rs1/rs2/rd.
    ///    - `val_coeff` = value *before* any access at that cycle (prev_val)
    ///    - `next_val`  = value *after* all accesses at that cycle
    /// 2. Sort entries by `(col, row)` for address-major order.
    /// 3. Build SoA vectors and compact RA/WA flags (`u8` with `NO_ACCESS` sentinel).
    /// 4. Compute `val_final[k]` = final value of register `k` after all cycles.
    #[tracing::instrument(skip_all, name = "RegisterMatrixAddressMajorOptimized::from_trace")]
    pub fn from_trace(trace: &[Cycle]) -> Self {
        println!("RegisterMatrixAddressMajorOptimized::from_trace");
        let t_size = trace.len();

        if t_size == 0 {
            // No cycles: all registers stay at their initial value (zero).
            let val_init_vec: Vec<F> = vec![F::zero(); K];
            let mut eq_table = ExpandingTable::new(K, BindingOrder::LowToHigh);
            eq_table.reset(F::one());
            return Self {
                rows: Vec::new(),
                cols: Vec::new(),
                vals: Vec::new(),
                rs1_ras: Vec::new(),
                rs2_ras: Vec::new(),
                rd_was: Vec::new(),
                eq_table,
                val_init: MultilinearPolynomial::from(val_init_vec.clone()),
                val_final: MultilinearPolynomial::from(val_init_vec),
                num_row_bits: 0,
                num_col_bits: LOG_K,
            };
        }

        // Step 1: build per-(cycle, register) entries in parallel.
        //
        // We intentionally mirror the logic of `RegisterMatrixCycleMajor::from_trace`
        // so that the resulting data is semantically identical to constructing a
        // cycle-major matrix and then converting it to this optimized format.
        //
        // Temporary entry format:
        //   (row, col, prev_val, next_val, has_rs1, has_rs2, has_rd)
        type TmpEntry = (usize, u8, u64, u64, bool, bool, bool);

        let mut entries: Vec<TmpEntry> = trace
            .par_iter()
            .enumerate()
            .flat_map_iter(|(j, cycle)| {
                let (rs1_reg, rs1_val) = cycle.rs1_read();
                let (rs2_reg, rs2_val) = cycle.rs2_read();
                let (rd_reg, rd_pre_val, rd_post_val) = cycle.rd_write();

                // Collect unique registers accessed in this cycle.
                // Format: (reg, has_rs1, has_rs2, has_rd, prev_val, next_val)
                type RegAccess = (u8, bool, bool, bool, u64, u64);
                let mut regs: Vec<RegAccess> = Vec::with_capacity(3);

                // Helper to add or merge access for a register within this cycle.
                let mut add_access =
                    |reg: u8, is_rs1: bool, is_rs2: bool, is_rd: bool, val: u64, next: u64| {
                        if let Some(entry) = regs.iter_mut().find(|(r, ..)| *r == reg) {
                            if is_rs1 {
                                entry.1 = true;
                            }
                            if is_rs2 {
                                entry.2 = true;
                            }
                            if is_rd {
                                entry.3 = true;
                                // For writes, the post-write value is the final value
                                // of this register at this cycle.
                                entry.5 = next;
                            }
                        } else {
                            regs.push((reg, is_rs1, is_rs2, is_rd, val, next));
                        }
                    };

                // rs1: read-only, prev == next == value at this cycle.
                add_access(rs1_reg, true, false, false, rs1_val, rs1_val);
                // rs2: read-only.
                add_access(rs2_reg, false, true, false, rs2_val, rs2_val);
                // rd: write, prev_val -> next_val.
                add_access(rd_reg, false, false, true, rd_pre_val, rd_post_val);

                // Ensure registers are in increasing order for this cycle.
                regs.sort_unstable_by(|(reg_a, ..), (reg_b, ..)| reg_a.cmp(reg_b));

                regs.into_iter()
                    .map(move |(reg, has_rs1, has_rs2, has_rd, prev, next)| {
                        (j, reg, prev, next, has_rs1, has_rs2, has_rd)
                    })
            })
            .collect();

        // Step 2: sort by (col, row) for address-major order.
        entries.par_sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

        let n = entries.len();

        // Compute number of cycle bits: log2 of T (rounded up to next power of two).
        let num_row_bits = {
            let t = t_size.next_power_of_two();
            t.trailing_zeros() as usize
        };

        // Step 3: build SoA vectors and compact RA/WA flags.
        let mut rows = Vec::with_capacity(n);
        let mut cols = Vec::with_capacity(n);
        let mut vals = Vec::with_capacity(n);
        let mut rs1_ras = Vec::with_capacity(n);
        let mut rs2_ras = Vec::with_capacity(n);
        let mut rd_was = Vec::with_capacity(n);

        for (row, col, prev, _next, has_rs1, has_rs2, has_rd) in &entries {
            rows.push(*row);
            cols.push(*col);
            // val_coeff stores the value before any access at this cycle.
            vals.push(F::from_u64(*prev));

            // Compact RA/WA encoding: original register index or NO_ACCESS.
            rs1_ras.push(if *has_rs1 { *col } else { NO_ACCESS });
            rs2_ras.push(if *has_rs2 { *col } else { NO_ACCESS });
            rd_was.push(if *has_rd { *col } else { NO_ACCESS });
        }

        // Step 4: build val_init (all zeros for registers) and val_final.
        let val_init_vec: Vec<F> = vec![F::zero(); K];
        let mut val_final_vec: Vec<F> = val_init_vec.clone();

        // For each register, val_final[k] is the last next_val seen for that column.
        let mut i = 0;
        while i < n {
            let col = entries[i].1 as usize;
            let mut last = i;
            while last + 1 < n && entries[last + 1].1 == entries[i].1 {
                last += 1;
            }
            let next_val = entries[last].3;
            val_final_vec[col] = F::from_u64(next_val);
            i = last + 1;
        }

        // Initialize eq_table: starts with 1 entry (value = 1), will expand as we bind.
        let mut eq_table = ExpandingTable::new(K, BindingOrder::LowToHigh);
        eq_table.reset(F::one());

        Self {
            rows,
            cols,
            vals,
            rs1_ras,
            rs2_ras,
            rd_was,
            eq_table,
            val_init: MultilinearPolynomial::from(val_init_vec),
            val_final: MultilinearPolynomial::from(val_final_vec),
            num_row_bits,
            num_col_bits: LOG_K,
        }
    }

    /// Number of non-zero sparse entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.rows.len()
    }

    /// Get the rs1_ra coefficient for entry `i`.
    ///
    /// After binding `n` address variables, eq_table has 2^n entries.
    /// We look up the lowest n bits of the original register index.
    #[inline]
    pub fn get_rs1_ra(&self, i: usize) -> F {
        let k = self.rs1_ras[i];
        if k == NO_ACCESS {
            F::zero()
        } else {
            // k is original register index, look up partial eq using lowest bits
            let mask = self.eq_table.len() - 1;
            self.eq_table[(k as usize) & mask]
        }
    }

    /// Get the rs2_ra coefficient for entry `i`.
    #[inline]
    pub fn get_rs2_ra(&self, i: usize) -> F {
        let k = self.rs2_ras[i];
        if k == NO_ACCESS {
            F::zero()
        } else {
            let mask = self.eq_table.len() - 1;
            self.eq_table[(k as usize) & mask]
        }
    }

    /// Get the rd_wa coefficient for entry `i`.
    #[inline]
    pub fn get_rd_wa(&self, i: usize) -> F {
        let k = self.rd_was[i];
        if k == NO_ACCESS {
            F::zero()
        } else {
            let mask = self.eq_table.len() - 1;
            self.eq_table[(k as usize) & mask]
        }
    }

    /// Get the "next value" for entry `i`.
    /// If entry `i+1` is in the same column, returns `vals[i+1]`.
    /// Otherwise returns `val_final[cols[i]]`.
    #[inline]
    pub fn get_next_val(&self, i: usize) -> F {
        if i + 1 < self.nnz() && self.cols[i + 1] == self.cols[i] {
            self.vals[i + 1]
        } else {
            self.val_final.get_bound_coeff(self.cols[i] as usize)
        }
    }

    /// Helper to get next val inside parallel closure.
    #[inline]
    fn get_next_val_static(
        cols: &[u8],
        vals: &[F],
        val_final: &MultilinearPolynomial<F>,
        i: usize,
    ) -> F {
        if i + 1 < cols.len() && cols[i + 1] == cols[i] {
            vals[i + 1]
        } else {
            val_final.get_bound_coeff(cols[i] as usize)
        }
    }

    /// Bind one address variable with challenge `r`.
    ///
    /// This updates:
    /// - The eq_table (EXPANDS from 2^n to 2^(n+1) entries)
    /// - The sparse entry data (merge column pairs, but rs1_ras/rs2_ras/rd_was stay unchanged!)
    /// - val_init and val_final (shrink)
    pub fn bind(&mut self, r: F::Challenge) {
        // IMPORTANT ORDER: merge_column_pairs needs val_init BEFORE binding
        // to get checkpoint values for implicit entries.

        // First: expand eq_table
        self.eq_table.update(r);

        // Second: merge column pairs (uses current val_init for checkpoints)
        // NOTE: rs1_ras/rs2_ras/rd_was keep their ORIGINAL register indices!
        self.merge_column_pairs(r);

        // Third: bind val_init/val_final (shrinks them)
        rayon::join(
            || self.val_init.bind_parallel(r, BindingOrder::LowToHigh),
            || self.val_final.bind_parallel(r, BindingOrder::LowToHigh),
        );

        self.num_col_bits -= 1;
    }

    /// Merge column pairs during address variable binding.
    fn merge_column_pairs(&mut self, r: F::Challenge) {
        if self.nnz() == 0 {
            return;
        }

        // 1. Identify ranges for each column pair
        let pair_ranges = Self::find_column_pair_ranges(&self.cols);

        // 2. Process column pairs in parallel
        let chunks: Vec<_> = pair_ranges
            .into_par_iter()
            .map(|(pair_start, pair_end)| self.merge_single_column_pair(pair_start, pair_end, r))
            .collect();

        // 3. Merge chunks
        let total_len: usize = chunks.iter().map(|c| c.0.len()).sum();
        let mut new_rows = Vec::with_capacity(total_len);
        let mut new_cols = Vec::with_capacity(total_len);
        let mut new_vals = Vec::with_capacity(total_len);
        let mut new_rs1_ras = Vec::with_capacity(total_len);
        let mut new_rs2_ras = Vec::with_capacity(total_len);
        let mut new_rd_was = Vec::with_capacity(total_len);

        for (rows, cols, vals, rs1, rs2, rd) in chunks {
            new_rows.extend(rows);
            new_cols.extend(cols);
            new_vals.extend(vals);
            new_rs1_ras.extend(rs1);
            new_rs2_ras.extend(rs2);
            new_rd_was.extend(rd);
        }

        self.rows = new_rows;
        self.cols = new_cols;
        self.vals = new_vals;
        self.rs1_ras = new_rs1_ras;
        self.rs2_ras = new_rs2_ras;
        self.rd_was = new_rd_was;
    }

    /// Find ranges of entries that belong to the same column pair.
    fn find_column_pair_ranges(cols: &[u8]) -> Vec<(usize, usize)> {
        let mut ranges = Vec::new();
        let mut idx = 0;
        let n = cols.len();

        while idx < n {
            let col_pair = cols[idx] / 2;
            let start = idx;
            while idx < n && cols[idx] / 2 == col_pair {
                idx += 1;
            }
            ranges.push((start, idx));
        }
        ranges
    }

    /// Merge a single column pair (indices pair_start..pair_end).
    fn merge_single_column_pair(
        &self,
        pair_start: usize,
        pair_end: usize,
        r: F::Challenge,
    ) -> OptimizedColumnPairChunk<F> {
        let len = pair_end - pair_start;
        let mut chunk_rows = Vec::with_capacity(len);
        let mut chunk_cols = Vec::with_capacity(len);
        let mut chunk_vals = Vec::with_capacity(len);
        let mut chunk_rs1 = Vec::with_capacity(len);
        let mut chunk_rs2 = Vec::with_capacity(len);
        let mut chunk_rd = Vec::with_capacity(len);

        let col_pair = self.cols[pair_start] / 2;
        let even_col_idx = (col_pair * 2) as usize;
        let odd_col_idx = even_col_idx + 1;

        // Split into even and odd columns within this range
        let pivot = (pair_start..pair_end)
            .find(|&idx| self.cols[idx].is_odd())
            .unwrap_or(pair_end);

        // Create index-based views
        let even_indices: Vec<usize> = (pair_start..pivot).collect();
        let odd_indices: Vec<usize> = (pivot..pair_end).collect();

        // Initialize checkpoints for implicit values
        let mut even_checkpoint = self.val_init.get_bound_coeff(even_col_idx);
        let mut odd_checkpoint = self.val_init.get_bound_coeff(odd_col_idx);

        // Use TwoPointerMerge to iterate by row
        let merge = TwoPointerMerge::new(&even_indices, &odd_indices, |&idx| self.rows[idx]);

        for item in merge {
            match item {
                MergeItem::Both {
                    even: &ei,
                    odd: &oi,
                } => {
                    let row = self.rows[ei];
                    chunk_rows.push(row);
                    chunk_cols.push(col_pair);
                    chunk_vals.push(linear_interpolate(self.vals[ei], self.vals[oi], r));
                    chunk_rs1.push(Self::merge_register_index(
                        self.rs1_ras[ei],
                        self.rs1_ras[oi],
                    ));
                    chunk_rs2.push(Self::merge_register_index(
                        self.rs2_ras[ei],
                        self.rs2_ras[oi],
                    ));
                    chunk_rd.push(Self::merge_register_index(self.rd_was[ei], self.rd_was[oi]));

                    even_checkpoint =
                        Self::get_next_val_static(&self.cols, &self.vals, &self.val_final, ei);
                    odd_checkpoint =
                        Self::get_next_val_static(&self.cols, &self.vals, &self.val_final, oi);
                }
                MergeItem::EvenOnly(&ei) => {
                    let row = self.rows[ei];
                    chunk_rows.push(row);
                    chunk_cols.push(col_pair);
                    chunk_vals.push(linear_interpolate(self.vals[ei], odd_checkpoint, r));
                    chunk_rs1.push(Self::merge_register_index(self.rs1_ras[ei], NO_ACCESS));
                    chunk_rs2.push(Self::merge_register_index(self.rs2_ras[ei], NO_ACCESS));
                    chunk_rd.push(Self::merge_register_index(self.rd_was[ei], NO_ACCESS));

                    even_checkpoint =
                        Self::get_next_val_static(&self.cols, &self.vals, &self.val_final, ei);
                }
                MergeItem::OddOnly(&oi) => {
                    let row = self.rows[oi];
                    chunk_rows.push(row);
                    chunk_cols.push(col_pair);
                    chunk_vals.push(linear_interpolate(even_checkpoint, self.vals[oi], r));
                    chunk_rs1.push(Self::merge_register_index(NO_ACCESS, self.rs1_ras[oi]));
                    chunk_rs2.push(Self::merge_register_index(NO_ACCESS, self.rs2_ras[oi]));
                    chunk_rd.push(Self::merge_register_index(NO_ACCESS, self.rd_was[oi]));

                    odd_checkpoint =
                        Self::get_next_val_static(&self.cols, &self.vals, &self.val_final, oi);
                }
            }
        }

        (
            chunk_rows, chunk_cols, chunk_vals, chunk_rs1, chunk_rs2, chunk_rd,
        )
    }

    /// Merge two register indices when binding a variable.
    ///
    /// KEY INSIGHT: We keep the ORIGINAL register index, NOT divided by 2!
    /// The eq_table expands and handles the bit-by-bit eq computation.
    ///
    /// At any cycle, each access type (rs1, rs2, rd) accesses exactly ONE register.
    /// So when merging column pairs, at most one can have the access.
    #[inline]
    fn merge_register_index(even_idx: u8, odd_idx: u8) -> u8 {
        match (even_idx == NO_ACCESS, odd_idx == NO_ACCESS) {
            (true, true) => NO_ACCESS,
            (false, true) => even_idx, // Keep original index!
            (true, false) => odd_idx,  // Keep original index!
            (false, false) => {
                // Both have access - shouldn't happen for same access type at same row
                // (rs1 only reads one register per cycle)
                // But if it does happen, just keep even's index
                even_idx
            }
        }
    }

    /// Materialize to dense polynomials after all address variables are bound.
    ///
    /// Returns `(rs1_ra, rs2_ra, rd_wa, val)` as dense polynomials over cycle variables.
    pub fn materialize(
        self,
        t_size: usize,
    ) -> (
        MultilinearPolynomial<F>,
        MultilinearPolynomial<F>,
        MultilinearPolynomial<F>,
        MultilinearPolynomial<F>,
    ) {
        debug_assert_eq!(
            self.num_col_bits, 0,
            "Must bind all address variables before materializing"
        );
        debug_assert_eq!(
            self.eq_table.len(),
            K,
            "eq_table should have K={K} entries after all bindings"
        );

        // Materialize to dense T-sized polynomials
        let mut rs1_ra: Vec<F> = unsafe_allocate_zero_vec(t_size);
        let mut rs2_ra: Vec<F> = unsafe_allocate_zero_vec(t_size);
        let mut rd_wa: Vec<F> = unsafe_allocate_zero_vec(t_size);
        let mut val: Vec<F> = vec![self.val_init.final_sumcheck_claim(); t_size];

        // Fill in sparse entries
        // After all LOG_K bindings, eq_table[k] = eq(k, r_address)
        for i in 0..self.nnz() {
            let row = self.rows[i];
            // Look up eq(original_register, r_address) from eq_table
            rs1_ra[row] = self.get_rs1_ra(i);
            rs2_ra[row] = self.get_rs2_ra(i);
            rd_wa[row] = self.get_rd_wa(i);
            val[row] = self.vals[i];
        }

        (
            MultilinearPolynomial::from(rs1_ra),
            MultilinearPolynomial::from(rs2_ra),
            MultilinearPolynomial::from(rd_wa),
            MultilinearPolynomial::from(val),
        )
    }
}

/// Convert from cycle-major to optimized address-major representation.
impl<F: JoltField> From<RegisterMatrixCycleMajor<F>> for RegisterMatrixAddressMajorOptimized<F> {
    fn from(cycle_major: RegisterMatrixCycleMajor<F>) -> Self {
        let n = cycle_major.entries.len();
        let num_row_bits = if n > 0 {
            cycle_major
                .entries
                .iter()
                .map(|e| e.row)
                .max()
                .unwrap_or(0)
                .next_power_of_two()
                .trailing_zeros() as usize
        } else {
            0
        };

        if n == 0 {
            return Self {
                val_init: cycle_major.val_init.clone(),
                val_final: cycle_major.val_init,
                num_row_bits,
                ..Default::default()
            };
        }

        // Sort entries by (col, row) for address-major order
        let mut entries = cycle_major.entries;
        entries.par_sort_by(|a, b| a.col.cmp(&b.col).then(a.row.cmp(&b.row)));

        // Extract into SoA format
        let mut rows = Vec::with_capacity(n);
        let mut cols = Vec::with_capacity(n);
        let mut vals = Vec::with_capacity(n);
        let mut rs1_ras = Vec::with_capacity(n);
        let mut rs2_ras = Vec::with_capacity(n);
        let mut rd_was = Vec::with_capacity(n);

        for entry in &entries {
            rows.push(entry.row);
            cols.push(entry.col);
            vals.push(entry.val_coeff);

            // Store register index if accessed, NO_ACCESS otherwise
            rs1_ras.push(if entry.rs1_ra.is_some() {
                entry.col
            } else {
                NO_ACCESS
            });
            rs2_ras.push(if entry.rs2_ra.is_some() {
                entry.col
            } else {
                NO_ACCESS
            });
            rd_was.push(if entry.rd_wa.is_some() {
                entry.col
            } else {
                NO_ACCESS
            });
        }

        // Build val_final from last entry in each column
        let k_size = cycle_major.val_init.len();
        let mut val_final_vec: Vec<F> = (0..k_size)
            .map(|k| cycle_major.val_init.get_bound_coeff(k))
            .collect();

        // Update val_final for columns that have entries
        let mut i = 0;
        while i < n {
            let col = cols[i];
            let mut last = i;
            while last + 1 < n && cols[last + 1] == col {
                last += 1;
            }
            val_final_vec[col as usize] = entries[last].next_val_as_field();
            i = last + 1;
        }

        // Initialize eq_table: starts with 1 entry (value = 1), will expand as we bind
        let mut eq_table = ExpandingTable::new(K, BindingOrder::LowToHigh);
        eq_table.reset(F::one());

        Self {
            rows,
            cols,
            vals,
            rs1_ras,
            rs2_ras,
            rd_was,
            eq_table,
            val_init: cycle_major.val_init,
            val_final: MultilinearPolynomial::from(val_final_vec),
            num_row_bits,
            num_col_bits: LOG_K,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::JoltField;
    use ark_bn254::Fr;
    use ark_std::One;

    #[test]
    fn test_no_access_sentinel() {
        assert_eq!(NO_ACCESS, u8::MAX);
        assert!(NO_ACCESS > 127); // Larger than any valid register index
    }

    #[test]
    fn test_merge_register_index_keeps_original() {
        // Both NO_ACCESS
        assert_eq!(
            RegisterMatrixAddressMajorOptimized::<Fr>::merge_register_index(NO_ACCESS, NO_ACCESS),
            NO_ACCESS
        );

        // Only even has access - keeps ORIGINAL index (not divided by 2!)
        assert_eq!(
            RegisterMatrixAddressMajorOptimized::<Fr>::merge_register_index(10, NO_ACCESS),
            10 // NOT 5!
        );

        // Only odd has access - keeps ORIGINAL index
        assert_eq!(
            RegisterMatrixAddressMajorOptimized::<Fr>::merge_register_index(NO_ACCESS, 11),
            11 // NOT 5!
        );

        // Both have access (shouldn't happen for same access type, but just in case)
        assert_eq!(
            RegisterMatrixAddressMajorOptimized::<Fr>::merge_register_index(10, 11),
            10 // Keeps even's index
        );
    }

    #[test]
    fn test_eq_table_expanding() {
        // ExpandingTable starts at length 1, EXPANDS as we bind
        let mut eq_table = ExpandingTable::<Fr>::new(8, BindingOrder::LowToHigh);
        eq_table.reset(Fr::one());

        // Initially: length 1, value = 1
        assert_eq!(eq_table.len(), 1);
        assert_eq!(eq_table[0], Fr::one());

        // After binding r_0: length 2
        let r0: <Fr as JoltField>::Challenge = 3u128.into();
        let r0_f = Fr::one() * r0;
        eq_table.update(r0);

        assert_eq!(eq_table.len(), 2);
        assert_eq!(eq_table[0], Fr::one() - r0_f); // (1-r_0) for even registers
        assert_eq!(eq_table[1], r0_f); // r_0 for odd registers

        // After binding r_1: length 4
        let r1: <Fr as JoltField>::Challenge = 5u128.into();
        let r1_f = Fr::one() * r1;
        eq_table.update(r1);

        assert_eq!(eq_table.len(), 4);
        // eq_table[x] = eq((x_0, x_1), (r_0, r_1))
        // x=0 (00): (1-r_0)(1-r_1)
        // x=1 (01): r_0(1-r_1)
        // x=2 (10): (1-r_0)r_1
        // x=3 (11): r_0 r_1
        assert_eq!(eq_table[0], (Fr::one() - r0_f) * (Fr::one() - r1_f));
        assert_eq!(eq_table[1], r0_f * (Fr::one() - r1_f));
        assert_eq!(eq_table[2], (Fr::one() - r0_f) * r1_f);
        assert_eq!(eq_table[3], r0_f * r1_f);
    }

    #[test]
    fn test_eq_table_lookup_with_original_index() {
        // Simulate what happens when we look up eq coefficients
        let mut eq_table = ExpandingTable::<Fr>::new(8, BindingOrder::LowToHigh);
        eq_table.reset(Fr::one());

        let r0: <Fr as JoltField>::Challenge = 3u128.into();
        let r0_f = Fr::one() * r0;
        eq_table.update(r0);

        // Original register 4 (binary: 100) -> lowest bit is 0 (even)
        // Original register 5 (binary: 101) -> lowest bit is 1 (odd)
        let mask = eq_table.len() - 1; // = 1
        assert_eq!(eq_table[4 & mask], Fr::one() - r0_f); // eq_table[0]
        assert_eq!(eq_table[5 & mask], r0_f); // eq_table[1]

        // After another binding
        let r1: <Fr as JoltField>::Challenge = 5u128.into();
        let r1_f = Fr::one() * r1;
        eq_table.update(r1);

        let mask = eq_table.len() - 1; // = 3
                                       // Register 4 = 0b100, lowest 2 bits = 00
        assert_eq!(eq_table[4 & mask], (Fr::one() - r0_f) * (Fr::one() - r1_f));
        // Register 5 = 0b101, lowest 2 bits = 01
        assert_eq!(eq_table[5 & mask], r0_f * (Fr::one() - r1_f));
        // Register 6 = 0b110, lowest 2 bits = 10
        assert_eq!(eq_table[6 & mask], (Fr::one() - r0_f) * r1_f);
        // Register 7 = 0b111, lowest 2 bits = 11
        assert_eq!(eq_table[7 & mask], r0_f * r1_f);
    }

    #[test]
    fn test_default_eq_table() {
        let matrix: RegisterMatrixAddressMajorOptimized<Fr> = Default::default();
        // Default eq_table starts at length 1 with value 1
        assert_eq!(matrix.eq_table.len(), 1);
        assert_eq!(matrix.eq_table[0], Fr::one());
    }
}
