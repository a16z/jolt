//! Address-major (column-major) sparse matrix representation for RAM.
//!
//! Entries are sorted by `(col, row)` - optimal for binding address variables first.

use std::cmp::Ordering;
use std::mem::MaybeUninit;

use allocative::Allocative;
use ark_std::Zero;
use common::jolt_device::MemoryLayout;
use rayon::prelude::*;
use tracer::instruction::{Cycle, RAMAccess};

use crate::field::JoltField;
use crate::field::OptimizedMul;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::poly::unipoly::UniPoly;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::zkvm::ram::remap_address;

use super::cycle_major::ReadWriteMatrixCycleMajor;
use super::super::ColIndex;

/// Represents the ra(k, j) and Val(k, j) polynomials for the RAM
/// read/write-checking sumcheck in address-major (column-major) order.
///
/// # Memory Layout: Struct-of-Arrays (SoA)
///
/// Unlike `ReadWriteMatrixCycleMajor` which uses Array-of-Structs (`Vec<ReadWriteEntry>`),
/// this uses SoA for better cache performance during address binding:
///
/// ```text
/// Sparse entry arrays (variable size, doesn't halve predictably):
///   rows[i] = cycle index for entry i
///   cols[i] = address index for entry i
///   vals[i] = Val(k, j) coefficient for entry i
///   ras[i]  = ra(k, j) coefficient for entry i
///
/// Dense auxiliary arrays (size K, halves each round):
///   val_init[k]  = initial value at address k (before any access)
///   val_final[k] = final value at address k (after last access)
/// ```
///
/// # Deriving `next_val` Without Storing It
///
/// Key memory optimization: Instead of storing `prev_val`/`next_val` per entry
/// (which would add 2 field elements = 64 bytes per entry), we derive `next_val`
/// on-the-fly using `get_next_val(i)`:
///
/// ```text
/// If entry i+1 is in the same column:
///   next_val(i) = vals[i+1]   // Next access in same column
/// Else:
///   next_val(i) = val_final[cols[i]]  // Value persists until end
/// ```
///
/// This works because entries are sorted by `(col, row)`, so within each column,
/// entries appear in chronological order.
///
/// # Invariants
///
/// - Entries are sorted by `(col, row)` (address-major order)
/// - `rows`, `cols`, `vals`, `ras` all have the same length (`nnz()`)
/// - `val_init` and `val_final` have the same length (`K` = address space size)
/// - `val_final[k]` equals `next_val` of the last entry in column `k`
///
/// # Type Parameters
///
/// - `F`: The field type for coefficients.
/// - `I`: The column index type (e.g., `usize` for RAM, `u8` for registers).
#[derive(Allocative, Debug, Default, Clone)]
pub struct ReadWriteMatrixAddressMajor<F: JoltField, I: ColIndex = usize> {
    /// Row indices (cycle indices) for each sparse entry.
    pub rows: Vec<usize>,
    /// Column indices (address indices) for each sparse entry.
    pub cols: Vec<I>,
    /// Val(k, j) coefficients: the value read/written at each access.
    pub vals: Vec<F>,
    /// ra(k, j) coefficients: always 1 for explicit accesses, used for sumcheck.
    pub ras: Vec<F>,
    /// Initial Val polynomial over addresses: `val_init[k] = Val(k, j=0)`.
    /// This is the memory state before the first cycle.
    val_init: MultilinearPolynomial<F>,
    /// Final Val polynomial over addresses: `val_final[k]` = value at address `k`
    /// after the last access to it. Used to derive `next_val` for the last entry
    /// in each column. For columns with no accesses, `val_final[k] = val_init[k]`.
    val_final: MultilinearPolynomial<F>,
}

impl<F: JoltField, I: ColIndex> ReadWriteMatrixAddressMajor<F, I> {
    /// Number of non-zero sparse entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.rows.len()
    }

    /// Get the "next value" for entry `i`.
    /// If entry `i+1` is in the same column, returns `vals[i+1]`.
    /// Otherwise returns `val_final[cols[i]]` (value persists after last access).
    #[inline]
    pub fn get_next_val(&self, i: usize) -> F {
        if i + 1 < self.nnz() && self.cols[i + 1] == self.cols[i] {
            self.vals[i + 1]
        } else {
            self.val_final.get_bound_coeff(self.cols[i].to_usize())
        }
    }
}

impl<F: JoltField> ReadWriteMatrixAddressMajor<F, usize> {
    /// Creates a new `ReadWriteMatrixAddressMajor` directly from the execution trace.
    ///
    /// This is the primary constructor for **address-first sumcheck** where we bind
    /// address variables before cycle variables.
    ///
    /// # Arguments
    ///
    /// * `trace` - The execution trace containing RAM accesses
    /// * `val_init` - Initial memory state (value at each address before execution)
    /// * `memory_layout` - Memory layout for address remapping
    ///
    /// # Algorithm
    ///
    /// 1. Extract (row=cycle, col=address, val, ra) from each RAM access in trace
    /// 2. Sort entries by (col, row) for address-major order
    /// 3. Compute `val_final[k]` = value at address k after the last access to it
    #[tracing::instrument(skip_all, name = "ReadWriteMatrixAddressMajor::from_trace")]
    pub fn from_trace(trace: &[Cycle], val_init: Vec<F>, memory_layout: &MemoryLayout) -> Self {
        // Step 1: Extract entries from trace in parallel
        // Each entry is (row=cycle_idx, col=address, val_coeff, next_val)
        let mut entries: Vec<(usize, usize, F, F)> = trace
            .par_iter()
            .enumerate()
            .filter_map(|(j, cycle)| {
                let ram_op = cycle.ram_access();
                match ram_op {
                    RAMAccess::Write(write) => {
                        let pre_value = F::from_u64(write.pre_value);
                        let post_value = F::from_u64(write.post_value);
                        let col = remap_address(write.address, memory_layout)? as usize;
                        Some((j, col, pre_value, post_value))
                    }
                    RAMAccess::Read(read) => {
                        let read_value = F::from_u64(read.value);
                        let col = remap_address(read.address, memory_layout)? as usize;
                        Some((j, col, read_value, read_value))
                    }
                    _ => None,
                }
            })
            .collect();

        // Step 2: Sort by (col, row) - address-major order
        entries.par_sort_by(|a, b| match a.1.cmp(&b.1) {
            Ordering::Equal => a.0.cmp(&b.0),
            other => other,
        });

        let n = entries.len();
        let k_size = val_init.len();

        // Step 3: Build SoA arrays
        let rows: Vec<usize> = entries.par_iter().map(|(row, _, _, _)| *row).collect();
        let cols: Vec<usize> = entries.par_iter().map(|(_, col, _, _)| *col).collect();
        let ras: Vec<F> = vec![F::one(); n]; // ra = 1 for all explicit accesses

        // For vals, we need the val_coeff (pre_value for writes, read_value for reads)
        let vals: Vec<F> = entries.par_iter().map(|(_, _, val, _)| *val).collect();

        // Step 4: Compute val_final
        // Initialize from val_init, then update with last entry's next_val per column
        let mut val_final_vec: Vec<F> = (0..k_size).into_par_iter().map(|k| val_init[k]).collect();

        // Update val_final for columns that have entries
        // entries is sorted by (col, row), so consecutive entries with same col form groups
        let val_final_ptr = val_final_vec.as_mut_ptr() as usize;
        entries
            .par_chunk_by(|a, b| a.1 == b.1) // chunk by col
            .for_each(|column_entries| {
                let col = column_entries[0].1;
                let last_next_val = column_entries.last().unwrap().3; // next_val of last entry
                // SAFETY: Each column appears in exactly one chunk, so writes are disjoint
                unsafe {
                    let ptr = val_final_ptr as *mut F;
                    *ptr.add(col) = last_next_val;
                }
            });

        ReadWriteMatrixAddressMajor {
            rows,
            cols,
            vals,
            ras,
            val_init: val_init.into(),
            val_final: val_final_vec.into(),
        }
    }
}

impl<F: JoltField, I: ColIndex> From<ReadWriteMatrixCycleMajor<F, I>>
    for ReadWriteMatrixAddressMajor<F, I>
{
    #[tracing::instrument(skip_all, name = "ReadWriteMatrixAddressMajor::from")]
    fn from(mut cycle_major: ReadWriteMatrixCycleMajor<F, I>) -> Self {
        let mut entries = std::mem::take(&mut cycle_major.entries);
        let val_init = std::mem::take(&mut cycle_major.val_init);

        // Sort entries by (col, row) - address-major order
        entries.par_sort_by(|a, b| match a.col.cmp(&b.col) {
            Ordering::Less => Ordering::Less,
            Ordering::Greater => Ordering::Greater,
            Ordering::Equal => a.row.cmp(&b.row),
        });

        let k_size = val_init.len();

        // Build SoA arrays in parallel - each array built independently
        // This iterates entries 4 times but each pass is fully parallel and cache-friendly.
        let rows: Vec<usize> = entries.par_iter().map(|e| e.row).collect();
        let cols: Vec<I> = entries.par_iter().map(|e| e.col).collect();
        let vals: Vec<F> = entries.par_iter().map(|e| e.val_coeff).collect();
        let ras: Vec<F> = entries.par_iter().map(|e| e.ra_coeff).collect();

        // Initialize val_final from val_init
        let mut val_final_vec: Vec<F> = (0..k_size)
            .into_par_iter()
            .map(|k| val_init.get_bound_coeff(k))
            .collect();

        // Build and apply val_final updates in parallel.
        // Since entries are sorted by (col, row), consecutive entries with the same col
        // form contiguous groups. The last entry in each group has the correct next_val.
        //
        // Each column appears in exactly one chunk, so writes are disjoint.
        // We write directly to avoid intermediate Vec allocation.
        //
        // Note: We convert the pointer to usize to make it Sync (raw pointers aren't Sync).
        let val_final_ptr = val_final_vec.as_mut_ptr() as usize;
        entries
            .par_chunk_by(|a, b| a.col == b.col)
            .for_each(|column_entries| {
                let col = column_entries[0].col.to_usize();
                // Convert u64 to F (cycle-major stores prev_val/next_val as u64 for memory efficiency)
                let last_next_val = F::from_u64(column_entries.last().unwrap().next_val);
                // SAFETY: Each column appears in exactly one chunk (entries sorted by col),
                // so writes to val_final_vec[col] are disjoint across parallel iterations.
                // The pointer is valid for the lifetime of this closure.
                unsafe {
                    let ptr = val_final_ptr as *mut F;
                    *ptr.add(col) = last_next_val;
                }
            });

        ReadWriteMatrixAddressMajor {
            rows,
            cols,
            vals,
            ras,
            val_init,
            val_final: val_final_vec.into(),
        }
    }
}

impl<F: JoltField, I: ColIndex> ReadWriteMatrixAddressMajor<F, I> {
    /// Binds an address variable of the ra and Val polynomials represented by
    /// this matrix to the random challenge `r`.
    ///
    /// # Address Binding Algorithm
    ///
    /// Entries are grouped by column-pair `(2k, 2k+1)` and merged using the "checkpoint"
    /// pattern:
    ///
    /// - **Checkpoint concept**: When an entry exists in only one column of a pair (e.g.,
    ///   only even column at row R), the implicit value for the other column is the last
    ///   known value in that column, tracked via `even_checkpoint` / `odd_checkpoint`.
    ///
    /// - **Initial checkpoints**: `val_init[2k]` and `val_init[2k+1]` respectively.
    ///
    /// - **After each entry**: Update the checkpoint with `get_next_val(i)` to track
    ///   the value after that access.
    ///
    /// This avoids storing `prev_val`/`next_val` per entry (saving ~50% memory) while
    /// still correctly computing implicit values during binding.
    ///
    /// # Parallelization Strategy
    ///
    /// 1. Group entries by column-pair using indices (can't use `par_chunk_by` on SoA)
    /// 2. Compute bound lengths in parallel (dry run)
    /// 3. Pre-split output buffers into disjoint `MaybeUninit` slices
    /// 4. Parallel write with recursive divide-and-conquer within large column pairs
    #[tracing::instrument(skip_all, name = "ReadWriteMatrixAddressMajor::bind")]
    pub fn bind(&mut self, r: F::Challenge) {
        let n = self.nnz();
        if n == 0 {
            self.val_init.bind_parallel(r, BindingOrder::LowToHigh);
            self.val_final.bind_parallel(r, BindingOrder::LowToHigh);
            return;
        }

        // First pass: collect column-pair info with bound lengths.
        // Group entries by col/2; within each group, even column entries come first.
        struct ColPairInfo {
            input_start: usize,  // Start index in input arrays
            input_end: usize,    // End index in input arrays
            even_end: usize,     // Boundary between even and odd entries
            even_col_idx: usize, // As usize for indexing val_init
            bound_len: usize,
        }

        // Find column-pair boundaries and compute bound lengths in parallel
        let pair_ranges: Vec<(usize, usize)> = {
            let mut ranges = Vec::new();
            let mut idx = 0;
            while idx < n {
                let col_pair = self.cols[idx].to_usize() / 2;
                let mut j = idx + 1;
                while j < n && self.cols[j].to_usize() / 2 == col_pair {
                    j += 1;
                }
                ranges.push((idx, j));
                idx = j;
            }
            ranges
        };

        // Parallel pass: compute pair info and bound lengths
        let pairs: Vec<ColPairInfo> = pair_ranges
            .par_iter()
            .map(|&(start, end)| {
                let col_pair = self.cols[start].to_usize() / 2;

                // Find boundary between even and odd column entries
                let mut mid = start;
                while mid < end && self.cols[mid].is_even() {
                    mid += 1;
                }

                // Use recursive bind_cols for dry run (respects PAR_THRESHOLD)
                let bound_len = Self::bind_cols(
                    self,
                    start,
                    mid,
                    mid,
                    end,
                    F::zero(), // Checkpoints not needed for dry run
                    F::zero(),
                    r,
                    &mut [],
                    &mut [],
                    &mut [],
                    &mut [],
                    true,
                );

                ColPairInfo {
                    input_start: start,
                    input_end: end,
                    even_end: mid,
                    even_col_idx: 2 * col_pair,
                    bound_len,
                }
            })
            .collect();

        let total_bound: usize = pairs.iter().map(|p| p.bound_len).sum();

        // Allocate new SoA arrays using MaybeUninit for safe parallel writes
        let mut rows_new: Vec<MaybeUninit<usize>> = Vec::with_capacity(total_bound);
        let mut cols_new: Vec<MaybeUninit<I>> = Vec::with_capacity(total_bound);
        let mut vals_new: Vec<MaybeUninit<F>> = Vec::with_capacity(total_bound);
        let mut ras_new: Vec<MaybeUninit<F>> = Vec::with_capacity(total_bound);

        // SAFETY: We're about to write to all positions in parallel
        unsafe {
            rows_new.set_len(total_bound);
            cols_new.set_len(total_bound);
            vals_new.set_len(total_bound);
            ras_new.set_len(total_bound);
        }

        // Pre-split output buffers into disjoint slices for each column pair.
        // This is the key pattern that avoids unsafe raw pointer arithmetic.
        let mut rows_slices: Vec<&mut [MaybeUninit<usize>]> = Vec::with_capacity(pairs.len());
        let mut cols_slices: Vec<&mut [MaybeUninit<I>]> = Vec::with_capacity(pairs.len());
        let mut vals_slices: Vec<&mut [MaybeUninit<F>]> = Vec::with_capacity(pairs.len());
        let mut ras_slices: Vec<&mut [MaybeUninit<F>]> = Vec::with_capacity(pairs.len());

        let mut rows_remaining = rows_new.as_mut_slice();
        let mut cols_remaining = cols_new.as_mut_slice();
        let mut vals_remaining = vals_new.as_mut_slice();
        let mut ras_remaining = ras_new.as_mut_slice();

        for p in pairs.iter() {
            let (rows_slice, rows_rest) = rows_remaining.split_at_mut(p.bound_len);
            let (cols_slice, cols_rest) = cols_remaining.split_at_mut(p.bound_len);
            let (vals_slice, vals_rest) = vals_remaining.split_at_mut(p.bound_len);
            let (ras_slice, ras_rest) = ras_remaining.split_at_mut(p.bound_len);

            rows_slices.push(rows_slice);
            cols_slices.push(cols_slice);
            vals_slices.push(vals_slice);
            ras_slices.push(ras_slice);

            rows_remaining = rows_rest;
            cols_remaining = cols_rest;
            vals_remaining = vals_rest;
            ras_remaining = ras_rest;
        }

        // Second pass: perform actual column binding in parallel.
        // Each pair writes to its pre-allocated disjoint slice.
        pairs
            .par_iter()
            .zip(rows_slices.into_par_iter())
            .zip(cols_slices.into_par_iter())
            .zip(vals_slices.into_par_iter())
            .zip(ras_slices.into_par_iter())
            .for_each(|((((p, rows_out), cols_out), vals_out), ras_out)| {
                let even_checkpoint = self.val_init.get_bound_coeff(p.even_col_idx);
                let odd_checkpoint = self.val_init.get_bound_coeff(p.even_col_idx + 1);

                Self::bind_cols(
                    self,
                    p.input_start,
                    p.even_end,
                    p.even_end,
                    p.input_end,
                    even_checkpoint,
                    odd_checkpoint,
                    r,
                    rows_out,
                    cols_out,
                    vals_out,
                    ras_out,
                    false,
                );
            });

        // Convert MaybeUninit to initialized values
        // SAFETY: All positions were written by bind_cols
        self.rows = unsafe { std::mem::transmute::<Vec<MaybeUninit<usize>>, Vec<usize>>(rows_new) };
        self.cols = unsafe { std::mem::transmute::<Vec<MaybeUninit<I>>, Vec<I>>(cols_new) };
        self.vals = unsafe { std::mem::transmute::<Vec<MaybeUninit<F>>, Vec<F>>(vals_new) };
        self.ras = unsafe { std::mem::transmute::<Vec<MaybeUninit<F>>, Vec<F>>(ras_new) };

        // Bind the address variable on val_init and val_final (low-to-high)
        self.val_init.bind_parallel(r, BindingOrder::LowToHigh);
        self.val_final.bind_parallel(r, BindingOrder::LowToHigh);
    }

    /// Binds two adjacent columns in the sparse matrix together with the randomness `r`.
    ///
    /// This is a parallel, recursive function (similar to a parallel merge of two
    /// sorted lists) that assumes the even and odd column entries are sorted by row
    /// and writes the output to the `out` buffers in sorted order.
    ///
    /// Returns the number of entries in the bound column.
    ///
    /// If `dry_run` is true, ignores output buffers and just computes the output length.
    /// This is used to allocate exact memory before the real bind operation.
    #[allow(clippy::too_many_arguments)]
    fn bind_cols(
        &self,
        e0: usize,
        e1: usize, // Even column entries: indices [e0, e1)
        o0: usize,
        o1: usize, // Odd column entries: indices [o0, o1)
        even_checkpoint: F,
        odd_checkpoint: F,
        r: F::Challenge,
        rows_out: &mut [MaybeUninit<usize>],
        cols_out: &mut [MaybeUninit<I>],
        vals_out: &mut [MaybeUninit<F>],
        ras_out: &mut [MaybeUninit<F>],
        dry_run: bool,
    ) -> usize {
        /// Threshold where we stop parallelizing and do a plain linear merge.
        const PAR_THRESHOLD: usize = 32_768;

        let even_len = e1 - e0;
        let odd_len = o1 - o0;

        // Small inputs: do the O(n) sequential merge
        if even_len + odd_len <= PAR_THRESHOLD {
            return self.seq_bind_cols(
                e0,
                e1,
                o0,
                o1,
                even_checkpoint,
                odd_checkpoint,
                r,
                rows_out,
                cols_out,
                vals_out,
                ras_out,
                dry_run,
            );
        }

        // Split the longer column at its midpoint; find where that pivot lands in the other.
        let (even_pivot_idx, odd_pivot_idx) = if even_len > odd_len {
            let even_pivot_idx = e0 + even_len / 2;
            let pivot_row = self.rows[even_pivot_idx];
            let odd_pivot_idx = o0 + self.rows[o0..o1].partition_point(|&row| row < pivot_row);
            (even_pivot_idx, odd_pivot_idx)
        } else {
            let odd_pivot_idx = o0 + odd_len / 2;
            let pivot_row = self.rows[odd_pivot_idx];
            let even_pivot_idx = e0 + self.rows[e0..e1].partition_point(|&row| row < pivot_row);
            (even_pivot_idx, odd_pivot_idx)
        };

        // Compute the merged lengths of each half (dry run to get sizes)
        let (left_merged_len, right_merged_len) = rayon::join(
            || {
                self.bind_cols(
                    e0,
                    even_pivot_idx,
                    o0,
                    odd_pivot_idx,
                    F::zero(), // Checkpoints not needed for dry run
                    F::zero(),
                    r,
                    &mut [],
                    &mut [],
                    &mut [],
                    &mut [],
                    true,
                )
            },
            || {
                self.bind_cols(
                    even_pivot_idx,
                    e1,
                    odd_pivot_idx,
                    o1,
                    F::zero(),
                    F::zero(),
                    r,
                    &mut [],
                    &mut [],
                    &mut [],
                    &mut [],
                    true,
                )
            },
        );

        if !dry_run {
            let out_len = rows_out.len();
            debug_assert_eq!(out_len, left_merged_len + right_merged_len);

            // Split output buffers at the computed boundary
            let (left_out_rows, right_out_rows) = rows_out.split_at_mut(left_merged_len);
            let (left_out_cols, right_out_cols) = cols_out.split_at_mut(left_merged_len);
            let (left_out_vals, right_out_vals) = vals_out.split_at_mut(left_merged_len);
            let (left_out_ras, right_out_ras) = ras_out.split_at_mut(left_merged_len);

            // Compute checkpoints for the right half
            let right_even_checkpoint = if even_pivot_idx == e0 {
                even_checkpoint
            } else {
                self.get_next_val(even_pivot_idx - 1)
            };
            let right_odd_checkpoint = if odd_pivot_idx == o0 {
                odd_checkpoint
            } else {
                self.get_next_val(odd_pivot_idx - 1)
            };

            // Perform the actual merge in parallel
            rayon::join(
                || {
                    self.bind_cols(
                        e0,
                        even_pivot_idx,
                        o0,
                        odd_pivot_idx,
                        even_checkpoint,
                        odd_checkpoint,
                        r,
                        left_out_rows,
                        left_out_cols,
                        left_out_vals,
                        left_out_ras,
                        false,
                    )
                },
                || {
                    self.bind_cols(
                        even_pivot_idx,
                        e1,
                        odd_pivot_idx,
                        o1,
                        right_even_checkpoint,
                        right_odd_checkpoint,
                        r,
                        right_out_rows,
                        right_out_cols,
                        right_out_vals,
                        right_out_ras,
                        false,
                    )
                },
            );
        }

        left_merged_len + right_merged_len
    }

    /// Sequential column binding - the base case for `bind_cols`.
    ///
    /// Merges entries from even column [e0, e1) and odd column [o0, o1) into output buffers.
    #[allow(clippy::too_many_arguments)]
    fn seq_bind_cols(
        &self,
        e0: usize,
        e1: usize,
        o0: usize,
        o1: usize,
        mut even_checkpoint: F,
        mut odd_checkpoint: F,
        r: F::Challenge,
        rows_out: &mut [MaybeUninit<usize>],
        cols_out: &mut [MaybeUninit<I>],
        vals_out: &mut [MaybeUninit<F>],
        ras_out: &mut [MaybeUninit<F>],
        dry_run: bool,
    ) -> usize {
        let one = F::one();

        let mut i = e0;
        let mut j = o0;
        let mut k = 0;

        while i < e1 && j < o1 {
            let row_e = self.rows[i];
            let row_o = self.rows[j];

            if row_e == row_o {
                if !dry_run {
                    let new_col = I::from_usize(self.cols[i].to_usize() / 2);
                    let ra_even = self.ras[i];
                    let ra_odd = self.ras[j];
                    let val_even = self.vals[i];
                    let val_odd = self.vals[j];

                    rows_out[k] = MaybeUninit::new(row_e);
                    cols_out[k] = MaybeUninit::new(new_col);
                    ras_out[k] = MaybeUninit::new(ra_even + r.mul_0_optimized(ra_odd - ra_even));
                    vals_out[k] =
                        MaybeUninit::new(val_even + r.mul_0_optimized(val_odd - val_even));
                }
                even_checkpoint = self.get_next_val(i);
                odd_checkpoint = self.get_next_val(j);
                i += 1;
                j += 1;
                k += 1;
            } else if row_e < row_o {
                if !dry_run {
                    let new_col = I::from_usize(self.cols[i].to_usize() / 2);
                    let ra_even = self.ras[i];
                    let val_even = self.vals[i];

                    rows_out[k] = MaybeUninit::new(row_e);
                    cols_out[k] = MaybeUninit::new(new_col);
                    ras_out[k] = MaybeUninit::new((one - r).mul_1_optimized(ra_even));
                    vals_out[k] =
                        MaybeUninit::new(val_even + r.mul_0_optimized(odd_checkpoint - val_even));
                }
                even_checkpoint = self.get_next_val(i);
                i += 1;
                k += 1;
            } else {
                if !dry_run {
                    let new_col = I::from_usize(self.cols[j].to_usize() / 2);
                    let ra_odd = self.ras[j];
                    let val_odd = self.vals[j];

                    rows_out[k] = MaybeUninit::new(row_o);
                    cols_out[k] = MaybeUninit::new(new_col);
                    ras_out[k] = MaybeUninit::new(r.mul_1_optimized(ra_odd));
                    vals_out[k] = MaybeUninit::new(
                        even_checkpoint + r.mul_0_optimized(val_odd - even_checkpoint),
                    );
                }
                odd_checkpoint = self.get_next_val(j);
                j += 1;
                k += 1;
            }
        }

        // Remaining even-only entries
        while i < e1 {
            if !dry_run {
                let row_e = self.rows[i];
                let new_col = I::from_usize(self.cols[i].to_usize() / 2);
                let ra_even = self.ras[i];
                let val_even = self.vals[i];

                rows_out[k] = MaybeUninit::new(row_e);
                cols_out[k] = MaybeUninit::new(new_col);
                ras_out[k] = MaybeUninit::new((one - r).mul_1_optimized(ra_even));
                vals_out[k] =
                    MaybeUninit::new(val_even + r.mul_0_optimized(odd_checkpoint - val_even));
            }
            i += 1;
            k += 1;
        }

        // Remaining odd-only entries
        while j < o1 {
            if !dry_run {
                let row_o = self.rows[j];
                let new_col = I::from_usize(self.cols[j].to_usize() / 2);
                let ra_odd = self.ras[j];
                let val_odd = self.vals[j];

                rows_out[k] = MaybeUninit::new(row_o);
                cols_out[k] = MaybeUninit::new(new_col);
                ras_out[k] = MaybeUninit::new(r.mul_1_optimized(ra_odd));
                vals_out[k] = MaybeUninit::new(
                    even_checkpoint + r.mul_0_optimized(val_odd - even_checkpoint),
                );
            }
            j += 1;
            k += 1;
        }

        if !dry_run {
            debug_assert_eq!(k, rows_out.len());
        }
        k
    }

    /// Computes the prover's sumcheck message for the current round.
    ///
    /// # Algorithm
    ///
    /// Each column pair `(2k, 2k+1)` contributes to the sumcheck polynomial independently.
    /// Contributions are computed in parallel across column pairs, then reduced.
    ///
    /// For each entry, we compute evaluations at `x=0` and `x=2`:
    /// - `ra(x)` and `val(x)` are linear in the binding variable
    /// - The sumcheck polynomial is `sum_k eq(k) * ra(k,x) * (val(k,x) + gamma * (inc(k) + val(k,x)))`
    ///
    /// # Parallelization Strategy
    ///
    /// 1. Group entries by column-pair using indices
    /// 2. Compute contributions in parallel with recursive divide-and-conquer within large pairs
    /// 3. Use `fold_with` + `Unreduced` for delayed modular reduction
    pub fn compute_prover_message(
        &self,
        inc: &MultilinearPolynomial<F>,
        eq: &MultilinearPolynomial<F>,
        gamma: F,
        previous_claim: F,
    ) -> UniPoly<F> {
        let n = self.nnz();
        if n == 0 {
            return UniPoly::from_evals_and_hint(previous_claim, &[F::zero(), F::zero()]);
        }

        // Find column-pair boundaries (same as in bind())
        let pair_ranges: Vec<(usize, usize)> = {
            let mut ranges = Vec::new();
            let mut idx = 0;
            while idx < n {
                let col_pair = self.cols[idx].to_usize() / 2;
                let mut j = idx + 1;
                while j < n && self.cols[j].to_usize() / 2 == col_pair {
                    j += 1;
                }
                ranges.push((idx, j));
                idx = j;
            }
            ranges
        };

        // Parallel computation across column pairs with fold_with for efficient accumulation
        let evals = pair_ranges
            .par_iter()
            .map(|&(start, end)| {
                let col_pair = self.cols[start].to_usize() / 2;

                // Find boundary between even and odd column entries
                let mut mid = start;
                while mid < end && self.cols[mid].is_even() {
                    mid += 1;
                }

                let even_col_idx = 2 * col_pair;
                let even_checkpoint = self.val_init.get_bound_coeff(even_col_idx);
                let odd_checkpoint = self.val_init.get_bound_coeff(even_col_idx + 1);

                // Use recursive prover_message_contribution for within-pair parallelism
                self.prover_message_contribution(
                    start,
                    mid,
                    mid,
                    end,
                    even_checkpoint,
                    odd_checkpoint,
                    inc,
                    eq,
                    gamma,
                )
            })
            .fold_with([F::Unreduced::<5>::zero(); 2], |running, new| {
                [
                    running[0] + new[0].as_unreduced_ref(),
                    running[1] + new[1].as_unreduced_ref(),
                ]
            })
            .reduce(
                || [F::Unreduced::<5>::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        UniPoly::from_evals_and_hint(
            previous_claim,
            &[
                F::from_barrett_reduce(evals[0]),
                F::from_barrett_reduce(evals[1]),
            ],
        )
    }

    /// Computes the contribution of a column pair to the prover's sumcheck message.
    ///
    /// This is a parallel, recursive algorithm that uses divide-and-conquer for large
    /// column pairs (exceeding PAR_THRESHOLD).
    #[allow(clippy::too_many_arguments)]
    fn prover_message_contribution(
        &self,
        e0: usize,
        e1: usize, // Even column entries: indices [e0, e1)
        o0: usize,
        o1: usize, // Odd column entries: indices [o0, o1)
        even_checkpoint: F,
        odd_checkpoint: F,
        inc: &MultilinearPolynomial<F>,
        eq: &MultilinearPolynomial<F>,
        gamma: F,
    ) -> [F; 2] {
        /// Threshold where we stop parallelizing and do a plain linear merge.
        const PAR_THRESHOLD: usize = 32_768;

        let even_len = e1 - e0;
        let odd_len = o1 - o0;

        // Small inputs: do the O(n) sequential algorithm
        if even_len + odd_len <= PAR_THRESHOLD {
            return self.seq_prover_message_contribution(
                e0,
                e1,
                o0,
                o1,
                even_checkpoint,
                odd_checkpoint,
                inc,
                eq,
                gamma,
            );
        }

        // Split the longer column at its midpoint; find where that pivot lands in the other.
        let (even_pivot_idx, odd_pivot_idx) = if even_len > odd_len {
            let even_pivot_idx = e0 + even_len / 2;
            let pivot_row = self.rows[even_pivot_idx];
            let odd_pivot_idx = o0 + self.rows[o0..o1].partition_point(|&row| row < pivot_row);
            (even_pivot_idx, odd_pivot_idx)
        } else {
            let odd_pivot_idx = o0 + odd_len / 2;
            let pivot_row = self.rows[odd_pivot_idx];
            let even_pivot_idx = e0 + self.rows[e0..e1].partition_point(|&row| row < pivot_row);
            (even_pivot_idx, odd_pivot_idx)
        };

        // Compute checkpoints for the right half
        let right_even_checkpoint = if even_pivot_idx == e0 {
            even_checkpoint
        } else {
            self.get_next_val(even_pivot_idx - 1)
        };
        let right_odd_checkpoint = if odd_pivot_idx == o0 {
            odd_checkpoint
        } else {
            self.get_next_val(odd_pivot_idx - 1)
        };

        // Compute each half's contribution in parallel
        let (left_evals, right_evals) = rayon::join(
            || {
                self.prover_message_contribution(
                    e0,
                    even_pivot_idx,
                    o0,
                    odd_pivot_idx,
                    even_checkpoint,
                    odd_checkpoint,
                    inc,
                    eq,
                    gamma,
                )
            },
            || {
                self.prover_message_contribution(
                    even_pivot_idx,
                    e1,
                    odd_pivot_idx,
                    o1,
                    right_even_checkpoint,
                    right_odd_checkpoint,
                    inc,
                    eq,
                    gamma,
                )
            },
        );

        [
            left_evals[0] + right_evals[0],
            left_evals[1] + right_evals[1],
        ]
    }

    /// Sequential prover message contribution - the base case for `prover_message_contribution`.
    ///
    /// Uses `Unreduced<9>` accumulator to delay modular reductions for better performance.
    /// Each `compute_evals_*_unreduced` returns `Unreduced<8>` (no reduction on the final multiply),
    /// and we accumulate into `Unreduced<9>` for headroom. Only one Montgomery reduction at the end.
    #[allow(clippy::too_many_arguments)]
    fn seq_prover_message_contribution(
        &self,
        e0: usize,
        e1: usize,
        o0: usize,
        o1: usize,
        mut even_checkpoint: F,
        mut odd_checkpoint: F,
        inc: &MultilinearPolynomial<F>,
        eq: &MultilinearPolynomial<F>,
        gamma: F,
    ) -> [F; 2] {
        let mut i = e0;
        let mut j = o0;
        let mut evals_accumulator = [F::Unreduced::<9>::zero(); 2];

        while i < e1 && j < o1 {
            let row_e = self.rows[i];
            let row_o = self.rows[j];

            if row_e == row_o {
                let evals = self.compute_evals_both_unreduced(
                    i,
                    j,
                    inc.get_bound_coeff(row_e),
                    eq.get_bound_coeff(row_e),
                    gamma,
                );
                evals_accumulator[0] += evals[0];
                evals_accumulator[1] += evals[1];
                even_checkpoint = self.get_next_val(i);
                odd_checkpoint = self.get_next_val(j);
                i += 1;
                j += 1;
            } else if row_e < row_o {
                let evals = self.compute_evals_even_only_unreduced(
                    i,
                    odd_checkpoint,
                    inc.get_bound_coeff(row_e),
                    eq.get_bound_coeff(row_e),
                    gamma,
                );
                even_checkpoint = self.get_next_val(i);
                evals_accumulator[0] += evals[0];
                evals_accumulator[1] += evals[1];
                i += 1;
            } else {
                let evals = self.compute_evals_odd_only_unreduced(
                    j,
                    even_checkpoint,
                    inc.get_bound_coeff(row_o),
                    eq.get_bound_coeff(row_o),
                    gamma,
                );
                odd_checkpoint = self.get_next_val(j);
                evals_accumulator[0] += evals[0];
                evals_accumulator[1] += evals[1];
                j += 1;
            }
        }

        while i < e1 {
            let row_e = self.rows[i];
            let evals = self.compute_evals_even_only_unreduced(
                i,
                odd_checkpoint,
                inc.get_bound_coeff(row_e),
                eq.get_bound_coeff(row_e),
                gamma,
            );
            evals_accumulator[0] += evals[0];
            evals_accumulator[1] += evals[1];
            i += 1;
        }

        while j < o1 {
            let row_o = self.rows[j];
            let evals = self.compute_evals_odd_only_unreduced(
                j,
                even_checkpoint,
                inc.get_bound_coeff(row_o),
                eq.get_bound_coeff(row_o),
                gamma,
            );
            evals_accumulator[0] += evals[0];
            evals_accumulator[1] += evals[1];
            j += 1;
        }

        [
            F::from_montgomery_reduce(evals_accumulator[0]),
            F::from_montgomery_reduce(evals_accumulator[1]),
        ]
    }

    /// Compute evals when both even and odd entries are present.
    /// Returns `Unreduced<8>` to avoid the final Montgomery reduction.
    fn compute_evals_both_unreduced(
        &self,
        even_idx: usize,
        odd_idx: usize,
        inc_eval: F,
        eq_eval: F,
        gamma: F,
    ) -> [F::Unreduced<8>; 2] {
        debug_assert!(self.cols[even_idx].is_even());
        debug_assert!(self.cols[odd_idx].is_odd());
        debug_assert_eq!(self.rows[even_idx], self.rows[odd_idx]);

        let ra_even = self.ras[even_idx];
        let ra_odd = self.ras[odd_idx];
        let val_even = self.vals[even_idx];
        let val_odd = self.vals[odd_idx];

        let ra_evals = [ra_even, ra_odd + ra_odd - ra_even];
        let val_evals = [val_even, val_odd + val_odd - val_even];

        [
            eq_eval.mul_unreduced(ra_evals[0] * (val_evals[0] + gamma * (inc_eval + val_evals[0]))),
            eq_eval.mul_unreduced(ra_evals[1] * (val_evals[1] + gamma * (inc_eval + val_evals[1]))),
        ]
    }

    /// Compute evals when only even entry is present (odd is implicit).
    /// Returns `Unreduced<8>` to avoid the final Montgomery reduction.
    fn compute_evals_even_only_unreduced(
        &self,
        even_idx: usize,
        odd_checkpoint: F,
        inc_eval: F,
        eq_eval: F,
        gamma: F,
    ) -> [F::Unreduced<8>; 2] {
        let ra_even = self.ras[even_idx];
        let val_even = self.vals[even_idx];

        let ra_evals = [ra_even, -ra_even];
        let val_evals = [val_even, odd_checkpoint + odd_checkpoint - val_even];

        [
            eq_eval.mul_unreduced(ra_evals[0] * (val_evals[0] + gamma * (inc_eval + val_evals[0]))),
            eq_eval.mul_unreduced(ra_evals[1] * (val_evals[1] + gamma * (inc_eval + val_evals[1]))),
        ]
    }

    /// Compute evals when only odd entry is present (even is implicit).
    /// Returns `Unreduced<8>` to avoid the final Montgomery reduction.
    fn compute_evals_odd_only_unreduced(
        &self,
        odd_idx: usize,
        even_checkpoint: F,
        inc_eval: F,
        eq_eval: F,
        gamma: F,
    ) -> [F::Unreduced<8>; 2] {
        let ra_odd = self.ras[odd_idx];
        let val_odd = self.vals[odd_idx];

        let ra_evals = [F::zero(), ra_odd + ra_odd];
        let val_evals = [even_checkpoint, val_odd + val_odd - even_checkpoint];

        [
            F::Unreduced::<8>::zero(), // ra_evals[0] is zero
            eq_eval.mul_unreduced(ra_evals[1] * (val_evals[1] + gamma * (inc_eval + val_evals[1]))),
        ]
    }

    /// Materializes the ra and Val polynomials.
    /// Some number of cycle and address variables have already been bound, so at this point
    /// there are `K_prime` columns and `T_prime` rows left in the matrix.
    ///
    /// This expands the sparse representation to dense polynomials of size `K_prime * T_prime`.
    #[tracing::instrument(skip_all, name = "ReadWriteMatrixAddressMajor::materialize")]
    pub fn materialize(
        self,
        K_prime: usize,
        T_prime: usize,
    ) -> (MultilinearPolynomial<F>, MultilinearPolynomial<F>) {
        let len = K_prime * T_prime;

        // Initialize ra to zero, val will be filled column by column
        let mut ra: Vec<F> = unsafe_allocate_zero_vec(len);
        let mut val: Vec<F> = unsafe_allocate_zero_vec(len);

        let n = self.nnz();

        // Build a set of columns that have explicit entries
        let mut col_seen = vec![false; K_prime];
        for t in 0..n {
            let col_usize = self.cols[t].to_usize();
            if col_usize < K_prime {
                col_seen[col_usize] = true;
            }
        }

        // Process columns with explicit entries
        let mut i = 0;
        while i < n {
            let col = self.cols[i];
            let col_usize = col.to_usize();

            // Find end of this column's entries
            let mut j = i + 1;
            while j < n && self.cols[j] == col {
                j += 1;
            }

            let mut current_val = self.val_init.get_bound_coeff(col_usize);
            let mut ptr = i;

            for row in 0..T_prime {
                let idx_flat = col_usize * T_prime + row;

                if ptr < j && self.rows[ptr] == row {
                    // Explicit entry at (col, row)
                    ra[idx_flat] = self.ras[ptr];
                    val[idx_flat] = self.vals[ptr];
                    current_val = self.get_next_val(ptr);
                    ptr += 1;
                } else {
                    // Implicit entry: ra=0 (already zero), Val is carried forward
                    val[idx_flat] = current_val;
                }
            }

            i = j;
        }

        // Handle columns with no explicit entries - val is constant (init_val)
        for col in 0..K_prime {
            if !col_seen[col] {
                let init_val = self.val_init.get_bound_coeff(col);
                for row in 0..T_prime {
                    let idx_flat = col * T_prime + row;
                    // ra is already zero
                    val[idx_flat] = init_val;
                }
            }
        }

        (ra.into(), val.into())
    }
}

