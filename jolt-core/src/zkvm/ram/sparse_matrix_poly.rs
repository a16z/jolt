use std::cmp::Ordering;
use std::mem::MaybeUninit;
use std::sync::{Arc, Mutex};

use allocative::Allocative;
use num::Integer;
use rayon::prelude::*;

use crate::field::OptimizedMul;
use crate::poly::multilinear_polynomial::{BindingOrder, PolynomialBinding};
use crate::poly::unipoly::UniPoly;
use crate::zkvm::ram::remap_address;
use crate::{field::JoltField, poly::multilinear_polynomial::MultilinearPolynomial};
use ark_std::Zero;
use common::jolt_device::MemoryLayout;
use tracer::instruction::{Cycle, RAMAccess};

/// Represents a non-zero coefficient of the ra(k, j) polynomial and the
/// corresponding coefficient of the Val(k, j) polynomial. Conceptually,
/// both ra and Val can be seen as K x T matrices, hence `MatrixEntry`.
#[derive(Allocative, Debug, PartialEq, Clone, Copy)]
pub struct MatrixEntry<F: JoltField> {
    /// The row index. Before binding, row \in [0, T)
    pub row: usize,
    /// The column index. Before binding, col \in [0, K)
    pub col: usize,
    /// In round i, each MatrixEntry represents a coefficient
    ///   Val(k, j', r)
    /// which is some combination of Val(k, j', 00...0), ...
    /// Val(k, j', 11...1).
    /// `prev_val` contains the unbound coefficient before
    /// Val(k, j', 00...0) –– abusing notation, `prev_val` is
    /// Val(k, j'-1, 11...1)
    prev_val: F,
    /// In round i, each MatrixEntry represents a coefficient
    ///   Val(k, j', r)
    /// which is some combination of Val(k, j', 00...0), ...
    /// Val(k, j', 11...1).
    /// `next_val` contains the unbound coefficient after
    /// Val(k, j', 00...0) –– abusing notation, `next_val` is
    /// Val(k, j'+1, 00...0)
    next_val: F,
    /// The Val coefficient for this matrix entry.
    pub val_coeff: F,
    /// The ra coefficient for this matrix entry. Note that for RAM,
    /// ra and wa are the same polynomial.
    pub ra_coeff: F,
}

/// Represents the ra(k, j) and Val(k, j) polynomials for the RAM
/// read/write-checking sumcheck in cycle-major (row-major) order.
/// Conceptually, both ra and Val can be seen as K x T matrices.
/// This is far too large to explicitly store in memory, but we observe
/// that, while binding cycle variables, we only need a small fraction
/// of the coefficients for the purposes of sumcheck.
///
/// Entries are sorted by `(row, col)` i.e. cycle-major order.
/// This view is used for binding *cycle variables* first.
#[derive(Allocative, Debug, Default, Clone)]
pub struct ReadWriteMatrixCycleMajor<F: JoltField> {
    pub entries: Vec<MatrixEntry<F>>,
    val_init: MultilinearPolynomial<F>,
}

/// Represents the ra(k, j) and Val(k, j) polynomials for the RAM
/// read/write-checking sumcheck in address-major (column-major) order.
///
/// # Memory Layout: Struct-of-Arrays (SoA)
///
/// Unlike `ReadWriteMatrixCycleMajor` which uses Array-of-Structs (`Vec<MatrixEntry>`),
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
#[derive(Allocative, Debug, Default, Clone)]
pub struct ReadWriteMatrixAddressMajor<F: JoltField> {
    /// Row indices (cycle indices) for each sparse entry.
    pub rows: Vec<usize>,
    /// Column indices (address indices) for each sparse entry.
    pub cols: Vec<usize>,
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

impl<F: JoltField> ReadWriteMatrixAddressMajor<F> {
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
            self.val_final.get_bound_coeff(self.cols[i])
        }
    }
}

impl<F: JoltField> ReadWriteMatrixCycleMajor<F> {
    /// Creates a new `ReadWriteMatrixCycleMajor` to represent the ra and Val polynomials
    /// for the RAM read/write checking sumcheck.
    #[tracing::instrument(skip_all, name = "ReadWriteMatrixCycleMajor::new")]
    pub fn new(trace: &[Cycle], val_init: Vec<F>, memory_layout: &MemoryLayout) -> Self {
        let entries: Vec<_> = trace
            .par_iter()
            .enumerate()
            .filter_map(|(j, cycle)| {
                let ram_op = cycle.ram_access();
                match ram_op {
                    RAMAccess::Write(write) => {
                        let pre_value = F::from_u64(write.pre_value);
                        let post_value = F::from_u64(write.post_value);
                        Some(MatrixEntry {
                            row: j,
                            col: remap_address(write.address, memory_layout).unwrap() as usize,
                            ra_coeff: F::one(),
                            val_coeff: pre_value,
                            prev_val: pre_value,
                            next_val: post_value,
                        })
                    }
                    RAMAccess::Read(read) => {
                        let read_value = F::from_u64(read.value);
                        Some(MatrixEntry {
                            row: j,
                            col: remap_address(read.address, memory_layout).unwrap() as usize,
                            ra_coeff: F::one(),
                            val_coeff: read_value,
                            prev_val: read_value,
                            next_val: read_value,
                        })
                    }
                    _ => None,
                }
            })
            .collect();

        ReadWriteMatrixCycleMajor {
            entries,
            val_init: val_init.into(),
        }
    }

    /// Binds two adjacent rows in the sparse matrix together with the randomness `r`.
    /// This is a parallel, recursive function (similar to a parallel merge of two
    /// sorted lists) that assumes `even_row` and `odd_row` are sorted by column
    /// (i.e. address) and (b) writes the output (i.e. bound row) to the `out` buffer
    /// in sorted order as well.
    ///
    /// Returns the number of entries in the bound row.
    /// If the `dry_run` parameter is true, `bind_rows` ignores `out` and just computes
    /// the number of entries that would be in the bound row, which can be used to
    /// allocate the exact amount of memory needed in the subsequent "real" bind operation.
    fn bind_rows(
        even_row: &[MatrixEntry<F>],
        odd_row: &[MatrixEntry<F>],
        r: F::Challenge,
        out: &mut [MaybeUninit<MatrixEntry<F>>],
        dry_run: bool,
    ) -> usize {
        /// Threshold where we stop parallelizing and do a plain linear merge.
        const PAR_THRESHOLD: usize = 32_768;

        // small inputs: do the O(n) sequential merge
        if even_row.len() + odd_row.len() <= PAR_THRESHOLD {
            return Self::seq_bind_rows(even_row, odd_row, r, out, dry_run);
        }

        // Split the longer row at its midpoint; find where that pivot would land in the other row.
        let (even_pivot_idx, odd_pivot_idx) = if even_row.len() > odd_row.len() {
            let even_pivot_idx = even_row.len() / 2;
            let pivot = even_row[even_pivot_idx].col;
            let odd_pivot_idx = odd_row.partition_point(|x| x.col < pivot);
            (even_pivot_idx, odd_pivot_idx)
        } else {
            let odd_pivot_idx = odd_row.len() / 2;
            let pivot = odd_row[odd_pivot_idx].col;
            let even_pivot_idx = even_row.partition_point(|x| x.col < pivot);
            (even_pivot_idx, odd_pivot_idx)
        };

        let out_len = out.len();
        let (left_out, right_out) = if dry_run {
            // `out` may be empty in a dry run
            out.split_at_mut(0)
        } else {
            out.split_at_mut(even_pivot_idx + odd_pivot_idx)
        };

        // Now we know the global order: everything in even_row[..even_pivot_idx] and
        // odd_row[..odd_pivot_idx] comes before everything in even_row[even_pivot_idx..]
        // and odd_row[odd_pivot_idx..]. Compute the merged lengths of each half
        let (left_merged_len, right_merged_len) = rayon::join(
            || {
                Self::bind_rows(
                    &even_row[..even_pivot_idx],
                    &odd_row[..odd_pivot_idx],
                    r,
                    left_out,
                    true,
                )
            },
            || {
                Self::bind_rows(
                    &even_row[even_pivot_idx..],
                    &odd_row[odd_pivot_idx..],
                    r,
                    right_out,
                    true,
                )
            },
        );

        if !dry_run {
            assert_eq!(out_len, left_merged_len + right_merged_len);
            let (left_out, right_out) = out.split_at_mut(left_merged_len);
            // If not a dry run, perform the actual merge now.
            rayon::join(
                || {
                    Self::bind_rows(
                        &even_row[..even_pivot_idx],
                        &odd_row[..odd_pivot_idx],
                        r,
                        left_out,
                        false,
                    )
                },
                || {
                    Self::bind_rows(
                        &even_row[even_pivot_idx..],
                        &odd_row[odd_pivot_idx..],
                        r,
                        right_out,
                        false,
                    )
                },
            );
        }

        left_merged_len + right_merged_len
    }

    /// Binds two adjacent rows in the sparse matrix together with the randomness `r`.
    /// This is a sequential function (unlike `bind_rows`) that assumes `even_row` and
    /// `odd_row` are sorted by column (i.e. address) and (b) writes the output (i.e.
    /// bound row) to the `out` buffer in sorted order as well.
    ///
    /// Returns the number of entries in the bound row.
    /// If the `dry_run` parameter is true, `bind_rows` ignores `out` and just computes
    /// the number of entries that would be in the bound row, which can be used to
    /// allocate the exact amount of memory needed in the subsequent "real" bind operation.
    fn seq_bind_rows(
        even: &[MatrixEntry<F>],
        odd: &[MatrixEntry<F>],
        r: F::Challenge,
        out: &mut [MaybeUninit<MatrixEntry<F>>],
        dry_run: bool,
    ) -> usize {
        // Even index
        let mut i = 0;
        // Odd index
        let mut j = 0;
        // Out index
        let mut k = 0;

        while i < even.len() && j < odd.len() {
            if even[i].col == odd[j].col {
                if !dry_run {
                    let bound_entry = Self::bind_entries(Some(&even[i]), Some(&odd[j]), r);
                    out[k] = MaybeUninit::new(bound_entry);
                }
                i += 1;
                j += 1;
                k += 1;
            } else if even[i].col < odd[j].col {
                if !dry_run {
                    let bound_entry = Self::bind_entries(Some(&even[i]), None, r);
                    out[k] = MaybeUninit::new(bound_entry);
                }
                i += 1;
                k += 1;
            } else {
                if !dry_run {
                    let bound_entry = Self::bind_entries(None, Some(&odd[j]), r);
                    out[k] = MaybeUninit::new(bound_entry);
                }
                j += 1;
                k += 1;
            }
        }
        for remaining_even_entry in even[i..].iter() {
            if !dry_run {
                let bound_entry = Self::bind_entries(Some(remaining_even_entry), None, r);
                out[k] = MaybeUninit::new(bound_entry);
            }
            k += 1;
        }
        for remaining_odd_entry in odd[j..].iter() {
            if !dry_run {
                let bound_entry = Self::bind_entries(None, Some(remaining_odd_entry), r);
                out[k] = MaybeUninit::new(bound_entry);
            }
            k += 1;
        }
        if !dry_run {
            assert_eq!(out.len(), k);
        }
        k
    }

    /// Binds adjacent entries of the matrix together using the random challenge `r`.
    /// By "adjacent", here we mean entries that are in the same column and adjacent
    /// rows (rows 2j and 2j+1).
    /// Either `even` or `odd` may be `None`, indicating that the corresponding matrix
    /// entry is not explicitly represented in the `ReadWriteMatrixCycleMajor` data structure.
    /// Instead, we can infer its values from the matrix entry that is `Some`.
    fn bind_entries(
        even: Option<&MatrixEntry<F>>,
        odd: Option<&MatrixEntry<F>>,
        r: F::Challenge,
    ) -> MatrixEntry<F> {
        match (even, odd) {
            (Some(even), Some(odd)) => {
                debug_assert!(even.row.is_even());
                debug_assert!(odd.row.is_odd());
                debug_assert_eq!(even.col, odd.col);
                MatrixEntry {
                    row: even.row / 2,
                    col: even.col,
                    ra_coeff: even.ra_coeff + r.mul_0_optimized(odd.ra_coeff - even.ra_coeff),
                    val_coeff: even.val_coeff + r.mul_0_optimized(odd.val_coeff - even.val_coeff),
                    prev_val: even.prev_val,
                    next_val: odd.next_val,
                }
            }
            (Some(even), None) => {
                // For ReadWriteMatrixCycleMajor, the absence of a matrix entry implies
                // that its coeff has not been bound yet.
                // The absence of an odd-row entry in the same column as even
                // means that its implicit Val coeff is even.next, and its implicit
                // ra coeff is 0.
                let odd_val_coeff = even.next_val;
                MatrixEntry {
                    row: even.row / 2,
                    col: even.col,
                    ra_coeff: (F::one() - r).mul_1_optimized(even.ra_coeff),
                    val_coeff: even.val_coeff + r.mul_0_optimized(odd_val_coeff - even.val_coeff),
                    prev_val: even.prev_val,
                    next_val: even.next_val,
                }
            }
            (None, Some(odd)) => {
                // For ReadWriteMatrixCycleMajor, the absence of a matrix entry implies
                // that its coeff has not been bound yet.
                // The absence of an even-row entry in the same column as odd
                // means that its implicit Val coeff is odd.prev, and its implicit
                // ra coeff is 0.
                let even_val_coeff = odd.prev_val;
                MatrixEntry {
                    row: odd.row / 2,
                    col: odd.col,
                    ra_coeff: r.mul_1_optimized(odd.ra_coeff),
                    val_coeff: even_val_coeff + r.mul_0_optimized(odd.val_coeff - even_val_coeff),
                    prev_val: odd.prev_val,
                    next_val: odd.next_val,
                }
            }
            (None, None) => panic!("Both entries are None"),
        }
    }

    /// Binds a cycle variable of the ra and Val polynomials represented by
    /// this `ReadWriteMatrixCycleMajor` to the random challenge `r`.
    #[tracing::instrument(skip_all, name = "ReadWriteMatrixCycleMajor::bind")]
    pub fn bind(&mut self, r: F::Challenge) {
        let row_lengths: Vec<_> = self
            .entries
            .par_chunk_by(|x, y| x.row / 2 == y.row / 2)
            .map(|entries| {
                let odd_row_start_index = entries.partition_point(|entry| entry.row.is_even());
                let (even_row, odd_row) = entries.split_at(odd_row_start_index);
                // Dry run to compute output length
                let bound_len = Self::bind_rows(even_row, odd_row, r, &mut [], true);
                (entries.len(), bound_len)
            })
            .collect();

        let bound_length = row_lengths.iter().map(|(_, bound_len)| bound_len).sum();
        let mut bound_entries: Vec<MatrixEntry<F>> = Vec::with_capacity(bound_length);
        let mut bound_entries_slice = bound_entries.spare_capacity_mut();
        let mut unbound_entries_slice = self.entries.as_slice();

        let mut output_slices = Vec::with_capacity(row_lengths.len());
        let mut input_slices = Vec::with_capacity(row_lengths.len());

        // Split `self.entries` and the output buffer into vectors of non-overlapping slices
        // that can be zipped together and parallelized over.
        for (unbound_len, bound_len) in row_lengths.iter() {
            let output_slice;
            (output_slice, bound_entries_slice) = bound_entries_slice.split_at_mut(*bound_len);
            output_slices.push(output_slice);
            let input_slice;
            (input_slice, unbound_entries_slice) = unbound_entries_slice.split_at(*unbound_len);
            input_slices.push(input_slice);
        }

        input_slices
            .par_iter()
            .zip(output_slices.into_par_iter())
            .for_each(|(input_slice, output_slice)| {
                let odd_row_start_index = input_slice.partition_point(|entry| entry.row.is_even());
                let (even_row, odd_row) = input_slice.split_at(odd_row_start_index);
                let _ = Self::bind_rows(even_row, odd_row, r, output_slice, false);
            });

        unsafe {
            bound_entries.set_len(bound_length);
        }
        self.entries = bound_entries;
    }

    /// For the given pair of adjacent rows, computes the pair's contribution to the prover's
    /// sumcheck message. This is a recursive, parallel algorithm.
    pub fn prover_message_contribution(
        even_row: &[MatrixEntry<F>],
        odd_row: &[MatrixEntry<F>],
        inc_evals: [F; 2],
        gamma: F,
    ) -> [F; 2] {
        /// Threshold where we stop parallelizing and do a plain linear merge.
        const PAR_THRESHOLD: usize = 32_768;

        // small inputs: do the O(n) sequential algorithm
        if even_row.len() + odd_row.len() <= PAR_THRESHOLD {
            return Self::seq_prover_message_contribution(even_row, odd_row, inc_evals, gamma);
        }

        // Split the longer row at its midpoint; find where that pivot would land in the other row.
        let (even_pivot_idx, odd_pivot_idx) = if even_row.len() > odd_row.len() {
            let even_pivot_idx = even_row.len() / 2;
            let pivot = even_row[even_pivot_idx].col;
            let odd_pivot_idx = odd_row.partition_point(|x| x.col < pivot);
            (even_pivot_idx, odd_pivot_idx)
        } else {
            let odd_pivot_idx = odd_row.len() / 2;
            let pivot = odd_row[odd_pivot_idx].col;
            let even_pivot_idx = even_row.partition_point(|x| x.col < pivot);
            (even_pivot_idx, odd_pivot_idx)
        };

        // Now we know the global order: everything in even_row[..even_pivot_idx] and
        // odd_row[..odd_pivot_idx] comes before everything in even_row[even_pivot_idx..]
        // and odd_row[odd_pivot_idx..]. Compute each half's contribution in parallel.
        let (left_evals, right_evals) = rayon::join(
            || {
                Self::prover_message_contribution(
                    &even_row[..even_pivot_idx],
                    &odd_row[..odd_pivot_idx],
                    inc_evals,
                    gamma,
                )
            },
            || {
                Self::prover_message_contribution(
                    &even_row[even_pivot_idx..],
                    &odd_row[odd_pivot_idx..],
                    inc_evals,
                    gamma,
                )
            },
        );

        [
            left_evals[0] + right_evals[0],
            left_evals[1] + right_evals[1],
        ]
    }

    /// For the given pair of adjacent rows, computes the pair's contribution to the prover's
    /// sumcheck message. This is the sequential counterpart of `prover_message_contribution`.
    fn seq_prover_message_contribution(
        even: &[MatrixEntry<F>],
        odd: &[MatrixEntry<F>],
        inc_evals: [F; 2],
        gamma: F,
    ) -> [F; 2] {
        let mut i = 0;
        let mut j = 0;
        let mut evals_accumulator = [F::zero(); 2];

        while i < even.len() && j < odd.len() {
            if even[i].col == odd[j].col {
                let evals = Self::compute_evals(Some(&even[i]), Some(&odd[j]), inc_evals, gamma);
                evals_accumulator[0] += evals[0];
                evals_accumulator[1] += evals[1];
                i += 1;
                j += 1;
            } else if even[i].col < odd[j].col {
                let evals = Self::compute_evals(Some(&even[i]), None, inc_evals, gamma);
                evals_accumulator[0] += evals[0];
                evals_accumulator[1] += evals[1];
                i += 1;
            } else {
                let evals = Self::compute_evals(None, Some(&odd[j]), inc_evals, gamma);
                evals_accumulator[0] += evals[0];
                evals_accumulator[1] += evals[1];
                j += 1;
            }
        }
        for remaining_even_entry in even[i..].iter() {
            let evals = Self::compute_evals(Some(remaining_even_entry), None, inc_evals, gamma);
            evals_accumulator[0] += evals[0];
            evals_accumulator[1] += evals[1];
        }
        for remaining_odd_entry in odd[j..].iter() {
            let evals = Self::compute_evals(None, Some(remaining_odd_entry), inc_evals, gamma);
            evals_accumulator[0] += evals[0];
            evals_accumulator[1] += evals[1];
        }

        evals_accumulator
    }

    /// For the given pair of adjacent entries, computes the pair's contribution to the prover's
    /// sumcheck message. By "adjacent", here we mean entries that are in the same column and
    /// adjacent rows (rows 2j and 2j+1).
    /// Either `even` or `odd` may be `None`, indicating that the corresponding matrix
    /// entry is not explicitly represented in the `ReadWriteMatrixCycleMajor` data structure.
    /// Instead, we can infer its values from the matrix entry that is `Some`.
    fn compute_evals(
        even: Option<&MatrixEntry<F>>,
        odd: Option<&MatrixEntry<F>>,
        inc_evals: [F; 2],
        gamma: F,
    ) -> [F; 2] {
        match (even, odd) {
            (Some(even), Some(odd)) => {
                debug_assert!(even.row.is_even());
                debug_assert!(odd.row.is_odd());
                debug_assert_eq!(even.col, odd.col);
                let ra_evals = [even.ra_coeff, odd.ra_coeff - even.ra_coeff];
                let val_evals = [even.val_coeff, odd.val_coeff - even.val_coeff];
                [
                    ra_evals[0] * (val_evals[0] + gamma * (inc_evals[0] + val_evals[0])),
                    ra_evals[1] * (val_evals[1] + gamma * (inc_evals[1] + val_evals[1])),
                ]
            }
            (Some(even), None) => {
                // For ReadWriteMatrixCycleMajor, the absence of a matrix entry implies
                // that its coeff has not been bound yet.
                // The absence of an odd-row entry in the same column as even
                // means that its implicit Val coeff is even.next, and its implicit
                // ra coeff is 0.
                let odd_val_coeff = even.next_val;
                let ra_evals = [even.ra_coeff, -even.ra_coeff];
                let val_evals = [even.val_coeff, odd_val_coeff - even.val_coeff];
                [
                    ra_evals[0] * (val_evals[0] + gamma * (inc_evals[0] + val_evals[0])),
                    ra_evals[1] * (val_evals[1] + gamma * (inc_evals[1] + val_evals[1])),
                ]
            }
            (None, Some(odd)) => {
                // For ReadWriteMatrixCycleMajor, the absence of a matrix entry implies
                // that its coeff has not been bound yet.
                // The absence of an even-row entry in the same column as odd
                // means that its implicit Val coeff is odd.prev, and its implicit
                // ra coeff is 0.
                let even_val_coeff = odd.prev_val;
                let ra_evals = [F::zero(), odd.ra_coeff];
                let val_evals = [even_val_coeff, odd.val_coeff - even_val_coeff];
                [
                    F::zero(), // ra_evals[0] is zero
                    ra_evals[1] * (val_evals[1] + gamma * (inc_evals[1] + val_evals[1])),
                ]
            }
            (None, None) => panic!("Both entries are None"),
        }
    }

    /// Materializes the ra and Val polynomials represented by this `ReadWriteMatrixCycleMajor`.
    /// All cycle variables must be bound at this point, so the materialized ra and Val
    /// have K coefficients each.
    #[tracing::instrument(skip_all, name = "ReadWriteMatrixCycleMajor::materialize")]
    pub fn materialize(
        self,
        K: usize,
        val_init: &[F],
    ) -> (MultilinearPolynomial<F>, MultilinearPolynomial<F>) {
        // Initialize ra and Val to initial values
        let ra: Vec<Arc<Mutex<F>>> = (0..K)
            .into_par_iter()
            .map(|_| Arc::new(Mutex::new(F::zero())))
            .collect();
        let val: Vec<Arc<Mutex<F>>> = val_init
            .par_iter()
            .map(|&x| Arc::new(Mutex::new(x)))
            .collect();
        // Update some of the ra and Val coefficients based on
        // matrix entries.
        self.entries.into_par_iter().for_each(|entry| {
            debug_assert_eq!(entry.row, 0);
            let k = entry.col;
            *ra[k].lock().unwrap() = entry.ra_coeff;
            *val[k].lock().unwrap() = entry.val_coeff;
        });
        // Unwrap Arc<Mutex<F>> back into F
        let ra: Vec<F> = ra
            .into_par_iter()
            .map(|arc_mutex| *arc_mutex.lock().unwrap())
            .collect();
        let val: Vec<F> = val
            .into_par_iter()
            .map(|arc_mutex| *arc_mutex.lock().unwrap())
            .collect();
        // Convert Vec<F> to MultilinearPolynomial<F>
        (ra.into(), val.into())
    }
}

impl<F: JoltField> From<ReadWriteMatrixCycleMajor<F>> for ReadWriteMatrixAddressMajor<F> {
    #[tracing::instrument(skip_all, name = "ReadWriteMatrixAddressMajor::from")]
    fn from(mut cycle_major: ReadWriteMatrixCycleMajor<F>) -> Self {
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
        let cols: Vec<usize> = entries.par_iter().map(|e| e.col).collect();
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
                let col = column_entries[0].col;
                let last_next_val = column_entries.last().unwrap().next_val;
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

impl<F: JoltField> ReadWriteMatrixAddressMajor<F> {
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
            input_start: usize, // Start index in input arrays
            input_end: usize,   // End index in input arrays
            even_end: usize,    // Boundary between even and odd entries
            even_col_idx: usize,
            bound_len: usize,
        }

        // Find column-pair boundaries and compute bound lengths in parallel
        let pair_ranges: Vec<(usize, usize)> = {
            let mut ranges = Vec::new();
            let mut idx = 0;
            while idx < n {
                let col_pair = self.cols[idx] / 2;
                let mut j = idx + 1;
                while j < n && self.cols[j] / 2 == col_pair {
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
                let col_pair = self.cols[start] / 2;

                // Find boundary between even and odd column entries
                let mut mid = start;
                while mid < end && self.cols[mid] % 2 == 0 {
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
        let mut cols_new: Vec<MaybeUninit<usize>> = Vec::with_capacity(total_bound);
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
        let mut cols_slices: Vec<&mut [MaybeUninit<usize>]> = Vec::with_capacity(pairs.len());
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
        self.cols = unsafe { std::mem::transmute::<Vec<MaybeUninit<usize>>, Vec<usize>>(cols_new) };
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
        cols_out: &mut [MaybeUninit<usize>],
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
        cols_out: &mut [MaybeUninit<usize>],
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
                    let new_col = self.cols[i] / 2;
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
                    let new_col = self.cols[i] / 2;
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
                    let new_col = self.cols[j] / 2;
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
                let new_col = self.cols[i] / 2;
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
                let new_col = self.cols[j] / 2;
                let ra_odd = self.ras[j];
                let val_odd = self.vals[j];

                rows_out[k] = MaybeUninit::new(row_o);
                cols_out[k] = MaybeUninit::new(new_col);
                ras_out[k] = MaybeUninit::new(r.mul_1_optimized(ra_odd));
                vals_out[k] =
                    MaybeUninit::new(even_checkpoint + r.mul_0_optimized(val_odd - even_checkpoint));
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
                let col_pair = self.cols[idx] / 2;
                let mut j = idx + 1;
                while j < n && self.cols[j] / 2 == col_pair {
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
                let col_pair = self.cols[start] / 2;

                // Find boundary between even and odd column entries
                let mut mid = start;
                while mid < end && self.cols[mid] % 2 == 0 {
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
        let mut evals_accumulator = [F::zero(); 2];

        while i < e1 && j < o1 {
            let row_e = self.rows[i];
            let row_o = self.rows[j];

            if row_e == row_o {
                let evals = self.compute_evals_both(
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
                let evals = self.compute_evals_even_only(
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
                let evals = self.compute_evals_odd_only(
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
            let evals = self.compute_evals_even_only(
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
            let evals = self.compute_evals_odd_only(
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

        evals_accumulator
    }

    /// Compute evals when both even and odd entries are present.
    fn compute_evals_both(
        &self,
        even_idx: usize,
        odd_idx: usize,
        inc_eval: F,
        eq_eval: F,
        gamma: F,
    ) -> [F; 2] {
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
            eq_eval * ra_evals[0] * (val_evals[0] + gamma * (inc_eval + val_evals[0])),
            eq_eval * ra_evals[1] * (val_evals[1] + gamma * (inc_eval + val_evals[1])),
        ]
    }

    /// Compute evals when only even entry is present (odd is implicit).
    fn compute_evals_even_only(
        &self,
        even_idx: usize,
        odd_checkpoint: F,
        inc_eval: F,
        eq_eval: F,
        gamma: F,
    ) -> [F; 2] {
        let ra_even = self.ras[even_idx];
        let val_even = self.vals[even_idx];

        let ra_evals = [ra_even, -ra_even];
        let val_evals = [val_even, odd_checkpoint + odd_checkpoint - val_even];

        [
            eq_eval * ra_evals[0] * (val_evals[0] + gamma * (inc_eval + val_evals[0])),
            eq_eval * ra_evals[1] * (val_evals[1] + gamma * (inc_eval + val_evals[1])),
        ]
    }

    /// Compute evals when only odd entry is present (even is implicit).
    fn compute_evals_odd_only(
        &self,
        odd_idx: usize,
        even_checkpoint: F,
        inc_eval: F,
        eq_eval: F,
        gamma: F,
    ) -> [F; 2] {
        let ra_odd = self.ras[odd_idx];
        let val_odd = self.vals[odd_idx];

        let ra_evals = [F::zero(), ra_odd + ra_odd];
        let val_evals = [even_checkpoint, val_odd + val_odd - even_checkpoint];

        [
            F::zero(), // ra_evals[0] is zero
            eq_eval * ra_evals[1] * (val_evals[1] + gamma * (inc_eval + val_evals[1])),
        ]
    }

    /// Materializes the ra and Val polynomials.
    /// Some number of cycle and address variables have already been bound, so at this point
    /// there are `K_prime` columns and `T_prime` rows left in the matrix.
    #[tracing::instrument(skip_all, name = "ReadWriteMatrixAddressMajor::materialize")]
    pub fn materialize(
        self,
        K_prime: usize,
        T_prime: usize,
    ) -> (MultilinearPolynomial<F>, MultilinearPolynomial<F>) {
        // Initialize ra and Val to initial values
        let ra: Vec<Arc<Mutex<F>>> = (0..K_prime * T_prime)
            .into_par_iter()
            .map(|_| Arc::new(Mutex::new(F::zero())))
            .collect();
        let val: Vec<Arc<Mutex<F>>> = (0..K_prime * T_prime)
            .into_par_iter()
            .map(|_| Arc::new(Mutex::new(F::zero())))
            .collect();

        let n = self.nnz();

        // Process column by column using SoA data
        let mut i = 0;
        while i < n {
            let col = self.cols[i];

            // Find end of this column's entries
            let mut j = i + 1;
            while j < n && self.cols[j] == col {
                j += 1;
            }

            let mut current_val = self.val_init.get_bound_coeff(col);
            let mut ptr = i;

            for row in 0..T_prime {
                let idx_flat = col * T_prime + row;

                if ptr < j && self.rows[ptr] == row {
                    // Explicit entry at (col, row)
                    *ra[idx_flat].lock().unwrap() = self.ras[ptr];
                    *val[idx_flat].lock().unwrap() = self.vals[ptr];
                    current_val = self.get_next_val(ptr);
                    ptr += 1;
                } else {
                    // Implicit entry: ra=0, Val is carried forward
                    *val[idx_flat].lock().unwrap() = current_val;
                }
            }

            i = j;
        }

        // Handle columns with no explicit entries
        let mut col_seen = vec![false; K_prime];
        for t in 0..n {
            if self.cols[t] < K_prime {
                col_seen[self.cols[t]] = true;
            }
        }

        for col in 0..K_prime {
            if !col_seen[col] {
                let init_val = self.val_init.get_bound_coeff(col);
                for row in 0..T_prime {
                    let idx_flat = col * T_prime + row;
                    *val[idx_flat].lock().unwrap() = init_val;
                }
            }
        }

        // Unwrap Arc<Mutex<F>> back into F
        let ra: Vec<F> = ra
            .into_par_iter()
            .map(|arc_mutex| *arc_mutex.lock().unwrap())
            .collect();
        let val: Vec<F> = val
            .into_par_iter()
            .map(|arc_mutex| *arc_mutex.lock().unwrap())
            .collect();

        (ra.into(), val.into())
    }
}
