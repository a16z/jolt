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

pub type MatrixOrder = bool;
pub const ROW_MAJOR: MatrixOrder = false;
pub const COL_MAJOR: MatrixOrder = true;

/// Represents the ra(k, j) and Val(k, j) polynomials for the RAM
/// read/write-checking sumcheck. Conceptually, both ra and Val can
/// be seen as K x T matrices. This is far too large to explicitly
/// store in memory, but we observe that, while binding cycle variables,
/// we only need a small fraction of the coefficients for the purposes
/// of sumcheck. The coefficients we do need are stored in this data structure.
#[derive(Allocative, Debug, Default, Clone)]
pub struct SparseMatrixPolynomial<const ORDER: MatrixOrder, F: JoltField> {
    pub entries: Vec<MatrixEntry<F>>,
    val_init: MultilinearPolynomial<F>,
}

impl<F: JoltField> SparseMatrixPolynomial<ROW_MAJOR, F> {
    /// Creates a new `SparseMatrixPolynomial` to represent the ra and Val polynomials
    /// for the RAM read/write checking sumcheck.
    #[tracing::instrument(skip_all, name = "SparseMatrixPolynomial::new")]
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

        SparseMatrixPolynomial {
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
    /// entry is not explicitly represented in the `SparseMatrixPolynomial` data structure.
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
                // For SparseMatrixPolynomial, the absence of a matrix entry implies
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
                // For SparseMatrixPolynomial, the absence of a matrix entry implies
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
    /// this `SparseMatrixPolynomial` to the random challenge `r`.
    #[tracing::instrument(skip_all, name = "SparseMatrixPolynomial::bind")]
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
    /// entry is not explicitly represented in the `SparseMatrixPolynomial` data structure.
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
                // For SparseMatrixPolynomial, the absence of a matrix entry implies
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
                // For SparseMatrixPolynomial, the absence of a matrix entry implies
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

    /// Materializes the ra and Val polynomials represented by this `SparseMatrixPolynomial`.
    /// All cycle variables must be bound at this point, so the materialized ra and Val
    /// have K coefficients each.
    #[tracing::instrument(skip_all, name = "SparseMatrixPolynomial::materialize")]
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

impl<F: JoltField> From<SparseMatrixPolynomial<ROW_MAJOR, F>>
    for SparseMatrixPolynomial<COL_MAJOR, F>
{
    fn from(mut row_major: SparseMatrixPolynomial<ROW_MAJOR, F>) -> Self {
        let mut entries = std::mem::take(&mut row_major.entries);
        let val_init = std::mem::take(&mut row_major.val_init);
        entries.par_sort_by(|a, b| match a.col.cmp(&b.col) {
            Ordering::Less => Ordering::Less,
            Ordering::Greater => Ordering::Greater,
            Ordering::Equal => a.row.cmp(&b.row),
        });
        SparseMatrixPolynomial { entries, val_init }
    }
}

impl<F: JoltField> SparseMatrixPolynomial<COL_MAJOR, F> {
    /// Binds an address variable of the ra and Val polynomials represented by
    /// this `SparseMatrixPolynomial` to the random challenge `r`.
    #[tracing::instrument(skip_all, name = "SparseMatrixPolynomial::bind")]
    pub fn bind(&mut self, r: F::Challenge) {
        let col_lengths: Vec<_> = self
            .entries
            .par_chunk_by(|x, y| x.col / 2 == y.col / 2)
            .map(|entries| {
                let odd_col_start_index = entries.partition_point(|entry| entry.col.is_even());
                let (even_col, odd_col) = entries.split_at(odd_col_start_index);
                // Dry run to compute output length
                let bound_len = Self::bind_cols(
                    even_col,
                    odd_col,
                    // Don't need checkpoints for dry run
                    F::zero(),
                    F::zero(),
                    r,
                    &mut [],
                    true,
                );
                (entries.len(), bound_len)
            })
            .collect();

        let bound_length = col_lengths.iter().map(|(_, bound_len)| bound_len).sum();
        let mut bound_entries: Vec<MatrixEntry<F>> = Vec::with_capacity(bound_length);
        let mut bound_entries_slice = bound_entries.spare_capacity_mut();
        let mut unbound_entries_slice = self.entries.as_slice();

        let mut output_slices = Vec::with_capacity(col_lengths.len());
        let mut input_slices = Vec::with_capacity(col_lengths.len());

        // Split `self.entries` and the output buffer into vectors of non-overlapping slices
        // that can be zipped together and parallelized over.
        for (unbound_len, bound_len) in col_lengths.iter() {
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
                let odd_col_start_index = input_slice.partition_point(|entry| entry.col.is_even());
                let (even_col, odd_col) = input_slice.split_at(odd_col_start_index);
                let even_col_idx = 2 * (input_slice[0].col / 2);
                let odd_col_idx = even_col_idx + 1;
                let _ = Self::bind_cols(
                    even_col,
                    odd_col,
                    self.val_init.get_bound_coeff(even_col_idx),
                    self.val_init.get_bound_coeff(odd_col_idx),
                    r,
                    output_slice,
                    false,
                );
            });

        unsafe {
            bound_entries.set_len(bound_length);
        }
        self.entries = bound_entries;
        self.val_init.bind_parallel(r, BindingOrder::LowToHigh);
    }

    /// Binds two adjacent columns in the sparse matrix together with the randomness `r`.
    /// This is a parallel, recursive function (similar to a parallel merge of two
    /// sorted lists) that assumes `even_col` and `odd_col` are sorted by row
    /// (i.e. cycle) and (b) writes the output (i.e. bound column) to the `out` buffer
    /// in sorted order as well.
    ///
    /// Returns the number of entries in the bound column.
    /// If the `dry_run` parameter is true, `bind_cols` ignores `out` and just computes
    /// the number of entries that would be in the bound column, which can be used to
    /// allocate the exact amount of memory needed in the subsequent "real" bind operation.
    fn bind_cols(
        even_col: &[MatrixEntry<F>],
        odd_col: &[MatrixEntry<F>],
        even_checkpoint: F,
        odd_checkpoint: F,
        r: F::Challenge,
        out: &mut [MaybeUninit<MatrixEntry<F>>],
        dry_run: bool,
    ) -> usize {
        /// Threshold where we stop parallelizing and do a plain linear merge.
        const PAR_THRESHOLD: usize = 32_768;

        // small inputs: do the O(n) sequential merge
        if even_col.len() + odd_col.len() <= PAR_THRESHOLD {
            return Self::seq_bind_cols(
                even_col,
                odd_col,
                even_checkpoint,
                odd_checkpoint,
                r,
                out,
                dry_run,
            );
        }

        // Split the longer col at its midpoint; find where that pivot would land in the other col.
        let (even_pivot_idx, odd_pivot_idx) = if even_col.len() > odd_col.len() {
            let even_pivot_idx = even_col.len() / 2;
            let pivot = even_col[even_pivot_idx].row;
            let odd_pivot_idx = odd_col.partition_point(|x| x.row < pivot);
            (even_pivot_idx, odd_pivot_idx)
        } else {
            let odd_pivot_idx = odd_col.len() / 2;
            let pivot = odd_col[odd_pivot_idx].row;
            let even_pivot_idx = even_col.partition_point(|x| x.row < pivot);
            (even_pivot_idx, odd_pivot_idx)
        };

        let out_len = out.len();
        let (left_out, right_out) = if dry_run {
            // `out` may be empty in a dry run
            out.split_at_mut(0)
        } else {
            out.split_at_mut(even_pivot_idx + odd_pivot_idx)
        };

        // Now we know the global order: everything in even_col[..even_pivot_idx] and
        // odd_col[..odd_pivot_idx] comes before everything in even_col[even_pivot_idx..]
        // and odd_col[odd_pivot_idx..]. Compute the merged lengths of each half
        let (left_merged_len, right_merged_len) = rayon::join(
            || {
                Self::bind_cols(
                    &even_col[..even_pivot_idx],
                    &odd_col[..odd_pivot_idx],
                    // Don't need checkpoints for dry run
                    F::zero(),
                    F::zero(),
                    r,
                    left_out,
                    true,
                )
            },
            || {
                Self::bind_cols(
                    &even_col[even_pivot_idx..],
                    &odd_col[odd_pivot_idx..],
                    // Don't need checkpoints for dry run
                    F::zero(),
                    F::zero(),
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
                    Self::bind_cols(
                        &even_col[..even_pivot_idx],
                        &odd_col[..odd_pivot_idx],
                        even_checkpoint,
                        odd_checkpoint,
                        r,
                        left_out,
                        false,
                    )
                },
                || {
                    let even_checkpoint = if even_col.is_empty() {
                        even_checkpoint
                    } else if even_pivot_idx != 0 {
                        even_col[even_pivot_idx - 1].next_val
                    } else {
                        even_col[even_pivot_idx].prev_val
                    };
                    let odd_checkpoint = if odd_col.is_empty() {
                        odd_checkpoint
                    } else if odd_pivot_idx != 0 {
                        odd_col[odd_pivot_idx - 1].next_val
                    } else {
                        odd_col[odd_pivot_idx].prev_val
                    };
                    Self::bind_cols(
                        &even_col[even_pivot_idx..],
                        &odd_col[odd_pivot_idx..],
                        even_checkpoint,
                        odd_checkpoint,
                        r,
                        right_out,
                        false,
                    )
                },
            );
        }

        left_merged_len + right_merged_len
    }

    /// Binds two adjacent columns in the sparse matrix together with the randomness `r`.
    /// This is a sequential function (unlike `bind_cols`) that assumes `even_col` and
    /// `odd_col` are sorted by row (i.e. cycle) and (b) writes the output (i.e.
    /// bound column) to the `out` buffer in sorted order as well.
    ///
    /// Returns the number of entries in the bound column.
    /// If the `dry_run` parameter is true, `bind_cols` ignores `out` and just computes
    /// the number of entries that would be in the bound column, which can be used to
    /// allocate the exact amount of memory needed in the subsequent "real" bind operation.
    fn seq_bind_cols(
        even: &[MatrixEntry<F>],
        odd: &[MatrixEntry<F>],
        mut even_checkpoint: F,
        mut odd_checkpoint: F,
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
            if even[i].row == odd[j].row {
                if !dry_run {
                    let bound_entry = Self::bind_entries(
                        Some(&even[i]),
                        Some(&odd[j]),
                        even_checkpoint,
                        odd_checkpoint,
                        r,
                    );
                    out[k] = MaybeUninit::new(bound_entry);
                }
                even_checkpoint = even[i].next_val;
                odd_checkpoint = odd[j].next_val;
                i += 1;
                j += 1;
                k += 1;
            } else if even[i].row < odd[j].row {
                if !dry_run {
                    let bound_entry = Self::bind_entries(
                        Some(&even[i]),
                        None,
                        even_checkpoint,
                        odd_checkpoint,
                        r,
                    );
                    out[k] = MaybeUninit::new(bound_entry);
                }
                even_checkpoint = even[i].next_val;
                i += 1;
                k += 1;
            } else {
                if !dry_run {
                    let bound_entry =
                        Self::bind_entries(None, Some(&odd[j]), even_checkpoint, odd_checkpoint, r);
                    out[k] = MaybeUninit::new(bound_entry);
                }
                odd_checkpoint = odd[j].next_val;
                j += 1;
                k += 1;
            }
        }
        for remaining_even_entry in even[i..].iter() {
            if !dry_run {
                let bound_entry = Self::bind_entries(
                    Some(remaining_even_entry),
                    None,
                    even_checkpoint,
                    odd_checkpoint,
                    r,
                );
                out[k] = MaybeUninit::new(bound_entry);
            }
            k += 1;
        }
        for remaining_odd_entry in odd[j..].iter() {
            if !dry_run {
                let bound_entry = Self::bind_entries(
                    None,
                    Some(remaining_odd_entry),
                    even_checkpoint,
                    odd_checkpoint,
                    r,
                );
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
    /// By "adjacent", here we mean entries that are in the same row and adjacent
    /// columns (columns 2k and 2k+1).
    /// Either `even` or `odd` may be `None`, indicating that the corresponding matrix
    /// entry is not explicitly represented in the `SparseMatrixPolynomial` data structure.
    /// Instead, we can infer its values from the matrix entry that is `Some`, plus the
    /// given checkpoints
    fn bind_entries(
        even: Option<&MatrixEntry<F>>,
        odd: Option<&MatrixEntry<F>>,
        even_checkpoint: F,
        odd_checkpoint: F,
        r: F::Challenge,
    ) -> MatrixEntry<F> {
        match (even, odd) {
            (Some(even), Some(odd)) => {
                debug_assert!(even.col.is_even());
                debug_assert!(odd.col.is_odd());
                debug_assert_eq!(even.row, odd.row);
                MatrixEntry {
                    row: even.row,
                    col: even.col / 2,
                    ra_coeff: even.ra_coeff + r.mul_0_optimized(odd.ra_coeff - even.ra_coeff),
                    val_coeff: even.val_coeff + r.mul_0_optimized(odd.val_coeff - even.val_coeff),
                    prev_val: even.prev_val + r.mul_0_optimized(odd.prev_val - even.prev_val),
                    next_val: even.next_val + r.mul_0_optimized(odd.next_val - even.next_val),
                }
            }
            (Some(even), None) => {
                // For SparseMatrixPolynomial, the absence of a matrix entry implies
                // that its coeff has not been bound yet.
                // The absence of an odd-col entry in the same row as even
                // means that its implicit Val coeff is odd_checkpoint, and its implicit
                // ra coeff is 0.
                MatrixEntry {
                    row: even.row,
                    col: even.col / 2,
                    ra_coeff: (F::one() - r).mul_1_optimized(even.ra_coeff),
                    val_coeff: even.val_coeff + r.mul_0_optimized(odd_checkpoint - even.val_coeff),
                    prev_val: even.prev_val + r.mul_0_optimized(odd_checkpoint - even.prev_val),
                    next_val: even.next_val + r.mul_0_optimized(odd_checkpoint - even.next_val),
                }
            }
            (None, Some(odd)) => {
                // For SparseMatrixPolynomial, the absence of a matrix entry implies
                // that its coeff has not been bound yet.
                // The absence of an even-col entry in the same row as odd
                // means that its implicit Val coeff is even_checkpoint, and its implicit
                // ra coeff is 0.
                MatrixEntry {
                    row: odd.row,
                    col: odd.col / 2,
                    ra_coeff: r.mul_1_optimized(odd.ra_coeff),
                    val_coeff: even_checkpoint + r.mul_0_optimized(odd.val_coeff - even_checkpoint),
                    prev_val: even_checkpoint + r.mul_0_optimized(odd.prev_val - even_checkpoint),
                    next_val: even_checkpoint + r.mul_0_optimized(odd.next_val - even_checkpoint),
                }
            }
            (None, None) => panic!("Both entries are None"),
        }
    }

    pub fn compute_prover_message(
        &self,
        inc: &MultilinearPolynomial<F>,
        eq: &MultilinearPolynomial<F>,
        gamma: F,
        previous_claim: F,
    ) -> UniPoly<F> {
        let evals = self
            .entries
            .par_chunk_by(|x, y| x.col / 2 == y.col / 2)
            .map(|entries| {
                let odd_col_start_index = entries.partition_point(|entry| entry.col.is_even());
                let (even_col, odd_col) = entries.split_at(odd_col_start_index);
                let even_col_idx = 2 * (entries[0].col / 2);
                let odd_col_idx = even_col_idx + 1;
                Self::prover_message_contribution(
                    even_col,
                    odd_col,
                    self.val_init.get_bound_coeff(even_col_idx),
                    self.val_init.get_bound_coeff(odd_col_idx),
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

    /// For the given pair of adjacent columns, computes the pair's contribution to the prover's
    /// sumcheck message. This is a recursive, parallel algorithm.
    fn prover_message_contribution(
        even_col: &[MatrixEntry<F>],
        odd_col: &[MatrixEntry<F>],
        even_checkpoint: F,
        odd_checkpoint: F,
        inc: &MultilinearPolynomial<F>,
        eq: &MultilinearPolynomial<F>,
        gamma: F,
    ) -> [F; 2] {
        /// Threshold where we stop parallelizing and do a plain linear merge.
        const PAR_THRESHOLD: usize = 32_768;

        // small inputs: do the O(n) sequential algorithm
        if even_col.len() + odd_col.len() <= PAR_THRESHOLD {
            return Self::seq_prover_message_contribution(
                even_col,
                odd_col,
                even_checkpoint,
                odd_checkpoint,
                inc,
                eq,
                gamma,
            );
        }

        // Split the longer col at its midpoint; find where that pivot would land in the other col.
        let (even_pivot_idx, odd_pivot_idx) = if even_col.len() > odd_col.len() {
            let even_pivot_idx = even_col.len() / 2;
            let pivot = even_col[even_pivot_idx].row;
            let odd_pivot_idx = odd_col.partition_point(|x| x.row < pivot);
            (even_pivot_idx, odd_pivot_idx)
        } else {
            let odd_pivot_idx = odd_col.len() / 2;
            let pivot = odd_col[odd_pivot_idx].row;
            let even_pivot_idx = even_col.partition_point(|x| x.row < pivot);
            (even_pivot_idx, odd_pivot_idx)
        };

        // Now we know the global order: everything in even_col[..even_pivot_idx] and
        // odd_col[..odd_pivot_idx] comes before everything in even_col[even_pivot_idx..]
        // and odd_col[odd_pivot_idx..]. Compute each half's contribution in parallel.
        let (top_evals, bottom_evals) = rayon::join(
            || {
                Self::prover_message_contribution(
                    &even_col[..even_pivot_idx],
                    &odd_col[..odd_pivot_idx],
                    even_checkpoint,
                    odd_checkpoint,
                    inc,
                    eq,
                    gamma,
                )
            },
            || {
                let even_checkpoint = if even_col.is_empty() {
                    even_checkpoint
                } else if even_pivot_idx != 0 {
                    even_col[even_pivot_idx - 1].next_val
                } else {
                    even_col[even_pivot_idx].prev_val
                };
                let odd_checkpoint = if odd_col.is_empty() {
                    odd_checkpoint
                } else if odd_pivot_idx != 0 {
                    odd_col[odd_pivot_idx - 1].next_val
                } else {
                    odd_col[odd_pivot_idx].prev_val
                };
                Self::prover_message_contribution(
                    &even_col[even_pivot_idx..],
                    &odd_col[odd_pivot_idx..],
                    even_checkpoint,
                    odd_checkpoint,
                    inc,
                    eq,
                    gamma,
                )
            },
        );

        [
            top_evals[0] + bottom_evals[0],
            top_evals[1] + bottom_evals[1],
        ]
    }

    /// For the given pair of adjacent columns, computes the pair's contribution to the prover's
    /// sumcheck message. This is the sequential counterpart of `prover_message_contribution`.
    fn seq_prover_message_contribution(
        even: &[MatrixEntry<F>],
        odd: &[MatrixEntry<F>],
        mut even_checkpoint: F,
        mut odd_checkpoint: F,
        inc: &MultilinearPolynomial<F>,
        eq: &MultilinearPolynomial<F>,
        gamma: F,
    ) -> [F; 2] {
        let mut i = 0;
        let mut j = 0;
        let mut evals_accumulator = [F::zero(); 2];

        while i < even.len() && j < odd.len() {
            if even[i].row == odd[j].row {
                let evals = Self::compute_evals(
                    Some(&even[i]),
                    Some(&odd[j]),
                    even_checkpoint,
                    odd_checkpoint,
                    inc.get_bound_coeff(even[i].row),
                    eq.get_bound_coeff(even[i].row),
                    gamma,
                );
                evals_accumulator[0] += evals[0];
                evals_accumulator[1] += evals[1];
                even_checkpoint = even[i].next_val;
                odd_checkpoint = odd[j].next_val;
                i += 1;
                j += 1;
            } else if even[i].row < odd[j].row {
                let evals = Self::compute_evals(
                    Some(&even[i]),
                    None,
                    even_checkpoint,
                    odd_checkpoint,
                    inc.get_bound_coeff(even[i].row),
                    eq.get_bound_coeff(even[i].row),
                    gamma,
                );
                even_checkpoint = even[i].next_val;
                evals_accumulator[0] += evals[0];
                evals_accumulator[1] += evals[1];
                i += 1;
            } else {
                let evals = Self::compute_evals(
                    None,
                    Some(&odd[j]),
                    even_checkpoint,
                    odd_checkpoint,
                    inc.get_bound_coeff(odd[j].row),
                    eq.get_bound_coeff(odd[j].row),
                    gamma,
                );
                odd_checkpoint = odd[j].next_val;
                evals_accumulator[0] += evals[0];
                evals_accumulator[1] += evals[1];
                j += 1;
            }
        }
        for remaining_even_entry in even[i..].iter() {
            let evals = Self::compute_evals(
                Some(remaining_even_entry),
                None,
                even_checkpoint,
                odd_checkpoint,
                inc.get_bound_coeff(remaining_even_entry.row),
                eq.get_bound_coeff(remaining_even_entry.row),
                gamma,
            );
            evals_accumulator[0] += evals[0];
            evals_accumulator[1] += evals[1];
        }
        for remaining_odd_entry in odd[j..].iter() {
            let evals = Self::compute_evals(
                None,
                Some(remaining_odd_entry),
                even_checkpoint,
                odd_checkpoint,
                inc.get_bound_coeff(remaining_odd_entry.row),
                eq.get_bound_coeff(remaining_odd_entry.row),
                gamma,
            );
            evals_accumulator[0] += evals[0];
            evals_accumulator[1] += evals[1];
        }

        evals_accumulator
    }

    /// For the given pair of adjacent entries, computes the pair's contribution to the prover's
    /// sumcheck message. By "adjacent", here we mean entries that are in the same row and
    /// adjacent columns (columns 2k and 2k+1).
    /// Either `even` or `odd` may be `None`, indicating that the corresponding matrix
    /// entry is not explicitly represented in the `SparseMatrixPolynomial` data structure.
    /// Instead, we can infer its values from the matrix entry that is `Some`.
    fn compute_evals(
        even: Option<&MatrixEntry<F>>,
        odd: Option<&MatrixEntry<F>>,
        even_checkpoint: F,
        odd_checkpoint: F,
        inc_eval: F,
        eq_eval: F,
        gamma: F,
    ) -> [F; 2] {
        match (even, odd) {
            (Some(even), Some(odd)) => {
                debug_assert!(even.col.is_even());
                debug_assert!(odd.col.is_odd());
                debug_assert_eq!(even.row, odd.row);
                let ra_evals = [even.ra_coeff, odd.ra_coeff + odd.ra_coeff - even.ra_coeff];
                let val_evals = [
                    even.val_coeff,
                    odd.val_coeff + odd.val_coeff - even.val_coeff,
                ];
                [
                    eq_eval * ra_evals[0] * (val_evals[0] + gamma * (inc_eval + val_evals[0])),
                    eq_eval * ra_evals[1] * (val_evals[1] + gamma * (inc_eval + val_evals[1])),
                ]
            }
            (Some(even), None) => {
                // For SparseMatrixPolynomial, the absence of a matrix entry implies
                // that its coeff has not been bound yet.
                // The absence of an odd-row entry in the same column as even
                // means that its implicit Val coeff is odd_checkpoint, and its implicit
                // ra coeff is 0.
                let ra_evals = [even.ra_coeff, -even.ra_coeff];
                let val_evals = [
                    even.val_coeff,
                    odd_checkpoint + odd_checkpoint - even.val_coeff,
                ];
                [
                    eq_eval * ra_evals[0] * (val_evals[0] + gamma * (inc_eval + val_evals[0])),
                    eq_eval * ra_evals[1] * (val_evals[1] + gamma * (inc_eval + val_evals[1])),
                ]
            }
            (None, Some(odd)) => {
                // For SparseMatrixPolynomial, the absence of a matrix entry implies
                // that its coeff has not been bound yet.
                // The absence of an even-row entry in the same column as odd
                // means that its implicit Val coeff is even_checkpoint, and its implicit
                // ra coeff is 0.
                let ra_evals = [F::zero(), odd.ra_coeff + odd.ra_coeff];
                let val_evals = [
                    even_checkpoint,
                    odd.val_coeff + odd.val_coeff - even_checkpoint,
                ];
                [
                    F::zero(), // ra_evals[0] is zero
                    eq_eval * ra_evals[1] * (val_evals[1] + gamma * (inc_eval + val_evals[1])),
                ]
            }
            (None, None) => panic!("Both entries are None"),
        }
    }

    /// Materializes the ra and Val polynomials represented by this `SparseMatrixPolynomial`.
    /// Some number of cycle and address variables have already been bound, so at this point
    /// there are `K_prime` columns and `T_prime` rows left in the matrix.
    #[tracing::instrument(skip_all, name = "SparseMatrixPolynomial::materialize")]
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

        // Update some of the ra and Val coefficients based on
        // matrix entries.
        self.entries
            .par_chunk_by(|a, b| a.col == b.col)
            .for_each(|column| {
                let k = column[0].col;
                let mut current_val_coeff = self.val_init.get_bound_coeff(k);
                let mut column_iter = column.iter().peekable();
                for j in 0..T_prime {
                    let idx = k * T_prime + j;
                    if let Some(entry) = column_iter.next_if(|&entry| entry.row == j) {
                        *ra[idx].lock().unwrap() = entry.ra_coeff;
                        *val[idx].lock().unwrap() = entry.val_coeff;
                        current_val_coeff = entry.next_val;
                        continue;
                    }
                    // *ra[idx].lock().unwrap() = F::zero(); // Already zero
                    *val[idx].lock().unwrap() = current_val_coeff;
                    continue;
                }
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
