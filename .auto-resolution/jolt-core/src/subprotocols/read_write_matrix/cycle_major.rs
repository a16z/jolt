//! Cycle-major (row-major) sparse matrix representation.
//!
//! Entries are sorted by `(row, col)` - optimal for binding cycle variables first.

use std::mem::MaybeUninit;

use allocative::Allocative;
use ark_std::Zero;
use num::Integer;
use rayon::prelude::*;

use crate::field::JoltField;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;

pub trait CycleMajorMatrixEntry<F: JoltField>: Send + Sync + Sized {
    /// The row index. Before binding, row \in [0, T)
    fn row(&self) -> usize;

    /// The column index. Before binding, column \in [0, K)
    fn column(&self) -> usize;

    /// Binds adjacent entries of the matrix together using the random challenge `r`.
    /// By "adjacent", here we mean entries that are in the same column and adjacent
    /// rows (rows 2j and 2j+1).
    /// Either `even` or `odd` may be `None`, indicating that the corresponding matrix
    /// entry is not explicitly represented in the `ReadWriteMatrixCycleMajor` data structure.
    /// Instead, we can infer its values from the matrix entry that is `Some`.
    fn bind_entries(even: Option<&Self>, odd: Option<&Self>, r: F::Challenge) -> Self;

    /// For the given pair of adjacent entries, computes the pair's contribution to the prover's
    /// sumcheck message, returning `Unreduced<8>` to avoid Montgomery reduction.
    ///
    /// By "adjacent", here we mean entries that are in the same column and
    /// adjacent rows (rows 2j and 2j+1).
    /// Either `even` or `odd` may be `None`, indicating that the corresponding matrix
    /// entry is not explicitly represented in the `ReadWriteMatrixCycleMajor` data structure.
    /// Instead, we can infer its values from the matrix entry that is `Some`.
    ///
    /// The final `ra * (...)` uses `mul_unreduced` instead of regular multiplication.
    /// This is used in `seq_prover_message_contribution` for better performance when
    /// accumulating many entries.
    fn compute_evals(
        even: Option<&Self>,
        odd: Option<&Self>,
        inc_evals: [F; 2],
        gamma: F,
    ) -> [F::Unreduced<8>; 2];
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
///
/// # Type Parameters
///
/// - `F`: The field type for coefficients.
#[derive(Allocative, Debug, Default, Clone)]
pub struct ReadWriteMatrixCycleMajor<F: JoltField, E: CycleMajorMatrixEntry<F>> {
    pub entries: Vec<E>,
    pub(crate) val_init: MultilinearPolynomial<F>,
}

impl<F: JoltField, E: CycleMajorMatrixEntry<F>> ReadWriteMatrixCycleMajor<F, E> {
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
        even_row: &[E],
        odd_row: &[E],
        r: F::Challenge,
        out: &mut [MaybeUninit<E>],
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
            let pivot = even_row[even_pivot_idx].column();
            let odd_pivot_idx = odd_row.partition_point(|x| x.column() < pivot);
            (even_pivot_idx, odd_pivot_idx)
        } else {
            let odd_pivot_idx = odd_row.len() / 2;
            let pivot = odd_row[odd_pivot_idx].column();
            let even_pivot_idx = even_row.partition_point(|x| x.column() < pivot);
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
        even: &[E],
        odd: &[E],
        r: F::Challenge,
        out: &mut [MaybeUninit<E>],
        dry_run: bool,
    ) -> usize {
        // Even index
        let mut i = 0;
        // Odd index
        let mut j = 0;
        // Out index
        let mut k = 0;

        while i < even.len() && j < odd.len() {
            if even[i].column() == odd[j].column() {
                if !dry_run {
                    let bound_entry = E::bind_entries(Some(&even[i]), Some(&odd[j]), r);
                    out[k] = MaybeUninit::new(bound_entry);
                }
                i += 1;
                j += 1;
                k += 1;
            } else if even[i].column() < odd[j].column() {
                if !dry_run {
                    let bound_entry = E::bind_entries(Some(&even[i]), None, r);
                    out[k] = MaybeUninit::new(bound_entry);
                }
                i += 1;
                k += 1;
            } else {
                if !dry_run {
                    let bound_entry = E::bind_entries(None, Some(&odd[j]), r);
                    out[k] = MaybeUninit::new(bound_entry);
                }
                j += 1;
                k += 1;
            }
        }

        if dry_run {
            k += even[i..].len();
            k += odd[j..].len();
            return k;
        }

        for remaining_even_entry in even[i..].iter() {
            let bound_entry = E::bind_entries(Some(remaining_even_entry), None, r);
            out[k] = MaybeUninit::new(bound_entry);
            k += 1;
        }
        for remaining_odd_entry in odd[j..].iter() {
            let bound_entry = E::bind_entries(None, Some(remaining_odd_entry), r);
            out[k] = MaybeUninit::new(bound_entry);
            k += 1;
        }
        assert_eq!(out.len(), k);
        k
    }

    /// Binds a cycle variable of the ra and Val polynomials represented by
    /// this `ReadWriteMatrixCycleMajor` to the random challenge `r`.
    #[tracing::instrument(skip_all, name = "ReadWriteMatrixCycleMajor::bind")]
    pub fn bind(&mut self, r: F::Challenge) {
        let row_lengths: Vec<_> = self
            .entries
            .par_chunk_by(|x, y| x.row() / 2 == y.row() / 2)
            .map(|entries| {
                let odd_row_start_index = entries.partition_point(|entry| entry.row().is_even());
                let (even_row, odd_row) = entries.split_at(odd_row_start_index);
                // Dry run to compute output length
                let bound_len = Self::bind_rows(even_row, odd_row, r, &mut [], true);
                (entries.len(), bound_len)
            })
            .collect();

        let bound_length = row_lengths.iter().map(|(_, bound_len)| bound_len).sum();
        let mut bound_entries: Vec<E> = Vec::with_capacity(bound_length);
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
                let odd_row_start_index =
                    input_slice.partition_point(|entry| entry.row().is_even());
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
        even_row: &[E],
        odd_row: &[E],
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
            let pivot = even_row[even_pivot_idx].column();
            let odd_pivot_idx = odd_row.partition_point(|x| x.column() < pivot);
            (even_pivot_idx, odd_pivot_idx)
        } else {
            let odd_pivot_idx = odd_row.len() / 2;
            let pivot = odd_row[odd_pivot_idx].column();
            let even_pivot_idx = even_row.partition_point(|x| x.column() < pivot);
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

        std::array::from_fn(|i| left_evals[i] + right_evals[i])
    }

    /// For the given pair of adjacent rows, computes the pair's contribution to the prover's
    /// sumcheck message. This is the sequential counterpart of `prover_message_contribution`.
    ///
    /// Uses `Unreduced<9>` accumulator to delay modular reductions for better performance.
    /// Each `compute_evals_unreduced` returns `Unreduced<8>` (no reduction on the final multiply),
    /// and we accumulate into `Unreduced<9>` for headroom. Only one Montgomery reduction at the end.
    fn seq_prover_message_contribution(
        even: &[E],
        odd: &[E],
        inc_evals: [F; 2],
        gamma: F,
    ) -> [F; 2] {
        let mut i = 0;
        let mut j = 0;
        let mut evals_accumulator = [F::Unreduced::<9>::zero(); 2];

        while i < even.len() && j < odd.len() {
            if even[i].column() == odd[j].column() {
                let evals = E::compute_evals(Some(&even[i]), Some(&odd[j]), inc_evals, gamma);
                evals_accumulator[0] += evals[0];
                evals_accumulator[1] += evals[1];
                i += 1;
                j += 1;
            } else if even[i].column() < odd[j].column() {
                let evals = E::compute_evals(Some(&even[i]), None, inc_evals, gamma);
                evals_accumulator[0] += evals[0];
                evals_accumulator[1] += evals[1];
                i += 1;
            } else {
                let evals = E::compute_evals(None, Some(&odd[j]), inc_evals, gamma);
                evals_accumulator[0] += evals[0];
                evals_accumulator[1] += evals[1];
                j += 1;
            }
        }
        for remaining_even_entry in even[i..].iter() {
            let evals = E::compute_evals(Some(remaining_even_entry), None, inc_evals, gamma);
            evals_accumulator[0] += evals[0];
            evals_accumulator[1] += evals[1];
        }
        for remaining_odd_entry in odd[j..].iter() {
            let evals = E::compute_evals(None, Some(remaining_odd_entry), inc_evals, gamma);
            evals_accumulator[0] += evals[0];
            evals_accumulator[1] += evals[1];
        }

        std::array::from_fn(|i| F::from_montgomery_reduce(evals_accumulator[i]))
    }
}
