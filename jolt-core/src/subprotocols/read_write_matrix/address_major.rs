//! Address-major (column-major) sparse matrix representation.
//!
//! Entries are sorted by `(col, row)` - optimal for binding address variables first.

use std::cmp::Ordering;
use std::mem::MaybeUninit;

use allocative::Allocative;
use ark_std::Zero;
use num::Integer;
use rayon::prelude::*;

use crate::field::JoltField;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::subprotocols::read_write_matrix::cycle_major::CycleMajorMatrixEntry;

use super::cycle_major::ReadWriteMatrixCycleMajor;

pub trait AddressMajorMatrixEntry<F: JoltField>: Send + Sync + Sized {
    /// The row index. Before binding, row \in [0, T)
    fn row(&self) -> usize;

    /// The column index. Before binding, column \in [0, K)
    fn column(&self) -> usize;

    /// In round i, each entry represents a coefficient
    ///   Val(k, j', r)
    /// which is some combination of Val(k, j', 00...0), ...
    /// Val(k, j', 11...1).
    /// `prev_val` contains the unbound coefficient before
    /// Val(k, j', 00...0) –– abusing notation, `prev_val` is
    /// Val(k, j'-1, 11...1)
    fn prev_val(&self) -> F;

    /// In round i, each entry represents a coefficient
    ///   Val(k, j', r)
    /// which is some combination of Val(k, j', 00...0), ...
    /// Val(k, j', 11...1).
    /// `next_val` contains the unbound coefficient after
    /// Val(k, j', 00...0) –– abusing notation, `next_val` is
    /// Val(k, j'+1, 00...0)
    fn next_val(&self) -> F;

    /// Binds adjacent entries of the matrix together using the random challenge `r`.
    /// By "adjacent", here we mean entries that are in the same row and adjacent
    /// columns (columns 2k and 2k+1).
    /// Either `even` or `odd` may be `None`, indicating that the corresponding matrix
    /// entry is not explicitly represented in the `SparseMatrixPolynomial` data structure.
    /// Instead, we can infer its values from the matrix entry that is `Some`, plus the
    /// given checkpoints
    fn bind_entries(
        even: Option<&Self>,
        odd: Option<&Self>,
        even_checkpoint: F,
        odd_checkpoint: F,
        r: F::Challenge,
    ) -> Self;

    /// For the given pair of adjacent entries, computes the pair's contribution to the prover's
    /// sumcheck message. By "adjacent", here we mean entries that are in the same row and
    /// adjacent columns (columns 2k and 2k+1).
    /// Either `even` or `odd` may be `None`, indicating that the corresponding matrix
    /// entry is not explicitly represented in the `SparseMatrixPolynomial` data structure.
    /// Instead, we can infer its values from the matrix entry that is `Some`.
    fn compute_evals(
        even: Option<&Self>,
        odd: Option<&Self>,
        even_checkpoint: F,
        odd_checkpoint: F,
        inc_eval: F,
        eq_eval: F,
        gamma: F,
    ) -> [F::Unreduced<8>; 2];
}

/// Represents the ra(k, j) and Val(k, j) polynomials for the RAM
/// read/write-checking sumcheck in address-major (column-major) order.
///
/// # Type Parameters
///
/// - `F`: The field type for coefficients.
#[derive(Allocative, Debug, Default, Clone)]
pub struct ReadWriteMatrixAddressMajor<F: JoltField, E: AddressMajorMatrixEntry<F>> {
    pub entries: Vec<E>,
    pub(crate) val_init: MultilinearPolynomial<F>,
}

impl<F: JoltField, E1: CycleMajorMatrixEntry<F>, E2: AddressMajorMatrixEntry<F> + From<E1>>
    From<ReadWriteMatrixCycleMajor<F, E1>> for ReadWriteMatrixAddressMajor<F, E2>
{
    fn from(mut cycle_major: ReadWriteMatrixCycleMajor<F, E1>) -> Self {
        let mut entries = std::mem::take(&mut cycle_major.entries);
        let val_init = std::mem::take(&mut cycle_major.val_init);
        entries.par_sort_by(|a, b| match a.column().cmp(&b.column()) {
            Ordering::Less => Ordering::Less,
            Ordering::Greater => Ordering::Greater,
            Ordering::Equal => a.row().cmp(&b.row()),
        });
        let entries = entries.into_par_iter().map(|entry| entry.into()).collect();
        ReadWriteMatrixAddressMajor { entries, val_init }
    }
}

impl<F: JoltField, E: AddressMajorMatrixEntry<F>> ReadWriteMatrixAddressMajor<F, E> {
    /// Binds an address variable of the ra and Val polynomials represented by
    /// this `SparseMatrixPolynomial` to the random challenge `r`.
    #[tracing::instrument(skip_all, name = "SparseMatrixPolynomial::bind")]
    pub fn bind(&mut self, r: F::Challenge) {
        let col_lengths: Vec<_> = self
            .entries
            .par_chunk_by(|x, y| x.column() / 2 == y.column() / 2)
            .map(|entries| {
                let odd_col_start_index = entries.partition_point(|entry| entry.column().is_even());
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
        let mut bound_entries: Vec<E> = Vec::with_capacity(bound_length);
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
                let odd_col_start_index =
                    input_slice.partition_point(|entry| entry.column().is_even());
                let (even_col, odd_col) = input_slice.split_at(odd_col_start_index);
                let even_col_idx = 2 * (input_slice[0].column() / 2);
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
        even_col: &[E],
        odd_col: &[E],
        even_checkpoint: F,
        odd_checkpoint: F,
        r: F::Challenge,
        out: &mut [MaybeUninit<E>],
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
            let pivot = even_col[even_pivot_idx].row();
            let odd_pivot_idx = odd_col.partition_point(|x| x.row() < pivot);
            (even_pivot_idx, odd_pivot_idx)
        } else {
            let odd_pivot_idx = odd_col.len() / 2;
            let pivot = odd_col[odd_pivot_idx].row();
            let even_pivot_idx = even_col.partition_point(|x| x.row() < pivot);
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
                        even_col[even_pivot_idx - 1].next_val()
                    } else {
                        even_col[even_pivot_idx].prev_val()
                    };
                    let odd_checkpoint = if odd_col.is_empty() {
                        odd_checkpoint
                    } else if odd_pivot_idx != 0 {
                        odd_col[odd_pivot_idx - 1].next_val()
                    } else {
                        odd_col[odd_pivot_idx].prev_val()
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
        even: &[E],
        odd: &[E],
        mut even_checkpoint: F,
        mut odd_checkpoint: F,
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
            if even[i].row() == odd[j].row() {
                if !dry_run {
                    let bound_entry = E::bind_entries(
                        Some(&even[i]),
                        Some(&odd[j]),
                        even_checkpoint,
                        odd_checkpoint,
                        r,
                    );
                    out[k] = MaybeUninit::new(bound_entry);
                }
                even_checkpoint = even[i].next_val();
                odd_checkpoint = odd[j].next_val();
                i += 1;
                j += 1;
                k += 1;
            } else if even[i].row() < odd[j].row() {
                if !dry_run {
                    let bound_entry =
                        E::bind_entries(Some(&even[i]), None, even_checkpoint, odd_checkpoint, r);
                    out[k] = MaybeUninit::new(bound_entry);
                }
                even_checkpoint = even[i].next_val();
                i += 1;
                k += 1;
            } else {
                if !dry_run {
                    let bound_entry =
                        E::bind_entries(None, Some(&odd[j]), even_checkpoint, odd_checkpoint, r);
                    out[k] = MaybeUninit::new(bound_entry);
                }
                odd_checkpoint = odd[j].next_val();
                j += 1;
                k += 1;
            }
        }
        for remaining_even_entry in even[i..].iter() {
            if !dry_run {
                let bound_entry = E::bind_entries(
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
                let bound_entry = E::bind_entries(
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

    /// For the given pair of adjacent columns, computes the pair's contribution to the prover's
    /// sumcheck message. This is a recursive, parallel algorithm.
    pub fn prover_message_contribution(
        even_col: &[E],
        odd_col: &[E],
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
            let pivot = even_col[even_pivot_idx].row();
            let odd_pivot_idx = odd_col.partition_point(|x| x.row() < pivot);
            (even_pivot_idx, odd_pivot_idx)
        } else {
            let odd_pivot_idx = odd_col.len() / 2;
            let pivot = odd_col[odd_pivot_idx].row();
            let even_pivot_idx = even_col.partition_point(|x| x.row() < pivot);
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
                    even_col[even_pivot_idx - 1].next_val()
                } else {
                    even_col[even_pivot_idx].prev_val()
                };
                let odd_checkpoint = if odd_col.is_empty() {
                    odd_checkpoint
                } else if odd_pivot_idx != 0 {
                    odd_col[odd_pivot_idx - 1].next_val()
                } else {
                    odd_col[odd_pivot_idx].prev_val()
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
        even: &[E],
        odd: &[E],
        mut even_checkpoint: F,
        mut odd_checkpoint: F,
        inc: &MultilinearPolynomial<F>,
        eq: &MultilinearPolynomial<F>,
        gamma: F,
    ) -> [F; 2] {
        let mut i = 0;
        let mut j = 0;
        let mut evals_accumulator = [F::Unreduced::<9>::zero(); 2];

        while i < even.len() && j < odd.len() {
            if even[i].row() == odd[j].row() {
                let evals = E::compute_evals(
                    Some(&even[i]),
                    Some(&odd[j]),
                    even_checkpoint,
                    odd_checkpoint,
                    inc.get_bound_coeff(even[i].row()),
                    eq.get_bound_coeff(even[i].row()),
                    gamma,
                );
                evals_accumulator[0] += evals[0];
                evals_accumulator[1] += evals[1];
                even_checkpoint = even[i].next_val();
                odd_checkpoint = odd[j].next_val();
                i += 1;
                j += 1;
            } else if even[i].row() < odd[j].row() {
                let evals = E::compute_evals(
                    Some(&even[i]),
                    None,
                    even_checkpoint,
                    odd_checkpoint,
                    inc.get_bound_coeff(even[i].row()),
                    eq.get_bound_coeff(even[i].row()),
                    gamma,
                );
                even_checkpoint = even[i].next_val();
                evals_accumulator[0] += evals[0];
                evals_accumulator[1] += evals[1];
                i += 1;
            } else {
                let evals = E::compute_evals(
                    None,
                    Some(&odd[j]),
                    even_checkpoint,
                    odd_checkpoint,
                    inc.get_bound_coeff(odd[j].row()),
                    eq.get_bound_coeff(odd[j].row()),
                    gamma,
                );
                odd_checkpoint = odd[j].next_val();
                evals_accumulator[0] += evals[0];
                evals_accumulator[1] += evals[1];
                j += 1;
            }
        }
        for remaining_even_entry in even[i..].iter() {
            let evals = E::compute_evals(
                Some(remaining_even_entry),
                None,
                even_checkpoint,
                odd_checkpoint,
                inc.get_bound_coeff(remaining_even_entry.row()),
                eq.get_bound_coeff(remaining_even_entry.row()),
                gamma,
            );
            evals_accumulator[0] += evals[0];
            evals_accumulator[1] += evals[1];
        }
        for remaining_odd_entry in odd[j..].iter() {
            let evals = E::compute_evals(
                None,
                Some(remaining_odd_entry),
                even_checkpoint,
                odd_checkpoint,
                inc.get_bound_coeff(remaining_odd_entry.row()),
                eq.get_bound_coeff(remaining_odd_entry.row()),
                gamma,
            );
            evals_accumulator[0] += evals[0];
            evals_accumulator[1] += evals[1];
        }

        [
            F::from_montgomery_reduce(evals_accumulator[0]),
            F::from_montgomery_reduce(evals_accumulator[1]),
        ]
    }
}
