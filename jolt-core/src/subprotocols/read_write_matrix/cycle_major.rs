//! Cycle-major (row-major) sparse matrix representation.
//!
//! Entries are sorted by `(row, col)` - optimal for binding cycle variables first.

use std::mem::MaybeUninit;

use allocative::Allocative;
use ark_std::Zero;
use num::Integer;
use rayon::prelude::*;

use crate::field::JoltField;
use crate::field::OptimizedMul;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::zkvm::ram::remap_address;
use common::jolt_device::MemoryLayout;
use tracer::instruction::{Cycle, RAMAccess};

/// Represents a non-zero entry in the ra(k, j) and Val(k, j) polynomials.
/// Conceptually, both ra and Val can be seen as K x T matrices.
///
/// # Memory Optimization: `prev_val`/`next_val` as `u64`
///
/// These fields store raw memory values (not bound coefficients) because:
/// - Phase 1 (cycle binding) always starts from initial memory state
/// - We never switch to cycle-major after binding address variables
/// - They're only converted to `F` when needed in arithmetic
///
/// This saves 48 bytes per entry (~35% reduction).
///
/// # Type Parameters
///
/// - `F`: The field type for coefficients.
#[derive(Allocative, Debug, PartialEq, Clone, Copy)]
pub struct ReadWriteEntry<F: JoltField> {
    /// The row index. Before binding, row \in [0, T)
    pub row: usize,
    /// The column index. Before binding, col \in [0, K)
    pub col: usize,
    /// In round i, each ReadWriteEntry represents a coefficient
    ///   Val(k, j', r)
    /// which is some combination of Val(k, j', 00...0), ...
    /// Val(k, j', 11...1).
    /// `prev_val` contains the unbound coefficient before
    /// Val(k, j', 00...0) –– abusing notation, `prev_val` is
    /// Val(k, j'-1, 11...1)
    pub(crate) prev_val: u64,
    /// In round i, each ReadWriteEntry represents a coefficient
    ///   Val(k, j', r)
    /// which is some combination of Val(k, j', 00...0), ...
    /// Val(k, j', 11...1).
    /// `next_val` contains the unbound coefficient after
    /// Val(k, j', 00...0) –– abusing notation, `next_val` is
    /// Val(k, j'+1, 00...0)
    pub(crate) next_val: u64,
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
///
/// # Type Parameters
///
/// - `F`: The field type for coefficients.
#[derive(Allocative, Debug, Default, Clone)]
pub struct ReadWriteMatrixCycleMajor<F: JoltField> {
    pub entries: Vec<ReadWriteEntry<F>>,
    pub(crate) val_init: MultilinearPolynomial<F>,
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
                        let pre_value = write.pre_value;
                        let post_value = write.post_value;
                        Some(ReadWriteEntry {
                            row: j,
                            col: remap_address(write.address, memory_layout).unwrap() as usize,
                            ra_coeff: F::one(),
                            val_coeff: F::from_u64(pre_value),
                            prev_val: pre_value,
                            next_val: post_value,
                        })
                    }
                    RAMAccess::Read(read) => {
                        let read_value = read.value;
                        Some(ReadWriteEntry {
                            row: j,
                            col: remap_address(read.address, memory_layout).unwrap() as usize,
                            ra_coeff: F::one(),
                            val_coeff: F::from_u64(read_value),
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
}

impl<F: JoltField> ReadWriteMatrixCycleMajor<F> {
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
        even_row: &[ReadWriteEntry<F>],
        odd_row: &[ReadWriteEntry<F>],
        r: F::Challenge,
        out: &mut [MaybeUninit<ReadWriteEntry<F>>],
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
        even: &[ReadWriteEntry<F>],
        odd: &[ReadWriteEntry<F>],
        r: F::Challenge,
        out: &mut [MaybeUninit<ReadWriteEntry<F>>],
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
        even: Option<&ReadWriteEntry<F>>,
        odd: Option<&ReadWriteEntry<F>>,
        r: F::Challenge,
    ) -> ReadWriteEntry<F> {
        match (even, odd) {
            (Some(even), Some(odd)) => {
                debug_assert!(even.row.is_even());
                debug_assert!(odd.row.is_odd());
                debug_assert_eq!(even.col, odd.col);
                ReadWriteEntry {
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
                // means that its implicit Val coeff is even.next_val, and its implicit
                // ra coeff is 0.
                let odd_val_coeff = F::from_u64(even.next_val);
                ReadWriteEntry {
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
                // means that its implicit Val coeff is odd.prev_val, and its implicit
                // ra coeff is 0.
                let even_val_coeff = F::from_u64(odd.prev_val);
                ReadWriteEntry {
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
        let mut bound_entries: Vec<ReadWriteEntry<F>> = Vec::with_capacity(bound_length);
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
        even_row: &[ReadWriteEntry<F>],
        odd_row: &[ReadWriteEntry<F>],
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
    ///
    /// Uses `Unreduced<9>` accumulator to delay modular reductions for better performance.
    /// Each `compute_evals_unreduced` returns `Unreduced<8>` (no reduction on the final multiply),
    /// and we accumulate into `Unreduced<9>` for headroom. Only one Montgomery reduction at the end.
    fn seq_prover_message_contribution(
        even: &[ReadWriteEntry<F>],
        odd: &[ReadWriteEntry<F>],
        inc_evals: [F; 2],
        gamma: F,
    ) -> [F; 2] {
        let mut i = 0;
        let mut j = 0;
        let mut evals_accumulator = [F::Unreduced::<9>::zero(); 2];

        while i < even.len() && j < odd.len() {
            if even[i].col == odd[j].col {
                let evals =
                    Self::compute_evals_unreduced(Some(&even[i]), Some(&odd[j]), inc_evals, gamma);
                evals_accumulator[0] += evals[0];
                evals_accumulator[1] += evals[1];
                i += 1;
                j += 1;
            } else if even[i].col < odd[j].col {
                let evals = Self::compute_evals_unreduced(Some(&even[i]), None, inc_evals, gamma);
                evals_accumulator[0] += evals[0];
                evals_accumulator[1] += evals[1];
                i += 1;
            } else {
                let evals = Self::compute_evals_unreduced(None, Some(&odd[j]), inc_evals, gamma);
                evals_accumulator[0] += evals[0];
                evals_accumulator[1] += evals[1];
                j += 1;
            }
        }
        for remaining_even_entry in even[i..].iter() {
            let evals =
                Self::compute_evals_unreduced(Some(remaining_even_entry), None, inc_evals, gamma);
            evals_accumulator[0] += evals[0];
            evals_accumulator[1] += evals[1];
        }
        for remaining_odd_entry in odd[j..].iter() {
            let evals =
                Self::compute_evals_unreduced(None, Some(remaining_odd_entry), inc_evals, gamma);
            evals_accumulator[0] += evals[0];
            evals_accumulator[1] += evals[1];
        }

        [
            F::from_montgomery_reduce(evals_accumulator[0]),
            F::from_montgomery_reduce(evals_accumulator[1]),
        ]
    }

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
    fn compute_evals_unreduced(
        even: Option<&ReadWriteEntry<F>>,
        odd: Option<&ReadWriteEntry<F>>,
        inc_evals: [F; 2],
        gamma: F,
    ) -> [F::Unreduced<8>; 2] {
        match (even, odd) {
            (Some(even), Some(odd)) => {
                debug_assert!(even.row.is_even());
                debug_assert!(odd.row.is_odd());
                debug_assert_eq!(even.col, odd.col);
                let ra_evals = [even.ra_coeff, odd.ra_coeff - even.ra_coeff];
                let val_evals = [even.val_coeff, odd.val_coeff - even.val_coeff];
                [
                    ra_evals[0].mul_unreduced(val_evals[0] + gamma * (inc_evals[0] + val_evals[0])),
                    ra_evals[1].mul_unreduced(val_evals[1] + gamma * (inc_evals[1] + val_evals[1])),
                ]
            }
            (Some(even), None) => {
                let odd_val_coeff = F::from_u64(even.next_val);
                let ra_evals = [even.ra_coeff, -even.ra_coeff];
                let val_evals = [even.val_coeff, odd_val_coeff - even.val_coeff];
                [
                    ra_evals[0].mul_unreduced(val_evals[0] + gamma * (inc_evals[0] + val_evals[0])),
                    ra_evals[1].mul_unreduced(val_evals[1] + gamma * (inc_evals[1] + val_evals[1])),
                ]
            }
            (None, Some(odd)) => {
                let even_val_coeff = F::from_u64(odd.prev_val);
                let ra_evals = [F::zero(), odd.ra_coeff];
                let val_evals = [even_val_coeff, odd.val_coeff - even_val_coeff];
                [
                    F::Unreduced::<8>::zero(), // ra_evals[0] is zero
                    ra_evals[1].mul_unreduced(val_evals[1] + gamma * (inc_evals[1] + val_evals[1])),
                ]
            }
            (None, None) => panic!("Both entries are None"),
        }
    }

    /// Materializes the ra and Val polynomials represented by this `ReadWriteMatrixCycleMajor`.
    ///
    /// After partial binding of cycle and address variables, there are `K_prime` columns
    /// (remaining address positions) and `T_prime` rows (remaining cycle positions) in the matrix.
    /// This expands the sparse representation to dense polynomials of size `K_prime * T_prime`.
    ///
    /// The output layout is address-major: index(addr k, cycle t) = k * T_prime + t.
    /// This matches the layout expected by `phase3_compute_message`.
    ///
    /// When `T_prime == 1` (all cycle variables bound), this is equivalent to the simple case
    /// where each entry has `row == 0` and entries have distinct `col` values.
    #[tracing::instrument(skip_all, name = "ReadWriteMatrixCycleMajor::materialize")]
    pub fn materialize(
        self,
        K_prime: usize,
        T_prime: usize,
    ) -> (MultilinearPolynomial<F>, MultilinearPolynomial<F>) {
        let len = K_prime * T_prime;

        // Initialize ra to zero
        let mut ra: Vec<F> = unsafe_allocate_zero_vec(len);

        // Initialize val: expand val_init from size K to size K * T_prime
        // Each address k gets val_init[k] replicated across all T_prime cycle positions
        // Layout: address-major, index(k, t) = k * T_prime + t
        let mut val: Vec<F> = unsafe_allocate_zero_vec(len);
        // Extract the coefficient slice from self.val_init (which is a LargeScalars polynomial)
        let val_init_coeffs = match &self.val_init {
            MultilinearPolynomial::LargeScalars(poly) => &poly.Z,
            _ => panic!("val_init must be LargeScalars"),
        };
        val.par_chunks_mut(T_prime)
            .zip(val_init_coeffs.par_iter())
            .for_each(|(chunk, &v)| {
                chunk.fill(v);
            });

        // Update ra and val at positions where we have entries.
        // Index is col * T_prime + row (address-major layout).
        let ra_ptr = ra.as_mut_ptr() as usize;
        let val_ptr = val.as_mut_ptr() as usize;

        self.entries.into_par_iter().for_each(|entry| {
            debug_assert!(
                entry.row < T_prime,
                "row {} >= T_prime {T_prime}",
                entry.row
            );
            let idx = entry.col * T_prime + entry.row;
            // SAFETY: Each entry has a unique (row, col) pair,
            // so writes to ra[idx] and val[idx] are disjoint across parallel iterations.
            unsafe {
                let ra_p = ra_ptr as *mut F;
                let val_p = val_ptr as *mut F;
                *ra_p.add(idx) = entry.ra_coeff;
                *val_p.add(idx) = entry.val_coeff;
            }
        });

        (ra.into(), val.into())
    }
}
