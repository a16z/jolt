use std::sync::{Arc, Mutex};

use allocative::Allocative;
use num::Integer;
use rayon::prelude::*;

use crate::zkvm::ram::remap_address;
use crate::{field::JoltField, poly::multilinear_polynomial::MultilinearPolynomial};
use common::jolt_device::MemoryLayout;
use tracer::instruction::{Cycle, RAMAccess};

/// Represents a non-zero coefficient of the ra(k, j) polynomial and the
/// corresponding coefficient of the Val(k, j) polynomial. Conceptually,
/// both ra and Val can be seen as K x T matrices, hence `MatrixEntry`.
#[derive(Allocative, Debug, PartialEq)]
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
    prev_val: u64,
    /// In round i, each MatrixEntry represents a coefficient
    ///   Val(k, j', r)
    /// which is some combination of Val(k, j', 00...0), ...
    /// Val(k, j', 11...1).
    /// `next_val` contains the unbound coefficient after
    /// Val(k, j', 00...0) –– abusing notation, `next_val` is
    /// Val(k, j'+1, 00...0)
    next_val: u64,
    /// The Val coefficient for this matrix entry.
    pub val_coeff: F,
    /// The ra coefficient for this matrix entry. Note that for RAM,
    /// ra and wa are the same polynomial.
    pub ra_coeff: F,
}

/// Represents the ra(k, j) and Val(k, j) polynomials for the RAM
/// read/write-checking sumcheck. Conceptually, both ra and Val can
/// be seen as K x T matrices. This is far too large to explicitly
/// store in memory, but we observe that, while binding cycle variables,
/// we only need a small fraction of the coefficients for the purposes
/// of sumcheck. The coefficients we do need are stored in this data structure.
#[derive(Allocative, Debug, Default)]
pub struct SparseMatrixPolynomial<F: JoltField> {
    pub entries: Vec<MatrixEntry<F>>,
}

impl<F: JoltField> SparseMatrixPolynomial<F> {
    pub fn new(trace: &[Cycle], memory_layout: &MemoryLayout) -> Self {
        let entries: Vec<_> = trace
            .par_iter()
            .enumerate()
            .filter_map(|(j, cycle)| {
                let ram_op = cycle.ram_access();
                match ram_op {
                    RAMAccess::Write(write) => Some(MatrixEntry {
                        row: j,
                        col: remap_address(write.address, &memory_layout).unwrap() as usize,
                        ra_coeff: F::one(),
                        val_coeff: F::from_u64(write.pre_value),
                        prev_val: write.pre_value,
                        next_val: write.post_value,
                    }),
                    RAMAccess::Read(read) => Some(MatrixEntry {
                        row: j,
                        col: remap_address(read.address, &memory_layout).unwrap() as usize,
                        ra_coeff: F::one(),
                        val_coeff: F::from_u64(read.value),
                        prev_val: read.value,
                        next_val: read.value,
                    }),
                    _ => None,
                }
            })
            .collect();

        SparseMatrixPolynomial { entries }
    }

    fn bind_rows(
        even_row: &[MatrixEntry<F>],
        odd_row: &[MatrixEntry<F>],
        r: F::Challenge,
    ) -> Vec<MatrixEntry<F>> {
        /// Threshold where we stop parallelizing and do a plain linear merge.
        const PAR_THRESHOLD: usize = 32_768;

        // small inputs: do the O(n) sequential merge
        if even_row.len() + odd_row.len() <= PAR_THRESHOLD {
            return Self::seq_bind_rows(even_row, odd_row, r);
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
        // and odd_row[odd_pivot_idx..]. Merge those two regions in parallel.
        let (mut left, mut right) = rayon::join(
            || Self::bind_rows(&even_row[..even_pivot_idx], &odd_row[..odd_pivot_idx], r),
            || Self::bind_rows(&even_row[even_pivot_idx..], &odd_row[odd_pivot_idx..], r),
        );

        // Stitch results. (Avoids unsafe; one extra reallocation at most.)
        left.append(&mut right);
        left
    }

    fn seq_bind_rows(
        even: &[MatrixEntry<F>],
        odd: &[MatrixEntry<F>],
        r: F::Challenge,
    ) -> Vec<MatrixEntry<F>> {
        let mut i = 0;
        let mut j = 0;
        let mut out = Vec::with_capacity(even.len() + odd.len());

        while i < even.len() && j < odd.len() {
            if even[i].col == odd[j].col {
                let bound_entry = Self::bind_entries(Some(&even[i]), Some(&odd[j]), r);
                out.push(bound_entry);
                i += 1;
                j += 1;
            } else if even[i].col < odd[j].col {
                let bound_entry = Self::bind_entries(Some(&even[i]), None, r);
                out.push(bound_entry);
                i += 1;
            } else {
                let bound_entry = Self::bind_entries(None, Some(&odd[j]), r);
                out.push(bound_entry);
                j += 1;
            }
        }
        for remaining_even_entry in even[i..].iter() {
            let bound_entry = Self::bind_entries(Some(&remaining_even_entry), None, r);
            out.push(bound_entry);
        }
        for remaining_odd_entry in odd[j..].iter() {
            let bound_entry = Self::bind_entries(None, Some(&remaining_odd_entry), r);
            out.push(bound_entry);
        }
        out
    }

    // TODO(moodlezoup): Optimize for zeros in Val
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
                    ra_coeff: even.ra_coeff + r * (odd.ra_coeff - even.ra_coeff),
                    val_coeff: even.val_coeff + r * (odd.val_coeff - even.val_coeff),
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
                let odd_val_coeff = F::from_u64(even.next_val);
                MatrixEntry {
                    row: even.row / 2,
                    col: even.col,
                    ra_coeff: (F::one() - r) * even.ra_coeff,
                    val_coeff: even.val_coeff + r * (odd_val_coeff - even.val_coeff),
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
                let even_val_coeff = F::from_u64(odd.prev_val);
                MatrixEntry {
                    row: odd.row / 2,
                    col: odd.col,
                    ra_coeff: r * odd.ra_coeff,
                    val_coeff: even_val_coeff + r * (odd.val_coeff - even_val_coeff),
                    prev_val: odd.prev_val,
                    next_val: odd.next_val,
                }
            }
            (None, None) => panic!("Both entries are None"),
        }
    }

    #[tracing::instrument(skip_all, name = "SparseMatrixPolynomial::bind")]
    pub fn bind(&mut self, r: F::Challenge) {
        self.entries = self
            .entries
            .par_chunk_by(|x, y| x.row / 2 == y.row / 2)
            .flat_map(|entries| {
                let odd_row_start_index = entries.partition_point(|entry| entry.row.is_even());
                let (even_row, odd_row) = entries.split_at(odd_row_start_index);
                Self::bind_rows(&even_row, &odd_row, r)
            })
            .collect();
    }

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
        // and odd_row[odd_pivot_idx..]. Merge those two regions in parallel.
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
            let evals = Self::compute_evals(Some(&remaining_even_entry), None, inc_evals, gamma);
            evals_accumulator[0] += evals[0];
            evals_accumulator[1] += evals[1];
        }
        for remaining_odd_entry in odd[j..].iter() {
            let evals = Self::compute_evals(None, Some(&remaining_odd_entry), inc_evals, gamma);
            evals_accumulator[0] += evals[0];
            evals_accumulator[1] += evals[1];
        }

        evals_accumulator
    }

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
                let odd_val_coeff = F::from_u64(even.next_val);
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
                let even_val_coeff = F::from_u64(odd.prev_val);
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
