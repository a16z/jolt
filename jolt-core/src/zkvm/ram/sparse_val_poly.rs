use allocative::Allocative;
use num::Integer;
use rayon::prelude::*;

use crate::field::JoltField;
use crate::zkvm::ram::remap_address;
use common::jolt_device::MemoryLayout;
use tracer::instruction::{Cycle, RAMAccess};

/// Represents a coefficient of the Val(k, j) polynomial, conceptually
/// equivalent to an entry in a K x T matrix.
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
    /// `prev` contains the unbound coefficient before
    /// Val(k, j', 00...0) –– abusing notation, `prev` is
    /// Val(k, j'-1, 11...1)
    prev: u64,
    /// In round i, each MatrixEntry represents a coefficient
    ///   Val(k, j', r)
    /// which is some combination of Val(k, j', 00...0), ...
    /// Val(k, j', 11...1).
    /// `next` contains the unbound coefficient after
    /// Val(k, j', 00...0) –– abusing notation, `next` is
    /// Val(k, j'+1, 00...0)
    next: u64,
    /// The (potentially bound) Val coefficient for this matrix entry.
    pub coeff: F,
}

/// Represents the Val(k, j) polynomial for the RAM read/write-checking
/// sumcheck. Conceptually, the Val polynomial can be viewed as a K x T
/// matrix, where each row contains the state of memory at cycle j. This
/// matrix is far too large to explicitly store in memory, but we observe
/// that, while binding cycle variables, we only need a small fraction of
/// the KT total coefficients for the purposes of sumcheck. The coefficients
/// we do need are stored in this data structure.
#[derive(Allocative)]
pub struct SparseValPolynomial<F: JoltField> {
    pub entries: Vec<MatrixEntry<F>>,
}

impl<F: JoltField> SparseValPolynomial<F> {
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
                        coeff: F::from_u64(write.pre_value),
                        prev: write.pre_value,
                        next: write.post_value,
                    }),
                    RAMAccess::Read(read) => Some(MatrixEntry {
                        row: j,
                        col: remap_address(read.address, &memory_layout).unwrap() as usize,
                        coeff: F::from_u64(read.value),
                        prev: read.value,
                        next: read.value,
                    }),
                    _ => None,
                }
            })
            .collect();

        println!("{entries:?}");

        SparseValPolynomial { entries }
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
                    coeff: even.coeff + r * (odd.coeff - even.coeff),
                    prev: even.prev,
                    next: odd.next,
                }
            }
            (Some(even), None) => {
                // For SparseValPolynomial, the absence of a matrix entry implies
                // that its coeff has not been bound yet.
                // The absence of an odd-row entry in the same column as even[i]
                // means that its implicit coeff is even[i].next.
                let odd_coeff = F::from_u64(even.next);
                MatrixEntry {
                    row: even.row / 2,
                    col: even.col,
                    coeff: even.coeff + r * (odd_coeff - even.coeff),
                    prev: even.prev,
                    next: even.next,
                }
            }
            (None, Some(odd)) => {
                // For SparseValPolynomial, the absence of a matrix entry implies
                // that its coeff has not been bound yet.
                // The absence of an even-row entry in the same column as odd[j]
                // means that its implicit coeff is odd[j].prev.
                let even_coeff = F::from_u64(odd.prev);
                MatrixEntry {
                    row: odd.row / 2,
                    col: odd.col,
                    coeff: even_coeff + r * (odd.coeff - even_coeff),
                    prev: odd.prev,
                    next: odd.next,
                }
            }
            (None, None) => panic!("Both entries are None"),
        }
    }

    pub fn bind(&mut self, r: F::Challenge) {
        self.entries = self
            .entries
            .par_chunk_by(|x, y| x.row / 2 == y.row / 2)
            .flat_map(|rows| {
                let odd_row_start_index = rows.partition_point(|entry| entry.row.is_even());
                let (even_row, odd_row) = rows.split_at(odd_row_start_index);
                Self::bind_rows(&even_row, &odd_row, r)
            })
            .collect();
    }
}
