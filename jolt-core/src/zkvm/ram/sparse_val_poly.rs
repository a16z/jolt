use allocative::Allocative;
use rayon::prelude::*;

use crate::field::JoltField;
use crate::utils::hashmap_or_vec::HashMapOrVec;
use crate::zkvm::ram::remap_address;
use common::jolt_device::MemoryLayout;
use tracer::instruction::{Cycle, RAMAccess};

/// Tuple of (row, col, coeff)
#[derive(Allocative, Debug, PartialEq)]
pub struct MatrixEntry<F: JoltField> {
    pub row: usize,
    pub col: usize,
    prev: u64,
    next: u64,
    pub coeff: F,
}

#[derive(Allocative)]
pub struct SparseValPolynomial<F: JoltField> {
    /// Chunks of rows
    pub row_chunks: Vec<Vec<MatrixEntry<F>>>,
    K: usize,
}

impl<F: JoltField> SparseValPolynomial<F> {
    pub fn new(trace: &[Cycle], memory_layout: &MemoryLayout, K: usize) -> Self {
        let T = trace.len();
        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = T / num_chunks;

        let row_chunks: Vec<_> = trace
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, trace_chunk)| {
                let mut matrix_entries = Vec::with_capacity(trace_chunk.len());

                // Row index of the I matrix
                let mut j = chunk_index * chunk_size;
                for cycle in trace_chunk.iter() {
                    let ram_op = cycle.ram_access();
                    let k = remap_address(ram_op.address() as u64, &memory_layout).unwrap_or(0)
                        as usize;
                    match ram_op {
                        RAMAccess::Write(write) => {
                            matrix_entries.push(MatrixEntry {
                                row: j,
                                col: k,
                                coeff: F::from_u64(write.pre_value),
                                prev: write.pre_value,
                                next: write.post_value,
                            });
                        }
                        RAMAccess::Read(read) => {
                            matrix_entries.push(MatrixEntry {
                                row: j,
                                col: k,
                                coeff: F::from_u64(read.value),
                                prev: read.value,
                                next: read.value,
                            });
                        }
                        _ => {
                            // matrix_entries.push(MatrixEntry(j, k, F::zero()));
                        }
                    }
                    j += 1;
                }

                matrix_entries
            })
            .collect();

        println!("{row_chunks:?}");

        SparseValPolynomial { row_chunks, K }
    }

    pub fn bind(&mut self, r: F::Challenge) {
        self.row_chunks.par_iter_mut().for_each(|row_chunk| {
            // Bind a cycle variable LowToHigh
            // Each row_chunk is bound serially in-place

            let mut next_bound_index = 0;
            let mut bound_indices: HashMapOrVec<usize> = HashMapOrVec::new(self.K, row_chunk.len());

            for i in 0..row_chunk.len() {
                let MatrixEntry {
                    row,
                    col,
                    coeff,
                    prev,
                    next,
                } = row_chunk[i];

                if let Some(bound_index) = bound_indices.get(col) {
                    if row_chunk[bound_index].row == row / 2 {
                        debug_assert!(row % 2 == 1);
                        // Neighbor was already processed
                        let neighbor = &mut row_chunk[bound_index];
                        // Neighbor's coeff was eagerly bound to the following:
                        //   (1 - r) * coeff + r * next,
                        // We want to correct the value to replace `next` with the
                        // `coeff` we have now encountered.
                        neighbor.coeff += r * (coeff - F::from_u64(neighbor.next));
                        neighbor.next = next;
                        continue;
                    }
                }

                // Otherwise, this is the first time this col has been encountered
                // in this pair of rows
                let new_entry = if row % 2 == 0 {
                    // For SparseValPolynomial, the absence of a matrix entry implies
                    // that its coeff has not been bound yet.
                    //
                    // Here, we eagerly bind `coeff` with `next`, as if the next
                    // row doesn't have an entry in the same column. If we do encounter
                    // an entry in the next row, we will correct this bound value (see
                    // logic in above if-block).
                    MatrixEntry {
                        row: row / 2,
                        col,
                        // (1 - r) * coeff + r * next
                        coeff: coeff + r * (F::from_u64(next) - coeff),
                        prev,
                        next,
                    }
                } else {
                    // Odd row, where the previous row did not have a matrix entry in
                    // this column. The absence of a matrix entry implies that its
                    // coeff has not been bound yet, which means it is `prev`.
                    let prev_F = F::from_u64(prev);
                    MatrixEntry {
                        row: row / 2,
                        col,
                        // (1 - r) * prev + r * coeff
                        coeff: prev_F + r * (coeff - prev_F),
                        prev,
                        next,
                    }
                };
                row_chunk[next_bound_index] = new_entry;
                bound_indices[col] = next_bound_index;
                next_bound_index += 1;
            }
            row_chunk.truncate(next_bound_index);
        });
    }
}
