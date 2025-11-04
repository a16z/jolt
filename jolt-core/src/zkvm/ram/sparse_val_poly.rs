use allocative::Allocative;
use rayon::prelude::*;

use crate::field::JoltField;
use crate::utils::hashmap_or_vec::HashMapOrVec;
use crate::zkvm::ram::remap_address;
use common::jolt_device::MemoryLayout;
use tracer::instruction::{Cycle, RAMAccess};

/// Tuple of (row, col, coefficient)
#[derive(Allocative, Debug, PartialEq)]
pub struct MatrixEntry<F: JoltField>(pub usize, pub usize, pub F);

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
                let mut matrix_entries = Vec::with_capacity(trace_chunk.len() * 2);

                // Row index of the I matrix
                let mut j = chunk_index * chunk_size;
                for cycle in trace_chunk.iter() {
                    let ram_op = cycle.ram_access();
                    let k = remap_address(ram_op.address() as u64, &memory_layout).unwrap_or(0)
                        as usize;
                    match ram_op {
                        RAMAccess::Write(write) => {
                            matrix_entries.push(MatrixEntry(j, k, F::from_u64(write.pre_value)));
                            if j + 1 != (chunk_index + 1) * chunk_size {
                                // If next cycle is within the same chunk, append entry for
                                // write value
                                matrix_entries.push(MatrixEntry(
                                    j + 1,
                                    k,
                                    F::from_u64(write.post_value),
                                ));
                            }
                        }
                        RAMAccess::Read(read) => {
                            matrix_entries.push(MatrixEntry(j, k, F::from_u64(read.value)));
                        }
                        _ => {
                            matrix_entries.push(MatrixEntry(j, k, F::zero()));
                        }
                    }
                    j += 1;
                }

                matrix_entries.dedup();
                matrix_entries
            })
            .collect();

        SparseValPolynomial { row_chunks, K }
    }

    pub fn bind(&mut self, r: F::Challenge) {
        self.row_chunks.par_iter_mut().for_each(|row_chunk| {
            // Bind a cycle variable LowToHigh
            // Each row_chunk is bound serially in-place

            let mut next_bound_index = 0;
            let mut bound_indices: HashMapOrVec<usize> = HashMapOrVec::new(self.K, row_chunk.len());

            for i in 0..row_chunk.len() {
                let MatrixEntry(j_prime, k, coeff) = row_chunk[i];

                if let Some(bound_index) = bound_indices.get(k) {
                    if row_chunk[bound_index].0 == j_prime / 2 {
                        // Neighbor was already processed
                        debug_assert!(j_prime % 2 == 1);
                        let (lo, hi) = (row_chunk[bound_index].2, coeff);
                        row_chunk[bound_index].2 = lo + r * (hi - lo);
                        continue;
                    }
                }
                // Otherwise, this is the first time this k has been encountered
                // in this row

                // For SparseValPolynomial, the absence of a matrix entry implies
                // that its coefficient is equal to that of the the most recent matrix
                // entry in the same column.
                // So, we eagerly set bound_coeff := coeff, which is the correct
                // bound_coeff unless we encounter a neighboring coeff, which is
                // handled by the above if block.
                row_chunk[next_bound_index] = MatrixEntry(j_prime / 2, k, coeff);
                bound_indices[k] = next_bound_index;
                next_bound_index += 1;
            }
            row_chunk.truncate(next_bound_index);
        });
    }
}
