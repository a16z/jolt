//! This module provides the custom jolt instructions for the ONNX runtime.

use crate::jolt_onnx::tracer::tensor::QuantizedTensor;

pub mod max;
pub mod relu;
pub mod sigmoid;



// #[enum_dispatch]
// pub trait JoltONNXInstruction: Clone + Debug + Send + Sync + Serialize {
//     /// Converts this instruction's operands into a lookup index (as used in sparse-dense Shout).
//     /// By default, interleaves the two bits of the two operands together.
//     fn to_lookup_index(&self) -> u64 {
//         let (x, y) = self.operands();
//         interleave_bits(x as u32, y as u32)
//     }

//     /// Materializes the entire lookup table for this instruction (assuming an 8-bit word size).
//     #[cfg(test)]
//     fn materialize(&self) -> Vec<u64> {
//         (0..1 << 16).map(|i| self.materialize_entry(i)).collect()
//     }

//     /// Materialize the entry at the given `index` in the lookup table for this instruction.
//     fn materialize_entry(&self, index: u64) -> u64;

//     /// Evaluates the MLE of this lookup table on the given point `r`.
//     fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F;

//     /// Returns a tuple of the instruction's operands. If the instruction has only one operand,
//     /// one of the tuple values will be 0.
//     fn operands(&self) -> QuantizedTensor;

//     /// Combines `vals` according to the instruction's "collation" polynomial `g`.
//     /// If `vals` are subtable entries (as opposed to MLE evaluations), this function returns the
//     /// output of the instruction. This function can also be thought of as the low-degree extension
//     /// for the instruction.
//     ///
//     /// Params:
//     /// - `vals`: Subtable entries or MLE evaluations. Assumed to be ordered
//     ///   [T1_1, ..., T1_C, T2_1, ..., T2_C, ..., Tk_1, ..., Tk_C]
//     ///   where T1, ..., Tk are the unique subtable types used by this instruction, in the order
//     ///   given by the `subtables` method below. Note that some subtable values may be unused.
//     /// - `C`: The "dimension" of the decomposition, i.e. the number of values read from each subtable.
//     /// - `M`: The size of each subtable/memory.
//     ///
//     /// Returns: The combined value g(vals).
//     fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F;

//     /// The degree of the `g` polynomial described by `combine_lookups`
//     fn g_poly_degree(&self, C: usize) -> usize;

//     /// Returns a Vec of the unique subtable types used by this instruction. For some instructions,
//     /// e.g. SLL, the list of subtables depends on the dimension `C`.
//     fn subtables<F: JoltField>(
//         &self,
//         C: usize,
//         M: usize,
//     ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)>;

//     /// Converts the instruction operand(s) in their native word-sized representation into a Vec
//     /// of subtable lookups indices. The returned Vec is length `C`, with elements in [0, `log_M`).
//     fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize>;

//     /// Computes the output lookup entry for this instruction as a u64.
//     fn lookup_entry(&self) -> u64;

//     fn operand_chunks(&self, C: usize, log_M: usize) -> (Vec<u8>, Vec<u8>) {
//         assert!(
//             log_M % 2 == 0,
//             "log_M must be even for operand_chunks to work"
//         );
//         let (left_operand, right_operand) = self.operands();
//         (
//             chunk_operand(left_operand, C, log_M / 2),
//             chunk_operand(right_operand, C, log_M / 2),
//         )
//     }

//     fn random(&self, rng: &mut StdRng) -> Self;

//     fn slice_values<'a, F: JoltField>(&self, vals: &'a [F], C: usize, M: usize) -> Vec<&'a [F]> {
//         let mut offset = 0;
//         let mut slices = vec![];
//         for (_, indices) in self.subtables::<F>(C, M) {
//             slices.push(&vals[offset..offset + indices.len()]);
//             offset += indices.len();
//         }
//         assert_eq!(offset, vals.len());
//         slices
//     }
// }