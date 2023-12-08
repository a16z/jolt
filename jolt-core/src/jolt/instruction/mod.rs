use ark_ff::PrimeField;
use ark_std::log2;
use enum_dispatch::enum_dispatch;
use rand::prelude::StdRng;
use std::marker::Sync;

use crate::jolt::subtable::LassoSubtable;
use crate::utils::index_to_field_bitvector;

#[enum_dispatch]
pub trait JoltInstruction: Sync {
    /// Combines `vals` according to the instruction's "collation" polynomial `g`.
    /// If `vals` are subtable entries (as opposed to MLE evaluations), this function returns the
    /// output of the instruction. This function can also be thought of as the low-degree extension
    /// for the instruction.
    ///
    /// Params:
    /// - `vals`: Subtable entries or MLE evaluations. Assumed to be ordered
    ///           [T1_1, ..., T1_C, T2_1, ..., T2_C, ..., Tk_1, ..., Tk_C]
    ///           where T1, ..., Tk are the unique subtable types used by this instruction, in the order
    ///           given by the `subtables` method below. Note that some subtable values may be unused.
    /// - `C`: The "dimension" of the decomposition, i.e. the number of values read from each subtable.
    /// - `M`: The size of each subtable/memory.
    ///
    /// Returns: The combined value g(vals).
    fn combine_lookups<F: PrimeField>(&self, vals: &[F], C: usize, M: usize) -> F;
    /// The degree of the `g` polynomial described by `combine_lookups`
    fn g_poly_degree(&self, C: usize) -> usize;
    /// Returns a Vec of the unique subtable types used by this instruction. For some instructions,
    /// e.g. SLL, the list of subtables depends on the dimension `C`.
    fn subtables<F: PrimeField>(&self, C: usize) -> Vec<Box<dyn LassoSubtable<F>>>;
    /// Converts the instruction operand(s) in their native word-sized representation into a Vec
    /// of subtable lookups indices. The returned Vec is length `C`, with elements in [0, `log_M`).
    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize>;
    fn lookup_entry<F: PrimeField>(&self, C: usize, M: usize) -> F {
        let log_M = log2(M) as usize;

        let subtable_lookup_indices = self.to_indices(C, log2(M) as usize);

        let subtable_lookup_values: Vec<F> = self
            .subtables::<F>(C)
            .iter()
            .flat_map(|subtable| {
                subtable_lookup_indices.iter().map(|&lookup_index| {
                    subtable.evaluate_mle(&index_to_field_bitvector(lookup_index, log_M))
                })
            })
            .collect();

        self.combine_lookups(&subtable_lookup_values, C, M)
    }

    fn random(&self, rng: &mut StdRng) -> Self;
}

pub trait Opcode {
    /// Converts a variant of an instruction set enum into its canonical "opcode" value.
    fn to_opcode(&self) -> u8 {
        unsafe { *<*const _>::from(self).cast::<u8>() }
    }
}

pub mod add;
pub mod and;
pub mod beq;
pub mod bge;
pub mod bgeu;
pub mod blt;
pub mod bltu;
pub mod bne;
pub mod jal;
pub mod jalr;
pub mod or;
pub mod sll;
pub mod slt;
pub mod sltu;
pub mod sra;
pub mod srl;
pub mod sub;
pub mod xor;

#[cfg(test)]
pub mod test;
