//! Lookup-table-related traits.

use jolt_field::Field;
use jolt_riscv::{JoltInstruction, JoltTraceRow};
use std::fmt::Debug;

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::interleave::interleave_bits;
use crate::tables::LookupTableKind;

/// Materialize and MLE-evaluate a single lookup table.
pub trait LookupTable: Clone + Debug + Send + Sync {
    fn materialize_entry(&self, index: u128) -> u64;

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>;
}

/// Maps an instruction to the lookup table it decomposes into for the proving system.
///
/// Returns `None` for instructions that don't use lookup tables (loads, stores,
/// system instructions). The prover uses this to route instruction evaluations
/// to the correct table during the instruction sumcheck.
///
/// Generic over `XLEN` so the same instruction can be used at production word
/// size (XLEN=64) and at test sizes (XLEN=8).
pub trait InstructionLookupTable<const XLEN: usize> {
    fn lookup_table(&self) -> Option<LookupTableKind<XLEN>>;
}

macro_rules! impl_jolt_instruction_lookup_table {
    (
        instructions: [$($kind:ident => $variant:ident => ($tag:expr, $canonical_name:expr)),* $(,)?]
    ) => {
        impl<const XLEN: usize, T> InstructionLookupTable<XLEN>
            for JoltInstruction<T>
        {
            #[inline]
            fn lookup_table(&self) -> Option<LookupTableKind<XLEN>> {
                match self {
                    JoltInstruction::Noop(_) => None,
                    $(
                        JoltInstruction::$variant(instruction) => instruction.lookup_table(),
                    )*
                }
            }
        }
    };
}

jolt_riscv::for_each_jolt_instruction_kind!(impl_jolt_instruction_lookup_table);

impl<const XLEN: usize> InstructionLookupTable<XLEN> for JoltTraceRow {
    #[inline]
    fn lookup_table(&self) -> Option<LookupTableKind<XLEN>> {
        let instruction = self.instruction_kind()?;
        InstructionLookupTable::<XLEN>::lookup_table(&instruction)
    }
}

macro_rules! impl_lookup_table {
    ($instr:ident, Some($table:ident)) => {
        impl<const XLEN: usize, T> $crate::traits::InstructionLookupTable<XLEN> for $instr<T> {
            #[inline]
            fn lookup_table(&self) -> Option<$crate::tables::LookupTableKind<XLEN>> {
                Some($crate::tables::LookupTableKind::$table(
                    ::core::default::Default::default(),
                ))
            }
        }
    };
    ($instr:ident, None) => {
        impl<const XLEN: usize, T> $crate::traits::InstructionLookupTable<XLEN> for $instr<T> {
            #[inline]
            fn lookup_table(&self) -> Option<$crate::tables::LookupTableKind<XLEN>> {
                None
            }
        }
    };
}

pub(crate) use impl_lookup_table;

/// Lookup-related queries on a concrete RISC-V cycle.
///
/// The default `to_lookup_operands` returns the instruction inputs unchanged,
/// and `to_lookup_index` interleaves their bits. Combined-operand tables
/// (ADD, MUL, advice, etc.) override these.
pub trait LookupQuery<const XLEN: usize> {
    /// Returns a tuple of the instruction's inputs. If the instruction has
    /// only one input, one of the tuple values will be 0.
    fn to_instruction_inputs(&self) -> (u64, i128);

    /// Returns a tuple of the instruction's lookup operands. By default
    /// these match the instruction inputs; combined-operand tables (ADD,
    /// MUL, advice) override this.
    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = self.to_instruction_inputs();
        (x, (y as u64) as u128)
    }

    /// Converts this instruction's operands into a lookup index by
    /// interleaving the two operands' bits.
    fn to_lookup_index(&self) -> u128 {
        let (x, y) = LookupQuery::<XLEN>::to_lookup_operands(self);
        interleave_bits(x, y as u64)
    }

    /// Computes the output lookup entry for this instruction as a u64.
    fn to_lookup_output(&self) -> u64;
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_riscv::{
        instructions::{Add, Ld, Noop},
        JoltInstructionRow,
    };

    #[test]
    fn aggregate_instruction_dispatches_lookup_table() {
        let add = JoltInstruction::Add(Add(JoltInstructionRow::default()));
        let load = JoltInstruction::Ld(Ld(JoltInstructionRow::default()));
        let noop = JoltInstruction::Noop(Noop(JoltInstructionRow::default()));

        assert!(InstructionLookupTable::<64>::lookup_table(&add).is_some());
        assert!(InstructionLookupTable::<64>::lookup_table(&load).is_none());
        assert!(InstructionLookupTable::<64>::lookup_table(&noop).is_none());
    }
}
