//! Lookup-table-related traits.

use jolt_field::Field;
#[cfg(feature = "field-inline")]
use jolt_riscv::instructions::{
    FieldAdd, FieldAssertEq, FieldInv, FieldLoadFromX, FieldLoadImm, FieldMul, FieldStoreToX,
    FieldSub,
};
use jolt_riscv::{JoltCycle, JoltInstruction, JoltInstructionKind, JoltInstructionRowData};
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
        instructions: [$($(#[$meta:meta])* $kind:ident => $variant:ident => ($tag:expr, $canonical_name:expr)),* $(,)?]
    ) => {
        impl<const XLEN: usize, T: JoltInstructionRowData> InstructionLookupTable<XLEN>
            for JoltInstruction<T>
        {
            #[inline]
            fn lookup_table(&self) -> Option<LookupTableKind<XLEN>> {
                match self {
                    JoltInstruction::Noop(_) => None,
                    $(
                        $(#[$meta])*
                        JoltInstruction::$variant(instruction) => instruction.lookup_table(),
                    )*
                }
            }
        }
    };
}

jolt_riscv::for_each_jolt_instruction_kind!(impl_jolt_instruction_lookup_table);

macro_rules! impl_lookup_table {
    ($instr:ident, Some($table:ident)) => {
        impl<const XLEN: usize, T: jolt_riscv::JoltInstructionRowData>
            $crate::traits::InstructionLookupTable<XLEN> for $instr<T>
        {
            #[inline]
            fn lookup_table(&self) -> Option<$crate::tables::LookupTableKind<XLEN>> {
                Some($crate::tables::LookupTableKind::$table(
                    ::core::default::Default::default(),
                ))
            }
        }
    };
    ($instr:ident, None) => {
        impl<const XLEN: usize, T: jolt_riscv::JoltInstructionRowData>
            $crate::traits::InstructionLookupTable<XLEN> for $instr<T>
        {
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

#[cfg(feature = "field-inline")]
macro_rules! impl_field_inline_no_lookup {
    ($($instr:ident),* $(,)?) => {
        $(
            impl_lookup_table!($instr, None);

            impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN>
                for jolt_riscv::instructions::$instr<C>
            {
                #[inline]
                fn to_instruction_inputs(&self) -> (u64, i128) {
                    (0, 0)
                }

                #[inline]
                fn to_lookup_output(&self) -> u64 {
                    0
                }
            }
        )*
    };
}

#[cfg(feature = "field-inline")]
impl_field_inline_no_lookup!(
    FieldAdd,
    FieldSub,
    FieldMul,
    FieldInv,
    FieldAssertEq,
    FieldLoadFromX,
    FieldStoreToX,
    FieldLoadImm,
);

/// Lookup-query adapter for dynamic final Jolt instruction rows.
///
/// Per-instruction `LookupQuery` impls remain the source of operand packing,
/// lookup-index derivation, and output computation. This wrapper only routes a
/// runtime cycle to the typed instruction wrapper selected by its final Jolt
/// instruction kind.
#[derive(Clone, Copy, Debug)]
pub struct JoltLookupQuery<C> {
    pub instruction_kind: JoltInstructionKind,
    pub cycle: C,
}

impl<C> JoltLookupQuery<C> {
    #[inline]
    pub const fn new(instruction_kind: JoltInstructionKind, cycle: C) -> Self {
        Self {
            instruction_kind,
            cycle,
        }
    }
}

macro_rules! impl_jolt_lookup_query {
    (
        instructions: [$($(#[$meta:meta])* $kind:ident => $variant:ident => ($tag:expr, $canonical_name:expr)),* $(,)?]
    ) => {
        impl<const XLEN: usize, C> LookupQuery<XLEN> for JoltLookupQuery<C>
        where
            C: JoltCycle + Copy,
        {
            #[inline]
            fn to_instruction_inputs(&self) -> (u64, i128) {
                match self.instruction_kind {
                    JoltInstruction::Noop(_) => (0, 0),
                    $(
                        $(#[$meta])*
                        JoltInstruction::$variant(_) => {
                            let instruction = jolt_riscv::instructions::$variant(self.cycle);
                            LookupQuery::<XLEN>::to_instruction_inputs(&instruction)
                        }
                    )*
                }
            }

            #[inline]
            fn to_lookup_operands(&self) -> (u64, u128) {
                match self.instruction_kind {
                    JoltInstruction::Noop(_) => (0, 0),
                    $(
                        $(#[$meta])*
                        JoltInstruction::$variant(_) => {
                            let instruction = jolt_riscv::instructions::$variant(self.cycle);
                            LookupQuery::<XLEN>::to_lookup_operands(&instruction)
                        }
                    )*
                }
            }

            #[inline]
            fn to_lookup_index(&self) -> u128 {
                match self.instruction_kind {
                    JoltInstruction::Noop(_) => 0,
                    $(
                        $(#[$meta])*
                        JoltInstruction::$variant(_) => {
                            let instruction = jolt_riscv::instructions::$variant(self.cycle);
                            LookupQuery::<XLEN>::to_lookup_index(&instruction)
                        }
                    )*
                }
            }

            #[inline]
            fn to_lookup_output(&self) -> u64 {
                match self.instruction_kind {
                    JoltInstruction::Noop(_) => 0,
                    $(
                        $(#[$meta])*
                        JoltInstruction::$variant(_) => {
                            let instruction = jolt_riscv::instructions::$variant(self.cycle);
                            LookupQuery::<XLEN>::to_lookup_output(&instruction)
                        }
                    )*
                }
            }
        }
    };
}

jolt_riscv::for_each_jolt_instruction_kind!(impl_jolt_lookup_query);

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_riscv::{
        instructions::{Add, Ld, Noop},
        JoltInstructionRow, NormalizedOperands,
    };

    #[derive(Clone, Copy, Debug)]
    struct TestCycle {
        instruction: JoltInstructionRow,
        rs1: Option<u64>,
        rs2: Option<u64>,
        rd: Option<(u64, u64)>,
    }

    impl JoltCycle for TestCycle {
        type Instruction = JoltInstructionRow;

        fn instruction(&self) -> Self::Instruction {
            self.instruction
        }

        fn rs1_val(&self) -> Option<u64> {
            self.rs1
        }

        fn rs2_val(&self) -> Option<u64> {
            self.rs2
        }

        fn rd_vals(&self) -> Option<(u64, u64)> {
            self.rd
        }

        fn ram_access_address(&self) -> Option<u64> {
            None
        }

        fn ram_read_value(&self) -> Option<u64> {
            None
        }

        fn ram_write_value(&self) -> Option<u64> {
            None
        }
    }

    #[test]
    fn aggregate_instruction_dispatches_lookup_table() {
        let add = JoltInstruction::Add(Add(JoltInstructionRow::default()));
        let load = JoltInstruction::Ld(Ld(JoltInstructionRow::default()));
        let noop = JoltInstruction::Noop(Noop(JoltInstructionRow::default()));

        assert!(InstructionLookupTable::<64>::lookup_table(&add).is_some());
        assert!(InstructionLookupTable::<64>::lookup_table(&load).is_none());
        assert!(InstructionLookupTable::<64>::lookup_table(&noop).is_none());
    }

    #[test]
    fn dynamic_lookup_query_dispatches_to_instruction_impl() {
        let instruction = JoltInstructionRow {
            instruction_kind: JoltInstructionKind::ADDI,
            operands: NormalizedOperands {
                rd: Some(1),
                rs1: Some(2),
                rs2: None,
                imm: -1,
            },
            ..Default::default()
        };
        let cycle = TestCycle {
            instruction,
            rs1: Some(10),
            rs2: None,
            rd: Some((0, 9)),
        };
        let query = JoltLookupQuery::new(JoltInstructionKind::ADDI, cycle);

        assert_eq!(
            LookupQuery::<64>::to_lookup_index(&query),
            (1_u128 << 64) + 9
        );
    }
}
