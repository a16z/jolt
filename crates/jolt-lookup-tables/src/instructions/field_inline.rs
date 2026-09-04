//! The field-inline bridge lookup.
//!
//! `FIELD_STORE_TO_X` is the only field-inline instruction with a lookup:
//! the x-register write is range-bound through `RangeCheck` exactly the way
//! `VirtualAdvice` binds a prover-supplied word. The rd write value is the
//! non-interleaved lookup operand (the `Advice` flag frees
//! `RightLookupOperand` from the RV64 operand rows), `RangeCheck` returns its
//! low 64 bits into `LookupOutput`, and the FR bridge rows
//! (`jolt-r1cs` `field_constraints::{ROW_STORE_TO_X, ROW_STORE_TO_X_LOOKUP}`)
//! pin both the operand and the write to `FieldRs1Value`, so the statement
//! is satisfiable only when the field value already fits in 64 bits — the
//! same condition under which the tracer executes the store.
use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_riscv::instructions::FieldStoreToX;
use jolt_riscv::JoltCycle;

impl_lookup_table!(FieldStoreToX, Some(RangeCheck));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for FieldStoreToX<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (0, 0)
    }

    fn to_lookup_operands(&self) -> (u64, u128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            0,
            (self.0.rd_vals().map_or(0, |(_, post)| post) & mask) as u128,
        )
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_lookup_output(&self) -> u64 {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        self.0.rd_vals().map_or(0, |(_, post)| post) & mask
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use super::*;
    use crate::tables::LookupTableKind;
    use crate::traits::InstructionLookupTable;
    use jolt_riscv::{JoltInstructionKind, JoltInstructionRow, JoltInstructionRowData};

    /// A cycle carrying only the rd write the bridge lookup reads.
    #[derive(Clone, Copy, Debug)]
    struct StoreCycle {
        instruction: JoltInstructionRow,
        rd: Option<(u64, u64)>,
    }

    impl From<StoreCycle> for JoltInstructionRow {
        fn from(cycle: StoreCycle) -> Self {
            cycle.instruction
        }
    }

    impl From<JoltInstructionRow> for StoreCycle {
        fn from(instruction: JoltInstructionRow) -> Self {
            Self {
                instruction,
                rd: None,
            }
        }
    }

    impl JoltInstructionRowData for StoreCycle {}

    impl JoltCycle for StoreCycle {
        type Instruction = JoltInstructionRow;

        fn instruction(&self) -> Self::Instruction {
            self.instruction
        }

        fn rs1_val(&self) -> Option<u64> {
            None
        }

        fn rs2_val(&self) -> Option<u64> {
            None
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

    fn store(rd_post: u64) -> FieldStoreToX<StoreCycle> {
        let instruction = JoltInstructionRow {
            instruction_kind: JoltInstructionKind::FIELD_STORE_TO_X,
            ..Default::default()
        };
        FieldStoreToX(StoreCycle {
            instruction,
            rd: Some((0, rd_post)),
        })
    }

    /// The bridge is the advice pattern: the rd write is the whole
    /// (non-interleaved) lookup index and `RangeCheck` hands it back, so the
    /// R1CS equalities against `FieldRs1Value` hold exactly for u64 values.
    #[test]
    fn store_to_x_is_a_range_check_of_the_rd_write() {
        for value in [0u64, 1, 42, u64::MAX] {
            let instruction = store(value);
            let table = InstructionLookupTable::<64>::lookup_table(&instruction).unwrap();
            assert!(matches!(table, LookupTableKind::RangeCheck(_)));
            assert_eq!(
                LookupQuery::<64>::to_lookup_operands(&instruction),
                (0, u128::from(value))
            );
            let index = LookupQuery::<64>::to_lookup_index(&instruction);
            assert_eq!(index, u128::from(value));
            assert_eq!(LookupQuery::<64>::to_lookup_output(&instruction), value);
            assert_eq!(table.materialize_entry(index), value);
        }
    }
}
