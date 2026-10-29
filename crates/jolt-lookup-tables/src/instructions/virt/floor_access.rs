use crate::tables::address_offset::AddressOffsetTable;
use crate::traits::impl_lookup_table;
use crate::traits::{LookupQuery, LookupTable};
use jolt_riscv::instructions::{LoadB, LoadH, LoadW, StoreB, StoreH, StoreW};
use jolt_riscv::{JoltCycle, JoltInstructionRow};

// The floor-addressed rows keep the signed load/store immediate: R1CS
// constraint 0 forms `RamAddress` from the same `Imm` column that feeds this
// lookup index, so the `u64`-wrapped immediate of the other `AddOperands` rows
// would put the address `2^64` away. The index `rs1 + imm` is the effective
// address itself, never negative on an honest trace.
macro_rules! floor_access {
    ($instr:ident, $table:ident, $width:literal) => {
        impl_lookup_table!($instr, Some($table));

        impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for $instr<C> {
            fn to_instruction_inputs(&self) -> (u64, i128) {
                (
                    self.0.rs1_val().unwrap_or(0),
                    Into::<JoltInstructionRow>::into(self.0.instruction())
                        .operands
                        .imm,
                )
            }

            fn to_lookup_operands(&self) -> (u64, u128) {
                let (rs1, imm) = LookupQuery::<XLEN>::to_instruction_inputs(self);
                (0, (rs1 as i128 + imm) as u128)
            }

            fn to_lookup_index(&self) -> u128 {
                LookupQuery::<XLEN>::to_lookup_operands(self).1
            }

            fn to_lookup_output(&self) -> u64 {
                AddressOffsetTable::<XLEN, $width>
                    .materialize_entry(LookupQuery::<XLEN>::to_lookup_index(self))
            }
        }
    };
}

floor_access!(LoadB, AddressOffsetB, 1);
floor_access!(LoadH, AddressOffsetH, 2);
floor_access!(LoadW, AddressOffsetW, 4);
floor_access!(StoreB, AddressOffsetB, 1);
floor_access!(StoreH, AddressOffsetH, 2);
floor_access!(StoreW, AddressOffsetW, 4);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instruction_inputs_match_constraint_test;
    use crate::InstructionLookupTable;
    use crate::XLEN;
    use jolt_riscv::JoltInstruction;
    use tracer::instruction::virtual_load_b::VirtualLoadB;
    use tracer::instruction::virtual_load_h::VirtualLoadH;
    use tracer::instruction::virtual_load_w::VirtualLoadW;
    use tracer::instruction::virtual_store_b::VirtualStoreB;
    use tracer::instruction::virtual_store_h::VirtualStoreH;
    use tracer::instruction::virtual_store_w::VirtualStoreW;

    #[test]
    fn instruction_inputs_match_constraint_floor_access() {
        instruction_inputs_match_constraint_test!(LoadB, VirtualLoadB);
        instruction_inputs_match_constraint_test!(LoadH, VirtualLoadH);
        instruction_inputs_match_constraint_test!(LoadW, VirtualLoadW);
        instruction_inputs_match_constraint_test!(StoreB, VirtualStoreB);
        instruction_inputs_match_constraint_test!(StoreH, VirtualStoreH);
        instruction_inputs_match_constraint_test!(StoreW, VirtualStoreW);
    }

    /// Pins each row's bound table to the RISC-V natural-alignment rule for
    /// its access width: `RamAddress = ea − LookupOutput` is doubleword-aligned
    /// exactly when `ea` is a multiple of the width.
    #[test]
    #[expect(clippy::unwrap_used)]
    fn ram_address_is_aligned_exactly_for_naturally_aligned_accesses() {
        let row = JoltInstructionRow::default();
        let rows = [
            (JoltInstruction::LoadB(LoadB(row)), 1),
            (JoltInstruction::LoadH(LoadH(row)), 2),
            (JoltInstruction::LoadW(LoadW(row)), 4),
            (JoltInstruction::StoreB(StoreB(row)), 1),
            (JoltInstruction::StoreH(StoreH(row)), 2),
            (JoltInstruction::StoreW(StoreW(row)), 4),
        ];
        for (instruction, width) in rows {
            let table = InstructionLookupTable::<XLEN>::lookup_table(&instruction).unwrap();
            for ea in 0x8000_1000u64..0x8000_1010 {
                let ram_address = ea - table.materialize_entry(ea.into());
                assert_eq!(
                    ram_address.is_multiple_of(8),
                    ea.is_multiple_of(width),
                    "{instruction:?} at ea={ea:#x}"
                );
            }
        }
    }
}
