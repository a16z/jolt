use crate::zkvm::instruction::{InstructionFlags, NUM_INSTRUCTION_FLAGS};
use tracer::instruction::{jalr::JALR, RISCVCycle};

use crate::zkvm::lookup_table::{range_check::RangeCheckTable, LookupTables};

use super::{CircuitFlags, Flags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const XLEN: usize> InstructionLookup<XLEN> for JALR {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        Some(RangeCheckTable.into())
    }
}

impl Flags for JALR {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::Jump] = true;
        flags[CircuitFlags::AddOperands] = true;
        flags[CircuitFlags::VirtualInstruction] = self.virtual_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC] =
            self.virtual_sequence_remaining.unwrap_or(0) != 0;
        flags[CircuitFlags::IsFirstInSequence] = self.is_first_in_sequence;
        flags[CircuitFlags::IsCompressed] = self.is_compressed;
        flags
    }

    fn instruction_flags(&self) -> [bool; NUM_INSTRUCTION_FLAGS] {
        let mut flags = [false; NUM_INSTRUCTION_FLAGS];
        flags[InstructionFlags::LeftOperandIsRs1Value] = true;
        flags[InstructionFlags::RightOperandIsImm] = true;
        flags[InstructionFlags::IsRdNotZero] = self.operands.rd != 0;
        flags
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<JALR> {
    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (0, (x as i128 + y) as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_instruction_inputs(&self) -> (u64, i128) {
        match XLEN {
            #[cfg(test)]
            8 => (
                self.register_state.rs1 as u8 as u64,
                self.instruction.operands.imm as u8 as u64 as i128,
            ),
            32 => (
                self.register_state.rs1 as u32 as u64,
                self.instruction.operands.imm as u32 as u64 as i128,
            ),
            64 => (
                self.register_state.rs1,
                self.instruction.operands.imm as i128,
            ),
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        match XLEN {
            #[cfg(test)]
            8 => (x as i8).overflowing_add(y as i8).0 as u8 as u64,
            32 => (x as i32).overflowing_add(y as i32).0 as u32 as u64,
            64 => (x as i64).overflowing_add(y as i64).0 as u64,
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::zkvm::instruction::test::materialize_entry_test;

    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, JALR>();
    }
}
