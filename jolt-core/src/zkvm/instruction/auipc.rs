use crate::zkvm::instruction::{InstructionFlags, NUM_INSTRUCTION_FLAGS};
use tracer::instruction::{auipc::AUIPC, RISCVCycle};

use crate::zkvm::lookup_table::{range_check::RangeCheckTable, LookupTables};

use super::{CircuitFlags, Flags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const XLEN: usize> InstructionLookup<XLEN> for AUIPC {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        Some(RangeCheckTable.into())
    }
}

impl Flags for AUIPC {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::AddOperands] = true;
        flags[CircuitFlags::WriteLookupOutputToRD] = true;
        flags[CircuitFlags::VirtualInstruction] = self.virtual_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC] =
            self.virtual_sequence_remaining.unwrap_or(0) != 0;
        flags[CircuitFlags::IsFirstInSequence] = self.is_first_in_sequence;
        flags[CircuitFlags::IsCompressed] = self.is_compressed;
        flags
    }

    fn instruction_flags(&self) -> [bool; NUM_INSTRUCTION_FLAGS] {
        let mut flags = [false; NUM_INSTRUCTION_FLAGS];
        flags[InstructionFlags::LeftOperandIsPC] = true;
        flags[InstructionFlags::RightOperandIsImm] = true;
        flags[InstructionFlags::IsRdZero] = self.operands.rd == 0;
        flags
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<AUIPC> {
    fn to_lookup_operands(&self) -> (u64, u128) {
        let (pc, imm) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (0, (pc as i128 + imm) as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_instruction_inputs(&self) -> (u64, i128) {
        match XLEN {
            #[cfg(test)]
            8 => (
                self.instruction.address as u8 as u64,
                self.instruction.operands.imm as u8 as u64 as i128,
            ),
            32 => (
                self.instruction.address as u32 as u64,
                self.instruction.operands.imm as u32 as u64 as i128,
            ),
            64 => (
                self.instruction.address,
                self.instruction.operands.imm as i128,
            ),
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }

    fn to_lookup_output(&self) -> u64 {
        let (pc, imm) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        match XLEN {
            #[cfg(test)]
            8 => (pc as i8).overflowing_add(imm as i8).0 as u8 as u64,
            32 => (pc as i32).overflowing_add(imm as i32).0 as u32 as u64,
            64 => (pc as i64).overflowing_add(imm as i64).0 as u64,
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
        materialize_entry_test::<Fr, AUIPC>();
    }
}
