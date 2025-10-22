use crate::zkvm::instruction::{InstructionFlags, NUM_INSTRUCTION_FLAGS};
use tracer::instruction::{slti::SLTI, RISCVCycle};

use crate::zkvm::lookup_table::{signed_less_than::SignedLessThanTable, LookupTables};

use super::{CircuitFlags, Flags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const XLEN: usize> InstructionLookup<XLEN> for SLTI {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        Some(SignedLessThanTable.into())
    }
}

impl Flags for SLTI {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
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
        flags[InstructionFlags::LeftOperandIsRs1Value] = true;
        flags[InstructionFlags::RightOperandIsImm] = true;
        flags[InstructionFlags::IsRdZero] = self.operands.rd == 0;
        flags
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<SLTI> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        match XLEN {
            #[cfg(test)]
            8 => (
                self.register_state.rs1 as u8 as u64,
                self.instruction.operands.imm as u8 as i128,
            ),
            32 => (
                self.register_state.rs1 as u32 as u64,
                self.instruction.operands.imm as u32 as i128,
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
            8 => ((x as i8) < (y as i8)).into(),
            32 => ((x as i32) < (y as i32)).into(),
            64 => ((x as i64) < (y as i64)).into(),
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
        materialize_entry_test::<Fr, SLTI>();
    }
}
