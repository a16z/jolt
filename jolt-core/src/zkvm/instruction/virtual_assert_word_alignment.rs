use tracer::instruction::virtual_assert_word_alignment::VirtualAssertWordAlignment;
use tracer::instruction::RISCVCycle;

use crate::zkvm::lookup_table::word_alignment::WordAlignmentTable;
use crate::zkvm::lookup_table::LookupTables;

use super::{CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualAssertWordAlignment {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(WordAlignmentTable.into())
    }
}

impl InstructionFlags for VirtualAssertWordAlignment {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::Assert as usize] = true;
        flags[CircuitFlags::LeftOperandIsRs1Value as usize] = true;
        flags[CircuitFlags::RightOperandIsImm as usize] = true;
        flags[CircuitFlags::AddOperands as usize] = true;
        flags[CircuitFlags::InlineSequenceInstruction as usize] =
            self.inline_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
            self.inline_sequence_remaining.unwrap_or(0) != 0;
        flags[CircuitFlags::IsCompressed as usize] = self.is_compressed;
        flags
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for RISCVCycle<VirtualAssertWordAlignment> {
    fn to_lookup_operands(&self) -> (u64, u128) {
        let (address, offset) = LookupQuery::<WORD_SIZE>::to_instruction_inputs(self);
        (0, (address as i128 + offset) as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<WORD_SIZE>::to_lookup_operands(self).1
    }

    fn to_instruction_inputs(&self) -> (u64, i128) {
        match WORD_SIZE {
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
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn to_lookup_output(&self) -> u64 {
        LookupQuery::<WORD_SIZE>::to_lookup_index(self)
            .is_multiple_of(4)
            .into()
    }
}

#[cfg(test)]
mod test {
    use crate::zkvm::instruction::test::materialize_entry_test;

    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, VirtualAssertWordAlignment>();
    }
}
