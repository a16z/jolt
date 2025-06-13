use tracer::instruction::{
    virtual_assert_halfword_alignment::VirtualAssertHalfwordAlignment, RISCVCycle,
};

use crate::jolt::lookup_table::{halfword_alignment::HalfwordAlignmentTable, LookupTables};

use super::{CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualAssertHalfwordAlignment {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(HalfwordAlignmentTable.into())
    }
}

impl InstructionFlags for VirtualAssertHalfwordAlignment {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::Assert as usize] = true;
        flags[CircuitFlags::LeftOperandIsRs1Value as usize] = true;
        flags[CircuitFlags::RightOperandIsImm as usize] = true;
        flags[CircuitFlags::AddOperands as usize] = true;
        flags[CircuitFlags::InlineSequenceInstruction as usize] =
            self.virtual_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
            self.virtual_sequence_remaining.unwrap_or(0) != 0;
        flags
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for RISCVCycle<VirtualAssertHalfwordAlignment> {
    fn to_lookup_operands(&self) -> (u64, u64) {
        let (address, offset) = LookupQuery::<WORD_SIZE>::to_instruction_inputs(self);
        (0, (address as i64 + offset) as u64)
    }

    fn to_lookup_index(&self) -> u64 {
        LookupQuery::<WORD_SIZE>::to_lookup_operands(self).1
    }

    fn to_instruction_inputs(&self) -> (u64, i64) {
        match WORD_SIZE {
            #[cfg(test)]
            8 => (
                self.register_state.rs1 as u8 as u64,
                self.instruction.operands.imm,
            ),
            32 => (
                self.register_state.rs1 as u32 as u64,
                self.instruction.operands.imm,
            ),
            64 => (self.register_state.rs1, self.instruction.operands.imm),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn to_lookup_output(&self) -> u64 {
        (LookupQuery::<WORD_SIZE>::to_lookup_index(self) % 2 == 0).into()
    }
}

#[cfg(test)]
mod test {
    use crate::jolt::instruction::test::materialize_entry_test;

    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, VirtualAssertHalfwordAlignment>();
    }
}
