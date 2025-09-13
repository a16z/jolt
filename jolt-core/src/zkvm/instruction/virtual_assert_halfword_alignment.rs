use tracer::instruction::{
    virtual_assert_halfword_alignment::VirtualAssertHalfwordAlignment, RISCVCycle,
};

use crate::zkvm::lookup_table::{halfword_alignment::HalfwordAlignmentTable, LookupTables};

use super::{
    CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, RightInputValue,
    NUM_CIRCUIT_FLAGS,
};

impl<const XLEN: usize> InstructionLookup<XLEN> for VirtualAssertHalfwordAlignment {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
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
            self.inline_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
            self.inline_sequence_remaining.unwrap_or(0) != 0;
        flags[CircuitFlags::IsCompressed as usize] = self.is_compressed;
        flags
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualAssertHalfwordAlignment> {
    fn to_lookup_operands(&self) -> (u64, u128) {
        let (address, offset) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (0, (address as i128 + offset.as_i128()) as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_instruction_inputs(&self) -> (u64, RightInputValue) {
        match XLEN {
            #[cfg(test)]
            8 => (
                self.register_state.rs1 as u8 as u64,
                RightInputValue::Signed(self.instruction.operands.imm as u8 as i64),
            ),
            32 => (
                self.register_state.rs1 as u32 as u64,
                RightInputValue::Signed(self.instruction.operands.imm as u32 as i64),
            ),
            64 => (
                self.register_state.rs1,
                RightInputValue::Signed(self.instruction.operands.imm),
            ),
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }

    fn to_lookup_output(&self) -> u64 {
        LookupQuery::<XLEN>::to_lookup_index(self)
            .is_multiple_of(2)
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
        materialize_entry_test::<Fr, VirtualAssertHalfwordAlignment>();
    }
}
