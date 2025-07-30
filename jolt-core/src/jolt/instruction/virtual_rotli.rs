use tracer::instruction::{virtual_rotli::VirtualROTLI, RISCVCycle};

use crate::jolt::lookup_table::{virtual_rotl::VirtualRotlTable, LookupTables};

use super::{CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualROTLI {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(VirtualRotlTable.into())
    }
}

impl InstructionFlags for VirtualROTLI {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::LeftOperandIsRs1Value as usize] = true;
        flags[CircuitFlags::RightOperandIsImm as usize] = true;
        flags[CircuitFlags::WriteLookupOutputToRD as usize] = true;
        flags[CircuitFlags::InlineSequenceInstruction as usize] =
            self.virtual_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
            self.virtual_sequence_remaining.unwrap_or(0) != 0;
        flags
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for RISCVCycle<VirtualROTLI> {
    fn to_instruction_inputs(&self) -> (u64, i64) {
        (
            self.register_state.rs1,
            self.instruction.operands.imm as i64,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<WORD_SIZE>::to_instruction_inputs(self);
        match WORD_SIZE {
            #[cfg(test)]
            8 => (x as u8).rotate_left((y as u8).trailing_zeros()) as u64,
            32 => (x as u32).rotate_left((y as u32).trailing_zeros()) as u64,
            64 => x.rotate_left((y as u64).trailing_zeros()),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::jolt::instruction::test::materialize_entry_test;

    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, VirtualROTLI>();
    }
}
