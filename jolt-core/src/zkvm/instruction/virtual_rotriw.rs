use tracer::instruction::{virtual_rotriw::VirtualROTRIW, RISCVCycle};

use crate::zkvm::lookup_table::LookupTables;

use super::{CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualROTRIW {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        todo!()
    }
}

impl InstructionFlags for VirtualROTRIW {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::LeftOperandIsRs1Value as usize] = true;
        flags[CircuitFlags::RightOperandIsImm as usize] = true;
        flags[CircuitFlags::WriteLookupOutputToRD as usize] = true;
        flags[CircuitFlags::InlineSequenceInstruction as usize] =
            self.inline_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
            self.inline_sequence_remaining.unwrap_or(0) != 0;
        flags[CircuitFlags::IsCompressed as usize] = self.is_compressed;
        flags
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for RISCVCycle<VirtualROTRIW> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (
            self.register_state.rs1,
            self.instruction.operands.imm as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<WORD_SIZE>::to_instruction_inputs(self);
        match WORD_SIZE {
            #[cfg(test)]
            8 => {
                let (x, y) = (x as u8, (y as u8).trailing_zeros());
                (((x & 0x0F) >> (y % 4)) | (((x & 0x0F) << (4 - (y % 4))) & 0x0F)) as u64
            }
            32 => (x as u16).rotate_right((y as u32).trailing_zeros()) as u64,
            64 => (x as u32).rotate_right((y as u64).trailing_zeros()) as u64,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
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
        materialize_entry_test::<Fr, VirtualROTRIW>();
    }
}
