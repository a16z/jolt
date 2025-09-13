use tracer::instruction::{virtual_rotriw::VirtualROTRIW, RISCVCycle};

use crate::zkvm::lookup_table::{virtual_rotrw::VirtualRotrWTable, LookupTables};

use super::{
    CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, U64OrI64,
    NUM_CIRCUIT_FLAGS,
};

impl<const XLEN: usize> InstructionLookup<XLEN> for VirtualROTRIW {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        Some(VirtualRotrWTable.into())
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

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualROTRIW> {
    fn to_instruction_inputs(&self) -> (u64, U64OrI64) {
        (
            self.register_state.rs1,
            U64OrI64::Unsigned(self.instruction.operands.imm),
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        match XLEN {
            #[cfg(test)]
            8 => {
                let (x, y) = (x as u8, (y.as_u8()).trailing_zeros());
                (((x & 0x0F) >> (y % 4)) | (((x & 0x0F) << (4 - (y % 4))) & 0x0F)) as u64
            }
            32 => (x as u16).rotate_right((y.as_u32()).trailing_zeros().min(16)) as u64,
            64 => (x as u32).rotate_right((y.as_u64()).trailing_zeros().min(32)) as u64,
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
        materialize_entry_test::<Fr, VirtualROTRIW>();
    }
}
