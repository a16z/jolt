use tracer::instruction::{ori::ORI, RISCVCycle};

use crate::zkvm::lookup_table::{or::OrTable, LookupTables};

use super::{
    CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, RightInputValue,
    NUM_CIRCUIT_FLAGS,
};

impl<const XLEN: usize> InstructionLookup<XLEN> for ORI {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        Some(OrTable.into())
    }
}

impl InstructionFlags for ORI {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::WriteLookupOutputToRD as usize] = true;
        flags[CircuitFlags::LeftOperandIsRs1Value as usize] = true;
        flags[CircuitFlags::RightOperandIsImm as usize] = true;
        flags[CircuitFlags::InlineSequenceInstruction as usize] =
            self.inline_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
            self.inline_sequence_remaining.unwrap_or(0) != 0;
        flags[CircuitFlags::IsCompressed as usize] = self.is_compressed;
        flags
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<ORI> {
    fn to_instruction_inputs(&self) -> (u64, RightInputValue) {
        match XLEN {
            #[cfg(test)]
            8 => (
                self.register_state.rs1 as u8 as u64,
                RightInputValue::Signed(self.instruction.operands.imm as i8 as i64),
            ),
            32 => (
                self.register_state.rs1 as u32 as u64,
                RightInputValue::Signed(self.instruction.operands.imm as i32 as i64),
            ),
            64 => (
                self.register_state.rs1,
                RightInputValue::Signed(self.instruction.operands.imm as i64),
            ),
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        match XLEN {
            #[cfg(test)]
            8 => (x as u8 | y.as_u8()).into(),
            32 => (x as u32 | y.as_u32()).into(),
            64 => x | y.as_u64(),
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
        materialize_entry_test::<Fr, ORI>();
    }
}
