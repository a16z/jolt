use tracer::instruction::{virtual_change_divisor_w::VirtualChangeDivisorW, RISCVCycle};

use crate::zkvm::lookup_table::virtual_change_divisor_w::VirtualChangeDivisorWTable;
use crate::zkvm::lookup_table::LookupTables;

use super::{
    CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, RightInputValue,
    NUM_CIRCUIT_FLAGS,
};

impl<const XLEN: usize> InstructionLookup<XLEN> for VirtualChangeDivisorW {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        Some(VirtualChangeDivisorWTable.into())
    }
}

impl InstructionFlags for VirtualChangeDivisorW {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::LeftOperandIsRs1Value as usize] = true;
        flags[CircuitFlags::RightOperandIsRs2Value as usize] = true;
        flags[CircuitFlags::WriteLookupOutputToRD as usize] = true;
        flags[CircuitFlags::InlineSequenceInstruction as usize] =
            self.inline_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
            self.inline_sequence_remaining.unwrap_or(0) != 0;
        flags[CircuitFlags::IsCompressed as usize] = self.is_compressed;
        flags
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualChangeDivisorW> {
    fn to_instruction_inputs(&self) -> (u64, RightInputValue) {
        // Always treat as 32-bit values for W instructions
        (
            self.register_state.rs1 as u32 as u64,
            RightInputValue::Signed(self.register_state.rs2 as i32 as i64),
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (remainder, divisor) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let remainder = remainder as i32;
        let divisor = divisor.as_i32();

        if remainder == i32::MIN && divisor == -1 {
            1
        } else {
            // Sign-extend the 32-bit result to 64 bits
            divisor as i64 as u64
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
        materialize_entry_test::<Fr, VirtualChangeDivisorW>();
    }
}
