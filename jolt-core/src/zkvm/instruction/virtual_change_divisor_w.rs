use tracer::instruction::{virtual_change_divisor_w::VirtualChangeDivisorW, RISCVCycle};

use super::{CircuitFlags, Flags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};
use crate::zkvm::{
    instruction::{InstructionFlags, NUM_INSTRUCTION_FLAGS},
    lookup_table::{virtual_change_divisor_w::VirtualChangeDivisorWTable, LookupTables},
};

impl<const XLEN: usize> InstructionLookup<XLEN> for VirtualChangeDivisorW {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        Some(VirtualChangeDivisorWTable.into())
    }
}

impl Flags for VirtualChangeDivisorW {
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
        flags[InstructionFlags::RightOperandIsRs2Value] = true;
        flags[InstructionFlags::IsRdNotZero] = self.operands.rd != 0;
        flags
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualChangeDivisorW> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        // Always treat as 32-bit values for W instructions
        (
            self.register_state.rs1 as u32 as u64,
            self.register_state.rs2 as i32 as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (remainder, divisor) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let remainder = remainder as i32;
        let divisor = divisor as i32;

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
    use ark_bn254::Fr;

    use super::*;
    use crate::zkvm::instruction::test::materialize_entry_test;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, VirtualChangeDivisorW>();
    }
}
