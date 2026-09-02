use crate::zkvm::instruction::{InstructionFlags, NUM_INSTRUCTION_FLAGS};
use tracer::instruction::{virtual_shift_right_bitmask_w::VirtualShiftRightBitmaskW, RISCVCycle};

use crate::zkvm::lookup_table::{shift_right_bitmask_w::ShiftRightBitmaskWTable, LookupTables};

use super::{CircuitFlags, Flags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const XLEN: usize> InstructionLookup<XLEN> for VirtualShiftRightBitmaskW {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        Some(ShiftRightBitmaskWTable.into())
    }
}

impl Flags for VirtualShiftRightBitmaskW {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::AddOperands] = true;
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
        flags
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualShiftRightBitmaskW> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (self.register_state.rs1, 0)
    }

    fn to_lookup_operands(&self) -> (u64, u128) {
        (0, self.register_state.rs1 as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        self.register_state.rs1 as u128
    }

    fn to_lookup_output(&self) -> u64 {
        let half = XLEN / 2;
        let shift = self.register_state.rs1 as usize % half;
        ((1u128 << half) - (1u128 << shift)) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zkvm::instruction::test::{
        lookup_output_matches_trace_test, materialize_entry_test,
    };
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, VirtualShiftRightBitmaskW>();
    }

    #[test]
    fn lookup_output_matches_trace() {
        lookup_output_matches_trace_test::<VirtualShiftRightBitmaskW>();
    }
}
