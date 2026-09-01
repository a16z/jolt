use crate::zkvm::instruction::{InstructionFlags, NUM_INSTRUCTION_FLAGS};
use tracer::instruction::{virtual_srliw::VirtualSRLIW, RISCVCycle};

use crate::zkvm::lookup_table::{virtual_srlw::VirtualSRLWTable, LookupTables};

use super::{CircuitFlags, Flags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const XLEN: usize> InstructionLookup<XLEN> for VirtualSRLIW {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        Some(VirtualSRLWTable.into())
    }
}

impl Flags for VirtualSRLIW {
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
        flags[InstructionFlags::RightOperandIsImm] = true;
        flags
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualSRLIW> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (
            self.register_state.rs1,
            self.instruction.operands.imm as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let half = XLEN / 2;
        let shift = self.instruction.operands.imm.trailing_zeros() as usize;
        let result = (self.register_state.rs1 as u128 & ((1u128 << half) - 1)) >> shift;
        let sign_bit = (result >> (half - 1)) & 1;
        (result + sign_bit * ((1u128 << XLEN) - (1u128 << half))) as u64
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
        materialize_entry_test::<Fr, VirtualSRLIW>();
    }

    #[test]
    fn lookup_output_matches_trace() {
        lookup_output_matches_trace_test::<VirtualSRLIW>();
    }
}
