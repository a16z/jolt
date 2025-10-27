use tracer::instruction::{sd::SD, RISCVCycle};

use super::{CircuitFlags, Flags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};
use crate::zkvm::{instruction::NUM_INSTRUCTION_FLAGS, lookup_table::LookupTables};

impl<const XLEN: usize> InstructionLookup<XLEN> for SD {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        None
    }
}

impl Flags for SD {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::Store] = true;
        flags[CircuitFlags::VirtualInstruction] = self.virtual_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC] =
            self.virtual_sequence_remaining.unwrap_or(0) != 0;
        flags[CircuitFlags::IsFirstInSequence] = self.is_first_in_sequence;
        flags[CircuitFlags::IsCompressed] = self.is_compressed;
        flags
    }

    fn instruction_flags(&self) -> [bool; NUM_INSTRUCTION_FLAGS] {
        [false; NUM_INSTRUCTION_FLAGS]
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<SD> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (0, 0)
    }

    fn to_lookup_output(&self) -> u64 {
        0
    }
}
