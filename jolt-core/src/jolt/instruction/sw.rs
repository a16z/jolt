use tracer::instruction::{sw::SW, RISCVCycle};

use crate::jolt::lookup_table::LookupTables;

use super::{CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for SW {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        None
    }
}

impl InstructionFlags for SW {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::Store as usize] = true;
        flags[CircuitFlags::InlineSequenceInstruction as usize] =
            self.virtual_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
            self.virtual_sequence_remaining.unwrap_or(0) != 0;
        flags
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for RISCVCycle<SW> {
    fn to_instruction_inputs(&self) -> (u64, i64) {
        (0, 0)
    }

    fn to_lookup_output(&self) -> u64 {
        0
    }
}
