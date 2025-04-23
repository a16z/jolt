use tracer::instruction::{lw::LW, RISCVCycle};

use crate::jolt::lookup_table::LookupTables;

use super::{CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for LW {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        None
    }
}

impl InstructionFlags for LW {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::RightOperandIsImm as usize] = true;
        flags[CircuitFlags::Load as usize] = true;
        flags[CircuitFlags::Virtual as usize] = self.virtual_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdatePC as usize] =
            self.virtual_sequence_remaining.unwrap_or(0) != 0;
        flags
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for RISCVCycle<LW> {
    fn to_lookup_query(&self) -> (u64, u64) {
        (0, 0)
    }

    #[cfg(test)]
    fn to_lookup_output(&self) -> u64 {
        0
    }
}
