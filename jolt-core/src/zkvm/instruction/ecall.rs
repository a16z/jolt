use tracer::instruction::{ecall::ECALL, RISCVCycle};

use crate::zkvm::lookup_table::LookupTables;

use super::{InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for ECALL {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        None
    }
}

impl InstructionFlags for ECALL {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        [false; NUM_CIRCUIT_FLAGS]
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for RISCVCycle<ECALL> {
    fn to_instruction_inputs(&self) -> (u64, i64) {
        (0, 0)
    }

    fn to_lookup_output(&self) -> u64 {
        0
    }
}
