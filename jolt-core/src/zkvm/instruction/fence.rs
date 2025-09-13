use tracer::instruction::{fence::FENCE, RISCVCycle};

use crate::zkvm::lookup_table::LookupTables;

use super::{
    CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, U64OrI64, NUM_CIRCUIT_FLAGS,
};

impl<const XLEN: usize> InstructionLookup<XLEN> for FENCE {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        None
    }
}

impl InstructionFlags for FENCE {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::IsCompressed as usize] = self.is_compressed;
        flags
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<FENCE> {
    fn to_instruction_inputs(&self) -> (u64, U64OrI64) {
        (0, U64OrI64::Unsigned(0))
    }

    fn to_lookup_output(&self) -> u64 {
        0
    }
}
