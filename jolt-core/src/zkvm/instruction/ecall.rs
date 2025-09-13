use tracer::instruction::{ecall::ECALL, RISCVCycle};

use crate::zkvm::lookup_table::LookupTables;

use super::{
    CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, RightInputValue,
    NUM_CIRCUIT_FLAGS,
};

impl<const XLEN: usize> InstructionLookup<XLEN> for ECALL {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        None
    }
}

impl InstructionFlags for ECALL {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::IsCompressed as usize] = self.is_compressed;
        flags
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<ECALL> {
    fn to_instruction_inputs(&self) -> (u64, RightInputValue) {
        (0, RightInputValue::Unsigned(0))
    }

    fn to_lookup_output(&self) -> u64 {
        0
    }
}
