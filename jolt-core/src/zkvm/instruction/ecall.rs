use crate::zkvm::instruction::NUM_INSTRUCTION_FLAGS;
use tracer::instruction::{ecall::ECALL, RISCVCycle};

use crate::zkvm::lookup_table::LookupTables;

use super::{CircuitFlags, Flags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const XLEN: usize> InstructionLookup<XLEN> for ECALL {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        None
    }
}

impl Flags for ECALL {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::IsFirstInSequence] = self.is_first_in_sequence;
        flags[CircuitFlags::IsCompressed] = self.is_compressed;
        flags[CircuitFlags::IsRdZero] = self.operands.rd == 0;
        flags
    }

    fn instruction_flags(&self) -> [bool; NUM_INSTRUCTION_FLAGS] {
        [false; NUM_INSTRUCTION_FLAGS]
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<ECALL> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (0, 0)
    }

    fn to_lookup_output(&self) -> u64 {
        0
    }
}
