use crate::zkvm::instruction::NUM_INSTRUCTION_FLAGS;
use tracer::instruction::{virtual_advice_sb::VirtualAdviceSB, RISCVCycle};

use crate::zkvm::lookup_table::{range_check::RangeCheckTable, LookupTables};

use super::{CircuitFlags, Flags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const XLEN: usize> InstructionLookup<XLEN> for VirtualAdviceSB {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        Some(RangeCheckTable.into())
    }
}

impl Flags for VirtualAdviceSB {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::Advice] = true;
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

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualAdviceSB> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (0, 0)
    }

    fn to_lookup_output(&self) -> u64 {
        0
    }
}

#[cfg(test)]
mod test {
    use crate::zkvm::instruction::test::materialize_entry_test;

    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, VirtualAdviceSB>();
    }
}
