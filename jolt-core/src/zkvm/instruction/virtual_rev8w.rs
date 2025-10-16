use crate::zkvm::instruction::{InstructionFlags, NUM_INSTRUCTION_FLAGS};
use tracer::instruction::{
    virtual_rev8w::{rev8w, VirtualRev8W},
    RISCVCycle,
};

use crate::zkvm::lookup_table::{virtual_rev8w::VirtualRev8WTable, LookupTables};

use super::{CircuitFlags, Flags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const XLEN: usize> InstructionLookup<XLEN> for VirtualRev8W {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        Some(VirtualRev8WTable.into())
    }
}

impl Flags for VirtualRev8W {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::WriteLookupOutputToRD as usize] = true;
        flags[CircuitFlags::AddOperands as usize] = true;
        flags[CircuitFlags::InlineSequenceInstruction as usize] =
            self.inline_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
            self.inline_sequence_remaining.unwrap_or(0) != 0;
        flags[CircuitFlags::IsFirstInSequence as usize] = self.is_first_in_sequence;
        flags[CircuitFlags::IsCompressed as usize] = self.is_compressed;
        flags
    }

    fn instruction_flags(&self) -> [bool; NUM_INSTRUCTION_FLAGS] {
        let mut flags = [false; NUM_INSTRUCTION_FLAGS];
        flags[InstructionFlags::LeftOperandIsRs1Value as usize] = true;
        flags
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualRev8W> {
    fn to_lookup_operands(&self) -> (u64, u128) {
        (0, self.register_state.rs1.into())
    }

    fn to_instruction_inputs(&self) -> (u64, i128) {
        (self.register_state.rs1, 0)
    }

    fn to_lookup_index(&self) -> u128 {
        self.register_state.rs1.into()
    }

    fn to_lookup_output(&self) -> u64 {
        rev8w(self.register_state.rs1)
    }
}

#[cfg(test)]
mod test {
    use super::VirtualRev8W;
    use crate::zkvm::instruction::test::materialize_entry_test;
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, VirtualRev8W>();
    }
}
