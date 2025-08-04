use tracer::instruction::{virtual_extend::VirtualExtend, RISCVCycle};

use crate::zkvm::lookup_table::{lower_half_word::LowerHalfWordTable, LookupTables};

use super::{CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualExtend {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(LowerHalfWordTable.into())
    }
}

impl InstructionFlags for VirtualExtend {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::WriteLookupOutputToRD as usize] = true;
        flags[CircuitFlags::LeftOperandIsRs1Value as usize] = true;
        flags[CircuitFlags::AddOperands as usize] = true;
        flags[CircuitFlags::InlineSequenceInstruction as usize] =
            self.virtual_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
            self.virtual_sequence_remaining.unwrap_or(0) != 0;
        flags[CircuitFlags::IsCompressed as usize] = self.is_compressed;
        flags
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for RISCVCycle<VirtualExtend> {
    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = LookupQuery::<WORD_SIZE>::to_instruction_inputs(self);
        (0, u128::try_from(x as i128 + y as i128).unwrap())
    }

    fn to_instruction_inputs(&self) -> (u64, i64) {
        (self.register_state.rs1, 0)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<WORD_SIZE>::to_lookup_operands(self).1
    }

    fn to_lookup_output(&self) -> u64 {
        let (_, y) = LookupQuery::<WORD_SIZE>::to_lookup_operands(self);
        // Zero-extend: keep only the lower half of the word
        let half_word_size = WORD_SIZE / 2;
        y as u64 & ((1u64 << half_word_size) - 1)
    }
}

#[cfg(test)]
mod test {
    use crate::zkvm::instruction::test::materialize_entry_test;

    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, VirtualExtend>();
    }
}
