use tracer::instruction::{virtual_sign_extend::VirtualSignExtend, RISCVCycle};

use crate::zkvm::lookup_table::{sign_extend_half_word::SignExtendHalfWordTable, LookupTables};

use super::{CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualSignExtend {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(SignExtendHalfWordTable.into())
    }
}

impl InstructionFlags for VirtualSignExtend {
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

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for RISCVCycle<VirtualSignExtend> {
    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = LookupQuery::<WORD_SIZE>::to_instruction_inputs(self);
        (0, x as u128 + y as u64 as u128)
    }

    fn to_instruction_inputs(&self) -> (u64, i128) {
        (self.register_state.rs1, 0)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<WORD_SIZE>::to_lookup_operands(self).1
    }

    fn to_lookup_output(&self) -> u64 {
        let (_, y) = LookupQuery::<WORD_SIZE>::to_lookup_operands(self);
        // Sign-extend: keep lower half and sign-extend upper half
        let half_word_size = WORD_SIZE / 2;
        let lower_half = y as u64 & ((1u64 << half_word_size) - 1);
        let sign_bit = (lower_half >> (half_word_size - 1)) & 1;

        if sign_bit == 1 {
            // Sign extend with 1s
            lower_half | (((1u64 << half_word_size) - 1) << half_word_size)
        } else {
            // Sign extend with 0s
            lower_half
        }
    }
}

#[cfg(test)]
mod test {
    use crate::zkvm::instruction::test::materialize_entry_test;

    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, VirtualSignExtend>();
    }
}
