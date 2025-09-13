use tracer::instruction::{virtual_zero_extend_word::VirtualZeroExtendWord, RISCVCycle};

use crate::zkvm::lookup_table::{lower_half_word::LowerHalfWordTable, LookupTables};

use super::{
    CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, U64OrI64,
    NUM_CIRCUIT_FLAGS,
};

impl<const XLEN: usize> InstructionLookup<XLEN> for VirtualZeroExtendWord {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        Some(LowerHalfWordTable.into())
    }
}

impl InstructionFlags for VirtualZeroExtendWord {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::WriteLookupOutputToRD as usize] = true;
        flags[CircuitFlags::LeftOperandIsRs1Value as usize] = true;
        flags[CircuitFlags::AddOperands as usize] = true;
        flags[CircuitFlags::InlineSequenceInstruction as usize] =
            self.inline_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
            self.inline_sequence_remaining.unwrap_or(0) != 0;
        flags[CircuitFlags::IsCompressed as usize] = self.is_compressed;
        flags
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualZeroExtendWord> {
    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (0, u128::try_from(x as i128 + y.as_i128()).unwrap())
    }

    fn to_instruction_inputs(&self) -> (u64, U64OrI64) {
        (self.register_state.rs1, U64OrI64::Unsigned(0))
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_lookup_output(&self) -> u64 {
        let (_, y) = LookupQuery::<XLEN>::to_lookup_operands(self);
        // Zero-extend: keep only the lower half of the word
        let half_word_size = XLEN / 2;
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
        materialize_entry_test::<Fr, VirtualZeroExtendWord>();
    }
}
