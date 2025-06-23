use tracer::instruction::{sub::SUB, RISCVCycle};

use crate::jolt::lookup_table::{range_check::RangeCheckTable, LookupTables};

use super::{CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for SUB {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(RangeCheckTable.into())
    }
}

impl InstructionFlags for SUB {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::SubtractOperands as usize] = true;
        flags[CircuitFlags::LeftOperandIsRs1Value as usize] = true;
        flags[CircuitFlags::RightOperandIsRs2Value as usize] = true;
        flags[CircuitFlags::WriteLookupOutputToRD as usize] = true;
        flags[CircuitFlags::InlineSequenceInstruction as usize] =
            self.virtual_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
            self.virtual_sequence_remaining.unwrap_or(0) != 0;
        flags
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for RISCVCycle<SUB> {
    fn to_lookup_operands(&self) -> (u64, u64) {
        let (x, y) = LookupQuery::<WORD_SIZE>::to_instruction_inputs(self);
        let x = x as u128;
        let y = (1u128 << WORD_SIZE) - y as u128;
        (0, (x + y) as u64)
    }

    fn to_lookup_index(&self) -> u64 {
        LookupQuery::<WORD_SIZE>::to_lookup_operands(self).1
    }

    fn to_instruction_inputs(&self) -> (u64, i64) {
        match WORD_SIZE {
            #[cfg(test)]
            8 => (
                self.register_state.rs1 as u8 as u64,
                self.register_state.rs2 as u8 as i64,
            ),
            32 => (
                self.register_state.rs1 as u32 as u64,
                self.register_state.rs2 as u32 as i64,
            ),
            64 => (self.register_state.rs1, self.register_state.rs2 as i64),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<WORD_SIZE>::to_instruction_inputs(self);
        match WORD_SIZE {
            #[cfg(test)]
            8 => (x as u8).overflowing_sub(y as u8).0.into(),
            32 => (x as u32).overflowing_sub(y as u32).0.into(),
            64 => x.overflowing_sub(y as u64).0,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::jolt::instruction::test::materialize_entry_test;

    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, SUB>();
    }
}
