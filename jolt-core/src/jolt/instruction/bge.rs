use tracer::instruction::{bge::BGE, RISCVCycle};

use crate::jolt::lookup_table::{
    signed_greater_than_equal::SignedGreaterThanEqualTable, LookupTables,
};

use super::{CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for BGE {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(SignedGreaterThanEqualTable.into())
    }
}

impl InstructionFlags for BGE {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::LeftOperandIsRs1Value as usize] = true;
        flags[CircuitFlags::RightOperandIsRs2Value as usize] = true;
        flags[CircuitFlags::Branch as usize] = true;
        flags[CircuitFlags::InlineSequenceInstruction as usize] =
            self.virtual_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
            self.virtual_sequence_remaining.unwrap_or(0) != 0;
        flags
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for RISCVCycle<BGE> {
    fn to_instruction_inputs(&self) -> (u64, i64) {
        (self.register_state.rs1, self.register_state.rs2 as i64)
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<WORD_SIZE>::to_instruction_inputs(self);
        match WORD_SIZE {
            #[cfg(test)]
            8 => ((x as i8) >= (y as i8)) as u64,
            32 => ((x as i32) >= (y as i32)) as u64,
            64 => ((x as i64) >= y) as u64,
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
        materialize_entry_test::<Fr, BGE>();
    }
}
