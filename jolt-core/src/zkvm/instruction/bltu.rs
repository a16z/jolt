use crate::zkvm::lookup_table::{unsigned_less_than::UnsignedLessThanTable, LookupTables};
use tracer::instruction::{bltu::BLTU, RISCVCycle};

use super::{CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for BLTU {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(UnsignedLessThanTable.into())
    }
}

impl InstructionFlags for BLTU {
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

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for RISCVCycle<BLTU> {
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
            8 => ((x as u8) < (y as u8)) as u64,
            32 => ((x as u32) < (y as u32)) as u64,
            64 => (x < y as u64) as u64,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
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
        materialize_entry_test::<Fr, BLTU>();
    }
}
