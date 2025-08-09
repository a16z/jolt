use jolt_core::jolt::instruction::{InstructionLookup, LookupQuery};
use jolt_core::jolt::lookup_table::{LookupTables, range_check::RangeCheckTable};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct ConstInstruction<const WORD_SIZE: usize>(pub u64);

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for ConstInstruction<WORD_SIZE> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(RangeCheckTable.into())
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for ConstInstruction<WORD_SIZE> {
    fn to_instruction_inputs(&self) -> (u64, i64) {
        (0, 0)
    }

    fn to_lookup_operands(&self) -> (u64, u64) {
        match WORD_SIZE {
            #[cfg(test)]
            8 => (0, self.0 as u8 as u64),
            32 => (0, self.0 as u32 as u64),
            64 => (0, self.0),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn to_lookup_index(&self) -> u64 {
        LookupQuery::<WORD_SIZE>::to_lookup_operands(self).1
    }

    fn to_lookup_output(&self) -> u64 {
        match WORD_SIZE {
            #[cfg(test)]
            8 => (self.0 as u8).into(),
            32 => (self.0 as u32).into(),
            64 => self.0,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}
