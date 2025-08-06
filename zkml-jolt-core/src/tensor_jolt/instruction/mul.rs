use jolt_core::jolt::instruction::{InstructionLookup, LookupQuery};
use jolt_core::jolt::lookup_table::{LookupTables, range_check::RangeCheckTable};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct MUL<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for MUL<WORD_SIZE> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(RangeCheckTable.into())
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for MUL<WORD_SIZE> {
    fn to_lookup_operands(&self) -> (u64, u64) {
        let (x, y) = LookupQuery::<WORD_SIZE>::to_instruction_inputs(self);
        (0, x * y as u64)
    }

    fn to_lookup_index(&self) -> u64 {
        LookupQuery::<WORD_SIZE>::to_lookup_operands(self).1
    }

    fn to_instruction_inputs(&self) -> (u64, i64) {
        match WORD_SIZE {
            #[cfg(test)]
            8 => (self.0 as u8 as u64, self.1 as u8 as i64),
            32 => (self.0 as u32 as u64, self.1 as u32 as i64),
            64 => (self.0, self.1 as i64),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<WORD_SIZE>::to_instruction_inputs(self);
        match WORD_SIZE {
            #[cfg(test)]
            8 => (x as i8).wrapping_mul(y as i8) as u8 as u64,
            32 => (x as i32).wrapping_mul(y as i32) as u32 as u64,
            64 => (x as i64).wrapping_mul(y) as u64,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}
