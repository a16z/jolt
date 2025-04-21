use tracer::instruction::{virtual_advice::VirtualAdvice, RISCVCycle};

use crate::jolt::lookup_table::{range_check::RangeCheckTable, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for RISCVCycle<VirtualAdvice> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(RangeCheckTable.into())
    }

    fn to_lookup_index(&self) -> u64 {
        self.instruction.advice
    }

    fn to_lookup_query(&self) -> (u64, u64) {
        (self.instruction.advice, 0)
    }

    fn to_lookup_output(&self) -> u64 {
        match WORD_SIZE {
            #[cfg(test)]
            8 => (self.instruction.advice as u8).into(),
            32 => (self.instruction.advice as u32).into(),
            64 => self.instruction.advice as u64,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}
