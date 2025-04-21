use tracer::instruction::{virtual_advice::VirtualAdvice, RISCVCycle};

use crate::jolt::lookup_table::{range_check::RangeCheckTable, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualAdvice {
    fn lookup_table() -> Option<LookupTables<WORD_SIZE>> {
        Some(RangeCheckTable.into())
    }

    fn to_lookup_index(cycle: &RISCVCycle<Self>) -> u64 {
        cycle.instruction.advice
    }

    fn lookup_query(cycle: &RISCVCycle<Self>) -> (u64, u64) {
        (cycle.instruction.advice, 0)
    }

    fn lookup_entry(cycle: &RISCVCycle<Self>) -> u64 {
        match WORD_SIZE {
            #[cfg(test)]
            8 => (cycle.instruction.advice as u8).into(),
            32 => (cycle.instruction.advice as u32).into(),
            64 => cycle.instruction.advice as u64,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}
