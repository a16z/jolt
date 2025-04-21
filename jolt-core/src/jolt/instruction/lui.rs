use tracer::instruction::{lui::LUI, RISCVCycle};

use crate::jolt::lookup_table::{range_check::RangeCheckTable, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for LUI {
    fn lookup_table() -> Option<LookupTables<WORD_SIZE>> {
        Some(RangeCheckTable.into())
    }

    fn to_lookup_index(cycle: &RISCVCycle<Self>) -> u64 {
        cycle.instruction.operands.imm as u64
    }

    fn lookup_query(cycle: &RISCVCycle<Self>) -> (u64, u64) {
        (cycle.instruction.operands.imm as u64, 0)
    }

    fn lookup_entry(cycle: &RISCVCycle<Self>) -> u64 {
        match WORD_SIZE {
            #[cfg(test)]
            8 => (cycle.instruction.operands.imm as u8).into(),
            32 => (cycle.instruction.operands.imm as u32).into(),
            64 => cycle.instruction.operands.imm as u64,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}
