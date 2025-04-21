use tracer::instruction::{jalr::JALR, RISCVCycle};

use crate::jolt::lookup_table::{range_check::RangeCheckTable, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for JALR {
    fn lookup_table() -> Option<LookupTables<WORD_SIZE>> {
        Some(RangeCheckTable.into())
    }

    fn to_lookup_index(cycle: &RISCVCycle<Self>) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::lookup_query(cycle);
        match WORD_SIZE {
            #[cfg(test)]
            8 => x + y,
            32 => x + y,
            // 64 => x.wrapping_add(y),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn lookup_query(cycle: &RISCVCycle<Self>) -> (u64, u64) {
        (
            cycle.register_state.rs1,
            cycle.instruction.operands.imm as u64,
        )
    }

    fn lookup_entry(cycle: &RISCVCycle<Self>) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::lookup_query(cycle);
        match WORD_SIZE {
            #[cfg(test)]
            8 => (x as u8).wrapping_add(y as u8) as u64,
            32 => (x as u32).wrapping_add(y as u32) as u64,
            64 => x.wrapping_add(y),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}
