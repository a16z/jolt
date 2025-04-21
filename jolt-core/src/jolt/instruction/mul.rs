use tracer::instruction::{mul::MUL, RISCVCycle};

use crate::jolt::lookup_table::{range_check::RangeCheckTable, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for MUL {
    fn lookup_table() -> Option<LookupTables<WORD_SIZE>> {
        Some(RangeCheckTable.into())
    }

    fn to_lookup_index(cycle: &RISCVCycle<Self>) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::lookup_query(cycle);
        match WORD_SIZE {
            #[cfg(test)]
            8 => x * y,
            32 => x * y,
            // 64 => x * y,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn lookup_query(cycle: &RISCVCycle<Self>) -> (u64, u64) {
        (cycle.register_state.rs1, cycle.register_state.rs2)
    }

    fn lookup_entry(cycle: &RISCVCycle<Self>) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::lookup_query(cycle);
        match WORD_SIZE {
            #[cfg(test)]
            8 => (x as i8).wrapping_mul(y as i8) as u8 as u64,
            32 => (x as i32).wrapping_mul(y as i32) as u32 as u64,
            64 => (x as i64).wrapping_mul(y as i64) as u64,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}
