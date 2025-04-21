use tracer::instruction::{sub::SUB, RISCVCycle};

use crate::jolt::lookup_table::{range_check::RangeCheckTable, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for SUB {
    fn lookup_table() -> Option<LookupTables<WORD_SIZE>> {
        Some(RangeCheckTable.into())
    }

    fn to_lookup_index(cycle: &RISCVCycle<Self>) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::lookup_query(cycle);
        let x = x as u128;
        let y = (1u128 << WORD_SIZE) - y as u128;
        (x + y) as u64
    }

    fn lookup_query(cycle: &RISCVCycle<Self>) -> (u64, u64) {
        (cycle.register_state.rs1, cycle.register_state.rs2)
    }

    fn lookup_entry(cycle: &RISCVCycle<Self>) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::lookup_query(cycle);
        match WORD_SIZE {
            #[cfg(test)]
            8 => (x as u8).overflowing_sub(y as u8).0.into(),
            32 => (x as u32).overflowing_sub(y as u32).0.into(),
            64 => x.overflowing_sub(y).0,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}
