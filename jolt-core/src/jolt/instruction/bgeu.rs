use tracer::instruction::{bgeu::BGEU, RISCVCycle};

use crate::jolt::lookup_table::{
    unsigned_greater_than_equal::UnsignedGreaterThanEqualTable, LookupTables,
};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for BGEU {
    fn lookup_table() -> Option<LookupTables<WORD_SIZE>> {
        Some(UnsignedGreaterThanEqualTable.into())
    }

    fn lookup_query(cycle: &RISCVCycle<Self>) -> (u64, u64) {
        (cycle.register_state.rs1, cycle.register_state.rs2)
    }

    fn lookup_entry(cycle: &RISCVCycle<Self>) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::lookup_query(cycle);
        match WORD_SIZE {
            #[cfg(test)]
            8 => (x >= y).into(),
            32 => (x >= y).into(),
            64 => (x >= y).into(),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}
