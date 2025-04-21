use tracer::instruction::{bge::BGE, RISCVCycle};

use crate::jolt::lookup_table::{
    signed_greater_than_equal::SignedGreaterThanEqualTable, LookupTables,
};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for BGE {
    fn lookup_table() -> Option<LookupTables<WORD_SIZE>> {
        Some(SignedGreaterThanEqualTable.into())
    }

    fn lookup_query(cycle: &RISCVCycle<Self>) -> (u64, u64) {
        (cycle.register_state.rs1, cycle.register_state.rs2)
    }

    fn lookup_entry(cycle: &RISCVCycle<Self>) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::lookup_query(cycle);
        match WORD_SIZE {
            #[cfg(test)]
            8 => ((x as i8) >= (y as i8)) as u64,
            32 => ((x as i32) >= (y as i32)) as u64,
            64 => ((x as i64) >= (y as i64)) as u64,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}
