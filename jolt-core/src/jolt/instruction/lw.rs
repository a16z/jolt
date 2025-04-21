use tracer::instruction::{lw::LW, RISCVCycle};

use crate::jolt::lookup_table::LookupTables;

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for LW {
    fn lookup_table() -> Option<LookupTables<WORD_SIZE>> {
        None
    }

    fn lookup_query(_: &RISCVCycle<Self>) -> (u64, u64) {
        (0, 0)
    }

    fn lookup_entry(_: &RISCVCycle<Self>) -> u64 {
        0
    }
}
