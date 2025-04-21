use tracer::instruction::{lw::LW, RISCVCycle};

use crate::jolt::lookup_table::LookupTables;

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for RISCVCycle<LW> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        None
    }

    fn to_lookup_query(&self) -> (u64, u64) {
        (0, 0)
    }

    fn to_lookup_output(&self) -> u64 {
        0
    }
}
