use tracer::instruction::{virtual_assert_eq::VirtualAssertEQ, RISCVCycle};

use crate::jolt::lookup_table::{equal::EqualTable, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualAssertEQ {
    fn lookup_table() -> Option<LookupTables<WORD_SIZE>> {
        Some(EqualTable.into())
    }

    fn lookup_query(cycle: &RISCVCycle<Self>) -> (u64, u64) {
        (cycle.register_state.rs1, cycle.register_state.rs2)
    }

    fn lookup_entry(cycle: &RISCVCycle<Self>) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::lookup_query(cycle);
        (x == y).into()
    }
}
