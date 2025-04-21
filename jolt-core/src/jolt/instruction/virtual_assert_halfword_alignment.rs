use tracer::instruction::{
    virtual_assert_halfword_alignment::VirtualAssertHalfwordAlignment, RISCVCycle,
};

use crate::jolt::lookup_table::{halfword_alignment::HalfwordAlignmentTable, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualAssertHalfwordAlignment {
    fn lookup_table() -> Option<LookupTables<WORD_SIZE>> {
        Some(HalfwordAlignmentTable.into())
    }

    fn lookup_query(cycle: &RISCVCycle<Self>) -> (u64, u64) {
        (
            cycle.register_state.rs1,
            cycle.instruction.operands.imm as u64,
        )
    }

    fn lookup_entry(cycle: &RISCVCycle<Self>) -> u64 {
        let (addr, offset) = InstructionLookup::<WORD_SIZE>::lookup_query(cycle);
        ((addr as i64 + offset as i64) % 2 == 0).into()
    }
}
