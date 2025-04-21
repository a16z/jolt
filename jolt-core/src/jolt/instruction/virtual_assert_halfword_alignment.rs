use tracer::instruction::{
    virtual_assert_halfword_alignment::VirtualAssertHalfwordAlignment, RISCVCycle,
};

use crate::jolt::lookup_table::{halfword_alignment::HalfwordAlignmentTable, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE>
    for RISCVCycle<VirtualAssertHalfwordAlignment>
{
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(HalfwordAlignmentTable.into())
    }

    fn to_lookup_query(&self) -> (u64, u64) {
        (
            self.register_state.rs1,
            self.instruction.operands.imm as u64,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (addr, offset) = InstructionLookup::<WORD_SIZE>::to_lookup_query(self);
        ((addr as i64 + offset as i64) % 2 == 0).into()
    }
}
