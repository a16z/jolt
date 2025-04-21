use tracer::instruction::{virtual_assert_lte::VirtualAssertLTE, RISCVCycle};

use crate::jolt::lookup_table::{
    unsigned_less_than_equal::UnsignedLessThanEqualTable, LookupTables,
};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for RISCVCycle<VirtualAssertLTE> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(UnsignedLessThanEqualTable.into())
    }

    fn to_lookup_query(&self) -> (u64, u64) {
        (self.register_state.rs1, self.register_state.rs2)
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::to_lookup_query(self);
        (x <= y).into()
    }
}
