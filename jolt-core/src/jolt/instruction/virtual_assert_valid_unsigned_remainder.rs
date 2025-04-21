use tracer::instruction::{
    virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder, RISCVCycle,
};

use crate::jolt::lookup_table::{
    valid_unsigned_remainder::ValidUnsignedRemainderTable, LookupTables,
};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualAssertValidUnsignedRemainder {
    fn lookup_table() -> Option<LookupTables<WORD_SIZE>> {
        Some(ValidUnsignedRemainderTable.into())
    }

    fn lookup_query(cycle: &RISCVCycle<Self>) -> (u64, u64) {
        (cycle.register_state.rs1, cycle.register_state.rs2)
    }

    fn lookup_entry(cycle: &RISCVCycle<Self>) -> u64 {
        let (remainder, divisor) = InstructionLookup::<WORD_SIZE>::lookup_query(cycle);
        (divisor == 0 || remainder < divisor).into()
    }
}
