use tracer::instruction::{
    virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder, RISCVCycle,
};

use crate::jolt::lookup_table::{
    valid_unsigned_remainder::ValidUnsignedRemainderTable, LookupTables,
};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE>
    for RISCVCycle<VirtualAssertValidUnsignedRemainder>
{
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(ValidUnsignedRemainderTable.into())
    }

    fn to_lookup_query(&self) -> (u64, u64) {
        (self.register_state.rs1, self.register_state.rs2)
    }

    fn to_lookup_output(&self) -> u64 {
        let (remainder, divisor) = InstructionLookup::<WORD_SIZE>::to_lookup_query(self);
        (divisor == 0 || remainder < divisor).into()
    }
}
