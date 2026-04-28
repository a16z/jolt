use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::AssertValidUnsignedRemainder;
use tracer::instruction::{
    virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder, RISCVCycle,
};

impl_lookup_table!(AssertValidUnsignedRemainder, Some(ValidUnsignedRemainder));

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualAssertValidUnsignedRemainder> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.register_state.rs1 & mask,
            (self.register_state.rs2 & mask) as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (remainder, divisor) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (divisor == 0 || remainder < divisor as u64).into()
    }
}
