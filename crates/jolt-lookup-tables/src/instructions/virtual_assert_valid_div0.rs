use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::AssertValidDiv0;
use tracer::instruction::{virtual_assert_valid_div0::VirtualAssertValidDiv0, RISCVCycle};

impl_lookup_table!(AssertValidDiv0, Some(ValidDiv0));

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualAssertValidDiv0> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.register_state.rs1 & mask,
            (self.register_state.rs2 & mask) as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (divisor, quotient) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let max_val = (1u128 << XLEN).wrapping_sub(1) as u64;
        if divisor == 0 {
            (quotient as u64 == max_val).into()
        } else {
            1
        }
    }
}
