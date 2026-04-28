use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::AssertMulUNoOverflow;
use tracer::instruction::{
    virtual_assert_mulu_no_overflow::VirtualAssertMulUNoOverflow, RISCVCycle,
};

impl_lookup_table!(AssertMulUNoOverflow, Some(MulUNoOverflow));

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualAssertMulUNoOverflow> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.register_state.rs1 & mask,
            (self.register_state.rs2 & mask) as i128,
        )
    }

    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (0, x as u128 * y as u64 as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_lookup_output(&self) -> u64 {
        let (rs1, rs2) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let result = (rs1 as u128) * (rs2 as u64 as u128);
        let max_val = (1u128 << XLEN).wrapping_sub(1);
        (result <= max_val) as u64
    }
}
