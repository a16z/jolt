use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::Pow2;
use tracer::instruction::{virtual_pow2::VirtualPow2, RISCVCycle};

impl_lookup_table!(Pow2, Some(Pow2));

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualPow2> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (self.register_state.rs1 & mask, 0)
    }

    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (0, x as u128 + y as u64 as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_lookup_output(&self) -> u64 {
        let y = LookupQuery::<XLEN>::to_lookup_index(self);
        1u64 << (y & ((XLEN as u128) - 1))
    }
}
