use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::VirtualAdvice;
use tracer::instruction::{virtual_advice::VirtualAdvice as TracerVirtualAdvice, RISCVCycle};

impl_lookup_table!(VirtualAdvice, Some(RangeCheck));

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<TracerVirtualAdvice> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (0, 0)
    }

    fn to_lookup_operands(&self) -> (u64, u128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (0, (self.instruction.advice & mask) as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_lookup_output(&self) -> u64 {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        self.instruction.advice & mask
    }
}
