use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::VirtualChangeDivisorW;
use tracer::instruction::{
    virtual_change_divisor_w::VirtualChangeDivisorW as TracerVirtualChangeDivisorW, RISCVCycle,
};

impl_lookup_table!(VirtualChangeDivisorW, Some(VirtualChangeDivisorW));

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<TracerVirtualChangeDivisorW> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (self.register_state.rs1, self.register_state.rs2 as i128)
    }

    fn to_lookup_output(&self) -> u64 {
        let (dividend, divisor) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let dividend = dividend as i32;
        let divisor = divisor as i32;
        if dividend == i32::MIN && divisor == -1 {
            1
        } else {
            divisor as i64 as u64
        }
    }
}
