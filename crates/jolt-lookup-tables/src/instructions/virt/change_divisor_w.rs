use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::VirtualChangeDivisorW;
use jolt_trace::JoltCycle;

impl_lookup_table!(VirtualChangeDivisorW, Some(VirtualChangeDivisorW));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualChangeDivisorW<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (
            self.0.rs1_val().unwrap_or(0),
            self.0.rs2_val().unwrap_or(0) as i128,
        )
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
