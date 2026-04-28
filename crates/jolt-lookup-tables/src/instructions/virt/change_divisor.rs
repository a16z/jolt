use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::VirtualChangeDivisor;
use jolt_trace::JoltCycle;

impl_lookup_table!(VirtualChangeDivisor, Some(VirtualChangeDivisor));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualChangeDivisor<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.0.rs1_val().unwrap_or(0) & mask,
            (self.0.rs2_val().unwrap_or(0) & mask) as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (dividend, divisor) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let shift = 64 - XLEN;
        let signed_dividend = ((dividend as i64) << shift) >> shift;
        let signed_divisor = ((divisor as i64) << shift) >> shift;
        let min_val = 1i64 << (XLEN - 1);
        if signed_dividend == -min_val && signed_divisor == -1 {
            1
        } else {
            let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
            signed_divisor as u64 & mask
        }
    }
}
