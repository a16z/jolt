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
        // Sign-extended XLEN-bit minimum (e.g. `i64::MIN` at XLEN=64,
        // `-128` at XLEN=8). Avoids `1 << (XLEN-1)` overflowing at XLEN=64.
        let signed_min = i64::MIN >> shift;
        if signed_dividend == signed_min && signed_divisor == -1 {
            1
        } else {
            let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
            signed_divisor as u64 & mask
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materialize_entry_test;

    #[test]
    fn materialize_entry_virtualchangedivisor() {
        materialize_entry_test!(VirtualChangeDivisor, tracer::instruction::virtual_change_divisor::VirtualChangeDivisor);
    }
}
