use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::Blt;
use jolt_trace::JoltCycle;

impl_lookup_table!(Blt, Some(SignedLessThan));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for Blt<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        let x = self.0.rs1_val().unwrap_or(0) & mask;
        let y = self.0.rs2_val().unwrap_or(0) & mask;
        // Sign-extend both operands for signed comparison.
        let shift = 64 - XLEN as u32;
        let x_signed = ((x as i64) << shift) >> shift;
        let y_signed = ((y as i64) << shift) >> shift;
        (x_signed as u64, y_signed as i128)
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        ((x as i64) < (y as i64)) as u64
    }
}
