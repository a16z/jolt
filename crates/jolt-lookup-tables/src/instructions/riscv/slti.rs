use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::SltI;
use jolt_trace::{JoltCycle, JoltInstruction};

impl_lookup_table!(SltI, Some(SignedLessThan));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for SltI<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.0.rs1_val().unwrap_or(0) & mask,
            self.0.instruction().imm() & mask as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let sx = ((x as i64) << (64 - XLEN)) >> (64 - XLEN);
        let sy = ((y as i64) << (64 - XLEN)) >> (64 - XLEN);
        (sx < sy) as u64
    }
}
