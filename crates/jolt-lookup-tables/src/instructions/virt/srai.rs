use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::VirtualSrai;
use jolt_trace::{JoltCycle, JoltInstruction};

impl_lookup_table!(VirtualSrai, Some(VirtualSRA));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualSrai<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (self.0.rs1_val().unwrap_or(0), self.0.imm())
    }

    fn to_lookup_output(&self) -> u64 {
        let (rs1, imm) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        let shift = (imm as u64).trailing_zeros();
        let signed = ((rs1 as i64) << (64 - XLEN)) >> (64 - XLEN);
        ((signed >> shift) as u64) & mask
    }
}
