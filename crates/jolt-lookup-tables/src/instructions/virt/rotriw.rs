use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::VirtualRotriw;
use jolt_trace::{JoltCycle, JoltInstruction};

impl_lookup_table!(VirtualRotriw, Some(VirtualROTRW));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualRotriw<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (self.0.rs1_val().unwrap_or(0), self.0.imm())
    }

    fn to_lookup_output(&self) -> u64 {
        let (rs1, imm) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let half = XLEN / 2;
        let mask = (1u128 << half).wrapping_sub(1) as u64;
        let r = (imm as u64).trailing_zeros() as usize % half;
        let v = (rs1 & mask) as u128;
        if r == 0 {
            rs1 & mask
        } else {
            (((v >> r) | (v << (half - r))) as u64) & mask
        }
    }
}
