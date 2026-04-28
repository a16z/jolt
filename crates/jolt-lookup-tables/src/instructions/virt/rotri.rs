use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::VirtualRotri;
use jolt_trace::JoltCycle;

impl_lookup_table!(VirtualRotri, Some(VirtualROTR));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualRotri<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (self.0.rs1_val().unwrap_or(0), self.0.imm())
    }

    fn to_lookup_output(&self) -> u64 {
        let (rs1, imm) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        let r = (imm as u64).trailing_zeros() as usize % XLEN;
        let v = (rs1 & mask) as u128;
        if r == 0 {
            rs1 & mask
        } else {
            (((v >> r) | (v << (XLEN - r))) as u64) & mask
        }
    }
}
