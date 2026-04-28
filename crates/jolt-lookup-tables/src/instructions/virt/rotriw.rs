use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::VirtualRotriw;
use tracer::instruction::{virtual_rotriw::VirtualROTRIW, RISCVCycle};

impl_lookup_table!(VirtualRotriw, Some(VirtualROTRW));

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualROTRIW> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (
            self.register_state.rs1,
            self.instruction.operands.imm as i128,
        )
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
