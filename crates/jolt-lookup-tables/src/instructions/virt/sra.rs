use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::VirtualSra;
use tracer::instruction::{virtual_sra::VirtualSRA, RISCVCycle};

impl_lookup_table!(VirtualSra, Some(VirtualSRA));

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualSRA> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (self.register_state.rs1, self.register_state.rs2 as i128)
    }

    fn to_lookup_output(&self) -> u64 {
        let (rs1, rs2) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        let shift = (rs2 as u64).trailing_zeros() as u32;
        let signed = ((rs1 as i64) << (64 - XLEN)) >> (64 - XLEN);
        ((signed >> shift) as u64) & mask
    }
}
