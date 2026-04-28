use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::Bge;
use tracer::instruction::{bge::BGE, RISCVCycle};

impl_lookup_table!(Bge, Some(SignedGreaterThanEqual));

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<BGE> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        let rs1 = self.register_state.rs1 & mask;
        let rs2 = self.register_state.rs2 & mask;
        let shift = 64 - XLEN as u32;
        let x = ((rs1 as i64) << shift >> shift) as u64;
        let y = ((rs2 as i64) << shift >> shift) as i128;
        (x, y)
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        ((x as i64) >= (y as i64)).into()
    }
}
