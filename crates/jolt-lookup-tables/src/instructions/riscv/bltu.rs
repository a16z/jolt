use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::BltU;
use tracer::instruction::{bltu::BLTU, RISCVCycle};

impl_lookup_table!(BltU, Some(UnsignedLessThan));

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<BLTU> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.register_state.rs1 & mask,
            (self.register_state.rs2 & mask) as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (x < y as u64) as u64
    }
}
