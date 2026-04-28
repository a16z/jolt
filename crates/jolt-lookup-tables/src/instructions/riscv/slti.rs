use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::SltI;
use tracer::instruction::{slti::SLTI, RISCVCycle};

impl_lookup_table!(SltI, Some(SignedLessThan));

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<SLTI> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.register_state.rs1 & mask,
            (self.instruction.operands.imm & mask) as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let sx = ((x as i64) << (64 - XLEN)) >> (64 - XLEN);
        let sy = ((y as i64) << (64 - XLEN)) >> (64 - XLEN);
        (sx < sy) as u64
    }
}
