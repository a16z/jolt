use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::OrI;
use tracer::instruction::{ori::ORI, RISCVCycle};

impl_lookup_table!(OrI, Some(Or));

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<ORI> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.register_state.rs1 & mask,
            (self.instruction.operands.imm & mask) as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        x | y as u64
    }
}
