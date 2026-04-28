use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::AndI;
use tracer::instruction::{andi::ANDI, RISCVCycle};

impl_lookup_table!(AndI, Some(And));

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<ANDI> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.register_state.rs1 & mask,
            (self.instruction.operands.imm as u64 & mask) as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        x & y as u64 & mask
    }
}
