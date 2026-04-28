use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::MulI;
use tracer::instruction::{virtual_muli::VirtualMULI, RISCVCycle};

impl_lookup_table!(MulI, Some(RangeCheck));

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualMULI> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.register_state.rs1 & mask,
            (self.instruction.operands.imm as u64 & mask) as i128,
        )
    }

    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (0, x as u128 * y as u64 as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let shift = 64 - XLEN;
        let signed_x = ((x as i64) << shift) >> shift;
        let signed_y = ((y as i64) << shift) >> shift;
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        signed_x.wrapping_mul(signed_y) as u64 & mask
    }
}
