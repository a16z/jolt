use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::Pow2IW;
use tracer::instruction::{virtual_pow2i_w::VirtualPow2IW, RISCVCycle};

impl_lookup_table!(Pow2IW, Some(Pow2W));

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualPow2IW> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (0, self.instruction.operands.imm as i128)
    }

    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (0, x as u128 + y as u64 as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_lookup_output(&self) -> u64 {
        let y = LookupQuery::<XLEN>::to_lookup_index(self);
        1u64 << (y & ((XLEN as u128 / 2) - 1))
    }
}
