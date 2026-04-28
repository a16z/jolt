use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::VirtualSignExtendWord;
use tracer::instruction::{
    virtual_sign_extend_word::VirtualSignExtendWord as TracerVirtualSignExtendWord, RISCVCycle,
};

impl_lookup_table!(VirtualSignExtendWord, Some(RangeCheck));

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<TracerVirtualSignExtendWord> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (self.register_state.rs1, 0)
    }

    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (0, x as u128 + y as u64 as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_lookup_output(&self) -> u64 {
        let (_, y) = LookupQuery::<XLEN>::to_lookup_operands(self);
        let half_word_size = XLEN / 2;
        let lower_half = y as u64 & ((1u64 << half_word_size) - 1);
        let sign_bit = (lower_half >> (half_word_size - 1)) & 1;
        if sign_bit == 1 {
            lower_half | (((1u64 << half_word_size) - 1) << half_word_size)
        } else {
            lower_half
        }
    }
}
