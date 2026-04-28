use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::VirtualZeroExtendWord;
use tracer::instruction::{
    virtual_zero_extend_word::VirtualZeroExtendWord as TracerVirtualZeroExtendWord, RISCVCycle,
};

impl_lookup_table!(VirtualZeroExtendWord, Some(RangeCheck));

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<TracerVirtualZeroExtendWord> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (self.register_state.rs1, 0)
    }

    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (0, u128::try_from(x as i128 + y).unwrap())
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_lookup_output(&self) -> u64 {
        let (_, y) = LookupQuery::<XLEN>::to_lookup_operands(self);
        let half_word_size = XLEN / 2;
        y as u64 & ((1u64 << half_word_size) - 1)
    }
}
