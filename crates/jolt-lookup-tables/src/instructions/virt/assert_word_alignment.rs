use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::AssertWordAlignment;
use tracer::instruction::{virtual_assert_word_alignment::VirtualAssertWordAlignment, RISCVCycle};

impl_lookup_table!(AssertWordAlignment, Some(WordAlignment));

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualAssertWordAlignment> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.register_state.rs1 & mask,
            (self.instruction.operands.imm as u64 & mask) as i128,
        )
    }

    fn to_lookup_operands(&self) -> (u64, u128) {
        let (address, offset) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (0, (address as i128 + offset) as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_lookup_output(&self) -> u64 {
        LookupQuery::<XLEN>::to_lookup_index(self)
            .is_multiple_of(4)
            .into()
    }
}
