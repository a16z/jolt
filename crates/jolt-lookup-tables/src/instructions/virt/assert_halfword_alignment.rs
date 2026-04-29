use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::AssertHalfwordAlignment;
use jolt_trace::{JoltCycle, JoltInstruction};

impl_lookup_table!(AssertHalfwordAlignment, Some(HalfwordAlignment));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for AssertHalfwordAlignment<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.0.rs1_val().unwrap_or(0) & mask,
            self.0.instruction().imm(),
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
            .is_multiple_of(2)
            .into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{instruction_inputs_match_constraint_test, materialize_entry_test};

    #[test]
    fn materialize_entry_virtualasserthalfwordalignment() {
        materialize_entry_test!(AssertHalfwordAlignment, tracer::instruction::virtual_assert_halfword_alignment::VirtualAssertHalfwordAlignment);
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualasserthalfwordalignment() {
        instruction_inputs_match_constraint_test!(AssertHalfwordAlignment, tracer::instruction::virtual_assert_halfword_alignment::VirtualAssertHalfwordAlignment);
    }
}
