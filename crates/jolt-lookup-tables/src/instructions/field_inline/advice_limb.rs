//! Range-check the advised x-register limb through its non-interleaved lookup
//! operand. The field-inline constraints separately bind the limb and quotient
//! to the source field value; canonical readout requires a guest range check.

use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_riscv::instructions::FieldAdviceLimb;
use jolt_riscv::JoltCycle;

impl_lookup_table!(FieldAdviceLimb, Some(RangeCheck));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for FieldAdviceLimb<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (0, 0)
    }

    fn to_lookup_operands(&self) -> (u64, u128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            0,
            (self.0.rd_vals().map_or(0, |(_, post)| post) & mask) as u128,
        )
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_lookup_output(&self) -> u64 {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        self.0.rd_vals().map_or(0, |(_, post)| post) & mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        instruction_inputs_match_constraint_test, lookup_output_matches_trace_test,
        materialize_entry_test,
    };
    use tracer::instruction::field_inline::FIELD_ADVICE_LIMB;

    #[test]
    fn materialize_entry_field_advice_limb() {
        materialize_entry_test!(FieldAdviceLimb, FIELD_ADVICE_LIMB);
    }

    #[test]
    fn instruction_inputs_match_constraint_field_advice_limb() {
        instruction_inputs_match_constraint_test!(FieldAdviceLimb, FIELD_ADVICE_LIMB);
    }

    #[test]
    fn lookup_output_matches_trace_field_advice_limb() {
        lookup_output_matches_trace_test!(FieldAdviceLimb, FIELD_ADVICE_LIMB);
    }
}
