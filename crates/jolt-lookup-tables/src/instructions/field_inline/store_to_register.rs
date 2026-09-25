//! Range-check the x-register write as a non-interleaved advice operand.
//! The field-inline R1CS rows `ROW_STORE_TO_REGISTER` and
//! `ROW_STORE_TO_REGISTER_LOOKUP` pin the write and operand to `FieldRs1Value`,
//! so the field value must already fit in 64 bits.

use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_riscv::instructions::FieldStoreToRegister;
use jolt_riscv::JoltCycle;

impl_lookup_table!(FieldStoreToRegister, Some(RangeCheck));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for FieldStoreToRegister<C> {
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
    use tracer::instruction::field_inline::FIELD_STORE_TO_REGISTER;

    #[test]
    fn materialize_entry_field_store_to_register() {
        materialize_entry_test!(FieldStoreToRegister, FIELD_STORE_TO_REGISTER);
    }

    #[test]
    fn instruction_inputs_match_constraint_field_store_to_register() {
        instruction_inputs_match_constraint_test!(FieldStoreToRegister, FIELD_STORE_TO_REGISTER);
    }

    #[test]
    fn lookup_output_matches_trace_field_store_to_register() {
        lookup_output_matches_trace_test!(FieldStoreToRegister, FIELD_STORE_TO_REGISTER);
    }
}
