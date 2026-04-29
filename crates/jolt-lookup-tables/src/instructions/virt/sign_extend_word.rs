use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::VirtualSignExtendWord;
use jolt_trace::JoltCycle;

impl_lookup_table!(VirtualSignExtendWord, Some(SignExtendHalfWord));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualSignExtendWord<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (self.0.rs1_val().unwrap_or(0), 0)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{instruction_inputs_match_constraint_test, materialize_entry_test};

    #[test]
    fn materialize_entry_virtualsignextendword() {
        materialize_entry_test!(VirtualSignExtendWord, tracer::instruction::virtual_sign_extend_word::VirtualSignExtendWord);
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualsignextendword() {
        instruction_inputs_match_constraint_test!(VirtualSignExtendWord, tracer::instruction::virtual_sign_extend_word::VirtualSignExtendWord);
    }
}
