use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::Lui;
use jolt_trace::{JoltCycle, JoltInstruction};

impl_lookup_table!(Lui, Some(RangeCheck));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for Lui<C> {
    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (0, self.0.instruction().imm() & mask as i128)
    }

    fn to_lookup_output(&self) -> u64 {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        self.0.instruction().imm() as u64 & mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{instruction_inputs_match_constraint_test, materialize_entry_test};

    #[test]
    fn materialize_entry_lui() {
        materialize_entry_test!(Lui, tracer::instruction::lui::LUI);
    }

    #[test]
    fn instruction_inputs_match_constraint_lui() {
        instruction_inputs_match_constraint_test!(Lui, tracer::instruction::lui::LUI);
    }
}
