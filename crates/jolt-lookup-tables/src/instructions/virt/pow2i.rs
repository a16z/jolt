use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::Pow2I;
use jolt_trace::{JoltCycle, JoltInstruction};

impl_lookup_table!(Pow2I, Some(Pow2));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for Pow2I<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (0, self.0.instruction().imm() & mask as i128)
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
        1u64 << (y & ((XLEN as u128) - 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{instruction_inputs_match_constraint_test, materialize_entry_test};

    #[test]
    fn materialize_entry_virtualpow2i() {
        materialize_entry_test!(Pow2I, tracer::instruction::virtual_pow2i::VirtualPow2I);
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualpow2i() {
        instruction_inputs_match_constraint_test!(Pow2I, tracer::instruction::virtual_pow2i::VirtualPow2I);
    }
}
