use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::Pow2W;
use jolt_trace::JoltCycle;

impl_lookup_table!(Pow2W, Some(Pow2W));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for Pow2W<C> {
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
        let y = LookupQuery::<XLEN>::to_lookup_index(self);
        1u64 << (y & ((XLEN as u128 / 2) - 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{instruction_inputs_match_constraint_test, materialize_entry_test};

    #[test]
    fn materialize_entry_virtualpow2w() {
        materialize_entry_test!(Pow2W, tracer::instruction::virtual_pow2_w::VirtualPow2W);
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualpow2w() {
        instruction_inputs_match_constraint_test!(Pow2W, tracer::instruction::virtual_pow2_w::VirtualPow2W);
    }
}
