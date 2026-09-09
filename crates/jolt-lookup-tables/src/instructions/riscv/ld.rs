use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_riscv::instructions::Ld;
use jolt_riscv::JoltCycle;

impl_lookup_table!(Ld, Some(DoublewordAlignment));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for Ld<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        super::doubleword_memory_instruction_inputs::<XLEN, _>(&self.0)
    }

    fn to_lookup_operands(&self) -> (u64, u128) {
        super::doubleword_memory_lookup_operands::<XLEN, _>(&self.0)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_lookup_output(&self) -> u64 {
        super::doubleword_memory_lookup_output::<XLEN, _>(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{instruction_inputs_match_constraint_test, materialize_entry_test};

    #[test]
    fn materialize_entry_ld() {
        materialize_entry_test!(Ld, tracer::instruction::ld::LD);
    }

    #[test]
    fn instruction_inputs_match_constraint_ld() {
        instruction_inputs_match_constraint_test!(Ld, tracer::instruction::ld::LD);
    }
}
