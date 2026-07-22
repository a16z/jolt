use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_riscv::instructions::Pow2IW;
use jolt_riscv::JoltCycle;

impl_lookup_table!(Pow2IW, Some(Pow2W));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for Pow2IW<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (
            0,
            Into::<jolt_riscv::JoltInstructionRow>::into(self.0.instruction())
                .operands
                .imm,
        )
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
    use crate::{
        instruction_inputs_match_constraint_test, lookup_output_matches_trace_test,
        materialize_entry_test,
    };

    #[test]
    fn materialize_entry_virtualpow2iw() {
        materialize_entry_test!(Pow2IW, jolt_tracer::instruction::virtual_pow2i_w::VirtualPow2IW);
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualpow2iw() {
        instruction_inputs_match_constraint_test!(
            Pow2IW,
            jolt_tracer::instruction::virtual_pow2i_w::VirtualPow2IW
        );
    }

    #[test]
    fn lookup_output_matches_trace_virtualpow2iw() {
        lookup_output_matches_trace_test!(
            Pow2IW,
            jolt_tracer::instruction::virtual_pow2i_w::VirtualPow2IW
        );
    }
}
