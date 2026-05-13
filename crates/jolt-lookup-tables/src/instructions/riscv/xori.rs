use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_riscv::instructions::XorI;
use jolt_riscv::JoltCycle;

impl_lookup_table!(XorI, Some(Xor));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for XorI<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.0.rs1_val().unwrap_or(0) & mask,
            Into::<jolt_riscv::JoltRow>::into(self.0.instruction())
                .operands
                .imm
                & mask as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        x ^ y as u64
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
    fn materialize_entry_xori() {
        materialize_entry_test!(XorI, tracer::instruction::xori::XORI);
    }

    #[test]
    fn instruction_inputs_match_constraint_xori() {
        instruction_inputs_match_constraint_test!(XorI, tracer::instruction::xori::XORI);
    }

    #[test]
    fn lookup_output_matches_trace_xori() {
        lookup_output_matches_trace_test!(XorI, tracer::instruction::xori::XORI);
    }
}
