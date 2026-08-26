use crate::traits::{impl_lookup_table, LookupQuery};
use jolt_riscv::instructions::VirtualNegateIf;
use jolt_riscv::JoltCycle;

impl_lookup_table!(VirtualNegateIf, Some(VirtualNegateIf));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualNegateIf<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.0.rs1_val().unwrap_or(0) & mask,
            (self.0.rs2_val().unwrap_or(0) & mask) as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (sign_source, value) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        if sign_source & (1 << (XLEN - 1)) == 0 {
            value as u64
        } else {
            (value as u64).wrapping_neg() & mask
        }
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
    fn materialize_entry_virtualnegateif() {
        materialize_entry_test!(
            VirtualNegateIf,
            tracer::instruction::virtual_negate_if::VirtualNegateIf
        );
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualnegateif() {
        instruction_inputs_match_constraint_test!(
            VirtualNegateIf,
            tracer::instruction::virtual_negate_if::VirtualNegateIf
        );
    }

    #[test]
    fn lookup_output_matches_trace_virtualnegateif() {
        lookup_output_matches_trace_test!(
            VirtualNegateIf,
            tracer::instruction::virtual_negate_if::VirtualNegateIf
        );
    }
}
