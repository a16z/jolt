use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_riscv::instructions::MulC;
use jolt_riscv::JoltCycle;

impl_lookup_table!(MulC, Some(RangeCheck));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for MulC<C> {
    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        // rs1 * rs2 + carry <= (2^64-1)^2 + (2^64-1) < 2^128: no overflow.
        (0, x as u128 * (y as u64 as u128) + self.0.carry() as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.0.rs1_val().unwrap_or(0) & mask,
            (self.0.rs2_val().unwrap_or(0) & mask) as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        x.wrapping_mul(y as u64).wrapping_add(self.0.carry()) & mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        instruction_inputs_match_constraint_test, lookup_output_matches_trace_test,
        materialize_entry_test,
    };
    use tracer::instruction::mulc::MULC;

    #[test]
    fn materialize_entry_mulc() {
        materialize_entry_test!(MulC, MULC);
    }

    #[test]
    fn instruction_inputs_match_constraint_mulc() {
        instruction_inputs_match_constraint_test!(MulC, MULC);
    }

    #[test]
    fn lookup_output_matches_trace_mulc() {
        lookup_output_matches_trace_test!(MulC, MULC);
    }
}
