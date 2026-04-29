use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::Bge;
use jolt_trace::JoltCycle;

impl_lookup_table!(Bge, Some(SignedGreaterThanEqual));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for Bge<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.0.rs1_val().unwrap_or(0) & mask,
            (self.0.rs2_val().unwrap_or(0) & mask) as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let shift = 64 - XLEN as u32;
        let x_signed = ((x as i64) << shift) >> shift;
        let y_signed = ((y as i64) << shift) >> shift;
        (x_signed >= y_signed) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{instruction_inputs_match_constraint_test, materialize_entry_test};

    #[test]
    fn materialize_entry_bge() {
        materialize_entry_test!(Bge, tracer::instruction::bge::BGE);
    }

    #[test]
    fn instruction_inputs_match_constraint_bge() {
        instruction_inputs_match_constraint_test!(Bge, tracer::instruction::bge::BGE);
    }
}
