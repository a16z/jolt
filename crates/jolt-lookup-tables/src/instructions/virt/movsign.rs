use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::MovSign;
use jolt_trace::{JoltCycle, JoltInstruction};

impl_lookup_table!(MovSign, Some(SignMask));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for MovSign<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.0.rs1_val().unwrap_or(0) & mask,
            self.0.instruction().imm() & mask as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, _) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let msb = 1u64 << (XLEN - 1);
        if x & msb != 0 {
            (1u128 << XLEN).wrapping_sub(1) as u64
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{instruction_inputs_match_constraint_test, materialize_entry_test};

    #[test]
    fn materialize_entry_virtualmovsign() {
        materialize_entry_test!(MovSign, tracer::instruction::virtual_movsign::VirtualMovsign);
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualmovsign() {
        instruction_inputs_match_constraint_test!(MovSign, tracer::instruction::virtual_movsign::VirtualMovsign);
    }
}
