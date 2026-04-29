use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_trace::instructions::VirtualSrl;
use jolt_trace::JoltCycle;

impl_lookup_table!(VirtualSrl, Some(VirtualSRL));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualSrl<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (
            self.0.rs1_val().unwrap_or(0),
            self.0.rs2_val().unwrap_or(0) as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (rs1, rs2) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        let shift = (rs2 as u64).trailing_zeros();
        (rs1 & mask) >> shift
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{instruction_inputs_match_constraint_test, materialize_entry_test};

    #[test]
    fn materialize_entry_virtualsrl() {
        materialize_entry_test!(VirtualSrl, tracer::instruction::virtual_srl::VirtualSRL);
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualsrl() {
        instruction_inputs_match_constraint_test!(VirtualSrl, tracer::instruction::virtual_srl::VirtualSRL);
    }
}
