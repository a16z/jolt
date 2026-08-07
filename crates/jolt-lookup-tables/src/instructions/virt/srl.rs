use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_riscv::instructions::VirtualSrl;
use jolt_riscv::JoltCycle;

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
        let mut x = rs1 & mask;
        let mut y = rs2 as u64 & mask;
        let mut result = 0;
        for _ in 0..XLEN {
            let x_i = x >> (XLEN - 1) & 1;
            let y_i = y >> (XLEN - 1) & 1;
            result *= 1 + y_i;
            result += x_i * y_i;
            x <<= 1;
            y <<= 1;
        }
        result
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
    fn materialize_entry_virtualsrl() {
        materialize_entry_test!(VirtualSrl, tracer::instruction::virtual_srl::VirtualSRL);
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualsrl() {
        instruction_inputs_match_constraint_test!(
            VirtualSrl,
            tracer::instruction::virtual_srl::VirtualSRL
        );
    }

    #[test]
    fn lookup_output_matches_trace_virtualsrl() {
        lookup_output_matches_trace_test!(VirtualSrl, tracer::instruction::virtual_srl::VirtualSRL);
    }
}
