use crate::instructions::virt::srlw::srlw;
use crate::traits::{impl_lookup_table, LookupQuery};
use jolt_riscv::instructions::VirtualSrliw;
use jolt_riscv::JoltCycle;

impl_lookup_table!(VirtualSrliw, Some(VirtualSRLW));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualSrliw<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (
            self.0.rs1_val().unwrap_or(0),
            Into::<jolt_riscv::JoltInstructionRow>::into(self.0.instruction())
                .operands
                .imm,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        srlw::<XLEN>(x, y as u64)
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
    fn materialize_entry_virtualsrliw() {
        materialize_entry_test!(
            VirtualSrliw,
            tracer::instruction::virtual_srliw::VirtualSRLIW
        );
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualsrliw() {
        instruction_inputs_match_constraint_test!(
            VirtualSrliw,
            tracer::instruction::virtual_srliw::VirtualSRLIW
        );
    }

    #[test]
    fn lookup_output_matches_trace_virtualsrliw() {
        lookup_output_matches_trace_test!(
            VirtualSrliw,
            tracer::instruction::virtual_srliw::VirtualSRLIW
        );
    }
}
