use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_riscv::instructions::VirtualSrli;
use jolt_riscv::JoltCycle;

impl_lookup_table!(VirtualSrli, Some(VirtualSRL));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualSrli<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (
            self.0.rs1_val().unwrap_or(0),
            Into::<jolt_riscv::JoltInstructionRow>::into(self.0.instruction())
                .operands
                .imm,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (rs1, imm) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        let shift = (imm as u64).trailing_zeros();
        (rs1 & mask).checked_shr(shift).unwrap_or(0)
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
    #[expect(clippy::unwrap_used)]
    fn zero_mask_output_matches_table() {
        use crate::{InstructionLookupTable, XLEN};
        use tracer::instruction::{virtual_srli::VirtualSRLI, RISCVCycle};

        let mut cycle = RISCVCycle::<VirtualSRLI>::default();
        cycle.register_state.rs1 = u64::MAX;
        cycle.instruction.operands.imm = 0;
        let table =
            InstructionLookupTable::<XLEN>::lookup_table(&VirtualSrli(cycle.instruction)).unwrap();
        let cycle = VirtualSrli(cycle);
        assert_eq!(LookupQuery::<XLEN>::to_lookup_output(&cycle), 0);
        assert_eq!(
            table.materialize_entry(LookupQuery::<XLEN>::to_lookup_index(&cycle)),
            0
        );
    }

    #[test]
    fn materialize_entry_virtualsrli() {
        materialize_entry_test!(VirtualSrli, tracer::instruction::virtual_srli::VirtualSRLI);
    }

    #[test]
    fn instruction_inputs_match_constraint_virtualsrli() {
        instruction_inputs_match_constraint_test!(
            VirtualSrli,
            tracer::instruction::virtual_srli::VirtualSRLI
        );
    }

    #[test]
    fn lookup_output_matches_trace_virtualsrli() {
        lookup_output_matches_trace_test!(
            VirtualSrli,
            tracer::instruction::virtual_srli::VirtualSRLI
        );
    }
}
