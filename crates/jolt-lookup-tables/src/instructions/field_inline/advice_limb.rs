//! Range-check the advised x-register limb through its non-interleaved lookup
//! operand. The field-inline constraints separately bind the limb and quotient
//! to the source field value; canonical readout requires a guest range check.

use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_riscv::instructions::FieldAdviceLimb;
use jolt_riscv::JoltCycle;

impl_lookup_table!(FieldAdviceLimb, Some(RangeCheck));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for FieldAdviceLimb<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (0, 0)
    }

    fn to_lookup_operands(&self) -> (u64, u128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            0,
            (self.0.rd_vals().map_or(0, |(_, post)| post) & mask) as u128,
        )
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_lookup_output(&self) -> u64 {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        self.0.rd_vals().map_or(0, |(_, post)| post) & mask
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use super::*;
    use crate::instructions::field_inline::test::{initialize_cpu, values};
    use crate::{
        instruction_inputs_match_constraint_test, lookup_output_matches_trace_test,
        materialize_entry_test, XLEN,
    };
    use jolt_riscv::{CircuitFlags, FieldInlineOp, Flags, InstructionFlagSet};
    use tracer::instruction::field_inline::FIELD_ADVICE_LIMB;
    use tracer::instruction::format::format_field_inline::RegisterStateFormatFieldInline;
    use tracer::instruction::{RISCVCycle, RISCVInstruction};

    fn cycle(limb: u64) -> RISCVCycle<FIELD_ADVICE_LIMB> {
        RISCVCycle {
            instruction: FIELD_ADVICE_LIMB::new(
                FieldInlineOp::AdviceLimb.instruction_match() | (3 << 7) | (1 << 15) | (2 << 20),
                0x8000_0008,
                false,
                false,
            ),
            register_state: RegisterStateFormatFieldInline {
                rs1: None,
                rd_pre: Some(!limb),
                rd_post: Some(limb),
            },
            ram_access: Default::default(),
        }
    }

    #[test]
    fn materialize_entry_field_advice_limb() {
        // RangeCheck truncates high bits, so materialization alone cannot pin the index.
        let query = FieldAdviceLimb(cycle(u64::MAX));
        assert_eq!(
            LookupQuery::<XLEN>::to_lookup_operands(&query),
            (0, u128::from(u64::MAX))
        );
        assert_eq!(
            LookupQuery::<XLEN>::to_lookup_index(&query),
            u128::from(u64::MAX)
        );
        materialize_entry_test!(
            FieldAdviceLimb,
            FIELD_ADVICE_LIMB,
            cycles = values().map(cycle)
        );
    }

    #[test]
    fn instruction_inputs_match_constraint_field_advice_limb() {
        let instruction = FieldAdviceLimb(cycle(0).instruction);
        assert_eq!(
            instruction.instruction_flags(),
            InstructionFlagSet::default()
        );
        let flags = instruction.circuit_flags();
        assert!(flags[CircuitFlags::Advice]);
        assert!(flags[CircuitFlags::WriteLookupOutputToRD]);
        instruction_inputs_match_constraint_test!(
            FieldAdviceLimb,
            FIELD_ADVICE_LIMB,
            cycles = values().map(cycle),
        );
    }

    #[test]
    fn lookup_output_matches_trace_field_advice_limb() {
        lookup_output_matches_trace_test!(
            FieldAdviceLimb,
            FIELD_ADVICE_LIMB,
            cycles = values().map(cycle),
            initialize = |cycle, cpu| {
                let limb = cycle.rd_vals().unwrap().1;
                initialize_cpu(cpu, (1u128 << 64) + u128::from(limb));
            },
        );
    }
}
