use crate::interleave_bits;
use crate::tables::funnel_high::FunnelHighTable;
use crate::traits::impl_lookup_table;
use crate::traits::{LookupQuery, LookupTable};
use jolt_riscv::instructions::FunnelHigh;
use jolt_riscv::JoltCycle;

impl_lookup_table!(FunnelHigh, Some(FunnelHigh));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for FunnelHigh<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (
            self.0.rs1_val().unwrap_or(0),
            self.0.rs2_val().unwrap_or(0) as i128,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (rs1, rs2) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        FunnelHighTable::<XLEN>.materialize_entry(interleave_bits(rs1 & mask, rs2 as u64 & mask))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{instruction_inputs_match_constraint_test, lookup_output_matches_trace_test};
    use tracer::instruction::virtual_funnel_high::VirtualFunnelHigh;

    #[test]
    fn instruction_inputs_match_constraint_funnelhigh() {
        instruction_inputs_match_constraint_test!(FunnelHigh, VirtualFunnelHigh);
    }

    #[test]
    fn lookup_output_matches_trace_funnelhigh() {
        lookup_output_matches_trace_test!(FunnelHigh, VirtualFunnelHigh);
    }
}
