use crate::traits::impl_lookup_table;
use crate::traits::LookupQuery;
use jolt_riscv::instructions::ExtractByteUnsigned;
use jolt_riscv::{JoltCycle, JoltInstructionRow};

impl_lookup_table!(ExtractByteUnsigned, Some(ExtractByteUnsigned));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for ExtractByteUnsigned<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        (
            self.0.rs1_val().unwrap_or(0),
            self.0.rs2_val().unwrap_or(0) as i128
                + Into::<JoltInstructionRow>::into(self.0.instruction())
                    .operands
                    .imm,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (rs1, ea) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let eighth = XLEN / 8;
        let lane_mask = ((1u128 << eighth) - 1) as u64;
        let offset = ea as u64 & 7;
        (rs1 >> (eighth as u32 * offset as u32)) & lane_mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        instruction_inputs_match_constraint_test, lookup_output_matches_trace_test,
        materialize_entry_test,
    };
    use tracer::instruction::virtual_extract_byte_unsigned::VirtualExtractByteUnsigned;

    #[test]
    fn materialize_entry_extractbyteunsigned() {
        materialize_entry_test!(ExtractByteUnsigned, VirtualExtractByteUnsigned);
    }

    #[test]
    fn instruction_inputs_match_constraint_extractbyteunsigned() {
        instruction_inputs_match_constraint_test!(ExtractByteUnsigned, VirtualExtractByteUnsigned);
    }

    #[test]
    fn lookup_output_matches_trace_extractbyteunsigned() {
        lookup_output_matches_trace_test!(ExtractByteUnsigned, VirtualExtractByteUnsigned);
    }
}
