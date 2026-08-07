use crate::tables::lane_extract_s::signed_extract;
use crate::tables::lane_mask::lane_value;
use crate::traits::{impl_lookup_table, LookupQuery};
use jolt_riscv::instructions::{
    VirtualAlignAddr, VirtualLaneExtractS, VirtualLaneMaskB, VirtualLaneMaskH, VirtualLaneMaskW,
    VirtualPow2Lane,
};
use jolt_riscv::JoltCycle;

macro_rules! impl_address_lookup {
    ($instruction:ident, $table:ident, $output:expr) => {
        impl_lookup_table!($instruction, Some($table));

        impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for $instruction<C> {
            fn to_instruction_inputs(&self) -> (u64, i128) {
                let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
                (
                    self.0.rs1_val().unwrap_or(0) & mask,
                    Into::<jolt_riscv::JoltInstructionRow>::into(self.0.instruction())
                        .operands
                        .imm,
                )
            }

            fn to_lookup_operands(&self) -> (u64, u128) {
                let (address, offset) = LookupQuery::<XLEN>::to_instruction_inputs(self);
                (0, (address as i128 + offset) as u128)
            }

            fn to_lookup_index(&self) -> u128 {
                LookupQuery::<XLEN>::to_lookup_operands(self).1
            }

            fn to_lookup_output(&self) -> u64 {
                let index = LookupQuery::<XLEN>::to_lookup_index(self);
                ($output)(index)
            }
        }
    };
}

impl_address_lookup!(VirtualAlignAddr, AlignAddr, |index: u128| index as u64 & !7);
impl_address_lookup!(VirtualLaneMaskB, LaneMaskB, |index: u128| lane_value::<1>(
    index as u64
));
impl_address_lookup!(VirtualLaneMaskH, LaneMaskH, |index: u128| lane_value::<2>(
    index as u64
));
impl_address_lookup!(VirtualLaneMaskW, LaneMaskW, |index: u128| lane_value::<4>(
    index as u64
));
impl_address_lookup!(VirtualPow2Lane, Pow2Lane, |index: u128| lane_value::<0>(
    index as u64
));

impl_lookup_table!(VirtualLaneExtractS, Some(LaneExtractS));

impl<const XLEN: usize, C: JoltCycle> LookupQuery<XLEN> for VirtualLaneExtractS<C> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        let mask = (1u128 << XLEN).wrapping_sub(1) as u64;
        (
            self.0.rs1_val().unwrap_or(0) & mask,
            i128::from(self.0.rs2_val().unwrap_or(0) & mask),
        )
    }

    fn to_lookup_output(&self) -> u64 {
        signed_extract::<XLEN>(LookupQuery::<XLEN>::to_lookup_index(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        instruction_inputs_match_constraint_test, lookup_output_matches_trace_test,
        materialize_entry_test,
    };

    macro_rules! instruction_tests {
        ($module:ident, $instruction:ident, $tracer:path) => {
            mod $module {
                use super::*;

                #[test]
                fn materialize_entry() {
                    materialize_entry_test!($instruction, $tracer);
                }

                #[test]
                fn instruction_inputs_match_constraint() {
                    instruction_inputs_match_constraint_test!($instruction, $tracer);
                }

                #[test]
                fn lookup_output_matches_trace() {
                    lookup_output_matches_trace_test!($instruction, $tracer);
                }
            }
        };
    }

    instruction_tests!(
        align_addr,
        VirtualAlignAddr,
        tracer::instruction::virtual_align_addr::VirtualAlignAddr
    );
    instruction_tests!(
        lane_mask_b,
        VirtualLaneMaskB,
        tracer::instruction::virtual_lane_mask_b::VirtualLaneMaskB
    );
    instruction_tests!(
        lane_mask_h,
        VirtualLaneMaskH,
        tracer::instruction::virtual_lane_mask_h::VirtualLaneMaskH
    );
    instruction_tests!(
        lane_mask_w,
        VirtualLaneMaskW,
        tracer::instruction::virtual_lane_mask_w::VirtualLaneMaskW
    );
    instruction_tests!(
        pow2_lane,
        VirtualPow2Lane,
        tracer::instruction::virtual_pow2_lane::VirtualPow2Lane
    );
    instruction_tests!(
        lane_extract_s,
        VirtualLaneExtractS,
        tracer::instruction::virtual_lane_extract_s::VirtualLaneExtractS
    );
}
