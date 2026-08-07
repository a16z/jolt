use crate::zkvm::instruction::{InstructionFlags, NUM_INSTRUCTION_FLAGS};
use crate::zkvm::lookup_table::lane_mask::lane_value;
use crate::zkvm::lookup_table::LookupTables;
use tracer::instruction::{
    virtual_align_addr::VirtualAlignAddr, virtual_lane_mask_b::VirtualLaneMaskB,
    virtual_lane_mask_h::VirtualLaneMaskH, virtual_lane_mask_w::VirtualLaneMaskW,
    virtual_pow2_lane::VirtualPow2Lane, RISCVCycle,
};

use super::{CircuitFlags, Flags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

macro_rules! impl_address_lookup {
    ($instruction:ident, $table:ident, $output:expr) => {
        impl<const XLEN: usize> InstructionLookup<XLEN> for $instruction {
            fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
                Some(LookupTables::$table(Default::default()))
            }
        }

        impl Flags for $instruction {
            fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
                let mut flags = [false; NUM_CIRCUIT_FLAGS];
                flags[CircuitFlags::AddOperands] = true;
                flags[CircuitFlags::WriteLookupOutputToRD] = true;
                flags[CircuitFlags::VirtualInstruction] = self.virtual_sequence_remaining.is_some();
                flags[CircuitFlags::DoNotUpdateUnexpandedPC] =
                    self.virtual_sequence_remaining.unwrap_or(0) != 0;
                flags[CircuitFlags::IsFirstInSequence] = self.is_first_in_sequence;
                flags[CircuitFlags::IsCompressed] = self.is_compressed;
                flags
            }

            fn instruction_flags(&self) -> [bool; NUM_INSTRUCTION_FLAGS] {
                let mut flags = [false; NUM_INSTRUCTION_FLAGS];
                flags[InstructionFlags::LeftOperandIsRs1Value] = true;
                flags[InstructionFlags::RightOperandIsImm] = true;
                flags
            }
        }

        impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<$instruction> {
            fn to_instruction_inputs(&self) -> (u64, i128) {
                match XLEN {
                    #[cfg(test)]
                    8 => (
                        self.register_state.rs1 as u8 as u64,
                        self.instruction.operands.imm as u8 as i128,
                    ),
                    32 => (
                        self.register_state.rs1 as u32 as u64,
                        self.instruction.operands.imm as u32 as i128,
                    ),
                    64 => (
                        self.register_state.rs1,
                        self.instruction.operands.imm as i128,
                    ),
                    _ => panic!("{XLEN}-bit word size is unsupported"),
                }
            }

            fn to_lookup_operands(&self) -> (u64, u128) {
                let (address, offset) = LookupQuery::<XLEN>::to_instruction_inputs(self);
                (0, (address as i128 + offset) as u128)
            }

            fn to_lookup_index(&self) -> u128 {
                LookupQuery::<XLEN>::to_lookup_operands(self).1
            }

            fn to_lookup_output(&self) -> u64 {
                ($output)(LookupQuery::<XLEN>::to_lookup_index(self))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zkvm::instruction::test::{
        lookup_output_matches_trace_test, materialize_entry_test,
    };
    use ark_bn254::Fr;

    macro_rules! instruction_tests {
        ($module:ident, $instruction:ident) => {
            mod $module {
                use super::*;

                #[test]
                fn materialize_entry() {
                    materialize_entry_test::<Fr, $instruction>();
                }

                #[test]
                fn lookup_output_matches_trace() {
                    lookup_output_matches_trace_test::<$instruction>();
                }
            }
        };
    }

    instruction_tests!(align_addr, VirtualAlignAddr);
    instruction_tests!(lane_mask_b, VirtualLaneMaskB);
    instruction_tests!(lane_mask_h, VirtualLaneMaskH);
    instruction_tests!(lane_mask_w, VirtualLaneMaskW);
    instruction_tests!(pow2_lane, VirtualPow2Lane);
}
