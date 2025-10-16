use crate::zkvm::instruction::{InstructionFlags, NUM_INSTRUCTION_FLAGS};
use tracer::instruction::{virtual_movsign::VirtualMovsign, RISCVCycle};

use crate::zkvm::lookup_table::{movsign::MovsignTable, LookupTables};

use super::{CircuitFlags, Flags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const XLEN: usize> InstructionLookup<XLEN> for VirtualMovsign {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        Some(MovsignTable.into())
    }
}

impl Flags for VirtualMovsign {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::WriteLookupOutputToRD as usize] = true;
        flags[CircuitFlags::InlineSequenceInstruction as usize] =
            self.inline_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
            self.inline_sequence_remaining.unwrap_or(0) != 0;
        flags[CircuitFlags::IsFirstInSequence as usize] = self.is_first_in_sequence;
        flags[CircuitFlags::IsCompressed as usize] = self.is_compressed;
        flags
    }

    fn instruction_flags(&self) -> [bool; NUM_INSTRUCTION_FLAGS] {
        let mut flags = [false; NUM_INSTRUCTION_FLAGS];
        flags[InstructionFlags::LeftOperandIsRs1Value as usize] = true;
        flags[InstructionFlags::RightOperandIsImm as usize] = true;
        flags
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualMovsign> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        match XLEN {
            #[cfg(test)]
            8 => (
                self.register_state.rs1 as u8 as u64,
                self.instruction.operands.imm as u8 as i128, // Unused
            ),
            32 => (
                self.register_state.rs1 as u32 as u64,
                self.instruction.operands.imm as u32 as i128, // Unused
            ),
            64 => (
                self.register_state.rs1,
                self.instruction.operands.imm as i128, // Unused
            ),
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, _) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        match XLEN {
            #[cfg(test)]
            8 => {
                if x & (1 << 7) != 0 {
                    0xFF
                } else {
                    0
                }
            }
            32 => {
                if x & (1 << 31) != 0 {
                    0xFFFFFFFF
                } else {
                    0
                }
            }
            64 => {
                if x & (1 << 63) != 0 {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            }
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::zkvm::instruction::test::materialize_entry_test;

    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, VirtualMovsign>();
    }
}
