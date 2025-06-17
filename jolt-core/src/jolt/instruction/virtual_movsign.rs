use tracer::instruction::{virtual_movsign::VirtualMovsign, RISCVCycle};

use crate::jolt::lookup_table::{movsign::MovsignTable, LookupTables};

use super::{CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualMovsign {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(MovsignTable.into())
    }
}

impl InstructionFlags for VirtualMovsign {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::LeftOperandIsRs1Value as usize] = true;
        flags[CircuitFlags::RightOperandIsImm as usize] = true;
        flags[CircuitFlags::WriteLookupOutputToRD as usize] = true;
        flags[CircuitFlags::InlineSequenceInstruction as usize] =
            self.virtual_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
            self.virtual_sequence_remaining.unwrap_or(0) != 0;
        flags
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for RISCVCycle<VirtualMovsign> {
    fn to_instruction_inputs(&self) -> (u64, i64) {
        (
            self.register_state.rs1,
            self.instruction.operands.imm as i64, // Unused
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, _) = LookupQuery::<WORD_SIZE>::to_instruction_inputs(self);
        match WORD_SIZE {
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
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::jolt::instruction::test::materialize_entry_test;

    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, VirtualMovsign>();
    }
}
