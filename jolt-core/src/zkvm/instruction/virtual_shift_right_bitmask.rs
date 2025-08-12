use tracer::instruction::{virtual_shift_right_bitmask::VirtualShiftRightBitmask, RISCVCycle};

use crate::zkvm::lookup_table::{shift_right_bitmask::ShiftRightBitmaskTable, LookupTables};

use super::{CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualShiftRightBitmask {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(ShiftRightBitmaskTable.into())
    }
}

impl InstructionFlags for VirtualShiftRightBitmask {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::LeftOperandIsRs1Value as usize] = true;
        flags[CircuitFlags::RightOperandIsImm as usize] = true;
        flags[CircuitFlags::WriteLookupOutputToRD as usize] = true;
        flags[CircuitFlags::AddOperands as usize] = true;
        flags[CircuitFlags::InlineSequenceInstruction as usize] =
            self.inline_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
            self.inline_sequence_remaining.unwrap_or(0) != 0;
        flags[CircuitFlags::IsCompressed as usize] = self.is_compressed;
        flags
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for RISCVCycle<VirtualShiftRightBitmask> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        match WORD_SIZE {
            #[cfg(test)]
            8 => (self.register_state.rs1 as u8 as u64, 0),
            32 => (self.register_state.rs1 as u32 as u64, 0),
            64 => (self.register_state.rs1, 0),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = LookupQuery::<WORD_SIZE>::to_instruction_inputs(self);
        (0, x as u128 + y as u64 as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<WORD_SIZE>::to_lookup_operands(self).1
    }

    fn to_lookup_output(&self) -> u64 {
        let y = LookupQuery::<WORD_SIZE>::to_lookup_index(self);
        match WORD_SIZE {
            #[cfg(test)]
            8 => {
                let shift = (y % 8) as u64;
                let ones = (1u64 << (8 - shift)) - 1;
                ones << shift
            }
            32 => {
                let shift = (y % 32) as u64;
                let ones = (1u64 << (32 - shift)) - 1;
                ones << shift
            }
            64 => {
                let shift = (y % 64) as u64;
                let ones = (1u128 << (64 - shift)) - 1;
                (ones << shift) as u64
            }
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
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
        materialize_entry_test::<Fr, VirtualShiftRightBitmask>();
    }
}
