use tracer::instruction::{virtual_shift_right_bitmaski::VirtualShiftRightBitmaskI, RISCVCycle};

use crate::zkvm::lookup_table::{shift_right_bitmask::ShiftRightBitmaskTable, LookupTables};

use super::{CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, RightInputValue, NUM_CIRCUIT_FLAGS};

impl<const XLEN: usize> InstructionLookup<XLEN> for VirtualShiftRightBitmaskI {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        Some(ShiftRightBitmaskTable.into())
    }
}
impl InstructionFlags for VirtualShiftRightBitmaskI {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::WriteLookupOutputToRD as usize] = true;
        flags[CircuitFlags::AddOperands as usize] = true;
        flags[CircuitFlags::RightOperandIsImm as usize] = true;
        flags[CircuitFlags::InlineSequenceInstruction as usize] =
            self.inline_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
            self.inline_sequence_remaining.unwrap_or(0) != 0;
        flags[CircuitFlags::IsCompressed as usize] = self.is_compressed;
        flags
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualShiftRightBitmaskI> {
    fn to_instruction_inputs(&self) -> (u64, RightInputValue) {
        match XLEN {
            #[cfg(test)]
            8 => (0, RightInputValue::Unsigned(self.instruction.operands.imm as u8 as u64)),
            32 => (0, RightInputValue::Unsigned(self.instruction.operands.imm as u32 as u64)),
            64 => (0, RightInputValue::Unsigned(self.instruction.operands.imm)),
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }

    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (0, x as u128 + y.as_u64() as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_lookup_output(&self) -> u64 {
        let y = LookupQuery::<XLEN>::to_lookup_index(self);
        match XLEN {
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
        materialize_entry_test::<Fr, VirtualShiftRightBitmaskI>();
    }
}
