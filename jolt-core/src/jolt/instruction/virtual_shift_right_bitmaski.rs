use tracer::instruction::{virtual_shift_right_bitmaski::VirtualShiftRightBitmaskI, RISCVCycle};

use crate::jolt::lookup_table::{shift_right_bitmask::ShiftRightBitmaskTable, LookupTables};

use super::{CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualShiftRightBitmaskI {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(ShiftRightBitmaskTable.into())
    }
}
impl InstructionFlags for VirtualShiftRightBitmaskI {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::WriteLookupOutputToRD as usize] = true;
        flags[CircuitFlags::RightOperandIsImm as usize] = true;
        flags[CircuitFlags::Virtual as usize] = self.virtual_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdatePC as usize] =
            self.virtual_sequence_remaining.unwrap_or(0) != 0;
        flags
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for RISCVCycle<VirtualShiftRightBitmaskI> {
    fn to_lookup_query(&self) -> (u64, u64) {
        (self.instruction.operands.imm as u64, 0)
    }

    fn to_lookup_index(&self) -> u64 {
        self.instruction.operands.imm as u64
    }

    #[cfg(test)]
    fn to_lookup_output(&self) -> u64 {
        let (x, _) = LookupQuery::<WORD_SIZE>::to_lookup_query(self);
        match WORD_SIZE {
            #[cfg(test)]
            8 => {
                let shift = x % 8;
                let ones = (1u64 << (8 - shift)) - 1;
                (ones << shift) as u64
            }
            32 => {
                let shift = x % 32;
                let ones = (1u64 << (32 - shift)) - 1;
                (ones << shift) as u64
            }
            64 => {
                let shift = x % 64;
                let ones = (1u128 << (64 - shift)) - 1;
                (ones << shift) as u64
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
        materialize_entry_test::<Fr, VirtualShiftRightBitmaskI>();
    }
}
