use tracer::instruction::{virtual_shift_right_bitmaski::VirtualShiftRightBitmaskI, RISCVCycle};

use crate::jolt::lookup_table::{shift_right_bitmask::ShiftRightBitmaskTable, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualShiftRightBitmaskI {
    fn lookup_table() -> Option<LookupTables<WORD_SIZE>> {
        Some(ShiftRightBitmaskTable.into())
    }

    fn lookup_query(cycle: &RISCVCycle<Self>) -> (u64, u64) {
        (cycle.instruction.operands.imm as u64, 0)
    }

    fn lookup_entry(cycle: &RISCVCycle<Self>) -> u64 {
        let (x, _) = InstructionLookup::<WORD_SIZE>::lookup_query(cycle);
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
