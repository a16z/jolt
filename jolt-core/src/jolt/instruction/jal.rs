use tracer::instruction::{jal::JAL, RISCVCycle};

use crate::jolt::lookup_table::{range_check::RangeCheckTable, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for JAL {
    fn lookup_table() -> Option<LookupTables<WORD_SIZE>> {
        Some(RangeCheckTable.into())
    }

    fn to_lookup_index(cycle: &RISCVCycle<Self>) -> u64 {
        let (pc, imm) = InstructionLookup::<WORD_SIZE>::lookup_query(cycle);
        match WORD_SIZE {
            #[cfg(test)]
            8 => pc + imm,
            32 => pc + imm,
            // 64 => pc.overflowing_add(imm).0,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn lookup_query(cycle: &RISCVCycle<Self>) -> (u64, u64) {
        (
            cycle.instruction.address,
            cycle.instruction.operands.imm as u64,
        )
    }

    fn lookup_entry(cycle: &RISCVCycle<Self>) -> u64 {
        let (pc, imm) = InstructionLookup::<WORD_SIZE>::lookup_query(cycle);
        match WORD_SIZE {
            #[cfg(test)]
            8 => (pc as u8).overflowing_add(imm as u8).0.into(),
            32 => (pc as u32).overflowing_add(imm as u32).0.into(),
            64 => pc.overflowing_add(imm).0,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}
