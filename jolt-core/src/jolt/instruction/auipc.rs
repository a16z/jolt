use tracer::instruction::{auipc::AUIPC, RISCVCycle};

use crate::jolt::lookup_table::{range_check::RangeCheckTable, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for RISCVCycle<AUIPC> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(RangeCheckTable.into())
    }

    fn to_lookup_index(&self) -> u64 {
        let (pc, imm) = InstructionLookup::<WORD_SIZE>::to_lookup_query(self);
        match WORD_SIZE {
            #[cfg(test)]
            8 => pc + imm,
            32 => pc + imm,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn to_lookup_query(&self) -> (u64, u64) {
        (
            self.instruction.address,
            self.instruction.operands.imm as u64,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (pc, imm) = InstructionLookup::<WORD_SIZE>::to_lookup_query(self);
        match WORD_SIZE {
            #[cfg(test)]
            8 => (pc as u8).overflowing_add(imm as u8).0.into(),
            32 => (pc as u32).overflowing_add(imm as u32).0.into(),
            64 => pc.overflowing_add(imm).0,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}
