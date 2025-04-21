use tracer::instruction::{sltiu::SLTIU, RISCVCycle};

use crate::jolt::lookup_table::{unsigned_less_than::UnsignedLessThanTable, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for RISCVCycle<SLTIU> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(UnsignedLessThanTable.into())
    }

    fn to_lookup_query(&self) -> (u64, u64) {
        (
            self.register_state.rs1,
            self.instruction.operands.imm as u64,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::to_lookup_query(self);
        match WORD_SIZE {
            #[cfg(test)]
            8 => ((x as u8) < y as u8).into(),
            32 => ((x as u32) < y as u32).into(),
            64 => (x < y).into(),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}
