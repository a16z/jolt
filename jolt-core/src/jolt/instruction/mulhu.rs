use tracer::instruction::{mulhu::MULHU, RISCVCycle};

use crate::jolt::lookup_table::{upper_word::UpperWordTable, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for RISCVCycle<MULHU> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(UpperWordTable.into())
    }

    fn to_lookup_index(&self) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::to_lookup_query(self);
        match WORD_SIZE {
            #[cfg(test)]
            8 => x * y,
            32 => x * y,
            // 64 => x * y,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn to_lookup_query(&self) -> (u64, u64) {
        (self.register_state.rs1, self.register_state.rs2)
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::to_lookup_query(self);
        match WORD_SIZE {
            #[cfg(test)]
            8 => (x * y) >> 8,
            32 => (x * y) >> 32,
            64 => ((x as u128).wrapping_mul(y as u128) >> 64) as u64,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}
