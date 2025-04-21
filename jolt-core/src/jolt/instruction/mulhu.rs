use tracer::instruction::{mulhu::MULHU, RISCVCycle};

use crate::jolt::lookup_table::{upper_word::UpperWordTable, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for MULHU {
    fn lookup_table() -> Option<LookupTables<WORD_SIZE>> {
        Some(UpperWordTable.into())
    }

    fn to_lookup_index(cycle: &RISCVCycle<Self>) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::lookup_query(cycle);
        match WORD_SIZE {
            #[cfg(test)]
            8 => x * y,
            32 => x * y,
            // 64 => x * y,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn lookup_query(cycle: &RISCVCycle<Self>) -> (u64, u64) {
        (cycle.register_state.rs1, cycle.register_state.rs2)
    }

    fn lookup_entry(cycle: &RISCVCycle<Self>) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::lookup_query(cycle);
        match WORD_SIZE {
            #[cfg(test)]
            8 => (x * y) >> 8,
            32 => (x * y) >> 32,
            64 => ((x as u128).wrapping_mul(y as u128) >> 64) as u64,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}
