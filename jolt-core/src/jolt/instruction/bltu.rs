use super::InstructionLookup;
use crate::jolt::lookup_table::{unsigned_less_than::UnsignedLessThanTable, LookupTables};
use tracer::instruction::{bltu::BLTU, RISCVCycle};

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for BLTU {
    fn lookup_table() -> Option<LookupTables<WORD_SIZE>> {
        Some(UnsignedLessThanTable.into())
    }

    fn lookup_query(cycle: &RISCVCycle<Self>) -> (u64, u64) {
        (cycle.register_state.rs1, cycle.register_state.rs2)
    }

    fn lookup_entry(cycle: &RISCVCycle<Self>) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::lookup_query(cycle);
        match WORD_SIZE {
            #[cfg(test)]
            8 => (x as u8) < (y as u8) as u64,
            32 => ((x as u32) < (y as u32)) as u64,
            64 => (x < y) as u64,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}
