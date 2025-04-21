use tracer::instruction::{bgeu::BGEU, RISCVCycle};

use crate::jolt::lookup_table::{
    unsigned_greater_than_equal::UnsignedGreaterThanEqualTable, LookupTables,
};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for RISCVCycle<BGEU> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(UnsignedGreaterThanEqualTable.into())
    }

    fn to_lookup_query(&self) -> (u64, u64) {
        (self.register_state.rs1, self.register_state.rs2)
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::to_lookup_query(self);
        match WORD_SIZE {
            #[cfg(test)]
            8 => (x >= y).into(),
            32 => (x >= y).into(),
            64 => (x >= y).into(),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}
