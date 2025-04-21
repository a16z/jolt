use crate::jolt::lookup_table::LookupTables;
use crate::jolt::{instruction::InstructionLookup, lookup_table::range_check::RangeCheckTable};
use tracer::instruction::{addi::ADDI, RISCVCycle};

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for RISCVCycle<ADDI> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(RangeCheckTable.into())
    }

    fn to_lookup_index(&self) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::to_lookup_query(self);
        match WORD_SIZE {
            #[cfg(test)]
            8 => x + y,
            32 => x + y,
            // 64 => x + y,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
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
            8 => (x as u8).overflowing_add(y as u8).0.into(),
            32 => (x as u32).overflowing_add(y as u32).0.into(),
            64 => x.overflowing_add(y).0,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}
