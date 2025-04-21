use tracer::instruction::{xori::XORI, RISCVCycle};

use crate::jolt::lookup_table::{xor::XorTable, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for XORI {
    fn lookup_table() -> Option<LookupTables<WORD_SIZE>> {
        Some(XorTable.into())
    }

    fn lookup_query(cycle: &RISCVCycle<Self>) -> (u64, u64) {
        (
            cycle.register_state.rs1,
            cycle.instruction.operands.imm as u64,
        )
    }

    fn lookup_entry(cycle: &RISCVCycle<Self>) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::lookup_query(cycle);
        match WORD_SIZE {
            #[cfg(test)]
            8 => (x as u8 ^ y as u8).into(),
            32 => (x as u32 ^ y as u32).into(),
            64 => x ^ y,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}
