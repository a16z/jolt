use tracer::instruction::{virtual_pow2i::VirtualPow2I, RISCVCycle};

use crate::jolt::lookup_table::{pow2::Pow2Table, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualPow2I {
    fn lookup_table() -> Option<LookupTables<WORD_SIZE>> {
        Some(Pow2Table.into())
    }

    fn lookup_query(cycle: &RISCVCycle<Self>) -> (u64, u64) {
        (cycle.instruction.operands.imm as u64, 0)
    }

    fn lookup_entry(cycle: &RISCVCycle<Self>) -> u64 {
        let (x, _) = InstructionLookup::<WORD_SIZE>::lookup_query(cycle);
        match WORD_SIZE {
            #[cfg(test)]
            8 => 1u64 << (x % 8),
            32 => 1u64 << (x % 32),
            64 => 1u64 << (x % 64),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}
