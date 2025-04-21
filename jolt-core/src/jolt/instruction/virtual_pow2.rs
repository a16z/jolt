use tracer::instruction::{virtual_pow2::VirtualPow2, RISCVCycle};

use crate::jolt::lookup_table::{pow2::Pow2Table, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualPow2 {
    fn lookup_table() -> Option<LookupTables<WORD_SIZE>> {
        Some(Pow2Table.into())
    }

    fn lookup_query(cycle: &RISCVCycle<Self>) -> (u64, u64) {
        (cycle.register_state.rs1, 0)
    }

    fn lookup_entry(cycle: &RISCVCycle<Self>) -> u64 {
        let (x, _) = InstructionLookup::<WORD_SIZE>::lookup_query(cycle);
        match WORD_SIZE {
            #[cfg(test)]
            8 => (1u64 << (x % 8)) as u64,
            32 => (1u64 << (x % 32)) as u64,
            64 => 1u64 << (x % 64),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}
