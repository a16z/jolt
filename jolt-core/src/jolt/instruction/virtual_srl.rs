use tracer::instruction::{virtual_srl::VirtualSRL, RISCVCycle};

use crate::jolt::lookup_table::{virtual_srl::VirtualSRLTable, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualSRL {
    fn lookup_table() -> Option<LookupTables<WORD_SIZE>> {
        Some(VirtualSRLTable.into())
    }

    fn lookup_query(cycle: &RISCVCycle<Self>) -> (u64, u64) {
        (cycle.register_state.rs1, cycle.register_state.rs2)
    }

    fn lookup_entry(cycle: &RISCVCycle<Self>) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::lookup_query(cycle);
        match WORD_SIZE {
            #[cfg(test)]
            8 => (x as u8 >> y.trailing_zeros()) as u64,
            32 => ((x as u32) >> y.trailing_zeros()) as u64,
            64 => x >> y.trailing_zeros(),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}
