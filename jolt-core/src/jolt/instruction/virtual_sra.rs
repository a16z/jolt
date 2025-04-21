use tracer::instruction::{virtual_sra::VirtualSRA, RISCVCycle};

use crate::jolt::lookup_table::{virtual_sra::VirtualSRATable, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualSRA {
    fn lookup_table() -> Option<LookupTables<WORD_SIZE>> {
        Some(VirtualSRATable.into())
    }

    fn lookup_query(cycle: &RISCVCycle<Self>) -> (u64, u64) {
        (cycle.register_state.rs1, cycle.register_state.rs2)
    }

    fn lookup_entry(cycle: &RISCVCycle<Self>) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::lookup_query(cycle);
        match WORD_SIZE {
            #[cfg(test)]
            8 => {
                let shift = y.trailing_zeros();
                ((x as i8) >> shift) as u8 as u64
            }
            32 => {
                let shift = y.trailing_zeros();
                ((x as i32) >> shift) as u32 as u64
            }
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}
