use tracer::instruction::{virtual_movsign::VirtualMovsign, RISCVCycle};

use crate::jolt::lookup_table::{movsign::MovsignTable, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualMovsign {
    fn lookup_table() -> Option<LookupTables<WORD_SIZE>> {
        Some(MovsignTable.into())
    }

    fn lookup_query(cycle: &RISCVCycle<Self>) -> (u64, u64) {
        (cycle.register_state.rs1, 0)
    }

    fn lookup_entry(cycle: &RISCVCycle<Self>) -> u64 {
        let (x, _) = InstructionLookup::<WORD_SIZE>::lookup_query(cycle);
        match WORD_SIZE {
            #[cfg(test)]
            8 => {
                if x & (1 << 7) != 0 {
                    0xFF
                } else {
                    0
                }
            }
            32 => {
                if x & (1 << 31) != 0 {
                    0xFFFFFFFF
                } else {
                    0
                }
            }
            64 => {
                if x & (1 << 63) != 0 {
                    0xFFFFFFFFFFFFFFFF
                } else {
                    0
                }
            }
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}
