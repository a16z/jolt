use tracer::instruction::{virtual_sra::VirtualSRA, RISCVCycle};

use crate::jolt::lookup_table::{virtual_sra::VirtualSRATable, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for RISCVCycle<VirtualSRA> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(VirtualSRATable.into())
    }

    fn to_lookup_query(&self) -> (u64, u64) {
        (self.register_state.rs1, self.register_state.rs2)
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::to_lookup_query(self);
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
