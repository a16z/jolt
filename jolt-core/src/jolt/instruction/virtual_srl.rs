use tracer::instruction::{virtual_srl::VirtualSRL, RISCVCycle};

use crate::jolt::lookup_table::{virtual_srl::VirtualSRLTable, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for RISCVCycle<VirtualSRL> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(VirtualSRLTable.into())
    }

    fn to_lookup_query(&self) -> (u64, u64) {
        (self.register_state.rs1, self.register_state.rs2)
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::to_lookup_query(self);
        match WORD_SIZE {
            #[cfg(test)]
            8 => (x as u8 >> y.trailing_zeros()) as u64,
            32 => ((x as u32) >> y.trailing_zeros()) as u64,
            64 => x >> y.trailing_zeros(),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}
