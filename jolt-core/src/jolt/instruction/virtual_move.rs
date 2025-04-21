use tracer::instruction::{virtual_move::VirtualMove, RISCVCycle};

use crate::jolt::lookup_table::{range_check::RangeCheckTable, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for RISCVCycle<VirtualMove> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(RangeCheckTable.into())
    }

    fn to_lookup_index(&self) -> u64 {
        self.register_state.rs1
    }

    fn to_lookup_query(&self) -> (u64, u64) {
        (self.register_state.rs1, 0)
    }

    fn to_lookup_output(&self) -> u64 {
        match WORD_SIZE {
            #[cfg(test)]
            8 => (self.register_state.rs1 as u8).into(),
            32 => (self.register_state.rs1 as u32).into(),
            64 => self.register_state.rs1 as u64,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::jolt::instruction::test::materialize_entry_test;

    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, VirtualMove>();
    }
}
