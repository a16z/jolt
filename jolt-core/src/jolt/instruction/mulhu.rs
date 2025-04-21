use tracer::instruction::{mulhu::MULHU, RISCVCycle};

use crate::jolt::lookup_table::{upper_word::UpperWordTable, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for RISCVCycle<MULHU> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(UpperWordTable.into())
    }

    fn to_lookup_index(&self) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::to_lookup_query(self);
        x * y
    }

    fn to_lookup_query(&self) -> (u64, u64) {
        match WORD_SIZE {
            #[cfg(test)]
            8 => (
                self.register_state.rs1 as u8 as u64,
                self.register_state.rs2 as u8 as u64,
            ),
            32 => (
                self.register_state.rs1 as u32 as u64,
                self.register_state.rs2 as u32 as u64,
            ),
            64 => (self.register_state.rs1, self.register_state.rs2),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::to_lookup_query(self);
        match WORD_SIZE {
            #[cfg(test)]
            8 => (x * y) >> 8,
            32 => (x * y) >> 32,
            64 => ((x as u128).wrapping_mul(y as u128) >> 64) as u64,
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
        materialize_entry_test::<Fr, MULHU>();
    }
}
