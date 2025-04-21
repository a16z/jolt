use tracer::instruction::{lui::LUI, RISCVCycle};

use crate::jolt::lookup_table::{range_check::RangeCheckTable, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for RISCVCycle<LUI> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(RangeCheckTable.into())
    }

    fn to_lookup_index(&self) -> u64 {
        self.instruction.operands.imm as u64
    }

    fn to_lookup_query(&self) -> (u64, u64) {
        (self.instruction.operands.imm as u64, 0)
    }

    fn to_lookup_output(&self) -> u64 {
        match WORD_SIZE {
            #[cfg(test)]
            8 => (self.instruction.operands.imm as u8).into(),
            32 => (self.instruction.operands.imm as u32).into(),
            64 => self.instruction.operands.imm as u64,
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
        materialize_entry_test::<Fr, LUI>();
    }
}
