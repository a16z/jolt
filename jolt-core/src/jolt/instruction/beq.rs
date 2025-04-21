use tracer::instruction::{beq::BEQ, RISCVCycle};

use crate::jolt::lookup_table::{equal::EqualTable, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for RISCVCycle<BEQ> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(EqualTable.into())
    }

    fn to_lookup_query(&self) -> (u64, u64) {
        (self.register_state.rs1, self.register_state.rs2)
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::to_lookup_query(self);
        (x == y).into()
    }
}

#[cfg(test)]
mod test {
    use crate::jolt::instruction::test::materialize_entry_test;

    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, BEQ>();
    }
}
