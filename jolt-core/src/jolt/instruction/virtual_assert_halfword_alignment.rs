use tracer::instruction::{
    virtual_assert_halfword_alignment::VirtualAssertHalfwordAlignment, RISCVCycle,
};

use crate::jolt::lookup_table::{halfword_alignment::HalfwordAlignmentTable, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE>
    for RISCVCycle<VirtualAssertHalfwordAlignment>
{
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(HalfwordAlignmentTable.into())
    }

    fn to_lookup_index(&self) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::to_lookup_query(self);
        x + y
    }

    fn to_lookup_query(&self) -> (u64, u64) {
        match WORD_SIZE {
            #[cfg(test)]
            8 => (
                self.register_state.rs1 as u8 as u64,
                self.instruction.operands.imm as u8 as u64,
            ),
            32 => (
                self.register_state.rs1 as u32 as u64,
                self.instruction.operands.imm as u32 as u64,
            ),
            64 => (
                self.register_state.rs1,
                self.instruction.operands.imm as u64,
            ),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn to_lookup_output(&self) -> u64 {
        (InstructionLookup::<WORD_SIZE>::to_lookup_index(self) % 2 == 0).into()
    }
}

#[cfg(test)]
mod test {
    use crate::jolt::instruction::test::materialize_entry_test;

    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, VirtualAssertHalfwordAlignment>();
    }
}
