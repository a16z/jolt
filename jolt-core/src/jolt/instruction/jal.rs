use tracer::instruction::{jal::JAL, RISCVCycle};

use crate::jolt::lookup_table::{range_check::RangeCheckTable, LookupTables};

use super::{CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for JAL {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(RangeCheckTable.into())
    }
}

impl InstructionFlags for JAL {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::LeftOperandIsPC as usize] = true;
        flags[CircuitFlags::RightOperandIsImm as usize] = true;
        flags[CircuitFlags::Jump as usize] = true;
        flags[CircuitFlags::Virtual as usize] = self.virtual_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdatePC as usize] =
            self.virtual_sequence_remaining.unwrap_or(0) != 0;
        flags
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for RISCVCycle<JAL> {
    fn to_lookup_index(&self) -> u64 {
        let (pc, imm) = LookupQuery::<WORD_SIZE>::to_lookup_query(self);
        pc + imm
    }

    fn to_lookup_query(&self) -> (u64, u64) {
        match WORD_SIZE {
            #[cfg(test)]
            8 => (
                self.instruction.address,
                self.instruction.operands.imm as u8 as u64,
            ),
            32 => (
                self.instruction.address,
                self.instruction.operands.imm as u32 as u64,
            ),
            64 => (
                self.instruction.address,
                self.instruction.operands.imm as u64,
            ),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    #[cfg(test)]
    fn to_lookup_output(&self) -> u64 {
        let (pc, imm) = LookupQuery::<WORD_SIZE>::to_lookup_query(self);
        match WORD_SIZE {
            #[cfg(test)]
            8 => (pc as u8).overflowing_add(imm as u8).0.into(),
            32 => (pc as u32).overflowing_add(imm as u32).0.into(),
            64 => pc.overflowing_add(imm).0,
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
        materialize_entry_test::<Fr, JAL>();
    }
}
