use tracer::instruction::{andi::ANDI, RISCVCycle};

use crate::jolt::lookup_table::{and::AndTable, LookupTables};

use super::{CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for ANDI {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(AndTable.into())
    }
}

impl InstructionFlags for ANDI {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::RightOperandIsImm as usize] = true;
        flags[CircuitFlags::WriteLookupOutputToRD as usize] = true;
        flags[CircuitFlags::Virtual as usize] = self.virtual_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdatePC as usize] =
            self.virtual_sequence_remaining.unwrap_or(0) != 0;
        flags
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for RISCVCycle<ANDI> {
    fn to_lookup_query(&self) -> (u64, u64) {
        (
            self.register_state.rs1,
            self.instruction.operands.imm as u64,
        )
    }

    #[cfg(test)]
    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<WORD_SIZE>::to_lookup_query(self);
        match WORD_SIZE {
            #[cfg(test)]
            8 => (x as u8 & y as u8) as u64,
            32 => (x as u32 & y as u32) as u64,
            64 => x & y,
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
        materialize_entry_test::<Fr, ANDI>();
    }
}
