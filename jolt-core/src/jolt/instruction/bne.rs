use tracer::instruction::{bne::BNE, RISCVCycle};

use crate::jolt::lookup_table::{not_equal::NotEqualTable, LookupTables};

use super::{CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for BNE {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(NotEqualTable.into())
    }
}

impl InstructionFlags for BNE {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::Branch as usize] = true;
        flags[CircuitFlags::Virtual as usize] = self.virtual_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdatePC as usize] =
            self.virtual_sequence_remaining.unwrap_or(0) != 0;
        flags
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for RISCVCycle<BNE> {
    fn to_instruction_inputs(&self) -> (u64, u64) {
        (self.register_state.rs1, self.register_state.rs2)
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<WORD_SIZE>::to_instruction_inputs(self);
        (x != y).into()
    }
}

#[cfg(test)]
mod test {
    use crate::jolt::instruction::test::materialize_entry_test;

    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, BNE>();
    }
}
