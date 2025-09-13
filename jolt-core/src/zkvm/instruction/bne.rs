use tracer::instruction::{bne::BNE, RISCVCycle};

use crate::zkvm::lookup_table::{not_equal::NotEqualTable, LookupTables};

use super::{CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, RightInputValue, NUM_CIRCUIT_FLAGS};

impl<const XLEN: usize> InstructionLookup<XLEN> for BNE {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        Some(NotEqualTable.into())
    }
}

impl InstructionFlags for BNE {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::LeftOperandIsRs1Value as usize] = true;
        flags[CircuitFlags::RightOperandIsRs2Value as usize] = true;
        flags[CircuitFlags::Branch as usize] = true;
        flags[CircuitFlags::InlineSequenceInstruction as usize] =
            self.inline_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
            self.inline_sequence_remaining.unwrap_or(0) != 0;
        flags[CircuitFlags::IsCompressed as usize] = self.is_compressed;
        flags
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<BNE> {
    fn to_instruction_inputs(&self) -> (u64, RightInputValue) {
        match XLEN {
            #[cfg(test)]
            8 => (
                self.register_state.rs1 as u8 as u64,
                RightInputValue::Unsigned(self.register_state.rs2 as u8 as u64),
            ),
            32 => (
                self.register_state.rs1 as u32 as u64,
                RightInputValue::Unsigned(self.register_state.rs2 as u32 as u64),
            ),
            64 => (self.register_state.rs1, RightInputValue::Unsigned(self.register_state.rs2)),
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (x != y.as_u64()).into()
    }
}

#[cfg(test)]
mod test {
    use crate::zkvm::instruction::test::materialize_entry_test;

    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, BNE>();
    }
}
