use tracer::instruction::{virtual_pow2::VirtualPow2, RISCVCycle};

use crate::zkvm::lookup_table::{pow2::Pow2Table, LookupTables};

use super::{CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualPow2 {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(Pow2Table.into())
    }
}

impl InstructionFlags for VirtualPow2 {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::LeftOperandIsRs1Value as usize] = true;
        flags[CircuitFlags::RightOperandIsImm as usize] = true;
        flags[CircuitFlags::AddOperands as usize] = true;
        flags[CircuitFlags::WriteLookupOutputToRD as usize] = true;
        flags[CircuitFlags::InlineSequenceInstruction as usize] =
            self.inline_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
            self.inline_sequence_remaining.unwrap_or(0) != 0;
        flags[CircuitFlags::IsCompressed as usize] = self.is_compressed;
        flags
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for RISCVCycle<VirtualPow2> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        match WORD_SIZE {
            #[cfg(test)]
            8 => (self.register_state.rs1 as u8 as u64, 0),
            32 => (self.register_state.rs1 as u32 as u64, 0),
            64 => (self.register_state.rs1, 0),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = LookupQuery::<WORD_SIZE>::to_instruction_inputs(self);
        (0, x as u128 + y as u64 as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<WORD_SIZE>::to_lookup_operands(self).1
    }

    fn to_lookup_output(&self) -> u64 {
        let y = LookupQuery::<WORD_SIZE>::to_lookup_index(self);
        match WORD_SIZE {
            #[cfg(test)]
            8 => 1u64 << ((y % 8) as u64),
            32 => 1u64 << ((y % 32) as u64),
            64 => 1u64 << ((y % 64) as u64),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::zkvm::instruction::test::materialize_entry_test;

    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, VirtualPow2>();
    }
}
