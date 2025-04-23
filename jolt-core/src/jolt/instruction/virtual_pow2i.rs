use tracer::instruction::{virtual_pow2i::VirtualPow2I, RISCVCycle};

use crate::jolt::lookup_table::{pow2::Pow2Table, LookupTables};

use super::{CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualPow2I {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(Pow2Table.into())
    }
}

impl InstructionFlags for VirtualPow2I {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::RightOperandIsImm as usize] = self.virtual_sequence_remaining.is_some();
        flags[CircuitFlags::WriteLookupOutputToRD as usize] = true;
        flags[CircuitFlags::Virtual as usize] = self.virtual_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdatePC as usize] =
            self.virtual_sequence_remaining.unwrap_or(0) != 0;
        flags
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for RISCVCycle<VirtualPow2I> {
    fn to_lookup_query(&self) -> (u64, u64) {
        (self.instruction.operands.imm as u64, 0)
    }

    fn to_lookup_index(&self) -> u64 {
        self.instruction.operands.imm as u64
    }

    #[cfg(test)]
    fn to_lookup_output(&self) -> u64 {
        let (x, _) = LookupQuery::<WORD_SIZE>::to_lookup_query(self);
        match WORD_SIZE {
            #[cfg(test)]
            8 => 1u64 << (x % 8),
            32 => 1u64 << (x % 32),
            64 => 1u64 << (x % 64),
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
        materialize_entry_test::<Fr, VirtualPow2I>();
    }
}
