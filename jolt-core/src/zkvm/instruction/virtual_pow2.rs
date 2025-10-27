use tracer::instruction::{virtual_pow2::VirtualPow2, RISCVCycle};

use super::{CircuitFlags, Flags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};
use crate::zkvm::{
    instruction::{InstructionFlags, NUM_INSTRUCTION_FLAGS},
    lookup_table::{pow2::Pow2Table, LookupTables},
};

impl<const XLEN: usize> InstructionLookup<XLEN> for VirtualPow2 {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        Some(Pow2Table.into())
    }
}

impl Flags for VirtualPow2 {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::AddOperands] = true;
        flags[CircuitFlags::WriteLookupOutputToRD] = true;
        flags[CircuitFlags::VirtualInstruction] = self.virtual_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC] =
            self.virtual_sequence_remaining.unwrap_or(0) != 0;
        flags[CircuitFlags::IsFirstInSequence] = self.is_first_in_sequence;
        flags[CircuitFlags::IsCompressed] = self.is_compressed;
        flags
    }

    fn instruction_flags(&self) -> [bool; NUM_INSTRUCTION_FLAGS] {
        let mut flags = [false; NUM_INSTRUCTION_FLAGS];
        flags[InstructionFlags::LeftOperandIsRs1Value] = true;
        flags[InstructionFlags::RightOperandIsImm] = true;
        flags[InstructionFlags::IsRdNotZero] = self.operands.rd != 0;
        flags
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualPow2> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        match XLEN {
            #[cfg(test)]
            8 => (self.register_state.rs1 as u8 as u64, 0),
            32 => (self.register_state.rs1 as u32 as u64, 0),
            64 => (self.register_state.rs1, 0),
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }

    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (0, x as u128 + y as u64 as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_lookup_output(&self) -> u64 {
        let y = LookupQuery::<XLEN>::to_lookup_index(self);
        match XLEN {
            #[cfg(test)]
            8 => 1u64 << ((y % 8) as u64),
            32 => 1u64 << ((y % 32) as u64),
            64 => 1u64 << ((y % 64) as u64),
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::*;
    use crate::zkvm::instruction::test::materialize_entry_test;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, VirtualPow2>();
    }
}
