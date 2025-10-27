use tracer::instruction::{
    virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder,
    RISCVCycle,
};

use super::{CircuitFlags, Flags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};
use crate::zkvm::{
    instruction::{InstructionFlags, NUM_INSTRUCTION_FLAGS},
    lookup_table::{valid_unsigned_remainder::ValidUnsignedRemainderTable, LookupTables},
};

impl<const XLEN: usize> InstructionLookup<XLEN> for VirtualAssertValidUnsignedRemainder {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        Some(ValidUnsignedRemainderTable.into())
    }
}

impl Flags for VirtualAssertValidUnsignedRemainder {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::Assert] = true;
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
        flags[InstructionFlags::RightOperandIsRs2Value] = true;
        flags
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualAssertValidUnsignedRemainder> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        match XLEN {
            #[cfg(test)]
            8 => (
                self.register_state.rs1 as u8 as u64,
                self.register_state.rs2 as u8 as i128,
            ),
            32 => (
                self.register_state.rs1 as u32 as u64,
                self.register_state.rs2 as u32 as i128,
            ),
            64 => (self.register_state.rs1, self.register_state.rs2 as i128),
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }

    fn to_lookup_output(&self) -> u64 {
        let (remainder, divisor) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (divisor == 0 || remainder < divisor as u64).into()
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::*;
    use crate::zkvm::instruction::test::materialize_entry_test;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, VirtualAssertValidUnsignedRemainder>();
    }
}
