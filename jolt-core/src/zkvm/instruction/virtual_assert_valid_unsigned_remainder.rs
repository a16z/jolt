use tracer::instruction::{
    virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder, RISCVCycle,
};

use crate::zkvm::lookup_table::{
    valid_unsigned_remainder::ValidUnsignedRemainderTable, LookupTables,
};

use super::{CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, RightInputValue, NUM_CIRCUIT_FLAGS};

impl<const XLEN: usize> InstructionLookup<XLEN> for VirtualAssertValidUnsignedRemainder {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        Some(ValidUnsignedRemainderTable.into())
    }
}

impl InstructionFlags for VirtualAssertValidUnsignedRemainder {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::Assert as usize] = true;
        flags[CircuitFlags::LeftOperandIsRs1Value as usize] = true;
        flags[CircuitFlags::RightOperandIsRs2Value as usize] = true;
        flags[CircuitFlags::InlineSequenceInstruction as usize] =
            self.inline_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
            self.inline_sequence_remaining.unwrap_or(0) != 0;
        flags[CircuitFlags::IsCompressed as usize] = self.is_compressed;
        flags
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualAssertValidUnsignedRemainder> {
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
        let (remainder, divisor) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let divisor_u64 = divisor.as_u64();
        (divisor_u64 == 0 || remainder < divisor_u64).into()
    }
}

#[cfg(test)]
mod test {
    use crate::zkvm::instruction::test::materialize_entry_test;

    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, VirtualAssertValidUnsignedRemainder>();
    }
}
