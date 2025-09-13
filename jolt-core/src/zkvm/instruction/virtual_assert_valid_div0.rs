use tracer::instruction::{virtual_assert_valid_div0::VirtualAssertValidDiv0, RISCVCycle};

use crate::zkvm::lookup_table::{valid_div0::ValidDiv0Table, LookupTables};

use super::{
    CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, U64OrI64, NUM_CIRCUIT_FLAGS,
};

impl<const XLEN: usize> InstructionLookup<XLEN> for VirtualAssertValidDiv0 {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        Some(ValidDiv0Table.into())
    }
}

impl InstructionFlags for VirtualAssertValidDiv0 {
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

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualAssertValidDiv0> {
    fn to_instruction_inputs(&self) -> (u64, U64OrI64) {
        match XLEN {
            #[cfg(test)]
            8 => (
                self.register_state.rs1 as u8 as u64,
                U64OrI64::Unsigned(self.register_state.rs2 as u8 as u64),
            ),
            32 => (
                self.register_state.rs1 as u32 as u64,
                U64OrI64::Unsigned(self.register_state.rs2 as u32 as u64),
            ),
            64 => (
                self.register_state.rs1,
                U64OrI64::Unsigned(self.register_state.rs2),
            ),
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }

    fn to_lookup_output(&self) -> u64 {
        let (divisor, quotient) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        if divisor == 0 {
            match XLEN {
                32 => (quotient.as_u64() == u32::MAX as u64).into(),
                64 => (quotient.as_u64() == u64::MAX).into(),
                _ => panic!("Unsupported XLEN: {XLEN}"),
            }
        } else {
            1
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
        materialize_entry_test::<Fr, VirtualAssertValidDiv0>();
    }
}
