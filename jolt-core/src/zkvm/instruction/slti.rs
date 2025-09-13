use tracer::instruction::{slti::SLTI, RISCVCycle};

use crate::zkvm::lookup_table::{signed_less_than::SignedLessThanTable, LookupTables};

use super::{
    CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, U64OrI64,
    NUM_CIRCUIT_FLAGS,
};

impl<const XLEN: usize> InstructionLookup<XLEN> for SLTI {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        Some(SignedLessThanTable.into())
    }
}

impl InstructionFlags for SLTI {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::WriteLookupOutputToRD as usize] = true;
        flags[CircuitFlags::LeftOperandIsRs1Value as usize] = true;
        flags[CircuitFlags::RightOperandIsImm as usize] = true;
        flags[CircuitFlags::InlineSequenceInstruction as usize] =
            self.inline_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
            self.inline_sequence_remaining.unwrap_or(0) != 0;
        flags[CircuitFlags::IsCompressed as usize] = self.is_compressed;
        flags
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<SLTI> {
    fn to_instruction_inputs(&self) -> (u64, U64OrI64) {
        match XLEN {
            #[cfg(test)]
            8 => (
                self.register_state.rs1 as u8 as u64,
                U64OrI64::Unsigned(self.instruction.operands.imm as u8 as u64),
            ),
            32 => (
                self.register_state.rs1 as u32 as u64,
                U64OrI64::Unsigned(self.instruction.operands.imm as u32 as u64),
            ),
            64 => (
                self.register_state.rs1,
                U64OrI64::Unsigned(self.instruction.operands.imm),
            ),
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        match XLEN {
            #[cfg(test)]
            8 => ((x as i8) < (y.as_i8())).into(),
            32 => ((x as i32) < (y.as_i32())).into(),
            64 => ((x as i64) < (y.as_i64())).into(),
            _ => panic!("{XLEN}-bit word size is unsupported"),
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
        materialize_entry_test::<Fr, SLTI>();
    }
}
