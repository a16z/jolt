use tracer::instruction::{virtual_muli::VirtualMULI, RISCVCycle};

use crate::zkvm::lookup_table::range_check::RangeCheckTable;
use crate::zkvm::lookup_table::LookupTables;

use super::{
    CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, U64OrI64, NUM_CIRCUIT_FLAGS,
};

impl<const XLEN: usize> InstructionLookup<XLEN> for VirtualMULI {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        Some(RangeCheckTable.into())
    }
}

impl InstructionFlags for VirtualMULI {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::LeftOperandIsRs1Value as usize] = true;
        flags[CircuitFlags::RightOperandIsImm as usize] = true;
        flags[CircuitFlags::MultiplyOperands as usize] = true;
        flags[CircuitFlags::WriteLookupOutputToRD as usize] = true;
        flags[CircuitFlags::InlineSequenceInstruction as usize] =
            self.inline_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
            self.inline_sequence_remaining.unwrap_or(0) != 0;
        flags[CircuitFlags::IsCompressed as usize] = self.is_compressed;
        flags
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualMULI> {
    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (0, x as u128 * y.as_u64() as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

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
            8 => (x as i8).wrapping_mul(y.as_i8()) as u8 as u64,
            32 => (x as i32).wrapping_mul(y.as_i32()) as u32 as u64,
            64 => (x as i64).wrapping_mul(y.as_i64()) as u64,
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
        materialize_entry_test::<Fr, VirtualMULI>();
    }
}
