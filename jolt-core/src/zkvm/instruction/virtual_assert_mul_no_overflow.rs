use super::{CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};
use crate::zkvm::lookup_table::{mul_no_overflow::MulNoOverflowTable, LookupTables};
use tracer::instruction::{virtual_assert_mul_no_overflow::VirtualAssertMulNoOverflow, RISCVCycle};

impl<const XLEN: usize> InstructionLookup<XLEN> for VirtualAssertMulNoOverflow {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        Some(MulNoOverflowTable.into())
    }
}

impl InstructionFlags for VirtualAssertMulNoOverflow {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::MultiplyOperands as usize] = true;
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

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualAssertMulNoOverflow> {
    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        // For signed multiplication, we need to handle the sign bits properly
        let result = (x as i64 as i128) * (y as i128);
        (0, result as u128)
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_instruction_inputs(&self) -> (u64, i128) {
        match XLEN {
            #[cfg(test)]
            8 => (
                self.register_state.rs1 as i8 as i64 as u64,
                self.register_state.rs2 as i8 as i128,
            ),
            32 => (
                self.register_state.rs1 as i32 as i64 as u64,
                self.register_state.rs2 as i32 as i128,
            ),
            64 => (self.register_state.rs1, self.register_state.rs2 as i128),
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }

    fn to_lookup_output(&self) -> u64 {
        let (rs1, rs2) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let result = (rs1 as i64 as i128) * (rs2 as i64 as i128);

        match XLEN {
            #[cfg(test)]
            8 => {
                let min = i8::MIN as i128;
                let max = i8::MAX as i128;
                (result >= min && result <= max) as u64
            }
            32 => {
                let min = i32::MIN as i128;
                let max = i32::MAX as i128;
                (result >= min && result <= max) as u64
            }
            64 => {
                let min = i64::MIN as i128;
                let max = i64::MAX as i128;
                (result >= min && result <= max) as u64
            }
            _ => panic!("Unsupported XLEN: {XLEN}"),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::zkvm::instruction::test::materialize_entry_test;
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, VirtualAssertMulNoOverflow>();
    }
}
