use tracer::instruction::{virtual_srai::VirtualSRAI, RISCVCycle};

use crate::zkvm::lookup_table::{virtual_sra::VirtualSRATable, LookupTables};

use super::{
    CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, U64OrI64, NUM_CIRCUIT_FLAGS,
};

impl<const XLEN: usize> InstructionLookup<XLEN> for VirtualSRAI {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        Some(VirtualSRATable.into())
    }
}

impl InstructionFlags for VirtualSRAI {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::LeftOperandIsRs1Value as usize] = true;
        flags[CircuitFlags::RightOperandIsImm as usize] = true;
        flags[CircuitFlags::WriteLookupOutputToRD as usize] = true;
        flags[CircuitFlags::InlineSequenceInstruction as usize] =
            self.inline_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
            self.inline_sequence_remaining.unwrap_or(0) != 0;
        flags[CircuitFlags::IsCompressed as usize] = self.is_compressed;
        flags
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualSRAI> {
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
        use crate::utils::lookup_bits::LookupBits;
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let mut x = LookupBits::new(x as u128, XLEN);
        let mut y = LookupBits::new(y.as_i128() as u128, XLEN);

        let sign_bit = if x.leading_ones() == 0 { 0 } else { 1 };
        let mut entry = 0;
        let mut sign_extension = 0;
        for i in 0..XLEN {
            let x_i = x.pop_msb() as u64;
            let y_i = y.pop_msb() as u64;
            entry *= 1 + y_i;
            entry += x_i * y_i;
            if i != 0 {
                sign_extension += (1 << i) * (1 - y_i);
            }
        }
        entry + sign_bit * sign_extension
    }
}

#[cfg(test)]
mod test {
    use crate::zkvm::instruction::test::materialize_entry_test;

    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, VirtualSRAI>();
    }
}
