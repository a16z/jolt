use tracer::instruction::{virtual_srli::VirtualSRLI, RISCVCycle};

use crate::zkvm::lookup_table::{virtual_srl::VirtualSRLTable, LookupTables};

use super::{CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualSRLI {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(VirtualSRLTable.into())
    }
}

impl InstructionFlags for VirtualSRLI {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::LeftOperandIsRs1Value as usize] = true;
        flags[CircuitFlags::RightOperandIsImm as usize] = true;
        flags[CircuitFlags::WriteLookupOutputToRD as usize] = true;
        flags[CircuitFlags::InlineSequenceInstruction as usize] =
            self.virtual_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
            self.virtual_sequence_remaining.unwrap_or(0) != 0;
        flags
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for RISCVCycle<VirtualSRLI> {
    fn to_instruction_inputs(&self) -> (u64, i64) {
        match WORD_SIZE {
            #[cfg(test)]
            8 => (
                self.register_state.rs1 as u8 as u64,
                self.instruction.operands.imm as u8 as i64,
            ),
            32 => (
                self.register_state.rs1 as u32 as u64,
                self.instruction.operands.imm as u32 as i64,
            ),
            64 => (
                self.register_state.rs1,
                self.instruction.operands.imm as i64,
            ),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn to_lookup_output(&self) -> u64 {
        use crate::utils::lookup_bits::LookupBits;
        let (x, y) = LookupQuery::<WORD_SIZE>::to_instruction_inputs(self);
        let mut x = LookupBits::new(x as u128, WORD_SIZE);
        let mut y = LookupBits::new(y as u128, WORD_SIZE);

        let mut entry = 0;
        for _ in 0..WORD_SIZE {
            let x_i = x.pop_msb();
            let y_i = y.pop_msb();
            entry *= 1 + y_i as u64;
            entry += (x_i * y_i) as u64;
        }
        entry
    }
}

#[cfg(test)]
mod test {
    use crate::zkvm::instruction::test::materialize_entry_test;

    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, VirtualSRLI>();
    }
}
