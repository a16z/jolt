use crate::zkvm::instruction::{InstructionFlags, NUM_INSTRUCTION_FLAGS};
use tracer::instruction::{virtual_srli::VirtualSRLI, RISCVCycle};

use crate::zkvm::lookup_table::{virtual_srl::VirtualSRLTable, LookupTables};

use super::{CircuitFlags, Flags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const XLEN: usize> InstructionLookup<XLEN> for VirtualSRLI {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        Some(VirtualSRLTable.into())
    }
}

impl Flags for VirtualSRLI {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
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
        flags[InstructionFlags::IsRdZero] = self.operands.rd == 0;
        flags
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualSRLI> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        match XLEN {
            #[cfg(test)]
            8 => (
                self.register_state.rs1 as u8 as u64,
                self.instruction.operands.imm as u8 as i128,
            ),
            32 => (
                self.register_state.rs1 as u32 as u64,
                self.instruction.operands.imm as u32 as i128,
            ),
            64 => (
                self.register_state.rs1,
                self.instruction.operands.imm as i128,
            ),
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }

    fn to_lookup_output(&self) -> u64 {
        use crate::utils::lookup_bits::LookupBits;
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let mut x = LookupBits::new(x as u128, XLEN);
        let mut y = LookupBits::new(y as u128, XLEN);

        let mut entry = 0;
        for _ in 0..XLEN {
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
