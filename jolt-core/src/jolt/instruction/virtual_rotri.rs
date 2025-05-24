use tracer::instruction::{virtual_rotri::VirtualROTRI, RISCVCycle};

use crate::jolt::lookup_table::{virtual_rotr::VirtualRotrTable, LookupTables};
use crate::subprotocols::sparse_dense_shout::LookupBits;

use super::{CircuitFlags, InstructionFlags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualROTRI {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        Some(VirtualRotrTable.into())
    }
}

impl InstructionFlags for VirtualROTRI {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::LeftOperandIsRs1Value as usize] = true;
        flags[CircuitFlags::RightOperandIsImm as usize] = true;
        flags[CircuitFlags::WriteLookupOutputToRD as usize] = true;
        flags[CircuitFlags::Virtual as usize] = self.virtual_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdatePC as usize] =
            self.virtual_sequence_remaining.unwrap_or(0) != 0;
        flags
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for RISCVCycle<VirtualROTRI> {
    fn to_instruction_inputs(&self) -> (u64, i64) {
        (
            self.register_state.rs1,
            self.instruction.operands.imm as i64,
        )
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<WORD_SIZE>::to_instruction_inputs(self);

        let mut x_bits = LookupBits::new(x as u64, WORD_SIZE);
        let mut y_bits = LookupBits::new(y as u64, WORD_SIZE);

        // First collect all bits to determine rotation amount
        let mut x_arr = [0u8; 32]; // Max WORD_SIZE
        let mut y_arr = [0u8; 32];
        for i in 0..WORD_SIZE {
            x_arr[i] = x_bits.pop_msb();
            y_arr[i] = y_bits.pop_msb();
        }

        // Count trailing zeros in y (from LSB side)
        let mut rotation = 0;
        for i in (0..WORD_SIZE).rev() {
            if y_arr[i] == 0 {
                rotation += 1;
            } else {
                break;
            }
        }

        // Build rotated result bit by bit from MSB
        let mut entry = 0;
        for i in 0..WORD_SIZE {
            entry <<= 1;
            // For ROTR by k: bit at position i comes from position (i + k) % WORD_SIZE
            let src_idx = (i + rotation) % WORD_SIZE;
            entry |= x_arr[src_idx] as u64;
        }

        entry
    }
}

#[cfg(test)]
mod test {
    use crate::jolt::instruction::test::materialize_entry_test;

    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, VirtualROTRI>();
    }
}
