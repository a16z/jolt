use crate::zkvm::instruction::{InstructionFlags, NUM_INSTRUCTION_FLAGS};
use tracer::instruction::{virtual_xor_rot::VirtualXORROT, RISCVCycle};

use crate::zkvm::lookup_table::LookupTables;

use super::{CircuitFlags, Flags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const XLEN: usize> InstructionLookup<XLEN> for VirtualXORROT {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        Some(LookupTables::xor_rot(self.operands.rotation))
    }
}

impl Flags for VirtualXORROT {
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
        flags[InstructionFlags::RightOperandIsRs2Value] = true;
        flags
    }
}

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<VirtualXORROT> {
    fn to_instruction_inputs(&self) -> (u64, i128) {
        match XLEN {
            #[cfg(test)]
            8 => (
                self.register_state.rs1 as u8 as u64,
                self.register_state.rs2 as u8 as i128,
            ),
            64 => (self.register_state.rs1, self.register_state.rs2 as i128),
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        let rotation = self.instruction.operands.rotation;
        match XLEN {
            #[cfg(test)]
            8 => ((x as u8) ^ (y as u8)).rotate_right(rotation).into(),
            64 => (x ^ (y as u64)).rotate_right(rotation),
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::zkvm::instruction::test::lookup_output_matches_trace_test;
    use common::constants::XLEN;
    use rand::{rngs::StdRng, SeedableRng};

    /// Each random cycle selects its own table through the rotation
    /// immediate, so the shared per-table helper does not apply.
    #[test]
    fn materialize_entry() {
        let cycle: RISCVCycle<VirtualXORROT> = Default::default();
        let mut rng = StdRng::seed_from_u64(12345);
        for _ in 0..10000 {
            let random_cycle = cycle.random(&mut rng);
            let table = InstructionLookup::<XLEN>::lookup_table(&random_cycle.instruction).unwrap();
            assert_eq!(
                LookupQuery::<XLEN>::to_lookup_output(&random_cycle),
                table.materialize_entry(LookupQuery::<XLEN>::to_lookup_index(&random_cycle)),
                "{:?}",
                random_cycle.instruction
            );
        }
    }

    #[test]
    fn lookup_output_matches_trace() {
        lookup_output_matches_trace_test::<VirtualXORROT>();
    }
}
