use tracer::instruction::{addc::ADDC, RISCVCycle};

use crate::zkvm::instruction::{InstructionFlags, NUM_INSTRUCTION_FLAGS};
use crate::zkvm::lookup_table::{range_check::RangeCheckTable, LookupTables};

use super::{CircuitFlags, Flags, InstructionLookup, LookupQuery, NUM_CIRCUIT_FLAGS};

impl<const XLEN: usize> InstructionLookup<XLEN> for ADDC {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        Some(RangeCheckTable.into())
    }
}

impl Flags for ADDC {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        flags[CircuitFlags::AddOperands] = true;
        flags[CircuitFlags::UsePreviousAux] = true;
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

impl<const XLEN: usize> LookupQuery<XLEN> for RISCVCycle<ADDC> {
    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        (
            0,
            x as u128 + y as u64 as u128 + self.instruction.prev_aux as u128,
        )
    }

    fn to_lookup_index(&self) -> u128 {
        LookupQuery::<XLEN>::to_lookup_operands(self).1
    }

    fn to_instruction_inputs(&self) -> (u64, i128) {
        match XLEN {
            #[cfg(test)]
            8 => (
                self.register_state.rs1 as u8 as u64,
                self.register_state.rs2 as u8 as i128,
            ),
            32 => (
                self.register_state.rs1 as u32 as u64,
                self.register_state.rs2 as u32 as i128,
            ),
            64 => (self.register_state.rs1, self.register_state.rs2 as i128),
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }

    fn to_lookup_output(&self) -> u64 {
        let (x, y) = LookupQuery::<XLEN>::to_instruction_inputs(self);
        match XLEN {
            #[cfg(test)]
            8 => (x as u8)
                .overflowing_add(y as u8)
                .0
                .overflowing_add(self.instruction.prev_aux as u8)
                .0
                .into(),
            32 => (x as u32)
                .overflowing_add(y as u32)
                .0
                .overflowing_add(self.instruction.prev_aux as u32)
                .0
                .into(),
            64 => {
                x.overflowing_add(y as u64)
                    .0
                    .overflowing_add(self.instruction.prev_aux)
                    .0
            }
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }
}

#[cfg(test)]
mod test {
    use std::boxed::Box;

    use common::constants::XLEN;
    use jolt_riscv::JoltInstructionRowData;
    use rand::{rngs::StdRng, SeedableRng};
    use tracer::emulator::cpu::Cpu;
    use tracer::emulator::terminal::DummyTerminal;
    use tracer::instruction::format::InstructionRegisterState;
    use tracer::instruction::RISCVTrace;

    use crate::zkvm::instruction::test::materialize_entry_test;

    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn materialize_entry() {
        materialize_entry_test::<Fr, ADDC>();
    }

    #[test]
    fn lookup_output_matches_trace() {
        let cycle: RISCVCycle<ADDC> = Default::default();
        let mut rng = StdRng::seed_from_u64(12345);
        for _ in 0..10000 {
            let random_cycle = cycle.random(&mut rng);
            let normalized_instr = random_cycle.instruction.jolt_instruction_row();
            let normalized_operands = normalized_instr.operands;

            let mut cpu = Cpu::new(Box::new(DummyTerminal::default()));
            cpu.set_last_lookup_high_word(random_cycle.instruction.prev_aux);
            if let Some(rs1_val) = random_cycle.register_state.rs1_value() {
                cpu.write_register(normalized_operands.rs1.unwrap() as usize, rs1_val as i64);
            }
            if let Some(rs2_val) = random_cycle.register_state.rs2_value() {
                cpu.write_register(normalized_operands.rs2.unwrap() as usize, rs2_val as i64);
            }

            random_cycle.instruction.trace(&mut cpu, None);
            let cpu_result = cpu.x[normalized_operands.rd.unwrap() as usize] as u64;
            let lookup_result = LookupQuery::<XLEN>::to_lookup_output(&random_cycle);
            assert_eq!(cpu_result, lookup_result, "{random_cycle:?}");
        }
    }
}
