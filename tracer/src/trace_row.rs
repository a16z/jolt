//! Conversion from completed tracer cycles into final, program-bound rows.
//!
//! Raw cycles retain instruction-specific observations. The unified row
//! constructor owns validation and packing after source expansion.

use jolt_program::{execution::TraceError, preprocess::BytecodePCMapper};
use jolt_riscv::{JoltTraceRow, RegisterRead, RegisterState, RegisterWrite};

use crate::instruction::Cycle;

/// Converts a completed final cycle, rejecting source-only instructions and
/// cycles absent from the program's expanded bytecode.
pub fn cycle_to_trace_row(
    cycle: &Cycle,
    bytecode: &BytecodePCMapper,
) -> Result<JoltTraceRow, TraceError> {
    let instruction = cycle
        .instruction()
        .try_jolt_instruction_row()
        .map_err(TraceError::SourceOnlyCycle)?;
    let pc = bytecode
        .get_instruction_pc(&instruction)
        .ok_or(TraceError::MissingBytecodePc {
            address: instruction.address as u64,
            virtual_sequence_remaining: instruction.virtual_sequence_remaining,
        })?;
    let bytecode_pc = u32::try_from(pc).map_err(|_| TraceError::BytecodePcTooWide { pc })?;
    let registers = RegisterState {
        rs1: cycle
            .rs1_read()
            .map(|(register, value)| RegisterRead { register, value }),
        rs2: cycle
            .rs2_read()
            .map(|(register, value)| RegisterRead { register, value }),
        rd: cycle
            .rd_write()
            .map(|(register, pre_value, post_value)| RegisterWrite {
                register,
                pre_value,
                post_value,
            }),
    };
    Ok(JoltTraceRow::new(
        instruction,
        registers,
        cycle.ram_access(),
        bytecode_pc,
    )?)
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test-only assertions")]
mod tests {
    use super::*;
    use crate::emulator::cpu::Cpu;
    use crate::emulator::mmu::DRAM_BASE;
    use crate::emulator::terminal::DummyTerminal;
    use crate::instruction::{Instruction, RISCVCycle};
    use jolt_program::preprocess::BytecodePreprocessing;
    use jolt_riscv::{JoltInstructionKind, TraceRowError, RV64IMAC_JOLT};

    const TEXT: u64 = 0x8000_0000;

    const ADD_WORD: u32 = (2 << 20) | (1 << 15) | (3 << 7) | 0x33;
    const LD_WORD: u32 = (1 << 15) | (0b011 << 12) | (3 << 7) | 0x03;
    const SD_WORD: u32 = (2 << 20) | (1 << 15) | (0b011 << 12) | (8 << 7) | 0x23;

    fn program() -> Vec<Instruction> {
        vec![
            Instruction::decode(ADD_WORD, TEXT, false).unwrap(),
            Instruction::decode(LD_WORD, TEXT + 4, false).unwrap(),
            Instruction::decode(SD_WORD, TEXT + 8, false).unwrap(),
        ]
    }

    fn preprocessing() -> BytecodePreprocessing {
        let rows = program()
            .iter()
            .map(|instruction| instruction.try_jolt_instruction_row().unwrap())
            .collect();
        BytecodePreprocessing::preprocess(rows, TEXT, RV64IMAC_JOLT).unwrap()
    }

    fn traced_cycles() -> Vec<Cycle> {
        let mut cpu = Cpu::new(Box::new(DummyTerminal::default()));
        cpu.get_mut_mmu().init_memory(1 << 16);
        let base = DRAM_BASE + 0x100;
        cpu.write_register(1, base as i64);
        cpu.write_register(2, 7);
        cpu.mmu.store_doubleword(base, 0x1234_5678).unwrap();

        let mut cycles = Vec::new();
        for instruction in program() {
            instruction.trace(&mut cpu, Some(&mut cycles));
        }
        assert_eq!(cycles.len(), 3, "each instruction traces one cycle");
        cycles
    }

    #[test]
    fn source_only_cycles_are_rejected_at_the_phase_boundary() {
        // DIV never appears in final bytecode; its cycle must be refused.
        let div_word: u32 = (0x01 << 25) | (2 << 20) | (1 << 15) | (0b100 << 12) | (3 << 7) | 0x33;
        let instruction = Instruction::decode(div_word, TEXT, false).unwrap();
        let Instruction::DIV(div) = instruction else {
            panic!("expected DIV");
        };
        let cycle: Cycle = RISCVCycle {
            instruction: div,
            register_state: Default::default(),
            ram_access: Default::default(),
        }
        .into();

        let err = cycle_to_trace_row(&cycle, &preprocessing().pc_map).unwrap_err();
        assert!(matches!(err, TraceError::SourceOnlyCycle(_)));
    }

    #[test]
    fn unknown_addresses_report_missing_bytecode_pc() {
        let preprocessing = preprocessing();
        let stray = Instruction::decode(ADD_WORD, TEXT + 0x1000, false).unwrap();
        let mut cpu = Cpu::new(Box::new(DummyTerminal::default()));
        cpu.get_mut_mmu().init_memory(64);
        let mut cycles = Vec::new();
        stray.trace(&mut cpu, Some(&mut cycles));

        let err = cycle_to_trace_row(&cycles[0], &preprocessing.pc_map).unwrap_err();
        assert!(matches!(
            err,
            TraceError::MissingBytecodePc {
                address,
                virtual_sequence_remaining: None,
            } if address == TEXT + 0x1000
        ));
    }

    #[test]
    fn tampered_load_and_store_values_violate_the_memory_row_contract() {
        let preprocessing = preprocessing();
        let cycles = traced_cycles();

        let Cycle::LD(mut ld_cycle) = cycles[1] else {
            panic!("expected an LD cycle");
        };
        ld_cycle.ram_access.value ^= 1;
        let err = cycle_to_trace_row(&ld_cycle.into(), &preprocessing.pc_map).unwrap_err();
        assert!(
            matches!(
                &err,
                TraceError::InvalidRow(TraceRowError::MemoryValueMismatch { kind })
                    if *kind == JoltInstructionKind::LD
            ),
            "got {err:?}"
        );

        let Cycle::SD(mut sd_cycle) = cycles[2] else {
            panic!("expected an SD cycle");
        };
        sd_cycle.ram_access.post_value ^= 1;
        let err = cycle_to_trace_row(&sd_cycle.into(), &preprocessing.pc_map).unwrap_err();
        assert!(
            matches!(
                &err,
                TraceError::InvalidRow(TraceRowError::MemoryValueMismatch { kind })
                    if *kind == JoltInstructionKind::SD
            ),
            "got {err:?}"
        );
    }
}
