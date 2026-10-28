//! Conversion from completed tracer cycles into final, program-bound rows.
//!
//! Raw cycles retain instruction-specific observations. The unified row
//! constructor owns validation and packing after source expansion.

use jolt_program::{execution::TraceError, preprocess::BytecodePreprocessing};
use jolt_riscv::{JoltTraceRow, RegisterRead, RegisterState, RegisterWrite};

use crate::instruction::Cycle;

/// Converts a completed final cycle, rejecting source-only instructions and
/// cycles absent from or differing from the program's expanded bytecode.
pub fn cycle_to_trace_row(
    cycle: &Cycle,
    bytecode: &BytecodePreprocessing,
) -> Result<JoltTraceRow, TraceError> {
    let instruction = cycle
        .instruction()
        .try_jolt_instruction_row()
        .map_err(TraceError::SourceOnlyCycle)?;
    let pc = bytecode
        .get_pc(&instruction)
        .ok_or(TraceError::MissingBytecodePc {
            address: instruction.address as u64,
            virtual_sequence_remaining: instruction.virtual_sequence_remaining,
        })?;
    let bytecode_pc = u32::try_from(pc).map_err(|_| TraceError::BytecodePcTooWide { pc })?;
    let expected = bytecode
        .bytecode
        .get(pc)
        .ok_or(TraceError::MissingBytecodePc {
            address: instruction.address as u64,
            virtual_sequence_remaining: instruction.virtual_sequence_remaining,
        })?;
    if *expected != instruction {
        return Err(TraceError::BytecodeInstructionMismatch {
            pc: bytecode_pc,
            expected: Box::new(*expected),
            actual: Box::new(instruction),
        });
    }
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
    fn executed_instruction_must_match_preprocessed_metadata() {
        // Frozen RV64 words: ADD x3,x1,x2 and individual operand/opcode changes.
        for (original, changed, rd, rs1, rs2, imm) in [
            (0x0020_81b3, 0x0020_8233, 4, 1, Some(2), 0),
            (0x0020_81b3, 0x0022_81b3, 3, 5, Some(2), 0),
            (0x0020_81b3, 0x0060_81b3, 3, 1, Some(6), 0),
            (0x0020_81b3, 0x4020_81b3, 3, 1, Some(2), 0),
            (0x0010_8193, 0x0020_8193, 3, 1, None, 2),
        ] {
            let original = Instruction::decode(original, TEXT, false).unwrap();
            let expected = original.try_jolt_instruction_row().unwrap();
            let bytecode =
                BytecodePreprocessing::preprocess(vec![expected], TEXT, RV64IMAC_JOLT).unwrap();
            let mut cpu = Cpu::new(Box::new(DummyTerminal::default()));
            let mut cycles = Vec::new();
            original.trace(&mut cpu, Some(&mut cycles));
            Instruction::decode(changed, TEXT, false)
                .unwrap()
                .trace(&mut cpu, Some(&mut cycles));
            assert_eq!(cycle_to_trace_row(&cycles[0], &bytecode).unwrap().pc(), 1);
            let TraceError::BytecodeInstructionMismatch {
                pc,
                expected: reported,
                actual,
            } = cycle_to_trace_row(&cycles[1], &bytecode).unwrap_err()
            else {
                panic!("expected bytecode mismatch");
            };
            assert_eq!(pc, 1);
            assert_eq!(*reported, expected);
            assert_eq!(actual.address, TEXT as usize);
            assert_eq!(actual.operands.rd, Some(rd));
            assert_eq!(actual.operands.rs1, Some(rs1));
            assert_eq!(actual.operands.rs2, rs2);
            assert_eq!(actual.operands.imm, imm);
            assert_ne!(*actual, expected);
        }

        let bytecode = preprocessing();
        for compressed in [true, false] {
            let mut instruction = Instruction::decode(ADD_WORD, TEXT, compressed).unwrap();
            instruction.set_is_first_in_sequence(!compressed);
            let mut cpu = Cpu::new(Box::new(DummyTerminal::default()));
            let mut cycles = Vec::new();
            instruction.trace(&mut cpu, Some(&mut cycles));
            assert!(matches!(
                cycle_to_trace_row(&cycles[0], &bytecode),
                Err(TraceError::BytecodeInstructionMismatch { .. })
            ));
            if compressed {
                let matching = BytecodePreprocessing::preprocess(
                    vec![instruction.try_jolt_instruction_row().unwrap()],
                    TEXT,
                    RV64IMAC_JOLT,
                )
                .unwrap();
                assert_eq!(cycle_to_trace_row(&cycles[0], &matching).unwrap().pc(), 1);
            }
        }
        assert_eq!(cycle_to_trace_row(&Cycle::NoOp, &bytecode).unwrap().pc(), 0);
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

        let err = cycle_to_trace_row(&cycle, &preprocessing()).unwrap_err();
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

        let err = cycle_to_trace_row(&cycles[0], &preprocessing).unwrap_err();
        assert!(matches!(
            err,
            TraceError::MissingBytecodePc {
                address,
                virtual_sequence_remaining: None,
            } if address == TEXT + 0x1000
        ));

        let mut instruction = program()[0];
        instruction.set_virtual_sequence_remaining(Some(1));
        let mut cycles = Vec::new();
        instruction.trace(&mut cpu, Some(&mut cycles));
        assert!(matches!(
            cycle_to_trace_row(&cycles[0], &preprocessing),
            Err(TraceError::MissingBytecodePc {
                virtual_sequence_remaining: Some(1),
                ..
            })
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
        let err = cycle_to_trace_row(&ld_cycle.into(), &preprocessing).unwrap_err();
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
        let err = cycle_to_trace_row(&sd_cycle.into(), &preprocessing).unwrap_err();
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
