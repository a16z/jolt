//! Conversion from tracer [`Cycle`]s to the proof-facing [`JoltTraceRow`].
//!
//! `JoltTraceRow` lives in `jolt-riscv` and is deliberately tracer-free. This
//! module owns the one direction that needs tracer types: extracting a cycle's
//! dynamic witness values into a typed [`CapturedState`] and pairing it with the
//! cycle's final bytecode index. That keeps the dependency edge
//! `tracer -> jolt-riscv` (not the reverse).
//!
//! Because [`CapturedState`] collapses the equal/aliased memory columns into a
//! single field per class, this is also where the final memory-row contract is
//! verified: the cycle's separate raw `RamReadValue`/`RamWriteValue`/
//! `RdWriteValue`/`Rs2Value` are checked to actually collapse before the typed
//! state is built.

use jolt_program::preprocess::BytecodePreprocessing;
use jolt_riscv::{
    CapturedState, CircuitFlags, Flags, JoltInstruction, JoltInstructionKind, JoltInstructionRow,
    JoltTraceRow, LoadState, NonMemoryState, SourceInstructionKind, StoreState, TraceRowError,
};

use crate::instruction::{Cycle, RAMAccess};

/// Error raised while converting a tracer cycle into a [`JoltTraceRow`].
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CycleConversionError {
    /// The cycle is a source-only / structural cycle with no final Jolt row.
    #[error("cycle has no final Jolt instruction row: {0:?}")]
    SourceOnlyCycle(SourceInstructionKind),
    /// Bytecode preprocessing has no index for this cycle's instruction.
    #[error("no bytecode index for instruction at {address:#x} (virtual_sequence_remaining={virtual_sequence_remaining:?})")]
    MissingBytecodePc {
        address: u64,
        virtual_sequence_remaining: Option<u16>,
    },
    /// The bytecode index exceeds the compact `u32` trace-row storage budget.
    #[error("bytecode index {pc} exceeds u32 trace-row storage budget")]
    BytecodePcTooWide { pc: usize },
    /// The cycle's raw values do not collapse to the final memory-row contract
    /// for its class (e.g. a load whose RAM read value differs from its written
    /// register value).
    #[error("memory-row contract violated for {kind:?}: {detail}")]
    MemoryRowContractViolation {
        kind: JoltInstructionKind,
        detail: &'static str,
    },
    /// The row's components violate the trace-row layout constraints.
    #[error(transparent)]
    Row(TraceRowError),
}

/// Build the typed [`CapturedState`] from a cycle, verifying that the cycle's
/// raw values satisfy the final memory-row contract for its class.
fn captured_state(
    cycle: &Cycle,
    instruction: &JoltInstructionRow,
) -> Result<CapturedState, CycleConversionError> {
    let circuit_flags = JoltInstruction::try_from(*instruction)
        .map(|instruction| instruction.circuit_flags())
        .unwrap_or_default();
    let is_load = circuit_flags.get(CircuitFlags::Load);
    let is_store = circuit_flags.get(CircuitFlags::Store);
    let kind = instruction.instruction_kind;

    let rs1_value = cycle.rs1_read().map_or(0, |(_, value)| value);
    let rs2_value = cycle.rs2_read().map_or(0, |(_, value)| value);
    let rd = cycle.rd_write();
    let rd_pre_value = rd.map_or(0, |(_, pre, _)| pre);
    let rd_write_value = rd.map_or(0, |(_, _, post)| post);
    let (ram_read_value, ram_write_value) = match cycle.ram_access() {
        RAMAccess::Read(read) => (read.value, read.value),
        RAMAccess::Write(write) => (write.pre_value, write.post_value),
        RAMAccess::NoOp => (0, 0),
    };
    let ram_address = cycle.ram_access().address() as u64;

    if is_load {
        // RamReadValue = RamWriteValue = RdWriteValue; no rs2.
        if rs2_value != 0 {
            return Err(contract(kind, "load row has non-zero Rs2Value"));
        }
        if ram_read_value != rd_write_value || ram_write_value != rd_write_value {
            return Err(contract(
                kind,
                "load RamReadValue/RamWriteValue must equal RdWriteValue",
            ));
        }
        Ok(CapturedState::Load(LoadState {
            rs1_value,
            ram_address,
            rd_pre_value,
            rd_write_value,
        }))
    } else if is_store {
        // RamWriteValue = Rs2Value; no rd.
        if rd_pre_value != 0 || rd_write_value != 0 {
            return Err(contract(kind, "store row writes rd"));
        }
        if ram_write_value != rs2_value {
            return Err(contract(kind, "store RamWriteValue must equal Rs2Value"));
        }
        Ok(CapturedState::Store(StoreState {
            rs1_value,
            rs2_value,
            ram_read_value,
            ram_address,
        }))
    } else {
        if ram_address != 0 || ram_read_value != 0 || ram_write_value != 0 {
            return Err(contract(kind, "non-memory row carries RAM values"));
        }
        Ok(CapturedState::NonMemory(NonMemoryState {
            rs1_value,
            rs2_value,
            rd_pre_value,
            rd_write_value,
        }))
    }
}

#[inline]
fn contract(kind: JoltInstructionKind, detail: &'static str) -> CycleConversionError {
    CycleConversionError::MemoryRowContractViolation { kind, detail }
}

/// Convert a single tracer cycle into a [`JoltTraceRow`].
///
/// Rejects source-only cycles, so this is the phase-boundary check: every row
/// in the materialized trace is backed by a final Jolt instruction.
pub fn cycle_to_trace_row(
    cycle: &Cycle,
    bytecode: &BytecodePreprocessing,
) -> Result<JoltTraceRow, CycleConversionError> {
    let instruction = cycle
        .instruction()
        .try_jolt_instruction_row()
        .map_err(CycleConversionError::SourceOnlyCycle)?;
    let pc = bytecode
        .get_pc(&instruction)
        .ok_or(CycleConversionError::MissingBytecodePc {
            address: instruction.address as u64,
            virtual_sequence_remaining: instruction.virtual_sequence_remaining,
        })?;
    let bytecode_pc =
        u32::try_from(pc).map_err(|_| CycleConversionError::BytecodePcTooWide { pc })?;
    let state = captured_state(cycle, &instruction)?;
    JoltTraceRow::from_components(state, &instruction, bytecode_pc)
        .map_err(CycleConversionError::Row)
}

/// Materialize the full proof-facing trace once from a `Vec<Cycle>`.
pub fn build_trace_rows(
    trace: &[Cycle],
    bytecode: &BytecodePreprocessing,
) -> Result<Vec<JoltTraceRow>, CycleConversionError> {
    trace
        .iter()
        .map(|cycle| cycle_to_trace_row(cycle, bytecode))
        .collect()
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test-only assertions")]
mod tests {
    use super::*;
    use crate::emulator::cpu::Cpu;
    use crate::emulator::mmu::DRAM_BASE;
    use crate::emulator::terminal::DummyTerminal;
    use crate::instruction::Instruction;
    use jolt_riscv::RV64IMAC_JOLT;

    const TEXT: u64 = 0x8000_0000;

    // add x3, x1, x2 ; ld x3, 0(x1) ; sd x2, 8(x1)
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

    /// Trace the three-instruction program on a real CPU and return its cycles.
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
    fn cycles_convert_to_rows_with_their_bytecode_pc_and_captured_state() {
        let preprocessing = preprocessing();
        let cycles = traced_cycles();
        let rows = build_trace_rows(&cycles, &preprocessing).unwrap();

        let base = DRAM_BASE + 0x100;

        // ADD: non-memory row; pc 1 (index 0 is the injected no-op)
        assert_eq!(rows[0].pc(), 1);
        assert_eq!(rows[0].rs1_value(), base);
        assert_eq!(rows[0].rs2_value(), 7);
        assert_eq!(rows[0].rd_write_value(), base + 7);
        assert_eq!(rows[0].ram_address(), 0);

        // LD: load row collapses RamRead/RamWrite/RdWrite into one value
        assert_eq!(rows[1].pc(), 2);
        assert_eq!(rows[1].ram_address(), base);
        assert_eq!(rows[1].rd_write_value(), 0x1234_5678);
        assert_eq!(rows[1].ram_read_value(), 0x1234_5678);
        assert_eq!(rows[1].ram_write_value(), 0x1234_5678);

        // SD: store row writes rs2 into RAM, no rd
        assert_eq!(rows[2].pc(), 3);
        assert_eq!(rows[2].ram_address(), base + 8);
        assert_eq!(rows[2].rs2_value(), 7);
        assert_eq!(rows[2].ram_write_value(), 7);
        assert_eq!(rows[2].rd_write_value(), 0);
    }

    #[test]
    fn source_only_cycles_are_rejected_at_the_phase_boundary() {
        // DIV never appears in final bytecode; its cycle must be refused.
        let div_word: u32 = (0x01 << 25) | (2 << 20) | (1 << 15) | (0b100 << 12) | (3 << 7) | 0x33;
        let instruction = Instruction::decode(div_word, TEXT, false).unwrap();
        let Instruction::DIV(div) = instruction else {
            panic!("expected DIV");
        };
        let cycle: Cycle = crate::instruction::RISCVCycle {
            instruction: div,
            register_state: Default::default(),
            ram_access: Default::default(),
        }
        .into();

        let err = cycle_to_trace_row(&cycle, &preprocessing()).unwrap_err();
        assert!(matches!(err, CycleConversionError::SourceOnlyCycle(_)));
    }

    #[test]
    fn unknown_addresses_report_missing_bytecode_pc() {
        let preprocessing = preprocessing();
        // Same ADD but at an address outside the preprocessed bytecode
        let stray = Instruction::decode(ADD_WORD, TEXT + 0x1000, false).unwrap();
        let mut cpu = Cpu::new(Box::new(DummyTerminal::default()));
        cpu.get_mut_mmu().init_memory(64);
        let mut cycles = Vec::new();
        stray.trace(&mut cpu, Some(&mut cycles));

        let err = cycle_to_trace_row(&cycles[0], &preprocessing).unwrap_err();
        assert_eq!(
            err,
            CycleConversionError::MissingBytecodePc {
                address: TEXT + 0x1000,
                virtual_sequence_remaining: None,
            }
        );
    }

    #[test]
    fn tampered_load_and_store_values_violate_the_memory_row_contract() {
        let preprocessing = preprocessing();
        let cycles = traced_cycles();

        // Load whose RAM value disagrees with the register write
        let Cycle::LD(mut ld_cycle) = cycles[1] else {
            panic!("expected an LD cycle");
        };
        ld_cycle.ram_access.value ^= 1;
        let err = cycle_to_trace_row(&ld_cycle.into(), &preprocessing).unwrap_err();
        assert!(
            matches!(
                &err,
                CycleConversionError::MemoryRowContractViolation { detail, .. }
                    if detail.contains("RamReadValue/RamWriteValue must equal RdWriteValue")
            ),
            "got {err:?}"
        );

        // Store whose written value disagrees with rs2
        let Cycle::SD(mut sd_cycle) = cycles[2] else {
            panic!("expected an SD cycle");
        };
        sd_cycle.ram_access.post_value ^= 1;
        let err = cycle_to_trace_row(&sd_cycle.into(), &preprocessing).unwrap_err();
        assert!(
            matches!(
                &err,
                CycleConversionError::MemoryRowContractViolation { detail, .. }
                    if detail.contains("RamWriteValue must equal Rs2Value")
            ),
            "got {err:?}"
        );
    }
}
