use std::path::PathBuf;

use jolt_program::execution::{
    ChunkedExecutionBackend, ExecutionBackend, ExecutionSummary, JoltProgram, MemoryImage,
    OwnedTrace, RamAccess as ProgramRamAccess, RamRead as ProgramRamRead,
    RamWrite as ProgramRamWrite, RegisterRead, RegisterState, RegisterWrite, TraceError,
    TraceInputs, TraceOutput, TraceRow,
};
use jolt_riscv::JoltInstructionRow;

use crate::emulator::cpu::AdviceTape;
use crate::instruction::{Cycle, RAMAccess};
use crate::{Checkpoint, CheckpointingTracer, GeneralizedLazyTraceIter, LazyTracer};

#[derive(Default, Debug, Clone)]
pub struct TracerBackend {
    pub elf_path: Option<PathBuf>,
}

impl TracerBackend {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_elf_path(elf_path: PathBuf) -> Self {
        Self {
            elf_path: Some(elf_path),
        }
    }
}

impl ExecutionBackend for TracerBackend {
    type Trace = OwnedTrace;

    fn trace(
        &mut self,
        program: &JoltProgram,
        inputs: TraceInputs,
    ) -> Result<TraceOutput<Self::Trace>, TraceError> {
        if program.elf_bytes().is_empty() {
            return Err(TraceError::MissingElfBytes);
        }

        let (_lazy_trace, cycles, final_memory, device, advice_tape) = crate::trace(
            program.elf_bytes(),
            self.elf_path.as_ref(),
            &inputs.inputs,
            &inputs.untrusted_advice,
            &inputs.trusted_advice,
            &inputs.memory_config,
            inputs.advice_tape.map(AdviceTape::from_bytes),
        );

        let rows = cycles
            .into_iter()
            .map(trace_row_from_cycle)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(TraceOutput::new(
            OwnedTrace::new(rows),
            device,
            Some(MemoryImage {
                bytes: final_memory.materialized_nonzero_bytes(),
            }),
        )
        .with_advice_tape(Some(advice_tape.into_bytes())))
    }
}

/// A [`Checkpoint`] wrapped for the chunked-execution contract.
///
/// Replay only reads the inner checkpoint (via `Clone`); mutation happens on
/// the per-replay clone.
pub struct TracerChunkCheckpoint(Checkpoint);

// SAFETY: `Checkpoint` is already `Send` (owned data only); shared references
// are only used to clone it, and it has no interior mutability, so concurrent
// reads are safe.
unsafe impl Sync for TracerChunkCheckpoint {}

impl ChunkedExecutionBackend for TracerBackend {
    type Checkpoint = TracerChunkCheckpoint;

    fn execute(
        &mut self,
        program: &JoltProgram,
        inputs: TraceInputs,
        chunk_size: usize,
    ) -> Result<ExecutionSummary<Self::Checkpoint>, TraceError> {
        if program.elf_bytes().is_empty() {
            return Err(TraceError::MissingElfBytes);
        }
        if chunk_size == 0 {
            return Err(TraceError::Backend("chunk_size must be nonzero"));
        }

        let mut iter = GeneralizedLazyTraceIter::new(CheckpointingTracer::new(
            crate::setup_emulator_with_backtraces(
                program.elf_bytes(),
                self.elf_path.as_ref(),
                &inputs.inputs,
                &inputs.untrusted_advice,
                &inputs.trusted_advice,
                &inputs.memory_config,
                inputs.advice_tape.map(AdviceTape::from_bytes),
            ),
        ));
        iter.lazy_tracer.start_saving_checkpoints();

        let mut checkpoints = Vec::new();
        let mut trace_len = 0usize;
        loop {
            let mut rows_in_chunk = 0usize;
            while rows_in_chunk < chunk_size {
                // The fast pass drops the produced cycles instead of
                // materializing rows; the wrapped interpreter still constructs
                // them per tick (invariant: reference machinery untouched).
                match iter.next() {
                    Some(_) => rows_in_chunk += 1,
                    None => break,
                }
            }
            trace_len += rows_in_chunk;
            let checkpoint = iter.lazy_tracer.save_checkpoint();
            if rows_in_chunk > 0 {
                checkpoints.push(TracerChunkCheckpoint(checkpoint));
            }
            if iter.lazy_tracer.has_terminated() {
                break;
            }
        }

        let advice_tape = iter.lazy_tracer.take_advice_tape().into_bytes();
        let final_memory = iter
            .lazy_tracer
            .final_memory_state
            .take()
            .map(|memory| MemoryImage {
                bytes: memory.materialized_nonzero_bytes(),
            });
        let device = iter.lazy_tracer.get_jolt_device();

        Ok(ExecutionSummary {
            checkpoints,
            trace_len,
            device,
            final_memory,
            advice_tape: Some(advice_tape),
        })
    }

    fn replay_chunk(&self, checkpoint: &Self::Checkpoint) -> Result<Self::Trace, TraceError> {
        let mut replay = checkpoint.0.clone();
        let mut rows = Vec::new();
        while !replay.has_terminated() {
            let Some(cycle) = replay.lazy_step_cycle() else {
                break;
            };
            rows.push(trace_row_from_cycle(cycle)?);
        }
        Ok(OwnedTrace::new(rows))
    }
}

fn trace_row_from_cycle(cycle: Cycle) -> Result<TraceRow, TraceError> {
    Ok(TraceRow {
        instruction: jolt_instruction_row(&cycle)?,
        registers: register_state(&cycle),
        ram_access: cycle.ram_access().into(),
        #[cfg(feature = "field-inline")]
        field_inline: cycle.field_inline_trace().map(Into::into),
    })
}

fn jolt_instruction_row(cycle: &Cycle) -> Result<JoltInstructionRow, TraceError> {
    let instruction = cycle.instruction();
    instruction
        .try_jolt_instruction_row()
        .map_err(|_| TraceError::Backend("execution trace contained a source-only instruction"))
}

fn register_state(cycle: &Cycle) -> RegisterState {
    RegisterState {
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
    }
}

impl From<RAMAccess> for ProgramRamAccess {
    fn from(access: RAMAccess) -> Self {
        match access {
            RAMAccess::Read(read) => Self::Read(ProgramRamRead {
                address: read.address,
                value: read.value,
            }),
            RAMAccess::Write(write) => Self::Write(ProgramRamWrite {
                address: write.address,
                pre_value: write.pre_value,
                post_value: write.post_value,
            }),
            RAMAccess::NoOp => Self::NoOp,
        }
    }
}

#[cfg(test)]
mod chunked_tests {
    use super::*;
    use crate::test_utils::build_muldiv_guest;
    use common::jolt_device::MemoryConfig;
    use jolt_program::execution::build_jolt_program;

    const INPUTS: [u8; 6] = [0xbd, 0xaa, 0xde, 0x5, 0x11, 0x5c];

    fn muldiv_setup() -> (JoltProgram, TraceInputs) {
        let elf = build_muldiv_guest();
        let memory_config = MemoryConfig {
            program_size: Some(elf.len() as u64),
            ..Default::default()
        };
        let program = build_jolt_program(&elf).expect("failed to build Jolt program");
        let inputs = TraceInputs::new(INPUTS.to_vec(), Vec::new(), Vec::new(), memory_config);
        (program, inputs)
    }

    /// Invariant 3 of `specs/x86-tracer-backend.md` for the reference backend:
    /// the concatenation of `replay_chunk` outputs equals the eager `trace()`
    /// row stream for every chunk size, including degenerate ones, regardless
    /// of replay order.
    #[test]
    fn chunked_execution_composes_to_eager_trace() {
        let (program, inputs) = muldiv_setup();
        let mut backend = TracerBackend::new();
        let eager = backend
            .trace(&program, inputs.clone())
            .expect("eager trace failed");
        let eager_rows = eager.trace.rows();
        assert!(!eager_rows.is_empty());

        // Chunk size 1 forces checkpoint marks inside multi-row expansions.
        for chunk_size in [1usize, 100, 1 << 18, eager_rows.len() + 1] {
            let mut backend = TracerBackend::new();
            let summary = backend
                .execute(&program, inputs.clone(), chunk_size)
                .expect("execute failed");

            assert_eq!(
                summary.trace_len,
                eager_rows.len(),
                "chunk_size {chunk_size}"
            );
            assert_eq!(
                summary.checkpoints.len(),
                eager_rows.len().div_ceil(chunk_size),
                "chunk_size {chunk_size}"
            );
            assert_eq!(summary.device, eager.device, "chunk_size {chunk_size}");
            assert_eq!(
                summary.final_memory, eager.final_memory,
                "chunk_size {chunk_size}"
            );
            assert_eq!(
                summary.advice_tape, eager.advice_tape,
                "chunk_size {chunk_size}"
            );

            // Replay in reverse order to exercise order-independence.
            let mut replayed: Vec<Vec<TraceRow>> = summary
                .checkpoints
                .iter()
                .rev()
                .map(|checkpoint| {
                    backend
                        .replay_chunk(checkpoint)
                        .expect("replay failed")
                        .into_rows()
                })
                .collect();
            replayed.reverse();

            let last = replayed.len() - 1;
            for (i, rows) in replayed.iter().enumerate() {
                if i < last {
                    assert_eq!(rows.len(), chunk_size, "chunk_size {chunk_size}, chunk {i}");
                }
            }
            let concat: Vec<TraceRow> = replayed.into_iter().flatten().collect();
            assert_eq!(concat.as_slice(), eager_rows, "chunk_size {chunk_size}");
        }
    }

    /// Advice-tape plumbing: a seeded tape reaches the emulator and the
    /// populated tape is captured on output, for both the eager and the
    /// chunked path (muldiv never consumes the tape, so it round-trips
    /// unchanged).
    #[test]
    fn advice_tape_seeds_and_captures() {
        let (program, inputs) = muldiv_setup();
        let seeded = vec![1u8, 2, 3, 4, 5];
        let inputs = inputs.with_advice_tape(Some(seeded.clone()));

        let mut backend = TracerBackend::new();
        let output = backend
            .trace(&program, inputs.clone())
            .expect("eager trace failed");
        assert_eq!(output.advice_tape, Some(seeded.clone()));

        let summary = backend
            .execute(&program, inputs, 100)
            .expect("execute failed");
        assert_eq!(summary.advice_tape, Some(seeded));
    }
}

#[cfg(test)]
#[cfg_attr(feature = "field-inline", expect(clippy::unwrap_used))]
mod tests {
    #[cfg(feature = "field-inline")]
    use crate::{
        emulator::{cpu::Cpu, default_terminal::DefaultTerminal},
        instruction::Instruction,
    };
    #[cfg(feature = "field-inline")]
    use jolt_program::field_inline::{FieldEncodedValue, FieldInlineBridge};
    #[cfg(feature = "field-inline")]
    use jolt_riscv::{FieldInlineOp, FIELD_INLINE_OPCODE};

    #[cfg(feature = "field-inline")]
    fn field_inline_word(op: FieldInlineOp, rd: u8, rs1: u8, rs2_or_imm: u16) -> u32 {
        u32::from(FIELD_INLINE_OPCODE)
            | (u32::from(rd) << 7)
            | (u32::from(op.funct3()) << 12)
            | (u32::from(rs1) << 15)
            | (u32::from(rs2_or_imm) << 20)
    }

    #[cfg(feature = "field-inline")]
    #[test]
    fn trace_row_from_cycle_carries_field_inline_payload() {
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::default()));
        cpu.write_register(5, 11);
        let instruction = Instruction::decode(
            field_inline_word(FieldInlineOp::LoadFromX, 2, 5, 0),
            0x8000_0000,
            false,
        )
        .unwrap();
        let mut trace = Vec::new();
        instruction.trace(&mut cpu, Some(&mut trace));
        assert_eq!(trace.len(), 1);

        let row = super::trace_row_from_cycle(trace.remove(0)).unwrap();
        assert_eq!(row.registers.rs1.unwrap().register, 5);
        assert_eq!(row.registers.rs1.unwrap().value, 11);
        assert!(row.registers.rs2.is_none());
        assert!(row.registers.rd.is_none());
        let field_trace = row.field_inline.unwrap();
        assert_eq!(field_trace.op, Some(FieldInlineOp::LoadFromX));
        assert_eq!(
            field_trace.bridge,
            Some(FieldInlineBridge::LoadFromX {
                x_register: 5,
                x_value: 11,
                field_value: FieldEncodedValue::from_u64(11),
            })
        );
    }
}
