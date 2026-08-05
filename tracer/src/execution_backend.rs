use std::{mem::MaybeUninit, path::PathBuf};

use jolt_program::execution::{
    ExecutionBackend, JoltProgram, MemoryImage, OwnedTrace, RamAccess as ProgramRamAccess,
    RamRead as ProgramRamRead, RamWrite as ProgramRamWrite, RegisterRead, RegisterState,
    RegisterWrite, TraceError, TraceInputs, TraceOutput, TraceRow,
};
use jolt_riscv::JoltInstructionRow;

use crate::instruction::{Cycle, RAMAccess};

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

        let (_lazy_trace, cycles, final_memory, device, _advice_tape) = crate::trace(
            program.elf_bytes(),
            self.elf_path.as_ref(),
            &inputs.inputs,
            &inputs.untrusted_advice,
            &inputs.trusted_advice,
            &inputs.memory_config,
            None,
        );

        let rows = rows_from_cycles(cycles)?;
        Ok(TraceOutput::new(
            OwnedTrace::new(rows),
            device,
            Some(MemoryImage {
                bytes: final_memory.materialized_nonzero_bytes(),
            }),
        ))
    }
}

const MIN_ROWS_PER_CONVERSION_WORKER: usize = 1 << 14;

fn rows_from_cycles(cycles: Vec<Cycle>) -> Result<Vec<TraceRow>, TraceError> {
    let workers = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
        .min(256)
        .min(cycles.len().div_ceil(MIN_ROWS_PER_CONVERSION_WORKER).max(1));
    rows_from_cycles_with_workers(cycles, workers)
}

fn rows_from_cycles_with_workers(
    cycles: Vec<Cycle>,
    workers: usize,
) -> Result<Vec<TraceRow>, TraceError> {
    let _span = tracing::info_span!(
        "trace_rows_from_cycles",
        rows = cycles.len(),
        workers = workers
    )
    .entered();
    if workers <= 1 || cycles.len() < 2 {
        return cycles.into_iter().map(trace_row_from_cycle).collect();
    }

    struct InitializedRows<'a> {
        rows: &'a mut [MaybeUninit<TraceRow>],
        len: usize,
        keep: bool,
    }

    impl Drop for InitializedRows<'_> {
        fn drop(&mut self) {
            if !self.keep {
                for row in &mut self.rows[..self.len] {
                    // SAFETY: this worker increments `len` after initializing each slot.
                    unsafe { row.assume_init_drop() };
                }
            }
        }
    }

    let len = cycles.len();
    let chunk_len = len.div_ceil(workers);
    let mut rows = Vec::with_capacity(len);
    // SAFETY: `MaybeUninit<TraceRow>` permits uninitialized elements; workers fill all slots.
    unsafe { rows.set_len(len) };

    let statuses = std::thread::scope(|scope| {
        let handles: Vec<_> = cycles
            .chunks(chunk_len)
            .zip(rows.chunks_mut(chunk_len))
            .map(|(cycles, rows)| {
                scope.spawn(move || {
                    let mut initialized = InitializedRows {
                        rows,
                        len: 0,
                        keep: false,
                    };
                    for (index, cycle) in cycles.iter().enumerate() {
                        let row = trace_row_from_cycle(*cycle)?;
                        initialized.rows[index].write(row);
                        initialized.len += 1;
                    }
                    initialized.keep = true;
                    Ok::<usize, TraceError>(initialized.len)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join())
            .collect::<Vec<_>>()
    });

    if statuses.iter().any(|status| !matches!(status, Ok(Ok(_)))) {
        let mut error = None;
        let mut panic = None;
        for (status, chunk) in statuses.into_iter().zip(rows.chunks_mut(chunk_len)) {
            match status {
                Ok(Ok(initialized)) => {
                    for row in &mut chunk[..initialized] {
                        // SAFETY: the successful worker initialized exactly this prefix.
                        unsafe { row.assume_init_drop() };
                    }
                }
                Ok(Err(worker_error)) => {
                    error.get_or_insert(worker_error);
                }
                Err(worker_panic) => {
                    panic.get_or_insert(worker_panic);
                }
            };
        }
        if let Some(worker_panic) = panic {
            std::panic::resume_unwind(worker_panic);
        }
        if let Some(worker_error) = error {
            return Err(worker_error);
        }
        unreachable!("failed cycle conversion without an error or panic");
    }

    let capacity = rows.capacity();
    let pointer = rows.as_mut_ptr().cast::<TraceRow>();
    std::mem::forget(rows);
    // SAFETY: every slot was initialized exactly once; `MaybeUninit<T>` has T's layout.
    Ok(unsafe { Vec::from_raw_parts(pointer, len, capacity) })
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
#[expect(clippy::unwrap_used)]
mod tests {
    use crate::{
        emulator::{cpu::Cpu, default_terminal::DefaultTerminal},
        instruction::Instruction,
    };
    #[cfg(feature = "field-inline")]
    use jolt_program::field_inline::{FieldEncodedValue, FieldInlineBridge};
    #[cfg(feature = "field-inline")]
    use jolt_riscv::{FieldInlineOp, FIELD_INLINE_OPCODE};

    #[test]
    fn parallel_cycle_conversion_matches_serial() {
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::default()));
        let instruction = Instruction::decode(0x0010_0093, 0x8000_0000, false).unwrap();
        let mut trace = Vec::new();
        instruction.trace(&mut cpu, Some(&mut trace));
        assert_eq!(trace.len(), 1);

        let cycles = vec![trace[0]; 1 << 15];
        let serial = super::rows_from_cycles_with_workers(cycles.clone(), 1).unwrap();
        let parallel = super::rows_from_cycles_with_workers(cycles, 4).unwrap();
        assert_eq!(parallel, serial);
    }

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
