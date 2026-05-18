use std::path::PathBuf;

use jolt_program::execution::{
    ExecutionBackend, JoltProgram, MemoryImage, OwnedTrace, RamAccess as ProgramRamAccess,
    RamRead as ProgramRamRead, RamWrite as ProgramRamWrite, RegisterRead, RegisterState,
    RegisterWrite, TraceError, TraceInputs, TraceOutput, TraceRow,
};
use jolt_riscv::NormalizedInstruction;

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

        let (_lazy_trace, cycles, final_memory, device, _advice_tape, _field_reg_events) =
            crate::trace(
                program.elf_bytes(),
                self.elf_path.as_ref(),
                &inputs.inputs,
                &inputs.untrusted_advice,
                &inputs.trusted_advice,
                &inputs.memory_config,
                None,
            );

        let rows = cycles.into_iter().map(trace_row_from_cycle).collect();
        Ok(TraceOutput::new(
            OwnedTrace::new(rows),
            device,
            Some(MemoryImage {
                bytes: final_memory.materialized_nonzero_bytes(),
            }),
        ))
    }
}

/// Adapt a single tracer `Cycle` into the program-side `TraceRow` view
/// consumed by `jolt-kernels::trace::*` helpers. Public because the
/// modular SDK path (`jolt-host`) wants to drive `tracer::trace`
/// directly so it can also hand the tracer's `Memory` to
/// `jolt_core::zkvm::ram::gen_ram_memory_states`.
pub fn trace_row_from_cycle(cycle: Cycle) -> TraceRow {
    TraceRow {
        instruction: normalized_instruction(&cycle),
        registers: register_state(&cycle),
        ram_access: cycle.ram_access().into(),
    }
}

fn normalized_instruction(cycle: &Cycle) -> NormalizedInstruction {
    let instruction = cycle.instruction();
    (&instruction).into()
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
