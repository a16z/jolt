use common::constants::RAM_START_ADDRESS;
use common::jolt_device::{JoltDevice, MemoryConfig};
use std::path::PathBuf;
use tracer::emulator::memory::Memory;
use tracer::instruction::{Cycle, Instruction};
use tracer::utils::virtual_registers::VirtualRegisterAllocator;
use tracer::LazyTraceIterator;

/// Configuration for program runtime
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub max_input_size: u64,
    pub max_output_size: u64,
}

/// Guest program that handles decoding and tracing
pub struct Program {
    pub elf_contents: Vec<u8>,
    pub elf_compute_advice_contents: Option<Vec<u8>>,
    pub memory_config: MemoryConfig,
    pub elf: Option<PathBuf>,
    pub elf_compute_advice: Option<PathBuf>,
}

impl Program {
    pub fn new(elf_contents: &[u8], memory_config: &MemoryConfig) -> Self {
        Self {
            elf_contents: elf_contents.to_vec(),
            elf_compute_advice_contents: None,
            memory_config: *memory_config,
            elf: None,
            elf_compute_advice: None,
        }
    }

    pub fn new_with_advice(
        elf_contents: &[u8],
        elf_compute_advice_contents: &[u8],
        memory_config: &MemoryConfig,
    ) -> Self {
        Self {
            elf_contents: elf_contents.to_vec(),
            elf_compute_advice_contents: Some(elf_compute_advice_contents.to_vec()),
            memory_config: *memory_config,
            elf: None,
            elf_compute_advice: None,
        }
    }

    /// Decode the ELF file into instructions and memory initialization
    pub fn decode(&self) -> (Vec<Instruction>, Vec<(u64, u8)>, u64) {
        decode(&self.elf_contents)
    }

    /// Trace the program execution with given inputs
    pub fn trace(
        &self,
        inputs: &[u8],
        untrusted_advice: &[u8],
        trusted_advice: &[u8],
    ) -> (LazyTraceIterator, Vec<Cycle>, Memory, JoltDevice) {
        trace(
            &self.elf_contents,
            self.elf.as_ref(),
            inputs,
            untrusted_advice,
            trusted_advice,
            &self.memory_config,
        )
    }

    pub fn trace_to_file(
        &self,
        inputs: &[u8],
        untrusted_advice: &[u8],
        trusted_advice: &[u8],
        trace_file: &PathBuf,
    ) -> (Memory, JoltDevice) {
        trace_to_file(
            &self.elf_contents,
            self.elf.as_ref(),
            inputs,
            untrusted_advice,
            trusted_advice,
            &self.memory_config,
            trace_file,
        )
    }

}

pub fn decode(elf: &[u8]) -> (Vec<Instruction>, Vec<(u64, u8)>, u64) {
    let (mut instructions, raw_bytes, program_end, xlen) = tracer::decode(elf);
    let program_size = program_end - RAM_START_ADDRESS;
    let allocator = VirtualRegisterAllocator::default();

    // Expand virtual sequences
    instructions = instructions
        .into_iter()
        .flat_map(|instr| instr.inline_sequence(&allocator, xlen))
        .collect();

    (instructions, raw_bytes, program_size)
}

pub fn trace(
    elf_contents: &[u8],
    elf_path: Option<&PathBuf>,
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    memory_config: &MemoryConfig,
) -> (LazyTraceIterator, Vec<Cycle>, Memory, JoltDevice) {
    let (lazy_trace, trace, memory, jolt_device, _advice_tape) = trace_with_advice(
        elf_contents,
        elf_path,
        inputs,
        untrusted_advice,
        trusted_advice,
        memory_config,
        None,
    );
    (lazy_trace, trace, memory, jolt_device)
}

pub fn trace_with_advice(
    elf_contents: &[u8],
    elf_path: Option<&PathBuf>,
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    memory_config: &MemoryConfig,
    advice_tape: Option<tracer::AdviceTape>,
) -> (LazyTraceIterator, Vec<Cycle>, Memory, JoltDevice, tracer::AdviceTape) {
    tracer::trace_with_advice(
        elf_contents,
        elf_path,
        inputs,
        untrusted_advice,
        trusted_advice,
        memory_config,
        advice_tape,
    )
}

pub fn trace_to_file(
    elf_contents: &[u8],
    elf_path: Option<&PathBuf>,
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    memory_config: &MemoryConfig,
    trace_file: &PathBuf,
) -> (Memory, JoltDevice) {
    let (memory, io_device) = tracer::trace_to_file(
        elf_contents,
        elf_path,
        inputs,
        untrusted_advice,
        trusted_advice,
        memory_config,
        trace_file,
    );
    (memory, io_device)
}
