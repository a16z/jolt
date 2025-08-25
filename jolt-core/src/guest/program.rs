use common::constants::RAM_START_ADDRESS;
use common::jolt_device::{JoltDevice, MemoryConfig};
use tracer::emulator::memory::Memory;
use tracer::instruction::{RV32IMCycle, RV32IMInstruction};

/// Configuration for program runtime
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub max_input_size: u64,
    pub max_output_size: u64,
}

/// Guest program that handles decoding and tracing
pub struct Program {
    pub elf_contents: Vec<u8>,
    pub memory_config: MemoryConfig,
}

impl Program {
    pub fn new(elf_contents: &[u8], memory_config: &MemoryConfig) -> Self {
        Self {
            elf_contents: elf_contents.to_vec(),
            memory_config: *memory_config,
        }
    }

    /// Decode the ELF file into instructions and memory initialization
    pub fn decode(&self) -> (Vec<RV32IMInstruction>, Vec<(u64, u8)>, u64) {
        decode(&self.elf_contents)
    }

    /// Trace the program execution with given inputs
    pub fn trace(
        &self,
        memory_config: &MemoryConfig,
        inputs: &[u8],
    ) -> (Vec<RV32IMCycle>, Memory, JoltDevice) {
        trace(&self.elf_contents, inputs, memory_config)
    }
}

pub fn decode(elf: &[u8]) -> (Vec<RV32IMInstruction>, Vec<(u64, u8)>, u64) {
    let (mut instructions, raw_bytes, program_end, xlen) = tracer::decode(elf);
    let program_size = program_end - RAM_START_ADDRESS;

    // Expand virtual sequences
    instructions = instructions
        .into_iter()
        .flat_map(|instr| instr.inline_sequence(xlen))
        .collect();

    (instructions, raw_bytes, program_size)
}

pub fn trace(
    elf_contents: &[u8],
    inputs: &[u8],
    memory_config: &MemoryConfig,
) -> (Vec<RV32IMCycle>, Memory, JoltDevice) {
    let (trace, memory, io_device) = tracer::trace(elf_contents, inputs, memory_config);
    (trace, memory, io_device)
}
