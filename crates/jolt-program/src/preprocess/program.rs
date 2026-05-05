use common::jolt_device::MemoryLayout;
use jolt_riscv::NormalizedInstruction;

use crate::preprocess::{
    bytecode::BytecodePreprocessing, ram::RAMPreprocessing, PreprocessingError,
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct JoltProgramPreprocessing {
    pub bytecode: BytecodePreprocessing,
    pub ram: RAMPreprocessing,
    pub memory_layout: MemoryLayout,
    pub max_padded_trace_length: usize,
}

impl JoltProgramPreprocessing {
    pub fn new(
        bytecode: Vec<NormalizedInstruction>,
        memory_init: Vec<(u64, u8)>,
        memory_layout: MemoryLayout,
        entry_address: u64,
        max_padded_trace_length: usize,
    ) -> Result<Self, PreprocessingError> {
        Ok(Self {
            bytecode: BytecodePreprocessing::preprocess(bytecode, entry_address)?,
            ram: RAMPreprocessing::preprocess(memory_init),
            memory_layout,
            max_padded_trace_length,
        })
    }
}
