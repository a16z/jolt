use common::jolt_device::MemoryLayout;

use crate::preprocess::{bytecode::BytecodePreprocessing, ram::RAMPreprocessing};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct JoltProgramPreprocessing {
    pub bytecode: BytecodePreprocessing,
    pub ram: RAMPreprocessing,
    pub memory_layout: MemoryLayout,
    pub max_padded_trace_length: usize,
}
