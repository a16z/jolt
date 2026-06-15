use common::jolt_device::MemoryLayout;
use jolt_riscv::{JoltInstructionProfile, JoltInstructionRow};

use crate::preprocess::{
    bytecode::BytecodePreprocessing, ram::RAMPreprocessing, PreprocessingError,
};

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct JoltProgramPreprocessing {
    pub bytecode: BytecodePreprocessing,
    pub ram: RAMPreprocessing,
    pub memory_layout: MemoryLayout,
    pub max_padded_trace_length: usize,
}

impl JoltProgramPreprocessing {
    pub fn new(
        bytecode: Vec<JoltInstructionRow>,
        memory_init: Vec<(u64, u8)>,
        memory_layout: MemoryLayout,
        entry_address: u64,
        max_padded_trace_length: usize,
        profile: JoltInstructionProfile,
    ) -> Result<Self, PreprocessingError> {
        Ok(Self {
            bytecode: BytecodePreprocessing::preprocess(bytecode, entry_address, profile)?,
            ram: RAMPreprocessing::preprocess(memory_init),
            memory_layout,
            max_padded_trace_length,
        })
    }
}

/// Verifier-side program shape for committed program mode: the trusted
/// commitments replace the bytecode table and program image, so only this
/// metadata accompanies them. Mirrors `jolt-core`'s `ProgramMetadata`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct ProgramMetadata {
    pub entry_address: u64,
    pub min_bytecode_address: u64,
    pub entry_bytecode_index: usize,
    pub program_image_len_words: usize,
    pub bytecode_len: usize,
}
