use common::jolt_device::MemoryLayout;
use jolt_riscv::{JoltInstructionProfile, JoltInstructionRow};

use crate::preprocess::{
    bytecode::BytecodePreprocessing,
    ram::{compute_max_ram_k, RAMPreprocessing},
    PreprocessingError,
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
        let _ = compute_max_ram_k(&memory_layout)?;
        Ok(Self {
            bytecode: BytecodePreprocessing::preprocess(bytecode, entry_address, profile)?,
            ram: RAMPreprocessing::preprocess(memory_init),
            memory_layout,
            max_padded_trace_length,
        })
    }

    /// Bytes the bytecode table, its PC map and the RAM program image hold on
    /// the heap, at capacity.
    pub fn heap_bytes(&self) -> usize {
        self.bytecode.bytecode.capacity() * size_of::<JoltInstructionRow>()
            + self.bytecode.pc_map.heap_bytes()
            + self.ram.bytecode_words.capacity() * size_of::<u64>()
    }

    pub fn metadata(&self) -> Option<ProgramMetadata> {
        Some(ProgramMetadata {
            entry_address: self.bytecode.entry_address,
            min_bytecode_address: self.ram.min_bytecode_address,
            entry_bytecode_index: self.bytecode.entry_bytecode_index()?,
            program_image_len_words: self.ram.bytecode_words.len(),
            bytecode_len: self.bytecode.code_size,
        })
    }
}

/// Verifier-side program shape for committed program mode: the trusted
/// commitments replace the bytecode table and program image, so only this
/// metadata accompanies them.
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

#[cfg(test)]
mod tests {
    use super::JoltProgramPreprocessing;
    use crate::preprocess::{PreprocessingError, RamDomainError};
    use common::jolt_device::{MemoryConfig, MemoryLayout, MemoryLayoutError};
    use jolt_riscv::RV64IMAC_JOLT;

    #[test]
    fn full_preprocessing_rechecks_layout_admission() {
        let mut layout = MemoryLayout::new(&MemoryConfig {
            program_size: Some(1024),
            ..Default::default()
        });
        layout.heap_end = u64::MAX - 6;
        assert!(matches!(
            JoltProgramPreprocessing::new(
                Vec::new(),
                Vec::new(),
                layout,
                0,
                1 << 16,
                RV64IMAC_JOLT
            ),
            Err(PreprocessingError::RamDomain(RamDomainError::MemoryLayout(
                MemoryLayoutError::HeapEndTooHigh { .. }
            )))
        ));
    }
}
