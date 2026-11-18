use common::jolt_device::MemoryLayout;
use jolt_riscv::{JoltInstructionProfile, JoltInstructionRow};
use std::ops::Range;

use crate::preprocess::{
    bytecode::BytecodePreprocessing,
    ram::{bytecode_code_region, compute_max_ram_k, validate_code_region, RAMPreprocessing},
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
        let program = Self {
            bytecode: BytecodePreprocessing::preprocess(bytecode, entry_address, profile)?,
            ram: RAMPreprocessing::preprocess(memory_init),
            memory_layout,
            max_padded_trace_length,
        };
        let _ = program.code_region()?;
        Ok(program)
    }

    /// The code region of this program's bytecode (see [`bytecode_code_region`]),
    /// checked to lie in the RAM domain.
    pub fn code_region(&self) -> Result<Range<usize>, PreprocessingError> {
        let region = bytecode_code_region(
            &self.bytecode.bytecode,
            self.memory_layout.get_lowest_address(),
        )?;
        validate_code_region(&region, &self.memory_layout)?;
        Ok(region)
    }

    /// Bytes the bytecode table, its PC map and the RAM program image hold on
    /// the heap, at capacity.
    pub fn heap_bytes(&self) -> usize {
        self.bytecode.bytecode.capacity() * size_of::<JoltInstructionRow>()
            + self.bytecode.pc_map.heap_bytes()
            + self.ram.bytecode_words.capacity() * size_of::<u64>()
    }

    /// `None` when the entry has no bytecode row or the code region is invalid.
    pub fn metadata(&self) -> Option<ProgramMetadata> {
        Some(ProgramMetadata {
            entry_address: self.bytecode.entry_address,
            min_bytecode_address: self.ram.min_bytecode_address,
            entry_bytecode_index: self.bytecode.entry_bytecode_index()?,
            program_image_len_words: self.ram.bytecode_words.len(),
            bytecode_len: self.bytecode.code_size,
            code_region: self.code_region().ok()?,
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
    /// The bytecode's code region, which the verifier cannot derive here.
    pub code_region: Range<usize>,
}

impl ProgramMetadata {
    /// The committed key authenticates the region against the hidden bytecode;
    /// admission still checks that it lies in RAM and is canonical.
    pub fn validate_code_region(
        &self,
        memory_layout: &MemoryLayout,
    ) -> Result<(), PreprocessingError> {
        validate_code_region(&self.code_region, memory_layout)
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "fixture construction failures fail the tests"
)]
mod tests {
    use super::*;
    use common::{constants::RAM_START_ADDRESS, jolt_device::MemoryConfig};
    use jolt_riscv::{JoltInstructionKind as Kind, NormalizedOperands, RV64IMAC_JOLT};

    fn preprocessing() -> JoltProgramPreprocessing {
        let layout = MemoryLayout::new(&MemoryConfig {
            program_size: Some(64),
            ..Default::default()
        });
        // The last source instruction is compressed and expands to two rows;
        // only its final row records the two-byte extent.
        let instructions = [
            (RAM_START_ADDRESS as usize, None, false),
            (RAM_START_ADDRESS as usize + 6, Some(1), false),
            (RAM_START_ADDRESS as usize + 6, Some(0), true),
        ]
        .into_iter()
        .map(
            |(address, virtual_sequence_remaining, is_compressed)| JoltInstructionRow {
                instruction_kind: Kind::ADDI,
                address,
                operands: NormalizedOperands::default(),
                virtual_sequence_remaining,
                is_first_in_sequence: virtual_sequence_remaining == Some(1),
                is_compressed,
            },
        )
        .collect();
        JoltProgramPreprocessing::new(
            instructions,
            Vec::new(),
            layout,
            RAM_START_ADDRESS,
            8,
            RV64IMAC_JOLT,
        )
        .unwrap()
    }

    #[test]
    fn code_region_ends_at_the_anchor_of_a_compressed_expansion() {
        let program = preprocessing();
        let first_word = program
            .memory_layout
            .remapped_word_address(RAM_START_ADDRESS)
            .unwrap() as usize;
        assert_eq!(program.code_region(), Ok(first_word..first_word + 1));
        assert_eq!(
            program.metadata().unwrap().code_region,
            first_word..first_word + 1
        );
    }

    #[test]
    fn metadata_admission_rejects_reversed_outside_and_noncanonical_regions() {
        let program = preprocessing();
        let layout = &program.memory_layout;
        let metadata = program.metadata().unwrap();
        assert_eq!(metadata.validate_code_region(layout), Ok(()));
        let max_ram_k = compute_max_ram_k(layout).unwrap();
        for code_region in [Range { start: 5, end: 4 }, 0..max_ram_k + 1, 5..5] {
            let bad = ProgramMetadata {
                code_region,
                ..metadata.clone()
            };
            assert!(bad.validate_code_region(layout).is_err());
        }
        let empty = ProgramMetadata {
            code_region: 0..0,
            ..metadata
        };
        assert_eq!(empty.validate_code_region(layout), Ok(()));
    }
}

#[cfg(test)]
mod admission_tests {
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
