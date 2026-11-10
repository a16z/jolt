use common::jolt_device::MemoryLayout;
use jolt_riscv::{JoltInstructionProfile, JoltInstructionRow};

use crate::preprocess::{
    bytecode::BytecodePreprocessing,
    ram::{
        bytecode_code_word_intervals, compute_max_ram_k, validate_code_word_intervals,
        RAMPreprocessing,
    },
    PreprocessingError,
};

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize),
    serde(try_from = "ProgramPreprocessingWire")
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
        let mut ram = RAMPreprocessing::preprocess(memory_init);
        ram.code_word_intervals =
            bytecode_code_word_intervals(&bytecode, memory_layout.get_lowest_address())?;
        validate_code_word_intervals(&ram.code_word_intervals, &memory_layout)?;
        Ok(Self {
            bytecode: BytecodePreprocessing::preprocess(bytecode, entry_address, profile)?,
            ram,
            memory_layout,
            max_padded_trace_length,
        })
    }

    pub fn validate_code_word_intervals(&self) -> Result<(), PreprocessingError> {
        validate_code_word_intervals(&self.ram.code_word_intervals, &self.memory_layout)?;
        let expected = bytecode_code_word_intervals(
            &self.bytecode.bytecode,
            self.memory_layout.get_lowest_address(),
        )?;
        if self.ram.code_word_intervals != expected {
            return Err(PreprocessingError::InvalidCodeWordIntervals(
                "protected words do not match decoded source extents",
            ));
        }
        Ok(())
    }

    /// Bytes the bytecode table, its PC map and the RAM program image hold on
    /// the heap, at capacity.
    pub fn heap_bytes(&self) -> usize {
        self.bytecode.bytecode.capacity() * size_of::<JoltInstructionRow>()
            + self.bytecode.pc_map.heap_bytes()
            + self.ram.bytecode_words.capacity() * size_of::<u64>()
            + self.ram.code_word_intervals.capacity() * size_of::<(usize, usize)>()
    }

    pub fn metadata(&self) -> Option<ProgramMetadata> {
        Some(ProgramMetadata {
            entry_address: self.bytecode.entry_address,
            min_bytecode_address: self.ram.min_bytecode_address,
            entry_bytecode_index: self.bytecode.entry_bytecode_index()?,
            program_image_len_words: self.ram.bytecode_words.len(),
            bytecode_len: self.bytecode.code_size,
            code_word_intervals: self.ram.code_word_intervals.clone(),
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
    pub code_word_intervals: Vec<(usize, usize)>,
}

impl ProgramMetadata {
    /// Hidden bytecode coverage is authenticated by the committed key; admission
    /// still checks the public mask's canonical shape and RAM bounds.
    pub fn validate_code_word_intervals(
        &self,
        memory_layout: &MemoryLayout,
    ) -> Result<(), PreprocessingError> {
        validate_code_word_intervals(&self.code_word_intervals, memory_layout)
    }
}

#[cfg(feature = "serialization")]
#[derive(serde::Deserialize)]
struct ProgramPreprocessingWire {
    bytecode: BytecodePreprocessing,
    ram: RAMPreprocessing,
    memory_layout: MemoryLayout,
    max_padded_trace_length: usize,
}

#[cfg(feature = "serialization")]
impl TryFrom<ProgramPreprocessingWire> for JoltProgramPreprocessing {
    type Error = PreprocessingError;

    fn try_from(wire: ProgramPreprocessingWire) -> Result<Self, Self::Error> {
        let program = Self {
            bytecode: wire.bytecode,
            ram: wire.ram,
            memory_layout: wire.memory_layout,
            max_padded_trace_length: wire.max_padded_trace_length,
        };
        program.validate_code_word_intervals()?;
        Ok(program)
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "fixture construction failures and indexing fail the tests"
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
        let instructions = [
            (RAM_START_ADDRESS as usize, None, false),
            (RAM_START_ADDRESS as usize + 6, Some(1), false),
            (RAM_START_ADDRESS as usize + 6, Some(0), true),
            (RAM_START_ADDRESS as usize + 24, None, true),
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
    fn protection_metadata_covers_source_extents_and_not_virtual_rows_or_padding() {
        let program = preprocessing();
        let first_word = program
            .memory_layout
            .remapped_word_address(RAM_START_ADDRESS)
            .unwrap() as usize;
        assert_eq!(
            program.ram.code_word_intervals,
            vec![
                (first_word, first_word + 1),
                (first_word + 3, first_word + 4)
            ]
        );
        assert_eq!(program.validate_code_word_intervals(), Ok(()));
        assert_eq!(
            program.metadata().unwrap().code_word_intervals,
            program.ram.code_word_intervals
        );
    }

    #[test]
    fn admission_rejects_missing_extra_and_noncanonical_protection_metadata() {
        let program = preprocessing();
        let first_word = program.ram.code_word_intervals[0].0;
        for intervals in [
            vec![],
            vec![(first_word, first_word + 4)],
            vec![
                (first_word, first_word + 1),
                (first_word + 1, first_word + 4),
            ],
            vec![(first_word, usize::MAX)],
        ] {
            let mut bad = program.clone();
            bad.ram.code_word_intervals = intervals;
            assert!(bad.validate_code_word_intervals().is_err());
        }
    }

    #[cfg(feature = "serialization")]
    #[test]
    fn deserialization_rejects_tampered_protection_metadata() {
        let program = preprocessing();
        let encoded = serde_json::to_value(&program).unwrap();
        assert_eq!(
            serde_json::from_value::<JoltProgramPreprocessing>(encoded.clone()).unwrap(),
            program
        );
        let mut encoded = encoded;
        encoded["ram"]["code_word_intervals"] = serde_json::json!([]);
        assert!(serde_json::from_value::<JoltProgramPreprocessing>(encoded).is_err());
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
