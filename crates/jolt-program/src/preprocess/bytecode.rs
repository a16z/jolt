#[cfg(feature = "serialization")]
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_common::constants::{ALIGNMENT_FACTOR_BYTECODE, RAM_START_ADDRESS};
use jolt_riscv::{
    CircuitFlags, Flags, JoltInstruction, JoltInstructionKind, JoltInstructionProfile,
    JoltInstructionRow,
};

#[cfg(feature = "field-inline")]
use crate::field_inline::FieldInlineBytecodeMetadata;
use crate::preprocess::PreprocessingError;

#[derive(Default, Debug, Clone, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(
        CanonicalSerialize,
        CanonicalDeserialize,
        serde::Serialize,
        serde::Deserialize
    )
)]
pub struct BytecodePreprocessing {
    pub code_size: usize,
    pub bytecode: Vec<JoltInstructionRow>,
    /// Maps each unexpanded instruction address to its virtual bytecode index.
    pub pc_map: BytecodePCMapper,
    pub entry_address: u64,
    #[cfg(feature = "field-inline")]
    pub field_inline: Option<FieldInlineBytecodeMetadata>,
}

impl BytecodePreprocessing {
    pub fn preprocess(
        mut bytecode: Vec<JoltInstructionRow>,
        entry_address: u64,
        profile: JoltInstructionProfile,
    ) -> Result<Self, PreprocessingError> {
        for instruction in &bytecode {
            if !profile.supports_jolt(instruction.instruction_kind) {
                return Err(PreprocessingError::IllegalTargetInstruction(
                    instruction.instruction_kind,
                ));
            }
            check_store_rd_disjoint(instruction)?;
        }
        bytecode.insert(0, noop_instruction());
        let pc_map = BytecodePCMapper::try_new(&bytecode)?;

        let code_size = bytecode.len().next_power_of_two().max(2);
        bytecode.resize(code_size, noop_instruction());
        #[cfg(feature = "field-inline")]
        let field_inline = if profile.supports_field_inline() {
            Some(FieldInlineBytecodeMetadata::from_bytecode(
                &bytecode,
                profile.fingerprint(),
            )?)
        } else {
            None
        };

        Ok(Self {
            code_size,
            bytecode,
            pc_map,
            entry_address,
            #[cfg(feature = "field-inline")]
            field_inline,
        })
    }

    pub fn entry_bytecode_index(&self) -> Option<usize> {
        self.pc_map.get_first_pc(self.entry_address as usize)
    }

    pub fn get_pc(&self, instruction: &JoltInstructionRow) -> Option<usize> {
        if instruction.instruction_kind == JoltInstructionKind::NoOp {
            return Some(0);
        }
        self.pc_map.get_pc(
            instruction.address,
            instruction.virtual_sequence_remaining.unwrap_or(0),
        )
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(
        CanonicalSerialize,
        CanonicalDeserialize,
        serde::Serialize,
        serde::Deserialize
    )
)]
pub struct BytecodePCMapper {
    indices: Vec<Vec<(u16, usize)>>,
}

impl BytecodePCMapper {
    pub fn try_new(bytecode: &[JoltInstructionRow]) -> Result<Self, PreprocessingError> {
        let mut indices = vec![Vec::new(); Self::index_count(bytecode)?];
        let mut last_pc = 0;
        indices[0].push((0, last_pc));

        for instruction in bytecode {
            if instruction.address == 0 {
                continue;
            }

            last_pc += 1;
            let bytecode_index = Self::try_get_index(instruction.address)?;
            indices[bytecode_index]
                .push((instruction.virtual_sequence_remaining.unwrap_or(0), last_pc));
        }

        for (bytecode_index, entries) in indices.iter().enumerate() {
            Self::validate_indices(bytecode_index, entries)?;
        }

        Ok(Self { indices })
    }

    pub fn get_pc(&self, address: usize, virtual_sequence_remaining: u16) -> Option<usize> {
        let index = Self::try_get_index(address).ok()?;
        self.indices
            .get(index)?
            .iter()
            .find_map(|(sequence, pc)| (*sequence == virtual_sequence_remaining).then_some(*pc))
    }

    pub fn get_first_pc(&self, address: usize) -> Option<usize> {
        let index = if address == 0 {
            0
        } else {
            Self::try_get_index(address).ok()?
        };
        self.indices.get(index)?.first().map(|(_sequence, pc)| *pc)
    }

    fn try_get_index(address: usize) -> Result<usize, PreprocessingError> {
        if address < RAM_START_ADDRESS as usize
            || !address.is_multiple_of(ALIGNMENT_FACTOR_BYTECODE)
        {
            return Err(PreprocessingError::InvalidBytecodeAddress(address));
        }
        Ok(Self::get_index(address))
    }

    pub const fn get_index(address: usize) -> usize {
        assert!(address >= RAM_START_ADDRESS as usize);
        assert!(address.is_multiple_of(ALIGNMENT_FACTOR_BYTECODE));
        (address - RAM_START_ADDRESS as usize) / ALIGNMENT_FACTOR_BYTECODE + 1
    }

    const fn address_for_index(index: usize) -> usize {
        if index == 0 {
            0
        } else {
            RAM_START_ADDRESS as usize + (index - 1) * ALIGNMENT_FACTOR_BYTECODE
        }
    }

    fn validate_indices(
        bytecode_index: usize,
        entries: &[(u16, usize)],
    ) -> Result<(), PreprocessingError> {
        for window in entries.windows(2) {
            let [(previous_sequence, _), (new_sequence, _)] = window else {
                unreachable!("windows(2) always yields two entries");
            };
            let Some(expected_sequence) = previous_sequence.checked_sub(1) else {
                return Err(PreprocessingError::InvalidInlineSequence {
                    bytecode_index,
                    address: Self::address_for_index(bytecode_index),
                    previous_sequence: *previous_sequence,
                    expected_sequence: 0,
                    new_sequence: *new_sequence,
                });
            };
            if *new_sequence != expected_sequence {
                return Err(PreprocessingError::InvalidInlineSequence {
                    bytecode_index,
                    address: Self::address_for_index(bytecode_index),
                    previous_sequence: *previous_sequence,
                    expected_sequence,
                    new_sequence: *new_sequence,
                });
            }
        }
        Ok(())
    }

    fn index_count(bytecode: &[JoltInstructionRow]) -> Result<usize, PreprocessingError> {
        let max_address = bytecode
            .iter()
            .map(|instruction| instruction.address)
            .max()
            .unwrap_or(0);
        if max_address == 0 {
            Ok(1)
        } else {
            Ok(Self::try_get_index(max_address)? + 1)
        }
    }
}

/// The store/rd-write disjointness check on the public bytecode: a
/// `Store`-flagged instruction must not name an rd destination. This is the
/// offline half of the lattice fused-inc soundness argument (one committed
/// increment stream serves both RAM and rd because no cycle increments both
/// — see `specs/lattice-claims.md`); the trace-level converse (a RAM write
/// only ever comes from a `Store`-flagged row) is asserted during witness
/// generation.
fn check_store_rd_disjoint(instruction: &JoltInstructionRow) -> Result<(), PreprocessingError> {
    let decoded = JoltInstruction::try_from(*instruction).unwrap_or(JoltInstruction::Noop(
        jolt_riscv::instructions::Noop(*instruction),
    ));
    match instruction.operands.rd {
        Some(rd) if decoded.circuit_flags()[CircuitFlags::Store] => {
            Err(PreprocessingError::StoreWritesRd {
                address: instruction.address,
                rd,
            })
        }
        _ => Ok(()),
    }
}

const fn noop_instruction() -> JoltInstructionRow {
    JoltInstructionRow {
        instruction_kind: JoltInstructionKind::NoOp,
        address: 0,
        operands: jolt_riscv::NormalizedOperands {
            rs1: None,
            rs2: None,
            rd: None,
            imm: 0,
        },
        virtual_sequence_remaining: None,
        is_first_in_sequence: false,
        is_compressed: false,
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use jolt_riscv::{
        JoltInstructionKind, JoltInstructionProfile, JoltInstructionRow, NormalizedOperands,
        SourceExtension, RV64IMAC_JOLT,
    };

    use super::{BytecodePCMapper, BytecodePreprocessing, PreprocessingError};

    #[test]
    fn preprocess_prepends_and_pads_bytecode() {
        let bytecode = vec![instruction(0x8000_0000, None)];

        let preprocessing =
            BytecodePreprocessing::preprocess(bytecode, 0x8000_0000, RV64IMAC_JOLT).unwrap();

        assert_eq!(preprocessing.code_size, 2);
        assert_eq!(
            preprocessing.bytecode[0].instruction_kind,
            JoltInstructionKind::NoOp
        );
        assert_eq!(preprocessing.entry_bytecode_index(), Some(1));
    }

    #[test]
    fn maps_inline_sequence_pcs() {
        let bytecode = vec![
            instruction(0x8000_0004, Some(2)),
            instruction(0x8000_0004, Some(1)),
            instruction(0x8000_0004, Some(0)),
        ];

        let preprocessing =
            BytecodePreprocessing::preprocess(bytecode, 0x8000_0004, RV64IMAC_JOLT).unwrap();

        assert_eq!(preprocessing.entry_bytecode_index(), Some(1));
        assert_eq!(
            preprocessing.get_pc(&instruction(0x8000_0004, Some(2))),
            Some(1)
        );
        assert_eq!(
            preprocessing.get_pc(&instruction(0x8000_0004, Some(1))),
            Some(2)
        );
        assert_eq!(
            preprocessing.get_pc(&instruction(0x8000_0004, Some(0))),
            Some(3)
        );
    }

    #[test]
    fn rejects_invalid_inline_sequences() {
        let bytecode = vec![
            instruction(0x8000_0004, Some(1)),
            instruction(0x8000_0004, Some(1)),
        ];

        let err = BytecodePCMapper::try_new(&bytecode).unwrap_err();
        assert_eq!(
            err,
            PreprocessingError::InvalidInlineSequence {
                bytecode_index: BytecodePCMapper::get_index(0x8000_0004),
                address: 0x8000_0004,
                previous_sequence: 1,
                expected_sequence: 0,
                new_sequence: 1,
            }
        );
    }

    #[test]
    fn rejects_non_consecutive_inline_sequences() {
        let bytecode = vec![
            instruction(0x8000_0004, Some(2)),
            instruction(0x8000_0004, Some(0)),
        ];

        let err = BytecodePCMapper::try_new(&bytecode).unwrap_err();
        assert_eq!(
            err,
            PreprocessingError::InvalidInlineSequence {
                bytecode_index: BytecodePCMapper::get_index(0x8000_0004),
                address: 0x8000_0004,
                previous_sequence: 2,
                expected_sequence: 1,
                new_sequence: 0,
            }
        );
    }

    #[test]
    fn rejects_invalid_bytecode_addresses() {
        let bytecode = vec![instruction(0x7fff_fffc, None)];

        let err = BytecodePCMapper::try_new(&bytecode).unwrap_err();
        assert_eq!(err, PreprocessingError::InvalidBytecodeAddress(0x7fff_fffc));
    }

    fn instruction(address: usize, virtual_sequence_remaining: Option<u16>) -> JoltInstructionRow {
        JoltInstructionRow {
            instruction_kind: JoltInstructionKind::ADDI,
            address,
            operands: NormalizedOperands {
                rd: Some(1),
                rs1: Some(2),
                rs2: None,
                imm: 3,
            },
            virtual_sequence_remaining,
            is_first_in_sequence: virtual_sequence_remaining == Some(2),
            is_compressed: false,
        }
    }

    #[test]
    fn rejects_store_rows_that_write_rd() {
        let mut row = instruction(0x8000_0000, None);
        row.instruction_kind = JoltInstructionKind::SD;
        row.operands = NormalizedOperands {
            rd: Some(5),
            rs1: Some(2),
            rs2: Some(3),
            imm: 0,
        };

        let err =
            BytecodePreprocessing::preprocess(vec![row], 0x8000_0000, RV64IMAC_JOLT).unwrap_err();
        assert_eq!(
            err,
            PreprocessingError::StoreWritesRd {
                address: 0x8000_0000,
                rd: 5,
            }
        );

        // The same store without an rd destination passes.
        let mut clean = instruction(0x8000_0000, None);
        clean.instruction_kind = JoltInstructionKind::SD;
        clean.operands = NormalizedOperands {
            rd: None,
            rs1: Some(2),
            rs2: Some(3),
            imm: 0,
        };
        let preprocessed =
            BytecodePreprocessing::preprocess(vec![clean], 0x8000_0000, RV64IMAC_JOLT).unwrap();
        assert_eq!(preprocessed.code_size, 2);
    }

    #[test]
    fn rejects_profile_illegal_target_rows() {
        const RV64I_ONLY: JoltInstructionProfile = JoltInstructionProfile {
            source_extensions: &[SourceExtension::Rv64I],
            inline_extensions: &[],
        };

        let mut row = instruction(0x8000_0000, None);
        row.instruction_kind = JoltInstructionKind::MUL;

        let err =
            BytecodePreprocessing::preprocess(vec![row], 0x8000_0000, RV64I_ONLY).unwrap_err();
        assert_eq!(
            err,
            PreprocessingError::IllegalTargetInstruction(JoltInstructionKind::MUL)
        );
    }

    #[cfg(feature = "field-inline")]
    #[test]
    fn fr_off_preprocessing_rejects_field_inline_rows() {
        let mut row = instruction(0x8000_0000, None);
        row.instruction_kind = JoltInstructionKind::FIELD_MUL;
        row.operands = NormalizedOperands {
            rd: Some(1),
            rs1: Some(2),
            rs2: Some(3),
            imm: 0,
        };

        let err =
            BytecodePreprocessing::preprocess(vec![row], 0x8000_0000, RV64IMAC_JOLT).unwrap_err();
        assert_eq!(
            err,
            PreprocessingError::IllegalTargetInstruction(JoltInstructionKind::FIELD_MUL)
        );
    }

    #[cfg(feature = "field-inline")]
    #[test]
    fn fr_on_preprocessing_builds_clean_metadata_for_field_rows() {
        let mut row = instruction(0x8000_0000, None);
        row.instruction_kind = JoltInstructionKind::FIELD_MUL;
        row.operands = NormalizedOperands {
            rd: Some(1),
            rs1: Some(2),
            rs2: Some(3),
            imm: 0,
        };

        let preprocessing = BytecodePreprocessing::preprocess(
            vec![row],
            0x8000_0000,
            jolt_riscv::RV64IMAC_JOLT_FIELD_INLINE,
        )
        .unwrap();
        let metadata = preprocessing.field_inline.as_ref().unwrap();

        assert_eq!(metadata.rows.len(), preprocessing.bytecode.len());
        assert!(!metadata.rows[0].active);
        assert!(metadata.rows[1].active);
        assert_eq!(metadata.rows[1].op, Some(jolt_riscv::FieldInlineOp::Mul));
        assert_eq!(
            metadata.rows[1].rd.map(jolt_riscv::FieldRegister::index),
            Some(1)
        );
        assert_eq!(
            metadata.rows[1].rs1.map(jolt_riscv::FieldRegister::index),
            Some(2)
        );
        assert_eq!(
            metadata.rows[1].rs2.map(jolt_riscv::FieldRegister::index),
            Some(3)
        );
    }

    #[cfg(feature = "field-inline")]
    #[test]
    fn field_inline_metadata_rejects_out_of_bounds_field_registers() {
        let mut row = instruction(0x8000_0000, None);
        row.instruction_kind = JoltInstructionKind::FIELD_ADD;
        row.operands = NormalizedOperands {
            rd: Some(jolt_riscv::FIELD_REGISTER_COUNT),
            rs1: Some(1),
            rs2: Some(2),
            imm: 0,
        };

        let err = BytecodePreprocessing::preprocess(
            vec![row],
            0x8000_0000,
            jolt_riscv::RV64IMAC_JOLT_FIELD_INLINE,
        )
        .unwrap_err();

        assert!(matches!(
            err,
            PreprocessingError::InvalidFieldInlineMetadata(
                crate::field_inline::FieldInlineMetadataError::InvalidFieldRegister {
                    operand: "rd",
                    register
                }
            ) if register == jolt_riscv::FIELD_REGISTER_COUNT
        ));
    }
}
