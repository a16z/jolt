#[cfg(feature = "serialization")]
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::{ALIGNMENT_FACTOR_BYTECODE, RAM_START_ADDRESS};
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

/// An address's rows form a descending `virtual_sequence_remaining` run at
/// consecutive PCs, so these three bounds determine every `(address, vsr) -> pc`.
/// Field order keeps each per-address mapping to one 8-byte slot.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(
        CanonicalSerialize,
        CanonicalDeserialize,
        serde::Serialize,
        serde::Deserialize
    )
)]
struct PcSlot {
    /// PC of the address's first row; `u32::MAX` marks an unmapped slot.
    first_pc: u32,
    /// `virtual_sequence_remaining` of that first row — the run's upper bound.
    first_vsr: u16,
    /// `virtual_sequence_remaining` of the address's last row — the run's lower
    /// bound, and the cursor the build walk decrements. Always 0 for sequences
    /// the expander stamps; nonzero only for a run that stops short of its anchor.
    last_vsr: u16,
}

const EMPTY_SLOT: PcSlot = PcSlot {
    first_pc: u32::MAX,
    first_vsr: 0,
    last_vsr: 0,
};

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
    slots: Vec<PcSlot>,
}

impl BytecodePCMapper {
    pub fn try_new(bytecode: &[JoltInstructionRow]) -> Result<Self, PreprocessingError> {
        // One allocation at the final size; the no-op sentinel lives in the
        // first slot (`index_count` is always >= 1).
        let mut slots = vec![EMPTY_SLOT; Self::index_count(bytecode)?];
        if let Some(first) = slots.first_mut() {
            *first = PcSlot {
                first_pc: 0,
                first_vsr: 0,
                last_vsr: 0,
            };
        }
        let mut last_pc = 0u32;

        for (pc, instruction) in bytecode.iter().enumerate() {
            if instruction.address == 0 {
                if pc == 0 {
                    continue;
                }
                return Err(PreprocessingError::InvalidBytecodeAddress(0));
            }

            // `u32::MAX` is reserved as the unmapped-slot marker.
            last_pc = last_pc
                .checked_add(1)
                .filter(|pc| *pc < u32::MAX)
                .ok_or(PreprocessingError::BytecodeTooLarge)?;
            let bytecode_index = Self::try_get_index(instruction.address)?;
            let sequence = instruction.virtual_sequence_remaining.unwrap_or(0);
            let slot =
                slots
                    .get_mut(bytecode_index)
                    .ok_or(PreprocessingError::InvalidBytecodeAddress(
                        instruction.address,
                    ))?;
            if *slot == EMPTY_SLOT {
                *slot = PcSlot {
                    first_pc: last_pc,
                    first_vsr: sequence,
                    last_vsr: sequence,
                };
                continue;
            }
            // Each row at an address decrements the previous sequence number
            // by one.
            let expected_sequence =
                slot.last_vsr
                    .checked_sub(1)
                    .ok_or(PreprocessingError::InvalidInlineSequence {
                        bytecode_index,
                        address: instruction.address,
                        previous_sequence: slot.last_vsr,
                        expected_sequence: 0,
                        new_sequence: sequence,
                    })?;
            if sequence != expected_sequence {
                return Err(PreprocessingError::InvalidInlineSequence {
                    bytecode_index,
                    address: instruction.address,
                    previous_sequence: slot.last_vsr,
                    expected_sequence,
                    new_sequence: sequence,
                });
            }
            // The slot arithmetic assumes the address's bytecode indices are
            // consecutive; an inline sequence interleaved with another
            // address would break that silently, so reject it.
            if last_pc - slot.first_pc != (slot.first_vsr - sequence) as u32 {
                return Err(PreprocessingError::NonContiguousInlineSequence {
                    bytecode_index,
                    address: instruction.address,
                });
            }
            slot.last_vsr = sequence;
        }

        Ok(Self { slots })
    }

    pub fn get_pc(&self, address: usize, virtual_sequence_remaining: u16) -> Option<usize> {
        let index = Self::try_get_index(address).ok()?;
        let slot = *self.slots.get(index)?;
        if slot.first_pc == u32::MAX {
            return None;
        }
        (slot.last_vsr..=slot.first_vsr)
            .contains(&virtual_sequence_remaining)
            .then(|| {
                slot.first_pc as usize + (slot.first_vsr - virtual_sequence_remaining) as usize
            })
    }

    pub fn get_first_pc(&self, address: usize) -> Option<usize> {
        let index = if address == 0 {
            0
        } else {
            Self::try_get_index(address).ok()?
        };
        let slot = *self.slots.get(index)?;
        (slot.first_pc != u32::MAX).then_some(slot.first_pc as usize)
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
#[expect(clippy::indexing_slicing, reason = "tests index fixture data")]
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
    fn rejects_interleaved_inline_sequences() {
        let bytecode = vec![
            instruction(0x8000_0004, Some(1)),
            instruction(0x8000_0008, None),
            instruction(0x8000_0004, Some(0)),
        ];

        let err = BytecodePCMapper::try_new(&bytecode).unwrap_err();
        assert_eq!(
            err,
            PreprocessingError::NonContiguousInlineSequence {
                bytecode_index: BytecodePCMapper::get_index(0x8000_0004),
                address: 0x8000_0004,
            }
        );
    }

    #[test]
    fn rejects_zero_address_outside_sentinel() {
        let bytecode = vec![
            instruction(0x8000_0004, Some(1)),
            instruction(0, None),
            instruction(0x8000_0004, Some(0)),
        ];

        let err =
            BytecodePreprocessing::preprocess(bytecode, 0x8000_0004, RV64IMAC_JOLT).unwrap_err();
        assert_eq!(err, PreprocessingError::InvalidBytecodeAddress(0));
    }

    #[test]
    fn rejects_invalid_bytecode_addresses() {
        let bytecode = vec![instruction(0x7fff_fffc, None)];

        let err = BytecodePCMapper::try_new(&bytecode).unwrap_err();
        assert_eq!(err, PreprocessingError::InvalidBytecodeAddress(0x7fff_fffc));
    }

    /// Every `NoOp` lands on bytecode slot 0 regardless of its address, and
    /// an instruction with no mapping fails materialization outright. Together
    /// those make the witness layer's `BytecodePc` total — one column for both
    /// the read-RAF pushforward and the committed one-hot.
    #[test]
    fn noop_maps_to_bytecode_slot_zero() {
        let bytecode = vec![instruction(0x8000_0000, None)];
        let preprocessing =
            BytecodePreprocessing::preprocess(bytecode, 0x8000_0000, RV64IMAC_JOLT).unwrap();

        let mut noop = instruction(0x8000_0004, None);
        noop.instruction_kind = JoltInstructionKind::NoOp;
        assert_eq!(preprocessing.get_pc(&noop), Some(0));

        // Not merely because the address is unmapped: the same address as a
        // non-no-op has no slot at all.
        assert_eq!(preprocessing.get_pc(&instruction(0x8000_0004, None)), None);
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
