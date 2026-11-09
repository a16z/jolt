//! Static PCALL metadata, included in the feature-on preprocessing digest.
use super::PreprocessingError;
#[cfg(feature = "serialization")]
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::RAM_START_ADDRESS;
use jolt_riscv::{JoltInstructionKind as Kind, JoltInstructionRow};

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(
        CanonicalSerialize,
        CanonicalDeserialize,
        serde::Serialize,
        serde::Deserialize
    )
)]
pub struct PcallRow {
    /// Zero denotes an ordinary row; one denotes KECCAKF.
    pub family: u8,
    pub flavour: u8,
    pub ordinal: u8,
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
pub struct PcallPreprocessing {
    pub rows: Vec<PcallRow>,
}

impl PcallPreprocessing {
    pub fn new(bytecode: &[JoltInstructionRow]) -> Result<Self, PreprocessingError> {
        let mut rows = vec![PcallRow::default(); bytecode.len()];
        let mut cursor = 0;
        while let Some(call) = bytecode.get(cursor) {
            if !matches!(call.instruction_kind, Kind::PcallCall(_)) {
                if matches!(
                    call.instruction_kind,
                    Kind::PcallRead(_) | Kind::PcallWrite(_) | Kind::PcallReturn(_)
                ) {
                    return Err(PreprocessingError::InvalidPcallSequence {
                        bytecode_index: cursor,
                    });
                }
                cursor += 1;
                continue;
            }
            let (flavour, length, writes) = match call.virtual_sequence_remaining {
                Some(26) => (0, 27, 1),
                Some(43) => (1, 44, 18),
                _ => {
                    return Err(PreprocessingError::InvalidPcallSequence {
                        bytecode_index: cursor,
                    })
                }
            };
            let sequence = bytecode.get(cursor..cursor + length).ok_or(
                PreprocessingError::InvalidPcallSequence {
                    bytecode_index: cursor,
                },
            )?;
            let state_register = sequence.get(writes).and_then(|row| row.operands.rs1);
            let block_register = sequence.get(1).and_then(|row| row.operands.rs1);
            for (ordinal, row) in sequence.iter().enumerate() {
                let expected = if ordinal == 0 {
                    (
                        matches!(row.instruction_kind, Kind::PcallCall(_)),
                        state_register,
                        RAM_START_ADDRESS as usize,
                    )
                } else if ordinal == length - 1 {
                    (
                        matches!(row.instruction_kind, Kind::PcallReturn(_)),
                        if flavour == 1 {
                            block_register
                        } else {
                            state_register
                        },
                        RAM_START_ADDRESS as usize,
                    )
                } else if ordinal < writes {
                    (
                        matches!(row.instruction_kind, Kind::PcallRead(_)),
                        block_register,
                        8 * (ordinal - 1),
                    )
                } else {
                    (
                        matches!(row.instruction_kind, Kind::PcallWrite(_)),
                        state_register,
                        8 * (ordinal - writes),
                    )
                };
                if !expected.0
                    || row.operands.rs1 != expected.1
                    || expected.1.is_none()
                    || row.operands.imm != expected.2 as i128
                    || row.operands.rd.is_some()
                    || row.operands.rs2.is_some()
                    || row.address != call.address
                    || row.is_compressed
                    || row.is_first_in_sequence != (ordinal == 0)
                    || row.virtual_sequence_remaining != Some((length - ordinal - 1) as u16)
                {
                    return Err(PreprocessingError::InvalidPcallSequence {
                        bytecode_index: cursor + ordinal,
                    });
                }
                let target = rows.get_mut(cursor + ordinal).ok_or(
                    PreprocessingError::InvalidPcallSequence {
                        bytecode_index: cursor + ordinal,
                    },
                )?;
                *target = PcallRow {
                    family: 1,
                    flavour,
                    ordinal: ordinal as u8,
                };
            }
            cursor += length;
        }
        Ok(Self { rows })
    }
}
