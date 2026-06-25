use jolt_openings::{PackingAdviceKind, PackingFamilyId};
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::JoltAdviceKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum JoltPackingFamilyId {
    InstructionRa { index: usize },
    BytecodeRa { index: usize },
    RamRa { index: usize },
    UnsignedIncChunk { index: usize },
    UnsignedIncMsb,
    FieldRdIncByte { index: usize },
    FieldRdIncSign,
    AdviceBytes { kind: JoltAdviceKind, index: usize },
    BytecodeChunk { index: usize },
    BytecodeRegisterSelector { chunk: usize, selector: usize },
    BytecodeCircuitFlag { chunk: usize, flag: usize },
    BytecodeInstructionFlag { chunk: usize, flag: usize },
    BytecodeLookupSelector { chunk: usize },
    BytecodeRafFlag { chunk: usize },
    BytecodeUnexpandedPcBytes { chunk: usize },
    BytecodeImmBytes { chunk: usize },
    ProgramImageInit,
}

impl JoltPackingFamilyId {
    pub fn physical_id(self) -> PackingFamilyId {
        match self {
            Self::InstructionRa { index } => PackingFamilyId::InstructionRa { index },
            Self::BytecodeRa { index } => PackingFamilyId::BytecodeRa { index },
            Self::RamRa { index } => PackingFamilyId::RamRa { index },
            Self::UnsignedIncChunk { index } => PackingFamilyId::UnsignedIncChunk { index },
            Self::UnsignedIncMsb => PackingFamilyId::UnsignedIncMsb,
            Self::FieldRdIncByte { index } => PackingFamilyId::FieldRdIncByte { index },
            Self::FieldRdIncSign => PackingFamilyId::FieldRdIncSign,
            Self::AdviceBytes { kind, index } => PackingFamilyId::AdviceBytes {
                kind: packing_advice_kind(kind),
                index,
            },
            Self::BytecodeChunk { index } => PackingFamilyId::BytecodeChunk { index },
            Self::BytecodeRegisterSelector { chunk, selector } => {
                PackingFamilyId::BytecodeRegisterSelector { chunk, selector }
            }
            Self::BytecodeCircuitFlag { chunk, flag } => {
                PackingFamilyId::BytecodeCircuitFlag { chunk, flag }
            }
            Self::BytecodeInstructionFlag { chunk, flag } => {
                PackingFamilyId::BytecodeInstructionFlag { chunk, flag }
            }
            Self::BytecodeLookupSelector { chunk } => {
                PackingFamilyId::BytecodeLookupSelector { chunk }
            }
            Self::BytecodeRafFlag { chunk } => PackingFamilyId::BytecodeRafFlag { chunk },
            Self::BytecodeUnexpandedPcBytes { chunk } => {
                PackingFamilyId::BytecodeUnexpandedPcBytes { chunk }
            }
            Self::BytecodeImmBytes { chunk } => PackingFamilyId::BytecodeImmBytes { chunk },
            Self::ProgramImageInit => PackingFamilyId::ProgramImageInit,
        }
    }

    pub fn from_physical_id(family: &PackingFamilyId) -> Option<Self> {
        match family {
            PackingFamilyId::InstructionRa { index } => Some(Self::InstructionRa { index: *index }),
            PackingFamilyId::BytecodeRa { index } => Some(Self::BytecodeRa { index: *index }),
            PackingFamilyId::RamRa { index } => Some(Self::RamRa { index: *index }),
            PackingFamilyId::UnsignedIncChunk { index } => {
                Some(Self::UnsignedIncChunk { index: *index })
            }
            PackingFamilyId::UnsignedIncMsb => Some(Self::UnsignedIncMsb),
            PackingFamilyId::FieldRdIncByte { index } => {
                Some(Self::FieldRdIncByte { index: *index })
            }
            PackingFamilyId::FieldRdIncSign => Some(Self::FieldRdIncSign),
            PackingFamilyId::AdviceBytes { kind, index } => Some(Self::AdviceBytes {
                kind: jolt_advice_kind(*kind),
                index: *index,
            }),
            PackingFamilyId::BytecodeChunk { index } => Some(Self::BytecodeChunk { index: *index }),
            PackingFamilyId::BytecodeRegisterSelector { chunk, selector } => {
                Some(Self::BytecodeRegisterSelector {
                    chunk: *chunk,
                    selector: *selector,
                })
            }
            PackingFamilyId::BytecodeCircuitFlag { chunk, flag } => {
                Some(Self::BytecodeCircuitFlag {
                    chunk: *chunk,
                    flag: *flag,
                })
            }
            PackingFamilyId::BytecodeInstructionFlag { chunk, flag } => {
                Some(Self::BytecodeInstructionFlag {
                    chunk: *chunk,
                    flag: *flag,
                })
            }
            PackingFamilyId::BytecodeLookupSelector { chunk } => {
                Some(Self::BytecodeLookupSelector { chunk: *chunk })
            }
            PackingFamilyId::BytecodeRafFlag { chunk } => {
                Some(Self::BytecodeRafFlag { chunk: *chunk })
            }
            PackingFamilyId::BytecodeUnexpandedPcBytes { chunk } => {
                Some(Self::BytecodeUnexpandedPcBytes { chunk: *chunk })
            }
            PackingFamilyId::BytecodeImmBytes { chunk } => {
                Some(Self::BytecodeImmBytes { chunk: *chunk })
            }
            PackingFamilyId::ProgramImageInit => Some(Self::ProgramImageInit),
            PackingFamilyId::Custom { .. } => None,
        }
    }
}

impl From<JoltPackingFamilyId> for PackingFamilyId {
    fn from(family: JoltPackingFamilyId) -> Self {
        family.physical_id()
    }
}

pub fn packing_advice_kind(kind: JoltAdviceKind) -> PackingAdviceKind {
    match kind {
        JoltAdviceKind::Trusted => PackingAdviceKind::Trusted,
        JoltAdviceKind::Untrusted => PackingAdviceKind::Untrusted,
    }
}

pub fn jolt_advice_kind(kind: PackingAdviceKind) -> JoltAdviceKind {
    match kind {
        PackingAdviceKind::Trusted => JoltAdviceKind::Trusted,
        PackingAdviceKind::Untrusted => JoltAdviceKind::Untrusted,
    }
}
