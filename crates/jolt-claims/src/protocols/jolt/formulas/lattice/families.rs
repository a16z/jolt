use jolt_openings::{PackingAdviceKind, PackingFamilyId};
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::JoltAdviceKind;

const JOLT_PACKING_FAMILY_NAMESPACE: u64 = 0x6a6f_6c74_7063_7301;
const BYTECODE_REGISTER_SELECTOR_ID: u64 = 15;
const BYTECODE_CIRCUIT_FLAG_ID: u64 = 16;
const BYTECODE_INSTRUCTION_FLAG_ID: u64 = 17;

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
        let (id, index) = self.physical_parts();
        PackingFamilyId::new(JOLT_PACKING_FAMILY_NAMESPACE, id, index)
    }

    pub fn from_physical_id(family: &PackingFamilyId) -> Option<Self> {
        if family.namespace != JOLT_PACKING_FAMILY_NAMESPACE {
            return None;
        }
        let index = usize::try_from(family.index).ok()?;
        match family.id {
            0 => Some(Self::InstructionRa { index }),
            1 => Some(Self::BytecodeRa { index }),
            2 => Some(Self::RamRa { index }),
            3 => Some(Self::UnsignedIncChunk { index }),
            4 if family.index == 0 => Some(Self::UnsignedIncMsb),
            9 => Some(Self::FieldRdIncByte { index }),
            10 if family.index == 0 => Some(Self::FieldRdIncSign),
            11 => Some(Self::AdviceBytes {
                kind: JoltAdviceKind::Trusted,
                index,
            }),
            12 => Some(Self::AdviceBytes {
                kind: JoltAdviceKind::Untrusted,
                index,
            }),
            13 => Some(Self::BytecodeChunk { index }),
            14 if family.index == 0 => Some(Self::ProgramImageInit),
            BYTECODE_REGISTER_SELECTOR_ID => {
                let (chunk, selector) = split_two_indices(family.index)?;
                Some(Self::BytecodeRegisterSelector { chunk, selector })
            }
            BYTECODE_CIRCUIT_FLAG_ID => {
                let (chunk, flag) = split_two_indices(family.index)?;
                Some(Self::BytecodeCircuitFlag { chunk, flag })
            }
            BYTECODE_INSTRUCTION_FLAG_ID => {
                let (chunk, flag) = split_two_indices(family.index)?;
                Some(Self::BytecodeInstructionFlag { chunk, flag })
            }
            18 => Some(Self::BytecodeLookupSelector { chunk: index }),
            19 => Some(Self::BytecodeRafFlag { chunk: index }),
            20 => Some(Self::BytecodeUnexpandedPcBytes { chunk: index }),
            21 => Some(Self::BytecodeImmBytes { chunk: index }),
            _ => None,
        }
    }

    #[expect(
        clippy::expect_used,
        reason = "Jolt bytecode chunk and lane indices are protocol-sized and must fit in u32"
    )]
    fn physical_parts(self) -> (u64, u64) {
        match self {
            Self::InstructionRa { index } => (0, index as u64),
            Self::BytecodeRa { index } => (1, index as u64),
            Self::RamRa { index } => (2, index as u64),
            Self::UnsignedIncChunk { index } => (3, index as u64),
            Self::UnsignedIncMsb => (4, 0),
            Self::FieldRdIncByte { index } => (9, index as u64),
            Self::FieldRdIncSign => (10, 0),
            Self::AdviceBytes { kind, index } => match kind {
                JoltAdviceKind::Trusted => (11, index as u64),
                JoltAdviceKind::Untrusted => (12, index as u64),
            },
            Self::BytecodeChunk { index } => (13, index as u64),
            Self::ProgramImageInit => (14, 0),
            Self::BytecodeRegisterSelector { chunk, selector } => (
                BYTECODE_REGISTER_SELECTOR_ID,
                combine_two_indices(chunk, selector).expect("bytecode selector index fits in u32"),
            ),
            Self::BytecodeCircuitFlag { chunk, flag } => (
                BYTECODE_CIRCUIT_FLAG_ID,
                combine_two_indices(chunk, flag).expect("bytecode circuit flag index fits in u32"),
            ),
            Self::BytecodeInstructionFlag { chunk, flag } => (
                BYTECODE_INSTRUCTION_FLAG_ID,
                combine_two_indices(chunk, flag)
                    .expect("bytecode instruction flag index fits in u32"),
            ),
            Self::BytecodeLookupSelector { chunk } => (18, chunk as u64),
            Self::BytecodeRafFlag { chunk } => (19, chunk as u64),
            Self::BytecodeUnexpandedPcBytes { chunk } => (20, chunk as u64),
            Self::BytecodeImmBytes { chunk } => (21, chunk as u64),
        }
    }
}

fn combine_two_indices(left: usize, right: usize) -> Option<u64> {
    let left = u64::from(u32::try_from(left).ok()?);
    let right = u64::from(u32::try_from(right).ok()?);
    Some((left << 32) | right)
}

fn split_two_indices(value: u64) -> Option<(usize, usize)> {
    let left = usize::try_from(value >> 32).ok()?;
    let right = usize::try_from(value & u64::from(u32::MAX)).ok()?;
    Some((left, right))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytecode_subfamilies_round_trip_checked_composite_indices() {
        let family = JoltPackingFamilyId::BytecodeRegisterSelector {
            chunk: 7,
            selector: 5,
        };
        let physical = family.physical_id();
        assert_eq!(physical.namespace, JOLT_PACKING_FAMILY_NAMESPACE);
        assert_eq!(physical.id, BYTECODE_REGISTER_SELECTOR_ID);
        assert_eq!(physical.index, (7_u64 << 32) | 5);
        assert_eq!(
            JoltPackingFamilyId::from_physical_id(&physical),
            Some(family)
        );

        assert_eq!(
            JoltPackingFamilyId::from_physical_id(&PackingFamilyId::new(
                JOLT_PACKING_FAMILY_NAMESPACE,
                99,
                0,
            )),
            None
        );
    }
}
