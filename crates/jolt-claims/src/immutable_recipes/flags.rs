use jolt_riscv::{CircuitFlags, InstructionFlags};

use super::{KeySink, KeyVisit};

impl KeyVisit for CircuitFlags {
    fn visit(&self, sink: &mut dyn KeySink) -> bool {
        sink.word(u64::from(*self as u8))
    }
}

impl KeyVisit for InstructionFlags {
    fn visit(&self, sink: &mut dyn KeySink) -> bool {
        sink.word(u64::from(*self as u8))
    }
}

#[cfg(all(feature = "immutable-recipes", not(target_arch = "riscv64")))]
impl super::record::RenderId for CircuitFlags {
    fn render(&self, writer: &mut dyn std::fmt::Write) -> std::fmt::Result {
        let variant = match self {
            Self::AddOperands => "AddOperands",
            Self::SubtractOperands => "SubtractOperands",
            Self::MultiplyOperands => "MultiplyOperands",
            Self::Load => "Load",
            Self::Store => "Store",
            Self::Jump => "Jump",
            Self::WriteLookupOutputToRD => "WriteLookupOutputToRD",
            Self::VirtualInstruction => "VirtualInstruction",
            Self::Assert => "Assert",
            Self::DoNotUpdateUnexpandedPC => "DoNotUpdateUnexpandedPC",
            Self::Advice => "Advice",
            Self::IsCompressed => "IsCompressed",
            Self::IsFirstInSequence => "IsFirstInSequence",
            Self::IsLastInSequence => "IsLastInSequence",
            Self::FieldAdd => "FieldAdd",
            Self::FieldSub => "FieldSub",
            Self::FieldMul => "FieldMul",
            Self::FieldInv => "FieldInv",
            Self::FieldAssertEq => "FieldAssertEq",
            Self::FieldLoadAccumulateFromRegister => "FieldLoadAccumulateFromRegister",
            Self::FieldAssertZero => "FieldAssertZero",
            Self::FieldLoadImm => "FieldLoadImm",
            Self::FieldLoadAccumulateFromMemory => "FieldLoadAccumulateFromMemory",
            Self::FieldAdviceLimb => "FieldAdviceLimb",
        };
        write!(writer, "jolt_riscv::CircuitFlags::{variant}")
    }
}

#[cfg(not(target_arch = "riscv64"))]
impl super::record::RenderId for InstructionFlags {
    fn render(&self, writer: &mut dyn std::fmt::Write) -> std::fmt::Result {
        write!(writer, "jolt_riscv::InstructionFlags::{self:?}")
    }
}
