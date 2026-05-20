//! Jolt RISC-V instruction types and their static metadata.
//!
//! Organized by ISA extension (`i`, `m`, `a`, `c`) plus `virt` (virtual
//! instructions emitted by the tracer) and `assert` (virtual asserts used
//! inside virtual sequences). Each `Foo<T = ()>(pub T)` represents an
//! instruction kind: with the default `T = ()` it is a zero-sized marker
//! (used by `JoltInstruction` variants and static-flag tests); with `T` set
//! to an `Instruction`/`Cycle` payload it becomes the constructed form used
//! by `LookupQuery` impls. `#[derive(Flags)]` declares the R1CS circuit and
//! witness-generation flags. The `InstructionLookupTable` impls (in
//! `jolt-lookup-tables`) map instructions to lookup tables.

#[cfg(feature = "serialization")]
use serde::ser::SerializeStruct;
#[cfg(feature = "serialization")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub mod a;
pub mod assert;
pub mod i;
pub mod m;
pub mod virt;

use crate::{
    JoltInstructionKind, JoltInstructionRow, NormalizedOperands, SourceInlineKey,
    SourceInstructionKind, SourceInstructionRow,
};
pub use assert::AssertEq;
pub use assert::AssertHalfwordAlignment;
pub use assert::AssertLte;
pub use assert::AssertMulUNoOverflow;
pub use assert::AssertValidDiv0;
pub use assert::AssertValidUnsignedRemainder;
pub use assert::AssertWordAlignment;
pub use i::Add;
pub use i::AddW;
pub use i::Addi;
pub use i::AddiW;
pub use i::And;
pub use i::AndI;
pub use i::Andn;
pub use i::Auipc;
pub use i::Beq;
pub use i::Bge;
pub use i::BgeU;
pub use i::Blt;
pub use i::BltU;
pub use i::Bne;
pub use i::Ebreak;
pub use i::Ecall;
pub use i::Fence;
pub use i::Jal;
pub use i::Jalr;
pub use i::Lb;
pub use i::Lbu;
pub use i::Ld;
pub use i::Lh;
pub use i::Lhu;
pub use i::Lui;
pub use i::Lw;
pub use i::Lwu;
pub use i::Noop;
pub use i::Or;
pub use i::OrI;
pub use i::Sb;
pub use i::Sd;
pub use i::Sh;
pub use i::Sll;
pub use i::SllI;
pub use i::SllIW;
pub use i::SllW;
pub use i::Slt;
pub use i::SltI;
pub use i::SltIU;
pub use i::SltU;
pub use i::Sra;
pub use i::SraI;
pub use i::SraIW;
pub use i::SraW;
pub use i::Srl;
pub use i::SrlI;
pub use i::SrlIW;
pub use i::SrlW;
pub use i::Sub;
pub use i::SubW;
pub use i::Sw;
pub use i::Xor;
pub use i::XorI;
pub use m::Div;
pub use m::DivU;
pub use m::DivUW;
pub use m::DivW;
pub use m::Mul;
pub use m::MulH;
pub use m::MulHSU;
pub use m::MulHU;
pub use m::MulW;
pub use m::Rem;
pub use m::RemU;
pub use m::RemUW;
pub use m::RemW;
pub use virt::MovSign;
pub use virt::MulI;
pub use virt::Pow2;
pub use virt::Pow2I;
pub use virt::Pow2IW;
pub use virt::Pow2W;
pub use virt::VirtualAdvice;
pub use virt::VirtualAdviceLen;
pub use virt::VirtualAdviceLoad;
pub use virt::VirtualChangeDivisor;
pub use virt::VirtualChangeDivisorW;
pub use virt::VirtualHostIO;
pub use virt::VirtualRev8W;
pub use virt::VirtualRotri;
pub use virt::VirtualRotriw;
pub use virt::VirtualShiftRightBitmask;
pub use virt::VirtualShiftRightBitmaski;
pub use virt::VirtualSignExtendWord;
pub use virt::VirtualSra;
pub use virt::VirtualSrai;
pub use virt::VirtualSrl;
pub use virt::VirtualSrli;
pub use virt::VirtualXorRot16;
pub use virt::VirtualXorRot24;
pub use virt::VirtualXorRot32;
pub use virt::VirtualXorRot63;
pub use virt::VirtualXorRotW12;
pub use virt::VirtualXorRotW16;
pub use virt::VirtualXorRotW7;
pub use virt::VirtualXorRotW8;
pub use virt::VirtualZeroExtendWord;

// Atomic + system + advice-load + virtual lw/sw additions
pub use a::AmoAddD;
pub use a::AmoAddW;
pub use a::AmoAndD;
pub use a::AmoAndW;
pub use a::AmoMaxD;
pub use a::AmoMaxUD;
pub use a::AmoMaxUW;
pub use a::AmoMaxW;
pub use a::AmoMinD;
pub use a::AmoMinUD;
pub use a::AmoMinUW;
pub use a::AmoMinW;
pub use a::AmoOrD;
pub use a::AmoOrW;
pub use a::AmoSwapD;
pub use a::AmoSwapW;
pub use a::AmoXorD;
pub use a::AmoXorW;
pub use a::LrD;
pub use a::LrW;
pub use a::ScD;
pub use a::ScW;
pub use i::Csrrs;
pub use i::Csrrw;
pub use i::Mret;
pub use virt::AdviceLb;
pub use virt::AdviceLd;
pub use virt::AdviceLh;
pub use virt::AdviceLw;
pub use virt::VirtualLw;
pub use virt::VirtualSw;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct Unimpl<T = ()>(pub T);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct Inline<T = ()>(pub T);

macro_rules! define_source_instruction {
    (
        instructions: [$($instr:ident => $marker:ident => $canonical_name:expr),* $(,)?]
    ) => {
        /// Typed view over decoded source rows.
        ///
        /// This is the source-side phase boundary: decode produces this enum,
        /// expansion consumes it, and final bytecode is emitted as
        /// [`JoltInstructionRow`](crate::JoltInstructionRow). The enum variant is the source
        /// instruction identity; the row payload carries only row data, so the
        /// two cannot silently disagree.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum SourceInstruction<T = SourceInstructionRow> {
            Noop(Noop<T>),
            Unimplemented(Unimpl<T>),
            $(
                $marker($marker<T>),
            )*
            InlineDispatch(Inline<T>),
        }

        impl SourceInstruction<SourceInstructionRow> {
            pub const fn kind(&self) -> SourceInstructionKind {
                match self {
                    Self::Noop(_) => SourceInstruction::Noop(Noop(())),
                    Self::Unimplemented(_) => SourceInstruction::Unimplemented(Unimpl(())),
                    $(
                        Self::$marker(_) => SourceInstruction::$marker($marker(())),
                    )*
                    Self::InlineDispatch(_) => SourceInstruction::InlineDispatch(Inline(())),
                }
            }

            pub fn new(kind: SourceInstructionKind, row: SourceInstructionRow) -> Self {
                match kind {
                    SourceInstruction::Noop(_) => Self::Noop(Noop(row)),
                    SourceInstruction::Unimplemented(_) => Self::Unimplemented(Unimpl(row)),
                    $(
                        SourceInstruction::$marker(_) => Self::$marker($marker(row)),
                    )*
                    SourceInstruction::InlineDispatch(_) => Self::InlineDispatch(Inline(row)),
                }
            }

            pub const fn row(&self) -> &SourceInstructionRow {
                match self {
                    Self::Noop(instruction) => &instruction.0,
                    Self::Unimplemented(instruction) => &instruction.0,
                    $(
                        Self::$marker(instruction) => &instruction.0,
                    )*
                    Self::InlineDispatch(instruction) => &instruction.0,
                }
            }

            pub fn into_row(self) -> SourceInstructionRow {
                match self {
                    Self::Noop(instruction) => instruction.0,
                    Self::Unimplemented(instruction) => instruction.0,
                    $(
                        Self::$marker(instruction) => instruction.0,
                    )*
                    Self::InlineDispatch(instruction) => instruction.0,
                }
            }

            pub fn map_row(self, f: impl FnOnce(SourceInstructionRow) -> SourceInstructionRow) -> Self {
                let kind = self.kind();
                Self::new(kind, f(self.into_row()))
            }
        }

        impl TryFrom<&SourceInstruction<SourceInstructionRow>> for JoltInstructionRow {
            type Error = SourceInstructionKind;

            fn try_from(instruction: &SourceInstruction<SourceInstructionRow>) -> Result<Self, Self::Error> {
                let source_kind = instruction.kind();
                let Some(jolt_kind) = source_kind.jolt_kind() else {
                    return Err(source_kind);
                };
                let row = instruction.row().jolt_instruction_row(jolt_kind);
                JoltInstruction::try_from(row)
                    .map(|_| row)
                    .map_err(|_| source_kind)
            }
        }

        #[cfg(feature = "serialization")]
        impl Serialize for SourceInstruction<SourceInstructionRow> {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                let row = self.row();
                let mut state = serializer.serialize_struct("SourceInstruction", 5)?;
                state.serialize_field("instruction_kind", &self.kind())?;
                state.serialize_field("address", &row.address)?;
                state.serialize_field("operands", &row.operands)?;
                state.serialize_field("inline", &row.inline)?;
                state.serialize_field("is_compressed", &row.is_compressed)?;
                state.end()
            }
        }

        #[cfg(feature = "serialization")]
        impl<'de> Deserialize<'de> for SourceInstruction<SourceInstructionRow> {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                #[derive(Deserialize)]
                struct SerializedSourceInstruction {
                    instruction_kind: SourceInstructionKind,
                    address: usize,
                    operands: NormalizedOperands,
                    #[serde(default)]
                    inline: Option<SourceInlineKey>,
                    is_compressed: bool,
                }

                let instruction = SerializedSourceInstruction::deserialize(deserializer)?;
                Ok(Self::new(
                    instruction.instruction_kind,
                    SourceInstructionRow {
                        address: instruction.address,
                        operands: instruction.operands,
                        inline: instruction.inline,
                        is_compressed: instruction.is_compressed,
                    },
                ))
            }
        }
    };
}

crate::for_each_instruction_kind!(define_source_instruction);

/// Typed view over expanded rows that have static lookup/circuit metadata.
///
/// Each variant wraps an instruction newtype parameterized by the canonical
/// [`JoltInstructionRow`](crate::JoltInstructionRow) row. Static-flag
/// dispatch and the flag-exclusivity tests rely on this concretization to
/// satisfy `T: JoltInstructionRowData` on the `Flags` impls.
///
/// Deliberately omitted instruction kinds (declared and re-exported above
/// but not proven by Jolt): the Zicsr ops (`Csrrs`, `Csrrw`), `Mret`,
/// the atomic family (`Amo*`, `Lr*`, `Sc*`),
/// the advice-load helpers (`AdviceLb`/`Ld`/`Lh`/`Lw`), and `VirtualLw` /
/// `VirtualSw`. These are intentionally absent from `JoltInstruction` and
/// from the flag-exclusivity tests below.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum JoltInstruction<T = JoltInstructionRow> {
    Noop(Noop<T>),
    Add(Add<T>),
    Addi(Addi<T>),
    Sub(Sub<T>),
    Lui(Lui<T>),
    Auipc(Auipc<T>),
    Mul(Mul<T>),
    MulHU(MulHU<T>),
    And(And<T>),
    AndI(AndI<T>),
    Or(Or<T>),
    OrI(OrI<T>),
    Xor(Xor<T>),
    XorI(XorI<T>),
    Andn(Andn<T>),
    Slt(Slt<T>),
    SltI(SltI<T>),
    SltU(SltU<T>),
    SltIU(SltIU<T>),
    Beq(Beq<T>),
    Bne(Bne<T>),
    Blt(Blt<T>),
    Bge(Bge<T>),
    BltU(BltU<T>),
    BgeU(BgeU<T>),
    Ld(Ld<T>),
    Sd(Sd<T>),
    Fence(Fence<T>),
    Jal(Jal<T>),
    Jalr(Jalr<T>),
    AssertEq(AssertEq<T>),
    AssertLte(AssertLte<T>),
    AssertValidDiv0(AssertValidDiv0<T>),
    AssertValidUnsignedRemainder(AssertValidUnsignedRemainder<T>),
    AssertMulUNoOverflow(AssertMulUNoOverflow<T>),
    AssertWordAlignment(AssertWordAlignment<T>),
    AssertHalfwordAlignment(AssertHalfwordAlignment<T>),
    Pow2(Pow2<T>),
    Pow2I(Pow2I<T>),
    Pow2W(Pow2W<T>),
    Pow2IW(Pow2IW<T>),
    MulI(MulI<T>),
    MovSign(MovSign<T>),
    VirtualRev8W(VirtualRev8W<T>),
    VirtualChangeDivisor(VirtualChangeDivisor<T>),
    VirtualChangeDivisorW(VirtualChangeDivisorW<T>),
    VirtualSignExtendWord(VirtualSignExtendWord<T>),
    VirtualZeroExtendWord(VirtualZeroExtendWord<T>),
    VirtualSrl(VirtualSrl<T>),
    VirtualSrli(VirtualSrli<T>),
    VirtualSra(VirtualSra<T>),
    VirtualSrai(VirtualSrai<T>),
    VirtualShiftRightBitmask(VirtualShiftRightBitmask<T>),
    VirtualShiftRightBitmaski(VirtualShiftRightBitmaski<T>),
    VirtualRotri(VirtualRotri<T>),
    VirtualRotriw(VirtualRotriw<T>),
    VirtualXorRot32(VirtualXorRot32<T>),
    VirtualXorRot24(VirtualXorRot24<T>),
    VirtualXorRot16(VirtualXorRot16<T>),
    VirtualXorRot63(VirtualXorRot63<T>),
    VirtualXorRotW16(VirtualXorRotW16<T>),
    VirtualXorRotW12(VirtualXorRotW12<T>),
    VirtualXorRotW8(VirtualXorRotW8<T>),
    VirtualXorRotW7(VirtualXorRotW7<T>),
    VirtualAdvice(VirtualAdvice<T>),
    VirtualAdviceLen(VirtualAdviceLen<T>),
    VirtualAdviceLoad(VirtualAdviceLoad<T>),
    VirtualHostIO(VirtualHostIO<T>),
}

macro_rules! impl_jolt_instruction_try_from_row {
    (
        instructions: [$($instr:ident => $marker:ident => ($tag:expr, $canonical_name:expr)),* $(,)?]
    ) => {
        impl TryFrom<JoltInstructionRow> for JoltInstruction {
            type Error = JoltInstructionKind;

            fn try_from(instruction: JoltInstructionRow) -> Result<Self, Self::Error> {
                Ok(match instruction.instruction_kind {
                    JoltInstruction::Noop(_) => Self::Noop(Noop(instruction)),
                    $(
                        JoltInstruction::$marker(_) => Self::$marker($marker(instruction)),
                    )*
                })
            }
        }
    };
}

crate::for_each_jolt_instruction_kind!(impl_jolt_instruction_try_from_row);

macro_rules! impl_jolt_instructions_flags {
    ($($variant:ident => $kind:ident),* $(,)?) => {
        impl<T> JoltInstruction<T> {
            pub const fn kind(&self) -> JoltInstructionKind {
                match self {
                    Self::Noop(_) => JoltInstruction::Noop(Noop(())),
                    $(
                        Self::$variant(_) => JoltInstruction::$variant($variant(())),
                    )*
                }
            }

            pub const fn row(&self) -> &T {
                match self {
                    Self::Noop(instruction) => &instruction.0,
                    $(
                        Self::$variant(instruction) => &instruction.0,
                    )*
                }
            }

        }

        impl<T: crate::JoltInstructionRowData> JoltInstruction<T> {
            pub fn into_row(self) -> JoltInstructionRow {
                match self {
                    Self::Noop(instruction) => {
                        let mut row = instruction.0.into();
                        row.instruction_kind = JoltInstruction::Noop(Noop(()));
                        row
                    }
                    $(
                        Self::$variant(instruction) => {
                            let mut row = instruction.0.into();
                            row.instruction_kind = JoltInstruction::$variant($variant(()));
                            row
                        }
                    )*
                }
            }
        }

        impl<T: crate::JoltInstructionRowData> From<JoltInstruction<T>> for JoltInstructionRow {
            fn from(instruction: JoltInstruction<T>) -> Self {
                instruction.into_row()
            }
        }

        impl crate::flags::Flags for JoltInstruction<JoltInstructionRow> {
            fn circuit_flags(&self) -> crate::flags::CircuitFlagSet {
                match self {
                    JoltInstruction::Noop(_) =>{
                        crate::flags::CircuitFlagSet::default()
                            .set(crate::flags::CircuitFlags::DoNotUpdateUnexpandedPC)
                    },
                    $(JoltInstruction::$variant(t) => t.circuit_flags(),)*
                }
            }

            fn instruction_flags(&self) -> crate::flags::InstructionFlagSet {
                match self {
                    JoltInstruction::Noop(_) =>{
                        crate::flags::InstructionFlagSet::default()
                            .set(crate::flags::InstructionFlags::IsNoop)
                    },
                    $(JoltInstruction::$variant(t) => t.instruction_flags(),)*
                }
            }
        }

        impl JoltInstruction<JoltInstructionRow> {
            pub fn iter() -> impl Iterator<Item = Self> {
                [
                    Self::Noop(Noop(JoltInstructionRow::default())),
                    $(
                        Self::$variant($variant(JoltInstructionRow::default())),
                    )*
                ]
                .into_iter()
            }
        }
    };
}

impl_jolt_instructions_flags! {
    Add => ADD,
    Addi => ADDI,
    Sub => SUB,
    Lui => LUI,
    Auipc => AUIPC,
    Mul => MUL,
    MulHU => MULHU,
    And => AND,
    AndI => ANDI,
    Or => OR,
    OrI => ORI,
    Xor => XOR,
    XorI => XORI,
    Andn => ANDN,
    Slt => SLT,
    SltI => SLTI,
    SltU => SLTU,
    SltIU => SLTIU,
    Beq => BEQ,
    Bne => BNE,
    Blt => BLT,
    Bge => BGE,
    BltU => BLTU,
    BgeU => BGEU,
    Ld => LD,
    Sd => SD,
    Fence => FENCE,
    Jal => JAL,
    Jalr => JALR,
    AssertEq => VirtualAssertEQ,
    AssertLte => VirtualAssertLTE,
    AssertValidDiv0 => VirtualAssertValidDiv0,
    AssertValidUnsignedRemainder => VirtualAssertValidUnsignedRemainder,
    AssertMulUNoOverflow => VirtualAssertMulUNoOverflow,
    AssertWordAlignment => VirtualAssertWordAlignment,
    AssertHalfwordAlignment => VirtualAssertHalfwordAlignment,
    Pow2 => VirtualPow2,
    Pow2I => VirtualPow2I,
    Pow2W => VirtualPow2W,
    Pow2IW => VirtualPow2IW,
    MulI => VirtualMULI,
    MovSign => VirtualMovsign,
    VirtualRev8W => VirtualRev8W,
    VirtualChangeDivisor => VirtualChangeDivisor,
    VirtualChangeDivisorW => VirtualChangeDivisorW,
    VirtualSignExtendWord => VirtualSignExtendWord,
    VirtualZeroExtendWord => VirtualZeroExtendWord,
    VirtualSrl => VirtualSRL,
    VirtualSrli => VirtualSRLI,
    VirtualSra => VirtualSRA,
    VirtualSrai => VirtualSRAI,
    VirtualShiftRightBitmask => VirtualShiftRightBitmask,
    VirtualShiftRightBitmaski => VirtualShiftRightBitmaskI,
    VirtualRotri => VirtualROTRI,
    VirtualRotriw => VirtualROTRIW,
    VirtualXorRot32 => VirtualXORROT32,
    VirtualXorRot24 => VirtualXORROT24,
    VirtualXorRot16 => VirtualXORROT16,
    VirtualXorRot63 => VirtualXORROT63,
    VirtualXorRotW16 => VirtualXORROTW16,
    VirtualXorRotW12 => VirtualXORROTW12,
    VirtualXorRotW8 => VirtualXORROTW8,
    VirtualXorRotW7 => VirtualXORROTW7,
    VirtualAdvice => VirtualAdvice,
    VirtualAdviceLen => VirtualAdviceLen,
    VirtualAdviceLoad => VirtualAdviceLoad,
    VirtualHostIO => VirtualHostIO,
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::flags::{CircuitFlags, Flags, InstructionFlags};

    #[test]
    fn left_operand_exclusive() {
        for instr in JoltInstruction::iter() {
            let flags = instr.instruction_flags();
            assert!(
                !(flags[InstructionFlags::LeftOperandIsPC]
                    && flags[InstructionFlags::LeftOperandIsRs1Value]),
                "Left operand flags not exclusive for {instr:?}",
            );
        }
    }

    #[test]
    fn right_operand_exclusive() {
        for instr in JoltInstruction::iter() {
            let flags = instr.instruction_flags();
            assert!(
                !(flags[InstructionFlags::RightOperandIsRs2Value]
                    && flags[InstructionFlags::RightOperandIsImm]),
                "Right operand flags not exclusive for {instr:?}",
            );
        }
    }

    #[test]
    fn lookup_shape_exclusive() {
        for instr in JoltInstruction::iter() {
            let flags = instr.circuit_flags();
            let num_true = [
                flags[CircuitFlags::AddOperands],
                flags[CircuitFlags::SubtractOperands],
                flags[CircuitFlags::MultiplyOperands],
                flags[CircuitFlags::Advice],
            ]
            .iter()
            .filter(|&&b| b)
            .count();
            assert!(
                num_true <= 1,
                "Lookup shaping flags not exclusive for {instr:?}",
            );
        }
    }

    #[test]
    fn load_store_exclusive() {
        for instr in JoltInstruction::iter() {
            let flags = instr.circuit_flags();
            assert!(
                !(flags[CircuitFlags::Load] && flags[CircuitFlags::Store]),
                "Load/Store flags not exclusive for {instr:?}",
            );
        }
    }

    #[test]
    fn phase_specific_instruction_kinds_are_distinct() {
        let source_kind = crate::SourceInstructionKind::AMOADDW;

        assert_eq!(source_kind.jolt_kind(), None);
        assert!(source_kind.expands_to_jolt());
    }

    #[test]
    fn source_instruction_variant_is_the_source_identity() {
        let row = SourceInstructionRow {
            address: 0x8000_0000,
            operands: crate::NormalizedOperands {
                rd: Some(1),
                rs1: Some(2),
                rs2: Some(3),
                imm: 4,
            },
            inline: None,
            is_compressed: false,
        };

        let add = SourceInstruction::new(SourceInstructionKind::ADD, row);
        let beq = SourceInstruction::new(SourceInstructionKind::BEQ, row);

        assert_eq!(add.kind(), SourceInstructionKind::ADD);
        assert_eq!(beq.kind(), SourceInstructionKind::BEQ);
        assert_eq!(
            JoltInstructionRow::try_from(&add).map(|row| row.instruction_kind),
            Ok(JoltInstructionKind::ADD)
        );
        assert_eq!(
            JoltInstructionRow::try_from(&beq).map(|row| row.instruction_kind),
            Ok(JoltInstructionKind::BEQ)
        );
        assert!(matches!(add, SourceInstruction::Add(Add(..))));
        assert!(matches!(beq, SourceInstruction::Beq(Beq(..))));
    }

    #[test]
    fn source_instruction_uses_catalog_marker_types() {
        let row = SourceInstructionRow::default();

        assert!(matches!(
            SourceInstruction::new(SourceInstructionKind::VirtualAssertEQ, row),
            SourceInstruction::AssertEq(AssertEq(..))
        ));
        assert!(matches!(
            SourceInstruction::new(SourceInstructionKind::VirtualMULI, row),
            SourceInstruction::MulI(MulI(..))
        ));
        assert!(matches!(
            SourceInstruction::new(SourceInstructionKind::VirtualROTRI, row),
            SourceInstruction::VirtualRotri(VirtualRotri(..))
        ));
        assert!(matches!(
            SourceInstruction::new(SourceInstructionKind::AMOMAXUD, row),
            SourceInstruction::AmoMaxUD(AmoMaxUD(..))
        ));
        assert!(matches!(
            SourceInstruction::new(SourceInstructionKind::Inline, row),
            SourceInstruction::InlineDispatch(Inline(..))
        ));
        assert!(matches!(
            SourceInstruction::new(SourceInstructionKind::Unimpl, row),
            SourceInstruction::Unimplemented(Unimpl(..))
        ));
    }

    #[test]
    fn jolt_instruction_identifies_explicit_final_subset() {
        assert!(matches!(
            JoltInstruction::try_from(JoltInstructionRow {
                instruction_kind: JoltInstructionKind::ADD,
                ..Default::default()
            }),
            Ok(JoltInstruction::Add(..))
        ));
        for kind in [
            SourceInstructionKind::ADDW,
            SourceInstructionKind::DIV,
            SourceInstructionKind::LW,
            SourceInstructionKind::SW,
            SourceInstructionKind::AMOADDW,
            SourceInstructionKind::CSRRS,
            SourceInstructionKind::VirtualSW,
        ] {
            assert_eq!(
                JoltInstructionRow::try_from(&SourceInstruction::new(
                    kind,
                    SourceInstructionRow::default()
                )),
                Err(kind)
            );
        }
    }

    #[test]
    fn jolt_instruction_variant_normalizes_row_kind() {
        let row = JoltInstructionRow {
            instruction_kind: JoltInstructionKind::SUB,
            ..Default::default()
        };

        let normalized = JoltInstructionRow::from(JoltInstruction::Add(Add(row)));

        assert_eq!(normalized.instruction_kind, JoltInstructionKind::ADD);
    }

    #[test]
    fn terminal_virtual_instruction_marks_last_in_sequence() -> Result<(), JoltInstructionKind> {
        fn flags_for(
            row: JoltInstructionRow,
        ) -> Result<crate::CircuitFlagSet, JoltInstructionKind> {
            JoltInstruction::try_from(row).map(|instruction| instruction.circuit_flags())
        }

        let mut row = JoltInstructionRow {
            virtual_sequence_remaining: Some(0),
            ..Default::default()
        };

        row.instruction_kind = JoltInstructionKind::ADDI;
        let addi_flags = flags_for(row)?;
        assert!(addi_flags[CircuitFlags::VirtualInstruction]);
        assert!(addi_flags[CircuitFlags::IsLastInSequence]);

        row.instruction_kind = JoltInstructionKind::JALR;
        let jalr_flags = flags_for(row)?;
        assert!(jalr_flags[CircuitFlags::VirtualInstruction]);
        assert!(jalr_flags[CircuitFlags::IsLastInSequence]);

        row.virtual_sequence_remaining = Some(1);
        let nonterminal_flags = flags_for(row)?;
        assert!(nonterminal_flags[CircuitFlags::VirtualInstruction]);
        assert!(!nonterminal_flags[CircuitFlags::IsLastInSequence]);
        Ok(())
    }
}
