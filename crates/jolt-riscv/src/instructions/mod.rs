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
    JoltInstructionKind, JoltRow, NormalizedOperands, SourceInline, SourceInstructionKind,
    SourceRow,
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
        instructions: [$($instr:ident => $marker:ident => ($tag:expr, $canonical_name:expr)),* $(,)?]
    ) => {
        /// Typed view over decoded source rows.
        ///
        /// This is the source-side phase boundary: decode produces this enum,
        /// expansion consumes it, and final bytecode is emitted as
        /// [`JoltRow`](crate::JoltRow). The enum variant is the source
        /// instruction identity; the row payload carries only row data, so the
        /// two cannot silently disagree.
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub enum SourceInstruction<T = SourceRow> {
            NoOp(Noop<T>),
            Unimpl(Unimpl<T>),
            $(
                $instr($marker<T>),
            )*
            Inline(Inline<T>),
        }

        impl SourceInstruction<SourceRow> {
            pub const fn kind(&self) -> SourceInstructionKind {
                match self {
                    Self::NoOp(_) => SourceInstructionKind::NoOp,
                    Self::Unimpl(_) => SourceInstructionKind::Unimpl,
                    $(
                        Self::$instr(_) => SourceInstructionKind::$instr,
                    )*
                    Self::Inline(_) => SourceInstructionKind::Inline,
                }
            }

            pub fn new(kind: SourceInstructionKind, row: SourceRow) -> Self {
                match kind {
                    SourceInstructionKind::NoOp => Self::NoOp(Noop(row)),
                    SourceInstructionKind::Unimpl => Self::Unimpl(Unimpl(row)),
                    $(
                        SourceInstructionKind::$instr => Self::$instr($marker(row)),
                    )*
                    SourceInstructionKind::Inline => Self::Inline(Inline(row)),
                }
            }

            pub const fn row(&self) -> &SourceRow {
                match self {
                    Self::NoOp(instruction) => &instruction.0,
                    Self::Unimpl(instruction) => &instruction.0,
                    $(
                        Self::$instr(instruction) => &instruction.0,
                    )*
                    Self::Inline(instruction) => &instruction.0,
                }
            }

            pub fn jolt_row(&self) -> JoltRow {
                self.row().jolt_row(self.kind().jolt_kind())
            }

            pub fn into_row(self) -> SourceRow {
                match self {
                    Self::NoOp(instruction) => instruction.0,
                    Self::Unimpl(instruction) => instruction.0,
                    $(
                        Self::$instr(instruction) => instruction.0,
                    )*
                    Self::Inline(instruction) => instruction.0,
                }
            }

            pub fn map_row(self, f: impl FnOnce(SourceRow) -> SourceRow) -> Self {
                let kind = self.kind();
                Self::new(kind, f(self.into_row()))
            }
        }

        impl From<SourceInstruction<SourceRow>> for JoltRow {
            fn from(instruction: SourceInstruction<SourceRow>) -> Self {
                instruction.jolt_row()
            }
        }

        #[cfg(feature = "serialization")]
        impl Serialize for SourceInstruction<SourceRow> {
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
        impl<'de> Deserialize<'de> for SourceInstruction<SourceRow> {
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
                    inline: Option<SourceInline>,
                    is_compressed: bool,
                }

                let instruction = SerializedSourceInstruction::deserialize(deserializer)?;
                Ok(Self::new(
                    instruction.instruction_kind,
                    SourceRow {
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
/// [`JoltRow`](crate::JoltRow) row. Static-flag
/// dispatch and the flag-exclusivity tests rely on this concretization to
/// satisfy `T: JoltRowData` on the `Flags` impls.
///
/// Deliberately omitted instruction kinds (declared and re-exported above
/// but not proven by Jolt): the Zicsr ops (`Csrrs`, `Csrrw`), `Mret`,
/// the atomic family (`Amo*`, `Lr*`, `Sc*`),
/// the advice-load helpers (`AdviceLb`/`Ld`/`Lh`/`Lw`), and `VirtualLw` /
/// `VirtualSw`. These are intentionally absent from `JoltInstruction` and
/// from the flag-exclusivity tests below.

#[derive(Clone, Copy, Debug, PartialEq, strum::EnumIter)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub enum JoltInstruction {
    Noop,
    Add(Add<JoltRow>),
    Addi(Addi<JoltRow>),
    Sub(Sub<JoltRow>),
    Lui(Lui<JoltRow>),
    Auipc(Auipc<JoltRow>),
    Mul(Mul<JoltRow>),
    MulH(MulH<JoltRow>),
    MulHSU(MulHSU<JoltRow>),
    MulHU(MulHU<JoltRow>),
    Div(Div<JoltRow>),
    DivU(DivU<JoltRow>),
    Rem(Rem<JoltRow>),
    RemU(RemU<JoltRow>),
    AddW(AddW<JoltRow>),
    AddiW(AddiW<JoltRow>),
    SubW(SubW<JoltRow>),
    MulW(MulW<JoltRow>),
    DivW(DivW<JoltRow>),
    DivUW(DivUW<JoltRow>),
    RemW(RemW<JoltRow>),
    RemUW(RemUW<JoltRow>),
    And(And<JoltRow>),
    AndI(AndI<JoltRow>),
    Or(Or<JoltRow>),
    OrI(OrI<JoltRow>),
    Xor(Xor<JoltRow>),
    XorI(XorI<JoltRow>),
    Andn(Andn<JoltRow>),
    Sll(Sll<JoltRow>),
    SllI(SllI<JoltRow>),
    Srl(Srl<JoltRow>),
    SrlI(SrlI<JoltRow>),
    Sra(Sra<JoltRow>),
    SraI(SraI<JoltRow>),
    SllW(SllW<JoltRow>),
    SllIW(SllIW<JoltRow>),
    SrlW(SrlW<JoltRow>),
    SrlIW(SrlIW<JoltRow>),
    SraW(SraW<JoltRow>),
    SraIW(SraIW<JoltRow>),
    Slt(Slt<JoltRow>),
    SltI(SltI<JoltRow>),
    SltU(SltU<JoltRow>),
    SltIU(SltIU<JoltRow>),
    Beq(Beq<JoltRow>),
    Bne(Bne<JoltRow>),
    Blt(Blt<JoltRow>),
    Bge(Bge<JoltRow>),
    BltU(BltU<JoltRow>),
    BgeU(BgeU<JoltRow>),
    Lb(Lb<JoltRow>),
    Lbu(Lbu<JoltRow>),
    Lh(Lh<JoltRow>),
    Lhu(Lhu<JoltRow>),
    Lw(Lw<JoltRow>),
    Lwu(Lwu<JoltRow>),
    Ld(Ld<JoltRow>),
    Sb(Sb<JoltRow>),
    Sh(Sh<JoltRow>),
    Sw(Sw<JoltRow>),
    Sd(Sd<JoltRow>),
    Ecall(Ecall<JoltRow>),
    Ebreak(Ebreak<JoltRow>),
    Fence(Fence<JoltRow>),
    Jal(Jal<JoltRow>),
    Jalr(Jalr<JoltRow>),
    AssertEq(AssertEq<JoltRow>),
    AssertLte(AssertLte<JoltRow>),
    AssertValidDiv0(AssertValidDiv0<JoltRow>),
    AssertValidUnsignedRemainder(AssertValidUnsignedRemainder<JoltRow>),
    AssertMulUNoOverflow(AssertMulUNoOverflow<JoltRow>),
    AssertWordAlignment(AssertWordAlignment<JoltRow>),
    AssertHalfwordAlignment(AssertHalfwordAlignment<JoltRow>),
    Pow2(Pow2<JoltRow>),
    Pow2I(Pow2I<JoltRow>),
    Pow2W(Pow2W<JoltRow>),
    Pow2IW(Pow2IW<JoltRow>),
    MulI(MulI<JoltRow>),
    MovSign(MovSign<JoltRow>),
    VirtualRev8W(VirtualRev8W<JoltRow>),
    VirtualChangeDivisor(VirtualChangeDivisor<JoltRow>),
    VirtualChangeDivisorW(VirtualChangeDivisorW<JoltRow>),
    VirtualSignExtendWord(VirtualSignExtendWord<JoltRow>),
    VirtualZeroExtendWord(VirtualZeroExtendWord<JoltRow>),
    VirtualSrl(VirtualSrl<JoltRow>),
    VirtualSrli(VirtualSrli<JoltRow>),
    VirtualSra(VirtualSra<JoltRow>),
    VirtualSrai(VirtualSrai<JoltRow>),
    VirtualShiftRightBitmask(VirtualShiftRightBitmask<JoltRow>),
    VirtualShiftRightBitmaski(VirtualShiftRightBitmaski<JoltRow>),
    VirtualRotri(VirtualRotri<JoltRow>),
    VirtualRotriw(VirtualRotriw<JoltRow>),
    VirtualXorRot32(VirtualXorRot32<JoltRow>),
    VirtualXorRot24(VirtualXorRot24<JoltRow>),
    VirtualXorRot16(VirtualXorRot16<JoltRow>),
    VirtualXorRot63(VirtualXorRot63<JoltRow>),
    VirtualXorRotW16(VirtualXorRotW16<JoltRow>),
    VirtualXorRotW12(VirtualXorRotW12<JoltRow>),
    VirtualXorRotW8(VirtualXorRotW8<JoltRow>),
    VirtualXorRotW7(VirtualXorRotW7<JoltRow>),
    VirtualAdvice(VirtualAdvice<JoltRow>),
    VirtualAdviceLen(VirtualAdviceLen<JoltRow>),
    VirtualAdviceLoad(VirtualAdviceLoad<JoltRow>),
    VirtualHostIO(VirtualHostIO<JoltRow>),
}

impl TryFrom<JoltRow> for JoltInstruction {
    type Error = JoltInstructionKind;

    fn try_from(instruction: JoltRow) -> Result<Self, Self::Error> {
        Ok(match instruction.instruction_kind {
            JoltInstructionKind::NoOp => Self::Noop,
            JoltInstructionKind::ADD => Self::Add(Add(instruction)),
            JoltInstructionKind::ADDI => Self::Addi(Addi(instruction)),
            JoltInstructionKind::SUB => Self::Sub(Sub(instruction)),
            JoltInstructionKind::LUI => Self::Lui(Lui(instruction)),
            JoltInstructionKind::AUIPC => Self::Auipc(Auipc(instruction)),
            JoltInstructionKind::MUL => Self::Mul(Mul(instruction)),
            JoltInstructionKind::MULH => Self::MulH(MulH(instruction)),
            JoltInstructionKind::MULHSU => Self::MulHSU(MulHSU(instruction)),
            JoltInstructionKind::MULHU => Self::MulHU(MulHU(instruction)),
            JoltInstructionKind::DIV => Self::Div(Div(instruction)),
            JoltInstructionKind::DIVU => Self::DivU(DivU(instruction)),
            JoltInstructionKind::REM => Self::Rem(Rem(instruction)),
            JoltInstructionKind::REMU => Self::RemU(RemU(instruction)),
            JoltInstructionKind::ADDW => Self::AddW(AddW(instruction)),
            JoltInstructionKind::ADDIW => Self::AddiW(AddiW(instruction)),
            JoltInstructionKind::SUBW => Self::SubW(SubW(instruction)),
            JoltInstructionKind::MULW => Self::MulW(MulW(instruction)),
            JoltInstructionKind::DIVW => Self::DivW(DivW(instruction)),
            JoltInstructionKind::DIVUW => Self::DivUW(DivUW(instruction)),
            JoltInstructionKind::REMW => Self::RemW(RemW(instruction)),
            JoltInstructionKind::REMUW => Self::RemUW(RemUW(instruction)),
            JoltInstructionKind::AND => Self::And(And(instruction)),
            JoltInstructionKind::ANDI => Self::AndI(AndI(instruction)),
            JoltInstructionKind::OR => Self::Or(Or(instruction)),
            JoltInstructionKind::ORI => Self::OrI(OrI(instruction)),
            JoltInstructionKind::XOR => Self::Xor(Xor(instruction)),
            JoltInstructionKind::XORI => Self::XorI(XorI(instruction)),
            JoltInstructionKind::ANDN => Self::Andn(Andn(instruction)),
            JoltInstructionKind::SLL => Self::Sll(Sll(instruction)),
            JoltInstructionKind::SLLI => Self::SllI(SllI(instruction)),
            JoltInstructionKind::SRL => Self::Srl(Srl(instruction)),
            JoltInstructionKind::SRLI => Self::SrlI(SrlI(instruction)),
            JoltInstructionKind::SRA => Self::Sra(Sra(instruction)),
            JoltInstructionKind::SRAI => Self::SraI(SraI(instruction)),
            JoltInstructionKind::SLLW => Self::SllW(SllW(instruction)),
            JoltInstructionKind::SLLIW => Self::SllIW(SllIW(instruction)),
            JoltInstructionKind::SRLW => Self::SrlW(SrlW(instruction)),
            JoltInstructionKind::SRLIW => Self::SrlIW(SrlIW(instruction)),
            JoltInstructionKind::SRAW => Self::SraW(SraW(instruction)),
            JoltInstructionKind::SRAIW => Self::SraIW(SraIW(instruction)),
            JoltInstructionKind::SLT => Self::Slt(Slt(instruction)),
            JoltInstructionKind::SLTI => Self::SltI(SltI(instruction)),
            JoltInstructionKind::SLTU => Self::SltU(SltU(instruction)),
            JoltInstructionKind::SLTIU => Self::SltIU(SltIU(instruction)),
            JoltInstructionKind::BEQ => Self::Beq(Beq(instruction)),
            JoltInstructionKind::BNE => Self::Bne(Bne(instruction)),
            JoltInstructionKind::BLT => Self::Blt(Blt(instruction)),
            JoltInstructionKind::BGE => Self::Bge(Bge(instruction)),
            JoltInstructionKind::BLTU => Self::BltU(BltU(instruction)),
            JoltInstructionKind::BGEU => Self::BgeU(BgeU(instruction)),
            JoltInstructionKind::LB => Self::Lb(Lb(instruction)),
            JoltInstructionKind::LBU => Self::Lbu(Lbu(instruction)),
            JoltInstructionKind::LH => Self::Lh(Lh(instruction)),
            JoltInstructionKind::LHU => Self::Lhu(Lhu(instruction)),
            JoltInstructionKind::LW => Self::Lw(Lw(instruction)),
            JoltInstructionKind::LWU => Self::Lwu(Lwu(instruction)),
            JoltInstructionKind::LD => Self::Ld(Ld(instruction)),
            JoltInstructionKind::SB => Self::Sb(Sb(instruction)),
            JoltInstructionKind::SH => Self::Sh(Sh(instruction)),
            JoltInstructionKind::SW => Self::Sw(Sw(instruction)),
            JoltInstructionKind::SD => Self::Sd(Sd(instruction)),
            JoltInstructionKind::ECALL => Self::Ecall(Ecall(instruction)),
            JoltInstructionKind::EBREAK => Self::Ebreak(Ebreak(instruction)),
            JoltInstructionKind::FENCE => Self::Fence(Fence(instruction)),
            JoltInstructionKind::JAL => Self::Jal(Jal(instruction)),
            JoltInstructionKind::JALR => Self::Jalr(Jalr(instruction)),
            JoltInstructionKind::VirtualAssertEQ => Self::AssertEq(AssertEq(instruction)),
            JoltInstructionKind::VirtualAssertLTE => Self::AssertLte(AssertLte(instruction)),
            JoltInstructionKind::VirtualAssertValidDiv0 => {
                Self::AssertValidDiv0(AssertValidDiv0(instruction))
            }
            JoltInstructionKind::VirtualAssertValidUnsignedRemainder => {
                Self::AssertValidUnsignedRemainder(AssertValidUnsignedRemainder(instruction))
            }
            JoltInstructionKind::VirtualAssertMulUNoOverflow => {
                Self::AssertMulUNoOverflow(AssertMulUNoOverflow(instruction))
            }
            JoltInstructionKind::VirtualAssertWordAlignment => {
                Self::AssertWordAlignment(AssertWordAlignment(instruction))
            }
            JoltInstructionKind::VirtualAssertHalfwordAlignment => {
                Self::AssertHalfwordAlignment(AssertHalfwordAlignment(instruction))
            }
            JoltInstructionKind::VirtualPow2 => Self::Pow2(Pow2(instruction)),
            JoltInstructionKind::VirtualPow2I => Self::Pow2I(Pow2I(instruction)),
            JoltInstructionKind::VirtualPow2W => Self::Pow2W(Pow2W(instruction)),
            JoltInstructionKind::VirtualPow2IW => Self::Pow2IW(Pow2IW(instruction)),
            JoltInstructionKind::VirtualMULI => Self::MulI(MulI(instruction)),
            JoltInstructionKind::VirtualMovsign => Self::MovSign(MovSign(instruction)),
            JoltInstructionKind::VirtualRev8W => Self::VirtualRev8W(VirtualRev8W(instruction)),
            JoltInstructionKind::VirtualChangeDivisor => {
                Self::VirtualChangeDivisor(VirtualChangeDivisor(instruction))
            }
            JoltInstructionKind::VirtualChangeDivisorW => {
                Self::VirtualChangeDivisorW(VirtualChangeDivisorW(instruction))
            }
            JoltInstructionKind::VirtualSignExtendWord => {
                Self::VirtualSignExtendWord(VirtualSignExtendWord(instruction))
            }
            JoltInstructionKind::VirtualZeroExtendWord => {
                Self::VirtualZeroExtendWord(VirtualZeroExtendWord(instruction))
            }
            JoltInstructionKind::VirtualSRL => Self::VirtualSrl(VirtualSrl(instruction)),
            JoltInstructionKind::VirtualSRLI => Self::VirtualSrli(VirtualSrli(instruction)),
            JoltInstructionKind::VirtualSRA => Self::VirtualSra(VirtualSra(instruction)),
            JoltInstructionKind::VirtualSRAI => Self::VirtualSrai(VirtualSrai(instruction)),
            JoltInstructionKind::VirtualShiftRightBitmask => {
                Self::VirtualShiftRightBitmask(VirtualShiftRightBitmask(instruction))
            }
            JoltInstructionKind::VirtualShiftRightBitmaskI => {
                Self::VirtualShiftRightBitmaski(VirtualShiftRightBitmaski(instruction))
            }
            JoltInstructionKind::VirtualROTRI => Self::VirtualRotri(VirtualRotri(instruction)),
            JoltInstructionKind::VirtualROTRIW => Self::VirtualRotriw(VirtualRotriw(instruction)),
            JoltInstructionKind::VirtualXORROT32 => {
                Self::VirtualXorRot32(VirtualXorRot32(instruction))
            }
            JoltInstructionKind::VirtualXORROT24 => {
                Self::VirtualXorRot24(VirtualXorRot24(instruction))
            }
            JoltInstructionKind::VirtualXORROT16 => {
                Self::VirtualXorRot16(VirtualXorRot16(instruction))
            }
            JoltInstructionKind::VirtualXORROT63 => {
                Self::VirtualXorRot63(VirtualXorRot63(instruction))
            }
            JoltInstructionKind::VirtualXORROTW16 => {
                Self::VirtualXorRotW16(VirtualXorRotW16(instruction))
            }
            JoltInstructionKind::VirtualXORROTW12 => {
                Self::VirtualXorRotW12(VirtualXorRotW12(instruction))
            }
            JoltInstructionKind::VirtualXORROTW8 => {
                Self::VirtualXorRotW8(VirtualXorRotW8(instruction))
            }
            JoltInstructionKind::VirtualXORROTW7 => {
                Self::VirtualXorRotW7(VirtualXorRotW7(instruction))
            }
            JoltInstructionKind::VirtualAdvice => Self::VirtualAdvice(VirtualAdvice(instruction)),
            JoltInstructionKind::VirtualAdviceLen => {
                Self::VirtualAdviceLen(VirtualAdviceLen(instruction))
            }
            JoltInstructionKind::VirtualAdviceLoad => {
                Self::VirtualAdviceLoad(VirtualAdviceLoad(instruction))
            }
            JoltInstructionKind::VirtualHostIO => Self::VirtualHostIO(VirtualHostIO(instruction)),
            unsupported => return Err(unsupported),
        })
    }
}

macro_rules! impl_jolt_instructions_flags {
    ($($variant:ident),* $(,)?) => {
        impl crate::flags::Flags for JoltInstruction {
            fn circuit_flags(&self) -> crate::flags::CircuitFlagSet {
                match self {
                    JoltInstruction::Noop =>{
                        crate::flags::CircuitFlagSet::default()
                            .set(crate::flags::CircuitFlags::DoNotUpdateUnexpandedPC)
                    },
                    $(JoltInstruction::$variant(t) => t.circuit_flags(),)*
                }
            }

            fn instruction_flags(&self) -> crate::flags::InstructionFlagSet {
                match self {
                    JoltInstruction::Noop =>{
                        crate::flags::InstructionFlagSet::default()
                            .set(crate::flags::InstructionFlags::IsNoop)
                    },
                    $(JoltInstruction::$variant(t) => t.instruction_flags(),)*
                }
            }
        }
    };
}

impl_jolt_instructions_flags! {
    Add, Addi, Sub, Lui, Auipc,
    Mul, MulH, MulHSU, MulHU, Div, DivU, Rem, RemU,
    AddW, AddiW, SubW, MulW, DivW, DivUW, RemW, RemUW,
    And, AndI, Or, OrI, Xor, XorI, Andn,
    Sll, SllI, Srl, SrlI, Sra, SraI,
    SllW, SllIW, SrlW, SrlIW, SraW, SraIW,
    Slt, SltI, SltU, SltIU,
    Beq, Bne, Blt, Bge, BltU, BgeU,
    Lb, Lbu, Lh, Lhu, Lw, Lwu, Ld,
    Sb, Sh, Sw, Sd,
    Ecall, Ebreak, Fence,
    Jal, Jalr,
    AssertEq, AssertLte, AssertValidDiv0, AssertValidUnsignedRemainder,
    AssertMulUNoOverflow, AssertWordAlignment, AssertHalfwordAlignment,
    Pow2, Pow2I, Pow2W, Pow2IW, MulI,
    MovSign, VirtualRev8W,
    VirtualChangeDivisor, VirtualChangeDivisorW,
    VirtualSignExtendWord, VirtualZeroExtendWord,
    VirtualSrl, VirtualSrli, VirtualSra, VirtualSrai,
    VirtualShiftRightBitmask, VirtualShiftRightBitmaski,
    VirtualRotri, VirtualRotriw,
    VirtualXorRot32, VirtualXorRot24, VirtualXorRot16, VirtualXorRot63,
    VirtualXorRotW16, VirtualXorRotW12, VirtualXorRotW8, VirtualXorRotW7,
    VirtualAdvice, VirtualAdviceLen, VirtualAdviceLoad, VirtualHostIO,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flags::{CircuitFlags, Flags, InstructionFlags};
    use strum::IntoEnumIterator;

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
        let jolt_kind = source_kind.jolt_kind();

        assert_eq!(jolt_kind, JoltInstructionKind::AMOADDW);
        assert!(source_kind.expands_to_jolt());
        assert_eq!(
            JoltInstruction::try_from(JoltRow {
                instruction_kind: jolt_kind,
                ..Default::default()
            }),
            Err(jolt_kind)
        );
    }

    #[test]
    fn source_instruction_variant_is_the_source_identity() {
        let row = SourceRow {
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
            JoltRow::from(add).instruction_kind,
            JoltInstructionKind::ADD
        );
        assert_eq!(
            JoltRow::from(beq).instruction_kind,
            JoltInstructionKind::BEQ
        );
        assert!(matches!(add, SourceInstruction::ADD(Add(..))));
        assert!(matches!(beq, SourceInstruction::BEQ(Beq(..))));
    }

    #[test]
    fn source_instruction_uses_catalog_marker_types() {
        let row = SourceRow::default();

        assert!(matches!(
            SourceInstruction::new(SourceInstructionKind::VirtualAssertEQ, row),
            SourceInstruction::VirtualAssertEQ(AssertEq(..))
        ));
        assert!(matches!(
            SourceInstruction::new(SourceInstructionKind::VirtualMULI, row),
            SourceInstruction::VirtualMULI(MulI(..))
        ));
        assert!(matches!(
            SourceInstruction::new(SourceInstructionKind::VirtualROTRI, row),
            SourceInstruction::VirtualROTRI(VirtualRotri(..))
        ));
        assert!(matches!(
            SourceInstruction::new(SourceInstructionKind::AMOMAXUD, row),
            SourceInstruction::AMOMAXUD(AmoMaxUD(..))
        ));
        assert!(matches!(
            SourceInstruction::new(SourceInstructionKind::Inline, row),
            SourceInstruction::Inline(Inline(..))
        ));
        assert!(matches!(
            SourceInstruction::new(SourceInstructionKind::Unimpl, row),
            SourceInstruction::Unimpl(Unimpl(..))
        ));
    }

    #[test]
    fn jolt_instruction_identifies_explicit_final_subset() {
        assert!(matches!(
            JoltInstruction::try_from(JoltRow {
                instruction_kind: JoltInstructionKind::ADD,
                ..Default::default()
            }),
            Ok(JoltInstruction::Add(..))
        ));
        for kind in [
            JoltInstructionKind::AMOADDW,
            JoltInstructionKind::CSRRS,
            JoltInstructionKind::VirtualSW,
        ] {
            assert_eq!(
                JoltInstruction::try_from(JoltRow {
                    instruction_kind: kind,
                    ..Default::default()
                }),
                Err(kind)
            );
        }
    }

    #[test]
    fn terminal_virtual_instruction_marks_last_in_sequence() -> Result<(), JoltInstructionKind> {
        fn flags_for(row: JoltRow) -> Result<crate::CircuitFlagSet, JoltInstructionKind> {
            JoltInstruction::try_from(row).map(|instruction| instruction.circuit_flags())
        }

        let mut row = JoltRow {
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
