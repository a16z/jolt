//! Jolt RISC-V instruction types and their static metadata.
//!
//! Organized by ISA extension (`i`, `m`, `a`, `c`) plus `virt` (virtual
//! instructions emitted by the tracer) and `assert` (virtual asserts used
//! inside virtual sequences). Each `Foo<T = ()>(pub T)` represents an
//! instruction kind: with the default `T = ()` it is a zero-sized marker
//! (used by `JoltInstructions` variants and static-flag tests); with `T` set
//! to an `Instruction`/`Cycle` payload it becomes the constructed form used
//! by `LookupQuery` impls. `#[derive(Flags)]` declares the R1CS circuit and
//! witness-generation flags. The `InstructionLookupTable` impls (in
//! `jolt-lookup-tables`) map instructions to lookup tables.

use serde::{Deserialize, Serialize};

pub mod a;
pub mod assert;
pub mod i;
pub mod m;
pub mod virt;

use crate::NormalizedInstruction;
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

/// Enum with one variant per Jolt instruction.
///
/// Each variant wraps a Jolt newtype parameterized by the canonical
/// [`NormalizedInstruction`](crate::NormalizedInstruction) row. Static-flag
/// dispatch and the flag-exclusivity tests rely on this concretization to
/// satisfy `T: JoltInstruction` on the `Flags` impls.
///
/// Deliberately omitted instruction kinds (declared and re-exported above
/// but not proven by Jolt): the Zicsr ops (`Csrrs`, `Csrrw`), `Mret`,
/// the entire RV32A/RV64A atomic family (`Amo*`, `Lr*`, `Sc*`),
/// the advice-load helpers (`AdviceLb`/`Ld`/`Lh`/`Lw`), and `VirtualLw` /
/// `VirtualSw`. These are intentionally absent from `JoltInstructions` and
/// from the flag-exclusivity tests below.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, strum::EnumIter)]
pub enum JoltInstructions {
    Noop,
    Add(Add<NormalizedInstruction>),
    Addi(Addi<NormalizedInstruction>),
    Sub(Sub<NormalizedInstruction>),
    Lui(Lui<NormalizedInstruction>),
    Auipc(Auipc<NormalizedInstruction>),
    Mul(Mul<NormalizedInstruction>),
    MulH(MulH<NormalizedInstruction>),
    MulHSU(MulHSU<NormalizedInstruction>),
    MulHU(MulHU<NormalizedInstruction>),
    Div(Div<NormalizedInstruction>),
    DivU(DivU<NormalizedInstruction>),
    Rem(Rem<NormalizedInstruction>),
    RemU(RemU<NormalizedInstruction>),
    AddW(AddW<NormalizedInstruction>),
    AddiW(AddiW<NormalizedInstruction>),
    SubW(SubW<NormalizedInstruction>),
    MulW(MulW<NormalizedInstruction>),
    DivW(DivW<NormalizedInstruction>),
    DivUW(DivUW<NormalizedInstruction>),
    RemW(RemW<NormalizedInstruction>),
    RemUW(RemUW<NormalizedInstruction>),
    And(And<NormalizedInstruction>),
    AndI(AndI<NormalizedInstruction>),
    Or(Or<NormalizedInstruction>),
    OrI(OrI<NormalizedInstruction>),
    Xor(Xor<NormalizedInstruction>),
    XorI(XorI<NormalizedInstruction>),
    Andn(Andn<NormalizedInstruction>),
    Sll(Sll<NormalizedInstruction>),
    SllI(SllI<NormalizedInstruction>),
    Srl(Srl<NormalizedInstruction>),
    SrlI(SrlI<NormalizedInstruction>),
    Sra(Sra<NormalizedInstruction>),
    SraI(SraI<NormalizedInstruction>),
    SllW(SllW<NormalizedInstruction>),
    SllIW(SllIW<NormalizedInstruction>),
    SrlW(SrlW<NormalizedInstruction>),
    SrlIW(SrlIW<NormalizedInstruction>),
    SraW(SraW<NormalizedInstruction>),
    SraIW(SraIW<NormalizedInstruction>),
    Slt(Slt<NormalizedInstruction>),
    SltI(SltI<NormalizedInstruction>),
    SltU(SltU<NormalizedInstruction>),
    SltIU(SltIU<NormalizedInstruction>),
    Beq(Beq<NormalizedInstruction>),
    Bne(Bne<NormalizedInstruction>),
    Blt(Blt<NormalizedInstruction>),
    Bge(Bge<NormalizedInstruction>),
    BltU(BltU<NormalizedInstruction>),
    BgeU(BgeU<NormalizedInstruction>),
    Lb(Lb<NormalizedInstruction>),
    Lbu(Lbu<NormalizedInstruction>),
    Lh(Lh<NormalizedInstruction>),
    Lhu(Lhu<NormalizedInstruction>),
    Lw(Lw<NormalizedInstruction>),
    Lwu(Lwu<NormalizedInstruction>),
    Ld(Ld<NormalizedInstruction>),
    Sb(Sb<NormalizedInstruction>),
    Sh(Sh<NormalizedInstruction>),
    Sw(Sw<NormalizedInstruction>),
    Sd(Sd<NormalizedInstruction>),
    Ecall(Ecall<NormalizedInstruction>),
    Ebreak(Ebreak<NormalizedInstruction>),
    Fence(Fence<NormalizedInstruction>),
    Jal(Jal<NormalizedInstruction>),
    Jalr(Jalr<NormalizedInstruction>),
    AssertEq(AssertEq<NormalizedInstruction>),
    AssertLte(AssertLte<NormalizedInstruction>),
    AssertValidDiv0(AssertValidDiv0<NormalizedInstruction>),
    AssertValidUnsignedRemainder(AssertValidUnsignedRemainder<NormalizedInstruction>),
    AssertMulUNoOverflow(AssertMulUNoOverflow<NormalizedInstruction>),
    AssertWordAlignment(AssertWordAlignment<NormalizedInstruction>),
    AssertHalfwordAlignment(AssertHalfwordAlignment<NormalizedInstruction>),
    Pow2(Pow2<NormalizedInstruction>),
    Pow2I(Pow2I<NormalizedInstruction>),
    Pow2W(Pow2W<NormalizedInstruction>),
    Pow2IW(Pow2IW<NormalizedInstruction>),
    MulI(MulI<NormalizedInstruction>),
    MovSign(MovSign<NormalizedInstruction>),
    VirtualRev8W(VirtualRev8W<NormalizedInstruction>),
    VirtualChangeDivisor(VirtualChangeDivisor<NormalizedInstruction>),
    VirtualChangeDivisorW(VirtualChangeDivisorW<NormalizedInstruction>),
    VirtualSignExtendWord(VirtualSignExtendWord<NormalizedInstruction>),
    VirtualZeroExtendWord(VirtualZeroExtendWord<NormalizedInstruction>),
    VirtualSrl(VirtualSrl<NormalizedInstruction>),
    VirtualSrli(VirtualSrli<NormalizedInstruction>),
    VirtualSra(VirtualSra<NormalizedInstruction>),
    VirtualSrai(VirtualSrai<NormalizedInstruction>),
    VirtualShiftRightBitmask(VirtualShiftRightBitmask<NormalizedInstruction>),
    VirtualShiftRightBitmaski(VirtualShiftRightBitmaski<NormalizedInstruction>),
    VirtualRotri(VirtualRotri<NormalizedInstruction>),
    VirtualRotriw(VirtualRotriw<NormalizedInstruction>),
    VirtualXorRot32(VirtualXorRot32<NormalizedInstruction>),
    VirtualXorRot24(VirtualXorRot24<NormalizedInstruction>),
    VirtualXorRot16(VirtualXorRot16<NormalizedInstruction>),
    VirtualXorRot63(VirtualXorRot63<NormalizedInstruction>),
    VirtualXorRotW16(VirtualXorRotW16<NormalizedInstruction>),
    VirtualXorRotW12(VirtualXorRotW12<NormalizedInstruction>),
    VirtualXorRotW8(VirtualXorRotW8<NormalizedInstruction>),
    VirtualXorRotW7(VirtualXorRotW7<NormalizedInstruction>),
    VirtualAdvice(VirtualAdvice<NormalizedInstruction>),
    VirtualAdviceLen(VirtualAdviceLen<NormalizedInstruction>),
    VirtualAdviceLoad(VirtualAdviceLoad<NormalizedInstruction>),
    VirtualHostIO(VirtualHostIO<NormalizedInstruction>),
}

macro_rules! impl_jolt_instructions_flags {
    ($($variant:ident),* $(,)?) => {
        impl crate::flags::Flags for JoltInstructions {
            fn circuit_flags(&self) -> crate::flags::CircuitFlagSet {
                match self {
                    JoltInstructions::Noop =>{
                        crate::flags::CircuitFlagSet::default()
                            .set(crate::flags::CircuitFlags::DoNotUpdateUnexpandedPC)
                    },
                    $(JoltInstructions::$variant(t) => t.circuit_flags(),)*
                }
            }

            fn instruction_flags(&self) -> crate::flags::InstructionFlagSet {
                match self {
                    JoltInstructions::Noop =>{
                        crate::flags::InstructionFlagSet::default()
                            .set(crate::flags::InstructionFlags::IsNoop)
                    },
                    $(JoltInstructions::$variant(t) => t.instruction_flags(),)*
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
        for instr in JoltInstructions::iter() {
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
        for instr in JoltInstructions::iter() {
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
        for instr in JoltInstructions::iter() {
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
        for instr in JoltInstructions::iter() {
            let flags = instr.circuit_flags();
            assert!(
                !(flags[CircuitFlags::Load] && flags[CircuitFlags::Store]),
                "Load/Store flags not exclusive for {instr:?}",
            );
        }
    }
}
