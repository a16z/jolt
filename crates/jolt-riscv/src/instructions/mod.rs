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
pub mod c;
pub mod i;
pub mod m;
pub mod virt;

pub use i::Add;
pub use i::AddW;
pub use i::Addi;
pub use i::AddiW;
pub use i::And;
pub use i::AndI;
pub use i::Andn;
pub use assert::AssertEq;
pub use assert::AssertHalfwordAlignment;
pub use assert::AssertLte;
pub use assert::AssertMulUNoOverflow;
pub use assert::AssertValidDiv0;
pub use assert::AssertValidUnsignedRemainder;
pub use assert::AssertWordAlignment;
pub use i::Auipc;
pub use i::Beq;
pub use i::Bge;
pub use i::BgeU;
pub use i::Blt;
pub use i::BltU;
pub use i::Bne;
pub use m::Div;
pub use m::DivU;
pub use m::DivUW;
pub use m::DivW;
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
pub use virt::MovSign;
pub use m::Mul;
pub use m::MulH;
pub use m::MulHSU;
pub use m::MulHU;
pub use virt::MulI;
pub use m::MulW;
pub use i::Noop;
pub use i::Or;
pub use i::OrI;
pub use virt::Pow2;
pub use virt::Pow2I;
pub use virt::Pow2IW;
pub use virt::Pow2W;
pub use m::Rem;
pub use m::RemU;
pub use m::RemUW;
pub use m::RemW;
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
pub use i::Xor;
pub use i::XorI;

/// Enum with one variant per Jolt instruction.
///
/// Each variant carries the corresponding unit struct, enabling trait-based
/// dispatch (e.g. via `Flags` or `InstructionLookupTable`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, strum::EnumIter)]
pub enum JoltInstructions {
    Add(Add),
    Addi(Addi),
    Sub(Sub),
    Lui(Lui),
    Auipc(Auipc),
    Mul(Mul),
    MulH(MulH),
    MulHSU(MulHSU),
    MulHU(MulHU),
    Div(Div),
    DivU(DivU),
    Rem(Rem),
    RemU(RemU),
    AddW(AddW),
    AddiW(AddiW),
    SubW(SubW),
    MulW(MulW),
    DivW(DivW),
    DivUW(DivUW),
    RemW(RemW),
    RemUW(RemUW),
    And(And),
    AndI(AndI),
    Or(Or),
    OrI(OrI),
    Xor(Xor),
    XorI(XorI),
    Andn(Andn),
    Sll(Sll),
    SllI(SllI),
    Srl(Srl),
    SrlI(SrlI),
    Sra(Sra),
    SraI(SraI),
    SllW(SllW),
    SllIW(SllIW),
    SrlW(SrlW),
    SrlIW(SrlIW),
    SraW(SraW),
    SraIW(SraIW),
    Slt(Slt),
    SltI(SltI),
    SltU(SltU),
    SltIU(SltIU),
    Beq(Beq),
    Bne(Bne),
    Blt(Blt),
    Bge(Bge),
    BltU(BltU),
    BgeU(BgeU),
    Lb(Lb),
    Lbu(Lbu),
    Lh(Lh),
    Lhu(Lhu),
    Lw(Lw),
    Lwu(Lwu),
    Ld(Ld),
    Sb(Sb),
    Sh(Sh),
    Sw(Sw),
    Sd(Sd),
    Ecall(Ecall),
    Ebreak(Ebreak),
    Fence(Fence),
    Noop(Noop),
    Jal(Jal),
    Jalr(Jalr),
    AssertEq(AssertEq),
    AssertLte(AssertLte),
    AssertValidDiv0(AssertValidDiv0),
    AssertValidUnsignedRemainder(AssertValidUnsignedRemainder),
    AssertMulUNoOverflow(AssertMulUNoOverflow),
    AssertWordAlignment(AssertWordAlignment),
    AssertHalfwordAlignment(AssertHalfwordAlignment),
    Pow2(Pow2),
    Pow2I(Pow2I),
    Pow2W(Pow2W),
    Pow2IW(Pow2IW),
    MulI(MulI),
    MovSign(MovSign),
    VirtualRev8W(VirtualRev8W),
    VirtualChangeDivisor(VirtualChangeDivisor),
    VirtualChangeDivisorW(VirtualChangeDivisorW),
    VirtualSignExtendWord(VirtualSignExtendWord),
    VirtualZeroExtendWord(VirtualZeroExtendWord),
    VirtualSrl(VirtualSrl),
    VirtualSrli(VirtualSrli),
    VirtualSra(VirtualSra),
    VirtualSrai(VirtualSrai),
    VirtualShiftRightBitmask(VirtualShiftRightBitmask),
    VirtualShiftRightBitmaski(VirtualShiftRightBitmaski),
    VirtualRotri(VirtualRotri),
    VirtualRotriw(VirtualRotriw),
    VirtualXorRot32(VirtualXorRot32),
    VirtualXorRot24(VirtualXorRot24),
    VirtualXorRot16(VirtualXorRot16),
    VirtualXorRot63(VirtualXorRot63),
    VirtualXorRotW16(VirtualXorRotW16),
    VirtualXorRotW12(VirtualXorRotW12),
    VirtualXorRotW8(VirtualXorRotW8),
    VirtualXorRotW7(VirtualXorRotW7),
    VirtualAdvice(VirtualAdvice),
    VirtualAdviceLen(VirtualAdviceLen),
    VirtualAdviceLoad(VirtualAdviceLoad),
    VirtualHostIO(VirtualHostIO),
}

macro_rules! impl_jolt_instructions_flags {
    ($($variant:ident),* $(,)?) => {
        impl crate::flags::Flags for JoltInstructions {
            fn circuit_flags(&self) -> crate::flags::CircuitFlagSet {
                match self {
                    $(JoltInstructions::$variant(t) => t.circuit_flags(),)*
                }
            }

            fn instruction_flags(&self) -> crate::flags::InstructionFlagSet {
                match self {
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
    Ecall, Ebreak, Fence, Noop,
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
