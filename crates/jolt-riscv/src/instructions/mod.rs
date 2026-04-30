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
use tracer::instruction::{
    add::ADD,
    addi::ADDI,
    addiw::ADDIW,
    addw::ADDW,
    and::AND,
    andi::ANDI,
    andn::ANDN,
    auipc::AUIPC,
    beq::BEQ,
    bge::BGE,
    bgeu::BGEU,
    blt::BLT,
    bltu::BLTU,
    bne::BNE,
    div::DIV,
    divu::DIVU,
    divuw::DIVUW,
    divw::DIVW,
    ebreak::EBREAK,
    ecall::ECALL,
    fence::FENCE,
    jal::JAL,
    jalr::JALR,
    lb::LB,
    lbu::LBU,
    ld::LD,
    lh::LH,
    lhu::LHU,
    lui::LUI,
    lw::LW,
    lwu::LWU,
    mul::MUL,
    mulh::MULH,
    mulhsu::MULHSU,
    mulhu::MULHU,
    mulw::MULW,
    or::OR,
    ori::ORI,
    rem::REM,
    remu::REMU,
    remuw::REMUW,
    remw::REMW,
    sb::SB,
    sd::SD,
    sh::SH,
    sll::SLL,
    slli::SLLI,
    slliw::SLLIW,
    sllw::SLLW,
    slt::SLT,
    slti::SLTI,
    sltiu::SLTIU,
    sltu::SLTU,
    sra::SRA,
    srai::SRAI,
    sraiw::SRAIW,
    sraw::SRAW,
    srl::SRL,
    srli::SRLI,
    srliw::SRLIW,
    srlw::SRLW,
    sub::SUB,
    subw::SUBW,
    sw::SW,
    virtual_advice::VirtualAdvice as TracerVirtualAdvice,
    virtual_advice_len::VirtualAdviceLen as TracerVirtualAdviceLen,
    virtual_advice_load::VirtualAdviceLoad as TracerVirtualAdviceLoad,
    virtual_assert_eq::VirtualAssertEQ,
    virtual_assert_halfword_alignment::VirtualAssertHalfwordAlignment,
    virtual_assert_lte::VirtualAssertLTE,
    virtual_assert_mulu_no_overflow::VirtualAssertMulUNoOverflow,
    virtual_assert_valid_div0::VirtualAssertValidDiv0,
    virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder,
    virtual_assert_word_alignment::VirtualAssertWordAlignment,
    virtual_change_divisor::VirtualChangeDivisor as TracerVirtualChangeDivisor,
    virtual_change_divisor_w::VirtualChangeDivisorW as TracerVirtualChangeDivisorW,
    virtual_host_io::VirtualHostIO as TracerVirtualHostIO,
    virtual_movsign::VirtualMovsign,
    virtual_muli::VirtualMULI,
    virtual_pow2::VirtualPow2 as TracerVirtualPow2,
    virtual_pow2_w::VirtualPow2W as TracerVirtualPow2W,
    virtual_pow2i::VirtualPow2I as TracerVirtualPow2I,
    virtual_pow2i_w::VirtualPow2IW as TracerVirtualPow2IW,
    virtual_rev8w::VirtualRev8W as TracerVirtualRev8W,
    virtual_rotri::VirtualROTRI,
    virtual_rotriw::VirtualROTRIW,
    virtual_shift_right_bitmask::VirtualShiftRightBitmask as TracerVirtualShiftRightBitmask,
    virtual_shift_right_bitmaski::VirtualShiftRightBitmaskI,
    virtual_sign_extend_word::VirtualSignExtendWord as TracerVirtualSignExtendWord,
    virtual_sra::VirtualSRA,
    virtual_srai::VirtualSRAI,
    virtual_srl::VirtualSRL,
    virtual_srli::VirtualSRLI,
    virtual_xor_rot::{VirtualXORROT16, VirtualXORROT24, VirtualXORROT32, VirtualXORROT63},
    virtual_xor_rotw::{VirtualXORROTW12, VirtualXORROTW16, VirtualXORROTW7, VirtualXORROTW8},
    virtual_zero_extend_word::VirtualZeroExtendWord as TracerVirtualZeroExtendWord,
    xor::XOR,
    xori::XORI,
};
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
/// Each variant wraps a Jolt newtype parameterized by the corresponding tracer
/// instruction struct, so a `JoltInstructions` value carries a populated
/// instruction (not just a marker). Static-flag dispatch and the
/// flag-exclusivity tests rely on this concretization to satisfy
/// `T: JoltInstruction` on the `Flags` impls.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, strum::EnumIter)]
pub enum JoltInstructions {
    Noop,
    Add(Add<ADD>),
    Addi(Addi<ADDI>),
    Sub(Sub<SUB>),
    Lui(Lui<LUI>),
    Auipc(Auipc<AUIPC>),
    Mul(Mul<MUL>),
    MulH(MulH<MULH>),
    MulHSU(MulHSU<MULHSU>),
    MulHU(MulHU<MULHU>),
    Div(Div<DIV>),
    DivU(DivU<DIVU>),
    Rem(Rem<REM>),
    RemU(RemU<REMU>),
    AddW(AddW<ADDW>),
    AddiW(AddiW<ADDIW>),
    SubW(SubW<SUBW>),
    MulW(MulW<MULW>),
    DivW(DivW<DIVW>),
    DivUW(DivUW<DIVUW>),
    RemW(RemW<REMW>),
    RemUW(RemUW<REMUW>),
    And(And<AND>),
    AndI(AndI<ANDI>),
    Or(Or<OR>),
    OrI(OrI<ORI>),
    Xor(Xor<XOR>),
    XorI(XorI<XORI>),
    Andn(Andn<ANDN>),
    Sll(Sll<SLL>),
    SllI(SllI<SLLI>),
    Srl(Srl<SRL>),
    SrlI(SrlI<SRLI>),
    Sra(Sra<SRA>),
    SraI(SraI<SRAI>),
    SllW(SllW<SLLW>),
    SllIW(SllIW<SLLIW>),
    SrlW(SrlW<SRLW>),
    SrlIW(SrlIW<SRLIW>),
    SraW(SraW<SRAW>),
    SraIW(SraIW<SRAIW>),
    Slt(Slt<SLT>),
    SltI(SltI<SLTI>),
    SltU(SltU<SLTU>),
    SltIU(SltIU<SLTIU>),
    Beq(Beq<BEQ>),
    Bne(Bne<BNE>),
    Blt(Blt<BLT>),
    Bge(Bge<BGE>),
    BltU(BltU<BLTU>),
    BgeU(BgeU<BGEU>),
    Lb(Lb<LB>),
    Lbu(Lbu<LBU>),
    Lh(Lh<LH>),
    Lhu(Lhu<LHU>),
    Lw(Lw<LW>),
    Lwu(Lwu<LWU>),
    Ld(Ld<LD>),
    Sb(Sb<SB>),
    Sh(Sh<SH>),
    Sw(Sw<SW>),
    Sd(Sd<SD>),
    Ecall(Ecall<ECALL>),
    Ebreak(Ebreak<EBREAK>),
    Fence(Fence<FENCE>),
    Jal(Jal<JAL>),
    Jalr(Jalr<JALR>),
    AssertEq(AssertEq<VirtualAssertEQ>),
    AssertLte(AssertLte<VirtualAssertLTE>),
    AssertValidDiv0(AssertValidDiv0<VirtualAssertValidDiv0>),
    AssertValidUnsignedRemainder(AssertValidUnsignedRemainder<VirtualAssertValidUnsignedRemainder>),
    AssertMulUNoOverflow(AssertMulUNoOverflow<VirtualAssertMulUNoOverflow>),
    AssertWordAlignment(AssertWordAlignment<VirtualAssertWordAlignment>),
    AssertHalfwordAlignment(AssertHalfwordAlignment<VirtualAssertHalfwordAlignment>),
    Pow2(Pow2<TracerVirtualPow2>),
    Pow2I(Pow2I<TracerVirtualPow2I>),
    Pow2W(Pow2W<TracerVirtualPow2W>),
    Pow2IW(Pow2IW<TracerVirtualPow2IW>),
    MulI(MulI<VirtualMULI>),
    MovSign(MovSign<VirtualMovsign>),
    VirtualRev8W(VirtualRev8W<TracerVirtualRev8W>),
    VirtualChangeDivisor(VirtualChangeDivisor<TracerVirtualChangeDivisor>),
    VirtualChangeDivisorW(VirtualChangeDivisorW<TracerVirtualChangeDivisorW>),
    VirtualSignExtendWord(VirtualSignExtendWord<TracerVirtualSignExtendWord>),
    VirtualZeroExtendWord(VirtualZeroExtendWord<TracerVirtualZeroExtendWord>),
    VirtualSrl(VirtualSrl<VirtualSRL>),
    VirtualSrli(VirtualSrli<VirtualSRLI>),
    VirtualSra(VirtualSra<VirtualSRA>),
    VirtualSrai(VirtualSrai<VirtualSRAI>),
    VirtualShiftRightBitmask(VirtualShiftRightBitmask<TracerVirtualShiftRightBitmask>),
    VirtualShiftRightBitmaski(VirtualShiftRightBitmaski<VirtualShiftRightBitmaskI>),
    VirtualRotri(VirtualRotri<VirtualROTRI>),
    VirtualRotriw(VirtualRotriw<VirtualROTRIW>),
    VirtualXorRot32(VirtualXorRot32<VirtualXORROT32>),
    VirtualXorRot24(VirtualXorRot24<VirtualXORROT24>),
    VirtualXorRot16(VirtualXorRot16<VirtualXORROT16>),
    VirtualXorRot63(VirtualXorRot63<VirtualXORROT63>),
    VirtualXorRotW16(VirtualXorRotW16<VirtualXORROTW16>),
    VirtualXorRotW12(VirtualXorRotW12<VirtualXORROTW12>),
    VirtualXorRotW8(VirtualXorRotW8<VirtualXORROTW8>),
    VirtualXorRotW7(VirtualXorRotW7<VirtualXORROTW7>),
    VirtualAdvice(VirtualAdvice<TracerVirtualAdvice>),
    VirtualAdviceLen(VirtualAdviceLen<TracerVirtualAdviceLen>),
    VirtualAdviceLoad(VirtualAdviceLoad<TracerVirtualAdviceLoad>),
    VirtualHostIO(VirtualHostIO<TracerVirtualHostIO>),
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
