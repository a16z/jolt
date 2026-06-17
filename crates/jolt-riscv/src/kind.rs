#[cfg(feature = "serialization")]
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, Read, SerializationError, Valid, Validate,
    Write,
};
#[cfg(feature = "serialization")]
use serde::{de::Visitor, Deserializer, Serialize, Serializer};

use crate::instructions::*;
use crate::profile::{JoltTargetExtension, SourceExtension};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JoltInstructionTag(pub u16);

/// Source instruction identity, represented as the zero-payload source enum.
///
/// `SourceInstruction<T>` is the only source-instruction enum. Instantiating it
/// with `T = ()` gives the kind-level view; instantiating it with
/// `T = SourceInstructionRow` gives a concrete decoded row.
pub type SourceInstructionKind = SourceInstruction<()>;

/// Final Jolt instruction identity, represented as the zero-payload final enum.
///
/// `JoltInstruction<T>` is the only final-instruction enum. Instantiating it
/// with `T = ()` gives the kind-level view; instantiating it with
/// `T = JoltInstructionRow` gives a concrete final row.
pub type JoltInstructionKind = JoltInstruction<()>;

pub trait SourceInstructionMeta {
    const CANONICAL_NAME: &'static str;
    const SOURCE_EXTENSION: Option<SourceExtension>;
    const HAS_SIDE_EFFECTS: bool;
}

pub trait JoltInstructionMeta {
    const CANONICAL_NAME: &'static str;
    const JOLT_TAG: JoltInstructionTag;
    const TARGET_EXTENSION: Option<JoltTargetExtension>;
    const HAS_SIDE_EFFECTS: bool;
}

macro_rules! source_extension_for_marker {
    (Add) => {
        Some(SourceExtension::Rv64I)
    };
    (Addi) => {
        Some(SourceExtension::Rv64I)
    };
    (AddiW) => {
        Some(SourceExtension::Rv64I)
    };
    (AddW) => {
        Some(SourceExtension::Rv64I)
    };
    (And) => {
        Some(SourceExtension::Rv64I)
    };
    (AndI) => {
        Some(SourceExtension::Rv64I)
    };
    (Auipc) => {
        Some(SourceExtension::Rv64I)
    };
    (Beq) => {
        Some(SourceExtension::Rv64I)
    };
    (Bge) => {
        Some(SourceExtension::Rv64I)
    };
    (BgeU) => {
        Some(SourceExtension::Rv64I)
    };
    (Blt) => {
        Some(SourceExtension::Rv64I)
    };
    (BltU) => {
        Some(SourceExtension::Rv64I)
    };
    (Bne) => {
        Some(SourceExtension::Rv64I)
    };
    (Ebreak) => {
        Some(SourceExtension::Rv64I)
    };
    (Ecall) => {
        Some(SourceExtension::Rv64I)
    };
    (Fence) => {
        Some(SourceExtension::Rv64I)
    };
    (Jal) => {
        Some(SourceExtension::Rv64I)
    };
    (Jalr) => {
        Some(SourceExtension::Rv64I)
    };
    (Lb) => {
        Some(SourceExtension::Rv64I)
    };
    (Lbu) => {
        Some(SourceExtension::Rv64I)
    };
    (Ld) => {
        Some(SourceExtension::Rv64I)
    };
    (Lh) => {
        Some(SourceExtension::Rv64I)
    };
    (Lhu) => {
        Some(SourceExtension::Rv64I)
    };
    (Lui) => {
        Some(SourceExtension::Rv64I)
    };
    (Lw) => {
        Some(SourceExtension::Rv64I)
    };
    (Lwu) => {
        Some(SourceExtension::Rv64I)
    };
    (Or) => {
        Some(SourceExtension::Rv64I)
    };
    (OrI) => {
        Some(SourceExtension::Rv64I)
    };
    (Sb) => {
        Some(SourceExtension::Rv64I)
    };
    (Sd) => {
        Some(SourceExtension::Rv64I)
    };
    (Sh) => {
        Some(SourceExtension::Rv64I)
    };
    (Sll) => {
        Some(SourceExtension::Rv64I)
    };
    (SllI) => {
        Some(SourceExtension::Rv64I)
    };
    (SllIW) => {
        Some(SourceExtension::Rv64I)
    };
    (SllW) => {
        Some(SourceExtension::Rv64I)
    };
    (Slt) => {
        Some(SourceExtension::Rv64I)
    };
    (SltI) => {
        Some(SourceExtension::Rv64I)
    };
    (SltIU) => {
        Some(SourceExtension::Rv64I)
    };
    (SltU) => {
        Some(SourceExtension::Rv64I)
    };
    (Sra) => {
        Some(SourceExtension::Rv64I)
    };
    (SraI) => {
        Some(SourceExtension::Rv64I)
    };
    (SraIW) => {
        Some(SourceExtension::Rv64I)
    };
    (SraW) => {
        Some(SourceExtension::Rv64I)
    };
    (Srl) => {
        Some(SourceExtension::Rv64I)
    };
    (SrlI) => {
        Some(SourceExtension::Rv64I)
    };
    (SrlIW) => {
        Some(SourceExtension::Rv64I)
    };
    (SrlW) => {
        Some(SourceExtension::Rv64I)
    };
    (Sub) => {
        Some(SourceExtension::Rv64I)
    };
    (SubW) => {
        Some(SourceExtension::Rv64I)
    };
    (Sw) => {
        Some(SourceExtension::Rv64I)
    };
    (Xor) => {
        Some(SourceExtension::Rv64I)
    };
    (XorI) => {
        Some(SourceExtension::Rv64I)
    };
    (Mul) => {
        Some(SourceExtension::Rv64M)
    };
    (MulH) => {
        Some(SourceExtension::Rv64M)
    };
    (MulHSU) => {
        Some(SourceExtension::Rv64M)
    };
    (MulHU) => {
        Some(SourceExtension::Rv64M)
    };
    (MulW) => {
        Some(SourceExtension::Rv64M)
    };
    (Div) => {
        Some(SourceExtension::Rv64M)
    };
    (DivU) => {
        Some(SourceExtension::Rv64M)
    };
    (DivUW) => {
        Some(SourceExtension::Rv64M)
    };
    (DivW) => {
        Some(SourceExtension::Rv64M)
    };
    (Rem) => {
        Some(SourceExtension::Rv64M)
    };
    (RemU) => {
        Some(SourceExtension::Rv64M)
    };
    (RemUW) => {
        Some(SourceExtension::Rv64M)
    };
    (RemW) => {
        Some(SourceExtension::Rv64M)
    };
    (LrW) => {
        Some(SourceExtension::Rv64A)
    };
    (ScW) => {
        Some(SourceExtension::Rv64A)
    };
    (AmoSwapW) => {
        Some(SourceExtension::Rv64A)
    };
    (AmoAddW) => {
        Some(SourceExtension::Rv64A)
    };
    (AmoAndW) => {
        Some(SourceExtension::Rv64A)
    };
    (AmoOrW) => {
        Some(SourceExtension::Rv64A)
    };
    (AmoXorW) => {
        Some(SourceExtension::Rv64A)
    };
    (AmoMinW) => {
        Some(SourceExtension::Rv64A)
    };
    (AmoMaxW) => {
        Some(SourceExtension::Rv64A)
    };
    (AmoMinUW) => {
        Some(SourceExtension::Rv64A)
    };
    (AmoMaxUW) => {
        Some(SourceExtension::Rv64A)
    };
    (LrD) => {
        Some(SourceExtension::Rv64A)
    };
    (ScD) => {
        Some(SourceExtension::Rv64A)
    };
    (AmoSwapD) => {
        Some(SourceExtension::Rv64A)
    };
    (AmoAddD) => {
        Some(SourceExtension::Rv64A)
    };
    (AmoAndD) => {
        Some(SourceExtension::Rv64A)
    };
    (AmoOrD) => {
        Some(SourceExtension::Rv64A)
    };
    (AmoXorD) => {
        Some(SourceExtension::Rv64A)
    };
    (AmoMinD) => {
        Some(SourceExtension::Rv64A)
    };
    (AmoMaxD) => {
        Some(SourceExtension::Rv64A)
    };
    (AmoMinUD) => {
        Some(SourceExtension::Rv64A)
    };
    (AmoMaxUD) => {
        Some(SourceExtension::Rv64A)
    };
    (Csrrs) => {
        Some(SourceExtension::Zicsr)
    };
    (Csrrw) => {
        Some(SourceExtension::Zicsr)
    };
    (Mret) => {
        Some(SourceExtension::RvPrivileged)
    };
    (Andn) => {
        Some(SourceExtension::JoltCustom)
    };
    (AdviceLb) => {
        Some(SourceExtension::JoltCustom)
    };
    (AdviceLd) => {
        Some(SourceExtension::JoltCustom)
    };
    (AdviceLh) => {
        Some(SourceExtension::JoltCustom)
    };
    (AdviceLw) => {
        Some(SourceExtension::JoltCustom)
    };
    (AssertEq) => {
        Some(SourceExtension::JoltCustom)
    };
    (AssertHalfwordAlignment) => {
        Some(SourceExtension::JoltCustom)
    };
    (AssertWordAlignment) => {
        Some(SourceExtension::JoltCustom)
    };
    (AssertLte) => {
        Some(SourceExtension::JoltCustom)
    };
    (AssertValidDiv0) => {
        Some(SourceExtension::JoltCustom)
    };
    (AssertValidUnsignedRemainder) => {
        Some(SourceExtension::JoltCustom)
    };
    (AssertMulUNoOverflow) => {
        Some(SourceExtension::JoltCustom)
    };
    (VirtualAdvice) => {
        Some(SourceExtension::JoltCustom)
    };
    (VirtualAdviceLen) => {
        Some(SourceExtension::JoltCustom)
    };
    (VirtualAdviceLoad) => {
        Some(SourceExtension::JoltCustom)
    };
    (VirtualHostIO) => {
        Some(SourceExtension::JoltCustom)
    };
    (VirtualChangeDivisor) => {
        Some(SourceExtension::JoltCustom)
    };
    (VirtualChangeDivisorW) => {
        Some(SourceExtension::JoltCustom)
    };
    (VirtualLw) => {
        Some(SourceExtension::JoltCustom)
    };
    (VirtualSw) => {
        Some(SourceExtension::JoltCustom)
    };
    (VirtualZeroExtendWord) => {
        Some(SourceExtension::JoltCustom)
    };
    (VirtualSignExtendWord) => {
        Some(SourceExtension::JoltCustom)
    };
    (Pow2W) => {
        Some(SourceExtension::JoltCustom)
    };
    (Pow2IW) => {
        Some(SourceExtension::JoltCustom)
    };
    (MovSign) => {
        Some(SourceExtension::JoltCustom)
    };
    (MulI) => {
        Some(SourceExtension::JoltCustom)
    };
    (Pow2) => {
        Some(SourceExtension::JoltCustom)
    };
    (Pow2I) => {
        Some(SourceExtension::JoltCustom)
    };
    (VirtualRev8W) => {
        Some(SourceExtension::JoltCustom)
    };
    (VirtualRotri) => {
        Some(SourceExtension::JoltCustom)
    };
    (VirtualRotriw) => {
        Some(SourceExtension::JoltCustom)
    };
    (VirtualShiftRightBitmask) => {
        Some(SourceExtension::JoltCustom)
    };
    (VirtualShiftRightBitmaski) => {
        Some(SourceExtension::JoltCustom)
    };
    (VirtualSra) => {
        Some(SourceExtension::JoltCustom)
    };
    (VirtualSrai) => {
        Some(SourceExtension::JoltCustom)
    };
    (VirtualSrl) => {
        Some(SourceExtension::JoltCustom)
    };
    (VirtualSrli) => {
        Some(SourceExtension::JoltCustom)
    };
    (VirtualXorRot32) => {
        Some(SourceExtension::JoltCustom)
    };
    (VirtualXorRot24) => {
        Some(SourceExtension::JoltCustom)
    };
    (VirtualXorRot16) => {
        Some(SourceExtension::JoltCustom)
    };
    (VirtualXorRot63) => {
        Some(SourceExtension::JoltCustom)
    };
    (VirtualXorRotW16) => {
        Some(SourceExtension::JoltCustom)
    };
    (VirtualXorRotW12) => {
        Some(SourceExtension::JoltCustom)
    };
    (VirtualXorRotW8) => {
        Some(SourceExtension::JoltCustom)
    };
    (VirtualXorRotW7) => {
        Some(SourceExtension::JoltCustom)
    };
    (FieldAdd) => {
        Some(SourceExtension::FieldInline)
    };
    (FieldSub) => {
        Some(SourceExtension::FieldInline)
    };
    (FieldMul) => {
        Some(SourceExtension::FieldInline)
    };
    (FieldInv) => {
        Some(SourceExtension::FieldInline)
    };
    (FieldAssertEq) => {
        Some(SourceExtension::FieldInline)
    };
    (FieldLoadFromX) => {
        Some(SourceExtension::FieldInline)
    };
    (FieldStoreToX) => {
        Some(SourceExtension::FieldInline)
    };
    (FieldLoadImm) => {
        Some(SourceExtension::FieldInline)
    };
}

macro_rules! source_side_effects_for_marker {
    (AdviceLb) => {
        true
    };
    (AdviceLd) => {
        true
    };
    (AdviceLh) => {
        true
    };
    (AdviceLw) => {
        true
    };
    (AmoAddD) => {
        true
    };
    (AmoAddW) => {
        true
    };
    (AmoAndD) => {
        true
    };
    (AmoAndW) => {
        true
    };
    (AmoMaxD) => {
        true
    };
    (AmoMaxUD) => {
        true
    };
    (AmoMaxUW) => {
        true
    };
    (AmoMaxW) => {
        true
    };
    (AmoMinD) => {
        true
    };
    (AmoMinUD) => {
        true
    };
    (AmoMinUW) => {
        true
    };
    (AmoMinW) => {
        true
    };
    (AmoOrD) => {
        true
    };
    (AmoOrW) => {
        true
    };
    (AmoSwapD) => {
        true
    };
    (AmoSwapW) => {
        true
    };
    (AmoXorD) => {
        true
    };
    (AmoXorW) => {
        true
    };
    (Beq) => {
        true
    };
    (Bge) => {
        true
    };
    (BgeU) => {
        true
    };
    (Blt) => {
        true
    };
    (BltU) => {
        true
    };
    (Bne) => {
        true
    };
    (Csrrs) => {
        true
    };
    (Csrrw) => {
        true
    };
    (Ebreak) => {
        true
    };
    (Ecall) => {
        true
    };
    (Inline) => {
        true
    };
    (Jal) => {
        true
    };
    (Jalr) => {
        true
    };
    (Lb) => {
        true
    };
    (Lbu) => {
        true
    };
    (Ld) => {
        true
    };
    (Lh) => {
        true
    };
    (Lhu) => {
        true
    };
    (LrD) => {
        true
    };
    (LrW) => {
        true
    };
    (Lw) => {
        true
    };
    (Lwu) => {
        true
    };
    (Mret) => {
        true
    };
    (Sb) => {
        true
    };
    (ScD) => {
        true
    };
    (ScW) => {
        true
    };
    (Sd) => {
        true
    };
    (Sh) => {
        true
    };
    (Sw) => {
        true
    };
    (VirtualAdviceLoad) => {
        true
    };
    (VirtualHostIO) => {
        true
    };
    (VirtualSw) => {
        true
    };
    (Add) => {
        false
    };
    (Addi) => {
        false
    };
    (AddiW) => {
        false
    };
    (AddW) => {
        false
    };
    (And) => {
        false
    };
    (AndI) => {
        false
    };
    (Andn) => {
        false
    };
    (Auipc) => {
        false
    };
    (Div) => {
        false
    };
    (DivU) => {
        false
    };
    (DivUW) => {
        false
    };
    (DivW) => {
        false
    };
    (Fence) => {
        false
    };
    (Lui) => {
        false
    };
    (Mul) => {
        false
    };
    (MulH) => {
        false
    };
    (MulHSU) => {
        false
    };
    (MulHU) => {
        false
    };
    (MulW) => {
        false
    };
    (Or) => {
        false
    };
    (OrI) => {
        false
    };
    (Rem) => {
        false
    };
    (RemU) => {
        false
    };
    (RemUW) => {
        false
    };
    (RemW) => {
        false
    };
    (Sll) => {
        false
    };
    (SllI) => {
        false
    };
    (SllIW) => {
        false
    };
    (SllW) => {
        false
    };
    (Slt) => {
        false
    };
    (SltI) => {
        false
    };
    (SltIU) => {
        false
    };
    (SltU) => {
        false
    };
    (Sra) => {
        false
    };
    (SraI) => {
        false
    };
    (SraIW) => {
        false
    };
    (SraW) => {
        false
    };
    (Srl) => {
        false
    };
    (SrlI) => {
        false
    };
    (SrlIW) => {
        false
    };
    (SrlW) => {
        false
    };
    (Sub) => {
        false
    };
    (SubW) => {
        false
    };
    (Xor) => {
        false
    };
    (XorI) => {
        false
    };
    (AssertEq) => {
        false
    };
    (AssertHalfwordAlignment) => {
        false
    };
    (AssertWordAlignment) => {
        false
    };
    (AssertLte) => {
        false
    };
    (AssertValidDiv0) => {
        false
    };
    (AssertValidUnsignedRemainder) => {
        false
    };
    (AssertMulUNoOverflow) => {
        false
    };
    (VirtualAdvice) => {
        false
    };
    (VirtualAdviceLen) => {
        false
    };
    (VirtualChangeDivisor) => {
        false
    };
    (VirtualChangeDivisorW) => {
        false
    };
    (VirtualLw) => {
        false
    };
    (VirtualZeroExtendWord) => {
        false
    };
    (VirtualSignExtendWord) => {
        false
    };
    (Pow2W) => {
        false
    };
    (Pow2IW) => {
        false
    };
    (MovSign) => {
        false
    };
    (MulI) => {
        false
    };
    (Pow2) => {
        false
    };
    (Pow2I) => {
        false
    };
    (VirtualRev8W) => {
        false
    };
    (VirtualRotri) => {
        false
    };
    (VirtualRotriw) => {
        false
    };
    (VirtualShiftRightBitmask) => {
        false
    };
    (VirtualShiftRightBitmaski) => {
        false
    };
    (VirtualSra) => {
        false
    };
    (VirtualSrai) => {
        false
    };
    (VirtualSrl) => {
        false
    };
    (VirtualSrli) => {
        false
    };
    (VirtualXorRot32) => {
        false
    };
    (VirtualXorRot24) => {
        false
    };
    (VirtualXorRot16) => {
        false
    };
    (VirtualXorRot63) => {
        false
    };
    (VirtualXorRotW16) => {
        false
    };
    (VirtualXorRotW12) => {
        false
    };
    (VirtualXorRotW8) => {
        false
    };
    (VirtualXorRotW7) => {
        false
    };
    (FieldAdd) => {
        true
    };
    (FieldSub) => {
        true
    };
    (FieldMul) => {
        true
    };
    (FieldInv) => {
        true
    };
    (FieldAssertEq) => {
        true
    };
    (FieldLoadFromX) => {
        true
    };
    (FieldStoreToX) => {
        true
    };
    (FieldLoadImm) => {
        true
    };
}

macro_rules! jolt_target_extension_for_marker {
    (Add) => {
        Some(JoltTargetExtension::IntegerCore)
    };
    (Addi) => {
        Some(JoltTargetExtension::IntegerCore)
    };
    (And) => {
        Some(JoltTargetExtension::IntegerCore)
    };
    (AndI) => {
        Some(JoltTargetExtension::IntegerCore)
    };
    (Auipc) => {
        Some(JoltTargetExtension::IntegerCore)
    };
    (Lui) => {
        Some(JoltTargetExtension::IntegerCore)
    };
    (Or) => {
        Some(JoltTargetExtension::IntegerCore)
    };
    (OrI) => {
        Some(JoltTargetExtension::IntegerCore)
    };
    (Slt) => {
        Some(JoltTargetExtension::IntegerCore)
    };
    (SltI) => {
        Some(JoltTargetExtension::IntegerCore)
    };
    (SltIU) => {
        Some(JoltTargetExtension::IntegerCore)
    };
    (SltU) => {
        Some(JoltTargetExtension::IntegerCore)
    };
    (Sub) => {
        Some(JoltTargetExtension::IntegerCore)
    };
    (Xor) => {
        Some(JoltTargetExtension::IntegerCore)
    };
    (XorI) => {
        Some(JoltTargetExtension::IntegerCore)
    };
    (Mul) => {
        Some(JoltTargetExtension::IntegerMultiply)
    };
    (MulHU) => {
        Some(JoltTargetExtension::IntegerMultiply)
    };
    (Beq) => {
        Some(JoltTargetExtension::ControlFlow)
    };
    (Bge) => {
        Some(JoltTargetExtension::ControlFlow)
    };
    (BgeU) => {
        Some(JoltTargetExtension::ControlFlow)
    };
    (Blt) => {
        Some(JoltTargetExtension::ControlFlow)
    };
    (BltU) => {
        Some(JoltTargetExtension::ControlFlow)
    };
    (Bne) => {
        Some(JoltTargetExtension::ControlFlow)
    };
    (Fence) => {
        Some(JoltTargetExtension::ControlFlow)
    };
    (Jal) => {
        Some(JoltTargetExtension::ControlFlow)
    };
    (Jalr) => {
        Some(JoltTargetExtension::ControlFlow)
    };
    (Ld) => {
        Some(JoltTargetExtension::LoadStore64)
    };
    (Sd) => {
        Some(JoltTargetExtension::LoadStore64)
    };
    (VirtualAdvice) => {
        Some(JoltTargetExtension::Advice)
    };
    (VirtualAdviceLen) => {
        Some(JoltTargetExtension::Advice)
    };
    (VirtualAdviceLoad) => {
        Some(JoltTargetExtension::Advice)
    };
    (VirtualHostIO) => {
        Some(JoltTargetExtension::HostIO)
    };
    (AssertEq) => {
        Some(JoltTargetExtension::VirtualAssertions)
    };
    (AssertHalfwordAlignment) => {
        Some(JoltTargetExtension::VirtualAssertions)
    };
    (AssertWordAlignment) => {
        Some(JoltTargetExtension::VirtualAssertions)
    };
    (AssertLte) => {
        Some(JoltTargetExtension::VirtualAssertions)
    };
    (AssertValidDiv0) => {
        Some(JoltTargetExtension::VirtualAssertions)
    };
    (AssertValidUnsignedRemainder) => {
        Some(JoltTargetExtension::VirtualAssertions)
    };
    (AssertMulUNoOverflow) => {
        Some(JoltTargetExtension::VirtualAssertions)
    };
    (VirtualChangeDivisor) => {
        Some(JoltTargetExtension::VirtualArithmetic)
    };
    (VirtualChangeDivisorW) => {
        Some(JoltTargetExtension::VirtualArithmetic)
    };
    (VirtualZeroExtendWord) => {
        Some(JoltTargetExtension::VirtualArithmetic)
    };
    (VirtualSignExtendWord) => {
        Some(JoltTargetExtension::VirtualArithmetic)
    };
    (Pow2W) => {
        Some(JoltTargetExtension::VirtualArithmetic)
    };
    (Pow2IW) => {
        Some(JoltTargetExtension::VirtualArithmetic)
    };
    (MovSign) => {
        Some(JoltTargetExtension::VirtualArithmetic)
    };
    (MulI) => {
        Some(JoltTargetExtension::VirtualArithmetic)
    };
    (Pow2) => {
        Some(JoltTargetExtension::VirtualArithmetic)
    };
    (Pow2I) => {
        Some(JoltTargetExtension::VirtualArithmetic)
    };
    (VirtualRotri) => {
        Some(JoltTargetExtension::VirtualShifts)
    };
    (VirtualRotriw) => {
        Some(JoltTargetExtension::VirtualShifts)
    };
    (VirtualShiftRightBitmask) => {
        Some(JoltTargetExtension::VirtualShifts)
    };
    (VirtualShiftRightBitmaski) => {
        Some(JoltTargetExtension::VirtualShifts)
    };
    (VirtualSra) => {
        Some(JoltTargetExtension::VirtualShifts)
    };
    (VirtualSrai) => {
        Some(JoltTargetExtension::VirtualShifts)
    };
    (VirtualSrl) => {
        Some(JoltTargetExtension::VirtualShifts)
    };
    (VirtualSrli) => {
        Some(JoltTargetExtension::VirtualShifts)
    };
    (Andn) => {
        Some(JoltTargetExtension::BitManipulation)
    };
    (VirtualRev8W) => {
        Some(JoltTargetExtension::BitManipulation)
    };
    (VirtualXorRot32) => {
        Some(JoltTargetExtension::BitManipulation)
    };
    (VirtualXorRot24) => {
        Some(JoltTargetExtension::BitManipulation)
    };
    (VirtualXorRot16) => {
        Some(JoltTargetExtension::BitManipulation)
    };
    (VirtualXorRot63) => {
        Some(JoltTargetExtension::BitManipulation)
    };
    (VirtualXorRotW16) => {
        Some(JoltTargetExtension::BitManipulation)
    };
    (VirtualXorRotW12) => {
        Some(JoltTargetExtension::BitManipulation)
    };
    (VirtualXorRotW8) => {
        Some(JoltTargetExtension::BitManipulation)
    };
    (VirtualXorRotW7) => {
        Some(JoltTargetExtension::BitManipulation)
    };
    (FieldAdd) => {
        Some(JoltTargetExtension::FieldInline)
    };
    (FieldSub) => {
        Some(JoltTargetExtension::FieldInline)
    };
    (FieldMul) => {
        Some(JoltTargetExtension::FieldInline)
    };
    (FieldInv) => {
        Some(JoltTargetExtension::FieldInline)
    };
    (FieldAssertEq) => {
        Some(JoltTargetExtension::FieldInline)
    };
    (FieldLoadFromX) => {
        Some(JoltTargetExtension::FieldInline)
    };
    (FieldStoreToX) => {
        Some(JoltTargetExtension::FieldInline)
    };
    (FieldLoadImm) => {
        Some(JoltTargetExtension::FieldInline)
    };
}

macro_rules! jolt_side_effects_for_marker {
    (Beq) => {
        true
    };
    (Bge) => {
        true
    };
    (BgeU) => {
        true
    };
    (Blt) => {
        true
    };
    (BltU) => {
        true
    };
    (Bne) => {
        true
    };
    (Fence) => {
        true
    };
    (Jal) => {
        true
    };
    (Jalr) => {
        true
    };
    (Ld) => {
        true
    };
    (Sd) => {
        true
    };
    (VirtualAdviceLoad) => {
        true
    };
    (VirtualHostIO) => {
        true
    };
    (FieldAdd) => {
        true
    };
    (FieldSub) => {
        true
    };
    (FieldMul) => {
        true
    };
    (FieldInv) => {
        true
    };
    (FieldAssertEq) => {
        true
    };
    (FieldLoadFromX) => {
        true
    };
    (FieldStoreToX) => {
        true
    };
    (FieldLoadImm) => {
        true
    };
    (Add) => {
        false
    };
    (Addi) => {
        false
    };
    (And) => {
        false
    };
    (AndI) => {
        false
    };
    (Andn) => {
        false
    };
    (Auipc) => {
        false
    };
    (Lui) => {
        false
    };
    (Mul) => {
        false
    };
    (MulHU) => {
        false
    };
    (Or) => {
        false
    };
    (OrI) => {
        false
    };
    (Slt) => {
        false
    };
    (SltI) => {
        false
    };
    (SltIU) => {
        false
    };
    (SltU) => {
        false
    };
    (Sub) => {
        false
    };
    (Xor) => {
        false
    };
    (XorI) => {
        false
    };
    (AssertEq) => {
        false
    };
    (AssertHalfwordAlignment) => {
        false
    };
    (AssertWordAlignment) => {
        false
    };
    (AssertLte) => {
        false
    };
    (AssertValidDiv0) => {
        false
    };
    (AssertValidUnsignedRemainder) => {
        false
    };
    (AssertMulUNoOverflow) => {
        false
    };
    (VirtualAdvice) => {
        false
    };
    (VirtualAdviceLen) => {
        false
    };
    (VirtualChangeDivisor) => {
        false
    };
    (VirtualChangeDivisorW) => {
        false
    };
    (VirtualZeroExtendWord) => {
        false
    };
    (VirtualSignExtendWord) => {
        false
    };
    (Pow2W) => {
        false
    };
    (Pow2IW) => {
        false
    };
    (MovSign) => {
        false
    };
    (MulI) => {
        false
    };
    (Pow2) => {
        false
    };
    (Pow2I) => {
        false
    };
    (VirtualRev8W) => {
        false
    };
    (VirtualRotri) => {
        false
    };
    (VirtualRotriw) => {
        false
    };
    (VirtualShiftRightBitmask) => {
        false
    };
    (VirtualShiftRightBitmaski) => {
        false
    };
    (VirtualSra) => {
        false
    };
    (VirtualSrai) => {
        false
    };
    (VirtualSrl) => {
        false
    };
    (VirtualSrli) => {
        false
    };
    (VirtualXorRot32) => {
        false
    };
    (VirtualXorRot24) => {
        false
    };
    (VirtualXorRot16) => {
        false
    };
    (VirtualXorRot63) => {
        false
    };
    (VirtualXorRotW16) => {
        false
    };
    (VirtualXorRotW12) => {
        false
    };
    (VirtualXorRotW8) => {
        false
    };
    (VirtualXorRotW7) => {
        false
    };
}

impl<T> SourceInstructionMeta for Noop<T> {
    const CANONICAL_NAME: &'static str = "jolt.pseudo.noop";
    const SOURCE_EXTENSION: Option<SourceExtension> = None;
    const HAS_SIDE_EFFECTS: bool = false;
}

impl<T> SourceInstructionMeta for Unimpl<T> {
    const CANONICAL_NAME: &'static str = "jolt.pseudo.unimpl";
    const SOURCE_EXTENSION: Option<SourceExtension> = None;
    const HAS_SIDE_EFFECTS: bool = false;
}

impl<T> SourceInstructionMeta for Inline<T> {
    const CANONICAL_NAME: &'static str = "jolt.inline.dispatch";
    const SOURCE_EXTENSION: Option<SourceExtension> = Some(SourceExtension::JoltInline);
    const HAS_SIDE_EFFECTS: bool = true;
}

impl<T> JoltInstructionMeta for Noop<T> {
    const CANONICAL_NAME: &'static str = "jolt.pseudo.noop";
    const JOLT_TAG: JoltInstructionTag = JoltInstructionTag(0x0000);
    const TARGET_EXTENSION: Option<JoltTargetExtension> = None;
    const HAS_SIDE_EFFECTS: bool = false;
}

macro_rules! define_source_instruction_meta {
    (
        instructions: [$($(#[$meta:meta])* $instr:ident => $marker:ident => $canonical_name:expr),* $(,)?]
    ) => {
        $(
            $(#[$meta])*
            impl<T> SourceInstructionMeta for $marker<T> {
                const CANONICAL_NAME: &'static str = $canonical_name;
                const SOURCE_EXTENSION: Option<SourceExtension> =
                    source_extension_for_marker!($marker);
                const HAS_SIDE_EFFECTS: bool = source_side_effects_for_marker!($marker);
            }
        )*
    };
}

macro_rules! define_jolt_instruction_meta {
    (
        instructions: [$($(#[$meta:meta])* $instr:ident => $marker:ident => ($tag:expr, $canonical_name:expr)),* $(,)?]
    ) => {
        $(
            $(#[$meta])*
            impl<T> JoltInstructionMeta for $marker<T> {
                const CANONICAL_NAME: &'static str = $canonical_name;
                const JOLT_TAG: JoltInstructionTag = JoltInstructionTag($tag);
                const TARGET_EXTENSION: Option<JoltTargetExtension> =
                    jolt_target_extension_for_marker!($marker);
                const HAS_SIDE_EFFECTS: bool = jolt_side_effects_for_marker!($marker);
            }
        )*
    };
}

crate::for_each_instruction_kind!(define_source_instruction_meta);
crate::for_each_jolt_instruction_kind!(define_jolt_instruction_meta);

macro_rules! define_source_instruction_kind {
    (
        instructions: [$($(#[$meta:meta])* $instr:ident => $marker:ident => $canonical_name:expr),* $(,)?]
    ) => {
        #[expect(
            non_upper_case_globals,
            reason = "Kind constants preserve existing instruction spelling"
        )]
        impl SourceInstructionKind {
            pub const NoOp: Self = SourceInstruction::Noop(Noop(()));
            pub const Unimpl: Self = SourceInstruction::Unimplemented(Unimpl(()));
            $(
                $(#[$meta])*
                pub const $instr: Self = SourceInstruction::$marker($marker(()));
            )*
            pub const Inline: Self = SourceInstruction::InlineDispatch(Inline(()));

            pub const ALL: &'static [Self] = &[
                SourceInstruction::Noop(Noop(())),
                SourceInstruction::Unimplemented(Unimpl(())),
                $(
                    $(#[$meta])*
                    SourceInstruction::$marker($marker(())),
                )*
                SourceInstruction::InlineDispatch(Inline(())),
            ];

            pub const fn name(self) -> &'static str {
                match self {
                    SourceInstruction::Noop(_) => "NoOp",
                    SourceInstruction::Unimplemented(_) => "Unimpl",
                    $(
                        $(#[$meta])*
                        SourceInstruction::$marker(_) => stringify!($instr),
                    )*
                    SourceInstruction::InlineDispatch(_) => "Inline",
                }
            }

            pub const fn canonical_name(self) -> &'static str {
                match self {
                    SourceInstruction::Noop(_) => <Noop<()> as SourceInstructionMeta>::CANONICAL_NAME,
                    SourceInstruction::Unimplemented(_) => {
                        <Unimpl<()> as SourceInstructionMeta>::CANONICAL_NAME
                    }
                    $(
                        $(#[$meta])*
                        SourceInstruction::$marker(_) => {
                            <$marker<()> as SourceInstructionMeta>::CANONICAL_NAME
                        }
                    )*
                    SourceInstruction::InlineDispatch(_) => {
                        <Inline<()> as SourceInstructionMeta>::CANONICAL_NAME
                    }
                }
            }

            pub const fn source_extension(self) -> Option<SourceExtension> {
                match self {
                    SourceInstruction::Noop(_) => <Noop<()> as SourceInstructionMeta>::SOURCE_EXTENSION,
                    SourceInstruction::Unimplemented(_) => {
                        <Unimpl<()> as SourceInstructionMeta>::SOURCE_EXTENSION
                    }
                    $(
                        $(#[$meta])*
                        SourceInstruction::$marker(_) => {
                            <$marker<()> as SourceInstructionMeta>::SOURCE_EXTENSION
                        }
                    )*
                    SourceInstruction::InlineDispatch(_) => {
                        <Inline<()> as SourceInstructionMeta>::SOURCE_EXTENSION
                    }
                }
            }

            pub fn from_canonical_name(name: &str) -> Option<Self> {
                match name {
                    "jolt.pseudo.noop" => Some(SourceInstruction::Noop(Noop(()))),
                    "jolt.pseudo.unimpl" => Some(SourceInstruction::Unimplemented(Unimpl(()))),
                    $(
                        $(#[$meta])*
                        $canonical_name => Some(SourceInstruction::$marker($marker(()))),
                    )*
                    "jolt.inline.dispatch" => Some(SourceInstruction::InlineDispatch(Inline(()))),
                    _ => None,
                }
            }

            pub fn from_name(name: &str) -> Option<Self> {
                match name {
                    "NoOp" => Some(SourceInstruction::Noop(Noop(()))),
                    "Unimpl" => Some(SourceInstruction::Unimplemented(Unimpl(()))),
                    $(
                        $(#[$meta])*
                        stringify!($instr) => Some(SourceInstruction::$marker($marker(()))),
                    )*
                    "Inline" => Some(SourceInstruction::InlineDispatch(Inline(()))),
                    _ => None,
                }
            }

            pub const fn expands_to_jolt(self) -> bool {
                !matches!(
                    self,
                    SourceInstruction::Noop(_) | SourceInstruction::Unimplemented(_)
                )
            }

            pub const fn has_side_effects(self) -> bool {
                match self {
                    SourceInstruction::Noop(_) => <Noop<()> as SourceInstructionMeta>::HAS_SIDE_EFFECTS,
                    SourceInstruction::Unimplemented(_) => {
                        <Unimpl<()> as SourceInstructionMeta>::HAS_SIDE_EFFECTS
                    }
                    $(
                        $(#[$meta])*
                        SourceInstruction::$marker(_) => {
                            <$marker<()> as SourceInstructionMeta>::HAS_SIDE_EFFECTS
                        }
                    )*
                    SourceInstruction::InlineDispatch(_) => {
                        <Inline<()> as SourceInstructionMeta>::HAS_SIDE_EFFECTS
                    }
                }
            }
        }

        impl Default for SourceInstructionKind {
            fn default() -> Self {
                SourceInstruction::Noop(Noop(()))
            }
        }
    };
}

macro_rules! define_jolt_instruction_kind {
    (
        instructions: [$($(#[$meta:meta])* $instr:ident => $marker:ident => ($tag:expr, $canonical_name:expr)),* $(,)?]
    ) => {
        #[expect(
            non_upper_case_globals,
            reason = "Kind constants preserve existing instruction spelling"
        )]
        impl JoltInstructionKind {
            pub const NoOp: Self = JoltInstruction::Noop(Noop(()));
            $(
                $(#[$meta])*
                pub const $instr: Self = JoltInstruction::$marker($marker(()));
            )*

            pub const ALL: &'static [Self] = &[
                JoltInstruction::Noop(Noop(())),
                $(
                    $(#[$meta])*
                    JoltInstruction::$marker($marker(())),
                )*
            ];

            pub const fn name(self) -> &'static str {
                match self {
                    JoltInstruction::Noop(_) => "NoOp",
                    $(
                        $(#[$meta])*
                        JoltInstruction::$marker(_) => stringify!($instr),
                    )*
                }
            }

            pub const fn canonical_name(self) -> &'static str {
                match self {
                    JoltInstruction::Noop(_) => <Noop<()> as JoltInstructionMeta>::CANONICAL_NAME,
                    $(
                        $(#[$meta])*
                        JoltInstruction::$marker(_) => {
                            <$marker<()> as JoltInstructionMeta>::CANONICAL_NAME
                        }
                    )*
                }
            }

            pub const fn tag(self) -> JoltInstructionTag {
                match self {
                    JoltInstruction::Noop(_) => <Noop<()> as JoltInstructionMeta>::JOLT_TAG,
                    $(
                        $(#[$meta])*
                        JoltInstruction::$marker(_) => <$marker<()> as JoltInstructionMeta>::JOLT_TAG,
                    )*
                }
            }

            pub const fn target_extension(self) -> Option<JoltTargetExtension> {
                match self {
                    JoltInstruction::Noop(_) => <Noop<()> as JoltInstructionMeta>::TARGET_EXTENSION,
                    $(
                        $(#[$meta])*
                        JoltInstruction::$marker(_) => {
                            <$marker<()> as JoltInstructionMeta>::TARGET_EXTENSION
                        }
                    )*
                }
            }

            pub const fn from_tag(tag: JoltInstructionTag) -> Option<Self> {
                match tag.0 {
                    0x0000 => Some(JoltInstruction::Noop(Noop(()))),
                    $(
                        $(#[$meta])*
                        $tag => Some(JoltInstruction::$marker($marker(()))),
                    )*
                    _ => None,
                }
            }

            pub fn from_name(name: &str) -> Option<Self> {
                match name {
                    "NoOp" => Some(JoltInstruction::Noop(Noop(()))),
                    $(
                        $(#[$meta])*
                        stringify!($instr) => Some(JoltInstruction::$marker($marker(()))),
                    )*
                    _ => None,
                }
            }

            pub const fn has_side_effects(self) -> bool {
                match self {
                    JoltInstruction::Noop(_) => <Noop<()> as JoltInstructionMeta>::HAS_SIDE_EFFECTS,
                    $(
                        $(#[$meta])*
                        JoltInstruction::$marker(_) => {
                            <$marker<()> as JoltInstructionMeta>::HAS_SIDE_EFFECTS
                        }
                    )*
                }
            }
        }

        impl SourceInstructionKind {
            pub const fn from_jolt_kind(kind: JoltInstructionKind) -> Option<Self> {
                match kind {
                    JoltInstruction::Noop(_) => Some(SourceInstruction::Noop(Noop(()))),
                    $(
                        $(#[$meta])*
                        JoltInstruction::$marker(_) => Some(SourceInstruction::$marker($marker(()))),
                    )*
                }
            }

            pub const fn jolt_kind(self) -> Option<JoltInstructionKind> {
                match self {
                    SourceInstruction::Noop(_) => Some(JoltInstruction::Noop(Noop(()))),
                    $(
                        $(#[$meta])*
                        SourceInstruction::$marker(_) => Some(JoltInstruction::$marker($marker(()))),
                    )*
                    _ => None,
                }
            }
        }

        impl Default for JoltInstructionKind {
            fn default() -> Self {
                JoltInstruction::Noop(Noop(()))
            }
        }
    };
}

crate::for_each_instruction_kind!(define_source_instruction_kind);
crate::for_each_jolt_instruction_kind!(define_jolt_instruction_kind);

#[cfg(feature = "serialization")]
impl Serialize for SourceInstructionKind {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.canonical_name())
    }
}

#[cfg(feature = "serialization")]
impl<'de> serde::Deserialize<'de> for SourceInstructionKind {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct SourceInstructionKindVisitor;

        impl Visitor<'_> for SourceInstructionKindVisitor {
            type Value = SourceInstructionKind;

            fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                formatter.write_str("a source instruction canonical name")
            }

            fn visit_str<E>(self, name: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                SourceInstructionKind::from_canonical_name(name)
                    .ok_or_else(|| E::custom("unknown source instruction canonical name"))
            }
        }

        deserializer.deserialize_str(SourceInstructionKindVisitor)
    }
}

#[cfg(feature = "serialization")]
impl Serialize for JoltInstructionKind {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u16(self.tag().0)
    }
}

#[cfg(feature = "serialization")]
impl<'de> serde::Deserialize<'de> for JoltInstructionKind {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct JoltInstructionKindVisitor;

        impl Visitor<'_> for JoltInstructionKindVisitor {
            type Value = JoltInstructionKind;

            fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                formatter.write_str("a final Jolt instruction u16 tag")
            }

            fn visit_u16<E>(self, value: u16) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                JoltInstructionKind::from_tag(JoltInstructionTag(value))
                    .ok_or_else(|| E::custom("unknown final Jolt instruction tag"))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                let tag = u16::try_from(value)
                    .map_err(|_| E::custom("final Jolt instruction tag out of range"))?;
                self.visit_u16(tag)
            }
        }

        deserializer.deserialize_u16(JoltInstructionKindVisitor)
    }
}

#[cfg(feature = "serialization")]
impl CanonicalSerialize for SourceInstructionKind {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.canonical_name()
            .as_bytes()
            .to_vec()
            .serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.canonical_name()
            .as_bytes()
            .to_vec()
            .serialized_size(compress)
    }
}

#[cfg(feature = "serialization")]
impl CanonicalDeserialize for SourceInstructionKind {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let bytes = Vec::<u8>::deserialize_with_mode(reader, compress, validate)?;
        let name = std::str::from_utf8(&bytes).map_err(|_| SerializationError::InvalidData)?;
        Self::from_canonical_name(name).ok_or(SerializationError::InvalidData)
    }
}

#[cfg(feature = "serialization")]
impl Valid for SourceInstructionKind {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

#[cfg(feature = "serialization")]
impl CanonicalSerialize for JoltInstructionKind {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.tag().0.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.tag().0.serialized_size(compress)
    }
}

#[cfg(feature = "serialization")]
impl CanonicalDeserialize for JoltInstructionKind {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let value = u16::deserialize_with_mode(reader, compress, validate)?;
        Self::from_tag(JoltInstructionTag(value)).ok_or(SerializationError::InvalidData)
    }
}

#[cfg(feature = "serialization")]
impl Valid for JoltInstructionKind {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{JoltInstructionKind, JoltInstructionTag, SourceInstructionKind};
    use crate::instructions::{JoltInstruction, SourceInstruction, VirtualHostIO};
    use std::collections::HashSet;

    #[test]
    fn tags_are_stable_for_representative_final_rows() {
        assert_eq!(JoltInstructionKind::NoOp.tag(), JoltInstructionTag(0x0000));
        assert_eq!(JoltInstructionKind::ADD.tag(), JoltInstructionTag(0x0002));
        assert_eq!(
            JoltInstruction::VirtualHostIO(VirtualHostIO(())).tag(),
            JoltInstructionTag(0x0068)
        );
        assert_eq!(
            JoltInstructionKind::VirtualXORROTW7.tag(),
            JoltInstructionTag(0x0088)
        );
    }

    #[test]
    fn final_tags_round_trip_and_are_unique() {
        let mut seen = HashSet::new();
        for kind in JoltInstructionKind::ALL {
            let tag = kind.tag();
            assert!(seen.insert(tag), "duplicate tag {tag:?} for {kind:?}");
            assert_eq!(JoltInstructionKind::from_tag(tag), Some(*kind));
        }
    }

    #[test]
    fn source_kinds_use_canonical_names_instead_of_tags() {
        assert_eq!(SourceInstructionKind::ADD.canonical_name(), "rv64.add");
        assert_eq!(
            SourceInstructionKind::Inline.canonical_name(),
            "jolt.inline.dispatch"
        );
        assert_eq!(
            SourceInstructionKind::from_canonical_name("rv64.addw"),
            Some(SourceInstructionKind::ADDW)
        );
    }

    #[test]
    fn source_canonical_names_are_unique_and_non_empty() {
        let mut seen = HashSet::new();
        for kind in SourceInstructionKind::ALL {
            let name = kind.canonical_name();
            assert!(!name.is_empty(), "empty canonical name for {kind:?}");
            assert!(seen.insert(name), "duplicate canonical name {name:?}");
        }
    }

    #[test]
    fn source_to_final_mapping_is_partial() {
        assert_eq!(
            SourceInstructionKind::ADD.jolt_kind(),
            Some(JoltInstructionKind::ADD)
        );
        assert_eq!(
            SourceInstruction::VirtualHostIO(VirtualHostIO(())).jolt_kind(),
            Some(JoltInstruction::VirtualHostIO(VirtualHostIO(())))
        );
        assert_eq!(SourceInstructionKind::ADDW.jolt_kind(), None);
        assert_eq!(SourceInstructionKind::Inline.jolt_kind(), None);
        assert_eq!(SourceInstructionKind::Unimpl.jolt_kind(), None);
    }

    #[test]
    fn final_canonical_names_are_unique_and_non_empty() {
        let mut seen = HashSet::new();
        for kind in JoltInstructionKind::ALL {
            let name = kind.canonical_name();
            assert!(!name.is_empty(), "empty canonical name for {kind:?}");
            assert!(seen.insert(name), "duplicate canonical name {name:?}");
        }
    }

    #[cfg(feature = "serialization")]
    #[test]
    fn serde_uses_source_names_and_final_tags() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(
            serde_json::to_string(&SourceInstructionKind::ADDW)?,
            "\"rv64.addw\""
        );
        assert_eq!(
            serde_json::from_str::<SourceInstructionKind>("\"rv64.addw\"")?,
            SourceInstructionKind::ADDW
        );

        assert_eq!(
            serde_json::to_string(&JoltInstructionKind::ADD)?,
            JoltInstructionKind::ADD.tag().0.to_string()
        );
        assert_eq!(
            serde_json::from_str::<JoltInstructionKind>("2")?,
            JoltInstructionKind::ADD
        );
        assert!(serde_json::from_str::<JoltInstructionKind>("1").is_err());
        Ok(())
    }

    #[cfg(feature = "serialization")]
    #[test]
    fn canonical_serialization_round_trips_stable_identity() {
        use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

        let mut source_bytes = Vec::new();
        assert!(SourceInstructionKind::ADDW
            .serialize_compressed(&mut source_bytes)
            .is_ok());
        assert!(matches!(
            SourceInstructionKind::deserialize_compressed(&source_bytes[..]),
            Ok(SourceInstructionKind::ADDW)
        ));

        let mut final_bytes = Vec::new();
        assert!(JoltInstructionKind::ADD
            .serialize_compressed(&mut final_bytes)
            .is_ok());
        assert!(matches!(
            JoltInstructionKind::deserialize_compressed(&final_bytes[..]),
            Ok(JoltInstructionKind::ADD)
        ));
        assert_eq!(final_bytes, JoltInstructionKind::ADD.tag().0.to_le_bytes());
    }
}
