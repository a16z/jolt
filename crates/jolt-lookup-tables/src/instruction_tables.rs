//! Mapping from Jolt instruction structs to their lookup table decomposition.

use crate::tables::LookupTableKind;
use jolt_trace::instructions::{
    Add, AddW, Addi, AddiW, And, AndI, Andn, AssertEq, AssertHalfwordAlignment, AssertLte,
    AssertMulUNoOverflow, AssertValidDiv0, AssertValidUnsignedRemainder, AssertWordAlignment,
    Auipc, Bge, BgeU, Beq, Blt, BltU, Bne, Div, DivU, DivUW, DivW, Ebreak, Ecall, Fence, Jal,
    Jalr, Lb, Lbu, Ld, Lh, Lhu, Lui, Lw, Lwu, MovSign, Mul, MulH, MulHSU, MulHU, MulI, MulW, Noop,
    Or, OrI, Pow2, Pow2I, Pow2IW, Pow2W, Rem, RemU, RemUW, RemW, Sb, Sd, Sh, Sll, SllI, SllIW,
    SllW, Slt, SltI, SltIU, SltU, Sra, SraI, SraIW, SraW, Srl, SrlI, SrlIW, SrlW, Sub, SubW, Sw,
    VirtualAdvice, VirtualAdviceLen, VirtualAdviceLoad, VirtualChangeDivisor,
    VirtualChangeDivisorW, VirtualHostIO, VirtualRev8W, VirtualRotri, VirtualRotriw,
    VirtualShiftRightBitmask, VirtualShiftRightBitmaski, VirtualSignExtendWord, VirtualSra,
    VirtualSrai, VirtualSrl, VirtualSrli, VirtualXorRot16, VirtualXorRot24, VirtualXorRot32,
    VirtualXorRot63, VirtualXorRotW12, VirtualXorRotW16, VirtualXorRotW7, VirtualXorRotW8,
    VirtualZeroExtendWord, Xor, XorI,
};

/// Maps an instruction to the lookup table it decomposes into for the proving system.
///
/// Returns `None` for instructions that don't use lookup tables (loads, stores,
/// system instructions). The prover uses this to route instruction evaluations
/// to the correct table during the instruction sumcheck.
pub trait InstructionLookupTable {
    fn lookup_table(&self) -> Option<LookupTableKind>;
}

macro_rules! impl_lookup_table {
    ($instr:ty, Some($table:ident)) => {
        impl InstructionLookupTable for $instr {
            #[inline]
            fn lookup_table(&self) -> Option<LookupTableKind> {
                Some(LookupTableKind::$table)
            }
        }
    };
    ($instr:ty, None) => {
        impl InstructionLookupTable for $instr {
            #[inline]
            fn lookup_table(&self) -> Option<LookupTableKind> {
                None
            }
        }
    };
}

impl_lookup_table!(Add, Some(RangeCheck));
impl_lookup_table!(Addi, Some(RangeCheck));
impl_lookup_table!(Sub, Some(RangeCheck));
impl_lookup_table!(Lui, Some(RangeCheck));
impl_lookup_table!(Auipc, Some(RangeCheck));
impl_lookup_table!(Mul, Some(RangeCheck));
impl_lookup_table!(MulH, None);
impl_lookup_table!(MulHSU, None);
impl_lookup_table!(MulHU, Some(UpperWord));
impl_lookup_table!(Div, None);
impl_lookup_table!(DivU, None);
impl_lookup_table!(Rem, None);
impl_lookup_table!(RemU, None);

impl_lookup_table!(AddW, None);
impl_lookup_table!(AddiW, None);
impl_lookup_table!(SubW, None);
impl_lookup_table!(MulW, None);
impl_lookup_table!(DivW, None);
impl_lookup_table!(DivUW, None);
impl_lookup_table!(RemW, None);
impl_lookup_table!(RemUW, None);

impl_lookup_table!(And, Some(And));
impl_lookup_table!(AndI, Some(And));
impl_lookup_table!(Or, Some(Or));
impl_lookup_table!(OrI, Some(Or));
impl_lookup_table!(Xor, Some(Xor));
impl_lookup_table!(XorI, Some(Xor));
impl_lookup_table!(Andn, Some(Andn));

impl_lookup_table!(Sll, None);
impl_lookup_table!(SllI, None);
impl_lookup_table!(Srl, None);
impl_lookup_table!(SrlI, None);
impl_lookup_table!(Sra, None);
impl_lookup_table!(SraI, None);

impl_lookup_table!(SllW, None);
impl_lookup_table!(SllIW, None);
impl_lookup_table!(SrlW, None);
impl_lookup_table!(SrlIW, None);
impl_lookup_table!(SraW, None);
impl_lookup_table!(SraIW, None);

impl_lookup_table!(Slt, Some(SignedLessThan));
impl_lookup_table!(SltI, Some(SignedLessThan));
impl_lookup_table!(SltU, Some(UnsignedLessThan));
impl_lookup_table!(SltIU, Some(UnsignedLessThan));

impl_lookup_table!(Beq, Some(Equal));
impl_lookup_table!(Bne, Some(NotEqual));
impl_lookup_table!(Blt, Some(SignedLessThan));
impl_lookup_table!(Bge, Some(SignedGreaterThanEqual));
impl_lookup_table!(BltU, Some(UnsignedLessThan));
impl_lookup_table!(BgeU, Some(UnsignedGreaterThanEqual));

impl_lookup_table!(Lb, None);
impl_lookup_table!(Lbu, None);
impl_lookup_table!(Lh, None);
impl_lookup_table!(Lhu, None);
impl_lookup_table!(Lw, None);
impl_lookup_table!(Lwu, None);
impl_lookup_table!(Ld, None);

impl_lookup_table!(Sb, None);
impl_lookup_table!(Sh, None);
impl_lookup_table!(Sw, None);
impl_lookup_table!(Sd, None);

impl_lookup_table!(Ecall, None);
impl_lookup_table!(Ebreak, None);
impl_lookup_table!(Fence, None);
impl_lookup_table!(Noop, None);

impl_lookup_table!(Jal, Some(RangeCheck));
impl_lookup_table!(Jalr, Some(RangeCheckAligned));

impl_lookup_table!(AssertEq, Some(Equal));
impl_lookup_table!(AssertLte, Some(UnsignedLessThanEqual));
impl_lookup_table!(AssertValidDiv0, Some(ValidDiv0));
impl_lookup_table!(AssertValidUnsignedRemainder, Some(ValidUnsignedRemainder));
impl_lookup_table!(AssertMulUNoOverflow, Some(MulUNoOverflow));
impl_lookup_table!(AssertWordAlignment, Some(WordAlignment));
impl_lookup_table!(AssertHalfwordAlignment, Some(HalfwordAlignment));

impl_lookup_table!(Pow2, Some(Pow2));
impl_lookup_table!(Pow2I, Some(Pow2));
impl_lookup_table!(Pow2W, Some(Pow2W));
impl_lookup_table!(Pow2IW, Some(Pow2W));
impl_lookup_table!(MulI, Some(RangeCheck));

impl_lookup_table!(MovSign, Some(SignMask));

impl_lookup_table!(VirtualRev8W, Some(VirtualRev8W));

impl_lookup_table!(VirtualSrl, Some(VirtualSRL));
impl_lookup_table!(VirtualSrli, Some(VirtualSRL));
impl_lookup_table!(VirtualSra, Some(VirtualSRA));
impl_lookup_table!(VirtualSrai, Some(VirtualSRA));
impl_lookup_table!(VirtualShiftRightBitmask, Some(ShiftRightBitmask));
impl_lookup_table!(VirtualShiftRightBitmaski, Some(ShiftRightBitmask));
impl_lookup_table!(VirtualRotri, Some(VirtualROTR));
impl_lookup_table!(VirtualRotriw, Some(VirtualROTRW));

impl_lookup_table!(VirtualChangeDivisor, Some(VirtualChangeDivisor));
impl_lookup_table!(VirtualChangeDivisorW, Some(VirtualChangeDivisorW));

impl_lookup_table!(VirtualSignExtendWord, Some(RangeCheck));
impl_lookup_table!(VirtualZeroExtendWord, Some(RangeCheck));

impl_lookup_table!(VirtualXorRot32, Some(VirtualXORROT32));
impl_lookup_table!(VirtualXorRot24, Some(VirtualXORROT24));
impl_lookup_table!(VirtualXorRot16, Some(VirtualXORROT16));
impl_lookup_table!(VirtualXorRot63, Some(VirtualXORROT63));
impl_lookup_table!(VirtualXorRotW16, Some(VirtualXORROTW16));
impl_lookup_table!(VirtualXorRotW12, Some(VirtualXORROTW12));
impl_lookup_table!(VirtualXorRotW8, Some(VirtualXORROTW8));
impl_lookup_table!(VirtualXorRotW7, Some(VirtualXORROTW7));

impl_lookup_table!(VirtualAdvice, Some(RangeCheck));
impl_lookup_table!(VirtualAdviceLen, Some(RangeCheck));
impl_lookup_table!(VirtualAdviceLoad, Some(RangeCheck));
impl_lookup_table!(VirtualHostIO, None);
