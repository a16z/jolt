//! Extension trait mapping instructions to their lookup table decomposition.

use crate::tables::LookupTableKind;
use jolt_riscv::Instruction;

/// Maps an instruction to the lookup table it decomposes into for the proving system.
///
/// Returns `None` for instructions that don't use lookup tables (loads, stores,
/// system instructions). The prover uses this to route instruction evaluations
/// to the correct table during the instruction sumcheck.
pub trait InstructionLookupTable: Instruction {
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

// RV64I arithmetic
impl_lookup_table!(jolt_riscv::rv::arithmetic::Add, Some(RangeCheck));
impl_lookup_table!(jolt_riscv::rv::arithmetic::Addi, Some(RangeCheck));
impl_lookup_table!(jolt_riscv::rv::arithmetic::Sub, Some(RangeCheck));
impl_lookup_table!(jolt_riscv::rv::arithmetic::Lui, Some(RangeCheck));
impl_lookup_table!(jolt_riscv::rv::arithmetic::Auipc, Some(RangeCheck));
impl_lookup_table!(jolt_riscv::rv::arithmetic::Mul, Some(RangeCheck));
impl_lookup_table!(jolt_riscv::rv::arithmetic::MulH, None);
impl_lookup_table!(jolt_riscv::rv::arithmetic::MulHSU, None);
impl_lookup_table!(jolt_riscv::rv::arithmetic::MulHU, Some(UpperWord));
impl_lookup_table!(jolt_riscv::rv::arithmetic::Div, None);
impl_lookup_table!(jolt_riscv::rv::arithmetic::DivU, None);
impl_lookup_table!(jolt_riscv::rv::arithmetic::Rem, None);
impl_lookup_table!(jolt_riscv::rv::arithmetic::RemU, None);

// RV64 W-suffix arithmetic
impl_lookup_table!(jolt_riscv::rv::arithmetic_w::AddW, None);
impl_lookup_table!(jolt_riscv::rv::arithmetic_w::AddiW, None);
impl_lookup_table!(jolt_riscv::rv::arithmetic_w::SubW, None);
impl_lookup_table!(jolt_riscv::rv::arithmetic_w::MulW, None);
impl_lookup_table!(jolt_riscv::rv::arithmetic_w::DivW, None);
impl_lookup_table!(jolt_riscv::rv::arithmetic_w::DivUW, None);
impl_lookup_table!(jolt_riscv::rv::arithmetic_w::RemW, None);
impl_lookup_table!(jolt_riscv::rv::arithmetic_w::RemUW, None);

// RV64I logic
impl_lookup_table!(jolt_riscv::rv::logic::And, Some(And));
impl_lookup_table!(jolt_riscv::rv::logic::AndI, Some(And));
impl_lookup_table!(jolt_riscv::rv::logic::Or, Some(Or));
impl_lookup_table!(jolt_riscv::rv::logic::OrI, Some(Or));
impl_lookup_table!(jolt_riscv::rv::logic::Xor, Some(Xor));
impl_lookup_table!(jolt_riscv::rv::logic::XorI, Some(Xor));
impl_lookup_table!(jolt_riscv::rv::logic::Andn, Some(Andn));

// RV64I shifts
impl_lookup_table!(jolt_riscv::rv::shift::Sll, None);
impl_lookup_table!(jolt_riscv::rv::shift::SllI, None);
impl_lookup_table!(jolt_riscv::rv::shift::Srl, None);
impl_lookup_table!(jolt_riscv::rv::shift::SrlI, None);
impl_lookup_table!(jolt_riscv::rv::shift::Sra, None);
impl_lookup_table!(jolt_riscv::rv::shift::SraI, None);

// RV64 W-suffix shifts
impl_lookup_table!(jolt_riscv::rv::shift_w::SllW, None);
impl_lookup_table!(jolt_riscv::rv::shift_w::SllIW, None);
impl_lookup_table!(jolt_riscv::rv::shift_w::SrlW, None);
impl_lookup_table!(jolt_riscv::rv::shift_w::SrlIW, None);
impl_lookup_table!(jolt_riscv::rv::shift_w::SraW, None);
impl_lookup_table!(jolt_riscv::rv::shift_w::SraIW, None);

// RV64I compare
impl_lookup_table!(jolt_riscv::rv::compare::Slt, Some(SignedLessThan));
impl_lookup_table!(jolt_riscv::rv::compare::SltI, Some(SignedLessThan));
impl_lookup_table!(jolt_riscv::rv::compare::SltU, Some(UnsignedLessThan));
impl_lookup_table!(jolt_riscv::rv::compare::SltIU, Some(UnsignedLessThan));

// RV64I branch
impl_lookup_table!(jolt_riscv::rv::branch::Beq, Some(Equal));
impl_lookup_table!(jolt_riscv::rv::branch::Bne, Some(NotEqual));
impl_lookup_table!(jolt_riscv::rv::branch::Blt, Some(SignedLessThan));
impl_lookup_table!(jolt_riscv::rv::branch::Bge, Some(SignedGreaterThanEqual));
impl_lookup_table!(jolt_riscv::rv::branch::BltU, Some(UnsignedLessThan));
impl_lookup_table!(jolt_riscv::rv::branch::BgeU, Some(UnsignedGreaterThanEqual));

// RV64I load
impl_lookup_table!(jolt_riscv::rv::load::Lb, None);
impl_lookup_table!(jolt_riscv::rv::load::Lbu, None);
impl_lookup_table!(jolt_riscv::rv::load::Lh, None);
impl_lookup_table!(jolt_riscv::rv::load::Lhu, None);
impl_lookup_table!(jolt_riscv::rv::load::Lw, None);
impl_lookup_table!(jolt_riscv::rv::load::Lwu, None);
impl_lookup_table!(jolt_riscv::rv::load::Ld, None);

// RV64I store
impl_lookup_table!(jolt_riscv::rv::store::Sb, None);
impl_lookup_table!(jolt_riscv::rv::store::Sh, None);
impl_lookup_table!(jolt_riscv::rv::store::Sw, None);
impl_lookup_table!(jolt_riscv::rv::store::Sd, None);

// RV64I system
impl_lookup_table!(jolt_riscv::rv::system::Ecall, None);
impl_lookup_table!(jolt_riscv::rv::system::Ebreak, None);
impl_lookup_table!(jolt_riscv::rv::system::Fence, None);
impl_lookup_table!(jolt_riscv::rv::system::Noop, None);

// RV64I jump
impl_lookup_table!(jolt_riscv::rv::jump::Jal, Some(RangeCheck));
impl_lookup_table!(jolt_riscv::rv::jump::Jalr, Some(RangeCheckAligned));

// Virtual assert
impl_lookup_table!(jolt_riscv::virt::assert::AssertEq, Some(Equal));
impl_lookup_table!(
    jolt_riscv::virt::assert::AssertLte,
    Some(UnsignedLessThanEqual)
);
impl_lookup_table!(jolt_riscv::virt::assert::AssertValidDiv0, Some(ValidDiv0));
impl_lookup_table!(
    jolt_riscv::virt::assert::AssertValidUnsignedRemainder,
    Some(ValidUnsignedRemainder)
);
impl_lookup_table!(
    jolt_riscv::virt::assert::AssertMulUNoOverflow,
    Some(MulUNoOverflow)
);
impl_lookup_table!(
    jolt_riscv::virt::assert::AssertWordAlignment,
    Some(WordAlignment)
);
impl_lookup_table!(
    jolt_riscv::virt::assert::AssertHalfwordAlignment,
    Some(HalfwordAlignment)
);

// Virtual arithmetic
impl_lookup_table!(jolt_riscv::virt::arithmetic::Pow2, Some(Pow2));
impl_lookup_table!(jolt_riscv::virt::arithmetic::Pow2I, Some(Pow2));
impl_lookup_table!(jolt_riscv::virt::arithmetic::Pow2W, Some(Pow2W));
impl_lookup_table!(jolt_riscv::virt::arithmetic::Pow2IW, Some(Pow2W));
impl_lookup_table!(jolt_riscv::virt::arithmetic::MulI, Some(RangeCheck));

// Virtual bitwise
impl_lookup_table!(jolt_riscv::virt::bitwise::MovSign, Some(SignMask));

// Virtual byte
impl_lookup_table!(jolt_riscv::virt::byte::VirtualRev8W, Some(VirtualRev8W));

// Virtual shift
impl_lookup_table!(jolt_riscv::virt::shift::VirtualSrl, Some(VirtualSRL));
impl_lookup_table!(jolt_riscv::virt::shift::VirtualSrli, Some(VirtualSRL));
impl_lookup_table!(jolt_riscv::virt::shift::VirtualSra, Some(VirtualSRA));
impl_lookup_table!(jolt_riscv::virt::shift::VirtualSrai, Some(VirtualSRA));
impl_lookup_table!(
    jolt_riscv::virt::shift::VirtualShiftRightBitmask,
    Some(ShiftRightBitmask)
);
impl_lookup_table!(
    jolt_riscv::virt::shift::VirtualShiftRightBitmaski,
    Some(ShiftRightBitmask)
);
impl_lookup_table!(jolt_riscv::virt::shift::VirtualRotri, Some(VirtualROTR));
impl_lookup_table!(jolt_riscv::virt::shift::VirtualRotriw, Some(VirtualROTRW));

// Virtual division
impl_lookup_table!(
    jolt_riscv::virt::division::VirtualChangeDivisor,
    Some(VirtualChangeDivisor)
);
impl_lookup_table!(
    jolt_riscv::virt::division::VirtualChangeDivisorW,
    Some(VirtualChangeDivisorW)
);

// Virtual extension
impl_lookup_table!(
    jolt_riscv::virt::extension::VirtualSignExtendWord,
    Some(RangeCheck)
);
impl_lookup_table!(
    jolt_riscv::virt::extension::VirtualZeroExtendWord,
    Some(RangeCheck)
);

// Virtual XOR-rotate
impl_lookup_table!(
    jolt_riscv::virt::xor_rotate::VirtualXorRot32,
    Some(VirtualXORROT32)
);
impl_lookup_table!(
    jolt_riscv::virt::xor_rotate::VirtualXorRot24,
    Some(VirtualXORROT24)
);
impl_lookup_table!(
    jolt_riscv::virt::xor_rotate::VirtualXorRot16,
    Some(VirtualXORROT16)
);
impl_lookup_table!(
    jolt_riscv::virt::xor_rotate::VirtualXorRot63,
    Some(VirtualXORROT63)
);
impl_lookup_table!(
    jolt_riscv::virt::xor_rotate::VirtualXorRotW16,
    Some(VirtualXORROTW16)
);
impl_lookup_table!(
    jolt_riscv::virt::xor_rotate::VirtualXorRotW12,
    Some(VirtualXORROTW12)
);
impl_lookup_table!(
    jolt_riscv::virt::xor_rotate::VirtualXorRotW8,
    Some(VirtualXORROTW8)
);
impl_lookup_table!(
    jolt_riscv::virt::xor_rotate::VirtualXorRotW7,
    Some(VirtualXORROTW7)
);

// Virtual advice/IO
impl_lookup_table!(jolt_riscv::virt::advice::VirtualAdvice, Some(RangeCheck));
impl_lookup_table!(jolt_riscv::virt::advice::VirtualAdviceLen, Some(RangeCheck));
impl_lookup_table!(
    jolt_riscv::virt::advice::VirtualAdviceLoad,
    Some(RangeCheck)
);
impl_lookup_table!(jolt_riscv::virt::advice::VirtualHostIO, None);

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_riscv::rv::arithmetic::Add;
    use jolt_riscv::rv::load::Lw;

    #[test]
    fn lookup_table_assignment() {
        assert!(
            Add.lookup_table().is_some(),
            "ADD should have a lookup table"
        );
        assert!(
            Lw.lookup_table().is_none(),
            "LW should not have a lookup table"
        );
    }
}
