/// Assembly-style sugar over Jolt expansion builders.
///
/// Each statement uses RISC-V operand order: destination first. Instruction
/// kinds that are already final Jolt rows are emitted directly; source-only
/// kinds are recursively expanded by the builder.
///
/// ```ignore
/// jolt_asm!(asm, {
///     advice quotient;
///     mul product, quotient, divisor;
///     sub remainder, dividend, product;
/// });
/// ```
#[macro_export]
macro_rules! jolt_asm {
    ($asm:expr, { $($mnemonic:ident $($operand:expr),* ;)* }) => {
        $($crate::__jolt_asm_stmt!($asm, $mnemonic $($operand),*);)*
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __jolt_asm_stmt {
    ($asm:expr, add $rd:expr, $rs1:expr, $rs2:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_r(Kind::ADD, $rd, $rs1, $rs2)
    }};
    ($asm:expr, sub $rd:expr, $rs1:expr, $rs2:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_r(Kind::SUB, $rd, $rs1, $rs2)
    }};
    ($asm:expr, and $rd:expr, $rs1:expr, $rs2:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_r(Kind::AND, $rd, $rs1, $rs2)
    }};
    ($asm:expr, or $rd:expr, $rs1:expr, $rs2:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_r(Kind::OR, $rd, $rs1, $rs2)
    }};
    ($asm:expr, xor $rd:expr, $rs1:expr, $rs2:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_r(Kind::XOR, $rd, $rs1, $rs2)
    }};
    ($asm:expr, andn $rd:expr, $rs1:expr, $rs2:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_r(Kind::ANDN, $rd, $rs1, $rs2)
    }};
    ($asm:expr, mul $rd:expr, $rs1:expr, $rs2:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_r(Kind::MUL, $rd, $rs1, $rs2)
    }};
    ($asm:expr, mulhu $rd:expr, $rs1:expr, $rs2:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_r(Kind::MULHU, $rd, $rs1, $rs2)
    }};
    ($asm:expr, pext $rd:expr, $rs1:expr, $rs2:expr) => {{
        use $crate::expand::asm_support::SourceKind;
        $asm.emit_r(SourceKind::VirtualPext, $rd, $rs1, $rs2)
    }};
    ($asm:expr, pext_signed $rd:expr, $rs1:expr, $rs2:expr) => {{
        use $crate::expand::asm_support::SourceKind;
        $asm.emit_r(SourceKind::VirtualPextSigned, $rd, $rs1, $rs2)
    }};
    ($asm:expr, sltu $rd:expr, $rs1:expr, $rs2:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_r(Kind::SLTU, $rd, $rs1, $rs2)
    }};
    ($asm:expr, negate_if $rd:expr, $condition:expr, $value:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_r(Kind::VIRTUAL_NEGATE_IF, $rd, $condition, $value)
    }};
    ($asm:expr, xorrot16 $rd:expr, $rs1:expr, $rs2:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_r(Kind::VirtualXORROT16, $rd, $rs1, $rs2)
    }};
    ($asm:expr, xorrot24 $rd:expr, $rs1:expr, $rs2:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_r(Kind::VirtualXORROT24, $rd, $rs1, $rs2)
    }};
    ($asm:expr, xorrot32 $rd:expr, $rs1:expr, $rs2:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_r(Kind::VirtualXORROT32, $rd, $rs1, $rs2)
    }};
    ($asm:expr, xorrot63 $rd:expr, $rs1:expr, $rs2:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_r(Kind::VirtualXORROT63, $rd, $rs1, $rs2)
    }};
    ($asm:expr, xorrotw7 $rd:expr, $rs1:expr, $rs2:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_r(Kind::VirtualXORROTW7, $rd, $rs1, $rs2)
    }};
    ($asm:expr, xorrotw8 $rd:expr, $rs1:expr, $rs2:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_r(Kind::VirtualXORROTW8, $rd, $rs1, $rs2)
    }};
    ($asm:expr, xorrotw12 $rd:expr, $rs1:expr, $rs2:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_r(Kind::VirtualXORROTW12, $rd, $rs1, $rs2)
    }};
    ($asm:expr, xorrotw16 $rd:expr, $rs1:expr, $rs2:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_r(Kind::VirtualXORROTW16, $rd, $rs1, $rs2)
    }};
    ($asm:expr, xorrotw19 $rd:expr, $rs1:expr, $rs2:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_r(Kind::VirtualXORROTW19, $rd, $rs1, $rs2)
    }};
    ($asm:expr, xorrotw22 $rd:expr, $rs1:expr, $rs2:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_r(Kind::VirtualXORROTW22, $rd, $rs1, $rs2)
    }};
    ($asm:expr, xorrotw6 $rd:expr, $rs1:expr, $rs2:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_r(Kind::VirtualXORROTW6, $rd, $rs1, $rs2)
    }};
    ($asm:expr, addi $rd:expr, $rs1:expr, $imm:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_i(Kind::ADDI, $rd, $rs1, $imm)
    }};
    ($asm:expr, andi $rd:expr, $rs1:expr, $imm:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_i(Kind::ANDI, $rd, $rs1, $imm)
    }};
    ($asm:expr, ori $rd:expr, $rs1:expr, $imm:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_i(Kind::ORI, $rd, $rs1, $imm)
    }};
    ($asm:expr, xori $rd:expr, $rs1:expr, $imm:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_i(Kind::XORI, $rd, $rs1, $imm)
    }};
    ($asm:expr, slli $rd:expr, $rs1:expr, $imm:expr) => {{
        use $crate::expand::asm_support::SourceKind;
        $asm.emit_i(SourceKind::SLLI, $rd, $rs1, $imm)
    }};
    ($asm:expr, srli $rd:expr, $rs1:expr, $imm:expr) => {{
        use $crate::expand::asm_support::SourceKind;
        $asm.emit_i(SourceKind::SRLI, $rd, $rs1, $imm)
    }};
    ($asm:expr, srliw $rd:expr, $rs1:expr, $imm:expr) => {{
        use $crate::expand::asm_support::SourceKind;
        $asm.emit_i(SourceKind::SRLIW, $rd, $rs1, $imm)
    }};
    ($asm:expr, zextw $rd:expr, $rs1:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_i(Kind::VIRTUAL_ZERO_EXTEND_WORD, $rd, $rs1, 0)
    }};
    ($asm:expr, sextw $rd:expr, $rs1:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_i(Kind::VIRTUAL_SIGN_EXTEND_WORD, $rd, $rs1, 0)
    }};
    ($asm:expr, align_addr $rd:expr, $rs1:expr, $imm:expr) => {{
        use $crate::expand::asm_support::SourceKind;
        $asm.emit_i(SourceKind::VirtualAlignAddr, $rd, $rs1, $imm)
    }};
    ($asm:expr, window_mask_b $rd:expr, $rs1:expr, $imm:expr) => {{
        use $crate::expand::asm_support::SourceKind;
        $asm.emit_i(SourceKind::VirtualWindowMaskB, $rd, $rs1, $imm)
    }};
    ($asm:expr, window_mask_h $rd:expr, $rs1:expr, $imm:expr) => {{
        use $crate::expand::asm_support::SourceKind;
        $asm.emit_i(SourceKind::VirtualWindowMaskH, $rd, $rs1, $imm)
    }};
    ($asm:expr, window_mask_w $rd:expr, $rs1:expr, $imm:expr) => {{
        use $crate::expand::asm_support::SourceKind;
        $asm.emit_i(SourceKind::VirtualWindowMaskW, $rd, $rs1, $imm)
    }};
    ($asm:expr, ld $rd:expr, $rs1:expr, $imm:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_ld(Kind::LD, $rd, $rs1, $imm)
    }};
    ($asm:expr, lw $rd:expr, $rs1:expr, $imm:expr) => {{
        use $crate::expand::asm_support::SourceKind;
        $asm.emit_ld(SourceKind::LW, $rd, $rs1, $imm)
    }};
    ($asm:expr, sd $base:expr, $src:expr, $imm:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_s(Kind::SD, $base, $src, $imm)
    }};
    ($asm:expr, lui $rd:expr, $imm:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_u(Kind::LUI, $rd, $imm)
    }};
    ($asm:expr, advice $rd:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_j(Kind::VIRTUAL_ADVICE, $rd, 0)
    }};
    ($asm:expr, assert_eq $rs1:expr, $rs2:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_b(Kind::VirtualAssertEQ, $rs1, $rs2, 0)
    }};
    ($asm:expr, assert_lte $rs1:expr, $rs2:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_b(Kind::VirtualAssertLTE, $rs1, $rs2, 0)
    }};
    ($asm:expr, assert_halfword_alignment $rs1:expr, $imm:expr) => {{
        use $crate::expand::asm_support::SourceKind;
        $asm.emit_address(SourceKind::VirtualAssertHalfwordAlignment, $rs1, $imm)
    }};
    ($asm:expr, assert_word_alignment $rs1:expr, $imm:expr) => {{
        use $crate::expand::asm_support::SourceKind;
        $asm.emit_address(SourceKind::VirtualAssertWordAlignment, $rs1, $imm)
    }};
    ($asm:expr, assert_valid_div0 $divisor:expr, $quotient:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_b(Kind::VirtualAssertValidDiv0, $divisor, $quotient, 0)
    }};
    ($asm:expr, assert_mul_u_no_overflow $lhs:expr, $rhs:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_b(Kind::VirtualAssertMulUNoOverflow, $lhs, $rhs, 0)
    }};
    ($asm:expr, assert_valid_unsigned_remainder $remainder:expr, $divisor:expr) => {{
        use $crate::expand::asm_support::Kind;
        $asm.emit_b(
            Kind::VirtualAssertValidUnsignedRemainder,
            $remainder,
            $divisor,
            0,
        )
    }};
    ($asm:expr, rotl64 $rd:expr, $rs1:expr, $amount:expr) => {
        let _ = $asm.rotl64($crate::expand::Value::Reg($rs1), $amount, $rd);
    };
}
