//! BN254 Fr coprocessor instruction-word encoding.
//!
//! R-type layout throughout:
//!   bits [31:25] = funct7 (7 bits)
//!   bits [24:20] = rs2 (5 bits)
//!   bits [19:15] = rs1 (5 bits)
//!   bits [14:12] = funct3 (3 bits)
//!   bits [11:7]  = rd (5 bits)
//!   bits [6:0]   = opcode (7 bits)
//!
//! The 9 BN254 Fr instructions split across two funct7 sub-families
//! because funct3 is only 3 bits wide — the SLL family lives under
//! `BN254_FR_SLL_FUNCT7 = 0x41`, all other FR ops under
//! `BN254_FR_FUNCT7 = 0x40`. See
//! `specs/bn254-fr-coprocessor.md` §ISA and the tracer decode dispatch
//! in `tracer/src/instruction/mod.rs`.

/// Custom-0 opcode used by every BN254 Fr coprocessor instruction.
pub const FIELD_OP_OPCODE: u32 = 0x0B;

/// funct7 for the 2-input FReg↔FReg ops (FMUL, FADD, FSUB, FINV) plus the
/// 1-input bridge ops that read an integer register and write an FReg
/// (FieldAssertEq, FieldMov). Saturates the 8-slot funct3 space.
pub const BN254_FR_FUNCT7: u32 = 0x40;

/// funct7 for the 3 FieldSLL* shift-left bridge ops. funct3 re-used
/// (0x00/0x01/0x02) because the 0x40 family has no room.
pub const BN254_FR_SLL_FUNCT7: u32 = 0x41;

// --- funct3 selectors under BN254_FR_FUNCT7 = 0x40 ---

pub const FUNCT3_FMUL: u32 = 0x02;
pub const FUNCT3_FADD: u32 = 0x03;
pub const FUNCT3_FINV: u32 = 0x04;
pub const FUNCT3_FSUB: u32 = 0x05;
pub const FUNCT3_FIELD_ASSERT_EQ: u32 = 0x06;
pub const FUNCT3_FIELD_MOV: u32 = 0x07;

// --- funct3 selectors under BN254_FR_SLL_FUNCT7 = 0x41 ---

pub const FUNCT3_FIELD_SLL64: u32 = 0x00;
pub const FUNCT3_FIELD_SLL128: u32 = 0x01;
pub const FUNCT3_FIELD_SLL192: u32 = 0x02;

/// Assemble a raw 32-bit R-type instruction word.
#[inline]
pub const fn encode_r(funct7: u32, rs2: u32, rs1: u32, funct3: u32, rd: u32, opcode: u32) -> u32 {
    ((funct7 & 0x7F) << 25)
        | ((rs2 & 0x1F) << 20)
        | ((rs1 & 0x1F) << 15)
        | ((funct3 & 0x7) << 12)
        | ((rd & 0x1F) << 7)
        | (opcode & 0x7F)
}

/// `FMUL frd, frs1, frs2`: `FReg[frd] = FReg[frs1] · FReg[frs2]` (mod p).
#[inline]
pub const fn encode_fmul(frd: u32, frs1: u32, frs2: u32) -> u32 {
    encode_r(BN254_FR_FUNCT7, frs2, frs1, FUNCT3_FMUL, frd, FIELD_OP_OPCODE)
}

/// `FADD frd, frs1, frs2`: `FReg[frd] = FReg[frs1] + FReg[frs2]` (mod p).
#[inline]
pub const fn encode_fadd(frd: u32, frs1: u32, frs2: u32) -> u32 {
    encode_r(BN254_FR_FUNCT7, frs2, frs1, FUNCT3_FADD, frd, FIELD_OP_OPCODE)
}

/// `FSUB frd, frs1, frs2`: `FReg[frd] = FReg[frs1] − FReg[frs2]` (mod p).
#[inline]
pub const fn encode_fsub(frd: u32, frs1: u32, frs2: u32) -> u32 {
    encode_r(BN254_FR_FUNCT7, frs2, frs1, FUNCT3_FSUB, frd, FIELD_OP_OPCODE)
}

/// `FINV frd, frs1`: `FReg[frd] = FReg[frs1]⁻¹` (mod p; `0⁻¹` is
/// guest-undefined).
#[inline]
pub const fn encode_finv(frd: u32, frs1: u32) -> u32 {
    // rs2 ignored for FINV.
    encode_r(BN254_FR_FUNCT7, 0, frs1, FUNCT3_FINV, frd, FIELD_OP_OPCODE)
}

/// `FieldAssertEq frs1, frs2`: `assert FReg[frs1] == FReg[frs2]`; no
/// write. Tracer emits a no-op event at slot `frs1` to preserve the
/// "single FR access per cycle" invariant.
#[inline]
pub const fn encode_field_assert_eq(frs1: u32, frs2: u32) -> u32 {
    // rd ignored for FieldAssertEq.
    encode_r(
        BN254_FR_FUNCT7,
        frs2,
        frs1,
        FUNCT3_FIELD_ASSERT_EQ,
        0,
        FIELD_OP_OPCODE,
    )
}

/// `FieldMov frd, rs1`: `FReg[frd] = XReg[rs1] as Fr` (integer embeds as
/// low limb).
#[inline]
pub const fn encode_field_mov(frd: u32, rs1: u32) -> u32 {
    // rs2 ignored.
    encode_r(
        BN254_FR_FUNCT7,
        0,
        rs1,
        FUNCT3_FIELD_MOV,
        frd,
        FIELD_OP_OPCODE,
    )
}

/// `FieldSLL64 frd, rs1`: `FReg[frd] = XReg[rs1] · 2⁶⁴`.
#[inline]
pub const fn encode_field_sll64(frd: u32, rs1: u32) -> u32 {
    encode_r(
        BN254_FR_SLL_FUNCT7,
        0,
        rs1,
        FUNCT3_FIELD_SLL64,
        frd,
        FIELD_OP_OPCODE,
    )
}

/// `FieldSLL128 frd, rs1`: `FReg[frd] = XReg[rs1] · 2¹²⁸`.
#[inline]
pub const fn encode_field_sll128(frd: u32, rs1: u32) -> u32 {
    encode_r(
        BN254_FR_SLL_FUNCT7,
        0,
        rs1,
        FUNCT3_FIELD_SLL128,
        frd,
        FIELD_OP_OPCODE,
    )
}

/// `FieldSLL192 frd, rs1`: `FReg[frd] = XReg[rs1] · 2¹⁹²` (caller must
/// ensure the result is canonical < p).
#[inline]
pub const fn encode_field_sll192(frd: u32, rs1: u32) -> u32 {
    encode_r(
        BN254_FR_SLL_FUNCT7,
        0,
        rs1,
        FUNCT3_FIELD_SLL192,
        frd,
        FIELD_OP_OPCODE,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mask + match values in each tracer instruction module must
    /// align with the encoders here — this test pins both ends of the
    /// contract.
    #[test]
    fn encoder_round_trips_through_field_layout() {
        // FMUL frd=3, frs1=1, frs2=2
        let word = encode_fmul(3, 1, 2);
        assert_eq!((word >> 25) & 0x7F, BN254_FR_FUNCT7);
        assert_eq!((word >> 20) & 0x1F, 2);
        assert_eq!((word >> 15) & 0x1F, 1);
        assert_eq!((word >> 12) & 0x7, FUNCT3_FMUL);
        assert_eq!((word >> 7) & 0x1F, 3);
        assert_eq!(word & 0x7F, FIELD_OP_OPCODE);
    }

    #[test]
    fn sll_family_lives_under_funct7_0x41() {
        let w64 = encode_field_sll64(5, 10);
        let w128 = encode_field_sll128(5, 10);
        let w192 = encode_field_sll192(5, 10);
        assert_eq!((w64 >> 25) & 0x7F, BN254_FR_SLL_FUNCT7);
        assert_eq!((w128 >> 25) & 0x7F, BN254_FR_SLL_FUNCT7);
        assert_eq!((w192 >> 25) & 0x7F, BN254_FR_SLL_FUNCT7);
        assert_eq!((w64 >> 12) & 0x7, FUNCT3_FIELD_SLL64);
        assert_eq!((w128 >> 12) & 0x7, FUNCT3_FIELD_SLL128);
        assert_eq!((w192 >> 12) & 0x7, FUNCT3_FIELD_SLL192);
    }

    #[test]
    fn mov_and_assert_eq_share_main_funct7() {
        let mov = encode_field_mov(3, 7);
        let aeq = encode_field_assert_eq(1, 2);
        assert_eq!((mov >> 25) & 0x7F, BN254_FR_FUNCT7);
        assert_eq!((aeq >> 25) & 0x7F, BN254_FR_FUNCT7);
        assert_eq!((mov >> 12) & 0x7, FUNCT3_FIELD_MOV);
        assert_eq!((aeq >> 12) & 0x7, FUNCT3_FIELD_ASSERT_EQ);
    }

    #[test]
    fn field_regs_use_low_4_bits() {
        // frd=15 (the largest valid FR slot). Bit 4 of the 5-bit register
        // field must round-trip faithfully — the tracer masks it back to
        // `& 0xF` at access time, but the instruction word itself carries
        // the raw 5-bit value (so `encode` does NOT mask).
        let w = encode_fadd(15, 0, 1);
        assert_eq!((w >> 7) & 0x1F, 15);
    }
}
