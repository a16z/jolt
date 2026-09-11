/// Expand a 16-bit RVC instruction into the RV64 32-bit instruction word that
/// Jolt's decoder expects.
///
/// This helper deliberately has no RV32 mode. Program image decoding rejects
/// ELF32/RV32 before compressed-instruction normalization.
#[inline]
pub fn uncompress_rv64_instruction(halfword: u16) -> u32 {
    let halfword = u32::from(halfword);
    let op = halfword & 0x3;
    let funct3 = (halfword >> 13) & 0x7;

    match op {
        0 => match funct3 {
            0 => {
                let rd = (halfword >> 2) & 0x7;
                let nzuimm = ((halfword >> 7) & 0x30)
                    | ((halfword >> 1) & 0x3c0)
                    | ((halfword >> 4) & 0x4)
                    | ((halfword >> 2) & 0x8);
                if nzuimm != 0 {
                    return (nzuimm << 20) | (2 << 15) | ((rd + 8) << 7) | 0x13;
                }
            }
            1 => {
                let rd = (halfword >> 2) & 0x7;
                let rs1 = (halfword >> 7) & 0x7;
                let offset = ((halfword >> 7) & 0x38) | ((halfword << 1) & 0xc0);
                return (offset << 20) | ((rs1 + 8) << 15) | (3 << 12) | ((rd + 8) << 7) | 0x7;
            }
            2 => {
                let rs1 = (halfword >> 7) & 0x7;
                let rd = (halfword >> 2) & 0x7;
                let offset =
                    ((halfword >> 7) & 0x38) | ((halfword >> 4) & 0x4) | ((halfword << 1) & 0x40);
                return (offset << 20) | ((rs1 + 8) << 15) | (2 << 12) | ((rd + 8) << 7) | 0x3;
            }
            3 => {
                let rs1 = (halfword >> 7) & 0x7;
                let rd = (halfword >> 2) & 0x7;
                let offset = ((halfword >> 7) & 0x38) | ((halfword << 1) & 0xc0);
                return (offset << 20) | ((rs1 + 8) << 15) | (3 << 12) | ((rd + 8) << 7) | 0x3;
            }
            5 => {
                let rs1 = (halfword >> 7) & 0x7;
                let rs2 = (halfword >> 2) & 0x7;
                let offset = ((halfword >> 7) & 0x38) | ((halfword << 1) & 0xc0);
                let imm11_5 = (offset >> 5) & 0x7f;
                let imm4_0 = offset & 0x1f;
                return (imm11_5 << 25)
                    | ((rs2 + 8) << 20)
                    | ((rs1 + 8) << 15)
                    | (3 << 12)
                    | (imm4_0 << 7)
                    | 0x27;
            }
            6 => {
                let rs1 = (halfword >> 7) & 0x7;
                let rs2 = (halfword >> 2) & 0x7;
                let offset =
                    ((halfword >> 7) & 0x38) | ((halfword << 1) & 0x40) | ((halfword >> 4) & 0x4);
                let imm11_5 = (offset >> 5) & 0x7f;
                let imm4_0 = offset & 0x1f;
                return (imm11_5 << 25)
                    | ((rs2 + 8) << 20)
                    | ((rs1 + 8) << 15)
                    | (2 << 12)
                    | (imm4_0 << 7)
                    | 0x23;
            }
            7 => {
                let rs1 = (halfword >> 7) & 0x7;
                let rs2 = (halfword >> 2) & 0x7;
                let offset = ((halfword >> 7) & 0x38) | ((halfword << 1) & 0xc0);
                let imm11_5 = (offset >> 5) & 0x7f;
                let imm4_0 = offset & 0x1f;
                return (imm11_5 << 25)
                    | ((rs2 + 8) << 20)
                    | ((rs1 + 8) << 15)
                    | (3 << 12)
                    | (imm4_0 << 7)
                    | 0x23;
            }
            _ => {}
        },
        1 => match funct3 {
            0 => {
                let r = (halfword >> 7) & 0x1f;
                let imm = sign_bit_mask(halfword, 0x1000, 0xffff_ffc0)
                    | ((halfword >> 7) & 0x20)
                    | ((halfword >> 2) & 0x1f);
                if r == 0 || imm == 0 {
                    return 0x13;
                }
                return (imm << 20) | (r << 15) | (r << 7) | 0x13;
            }
            1 => {
                let r = (halfword >> 7) & 0x1f;
                let imm = sign_bit_mask(halfword, 0x1000, 0xffff_ffc0)
                    | ((halfword >> 7) & 0x20)
                    | ((halfword >> 2) & 0x1f);
                if r != 0 {
                    return (imm << 20) | (r << 15) | (r << 7) | 0x1b;
                }
            }
            2 => {
                let r = (halfword >> 7) & 0x1f;
                let imm = sign_bit_mask(halfword, 0x1000, 0xffff_ffc0)
                    | ((halfword >> 7) & 0x20)
                    | ((halfword >> 2) & 0x1f);
                if r != 0 {
                    return (imm << 20) | (r << 7) | 0x13;
                }
                return 0x13;
            }
            3 => {
                let r = (halfword >> 7) & 0x1f;
                if r == 2 {
                    let imm = sign_bit_mask(halfword, 0x1000, 0xffff_fc00)
                        | ((halfword >> 3) & 0x200)
                        | ((halfword >> 2) & 0x10)
                        | ((halfword << 1) & 0x40)
                        | ((halfword << 4) & 0x180)
                        | ((halfword << 3) & 0x20);
                    if imm != 0 {
                        return (imm << 20) | (r << 15) | (r << 7) | 0x13;
                    }
                }
                if r != 0 && r != 2 {
                    let nzimm = sign_bit_mask(halfword, 0x1000, 0xfffc_0000)
                        | ((halfword << 5) & 0x20000)
                        | ((halfword << 10) & 0x1f000);
                    if nzimm != 0 {
                        return nzimm | (r << 7) | 0x37;
                    }
                }
                if r == 0 {
                    return 0x13;
                }
            }
            4 => {
                let funct2 = (halfword >> 10) & 0x3;
                match funct2 {
                    0 => {
                        let shamt = ((halfword >> 7) & 0x20) | ((halfword >> 2) & 0x1f);
                        let rs1 = (halfword >> 7) & 0x7;
                        return (shamt << 20)
                            | ((rs1 + 8) << 15)
                            | (5 << 12)
                            | ((rs1 + 8) << 7)
                            | 0x13;
                    }
                    1 => {
                        let shamt = ((halfword >> 7) & 0x20) | ((halfword >> 2) & 0x1f);
                        let rs1 = (halfword >> 7) & 0x7;
                        return (0x20 << 25)
                            | (shamt << 20)
                            | ((rs1 + 8) << 15)
                            | (5 << 12)
                            | ((rs1 + 8) << 7)
                            | 0x13;
                    }
                    2 => {
                        let r = (halfword >> 7) & 0x7;
                        let imm = sign_bit_mask(halfword, 0x1000, 0xffff_ffc0)
                            | ((halfword >> 7) & 0x20)
                            | ((halfword >> 2) & 0x1f);
                        return (imm << 20) | ((r + 8) << 15) | (7 << 12) | ((r + 8) << 7) | 0x13;
                    }
                    3 => {
                        let funct1 = (halfword >> 12) & 1;
                        let funct2_2 = (halfword >> 5) & 0x3;
                        let rs1 = (halfword >> 7) & 0x7;
                        let rs2 = (halfword >> 2) & 0x7;
                        match (funct1, funct2_2) {
                            (0, 0) => {
                                return (0x20 << 25)
                                    | ((rs2 + 8) << 20)
                                    | ((rs1 + 8) << 15)
                                    | ((rs1 + 8) << 7)
                                    | 0x33;
                            }
                            (0, 1) => {
                                return ((rs2 + 8) << 20)
                                    | ((rs1 + 8) << 15)
                                    | (4 << 12)
                                    | ((rs1 + 8) << 7)
                                    | 0x33;
                            }
                            (0, 2) => {
                                return ((rs2 + 8) << 20)
                                    | ((rs1 + 8) << 15)
                                    | (6 << 12)
                                    | ((rs1 + 8) << 7)
                                    | 0x33;
                            }
                            (0, 3) => {
                                return ((rs2 + 8) << 20)
                                    | ((rs1 + 8) << 15)
                                    | (7 << 12)
                                    | ((rs1 + 8) << 7)
                                    | 0x33;
                            }
                            (1, 0) => {
                                return (0x20 << 25)
                                    | ((rs2 + 8) << 20)
                                    | ((rs1 + 8) << 15)
                                    | ((rs1 + 8) << 7)
                                    | 0x3b;
                            }
                            (1, 1) => {
                                return ((rs2 + 8) << 20)
                                    | ((rs1 + 8) << 15)
                                    | ((rs1 + 8) << 7)
                                    | 0x3b;
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
            5 => {
                let offset = j_offset(halfword);
                let imm = ((offset >> 1) & 0x80000)
                    | ((offset << 8) & 0x7fe00)
                    | ((offset >> 3) & 0x100)
                    | ((offset >> 12) & 0xff);
                return (imm << 12) | 0x6f;
            }
            6 => {
                let r = (halfword >> 7) & 0x7;
                let (imm2, imm1) = branch_immediates(halfword);
                return (imm2 << 25) | ((r + 8) << 20) | (imm1 << 7) | 0x63;
            }
            7 => {
                let r = (halfword >> 7) & 0x7;
                let (imm2, imm1) = branch_immediates(halfword);
                return (imm2 << 25) | ((r + 8) << 20) | (1 << 12) | (imm1 << 7) | 0x63;
            }
            _ => {}
        },
        2 => match funct3 {
            0 => {
                let r = (halfword >> 7) & 0x1f;
                let shamt = ((halfword >> 7) & 0x20) | ((halfword >> 2) & 0x1f);
                return (shamt << 20) | (r << 15) | (1 << 12) | (r << 7) | 0x13;
            }
            1 => {
                let rd = (halfword >> 7) & 0x1f;
                let offset =
                    ((halfword >> 7) & 0x20) | ((halfword >> 2) & 0x18) | ((halfword << 4) & 0x1c0);
                if rd != 0 {
                    return (offset << 20) | (2 << 15) | (3 << 12) | (rd << 7) | 0x7;
                }
            }
            2 => {
                let r = (halfword >> 7) & 0x1f;
                let offset =
                    ((halfword >> 7) & 0x20) | ((halfword >> 2) & 0x1c) | ((halfword << 4) & 0xc0);
                if r != 0 {
                    return (offset << 20) | (2 << 15) | (2 << 12) | (r << 7) | 0x3;
                }
            }
            3 => {
                let rd = (halfword >> 7) & 0x1f;
                let offset =
                    ((halfword >> 7) & 0x20) | ((halfword >> 2) & 0x18) | ((halfword << 4) & 0x1c0);
                if rd != 0 {
                    return (offset << 20) | (2 << 15) | (3 << 12) | (rd << 7) | 0x3;
                }
            }
            4 => {
                let funct1 = (halfword >> 12) & 1;
                let rs1 = (halfword >> 7) & 0x1f;
                let rs2 = (halfword >> 2) & 0x1f;
                match (funct1, rs1, rs2) {
                    (0, 0, 0) => {}
                    (0, r, 0) if r != 0 => return (rs1 << 15) | 0x67,
                    (0, 0, r2) if r2 != 0 => return 0x13,
                    (0, rd, rs2) => return (rs2 << 20) | (rd << 7) | 0x33,
                    (1, 0, 0) => return 0x0010_0073,
                    (1, rs1, 0) if rs1 != 0 => return (rs1 << 15) | (1 << 7) | 0x67,
                    (1, 0, rs2) if rs2 != 0 => return 0x13,
                    (1, rs1, rs2) => return (rs2 << 20) | (rs1 << 15) | (rs1 << 7) | 0x33,
                    _ => {}
                }
            }
            5 => {
                let rs2 = (halfword >> 2) & 0x1f;
                let offset = ((halfword >> 7) & 0x38) | ((halfword >> 1) & 0x1c0);
                let imm11_5 = (offset >> 5) & 0x3f;
                let imm4_0 = offset & 0x1f;
                return (imm11_5 << 25)
                    | (rs2 << 20)
                    | (2 << 15)
                    | (3 << 12)
                    | (imm4_0 << 7)
                    | 0x27;
            }
            6 => {
                let rs2 = (halfword >> 2) & 0x1f;
                let offset = ((halfword >> 7) & 0x3c) | ((halfword >> 1) & 0xc0);
                let imm11_5 = (offset >> 5) & 0x3f;
                let imm4_0 = offset & 0x1f;
                return (imm11_5 << 25)
                    | (rs2 << 20)
                    | (2 << 15)
                    | (2 << 12)
                    | (imm4_0 << 7)
                    | 0x23;
            }
            7 => {
                let rs2 = (halfword >> 2) & 0x1f;
                let offset = ((halfword >> 7) & 0x38) | ((halfword >> 1) & 0x1c0);
                let imm11_5 = (offset >> 5) & 0x3f;
                let imm4_0 = offset & 0x1f;
                return (imm11_5 << 25)
                    | (rs2 << 20)
                    | (2 << 15)
                    | (3 << 12)
                    | (imm4_0 << 7)
                    | 0x23;
            }
            _ => {}
        },
        _ => {}
    }

    0xffff_ffff
}

#[inline]
fn sign_bit_mask(value: u32, bit: u32, mask: u32) -> u32 {
    if value & bit == bit {
        mask
    } else {
        0
    }
}

#[inline]
fn j_offset(halfword: u32) -> u32 {
    sign_bit_mask(halfword, 0x1000, 0xffff_f000)
        | ((halfword >> 1) & 0x800)
        | ((halfword >> 7) & 0x10)
        | ((halfword >> 1) & 0x300)
        | ((halfword << 2) & 0x400)
        | ((halfword >> 1) & 0x40)
        | ((halfword << 1) & 0x80)
        | ((halfword >> 2) & 0xe)
        | ((halfword << 3) & 0x20)
}

#[inline]
fn branch_immediates(halfword: u32) -> (u32, u32) {
    let offset = sign_bit_mask(halfword, 0x1000, 0xffff_fe00)
        | ((halfword >> 4) & 0x100)
        | ((halfword >> 7) & 0x18)
        | ((halfword << 1) & 0xc0)
        | ((halfword >> 2) & 0x6)
        | ((halfword << 3) & 0x20);
    let imm2 = ((offset >> 6) & 0x40) | ((offset >> 5) & 0x3f);
    let imm1 = (offset & 0x1e) | ((offset >> 11) & 0x1);
    (imm2, imm1)
}

#[cfg(test)]
mod tests {
    use super::uncompress_rv64_instruction;

    fn expand(halfword: u32) -> u32 {
        assert!(
            halfword <= 0xffff,
            "test assembler produced a non-16-bit encoding: {halfword:#x}"
        );
        uncompress_rv64_instruction(halfword as u16)
    }

    fn bit(value: u32, index: u32) -> u32 {
        (value >> index) & 1
    }

    fn bits(value: u32, hi: u32, lo: u32) -> u32 {
        (value >> lo) & ((1 << (hi - lo + 1)) - 1)
    }

    // Expected-side assemblers, transcribed from the RV64I base instruction
    // formats (unprivileged spec §2.3), independent of the decoder above.

    fn i_type(imm: i32, rs1: u32, funct3: u32, rd: u32, opcode: u32) -> u32 {
        ((imm as u32 & 0xfff) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
    }

    fn s_type(imm: i32, rs2: u32, rs1: u32, funct3: u32, opcode: u32) -> u32 {
        let imm = imm as u32 & 0xfff;
        (bits(imm, 11, 5) << 25)
            | (rs2 << 20)
            | (rs1 << 15)
            | (funct3 << 12)
            | (bits(imm, 4, 0) << 7)
            | opcode
    }

    fn b_type(offset: i32, rs2: u32, rs1: u32, funct3: u32) -> u32 {
        let imm = offset as u32;
        (bit(imm, 12) << 31)
            | (bits(imm, 10, 5) << 25)
            | (rs2 << 20)
            | (rs1 << 15)
            | (funct3 << 12)
            | (bits(imm, 4, 1) << 8)
            | (bit(imm, 11) << 7)
            | 0x63
    }

    fn j_type(offset: i32, rd: u32) -> u32 {
        let imm = offset as u32;
        (bit(imm, 20) << 31)
            | (bits(imm, 10, 1) << 21)
            | (bit(imm, 11) << 20)
            | (bits(imm, 19, 12) << 12)
            | (rd << 7)
            | 0x6f
    }

    fn r_type(funct7: u32, rs2: u32, rs1: u32, funct3: u32, rd: u32, opcode: u32) -> u32 {
        (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
    }

    fn lui(imm17_12: i32, rd: u32) -> u32 {
        ((imm17_12 << 12) as u32) | (rd << 7) | 0x37
    }

    // Compressed-side assemblers, transcribed from the RVC format tables
    // (unprivileged spec §16.5). Immediate arguments are the *byte* offsets or
    // signed values; the helpers scatter them into instruction bits.

    fn c_addi4spn(rdp: u32, nzuimm: u32) -> u32 {
        (bits(nzuimm, 5, 4) << 11)
            | (bits(nzuimm, 9, 6) << 7)
            | (bit(nzuimm, 2) << 6)
            | (bit(nzuimm, 3) << 5)
            | (rdp << 2)
    }

    fn c_lw(rdp: u32, rs1p: u32, offset: u32) -> u32 {
        (0b010 << 13)
            | (bits(offset, 5, 3) << 10)
            | (rs1p << 7)
            | (bit(offset, 2) << 6)
            | (bit(offset, 6) << 5)
            | (rdp << 2)
    }

    fn c_ld(rdp: u32, rs1p: u32, offset: u32) -> u32 {
        (0b011 << 13)
            | (bits(offset, 5, 3) << 10)
            | (rs1p << 7)
            | (bits(offset, 7, 6) << 5)
            | (rdp << 2)
    }

    fn c_fld(rdp: u32, rs1p: u32, offset: u32) -> u32 {
        (0b001 << 13)
            | (bits(offset, 5, 3) << 10)
            | (rs1p << 7)
            | (bits(offset, 7, 6) << 5)
            | (rdp << 2)
    }

    fn c_sw(rs2p: u32, rs1p: u32, offset: u32) -> u32 {
        (0b110 << 13)
            | (bits(offset, 5, 3) << 10)
            | (rs1p << 7)
            | (bit(offset, 2) << 6)
            | (bit(offset, 6) << 5)
            | (rs2p << 2)
    }

    fn c_sd(rs2p: u32, rs1p: u32, offset: u32) -> u32 {
        (0b111 << 13)
            | (bits(offset, 5, 3) << 10)
            | (rs1p << 7)
            | (bits(offset, 7, 6) << 5)
            | (rs2p << 2)
    }

    fn c_fsd(rs2p: u32, rs1p: u32, offset: u32) -> u32 {
        (0b101 << 13)
            | (bits(offset, 5, 3) << 10)
            | (rs1p << 7)
            | (bits(offset, 7, 6) << 5)
            | (rs2p << 2)
    }

    /// CI format (quadrant 1): C.ADDI (funct3=000), C.ADDIW (001), C.LI (010).
    /// Also encodes C.LUI (011) where `imm` is the signed 6-bit value of
    /// nzimm[17:12].
    fn ci(funct3: u32, rd: u32, imm: i32) -> u32 {
        let imm = imm as u32;
        (funct3 << 13) | (bit(imm, 5) << 12) | (rd << 7) | (bits(imm, 4, 0) << 2) | 0b01
    }

    fn c_addi16sp(imm: i32) -> u32 {
        let imm = imm as u32;
        (0b011 << 13)
            | (bit(imm, 9) << 12)
            | (2 << 7)
            | (bit(imm, 4) << 6)
            | (bit(imm, 6) << 5)
            | (bits(imm, 8, 7) << 3)
            | (bit(imm, 5) << 2)
            | 0b01
    }

    /// CB format shifts/andi: funct2 = 00 (C.SRLI), 01 (C.SRAI), 10 (C.ANDI).
    fn cb_alu(funct2: u32, rdp: u32, imm: i32) -> u32 {
        let imm = imm as u32;
        (0b100 << 13)
            | (bit(imm, 5) << 12)
            | (funct2 << 10)
            | (rdp << 7)
            | (bits(imm, 4, 0) << 2)
            | 0b01
    }

    fn ca(funct1: u32, funct2: u32, rdp: u32, rs2p: u32) -> u32 {
        (0b100 << 13)
            | (funct1 << 12)
            | (0b11 << 10)
            | (rdp << 7)
            | (funct2 << 5)
            | (rs2p << 2)
            | 0b01
    }

    fn c_j(offset: i32) -> u32 {
        let o = offset as u32;
        (0b101 << 13)
            | (bit(o, 11) << 12)
            | (bit(o, 4) << 11)
            | (bits(o, 9, 8) << 9)
            | (bit(o, 10) << 8)
            | (bit(o, 6) << 7)
            | (bit(o, 7) << 6)
            | (bits(o, 3, 1) << 3)
            | (bit(o, 5) << 2)
            | 0b01
    }

    /// CB format branches: funct3 = 110 (C.BEQZ), 111 (C.BNEZ).
    fn cb_branch(funct3: u32, rs1p: u32, offset: i32) -> u32 {
        let o = offset as u32;
        (funct3 << 13)
            | (bit(o, 8) << 12)
            | (bits(o, 4, 3) << 10)
            | (rs1p << 7)
            | (bits(o, 7, 6) << 5)
            | (bits(o, 2, 1) << 3)
            | (bit(o, 5) << 2)
            | 0b01
    }

    fn c_slli(rd: u32, shamt: u32) -> u32 {
        (bit(shamt, 5) << 12) | (rd << 7) | (bits(shamt, 4, 0) << 2) | 0b10
    }

    fn c_lwsp(rd: u32, offset: u32) -> u32 {
        (0b010 << 13)
            | (bit(offset, 5) << 12)
            | (rd << 7)
            | (bits(offset, 4, 2) << 4)
            | (bits(offset, 7, 6) << 2)
            | 0b10
    }

    fn c_ldsp(rd: u32, offset: u32) -> u32 {
        (0b011 << 13)
            | (bit(offset, 5) << 12)
            | (rd << 7)
            | (bits(offset, 4, 3) << 5)
            | (bits(offset, 8, 6) << 2)
            | 0b10
    }

    fn c_fldsp(rd: u32, offset: u32) -> u32 {
        (0b001 << 13)
            | (bit(offset, 5) << 12)
            | (rd << 7)
            | (bits(offset, 4, 3) << 5)
            | (bits(offset, 8, 6) << 2)
            | 0b10
    }

    /// CR format: funct1 = 0 (C.JR/C.MV), 1 (C.EBREAK/C.JALR/C.ADD).
    fn cr(funct1: u32, rs1: u32, rs2: u32) -> u32 {
        (0b100 << 13) | (funct1 << 12) | (rs1 << 7) | (rs2 << 2) | 0b10
    }

    fn c_swsp(rs2: u32, offset: u32) -> u32 {
        (0b110 << 13) | (bits(offset, 5, 2) << 9) | (bits(offset, 7, 6) << 7) | (rs2 << 2) | 0b10
    }

    fn c_sdsp(rs2: u32, offset: u32) -> u32 {
        (0b111 << 13) | (bits(offset, 5, 3) << 10) | (bits(offset, 8, 6) << 7) | (rs2 << 2) | 0b10
    }

    fn c_fsdsp(rs2: u32, offset: u32) -> u32 {
        (0b101 << 13) | (bits(offset, 5, 3) << 10) | (bits(offset, 8, 6) << 7) | (rs2 << 2) | 0b10
    }

    #[test]
    fn uncompresses_nop() {
        assert_eq!(uncompress_rv64_instruction(0x0001), 0x13);
    }

    #[test]
    fn uncompresses_c_slli_x0() {
        assert_eq!(uncompress_rv64_instruction(0x107a), 0x03e0_1013);
    }

    /// Raw halfword/word pairs taken from binutils objdump output; no test
    /// helper is involved on either side.
    #[test]
    fn matches_known_toolchain_encodings() {
        assert_eq!(uncompress_rv64_instruction(0x1141), 0xff01_0113); // c.addi sp,-16
        assert_eq!(uncompress_rv64_instruction(0x4501), 0x0000_0513); // c.li a0,0
        assert_eq!(uncompress_rv64_instruction(0x8082), 0x0000_8067); // c.jr ra (ret)
        assert_eq!(uncompress_rv64_instruction(0x9002), 0x0010_0073); // c.ebreak
        assert_eq!(uncompress_rv64_instruction(0x852e), 0x00b0_0533); // c.mv a0,a1
        assert_eq!(uncompress_rv64_instruction(0xe022), 0x0081_3023); // c.sdsp s0,0(sp)
    }

    /// Guards the test-side compressed assemblers against the same toolchain
    /// halfwords, so an encoder bug cannot silently cancel a decoder bug.
    #[test]
    fn test_assemblers_match_known_toolchain_halfwords() {
        assert_eq!(ci(0b000, 2, -16), 0x1141); // c.addi sp,-16
        assert_eq!(ci(0b010, 10, 0), 0x4501); // c.li a0,0
        assert_eq!(cr(0, 1, 0), 0x8082); // c.jr ra
        assert_eq!(cr(1, 0, 0), 0x9002); // c.ebreak
        assert_eq!(cr(0, 10, 11), 0x852e); // c.mv a0,a1
        assert_eq!(c_sdsp(8, 0), 0xe022); // c.sdsp s0,0(sp)
        assert_eq!(c_slli(0, 62), 0x107a); // c.slli x0,62 (existing anchor)
    }

    #[test]
    fn c_addi4spn_expands_to_addi_from_sp() {
        assert_eq!(expand(c_addi4spn(0, 4)), i_type(4, 2, 0b000, 8, 0x13)); // minimum imm
        assert_eq!(
            expand(c_addi4spn(7, 1020)),
            i_type(1020, 2, 0b000, 15, 0x13)
        ); // maximum imm
           // 340 = bits 2,4,6,8 — one bit in each scattered source field
        assert_eq!(expand(c_addi4spn(3, 340)), i_type(340, 2, 0b000, 11, 0x13));
        for b in 2..=9 {
            let imm = 1u32 << b;
            assert_eq!(
                expand(c_addi4spn(1, imm)),
                i_type(imm as i32, 2, 0b000, 9, 0x13)
            );
        }
    }

    #[test]
    fn c_lw_expands_with_word_scaled_offset() {
        assert_eq!(expand(c_lw(7, 0, 0)), i_type(0, 8, 0b010, 15, 0x03));
        assert_eq!(expand(c_lw(0, 7, 124)), i_type(124, 15, 0b010, 8, 0x03)); // max offset
        assert_eq!(expand(c_lw(2, 1, 84)), i_type(84, 9, 0b010, 10, 0x03)); // bits 2,4,6
        for b in 2..=6 {
            let imm = 1u32 << b;
            assert_eq!(
                expand(c_lw(2, 1, imm)),
                i_type(imm as i32, 9, 0b010, 10, 0x03)
            );
        }
    }

    #[test]
    fn c_ld_expands_with_doubleword_scaled_offset() {
        assert_eq!(expand(c_ld(1, 2, 8)), i_type(8, 10, 0b011, 9, 0x03));
        assert_eq!(expand(c_ld(6, 5, 248)), i_type(248, 13, 0b011, 14, 0x03)); // max offset
        assert_eq!(expand(c_ld(4, 3, 168)), i_type(168, 11, 0b011, 12, 0x03)); // bits 3,5,7
        for b in 3..=7 {
            let imm = 1u32 << b;
            assert_eq!(
                expand(c_ld(4, 3, imm)),
                i_type(imm as i32, 11, 0b011, 12, 0x03)
            );
        }
    }

    #[test]
    fn c_fld_expands_to_fp_load() {
        for b in 3..=7 {
            let imm = 1u32 << b;
            assert_eq!(
                expand(c_fld(1, 2, imm)),
                i_type(imm as i32, 10, 0b011, 9, 0x07)
            );
        }
    }

    #[test]
    fn c_sw_expands_with_word_scaled_offset() {
        assert_eq!(expand(c_sw(7, 0, 0)), s_type(0, 15, 8, 0b010, 0x23));
        assert_eq!(expand(c_sw(0, 7, 124)), s_type(124, 8, 15, 0b010, 0x23));
        assert_eq!(expand(c_sw(1, 6, 84)), s_type(84, 9, 14, 0b010, 0x23));
        for b in 2..=6 {
            let imm = 1u32 << b;
            assert_eq!(
                expand(c_sw(1, 6, imm)),
                s_type(imm as i32, 9, 14, 0b010, 0x23)
            );
        }
    }

    #[test]
    fn c_sd_expands_with_doubleword_scaled_offset() {
        assert_eq!(expand(c_sd(2, 5, 248)), s_type(248, 10, 13, 0b011, 0x23));
        assert_eq!(expand(c_sd(3, 4, 168)), s_type(168, 11, 12, 0b011, 0x23));
        for b in 3..=7 {
            let imm = 1u32 << b;
            assert_eq!(
                expand(c_sd(3, 4, imm)),
                s_type(imm as i32, 11, 12, 0b011, 0x23)
            );
        }
    }

    #[test]
    fn c_fsd_expands_to_fp_store() {
        for b in 3..=7 {
            let imm = 1u32 << b;
            assert_eq!(
                expand(c_fsd(4, 3, imm)),
                s_type(imm as i32, 12, 11, 0b011, 0x27)
            );
        }
    }

    #[test]
    fn quadrant0_reserved_encodings_are_illegal() {
        assert_eq!(expand(0x0000), 0xffff_ffff); // defined-illegal all-zero halfword
        assert_eq!(expand(c_addi4spn(5, 0)), 0xffff_ffff); // nzuimm=0 reserved
        assert_eq!(expand(0x8000), 0xffff_ffff); // funct3=100 unallocated in quadrant 0
    }

    #[test]
    fn c_addi_expands_and_canonicalizes_hints_to_nop() {
        assert_eq!(expand(ci(0b000, 31, 1)), i_type(1, 31, 0b000, 31, 0x13));
        assert_eq!(expand(ci(0b000, 1, -32)), i_type(-32, 1, 0b000, 1, 0x13)); // min imm
        assert_eq!(expand(ci(0b000, 5, 21)), i_type(21, 5, 0b000, 5, 0x13)); // 0b010101
        assert_eq!(expand(ci(0b000, 0, 5)), 0x13); // rd=0 (C.NOP)
        assert_eq!(expand(ci(0b000, 7, 0)), 0x13); // imm=0 hint
        for b in 0..=4 {
            let imm = 1 << b;
            assert_eq!(expand(ci(0b000, 5, imm)), i_type(imm, 5, 0b000, 5, 0x13));
        }
    }

    #[test]
    fn c_addiw_expands_and_rejects_rd_zero() {
        assert_eq!(expand(ci(0b001, 1, -1)), i_type(-1, 1, 0b000, 1, 0x1b));
        assert_eq!(expand(ci(0b001, 31, 21)), i_type(21, 31, 0b000, 31, 0x1b));
        assert_eq!(expand(ci(0b001, 3, -32)), i_type(-32, 3, 0b000, 3, 0x1b));
        assert_eq!(expand(ci(0b001, 0, 5)), 0xffff_ffff); // rd=0 reserved
        for b in 0..=4 {
            let imm = 1 << b;
            assert_eq!(expand(ci(0b001, 3, imm)), i_type(imm, 3, 0b000, 3, 0x1b));
        }
    }

    #[test]
    fn c_li_expands_to_addi_from_x0() {
        assert_eq!(expand(ci(0b010, 5, -32)), i_type(-32, 0, 0b000, 5, 0x13));
        assert_eq!(expand(ci(0b010, 31, 31)), i_type(31, 0, 0b000, 31, 0x13));
        assert_eq!(expand(ci(0b010, 0, 7)), 0x13); // rd=0 hint
    }

    #[test]
    fn c_addi16sp_expands_16_byte_scaled_immediate() {
        assert_eq!(expand(c_addi16sp(496)), i_type(496, 2, 0b000, 2, 0x13)); // max
        assert_eq!(expand(c_addi16sp(-512)), i_type(-512, 2, 0b000, 2, 0x13)); // min
        assert_eq!(expand(c_addi16sp(-336)), i_type(-336, 2, 0b000, 2, 0x13)); // mixed bits
        assert_eq!(expand(c_addi16sp(0)), 0xffff_ffff); // nzimm=0 reserved
        for b in 4..=8 {
            let imm = 1 << b;
            assert_eq!(expand(c_addi16sp(imm)), i_type(imm, 2, 0b000, 2, 0x13));
        }
    }

    #[test]
    fn c_lui_expands_sign_extended_and_rejects_zero_imm() {
        assert_eq!(expand(ci(0b011, 7, 1)), lui(1, 7));
        assert_eq!(expand(ci(0b011, 1, -32)), lui(-32, 1)); // sign bit only
        assert_eq!(expand(ci(0b011, 3, -1)), lui(-1, 3)); // all six bits set
        assert_eq!(expand(ci(0b011, 4, 21)), lui(21, 4)); // 0b010101
        assert_eq!(expand(ci(0b011, 1, 0)), 0xffff_ffff); // nzimm=0 reserved
        assert_eq!(expand(ci(0b011, 0, 5)), 0x13); // rd=0 emits canonical nop
        for b in 0..=4 {
            let imm = 1 << b;
            assert_eq!(expand(ci(0b011, 5, imm)), lui(imm, 5));
        }
    }

    #[test]
    fn c_srli_c_srai_expand_rv64_shift_range() {
        assert_eq!(expand(cb_alu(0b00, 0, 1)), i_type(1, 8, 0b101, 8, 0x13));
        assert_eq!(expand(cb_alu(0b00, 7, 63)), i_type(63, 15, 0b101, 15, 0x13)); // shamt[5] set
        assert_eq!(expand(cb_alu(0b00, 2, 42)), i_type(42, 10, 0b101, 10, 0x13));
        // SRAI carries 0b010000 in imm[11:6]
        assert_eq!(
            expand(cb_alu(0b01, 1, 63)),
            i_type(0x400 | 0x3f, 9, 0b101, 9, 0x13)
        );
        assert_eq!(
            expand(cb_alu(0b01, 4, 42)),
            i_type(0x400 | 0x2a, 12, 0b101, 12, 0x13)
        );
        for b in 0..=5 {
            let shamt = 1 << b;
            assert_eq!(
                expand(cb_alu(0b00, 2, shamt)),
                i_type(shamt, 10, 0b101, 10, 0x13)
            );
        }
    }

    #[test]
    fn c_andi_expands_sign_extended_immediate() {
        assert_eq!(expand(cb_alu(0b10, 3, -1)), i_type(-1, 11, 0b111, 11, 0x13));
        assert_eq!(
            expand(cb_alu(0b10, 6, -32)),
            i_type(-32, 14, 0b111, 14, 0x13)
        );
        assert_eq!(expand(cb_alu(0b10, 5, 21)), i_type(21, 13, 0b111, 13, 0x13));
        for b in 0..=4 {
            let imm = 1 << b;
            assert_eq!(
                expand(cb_alu(0b10, 5, imm)),
                i_type(imm, 13, 0b111, 13, 0x13)
            );
        }
    }

    #[test]
    fn ca_format_expands_register_arithmetic() {
        assert_eq!(
            expand(ca(0, 0b00, 0, 7)),
            r_type(0x20, 15, 8, 0b000, 8, 0x33)
        ); // c.sub
        assert_eq!(expand(ca(0, 0b01, 1, 6)), r_type(0, 14, 9, 0b100, 9, 0x33)); // c.xor
        assert_eq!(
            expand(ca(0, 0b10, 2, 5)),
            r_type(0, 13, 10, 0b110, 10, 0x33)
        ); // c.or
        assert_eq!(
            expand(ca(0, 0b11, 3, 4)),
            r_type(0, 12, 11, 0b111, 11, 0x33)
        ); // c.and
        assert_eq!(
            expand(ca(1, 0b00, 7, 0)),
            r_type(0x20, 8, 15, 0b000, 15, 0x3b)
        ); // c.subw
        assert_eq!(expand(ca(1, 0b01, 6, 1)), r_type(0, 9, 14, 0b000, 14, 0x3b));
        // c.addw
    }

    #[test]
    fn ca_format_reserved_rv64_slots_are_illegal() {
        assert_eq!(expand(ca(1, 0b10, 0, 0)), 0xffff_ffff);
        assert_eq!(expand(ca(1, 0b11, 5, 3)), 0xffff_ffff);
    }

    #[test]
    fn c_j_expands_signed_scattered_offset() {
        assert_eq!(expand(c_j(2)), j_type(2, 0));
        assert_eq!(expand(c_j(-2)), j_type(-2, 0));
        assert_eq!(expand(c_j(2046)), j_type(2046, 0)); // max
        assert_eq!(expand(c_j(-2048)), j_type(-2048, 0)); // min
        assert_eq!(expand(c_j(1366)), j_type(1366, 0)); // bits 1,2,4,6,8,10
        assert_eq!(expand(c_j(-1366)), j_type(-1366, 0));
        for b in 1..=10 {
            let offset = 1 << b;
            assert_eq!(expand(c_j(offset)), j_type(offset, 0));
        }
    }

    #[test]
    fn c_beqz_c_bnez_expand_branches_against_x0() {
        // The expansion places rs1' in the rs2 slot and x0 in rs1; equality
        // comparison commutes so this is spec-equivalent.
        assert_eq!(expand(cb_branch(0b110, 1, -256)), b_type(-256, 9, 0, 0b000)); // min
        assert_eq!(expand(cb_branch(0b110, 0, 254)), b_type(254, 8, 0, 0b000)); // max
        assert_eq!(expand(cb_branch(0b110, 7, 170)), b_type(170, 15, 0, 0b000)); // bits 1,3,5,7
        assert_eq!(expand(cb_branch(0b111, 2, -86)), b_type(-86, 10, 0, 0b001));
        assert_eq!(expand(cb_branch(0b111, 5, 6)), b_type(6, 13, 0, 0b001));
        for b in 1..=7 {
            let offset = 1 << b;
            assert_eq!(
                expand(cb_branch(0b110, 1, offset)),
                b_type(offset, 9, 0, 0b000)
            );
        }
    }

    #[test]
    fn c_slli_expands_full_register_and_shamt_range() {
        assert_eq!(expand(c_slli(3, 63)), i_type(63, 3, 0b001, 3, 0x13));
        assert_eq!(expand(c_slli(31, 1)), i_type(1, 31, 0b001, 31, 0x13));
        assert_eq!(expand(c_slli(10, 42)), i_type(42, 10, 0b001, 10, 0x13));
        for b in 0..=5 {
            let shamt = 1u32 << b;
            assert_eq!(
                expand(c_slli(10, shamt)),
                i_type(shamt as i32, 10, 0b001, 10, 0x13)
            );
        }
    }

    #[test]
    fn c_lwsp_c_ldsp_expand_and_reject_rd_zero() {
        assert_eq!(expand(c_lwsp(4, 252)), i_type(252, 2, 0b010, 4, 0x03)); // max
        assert_eq!(expand(c_lwsp(31, 4)), i_type(4, 2, 0b010, 31, 0x03));
        assert_eq!(expand(c_lwsp(1, 168)), i_type(168, 2, 0b010, 1, 0x03)); // bits 3,5,7
        assert_eq!(expand(c_lwsp(0, 8)), 0xffff_ffff); // rd=0 reserved

        assert_eq!(expand(c_ldsp(9, 504)), i_type(504, 2, 0b011, 9, 0x03)); // max
        assert_eq!(expand(c_ldsp(1, 8)), i_type(8, 2, 0b011, 1, 0x03));
        assert_eq!(expand(c_ldsp(31, 336)), i_type(336, 2, 0b011, 31, 0x03)); // bits 4,6,8
        assert_eq!(expand(c_ldsp(0, 8)), 0xffff_ffff); // rd=0 reserved

        for b in 2..=7 {
            let imm = 1u32 << b;
            assert_eq!(
                expand(c_lwsp(4, imm)),
                i_type(imm as i32, 2, 0b010, 4, 0x03)
            );
        }
        for b in 3..=8 {
            let imm = 1u32 << b;
            assert_eq!(
                expand(c_ldsp(9, imm)),
                i_type(imm as i32, 2, 0b011, 9, 0x03)
            );
        }
    }

    #[test]
    fn c_fldsp_expands_to_fp_load_from_sp() {
        for b in 3..=8 {
            let imm = 1u32 << b;
            assert_eq!(
                expand(c_fldsp(8, imm)),
                i_type(imm as i32, 2, 0b011, 8, 0x07)
            );
        }
        // Spec allows rd=f0 here, but this decoder treats rd=0 as illegal.
        assert_eq!(expand(c_fldsp(0, 8)), 0xffff_ffff);
    }

    #[test]
    fn cr_group_expands_jr_mv_ebreak_jalr_add() {
        assert_eq!(expand(cr(0, 31, 0)), i_type(0, 31, 0b000, 0, 0x67)); // c.jr
        assert_eq!(expand(cr(0, 0, 0)), 0xffff_ffff); // rs1=0 reserved
        assert_eq!(expand(cr(0, 5, 6)), r_type(0, 6, 0, 0b000, 5, 0x33)); // c.mv
        assert_eq!(expand(cr(0, 0, 6)), 0x13); // c.mv rd=0 hint
        assert_eq!(expand(cr(1, 0, 0)), 0x0010_0073); // c.ebreak
        assert_eq!(expand(cr(1, 5, 0)), i_type(0, 5, 0b000, 1, 0x67)); // c.jalr
        assert_eq!(expand(cr(1, 5, 31)), r_type(0, 31, 5, 0b000, 5, 0x33)); // c.add
        assert_eq!(expand(cr(1, 0, 31)), 0x13); // c.add rd=0 hint
    }

    #[test]
    fn c_swsp_c_sdsp_c_fsdsp_expand_sp_relative_stores() {
        assert_eq!(expand(c_swsp(31, 252)), s_type(252, 31, 2, 0b010, 0x23)); // max
        assert_eq!(expand(c_swsp(1, 4)), s_type(4, 1, 2, 0b010, 0x23));
        assert_eq!(expand(c_swsp(8, 84)), s_type(84, 8, 2, 0b010, 0x23)); // bits 2,4,6

        assert_eq!(expand(c_sdsp(8, 504)), s_type(504, 8, 2, 0b011, 0x23)); // max
        assert_eq!(expand(c_sdsp(31, 8)), s_type(8, 31, 2, 0b011, 0x23));
        assert_eq!(expand(c_sdsp(2, 328)), s_type(328, 2, 2, 0b011, 0x23)); // bits 3,6,8

        assert_eq!(expand(c_fsdsp(9, 16)), s_type(16, 9, 2, 0b011, 0x27));

        for b in 2..=7 {
            let imm = 1u32 << b;
            assert_eq!(
                expand(c_swsp(8, imm)),
                s_type(imm as i32, 8, 2, 0b010, 0x23)
            );
        }
        for b in 3..=8 {
            let imm = 1u32 << b;
            assert_eq!(
                expand(c_sdsp(8, imm)),
                s_type(imm as i32, 8, 2, 0b011, 0x23)
            );
            assert_eq!(
                expand(c_fsdsp(9, imm)),
                s_type(imm as i32, 9, 2, 0b011, 0x27)
            );
        }
    }

    #[test]
    fn compressed_register_indices_map_to_x8_x15() {
        for rp in 0..8 {
            assert_eq!(expand(c_addi4spn(rp, 8)), i_type(8, 2, 0b000, rp + 8, 0x13));
            assert_eq!(
                expand(ca(0, 0b01, rp, 7 - rp)),
                r_type(0, 15 - rp, rp + 8, 0b100, rp + 8, 0x33)
            );
        }
    }

    #[test]
    fn non_compressed_or_unallocated_encodings_return_all_ones() {
        assert_eq!(expand(0x8093), 0xffff_ffff); // low bits 0b11: not compressed
        assert_eq!(expand(0xffff), 0xffff_ffff);
        assert_eq!(expand(0x0003), 0xffff_ffff);
    }
}
