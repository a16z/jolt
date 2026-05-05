/// Expand a 16-bit RVC instruction into the RV64 32-bit instruction word that
/// Jolt's decoder expects.
///
/// This helper deliberately has no RV32 mode. The new `jolt-program` pipeline
/// rejects ELF32/RV32 at the image boundary; tracer may keep legacy RV32 paths
/// until its own cleanup lands.
#[inline]
pub fn uncompress_rv64_instruction(halfword: u32) -> u32 {
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

    #[test]
    fn uncompresses_nop() {
        assert_eq!(uncompress_rv64_instruction(0x0001), 0x13);
    }

    #[test]
    fn uncompresses_c_slli_x0() {
        assert_eq!(uncompress_rv64_instruction(0x107a), 0x03e0_1013);
    }
}
