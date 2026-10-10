use super::*;
#[cfg(feature = "field-inline")]
use jolt_riscv::RV64IMAC_JOLT_FIELD_INLINE;
use jolt_riscv::{uncompress_rv64_instruction, RV64I, RV64IMAC_JOLT, RV64IM_JOLT};

#[test]
fn rv64i_decodes_every_base_instruction() {
    let cases = [
        (0x1234_50b7, SourceInstructionKind::LUI), // lui x1,0x12345
        (0x1234_5097, SourceInstructionKind::AUIPC), // auipc x1,0x12345
        (0x0040_00ef, SourceInstructionKind::JAL), // jal x1,4
        (0x0041_00e7, SourceInstructionKind::JALR), // jalr x1,4(x2)
        (0x0031_0263, SourceInstructionKind::BEQ), // beq x2,x3,4
        (0x0031_1263, SourceInstructionKind::BNE), // bne x2,x3,4
        (0x0031_4263, SourceInstructionKind::BLT), // blt x2,x3,4
        (0x0031_5263, SourceInstructionKind::BGE), // bge x2,x3,4
        (0x0031_6263, SourceInstructionKind::BLTU), // bltu x2,x3,4
        (0x0031_7263, SourceInstructionKind::BGEU), // bgeu x2,x3,4
        (0x0041_0083, SourceInstructionKind::LB),  // lb x1,4(x2)
        (0x0041_1083, SourceInstructionKind::LH),  // lh x1,4(x2)
        (0x0041_2083, SourceInstructionKind::LW),  // lw x1,4(x2)
        (0x0041_4083, SourceInstructionKind::LBU), // lbu x1,4(x2)
        (0x0041_5083, SourceInstructionKind::LHU), // lhu x1,4(x2)
        (0x0031_0223, SourceInstructionKind::SB),  // sb x3,4(x2)
        (0x0031_1223, SourceInstructionKind::SH),  // sh x3,4(x2)
        (0x0031_2223, SourceInstructionKind::SW),  // sw x3,4(x2)
        (0x0041_0093, SourceInstructionKind::ADDI), // addi x1,x2,4
        (0x0041_2093, SourceInstructionKind::SLTI), // slti x1,x2,4
        (0x0041_3093, SourceInstructionKind::SLTIU), // sltiu x1,x2,4
        (0x0041_4093, SourceInstructionKind::XORI), // xori x1,x2,4
        (0x0041_6093, SourceInstructionKind::ORI), // ori x1,x2,4
        (0x0041_7093, SourceInstructionKind::ANDI), // andi x1,x2,4
        (0x0041_1093, SourceInstructionKind::SLLI), // slli x1,x2,4
        (0x0041_5093, SourceInstructionKind::SRLI), // srli x1,x2,4
        (0x4041_5093, SourceInstructionKind::SRAI), // srai x1,x2,4
        (0x0031_00b3, SourceInstructionKind::ADD), // add x1,x2,x3
        (0x4031_00b3, SourceInstructionKind::SUB), // sub x1,x2,x3
        (0x0031_10b3, SourceInstructionKind::SLL), // sll x1,x2,x3
        (0x0031_20b3, SourceInstructionKind::SLT), // slt x1,x2,x3
        (0x0031_30b3, SourceInstructionKind::SLTU), // sltu x1,x2,x3
        (0x0031_40b3, SourceInstructionKind::XOR), // xor x1,x2,x3
        (0x0031_50b3, SourceInstructionKind::SRL), // srl x1,x2,x3
        (0x4031_50b3, SourceInstructionKind::SRA), // sra x1,x2,x3
        (0x0031_60b3, SourceInstructionKind::OR),  // or x1,x2,x3
        (0x0031_70b3, SourceInstructionKind::AND), // and x1,x2,x3
        (0x0ff0_000f, SourceInstructionKind::FENCE), // fence iorw,iorw
        (0x0000_0073, SourceInstructionKind::ECALL), // ecall
        (0x0010_0073, SourceInstructionKind::EBREAK), // ebreak
        (0x0041_6083, SourceInstructionKind::LWU), // lwu x1,4(x2)
        (0x0081_3083, SourceInstructionKind::LD),  // ld x1,8(x2)
        (0x0031_3423, SourceInstructionKind::SD),  // sd x3,8(x2)
        (0x0041_009b, SourceInstructionKind::ADDIW), // addiw x1,x2,4
        (0x0041_109b, SourceInstructionKind::SLLIW), // slliw x1,x2,4
        (0x0041_509b, SourceInstructionKind::SRLIW), // srliw x1,x2,4
        (0x4041_509b, SourceInstructionKind::SRAIW), // sraiw x1,x2,4
        (0x0031_00bb, SourceInstructionKind::ADDW), // addw x1,x2,x3
        (0x4031_00bb, SourceInstructionKind::SUBW), // subw x1,x2,x3
        (0x0031_10bb, SourceInstructionKind::SLLW), // sllw x1,x2,x3
        (0x0031_50bb, SourceInstructionKind::SRLW), // srlw x1,x2,x3
        (0x4031_50bb, SourceInstructionKind::SRAW), // sraw x1,x2,x3
    ];
    assert_eq!(cases.len(), 52);
    for (word, kind) in cases {
        let decoded = decode_instruction(word, 0x8000_0000, false, RV64I);
        assert!(
            matches!(decoded.as_ref().map(SourceInstruction::kind), Ok(actual) if actual == kind),
            "{kind:?}: {decoded:?}"
        );
    }
}

#[test]
fn rv64i_rejects_non_base_instructions_and_unknown_encodings() {
    for (word, kind) in [
        (0x0231_00b3, SourceInstructionKind::MUL), // mul x1,x2,x3
        (0x0031_20af, SourceInstructionKind::AMOADDW), // amoadd.w x1,x3,(x2)
        (0x3051_10f3, SourceInstructionKind::CSRRW), // csrrw x1,mtvec,x2
        (0x3020_0073, SourceInstructionKind::MRET), // mret
        (0x0620_d20b, SourceInstructionKind::Inline), // .insn r 0x0b,5,3,x4,x1,x2
    ] {
        assert!(matches!(
            decode_instruction(word, 0x8000_0000, false, RV64I),
            Err(ProgramError::IllegalSourceInstruction(actual)) if actual == kind
        ));
    }
    assert!(matches!(
        decode_instruction(0xffff_ffff, 0x8000_0000, false, RV64I),
        Err(ProgramError::MalformedImage("unknown RV64 opcode"))
    ));
}

#[test]
fn compressed_legality_precedes_kind_and_alignment_checks() {
    let compressed = 0x0085; // c.addi x1,1
    let word = uncompress_rv64_instruction(compressed);
    assert_eq!(word, 0x0010_8093); // addi x1,x1,1
    for profile in [RV64I, RV64IM_JOLT] {
        for (word, address) in [(word, 0x8000_0000), (0xffff_ffff, 0x8000_0002)] {
            let result = decode_instruction(word, address, true, profile);
            assert!(matches!(
                result,
                Err(ProgramError::IllegalCompressedInstruction { address: actual }) if actual == address
            ));
        }
    }
    let decoded = decode_instruction(word, 0x8000_0000, true, RV64IMAC_JOLT);
    assert!(matches!(
        decoded.as_ref().map(SourceInstruction::kind),
        Ok(SourceInstructionKind::ADDI)
    ));
}

#[test]
fn alignment_checks_precede_kind_decoding_only_without_rv64c() {
    let nop = 0x0000_0013; // addi x0,x0,0
    for address in [0x8000_0001, 0x8000_0002, 0x8000_0003] {
        for word in [nop, 0xffff_ffff] {
            assert!(matches!(
                decode_instruction(word, address, false, RV64I),
                Err(ProgramError::MalformedImage(
                    "instruction address is not 4-byte aligned"
                ))
            ));
        }
        for is_compressed in [false, true] {
            let decoded = decode_instruction(nop, address, is_compressed, RV64IMAC_JOLT);
            assert!(matches!(
                decoded.as_ref().map(SourceInstruction::kind),
                Ok(SourceInstructionKind::ADDI)
            ));
            let row = decoded.unwrap_or_else(|error| panic!("{error}"));
            assert_eq!(row.row().address, address as usize);
            assert_eq!(row.row().is_compressed, is_compressed);
            assert_eq!(
                row.row().operands,
                NormalizedOperands {
                    rd: Some(0),
                    rs1: Some(0),
                    rs2: None,
                    imm: 0,
                }
            );
            assert!(matches!(
                decode_instruction(0xffff_ffff, address, is_compressed, RV64IMAC_JOLT),
                Err(ProgramError::MalformedImage("unknown RV64 opcode"))
            ));
        }
    }
}

fn field_word(funct3: u32, rd: u8, rs1: u8, rs2_or_imm: u32) -> u32 {
    0x7b | (funct3 << 12) | (u32::from(rd) << 7) | (u32::from(rs1) << 15) | (rs2_or_imm << 20)
}

fn bit(value: u32, index: u32) -> u32 {
    (value >> index) & 1
}

fn field_bits(value: u32, hi: u32, lo: u32) -> u32 {
    (value >> lo) & ((1 << (hi - lo + 1)) - 1)
}

// Encoding-side assemblers transcribed from the RV64I base instruction
// formats (unprivileged spec §2.3); they scatter immediates independently
// of the reassembly code under test.

fn b_word(offset: i32, rs2: u32, rs1: u32, funct3: u32) -> u32 {
    let imm = offset as u32;
    (bit(imm, 12) << 31)
        | (field_bits(imm, 10, 5) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | (field_bits(imm, 4, 1) << 8)
        | (bit(imm, 11) << 7)
        | 0x63
}

fn j_word(offset: i32, rd: u32) -> u32 {
    let imm = offset as u32;
    (bit(imm, 20) << 31)
        | (field_bits(imm, 10, 1) << 21)
        | (bit(imm, 11) << 20)
        | (field_bits(imm, 19, 12) << 12)
        | (rd << 7)
        | 0x6f
}

fn s_word(imm: i32, rs2: u32, rs1: u32, funct3: u32) -> u32 {
    let imm = imm as u32 & 0xfff;
    (field_bits(imm, 11, 5) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | (field_bits(imm, 4, 0) << 7)
        | 0x23
}

fn u_word(imm31_12: u32, rd: u32, opcode: u32) -> u32 {
    (imm31_12 << 12) | (rd << 7) | opcode
}

fn decode_ok(word: u32, address: u64, is_compressed: bool) -> SourceInstruction {
    match decode_instruction(word, address, is_compressed, RV64IMAC_JOLT) {
        Ok(instruction) => instruction,
        Err(error) => panic!("decode failed for word {word:#010x}: {error:?}"),
    }
}

#[test]
fn sign_extension_helpers_handle_boundary_widths() {
    assert_eq!(sign_extend_i64(0x7ff, 12), 2047);
    assert_eq!(sign_extend_i64(0x800, 12), -2048);
    assert_eq!(sign_extend_i64(0xfff, 12), -1);
    assert_eq!(sign_extend_i64(0, 12), 0);
    assert_eq!(sign_extend_i64(0xffff_f7ff, 12), 2047);
    assert_eq!(sign_extend_i64(1, 1), -1);
    assert_eq!(sign_extend_i64(0, 1), 0);
    assert_eq!(sign_extend_i64(0x8000_0000, 32), i64::from(i32::MIN));
    assert_eq!(sign_extend_i64(0x7fff_ffff, 32), i64::from(i32::MAX));

    assert_eq!(sign_extend_u64(0x800, 12), 0xffff_ffff_ffff_f800);
    assert_eq!(sign_extend_u64(0x7ff, 12), 0x7ff);

    assert_eq!(
        sign_extension_mask(0x8000_0000, 0x8000_0000, 0xffff_f000),
        0xffff_f000
    );
    assert_eq!(
        sign_extension_mask(0x7fff_ffff, 0x8000_0000, 0xffff_f000),
        0
    );
}

#[test]
fn format_b_operands_reassembles_scattered_branch_immediate() {
    assert_eq!(
        format_b_operands(b_word(-4096, 2, 1, 0b000)),
        NormalizedOperands {
            rs1: Some(1),
            rs2: Some(2),
            rd: None,
            imm: -4096,
        }
    );
    assert_eq!(format_b_operands(b_word(4094, 31, 15, 0b000)).imm, 4094);
    assert_eq!(format_b_operands(b_word(-2, 0, 0, 0b000)).imm, -2);
    for b in 1..=11 {
        let offset = 1 << b;
        assert_eq!(
            format_b_operands(b_word(offset, 3, 4, 0b000)).imm,
            i128::from(offset)
        );
    }
}

#[test]
fn format_j_operands_reassembles_scattered_jump_immediate() {
    // JAL immediates carry the 64-bit two's-complement pattern
    // zero-extended into i128, not a negative i128
    let operands = format_j_operands(j_word(-2, 1));
    assert_eq!(operands.rd, Some(1));
    assert_eq!(operands.rs1, None);
    assert_eq!(operands.rs2, None);
    assert_eq!(operands.imm, i128::from(-2i64 as u64));
    assert_eq!(
        format_j_operands(j_word(-1_048_576, 0)).imm,
        i128::from(-1_048_576i64 as u64)
    );
    assert_eq!(format_j_operands(j_word(703_710, 0)).imm, 703_710);
    for b in 1..=19 {
        let offset = 1 << b;
        assert_eq!(format_j_operands(j_word(offset, 5)).imm, i128::from(offset));
    }
}

#[test]
fn format_s_operands_reassembles_split_store_immediate() {
    assert_eq!(
        format_s_operands(s_word(-2048, 10, 11, 0b011)),
        NormalizedOperands {
            rs1: Some(11),
            rs2: Some(10),
            rd: None,
            imm: -2048,
        }
    );
    assert_eq!(format_s_operands(s_word(2047, 1, 2, 0b011)).imm, 2047);
    assert_eq!(format_s_operands(s_word(-677, 1, 2, 0b011)).imm, -677);
    for b in 0..=10 {
        let imm = 1 << b;
        assert_eq!(
            format_s_operands(s_word(imm, 6, 7, 0b011)).imm,
            i128::from(imm)
        );
    }
}

#[test]
fn format_u_operands_zero_extends_the_64_bit_pattern() {
    assert_eq!(
        format_u_operands(u_word(0x12345, 3, 0x37)),
        NormalizedOperands {
            rs1: None,
            rs2: None,
            rd: Some(3),
            imm: 0x1234_5000,
        }
    );
    // sign bit set: the sign-extended u64 pattern appears as a large
    // positive i128
    assert_eq!(
        format_u_operands(u_word(0xfffff, 3, 0x37)).imm,
        i128::from(0xffff_ffff_ffff_f000u64)
    );
}

#[test]
fn decodes_r_format_add_with_register_operands() {
    let instruction = decode_ok(0x0073_02b3, 0x8000_0000, false); // add t0,t1,t2
    assert_eq!(instruction.kind(), SourceInstructionKind::ADD);
    assert_eq!(
        instruction.row().operands,
        NormalizedOperands {
            rs1: Some(6),
            rs2: Some(7),
            rd: Some(5),
            imm: 0,
        }
    );
}

#[test]
fn decodes_i_format_addi_and_records_row_metadata() {
    let instruction = decode_ok(0xff01_0113, 0x8000_0010, true); // addi sp,sp,-16
    assert_eq!(instruction.kind(), SourceInstructionKind::ADDI);
    // unlike loads (format_load_operands), plain I-format immediates carry
    // the zero-extended 64-bit two's-complement pattern in the i128
    assert_eq!(
        instruction.row().operands,
        NormalizedOperands {
            rs1: Some(2),
            rs2: None,
            rd: Some(2),
            imm: i128::from(-16i64 as u64),
        }
    );
    assert_eq!(instruction.row().address, 0x8000_0010);
    assert!(instruction.row().is_compressed);

    let instruction = decode_ok(0x0010_8093, 0x8000_0000, false); // addi ra,ra,1
    assert_eq!(instruction.kind(), SourceInstructionKind::ADDI);
    assert_eq!(instruction.row().operands.imm, 1);
    assert!(!instruction.row().is_compressed);
}

#[test]
fn decodes_loads_with_sign_extended_offset() {
    let instruction = decode_ok(0xff84_a503, 0x8000_0000, false); // lw a0,-8(s1)
    assert_eq!(instruction.kind(), SourceInstructionKind::LW);
    assert_eq!(
        instruction.row().operands,
        NormalizedOperands {
            rs1: Some(9),
            rs2: None,
            rd: Some(10),
            imm: -8,
        }
    );

    let instruction = decode_ok(0x0106_3583, 0x8000_0000, false); // ld a1,16(a2)
    assert_eq!(instruction.kind(), SourceInstructionKind::LD);
    assert_eq!(instruction.row().operands.imm, 16);
}

#[test]
fn decodes_s_format_store_with_negative_offset() {
    let instruction = decode_ok(s_word(-16, 10, 11, 0b011), 0x8000_0000, false);
    assert_eq!(instruction.kind(), SourceInstructionKind::SD);
    assert_eq!(
        instruction.row().operands,
        NormalizedOperands {
            rs1: Some(11),
            rs2: Some(10),
            rd: None,
            imm: -16,
        }
    );
}

#[test]
fn decodes_b_format_branch_with_negative_target() {
    let instruction = decode_ok(b_word(-4, 9, 8, 0b001), 0x8000_0000, false);
    assert_eq!(instruction.kind(), SourceInstructionKind::BNE);
    assert_eq!(
        instruction.row().operands,
        NormalizedOperands {
            rs1: Some(8),
            rs2: Some(9),
            rd: None,
            imm: -4,
        }
    );
}

#[test]
fn decodes_u_format_lui_and_auipc() {
    let instruction = decode_ok(u_word(0x12345, 7, 0x37), 0x8000_0000, false);
    assert_eq!(instruction.kind(), SourceInstructionKind::LUI);
    assert_eq!(instruction.row().operands.imm, 0x1234_5000);

    let instruction = decode_ok(u_word(0xfffff, 7, 0x17), 0x8000_0000, false);
    assert_eq!(instruction.kind(), SourceInstructionKind::AUIPC);
    assert_eq!(
        instruction.row().operands.imm,
        i128::from(0xffff_ffff_ffff_f000u64)
    );
}

#[test]
fn decodes_j_format_jal_with_negative_offset() {
    let instruction = decode_ok(j_word(-2, 1), 0x8000_0000, false);
    assert_eq!(instruction.kind(), SourceInstructionKind::JAL);
    assert_eq!(instruction.row().operands.rd, Some(1));
    assert_eq!(instruction.row().operands.imm, i128::from(-2i64 as u64));
}

#[test]
fn decodes_amo_and_ignores_aq_rl_bits() {
    // amoadd.w.aq.rl a0,a1,(a2): funct5 selects the operation; aq/rl
    // (bits 26:25) must not affect decoding
    let word = (0b11 << 25) | (11 << 20) | (12 << 15) | (0b010 << 12) | (10 << 7) | 0x2f;
    let instruction = decode_ok(word, 0x8000_0000, false);
    assert_eq!(instruction.kind(), SourceInstructionKind::AMOADDW);
    assert_eq!(
        instruction.row().operands,
        NormalizedOperands {
            rs1: Some(12),
            rs2: Some(11),
            rd: Some(10),
            imm: 0,
        }
    );

    let word = (0b00010 << 27) | (6 << 15) | (0b011 << 12) | (5 << 7) | 0x2f; // lr.d t0,(t1)
    assert_eq!(
        decode_ok(word, 0x8000_0000, false).kind(),
        SourceInstructionKind::LRD
    );
}

#[test]
fn decodes_system_instructions_exactly() {
    assert_eq!(
        decode_ok(0x0000_0073, 0x8000_0000, false).kind(),
        SourceInstructionKind::ECALL
    );
    assert_eq!(
        decode_ok(0x0010_0073, 0x8000_0000, false).kind(),
        SourceInstructionKind::EBREAK
    );
    assert_eq!(
        decode_ok(0x3020_0073, 0x8000_0000, false).kind(),
        SourceInstructionKind::MRET
    );

    let word = (0x305 << 20) | (1 << 15) | (0b001 << 12) | (3 << 7) | 0x73; // csrrw gp,mtvec,ra
    let instruction = decode_ok(word, 0x8000_0000, false);
    assert_eq!(instruction.kind(), SourceInstructionKind::CSRRW);
    assert_eq!(instruction.row().operands.imm, 0x305);

    let word = (0xc00 << 20) | (0b010 << 12) | (5 << 7) | 0x73; // csrrs t0,cycle,x0
    let instruction = decode_ok(word, 0x8000_0000, false);
    assert_eq!(instruction.kind(), SourceInstructionKind::CSRRS);
    // I-format sign extension leaves the u64 bit pattern in the i128
    assert_eq!(
        instruction.row().operands.imm,
        i128::from(0xffff_ffff_ffff_fc00u64)
    );

    assert!(matches!(
        decode_instruction(0x0000_00f3, 0x8000_0000, false, RV64IMAC_JOLT),
        Err(ProgramError::MalformedImage(
            "unsupported system instruction"
        ))
    ));
}

#[test]
fn decodes_inline_opcode_with_dispatch_key() {
    let word = (3 << 25) | (2 << 20) | (1 << 15) | (0b101 << 12) | (4 << 7) | 0x0b;
    let instruction = decode_ok(word, 0x8000_0000, false);
    assert_eq!(instruction.kind(), SourceInstructionKind::Inline);
    assert_eq!(
        instruction.row().inline,
        Some(SourceInlineKey {
            opcode: 0x0b,
            funct3: 0b101,
            funct7: 3,
        })
    );
    assert_eq!(
        instruction.row().operands,
        NormalizedOperands {
            rs1: Some(1),
            rs2: Some(2),
            rd: Some(4),
            imm: 0,
        }
    );
}

#[test]
fn rejects_invalid_encodings_with_exact_messages() {
    let cases: &[(u32, &str)] = &[
        (0x0000_007f, "unknown RV64 opcode"),
        (0x63 | (0b010 << 12), "invalid branch funct3"),
        (0x67 | (0b001 << 12), "invalid JALR funct3"),
        (0x03 | (0b111 << 12), "invalid load funct3"),
        (0x23 | (0b100 << 12), "invalid store funct3"),
        ((1 << 26) | (0b001 << 12) | 0x13, "invalid SLLI funct6"),
        (
            (0b100000 << 26) | (0b101 << 12) | 0x13,
            "invalid shift-immediate funct6",
        ),
        (0x1b | (0b010 << 12), "invalid RV64 op-imm-32 instruction"),
        ((0b0000010 << 25) | 0x33, "invalid op instruction"),
        (0x3b | (0b010 << 12), "invalid RV64 op-32 instruction"),
        (
            (0b00101 << 27) | (0b010 << 12) | 0x2f,
            "invalid atomic memory operation",
        ),
        (
            (0b00010 << 27) | (1 << 20) | (0b010 << 12) | 0x2f,
            "invalid LR rs2",
        ),
        (
            (0b00010 << 27) | (0b11 << 25) | (31 << 20) | (0b011 << 12) | 0x2f,
            "invalid LR rs2",
        ),
        ((0x3f << 25) | 0x5b, "invalid custom instruction"),
        (0x0f | (0b001 << 12), "invalid MISC-MEM funct3"),
        (0x0f | (0b010 << 12), "invalid MISC-MEM funct3"),
        (0x0f | (0b011 << 12), "invalid MISC-MEM funct3"),
        (0x0f | (0b100 << 12), "invalid MISC-MEM funct3"),
        (0x0f | (0b101 << 12), "invalid MISC-MEM funct3"),
        (0x0f | (0b110 << 12), "invalid MISC-MEM funct3"),
        (0x0f | (0b111 << 12), "invalid MISC-MEM funct3"),
    ];
    for (word, message) in cases {
        match decode_instruction(*word, 0x8000_0000, false, RV64IMAC_JOLT) {
            Err(ProgramError::MalformedImage(actual)) => {
                assert_eq!(actual, *message, "wrong message for word {word:#010x}");
            }
            Err(error) => panic!("expected MalformedImage for {word:#010x}, got {error:?}"),
            Ok(_) => panic!("expected MalformedImage for {word:#010x}, got Ok"),
        }
    }
}

#[test]
fn rejects_source_instructions_outside_the_profile() {
    let word = (11 << 20) | (12 << 15) | (0b010 << 12) | (10 << 7) | 0x2f;
    match decode_instruction(word, 0x8000_0000, false, RV64IM_JOLT) {
        Err(ProgramError::IllegalSourceInstruction(kind)) => {
            assert_eq!(kind, SourceInstructionKind::AMOADDW);
        }
        Err(error) => panic!("expected IllegalSourceInstruction, got {error:?}"),
        Ok(_) => panic!("expected IllegalSourceInstruction, got Ok"),
    }
}

#[cfg(feature = "field-inline")]
fn field_r_word(funct7: u32, funct3: u32, rd: u8, rs1: u8, rs2: u8) -> u32 {
    field_word(funct3, rd, rs1, u32::from(rs2)) | (funct7 << 25)
}

#[cfg(feature = "field-inline")]
#[test]
fn decodes_field_inline_source_rows_only_for_field_inline_profile() {
    let word = field_word(FieldInlineOp::Mul.funct3().into(), 1, 2, 3);
    let decoded_base = decode_instruction(word, 0x8000_0000, false, RV64IMAC_JOLT);
    assert!(matches!(
        decoded_base,
        Err(ProgramError::IllegalSourceInstruction(
            SourceInstruction::FieldMul(_)
        ))
    ));

    let decoded_field_inline =
        decode_instruction(word, 0x8000_0000, false, RV64IMAC_JOLT_FIELD_INLINE);
    let instruction = match decoded_field_inline {
        Ok(instruction) => instruction,
        Err(error) => panic!("field-inline decode failed: {error:?}"),
    };
    assert_eq!(instruction.kind(), SourceInstructionKind::FIELD_MUL);
    assert_eq!(instruction.row().operands.rd, Some(1));
    assert_eq!(instruction.row().operands.rs1, Some(2));
    assert_eq!(instruction.row().operands.rs2, Some(3));
}

#[cfg(feature = "field-inline")]
#[test]
fn rejects_unknown_field_inline_r_type_funct7() {
    let word = field_r_word(1, u32::from(FieldInlineOp::Mul.funct3()), 1, 2, 3);
    assert!(matches!(
        decode_instruction(word, 0x8000_0000, false, RV64IMAC_JOLT_FIELD_INLINE),
        Err(ProgramError::MalformedImage(
            "invalid field-inline encoding"
        ))
    ));
}

#[cfg(feature = "field-inline")]
#[test]
fn assert_zero_decodes_only_a_field_source_and_rejects_retired_store() {
    let word = field_r_word(2, 6, 0, 3, 0);
    let instruction = match decode_instruction(word, 0x8000_0000, false, RV64IMAC_JOLT_FIELD_INLINE)
    {
        Ok(instruction) => instruction,
        Err(error) => panic!("field-inline zero assertion decode failed: {error:?}"),
    };
    assert_eq!(instruction.kind(), SourceInstructionKind::FIELD_ASSERT_ZERO);
    assert_eq!(
        instruction.row().operands,
        NormalizedOperands {
            rs1: Some(3),
            ..Default::default()
        }
    );
    assert!(decode_instruction(
        field_r_word(0, 6, 1, 3, 0),
        0x8000_0000,
        false,
        RV64IMAC_JOLT_FIELD_INLINE
    )
    .is_err());
}

/// `fence` (`fence iorw, iorw`) decodes as FENCE, the only MISC-MEM
/// instruction in RV64IMAC; the other funct3 values are rejected in
/// `rejects_invalid_encodings_with_exact_messages`.
#[test]
fn decodes_fence() {
    let fence = decode_instruction(0x0ff0_000f, 0x8000_0000, false, RV64IMAC_JOLT);
    assert!(
        matches!(
            fence.as_ref().map(SourceInstruction::kind),
            Ok(SourceInstructionKind::FENCE)
        ),
        "{fence:?}"
    );
}

#[cfg(not(feature = "field-inline"))]
#[test]
fn field_inline_opcode_is_unknown_without_feature() {
    let word = field_word(2, 1, 2, 3);
    assert!(matches!(
        decode_instruction(word, 0x8000_0000, false, RV64IMAC_JOLT),
        Err(ProgramError::MalformedImage("unknown RV64 opcode"))
    ));
}
