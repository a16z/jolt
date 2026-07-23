#![expect(
    clippy::unreadable_literal,
    reason = "RISC-V decode tables are easiest to audit in ISA bit-field widths"
)]

#[cfg(feature = "field-inline")]
use jolt_riscv::{FieldInlineOp, FIELD_INLINE_OPCODE};
use jolt_riscv::{
    JoltInstructionProfile, NormalizedOperands, SourceInlineKey, SourceInstruction,
    SourceInstructionKind, SourceInstructionRow,
};

use crate::ProgramError;

pub fn decode_instruction(
    word: u32,
    address: u64,
    is_compressed: bool,
    profile: JoltInstructionProfile,
) -> Result<SourceInstruction, ProgramError> {
    let opcode = word & 0x7f;
    let kind = match opcode {
        0b0110111 => SourceInstructionKind::LUI,
        0b0010111 => SourceInstructionKind::AUIPC,
        0b1101111 => SourceInstructionKind::JAL,
        0b1100111 => match funct3(word) {
            0b000 => SourceInstructionKind::JALR,
            _ => return invalid("invalid JALR funct3"),
        },
        0b1100011 => match funct3(word) {
            0b000 => SourceInstructionKind::BEQ,
            0b001 => SourceInstructionKind::BNE,
            0b100 => SourceInstructionKind::BLT,
            0b101 => SourceInstructionKind::BGE,
            0b110 => SourceInstructionKind::BLTU,
            0b111 => SourceInstructionKind::BGEU,
            _ => return invalid("invalid branch funct3"),
        },
        0b0000011 => match funct3(word) {
            0b000 => SourceInstructionKind::LB,
            0b001 => SourceInstructionKind::LH,
            0b010 => SourceInstructionKind::LW,
            0b011 => SourceInstructionKind::LD,
            0b100 => SourceInstructionKind::LBU,
            0b101 => SourceInstructionKind::LHU,
            0b110 => SourceInstructionKind::LWU,
            _ => return invalid("invalid load funct3"),
        },
        0b0100011 => match funct3(word) {
            0b000 => SourceInstructionKind::SB,
            0b001 => SourceInstructionKind::SH,
            0b010 => SourceInstructionKind::SW,
            0b011 => SourceInstructionKind::SD,
            _ => return invalid("invalid store funct3"),
        },
        0b0010011 => decode_op_imm(word)?,
        0b0011011 => decode_op_imm_32(word)?,
        0b0110011 => decode_op(word)?,
        0b0111011 => decode_op_32(word)?,
        0b0001111 => SourceInstructionKind::FENCE,
        0b0101111 => decode_amo(word)?,
        0b1110011 => decode_system(word)?,
        0b0001011 | 0b0101011 => SourceInstructionKind::Inline,
        0b1011011 => decode_custom(word)?,
        #[cfg(feature = "field-inline")]
        opcode if opcode == u32::from(FIELD_INLINE_OPCODE) => decode_field_inline(word)?,
        _ => return invalid("unknown RV64 opcode"),
    };

    if !profile.supports_source(kind) {
        return Err(ProgramError::IllegalSourceInstruction(kind));
    }
    Ok(source_instruction(kind, word, address, is_compressed))
}

fn decode_op_imm(word: u32) -> Result<SourceInstructionKind, ProgramError> {
    match funct3(word) {
        0b001 if funct6(word) == 0 => Ok(SourceInstructionKind::SLLI),
        0b001 => invalid("invalid SLLI funct6"),
        0b101 if funct6(word) == 0b000000 => Ok(SourceInstructionKind::SRLI),
        0b101 if funct6(word) == 0b010000 => Ok(SourceInstructionKind::SRAI),
        0b101 => invalid("invalid shift-immediate funct6"),
        0b000 => Ok(SourceInstructionKind::ADDI),
        0b010 => Ok(SourceInstructionKind::SLTI),
        0b011 => Ok(SourceInstructionKind::SLTIU),
        0b100 => Ok(SourceInstructionKind::XORI),
        0b110 => Ok(SourceInstructionKind::ORI),
        0b111 => Ok(SourceInstructionKind::ANDI),
        _ => invalid("invalid op-imm funct3"),
    }
}

fn decode_op_imm_32(word: u32) -> Result<SourceInstructionKind, ProgramError> {
    match (funct3(word), funct7(word)) {
        (0b000, _) => Ok(SourceInstructionKind::ADDIW),
        (0b001, 0b0000000) => Ok(SourceInstructionKind::SLLIW),
        (0b101, 0b0000000) => Ok(SourceInstructionKind::SRLIW),
        (0b101, 0b0100000) => Ok(SourceInstructionKind::SRAIW),
        _ => invalid("invalid RV64 op-imm-32 instruction"),
    }
}

fn decode_op(word: u32) -> Result<SourceInstructionKind, ProgramError> {
    match (funct3(word), funct7(word)) {
        (0b000, 0b0000000) => Ok(SourceInstructionKind::ADD),
        (0b000, 0b0100000) => Ok(SourceInstructionKind::SUB),
        (0b001, 0b0000000) => Ok(SourceInstructionKind::SLL),
        (0b010, 0b0000000) => Ok(SourceInstructionKind::SLT),
        (0b011, 0b0000000) => Ok(SourceInstructionKind::SLTU),
        (0b100, 0b0000000) => Ok(SourceInstructionKind::XOR),
        (0b101, 0b0000000) => Ok(SourceInstructionKind::SRL),
        (0b101, 0b0100000) => Ok(SourceInstructionKind::SRA),
        (0b110, 0b0000000) => Ok(SourceInstructionKind::OR),
        (0b111, 0b0000000) => Ok(SourceInstructionKind::AND),
        (0b000, 0b0000001) => Ok(SourceInstructionKind::MUL),
        (0b001, 0b0000001) => Ok(SourceInstructionKind::MULH),
        (0b010, 0b0000001) => Ok(SourceInstructionKind::MULHSU),
        (0b011, 0b0000001) => Ok(SourceInstructionKind::MULHU),
        (0b100, 0b0000001) => Ok(SourceInstructionKind::DIV),
        (0b101, 0b0000001) => Ok(SourceInstructionKind::DIVU),
        (0b110, 0b0000001) => Ok(SourceInstructionKind::REM),
        (0b111, 0b0000001) => Ok(SourceInstructionKind::REMU),
        _ => invalid("invalid op instruction"),
    }
}

fn decode_op_32(word: u32) -> Result<SourceInstructionKind, ProgramError> {
    match (funct3(word), funct7(word)) {
        (0b000, 0b0000000) => Ok(SourceInstructionKind::ADDW),
        (0b000, 0b0100000) => Ok(SourceInstructionKind::SUBW),
        (0b001, 0b0000000) => Ok(SourceInstructionKind::SLLW),
        (0b100, 0b0000001) => Ok(SourceInstructionKind::DIVW),
        (0b101, 0b0000000) => Ok(SourceInstructionKind::SRLW),
        (0b101, 0b0100000) => Ok(SourceInstructionKind::SRAW),
        (0b000, 0b0000001) => Ok(SourceInstructionKind::MULW),
        (0b101, 0b0000001) => Ok(SourceInstructionKind::DIVUW),
        (0b110, 0b0000001) => Ok(SourceInstructionKind::REMW),
        (0b111, 0b0000001) => Ok(SourceInstructionKind::REMUW),
        _ => invalid("invalid RV64 op-32 instruction"),
    }
}

fn decode_amo(word: u32) -> Result<SourceInstructionKind, ProgramError> {
    match (funct3(word), (word >> 27) & 0x1f) {
        (0b010, 0b00010) => Ok(SourceInstructionKind::LRW),
        (0b011, 0b00010) => Ok(SourceInstructionKind::LRD),
        (0b010, 0b00011) => Ok(SourceInstructionKind::SCW),
        (0b011, 0b00011) => Ok(SourceInstructionKind::SCD),
        (0b010, 0b00001) => Ok(SourceInstructionKind::AMOSWAPW),
        (0b011, 0b00001) => Ok(SourceInstructionKind::AMOSWAPD),
        (0b010, 0b00000) => Ok(SourceInstructionKind::AMOADDW),
        (0b011, 0b00000) => Ok(SourceInstructionKind::AMOADDD),
        (0b010, 0b01100) => Ok(SourceInstructionKind::AMOANDW),
        (0b011, 0b01100) => Ok(SourceInstructionKind::AMOANDD),
        (0b010, 0b01000) => Ok(SourceInstructionKind::AMOORW),
        (0b011, 0b01000) => Ok(SourceInstructionKind::AMOORD),
        (0b010, 0b00100) => Ok(SourceInstructionKind::AMOXORW),
        (0b011, 0b00100) => Ok(SourceInstructionKind::AMOXORD),
        (0b010, 0b10000) => Ok(SourceInstructionKind::AMOMINW),
        (0b011, 0b10000) => Ok(SourceInstructionKind::AMOMIND),
        (0b010, 0b10100) => Ok(SourceInstructionKind::AMOMAXW),
        (0b011, 0b10100) => Ok(SourceInstructionKind::AMOMAXD),
        (0b010, 0b11000) => Ok(SourceInstructionKind::AMOMINUW),
        (0b011, 0b11000) => Ok(SourceInstructionKind::AMOMINUD),
        (0b010, 0b11100) => Ok(SourceInstructionKind::AMOMAXUW),
        (0b011, 0b11100) => Ok(SourceInstructionKind::AMOMAXUD),
        _ => invalid("invalid atomic memory operation"),
    }
}

fn decode_system(word: u32) -> Result<SourceInstructionKind, ProgramError> {
    match (funct3(word), funct7(word), (word >> 20) & 0x1f) {
        (0, 0, 0) if word == 0x00000073 => Ok(SourceInstructionKind::ECALL),
        (0, 0, 1) if word == 0x00100073 => Ok(SourceInstructionKind::EBREAK),
        (0, 0x18, 2) if word == 0x30200073 => Ok(SourceInstructionKind::MRET),
        (1, _, _) => Ok(SourceInstructionKind::CSRRW),
        (2, _, _) => Ok(SourceInstructionKind::CSRRS),
        _ => invalid("unsupported system instruction"),
    }
}

fn decode_custom(word: u32) -> Result<SourceInstructionKind, ProgramError> {
    let funct3 = funct3(word);
    let funct7 = funct7(word);
    match (funct3, funct7) {
        (0b000, 0x00) => Ok(SourceInstructionKind::AdviceLB),
        (0b000, 0x01) => Ok(SourceInstructionKind::AdviceLH),
        (0b000, 0x02) => Ok(SourceInstructionKind::AdviceLW),
        (0b000, 0x03) => Ok(SourceInstructionKind::AdviceLD),
        (0b000, 0x04) => Ok(SourceInstructionKind::VirtualAdviceLen(
            jolt_riscv::instructions::VirtualAdviceLen(()),
        )),
        (0b000, 0x05) => Ok(SourceInstructionKind::VirtualRev8W(
            jolt_riscv::instructions::VirtualRev8W(()),
        )),
        (0b001, _) => Ok(SourceInstructionKind::VirtualAssertEQ),
        (0b010, _) => Ok(SourceInstructionKind::VirtualHostIO(
            jolt_riscv::instructions::VirtualHostIO(()),
        )),
        _ => invalid("invalid custom instruction"),
    }
}

#[cfg(feature = "field-inline")]
fn decode_field_inline(word: u32) -> Result<SourceInstructionKind, ProgramError> {
    match FieldInlineOp::from_funct3(funct3(word) as u8) {
        Some(FieldInlineOp::Add) => Ok(SourceInstructionKind::FIELD_ADD),
        Some(FieldInlineOp::Sub) => Ok(SourceInstructionKind::FIELD_SUB),
        Some(FieldInlineOp::Mul) => Ok(SourceInstructionKind::FIELD_MUL),
        Some(FieldInlineOp::Inv) => Ok(SourceInstructionKind::FIELD_INV),
        Some(FieldInlineOp::AssertEq) => Ok(SourceInstructionKind::FIELD_ASSERT_EQ),
        Some(FieldInlineOp::LoadFromX) => Ok(SourceInstructionKind::FIELD_LOAD_FROM_X),
        Some(FieldInlineOp::StoreToX) => Ok(SourceInstructionKind::FIELD_STORE_TO_X),
        Some(FieldInlineOp::LoadImm) => Ok(SourceInstructionKind::FIELD_LOAD_IMM),
        None => invalid("invalid field-inline funct3"),
    }
}

fn source_instruction(
    instruction_kind: SourceInstructionKind,
    word: u32,
    address: u64,
    is_compressed: bool,
) -> SourceInstruction {
    let inline =
        (instruction_kind == SourceInstructionKind::Inline).then(|| source_inline_key(word));
    SourceInstruction::new(
        instruction_kind,
        SourceInstructionRow {
            address: address as usize,
            operands: operands(instruction_kind, word),
            inline,
            is_compressed,
        },
    )
}

fn operands(instruction_kind: SourceInstructionKind, word: u32) -> NormalizedOperands {
    match instruction_kind {
        SourceInstructionKind::LUI | SourceInstructionKind::AUIPC => format_u_operands(word),
        SourceInstructionKind::JAL => format_j_operands(word),
        SourceInstructionKind::BEQ
        | SourceInstructionKind::BNE
        | SourceInstructionKind::BLT
        | SourceInstructionKind::BGE
        | SourceInstructionKind::BLTU
        | SourceInstructionKind::BGEU
        | SourceInstructionKind::VirtualAssertEQ => format_b_operands(word),
        SourceInstructionKind::SB
        | SourceInstructionKind::SH
        | SourceInstructionKind::SW
        | SourceInstructionKind::SD => format_s_operands(word),
        SourceInstructionKind::LB
        | SourceInstructionKind::LH
        | SourceInstructionKind::LW
        | SourceInstructionKind::LD
        | SourceInstructionKind::LBU
        | SourceInstructionKind::LHU
        | SourceInstructionKind::LWU => format_load_operands(word),
        SourceInstructionKind::LRW
        | SourceInstructionKind::LRD
        | SourceInstructionKind::SCW
        | SourceInstructionKind::SCD
        | SourceInstructionKind::AMOSWAPW
        | SourceInstructionKind::AMOSWAPD
        | SourceInstructionKind::AMOADDW
        | SourceInstructionKind::AMOADDD
        | SourceInstructionKind::AMOANDW
        | SourceInstructionKind::AMOANDD
        | SourceInstructionKind::AMOORW
        | SourceInstructionKind::AMOORD
        | SourceInstructionKind::AMOXORW
        | SourceInstructionKind::AMOXORD
        | SourceInstructionKind::AMOMINW
        | SourceInstructionKind::AMOMIND
        | SourceInstructionKind::AMOMAXW
        | SourceInstructionKind::AMOMAXD
        | SourceInstructionKind::AMOMINUW
        | SourceInstructionKind::AMOMINUD
        | SourceInstructionKind::AMOMAXUW
        | SourceInstructionKind::AMOMAXUD => format_r_operands(word),
        SourceInstructionKind::AdviceLB
        | SourceInstructionKind::AdviceLH
        | SourceInstructionKind::AdviceLW
        | SourceInstructionKind::AdviceLD => format_advice_load_operands(word),
        SourceInstructionKind::VirtualRev8W(jolt_riscv::instructions::VirtualRev8W(())) => {
            format_t_operands(word)
        }
        #[cfg(feature = "field-inline")]
        SourceInstructionKind::FIELD_ADD
        | SourceInstructionKind::FIELD_SUB
        | SourceInstructionKind::FIELD_MUL => format_r_operands(word),
        // FIELD_ASSERT_EQ has no destination register; decoding it with `rd: None`
        // keeps the bytecode operands consistent with the tracer's parsed shape and
        // avoids the rd=x0 virtual-register rewrite during expansion.
        #[cfg(feature = "field-inline")]
        SourceInstructionKind::FIELD_ASSERT_EQ => format_field_binary_no_rd_operands(word),
        #[cfg(feature = "field-inline")]
        SourceInstructionKind::FIELD_INV
        | SourceInstructionKind::FIELD_LOAD_FROM_X
        | SourceInstructionKind::FIELD_STORE_TO_X => format_field_unary_operands(word),
        #[cfg(feature = "field-inline")]
        SourceInstructionKind::FIELD_LOAD_IMM => format_field_load_imm_operands(word),
        SourceInstructionKind::Inline => format_inline_operands(word),
        SourceInstructionKind::ECALL
        | SourceInstructionKind::EBREAK
        | SourceInstructionKind::MRET => format_i_operands(word),
        SourceInstructionKind::FENCE
        | SourceInstructionKind::NoOp
        | SourceInstructionKind::Unimpl => NormalizedOperands::default(),
        _ => format_i_or_r_operands(instruction_kind, word),
    }
}

#[cfg(feature = "field-inline")]
fn format_field_unary_operands(word: u32) -> NormalizedOperands {
    NormalizedOperands {
        rd: Some(rd(word)),
        rs1: Some(rs1(word)),
        rs2: None,
        imm: 0,
    }
}

#[cfg(feature = "field-inline")]
fn format_field_binary_no_rd_operands(word: u32) -> NormalizedOperands {
    NormalizedOperands {
        rd: None,
        rs1: Some(rs1(word)),
        rs2: Some(rs2(word)),
        imm: 0,
    }
}

#[cfg(feature = "field-inline")]
fn format_field_load_imm_operands(word: u32) -> NormalizedOperands {
    NormalizedOperands {
        rd: Some(rd(word)),
        rs1: None,
        rs2: None,
        imm: i128::from((word >> 20) & 0xfff),
    }
}

fn format_i_or_r_operands(
    instruction_kind: SourceInstructionKind,
    word: u32,
) -> NormalizedOperands {
    if uses_r_format(instruction_kind) {
        format_r_operands(word)
    } else {
        format_i_operands(word)
    }
}

fn uses_r_format(instruction_kind: SourceInstructionKind) -> bool {
    matches!(
        instruction_kind,
        SourceInstructionKind::ADD
            | SourceInstructionKind::SUB
            | SourceInstructionKind::SLL
            | SourceInstructionKind::SLT
            | SourceInstructionKind::SLTU
            | SourceInstructionKind::XOR
            | SourceInstructionKind::SRL
            | SourceInstructionKind::SRA
            | SourceInstructionKind::OR
            | SourceInstructionKind::AND
            | SourceInstructionKind::MUL
            | SourceInstructionKind::MULH
            | SourceInstructionKind::MULHSU
            | SourceInstructionKind::MULHU
            | SourceInstructionKind::DIV
            | SourceInstructionKind::DIVU
            | SourceInstructionKind::REM
            | SourceInstructionKind::REMU
            | SourceInstructionKind::ADDW
            | SourceInstructionKind::SUBW
            | SourceInstructionKind::SLLW
            | SourceInstructionKind::DIVW
            | SourceInstructionKind::SRLW
            | SourceInstructionKind::SRAW
            | SourceInstructionKind::MULW
            | SourceInstructionKind::DIVUW
            | SourceInstructionKind::REMW
            | SourceInstructionKind::REMUW
    )
}

fn format_r_operands(word: u32) -> NormalizedOperands {
    NormalizedOperands {
        rd: Some(rd(word)),
        rs1: Some(rs1(word)),
        rs2: Some(rs2(word)),
        imm: 0,
    }
}

fn format_i_operands(word: u32) -> NormalizedOperands {
    NormalizedOperands {
        rd: Some(rd(word)),
        rs1: Some(rs1(word)),
        rs2: None,
        imm: sign_extend_u64(word >> 20, 12) as i128,
    }
}

fn format_load_operands(word: u32) -> NormalizedOperands {
    NormalizedOperands {
        rd: Some(rd(word)),
        rs1: Some(rs1(word)),
        rs2: None,
        imm: sign_extend_i64(word >> 20, 12) as i128,
    }
}

fn format_advice_load_operands(word: u32) -> NormalizedOperands {
    NormalizedOperands {
        rd: Some(rd(word)),
        rs1: None,
        rs2: None,
        imm: 0,
    }
}

fn format_t_operands(word: u32) -> NormalizedOperands {
    NormalizedOperands {
        rd: Some(rd(word)),
        rs1: Some(rs1(word)),
        rs2: None,
        imm: 0,
    }
}

fn format_s_operands(word: u32) -> NormalizedOperands {
    let raw = ((word >> 20) & 0xfe0) | ((word >> 7) & 0x1f);
    NormalizedOperands {
        rd: None,
        rs1: Some(rs1(word)),
        rs2: Some(rs2(word)),
        imm: sign_extend_i64(raw, 12) as i128,
    }
}

fn format_b_operands(word: u32) -> NormalizedOperands {
    let raw = ((word << 4) & 0x00000800)
        | ((word >> 20) & 0x000007e0)
        | ((word >> 7) & 0x0000001e)
        | sign_extension_mask(word, 0x80000000, 0xfffff000);
    NormalizedOperands {
        rd: None,
        rs1: Some(rs1(word)),
        rs2: Some(rs2(word)),
        imm: raw as i32 as i128,
    }
}

fn format_j_operands(word: u32) -> NormalizedOperands {
    let raw = sign_extension_mask(word, 0x80000000, 0xfff00000)
        | (word & 0x000ff000)
        | ((word & 0x00100000) >> 9)
        | ((word & 0x7fe00000) >> 20);
    NormalizedOperands {
        rd: Some(rd(word)),
        rs1: None,
        rs2: None,
        imm: (raw as i32 as i64 as u64) as i128,
    }
}

fn format_u_operands(word: u32) -> NormalizedOperands {
    NormalizedOperands {
        rd: Some(rd(word)),
        rs1: None,
        rs2: None,
        imm: ((word & 0xfffff000) as i32 as i64 as u64) as i128,
    }
}

fn format_inline_operands(word: u32) -> NormalizedOperands {
    NormalizedOperands {
        rd: Some(rd(word)),
        rs1: Some(rs1(word)),
        rs2: Some(rs2(word)),
        imm: 0,
    }
}

fn source_inline_key(word: u32) -> SourceInlineKey {
    SourceInlineKey {
        opcode: (word & 0x7f) as u8,
        funct3: funct3(word) as u8,
        funct7: funct7(word) as u8,
    }
}

fn rd(word: u32) -> u8 {
    ((word >> 7) & 0x1f) as u8
}

fn rs1(word: u32) -> u8 {
    ((word >> 15) & 0x1f) as u8
}

fn rs2(word: u32) -> u8 {
    ((word >> 20) & 0x1f) as u8
}

fn funct3(word: u32) -> u32 {
    (word >> 12) & 0x7
}

fn funct6(word: u32) -> u32 {
    (word >> 26) & 0x3f
}

fn funct7(word: u32) -> u32 {
    (word >> 25) & 0x7f
}

fn sign_extend_u64(value: u32, bits: u32) -> u64 {
    sign_extend_i64(value, bits) as u64
}

fn sign_extend_i64(value: u32, bits: u32) -> i64 {
    let shift = 32 - bits;
    ((value << shift) as i32 >> shift) as i64
}

fn sign_extension_mask(value: u32, bit: u32, mask: u32) -> u32 {
    if value & bit == bit {
        mask
    } else {
        0
    }
}

fn invalid<T>(message: &'static str) -> Result<T, ProgramError> {
    Err(ProgramError::MalformedImage(message))
}

#[cfg(test)]
#[expect(clippy::panic, reason = "decode tests fail with contextual errors")]
mod tests {
    use super::*;
    use jolt_riscv::RV64IMAC_JOLT;

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
        // garbage above the extracted width must not leak into the result
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
        // one-hot sweep over every branch immediate bit position
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
        assert_eq!(format_j_operands(j_word(703_710, 0)).imm, 703_710); // 0xABCDE
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

        // ecall with rd=1 is not the canonical 0x00000073 encoding
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
            ((0x3f << 25) | 0x5b, "invalid custom instruction"),
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
        // amoadd.w decodes but the A extension is absent from RV64IM_JOLT
        let word = (11 << 20) | (12 << 15) | (0b010 << 12) | (10 << 7) | 0x2f;
        match decode_instruction(word, 0x8000_0000, false, jolt_riscv::RV64IM_JOLT) {
            Err(ProgramError::IllegalSourceInstruction(kind)) => {
                assert_eq!(kind, SourceInstructionKind::AMOADDW);
            }
            Err(error) => panic!("expected IllegalSourceInstruction, got {error:?}"),
            Ok(_) => panic!("expected IllegalSourceInstruction, got Ok"),
        }
    }

    #[cfg(feature = "field-inline")]
    #[test]
    fn decodes_field_inline_source_rows_only_for_fr_on_profile() {
        let word = field_word(jolt_riscv::FieldInlineOp::Mul.funct3().into(), 1, 2, 3);
        let fr_off = decode_instruction(word, 0x8000_0000, false, RV64IMAC_JOLT);
        assert!(matches!(
            fr_off,
            Err(ProgramError::IllegalSourceInstruction(
                jolt_riscv::SourceInstruction::FieldMul(_)
            ))
        ));

        let fr_on = decode_instruction(
            word,
            0x8000_0000,
            false,
            jolt_riscv::RV64IMAC_JOLT_FIELD_INLINE,
        );
        let instruction = match fr_on {
            Ok(instruction) => instruction,
            Err(error) => panic!("field-inline decode failed: {error:?}"),
        };
        assert_eq!(instruction.kind(), SourceInstructionKind::FIELD_MUL);
        assert_eq!(instruction.row().operands.rd, Some(1));
        assert_eq!(instruction.row().operands.rs1, Some(2));
        assert_eq!(instruction.row().operands.rs2, Some(3));
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
}
