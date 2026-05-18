#![expect(
    clippy::unreadable_literal,
    reason = "RISC-V decode tables are easiest to audit in ISA bit-field widths"
)]

use jolt_riscv::{
    JoltInstructionKind, NormalizedInstruction, NormalizedOperands, SourceInstructionKind,
};

use crate::ProgramError;

pub fn decode_instruction(
    word: u32,
    address: u64,
    is_compressed: bool,
) -> Result<NormalizedInstruction, ProgramError> {
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
        0b0001011 => decode_field_op_or_inline(word)?,
        0b0101011 => SourceInstructionKind::Inline,
        0b1011011 => decode_custom(word)?,
        _ => return invalid("unknown RV64 opcode"),
    };

    Ok(normalized(kind, word, address, is_compressed))
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

/// Custom-0 opcode (0x0B) is shared between the BN254 Fr coprocessor instructions
/// (FieldOp/FieldAssertEq/FieldMov/FieldSLL*) and the general Inline mechanism.
/// FR ops are recognized by their funct7 = 0x40 (FieldOp/AssertEq/Mov) or 0x41
/// (FieldSLL* family); anything else falls back to Inline dispatch.
fn decode_field_op_or_inline(word: u32) -> Result<SourceInstructionKind, ProgramError> {
    match (funct7(word), funct3(word)) {
        (0x40, 0x02) => Ok(SourceInstructionKind::FieldMul),
        (0x40, 0x03) => Ok(SourceInstructionKind::FieldAdd),
        (0x40, 0x04) => Ok(SourceInstructionKind::FieldInv),
        (0x40, 0x05) => Ok(SourceInstructionKind::FieldSub),
        (0x40, 0x06) => Ok(SourceInstructionKind::FieldAssertEq),
        (0x40, 0x07) => Ok(SourceInstructionKind::FieldMov),
        (0x41, 0x00) => Ok(SourceInstructionKind::FieldSLL64),
        (0x41, 0x01) => Ok(SourceInstructionKind::FieldSLL128),
        (0x41, 0x02) => Ok(SourceInstructionKind::FieldSLL192),
        _ => Ok(SourceInstructionKind::Inline),
    }
}

fn decode_custom(word: u32) -> Result<SourceInstructionKind, ProgramError> {
    match funct3(word) {
        0b000 => Ok(SourceInstructionKind::VirtualRev8W),
        0b001 => Ok(SourceInstructionKind::VirtualAssertEQ),
        0b010 => Ok(SourceInstructionKind::VirtualHostIO),
        0b011 => Ok(SourceInstructionKind::AdviceLB),
        0b100 => Ok(SourceInstructionKind::AdviceLH),
        0b101 => Ok(SourceInstructionKind::AdviceLW),
        0b110 => Ok(SourceInstructionKind::AdviceLD),
        0b111 => Ok(SourceInstructionKind::VirtualAdviceLen),
        _ => invalid("invalid custom instruction"),
    }
}

fn normalized(
    instruction_kind: SourceInstructionKind,
    word: u32,
    address: u64,
    is_compressed: bool,
) -> NormalizedInstruction {
    let jolt_kind = instruction_kind.jolt_kind();
    NormalizedInstruction {
        instruction_kind: jolt_kind,
        address: address as usize,
        operands: operands(jolt_kind, word),
        virtual_sequence_remaining: None,
        is_first_in_sequence: false,
        is_compressed,
    }
}

fn operands(instruction_kind: JoltInstructionKind, word: u32) -> NormalizedOperands {
    match instruction_kind {
        JoltInstructionKind::LUI | JoltInstructionKind::AUIPC => format_u_operands(word),
        JoltInstructionKind::JAL => format_j_operands(word),
        JoltInstructionKind::BEQ
        | JoltInstructionKind::BNE
        | JoltInstructionKind::BLT
        | JoltInstructionKind::BGE
        | JoltInstructionKind::BLTU
        | JoltInstructionKind::BGEU
        | JoltInstructionKind::VirtualAssertEQ => format_b_operands(word),
        JoltInstructionKind::SB
        | JoltInstructionKind::SH
        | JoltInstructionKind::SW
        | JoltInstructionKind::SD => format_s_operands(word),
        JoltInstructionKind::LB
        | JoltInstructionKind::LH
        | JoltInstructionKind::LW
        | JoltInstructionKind::LD
        | JoltInstructionKind::LBU
        | JoltInstructionKind::LHU
        | JoltInstructionKind::LWU => format_load_operands(word),
        JoltInstructionKind::LRW
        | JoltInstructionKind::LRD
        | JoltInstructionKind::SCW
        | JoltInstructionKind::SCD
        | JoltInstructionKind::AMOSWAPW
        | JoltInstructionKind::AMOSWAPD
        | JoltInstructionKind::AMOADDW
        | JoltInstructionKind::AMOADDD
        | JoltInstructionKind::AMOANDW
        | JoltInstructionKind::AMOANDD
        | JoltInstructionKind::AMOORW
        | JoltInstructionKind::AMOORD
        | JoltInstructionKind::AMOXORW
        | JoltInstructionKind::AMOXORD
        | JoltInstructionKind::AMOMINW
        | JoltInstructionKind::AMOMIND
        | JoltInstructionKind::AMOMAXW
        | JoltInstructionKind::AMOMAXD
        | JoltInstructionKind::AMOMINUW
        | JoltInstructionKind::AMOMINUD
        | JoltInstructionKind::AMOMAXUW
        | JoltInstructionKind::AMOMAXUD => format_r_operands(word),
        JoltInstructionKind::AdviceLB
        | JoltInstructionKind::AdviceLH
        | JoltInstructionKind::AdviceLW
        | JoltInstructionKind::AdviceLD => format_advice_load_operands(word),
        JoltInstructionKind::Inline => format_inline_operands(word),
        JoltInstructionKind::ECALL | JoltInstructionKind::EBREAK | JoltInstructionKind::MRET => {
            format_i_operands(word)
        }
        JoltInstructionKind::FENCE | JoltInstructionKind::NoOp | JoltInstructionKind::Unimpl => {
            NormalizedOperands::default()
        }
        _ => format_i_or_r_operands(instruction_kind, word),
    }
}

fn format_i_or_r_operands(instruction_kind: JoltInstructionKind, word: u32) -> NormalizedOperands {
    if uses_r_format(instruction_kind) {
        format_r_operands(word)
    } else {
        format_i_operands(word)
    }
}

fn uses_r_format(instruction_kind: JoltInstructionKind) -> bool {
    matches!(
        instruction_kind,
        JoltInstructionKind::ADD
            | JoltInstructionKind::SUB
            | JoltInstructionKind::SLL
            | JoltInstructionKind::SLT
            | JoltInstructionKind::SLTU
            | JoltInstructionKind::XOR
            | JoltInstructionKind::SRL
            | JoltInstructionKind::SRA
            | JoltInstructionKind::OR
            | JoltInstructionKind::AND
            | JoltInstructionKind::MUL
            | JoltInstructionKind::MULH
            | JoltInstructionKind::MULHSU
            | JoltInstructionKind::MULHU
            | JoltInstructionKind::DIV
            | JoltInstructionKind::DIVU
            | JoltInstructionKind::REM
            | JoltInstructionKind::REMU
            | JoltInstructionKind::ADDW
            | JoltInstructionKind::SUBW
            | JoltInstructionKind::SLLW
            | JoltInstructionKind::DIVW
            | JoltInstructionKind::SRLW
            | JoltInstructionKind::SRAW
            | JoltInstructionKind::MULW
            | JoltInstructionKind::DIVUW
            | JoltInstructionKind::REMW
            | JoltInstructionKind::REMUW
            // BN254 Fr coprocessor ops — R-type with frd/frs1/frs2 (or rs1
            // for the bridge ops; rs2 is reserved/0).
            | JoltInstructionKind::FieldMul
            | JoltInstructionKind::FieldAdd
            | JoltInstructionKind::FieldSub
            | JoltInstructionKind::FieldInv
            | JoltInstructionKind::FieldAssertEq
            | JoltInstructionKind::FieldMov
            | JoltInstructionKind::FieldSLL64
            | JoltInstructionKind::FieldSLL128
            | JoltInstructionKind::FieldSLL192
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
        imm: (word as i32 as i64 as u64) as i128,
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
        imm: inline_metadata(word) as i128,
    }
}

fn inline_metadata(word: u32) -> u32 {
    let opcode = word & 0x7f;
    let funct3 = funct3(word);
    let funct7 = funct7(word);
    opcode | (funct3 << 7) | (funct7 << 10)
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
