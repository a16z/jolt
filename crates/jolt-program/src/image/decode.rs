#![expect(
    clippy::unreadable_literal,
    reason = "RISC-V decode tables are easiest to audit in ISA bit-field widths"
)]

use jolt_riscv::{
    NormalizedOperands, SourceInline, SourceInstruction, SourceInstructionKind, SourceRow,
    RV64IMAC_JOLT,
};

use crate::ProgramError;

pub fn decode_instruction(
    word: u32,
    address: u64,
    is_compressed: bool,
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
        _ => return invalid("unknown RV64 opcode"),
    };

    if !RV64IMAC_JOLT.supports_source(kind) {
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

fn source_instruction(
    instruction_kind: SourceInstructionKind,
    word: u32,
    address: u64,
    is_compressed: bool,
) -> SourceInstruction {
    let inline = (instruction_kind == SourceInstructionKind::Inline).then(|| source_inline(word));
    SourceInstruction::new(
        instruction_kind,
        SourceRow {
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
        imm: 0,
    }
}

fn source_inline(word: u32) -> SourceInline {
    SourceInline {
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
