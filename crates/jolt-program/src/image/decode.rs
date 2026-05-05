#![expect(
    clippy::unreadable_literal,
    reason = "RISC-V decode tables are easiest to audit in ISA bit-field widths"
)]

use jolt_riscv::{InstructionKind, NormalizedInstruction, NormalizedOperands};

use crate::ProgramError;

pub fn decode_instruction(
    word: u32,
    address: u64,
    is_compressed: bool,
) -> Result<NormalizedInstruction, ProgramError> {
    let opcode = word & 0x7f;
    let kind = match opcode {
        0b0110111 => InstructionKind::LUI,
        0b0010111 => InstructionKind::AUIPC,
        0b1101111 => InstructionKind::JAL,
        0b1100111 => match funct3(word) {
            0b000 => InstructionKind::JALR,
            _ => return invalid("invalid JALR funct3"),
        },
        0b1100011 => match funct3(word) {
            0b000 => InstructionKind::BEQ,
            0b001 => InstructionKind::BNE,
            0b100 => InstructionKind::BLT,
            0b101 => InstructionKind::BGE,
            0b110 => InstructionKind::BLTU,
            0b111 => InstructionKind::BGEU,
            _ => return invalid("invalid branch funct3"),
        },
        0b0000011 => match funct3(word) {
            0b000 => InstructionKind::LB,
            0b001 => InstructionKind::LH,
            0b010 => InstructionKind::LW,
            0b011 => InstructionKind::LD,
            0b100 => InstructionKind::LBU,
            0b101 => InstructionKind::LHU,
            0b110 => InstructionKind::LWU,
            _ => return invalid("invalid load funct3"),
        },
        0b0100011 => match funct3(word) {
            0b000 => InstructionKind::SB,
            0b001 => InstructionKind::SH,
            0b010 => InstructionKind::SW,
            0b011 => InstructionKind::SD,
            _ => return invalid("invalid store funct3"),
        },
        0b0010011 => decode_op_imm(word)?,
        0b0011011 => decode_op_imm_32(word)?,
        0b0110011 => decode_op(word)?,
        0b0111011 => decode_op_32(word)?,
        0b0001111 => InstructionKind::FENCE,
        0b0101111 => decode_amo(word)?,
        0b1110011 => decode_system(word)?,
        0b0001011 | 0b0101011 => InstructionKind::Inline,
        0b1011011 => decode_custom(word)?,
        _ => return invalid("unknown RV64 opcode"),
    };

    Ok(normalized(kind, word, address, is_compressed))
}

fn decode_op_imm(word: u32) -> Result<InstructionKind, ProgramError> {
    match funct3(word) {
        0b001 if funct6(word) == 0 => Ok(InstructionKind::SLLI),
        0b001 => invalid("invalid SLLI funct6"),
        0b101 if funct6(word) == 0b000000 => Ok(InstructionKind::SRLI),
        0b101 if funct6(word) == 0b010000 => Ok(InstructionKind::SRAI),
        0b101 => invalid("invalid shift-immediate funct6"),
        0b000 => Ok(InstructionKind::ADDI),
        0b010 => Ok(InstructionKind::SLTI),
        0b011 => Ok(InstructionKind::SLTIU),
        0b100 => Ok(InstructionKind::XORI),
        0b110 => Ok(InstructionKind::ORI),
        0b111 => Ok(InstructionKind::ANDI),
        _ => invalid("invalid op-imm funct3"),
    }
}

fn decode_op_imm_32(word: u32) -> Result<InstructionKind, ProgramError> {
    match (funct3(word), funct7(word)) {
        (0b000, _) => Ok(InstructionKind::ADDIW),
        (0b001, 0b0000000) => Ok(InstructionKind::SLLIW),
        (0b101, 0b0000000) => Ok(InstructionKind::SRLIW),
        (0b101, 0b0100000) => Ok(InstructionKind::SRAIW),
        _ => invalid("invalid RV64 op-imm-32 instruction"),
    }
}

fn decode_op(word: u32) -> Result<InstructionKind, ProgramError> {
    match (funct3(word), funct7(word)) {
        (0b000, 0b0000000) => Ok(InstructionKind::ADD),
        (0b000, 0b0100000) => Ok(InstructionKind::SUB),
        (0b001, 0b0000000) => Ok(InstructionKind::SLL),
        (0b010, 0b0000000) => Ok(InstructionKind::SLT),
        (0b011, 0b0000000) => Ok(InstructionKind::SLTU),
        (0b100, 0b0000000) => Ok(InstructionKind::XOR),
        (0b101, 0b0000000) => Ok(InstructionKind::SRL),
        (0b101, 0b0100000) => Ok(InstructionKind::SRA),
        (0b110, 0b0000000) => Ok(InstructionKind::OR),
        (0b111, 0b0000000) => Ok(InstructionKind::AND),
        (0b000, 0b0000001) => Ok(InstructionKind::MUL),
        (0b001, 0b0000001) => Ok(InstructionKind::MULH),
        (0b010, 0b0000001) => Ok(InstructionKind::MULHSU),
        (0b011, 0b0000001) => Ok(InstructionKind::MULHU),
        (0b100, 0b0000001) => Ok(InstructionKind::DIV),
        (0b101, 0b0000001) => Ok(InstructionKind::DIVU),
        (0b110, 0b0000001) => Ok(InstructionKind::REM),
        (0b111, 0b0000001) => Ok(InstructionKind::REMU),
        _ => invalid("invalid op instruction"),
    }
}

fn decode_op_32(word: u32) -> Result<InstructionKind, ProgramError> {
    match (funct3(word), funct7(word)) {
        (0b000, 0b0000000) => Ok(InstructionKind::ADDW),
        (0b000, 0b0100000) => Ok(InstructionKind::SUBW),
        (0b001, 0b0000000) => Ok(InstructionKind::SLLW),
        (0b100, 0b0000001) => Ok(InstructionKind::DIVW),
        (0b101, 0b0000000) => Ok(InstructionKind::SRLW),
        (0b101, 0b0100000) => Ok(InstructionKind::SRAW),
        (0b000, 0b0000001) => Ok(InstructionKind::MULW),
        (0b101, 0b0000001) => Ok(InstructionKind::DIVUW),
        (0b110, 0b0000001) => Ok(InstructionKind::REMW),
        (0b111, 0b0000001) => Ok(InstructionKind::REMUW),
        _ => invalid("invalid RV64 op-32 instruction"),
    }
}

fn decode_amo(word: u32) -> Result<InstructionKind, ProgramError> {
    match (funct3(word), (word >> 27) & 0x1f) {
        (0b010, 0b00010) => Ok(InstructionKind::LRW),
        (0b011, 0b00010) => Ok(InstructionKind::LRD),
        (0b010, 0b00011) => Ok(InstructionKind::SCW),
        (0b011, 0b00011) => Ok(InstructionKind::SCD),
        (0b010, 0b00001) => Ok(InstructionKind::AMOSWAPW),
        (0b011, 0b00001) => Ok(InstructionKind::AMOSWAPD),
        (0b010, 0b00000) => Ok(InstructionKind::AMOADDW),
        (0b011, 0b00000) => Ok(InstructionKind::AMOADDD),
        (0b010, 0b01100) => Ok(InstructionKind::AMOANDW),
        (0b011, 0b01100) => Ok(InstructionKind::AMOANDD),
        (0b010, 0b01000) => Ok(InstructionKind::AMOORW),
        (0b011, 0b01000) => Ok(InstructionKind::AMOORD),
        (0b010, 0b00100) => Ok(InstructionKind::AMOXORW),
        (0b011, 0b00100) => Ok(InstructionKind::AMOXORD),
        (0b010, 0b10000) => Ok(InstructionKind::AMOMINW),
        (0b011, 0b10000) => Ok(InstructionKind::AMOMIND),
        (0b010, 0b10100) => Ok(InstructionKind::AMOMAXW),
        (0b011, 0b10100) => Ok(InstructionKind::AMOMAXD),
        (0b010, 0b11000) => Ok(InstructionKind::AMOMINUW),
        (0b011, 0b11000) => Ok(InstructionKind::AMOMINUD),
        (0b010, 0b11100) => Ok(InstructionKind::AMOMAXUW),
        (0b011, 0b11100) => Ok(InstructionKind::AMOMAXUD),
        _ => invalid("invalid atomic memory operation"),
    }
}

fn decode_system(word: u32) -> Result<InstructionKind, ProgramError> {
    match (funct3(word), funct7(word), (word >> 20) & 0x1f) {
        (0, 0, 0) if word == 0x00000073 => Ok(InstructionKind::ECALL),
        (0, 0, 1) if word == 0x00100073 => Ok(InstructionKind::EBREAK),
        (0, 0x18, 2) if word == 0x30200073 => Ok(InstructionKind::MRET),
        (1, _, _) => Ok(InstructionKind::CSRRW),
        (2, _, _) => Ok(InstructionKind::CSRRS),
        _ => invalid("unsupported system instruction"),
    }
}

fn decode_custom(word: u32) -> Result<InstructionKind, ProgramError> {
    match funct3(word) {
        0b000 => Ok(InstructionKind::VirtualRev8W),
        0b001 => Ok(InstructionKind::VirtualAssertEQ),
        0b010 => Ok(InstructionKind::VirtualHostIO),
        0b011 => Ok(InstructionKind::AdviceLB),
        0b100 => Ok(InstructionKind::AdviceLH),
        0b101 => Ok(InstructionKind::AdviceLW),
        0b110 => Ok(InstructionKind::AdviceLD),
        0b111 => Ok(InstructionKind::VirtualAdviceLen),
        _ => invalid("invalid custom instruction"),
    }
}

fn normalized(
    instruction_kind: InstructionKind,
    word: u32,
    address: u64,
    is_compressed: bool,
) -> NormalizedInstruction {
    NormalizedInstruction {
        instruction_kind,
        address: address as usize,
        operands: operands(instruction_kind, word),
        virtual_sequence_remaining: None,
        is_first_in_sequence: false,
        is_compressed,
    }
}

fn operands(instruction_kind: InstructionKind, word: u32) -> NormalizedOperands {
    match instruction_kind {
        InstructionKind::LUI | InstructionKind::AUIPC => format_u_operands(word),
        InstructionKind::JAL => format_j_operands(word),
        InstructionKind::BEQ
        | InstructionKind::BNE
        | InstructionKind::BLT
        | InstructionKind::BGE
        | InstructionKind::BLTU
        | InstructionKind::BGEU
        | InstructionKind::VirtualAssertEQ => format_b_operands(word),
        InstructionKind::SB | InstructionKind::SH | InstructionKind::SW | InstructionKind::SD => {
            format_s_operands(word)
        }
        InstructionKind::LB
        | InstructionKind::LH
        | InstructionKind::LW
        | InstructionKind::LD
        | InstructionKind::LBU
        | InstructionKind::LHU
        | InstructionKind::LWU => format_load_operands(word),
        InstructionKind::LRW
        | InstructionKind::LRD
        | InstructionKind::SCW
        | InstructionKind::SCD
        | InstructionKind::AMOSWAPW
        | InstructionKind::AMOSWAPD
        | InstructionKind::AMOADDW
        | InstructionKind::AMOADDD
        | InstructionKind::AMOANDW
        | InstructionKind::AMOANDD
        | InstructionKind::AMOORW
        | InstructionKind::AMOORD
        | InstructionKind::AMOXORW
        | InstructionKind::AMOXORD
        | InstructionKind::AMOMINW
        | InstructionKind::AMOMIND
        | InstructionKind::AMOMAXW
        | InstructionKind::AMOMAXD
        | InstructionKind::AMOMINUW
        | InstructionKind::AMOMINUD
        | InstructionKind::AMOMAXUW
        | InstructionKind::AMOMAXUD => format_r_operands(word),
        InstructionKind::AdviceLB
        | InstructionKind::AdviceLH
        | InstructionKind::AdviceLW
        | InstructionKind::AdviceLD => format_advice_load_operands(word),
        InstructionKind::Inline => format_inline_operands(word),
        InstructionKind::ECALL | InstructionKind::EBREAK | InstructionKind::MRET => {
            format_i_operands(word)
        }
        InstructionKind::FENCE | InstructionKind::NoOp | InstructionKind::Unimpl => {
            NormalizedOperands::default()
        }
        _ => format_i_or_r_operands(instruction_kind, word),
    }
}

fn format_i_or_r_operands(instruction_kind: InstructionKind, word: u32) -> NormalizedOperands {
    if uses_r_format(instruction_kind) {
        format_r_operands(word)
    } else {
        format_i_operands(word)
    }
}

fn uses_r_format(instruction_kind: InstructionKind) -> bool {
    matches!(
        instruction_kind,
        InstructionKind::ADD
            | InstructionKind::SUB
            | InstructionKind::SLL
            | InstructionKind::SLT
            | InstructionKind::SLTU
            | InstructionKind::XOR
            | InstructionKind::SRL
            | InstructionKind::SRA
            | InstructionKind::OR
            | InstructionKind::AND
            | InstructionKind::MUL
            | InstructionKind::MULH
            | InstructionKind::MULHSU
            | InstructionKind::MULHU
            | InstructionKind::DIV
            | InstructionKind::DIVU
            | InstructionKind::REM
            | InstructionKind::REMU
            | InstructionKind::ADDW
            | InstructionKind::SUBW
            | InstructionKind::SLLW
            | InstructionKind::DIVW
            | InstructionKind::SRLW
            | InstructionKind::SRAW
            | InstructionKind::MULW
            | InstructionKind::DIVUW
            | InstructionKind::REMW
            | InstructionKind::REMUW
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
