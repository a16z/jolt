#![expect(
    clippy::unreadable_literal,
    reason = "RISC-V decode tables are easiest to audit in ISA bit-field widths"
)]

use jolt_riscv::{
    JoltInstructionKind, NormalizedInstruction, NormalizedOperands, RiscvInstructionKind,
};

use crate::ProgramError;

pub fn decode_instruction(
    word: u32,
    address: u64,
    is_compressed: bool,
) -> Result<NormalizedInstruction, ProgramError> {
    let opcode = word & 0x7f;
    let kind = match opcode {
        0b0110111 => RiscvInstructionKind::LUI,
        0b0010111 => RiscvInstructionKind::AUIPC,
        0b1101111 => RiscvInstructionKind::JAL,
        0b1100111 => match funct3(word) {
            0b000 => RiscvInstructionKind::JALR,
            _ => return invalid("invalid JALR funct3"),
        },
        0b1100011 => match funct3(word) {
            0b000 => RiscvInstructionKind::BEQ,
            0b001 => RiscvInstructionKind::BNE,
            0b100 => RiscvInstructionKind::BLT,
            0b101 => RiscvInstructionKind::BGE,
            0b110 => RiscvInstructionKind::BLTU,
            0b111 => RiscvInstructionKind::BGEU,
            _ => return invalid("invalid branch funct3"),
        },
        0b0000011 => match funct3(word) {
            0b000 => RiscvInstructionKind::LB,
            0b001 => RiscvInstructionKind::LH,
            0b010 => RiscvInstructionKind::LW,
            0b011 => RiscvInstructionKind::LD,
            0b100 => RiscvInstructionKind::LBU,
            0b101 => RiscvInstructionKind::LHU,
            0b110 => RiscvInstructionKind::LWU,
            _ => return invalid("invalid load funct3"),
        },
        0b0100011 => match funct3(word) {
            0b000 => RiscvInstructionKind::SB,
            0b001 => RiscvInstructionKind::SH,
            0b010 => RiscvInstructionKind::SW,
            0b011 => RiscvInstructionKind::SD,
            _ => return invalid("invalid store funct3"),
        },
        0b0010011 => decode_op_imm(word)?,
        0b0011011 => decode_op_imm_32(word)?,
        0b0110011 => decode_op(word)?,
        0b0111011 => decode_op_32(word)?,
        0b0001111 => RiscvInstructionKind::FENCE,
        0b0101111 => decode_amo(word)?,
        0b1110011 => decode_system(word)?,
        0b0001011 | 0b0101011 => RiscvInstructionKind::Inline,
        0b1011011 => decode_custom(word)?,
        _ => return invalid("unknown RV64 opcode"),
    };

    Ok(normalized(kind, word, address, is_compressed))
}

fn decode_op_imm(word: u32) -> Result<RiscvInstructionKind, ProgramError> {
    match funct3(word) {
        0b001 if funct6(word) == 0 => Ok(RiscvInstructionKind::SLLI),
        0b001 => invalid("invalid SLLI funct6"),
        0b101 if funct6(word) == 0b000000 => Ok(RiscvInstructionKind::SRLI),
        0b101 if funct6(word) == 0b010000 => Ok(RiscvInstructionKind::SRAI),
        0b101 => invalid("invalid shift-immediate funct6"),
        0b000 => Ok(RiscvInstructionKind::ADDI),
        0b010 => Ok(RiscvInstructionKind::SLTI),
        0b011 => Ok(RiscvInstructionKind::SLTIU),
        0b100 => Ok(RiscvInstructionKind::XORI),
        0b110 => Ok(RiscvInstructionKind::ORI),
        0b111 => Ok(RiscvInstructionKind::ANDI),
        _ => invalid("invalid op-imm funct3"),
    }
}

fn decode_op_imm_32(word: u32) -> Result<RiscvInstructionKind, ProgramError> {
    match (funct3(word), funct7(word)) {
        (0b000, _) => Ok(RiscvInstructionKind::ADDIW),
        (0b001, 0b0000000) => Ok(RiscvInstructionKind::SLLIW),
        (0b101, 0b0000000) => Ok(RiscvInstructionKind::SRLIW),
        (0b101, 0b0100000) => Ok(RiscvInstructionKind::SRAIW),
        _ => invalid("invalid RV64 op-imm-32 instruction"),
    }
}

fn decode_op(word: u32) -> Result<RiscvInstructionKind, ProgramError> {
    match (funct3(word), funct7(word)) {
        (0b000, 0b0000000) => Ok(RiscvInstructionKind::ADD),
        (0b000, 0b0100000) => Ok(RiscvInstructionKind::SUB),
        (0b001, 0b0000000) => Ok(RiscvInstructionKind::SLL),
        (0b010, 0b0000000) => Ok(RiscvInstructionKind::SLT),
        (0b011, 0b0000000) => Ok(RiscvInstructionKind::SLTU),
        (0b100, 0b0000000) => Ok(RiscvInstructionKind::XOR),
        (0b101, 0b0000000) => Ok(RiscvInstructionKind::SRL),
        (0b101, 0b0100000) => Ok(RiscvInstructionKind::SRA),
        (0b110, 0b0000000) => Ok(RiscvInstructionKind::OR),
        (0b111, 0b0000000) => Ok(RiscvInstructionKind::AND),
        (0b000, 0b0000001) => Ok(RiscvInstructionKind::MUL),
        (0b001, 0b0000001) => Ok(RiscvInstructionKind::MULH),
        (0b010, 0b0000001) => Ok(RiscvInstructionKind::MULHSU),
        (0b011, 0b0000001) => Ok(RiscvInstructionKind::MULHU),
        (0b100, 0b0000001) => Ok(RiscvInstructionKind::DIV),
        (0b101, 0b0000001) => Ok(RiscvInstructionKind::DIVU),
        (0b110, 0b0000001) => Ok(RiscvInstructionKind::REM),
        (0b111, 0b0000001) => Ok(RiscvInstructionKind::REMU),
        _ => invalid("invalid op instruction"),
    }
}

fn decode_op_32(word: u32) -> Result<RiscvInstructionKind, ProgramError> {
    match (funct3(word), funct7(word)) {
        (0b000, 0b0000000) => Ok(RiscvInstructionKind::ADDW),
        (0b000, 0b0100000) => Ok(RiscvInstructionKind::SUBW),
        (0b001, 0b0000000) => Ok(RiscvInstructionKind::SLLW),
        (0b100, 0b0000001) => Ok(RiscvInstructionKind::DIVW),
        (0b101, 0b0000000) => Ok(RiscvInstructionKind::SRLW),
        (0b101, 0b0100000) => Ok(RiscvInstructionKind::SRAW),
        (0b000, 0b0000001) => Ok(RiscvInstructionKind::MULW),
        (0b101, 0b0000001) => Ok(RiscvInstructionKind::DIVUW),
        (0b110, 0b0000001) => Ok(RiscvInstructionKind::REMW),
        (0b111, 0b0000001) => Ok(RiscvInstructionKind::REMUW),
        _ => invalid("invalid RV64 op-32 instruction"),
    }
}

fn decode_amo(word: u32) -> Result<RiscvInstructionKind, ProgramError> {
    match (funct3(word), (word >> 27) & 0x1f) {
        (0b010, 0b00010) => Ok(RiscvInstructionKind::LRW),
        (0b011, 0b00010) => Ok(RiscvInstructionKind::LRD),
        (0b010, 0b00011) => Ok(RiscvInstructionKind::SCW),
        (0b011, 0b00011) => Ok(RiscvInstructionKind::SCD),
        (0b010, 0b00001) => Ok(RiscvInstructionKind::AMOSWAPW),
        (0b011, 0b00001) => Ok(RiscvInstructionKind::AMOSWAPD),
        (0b010, 0b00000) => Ok(RiscvInstructionKind::AMOADDW),
        (0b011, 0b00000) => Ok(RiscvInstructionKind::AMOADDD),
        (0b010, 0b01100) => Ok(RiscvInstructionKind::AMOANDW),
        (0b011, 0b01100) => Ok(RiscvInstructionKind::AMOANDD),
        (0b010, 0b01000) => Ok(RiscvInstructionKind::AMOORW),
        (0b011, 0b01000) => Ok(RiscvInstructionKind::AMOORD),
        (0b010, 0b00100) => Ok(RiscvInstructionKind::AMOXORW),
        (0b011, 0b00100) => Ok(RiscvInstructionKind::AMOXORD),
        (0b010, 0b10000) => Ok(RiscvInstructionKind::AMOMINW),
        (0b011, 0b10000) => Ok(RiscvInstructionKind::AMOMIND),
        (0b010, 0b10100) => Ok(RiscvInstructionKind::AMOMAXW),
        (0b011, 0b10100) => Ok(RiscvInstructionKind::AMOMAXD),
        (0b010, 0b11000) => Ok(RiscvInstructionKind::AMOMINUW),
        (0b011, 0b11000) => Ok(RiscvInstructionKind::AMOMINUD),
        (0b010, 0b11100) => Ok(RiscvInstructionKind::AMOMAXUW),
        (0b011, 0b11100) => Ok(RiscvInstructionKind::AMOMAXUD),
        _ => invalid("invalid atomic memory operation"),
    }
}

fn decode_system(word: u32) -> Result<RiscvInstructionKind, ProgramError> {
    match (funct3(word), funct7(word), (word >> 20) & 0x1f) {
        (0, 0, 0) if word == 0x00000073 => Ok(RiscvInstructionKind::ECALL),
        (0, 0, 1) if word == 0x00100073 => Ok(RiscvInstructionKind::EBREAK),
        (0, 0x18, 2) if word == 0x30200073 => Ok(RiscvInstructionKind::MRET),
        (1, _, _) => Ok(RiscvInstructionKind::CSRRW),
        (2, _, _) => Ok(RiscvInstructionKind::CSRRS),
        _ => invalid("unsupported system instruction"),
    }
}

fn decode_custom(word: u32) -> Result<RiscvInstructionKind, ProgramError> {
    match funct3(word) {
        0b000 => Ok(RiscvInstructionKind::VirtualRev8W),
        0b001 => Ok(RiscvInstructionKind::VirtualAssertEQ),
        0b010 => Ok(RiscvInstructionKind::VirtualHostIO),
        0b011 => Ok(RiscvInstructionKind::AdviceLB),
        0b100 => Ok(RiscvInstructionKind::AdviceLH),
        0b101 => Ok(RiscvInstructionKind::AdviceLW),
        0b110 => Ok(RiscvInstructionKind::AdviceLD),
        0b111 => Ok(RiscvInstructionKind::VirtualAdviceLen),
        _ => invalid("invalid custom instruction"),
    }
}

fn normalized(
    instruction_kind: RiscvInstructionKind,
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
