use serde::{Deserialize, Serialize};
use std::str::FromStr;
use strum::EnumCount;
use strum_macros::{EnumCount as EnumCountMacro, EnumIter, FromRepr};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ELFInstruction {
    pub address: u64,
    pub opcode: RV32IM,
    pub rs1: Option<u64>,
    pub rs2: Option<u64>,
    pub rd: Option<u64>,
    pub imm: Option<i64>,
    /// If this instruction is part of a "virtual sequence" (see Section 6.2 of the
    /// Jolt paper), then this contains the number of virtual instructions after this
    /// one in the sequence. I.e. if this is the last instruction in the sequence,
    /// `virtual_sequence_remaining` will be Some(0); if this is the penultimate instruction
    /// in the sequence, `virtual_sequence_remaining` will be Some(1); etc.
    pub virtual_sequence_remaining: Option<usize>,
}

/// Boolean flags used in Jolt's R1CS constraints (`opflags` in the Jolt paper).
/// Note that the flags below deviate slightly from those described in Appendix A.1
/// of the Jolt paper.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Hash, Ord, EnumCountMacro, EnumIter, Default,
)]
pub enum CircuitFlags {
    #[default] // Need a default so that we can derive EnumIter on `JoltR1CSInputs`
    /// 1 if the first lookup operand is the program counter; 0 otherwise (first lookup operand is RS1 value).
    LeftOperandIsPC,
    /// 1 if the second lookup operand is `imm`; 0 otherwise (second lookup operand is RS2 value).
    RightOperandIsImm,
    /// 1 if the instruction is a load (i.e. `LW`)
    Load,
    /// 1 if the instruction is a store (i.e. `SW`)
    Store,
    /// 1 if the instruction is a jump (i.e. `JAL`, `JALR`)
    Jump,
    /// 1 if the instruction is a branch (i.e. `BEQ`, `BNE`, etc.)
    Branch,
    /// 1 if the lookup output is to be stored in `rd` at the end of the step.
    WriteLookupOutputToRD,
    /// Indicates whether the instruction performs a concat-type lookup.
    ConcatLookupQueryChunks,
    /// 1 if the instruction is "virtual", as defined in Section 6.1 of the Jolt paper.
    Virtual,
    /// 1 if the instruction is an assert, as defined in Section 6.1.1 of the Jolt paper.
    Assert,
    /// Used in virtual sequences; the program counter should be the same for the full sequence.
    DoNotUpdatePC,
}
pub const NUM_CIRCUIT_FLAGS: usize = CircuitFlags::COUNT;

impl ELFInstruction {
    #[rustfmt::skip]
    pub fn to_circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];

        flags[CircuitFlags::LeftOperandIsPC as usize] = matches!(
            self.opcode,
            RV32IM::JAL | RV32IM::LUI | RV32IM::AUIPC,
        );

        flags[CircuitFlags::RightOperandIsImm as usize] = matches!(
            self.opcode,
            RV32IM::ADDI
            | RV32IM::XORI
            | RV32IM::ORI
            | RV32IM::ANDI
            | RV32IM::SLLI
            | RV32IM::SRLI
            | RV32IM::SRAI
            | RV32IM::SLTI
            | RV32IM::SLTIU
            | RV32IM::AUIPC
            | RV32IM::JAL
            | RV32IM::JALR
            | RV32IM::SW
            | RV32IM::LW
            | RV32IM::VIRTUAL_ASSERT_HALFWORD_ALIGNMENT,
        );

        flags[CircuitFlags::Load as usize] = matches!(
            self.opcode,
            RV32IM::LW,
        );

        flags[CircuitFlags::Store as usize] = matches!(
            self.opcode,
            RV32IM::SW,
        );

        flags[CircuitFlags::Jump as usize] = matches!(
            self.opcode,
            RV32IM::JAL | RV32IM::JALR,
        );

        flags[CircuitFlags::Branch as usize] = matches!(
            self.opcode,
            RV32IM::BEQ | RV32IM::BNE | RV32IM::BLT | RV32IM::BGE | RV32IM::BLTU | RV32IM::BGEU,
        );

        // Stores, branches, jumps, and asserts do not store the lookup output to rd (they may update rd in other ways)
        flags[CircuitFlags::WriteLookupOutputToRD as usize] = !matches!(
            self.opcode,
            RV32IM::SW
            | RV32IM::LW
            | RV32IM::BEQ
            | RV32IM::BNE
            | RV32IM::BLT
            | RV32IM::BGE
            | RV32IM::BLTU
            | RV32IM::BGEU
            | RV32IM::JAL
            | RV32IM::JALR
            | RV32IM::LUI
            | RV32IM::VIRTUAL_ASSERT_EQ
            | RV32IM::VIRTUAL_ASSERT_LTE
            | RV32IM::VIRTUAL_ASSERT_VALID_DIV0
            | RV32IM::VIRTUAL_ASSERT_VALID_SIGNED_REMAINDER
            | RV32IM::VIRTUAL_ASSERT_VALID_UNSIGNED_REMAINDER
            | RV32IM::VIRTUAL_ASSERT_HALFWORD_ALIGNMENT
        );

        flags[CircuitFlags::ConcatLookupQueryChunks as usize] = matches!(
            self.opcode,
            RV32IM::XOR
            | RV32IM::XORI
            | RV32IM::OR
            | RV32IM::ORI
            | RV32IM::AND
            | RV32IM::ANDI
            | RV32IM::SLL
            | RV32IM::SRL
            | RV32IM::SRA
            | RV32IM::SLLI
            | RV32IM::SRLI
            | RV32IM::SRAI
            | RV32IM::SLT
            | RV32IM::SLTU
            | RV32IM::SLTI
            | RV32IM::SLTIU
            | RV32IM::BEQ
            | RV32IM::BNE
            | RV32IM::BLT
            | RV32IM::BGE
            | RV32IM::BLTU
            | RV32IM::BGEU
            | RV32IM::VIRTUAL_ASSERT_EQ
            | RV32IM::VIRTUAL_ASSERT_LTE
            | RV32IM::VIRTUAL_ASSERT_VALID_SIGNED_REMAINDER
            | RV32IM::VIRTUAL_ASSERT_VALID_UNSIGNED_REMAINDER
            | RV32IM::VIRTUAL_ASSERT_VALID_DIV0,
        );

        flags[CircuitFlags::Virtual as usize] = self.virtual_sequence_remaining.is_some();

        flags[CircuitFlags::Assert as usize] = matches!(self.opcode,
            RV32IM::VIRTUAL_ASSERT_EQ                        |
            RV32IM::VIRTUAL_ASSERT_LTE                       |
            RV32IM::VIRTUAL_ASSERT_HALFWORD_ALIGNMENT        |
            RV32IM::VIRTUAL_ASSERT_VALID_SIGNED_REMAINDER    |
            RV32IM::VIRTUAL_ASSERT_VALID_UNSIGNED_REMAINDER  |
            RV32IM::VIRTUAL_ASSERT_VALID_DIV0
        );

        // All instructions in virtual sequence are mapped from the same
        // ELF address. Thus if an instruction is virtual (and not the last one
        // in its sequence), then we should *not* update the PC.
        flags[CircuitFlags::DoNotUpdatePC as usize] = match self.virtual_sequence_remaining {
            Some(i) => i != 0,
            None => false
        };

        flags
    }
}

// Reference: https://www.cs.sfu.ca/~ashriram/Courses/CS295/assets/notebooks/RISCV/RISCV_CARD.pdf
#[derive(Debug, PartialEq, Eq, Clone, Copy, FromRepr, Serialize, Deserialize, Hash)]
#[repr(u8)]
#[allow(non_camel_case_types)]
pub enum RV32IM {
    ADD,
    SUB,
    XOR,
    OR,
    AND,
    SLL,
    SRL,
    SRA,
    SLT,
    SLTU,
    ADDI,
    XORI,
    ORI,
    ANDI,
    SLLI,
    SRLI,
    SRAI,
    SLTI,
    SLTIU,
    LB,
    LH,
    LW,
    LBU,
    LHU,
    SB,
    SH,
    SW,
    BEQ,
    BNE,
    BLT,
    BGE,
    BLTU,
    BGEU,
    JAL,
    JALR,
    LUI,
    AUIPC,
    ECALL,
    EBREAK,
    MUL,
    MULH,
    MULHU,
    MULHSU,
    MULU,
    DIV,
    DIVU,
    REM,
    REMU,
    FENCE,
    UNIMPL,
    // Virtual instructions
    VIRTUAL_MOVSIGN,
    VIRTUAL_MOVE,
    VIRTUAL_ADVICE,
    VIRTUAL_ASSERT_LTE,
    VIRTUAL_ASSERT_VALID_UNSIGNED_REMAINDER,
    VIRTUAL_ASSERT_VALID_SIGNED_REMAINDER,
    VIRTUAL_ASSERT_EQ,
    VIRTUAL_ASSERT_VALID_DIV0,
    VIRTUAL_ASSERT_HALFWORD_ALIGNMENT,
    VIRTUAL_POW2,
    VIRTUAL_POW2I,
    VIRTUAL_SRA_PAD,
    VIRTUAL_SRA_PADI,
}

impl FromStr for RV32IM {
    type Err = String;

    fn from_str(s: &str) -> Result<RV32IM, String> {
        match s {
            "ADD" => Ok(Self::ADD),
            "SUB" => Ok(Self::SUB),
            "XOR" => Ok(Self::XOR),
            "OR" => Ok(Self::OR),
            "AND" => Ok(Self::AND),
            "SLL" => Ok(Self::SLL),
            "SRL" => Ok(Self::SRL),
            "SRA" => Ok(Self::SRA),
            "SLT" => Ok(Self::SLT),
            "SLTU" => Ok(Self::SLTU),
            "ADDI" => Ok(Self::ADDI),
            "XORI" => Ok(Self::XORI),
            "ORI" => Ok(Self::ORI),
            "ANDI" => Ok(Self::ANDI),
            "SLLI" => Ok(Self::SLLI),
            "SRLI" => Ok(Self::SRLI),
            "SRAI" => Ok(Self::SRAI),
            "SLTI" => Ok(Self::SLTI),
            "SLTIU" => Ok(Self::SLTIU),
            "LB" => Ok(Self::LB),
            "LH" => Ok(Self::LH),
            "LW" => Ok(Self::LW),
            "LBU" => Ok(Self::LBU),
            "LHU" => Ok(Self::LHU),
            "SB" => Ok(Self::SB),
            "SH" => Ok(Self::SH),
            "SW" => Ok(Self::SW),
            "BEQ" => Ok(Self::BEQ),
            "BNE" => Ok(Self::BNE),
            "BLT" => Ok(Self::BLT),
            "BGE" => Ok(Self::BGE),
            "BLTU" => Ok(Self::BLTU),
            "BGEU" => Ok(Self::BGEU),
            "JAL" => Ok(Self::JAL),
            "JALR" => Ok(Self::JALR),
            "LUI" => Ok(Self::LUI),
            "AUIPC" => Ok(Self::AUIPC),
            "ECALL" => Ok(Self::ECALL),
            "EBREAK" => Ok(Self::EBREAK),
            "MUL" => Ok(Self::MUL),
            "MULH" => Ok(Self::MULH),
            "MULHU" => Ok(Self::MULHU),
            "MULHSU" => Ok(Self::MULHSU),
            "MULU" => Ok(Self::MULU),
            "DIV" => Ok(Self::DIV),
            "DIVU" => Ok(Self::DIVU),
            "REM" => Ok(Self::REM),
            "REMU" => Ok(Self::REMU),
            "FENCE" => Ok(Self::FENCE),
            "UNIMPL" => Ok(Self::UNIMPL),
            _ => Err("Could not match instruction to RV32IM set.".to_string()),
        }
    }
}
