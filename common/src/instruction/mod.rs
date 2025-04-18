use add::ADD;
use addi::ADDI;
use and::AND;
use andi::ANDI;
use auipc::AUIPC;
use beq::BEQ;
use bge::BGE;
use bgeu::BGEU;
use blt::BLT;
use bltu::BLTU;
use bne::BNE;
use div::DIV;
use divu::DIVU;
use format::InstructionFormat;
use jal::JAL;
use jalr::JALR;
use lb::LB;
use lbu::LBU;
use lh::LH;
use lhu::LHU;
use lui::LUI;
use lw::LW;
use mul::MUL;
use mulh::MULH;
use mulhsu::MULHSU;
use mulhu::MULHU;
use mulu::MULU;
use or::OR;
use ori::ORI;
use rem::REM;
use remu::REMU;
use sb::SB;
use serde::{Deserialize, Serialize};
use sh::SH;
use sll::SLL;
use slli::SLLI;
use slt::SLT;
use slti::SLTI;
use sltiu::SLTIU;
use sltu::SLTU;
use sra::SRA;
use srai::SRAI;
use srl::SRL;
use srli::SRLI;
use std::str::FromStr;
use strum::EnumCount;
use strum_macros::{EnumCount as EnumCountMacro, EnumIter, FromRepr};
use sub::SUB;
use sw::SW;
use xor::XOR;
use xori::XORI;

pub mod format;

pub mod add;
pub mod addi;
pub mod and;
pub mod andi;
pub mod auipc;
pub mod beq;
pub mod bge;
pub mod bgeu;
pub mod blt;
pub mod bltu;
pub mod bne;
pub mod div;
pub mod divu;
pub mod jal;
pub mod jalr;
pub mod lb;
pub mod lbu;
pub mod lh;
pub mod lhu;
pub mod lui;
pub mod lw;
pub mod mul;
pub mod mulh;
pub mod mulhsu;
pub mod mulhu;
pub mod mulu;
pub mod or;
pub mod ori;
pub mod rem;
pub mod remu;
pub mod sb;
pub mod sh;
pub mod sll;
pub mod slli;
pub mod slt;
pub mod slti;
pub mod sltiu;
pub mod sltu;
pub mod sra;
pub mod srai;
pub mod srl;
pub mod srli;
pub mod sub;
pub mod sw;
pub mod xor;
pub mod xori;

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
            | RV32IM::VIRTUAL_POW2I
            | RV32IM::VIRTUAL_SHIFT_RIGHT_BITMASKI
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
    VIRTUAL_SHIFT_RIGHT_BITMASK,
    VIRTUAL_SHIFT_RIGHT_BITMASKI,
    VIRTUAL_SRL,
    VIRTUAL_SRA,
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

trait RISCVInstruction {
    type Format: InstructionFormat;
    type RAMAccess: Default;

    fn to_raw(&self) -> Self::Format;
    fn new(word: u32, address: u64) -> Self;
}

struct RISCVCycle<T: RISCVInstruction> {
    instruction: T,
    register_state: <T::Format as InstructionFormat>::RegisterState,
    memory_state: T::RAMAccess,
}

impl<T: RISCVInstruction> RISCVCycle<T> {
    fn capture_pre_execution_state(&mut self, registers: [i64; 64]) {
        self.instruction
            .to_raw()
            .capture_pre_execution_state(&mut self.register_state, registers);
    }

    fn capture_post_execution_state(&mut self, registers: [i64; 64]) {
        self.instruction
            .to_raw()
            .capture_post_execution_state(&mut self.register_state, registers);
    }
    // fn capture_memory_state(&mut self, state: MemoryState) {}
}

pub enum RV32IMInstruction {
    ADD(ADD),
    ADDI(ADDI),
    AND(AND),
    ANDI(ANDI),
    AUIPC(AUIPC),
    BEQ(BEQ),
    BGE(BGE),
    BGEU(BGEU),
    BLT(BLT),
    BLTU(BLTU),
    BNE(BNE),
    DIV(DIV),
    DIVU(DIVU),
    JAL(JAL),
    JALR(JALR),
    LB(LB),
    LBU(LBU),
    LH(LH),
    LHU(LHU),
    LUI(LUI),
    LW(LW),
    MUL(MUL),
    MULH(MULH),
    MULHSU(MULHSU),
    MULHU(MULHU),
    MULU(MULU),
    OR(OR),
    ORI(ORI),
    REM(REM),
    REMU(REMU),
    SB(SB),
    SH(SH),
    SLL(SLL),
    SLLI(SLLI),
    SLT(SLT),
    SLTI(SLTI),
    SLTIU(SLTIU),
    SLTU(SLTU),
    SRA(SRA),
    SRAI(SRAI),
    SRL(SRL),
    SRLI(SRLI),
    SUB(SUB),
    SW(SW),
    XOR(XOR),
    XORI(XORI),
}

pub enum RV32IMCycle {
    ADD(RISCVCycle<ADD>),
    ADDI(RISCVCycle<ADDI>),
    AND(RISCVCycle<AND>),
    ANDI(RISCVCycle<ANDI>),
    AUIPC(RISCVCycle<AUIPC>),
    BEQ(RISCVCycle<BEQ>),
    BGE(RISCVCycle<BGE>),
    BGEU(RISCVCycle<BGEU>),
    BLT(RISCVCycle<BLT>),
    BLTU(RISCVCycle<BLTU>),
    BNE(RISCVCycle<BNE>),
    DIV(RISCVCycle<DIV>),
    DIVU(RISCVCycle<DIVU>),
    JAL(RISCVCycle<JAL>),
    JALR(RISCVCycle<JALR>),
    LB(RISCVCycle<LB>),
    LBU(RISCVCycle<LBU>),
    LH(RISCVCycle<LH>),
    LHU(RISCVCycle<LHU>),
    LUI(RISCVCycle<LUI>),
    LW(RISCVCycle<LW>),
    MUL(RISCVCycle<MUL>),
    MULH(RISCVCycle<MULH>),
    MULHSU(RISCVCycle<MULHSU>),
    MULHU(RISCVCycle<MULHU>),
    MULU(RISCVCycle<MULU>),
    OR(RISCVCycle<OR>),
    ORI(RISCVCycle<ORI>),
    REM(RISCVCycle<REM>),
    REMU(RISCVCycle<REMU>),
    SB(RISCVCycle<SB>),
    SH(RISCVCycle<SH>),
    SLL(RISCVCycle<SLL>),
    SLLI(RISCVCycle<SLLI>),
    SLT(RISCVCycle<SLT>),
    SLTI(RISCVCycle<SLTI>),
    SLTIU(RISCVCycle<SLTIU>),
    SLTU(RISCVCycle<SLTU>),
    SRA(RISCVCycle<SRA>),
    SRAI(RISCVCycle<SRAI>),
    SRL(RISCVCycle<SRL>),
    SRLI(RISCVCycle<SRLI>),
    SUB(RISCVCycle<SUB>),
    SW(RISCVCycle<SW>),
    XOR(RISCVCycle<XOR>),
    XORI(RISCVCycle<XORI>),
}
