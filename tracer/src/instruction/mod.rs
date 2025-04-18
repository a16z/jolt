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
use derive_more::From;
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

#[derive(Default)]
pub struct MemoryRead(u64, u64); // (address, value)
#[derive(Default)]
pub struct MemoryWrite(u64, u64, u64); // (address, old_value, new_value)

pub trait RISCVInstruction: Sized + Copy {
    const MASK: u32;
    const MATCH: u32;

    type Format: InstructionFormat;
    type RAMAccess: Default;

    fn to_raw(&self) -> Self::Format;
    fn new(word: u32, address: u64) -> Self;
}

pub struct RISCVCycle<T: RISCVInstruction> {
    pub instruction: T,
    pub register_state: <T::Format as InstructionFormat>::RegisterState,
    pub memory_state: T::RAMAccess,
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

#[derive(From)]
pub enum RV32IMInstruction {
    ADD(ADD<32>),
    ADDI(ADDI<32>),
    AND(AND<32>),
    ANDI(ANDI<32>),
    AUIPC(AUIPC<32>),
    BEQ(BEQ<32>),
    BGE(BGE<32>),
    BGEU(BGEU<32>),
    BLT(BLT<32>),
    BLTU(BLTU<32>),
    BNE(BNE<32>),
    DIV(DIV<32>),
    DIVU(DIVU<32>),
    JAL(JAL<32>),
    JALR(JALR<32>),
    LB(LB<32>),
    LBU(LBU<32>),
    LH(LH<32>),
    LHU(LHU<32>),
    LUI(LUI<32>),
    LW(LW<32>),
    MUL(MUL<32>),
    MULH(MULH<32>),
    MULHSU(MULHSU<32>),
    MULHU(MULHU<32>),
    MULU(MULU<32>),
    OR(OR<32>),
    ORI(ORI<32>),
    REM(REM<32>),
    REMU(REMU<32>),
    SB(SB<32>),
    SH(SH<32>),
    SLL(SLL<32>),
    SLLI(SLLI<32>),
    SLT(SLT<32>),
    SLTI(SLTI<32>),
    SLTIU(SLTIU<32>),
    SLTU(SLTU<32>),
    SRA(SRA<32>),
    SRAI(SRAI<32>),
    SRL(SRL<32>),
    SRLI(SRLI<32>),
    SUB(SUB<32>),
    SW(SW<32>),
    XOR(XOR<32>),
    XORI(XORI<32>),
}

#[derive(From)]
pub enum RV32IMCycle {
    ADD(RISCVCycle<ADD<32>>),
    ADDI(RISCVCycle<ADDI<32>>),
    AND(RISCVCycle<AND<32>>),
    ANDI(RISCVCycle<ANDI<32>>),
    AUIPC(RISCVCycle<AUIPC<32>>),
    BEQ(RISCVCycle<BEQ<32>>),
    BGE(RISCVCycle<BGE<32>>),
    BGEU(RISCVCycle<BGEU<32>>),
    BLT(RISCVCycle<BLT<32>>),
    BLTU(RISCVCycle<BLTU<32>>),
    BNE(RISCVCycle<BNE<32>>),
    DIV(RISCVCycle<DIV<32>>),
    DIVU(RISCVCycle<DIVU<32>>),
    JAL(RISCVCycle<JAL<32>>),
    JALR(RISCVCycle<JALR<32>>),
    LB(RISCVCycle<LB<32>>),
    LBU(RISCVCycle<LBU<32>>),
    LH(RISCVCycle<LH<32>>),
    LHU(RISCVCycle<LHU<32>>),
    LUI(RISCVCycle<LUI<32>>),
    LW(RISCVCycle<LW<32>>),
    MUL(RISCVCycle<MUL<32>>),
    MULH(RISCVCycle<MULH<32>>),
    MULHSU(RISCVCycle<MULHSU<32>>),
    MULHU(RISCVCycle<MULHU<32>>),
    MULU(RISCVCycle<MULU<32>>),
    OR(RISCVCycle<OR<32>>),
    ORI(RISCVCycle<ORI<32>>),
    REM(RISCVCycle<REM<32>>),
    REMU(RISCVCycle<REMU<32>>),
    SB(RISCVCycle<SB<32>>),
    SH(RISCVCycle<SH<32>>),
    SLL(RISCVCycle<SLL<32>>),
    SLLI(RISCVCycle<SLLI<32>>),
    SLT(RISCVCycle<SLT<32>>),
    SLTI(RISCVCycle<SLTI<32>>),
    SLTIU(RISCVCycle<SLTIU<32>>),
    SLTU(RISCVCycle<SLTU<32>>),
    SRA(RISCVCycle<SRA<32>>),
    SRAI(RISCVCycle<SRAI<32>>),
    SRL(RISCVCycle<SRL<32>>),
    SRLI(RISCVCycle<SRLI<32>>),
    SUB(RISCVCycle<SUB<32>>),
    SW(RISCVCycle<SW<32>>),
    XOR(RISCVCycle<XOR<32>>),
    XORI(RISCVCycle<XORI<32>>),
}

impl RV32IMCycle {
    pub fn instruction(&self) -> RV32IMInstruction {
        match self {
            RV32IMCycle::ADD(cycle) => RV32IMInstruction::ADD(cycle.instruction),
            RV32IMCycle::ADDI(cycle) => RV32IMInstruction::ADDI(cycle.instruction),
            RV32IMCycle::AND(cycle) => RV32IMInstruction::AND(cycle.instruction),
            RV32IMCycle::ANDI(cycle) => RV32IMInstruction::ANDI(cycle.instruction),
            RV32IMCycle::AUIPC(cycle) => RV32IMInstruction::AUIPC(cycle.instruction),
            RV32IMCycle::BEQ(cycle) => RV32IMInstruction::BEQ(cycle.instruction),
            RV32IMCycle::BGE(cycle) => RV32IMInstruction::BGE(cycle.instruction),
            RV32IMCycle::BGEU(cycle) => RV32IMInstruction::BGEU(cycle.instruction),
            RV32IMCycle::BLT(cycle) => RV32IMInstruction::BLT(cycle.instruction),
            RV32IMCycle::BLTU(cycle) => RV32IMInstruction::BLTU(cycle.instruction),
            RV32IMCycle::BNE(cycle) => RV32IMInstruction::BNE(cycle.instruction),
            RV32IMCycle::DIV(cycle) => RV32IMInstruction::DIV(cycle.instruction),
            RV32IMCycle::DIVU(cycle) => RV32IMInstruction::DIVU(cycle.instruction),
            RV32IMCycle::JAL(cycle) => RV32IMInstruction::JAL(cycle.instruction),
            RV32IMCycle::JALR(cycle) => RV32IMInstruction::JALR(cycle.instruction),
            RV32IMCycle::LB(cycle) => RV32IMInstruction::LB(cycle.instruction),
            RV32IMCycle::LBU(cycle) => RV32IMInstruction::LBU(cycle.instruction),
            RV32IMCycle::LH(cycle) => RV32IMInstruction::LH(cycle.instruction),
            RV32IMCycle::LHU(cycle) => RV32IMInstruction::LHU(cycle.instruction),
            RV32IMCycle::LUI(cycle) => RV32IMInstruction::LUI(cycle.instruction),
            RV32IMCycle::LW(cycle) => RV32IMInstruction::LW(cycle.instruction),
            RV32IMCycle::MUL(cycle) => RV32IMInstruction::MUL(cycle.instruction),
            RV32IMCycle::MULH(cycle) => RV32IMInstruction::MULH(cycle.instruction),
            RV32IMCycle::MULHSU(cycle) => RV32IMInstruction::MULHSU(cycle.instruction),
            RV32IMCycle::MULHU(cycle) => RV32IMInstruction::MULHU(cycle.instruction),
            RV32IMCycle::MULU(cycle) => RV32IMInstruction::MULU(cycle.instruction),
            RV32IMCycle::OR(cycle) => RV32IMInstruction::OR(cycle.instruction),
            RV32IMCycle::ORI(cycle) => RV32IMInstruction::ORI(cycle.instruction),
            RV32IMCycle::REM(cycle) => RV32IMInstruction::REM(cycle.instruction),
            RV32IMCycle::REMU(cycle) => RV32IMInstruction::REMU(cycle.instruction),
            RV32IMCycle::SB(cycle) => RV32IMInstruction::SB(cycle.instruction),
            RV32IMCycle::SH(cycle) => RV32IMInstruction::SH(cycle.instruction),
            RV32IMCycle::SLL(cycle) => RV32IMInstruction::SLL(cycle.instruction),
            RV32IMCycle::SLLI(cycle) => RV32IMInstruction::SLLI(cycle.instruction),
            RV32IMCycle::SLT(cycle) => RV32IMInstruction::SLT(cycle.instruction),
            RV32IMCycle::SLTI(cycle) => RV32IMInstruction::SLTI(cycle.instruction),
            RV32IMCycle::SLTIU(cycle) => RV32IMInstruction::SLTIU(cycle.instruction),
            RV32IMCycle::SLTU(cycle) => RV32IMInstruction::SLTU(cycle.instruction),
            RV32IMCycle::SRA(cycle) => RV32IMInstruction::SRA(cycle.instruction),
            RV32IMCycle::SRAI(cycle) => RV32IMInstruction::SRAI(cycle.instruction),
            RV32IMCycle::SRL(cycle) => RV32IMInstruction::SRL(cycle.instruction),
            RV32IMCycle::SRLI(cycle) => RV32IMInstruction::SRLI(cycle.instruction),
            RV32IMCycle::SUB(cycle) => RV32IMInstruction::SUB(cycle.instruction),
            RV32IMCycle::SW(cycle) => RV32IMInstruction::SW(cycle.instruction),
            RV32IMCycle::XOR(cycle) => RV32IMInstruction::XOR(cycle.instruction),
            RV32IMCycle::XORI(cycle) => RV32IMInstruction::XORI(cycle.instruction),
        }
    }
}
