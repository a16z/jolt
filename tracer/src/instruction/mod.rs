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
use fence::FENCE;
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
use or::OR;
use ori::ORI;
use rem::REM;
use remu::REMU;
use sb::SB;
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
use sub::SUB;
use sw::SW;
use xor::XOR;
use xori::XORI;

use crate::emulator::cpu::Cpu;
use derive_more::From;
use format::InstructionFormat;

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
pub mod fence;
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
pub struct MemoryRead {
    pub(crate) address: u64,
    pub(crate) value: u64,
}
#[derive(Default)]
pub struct MemoryWrite {
    pub(crate) address: u64,
    pub(crate) pre_value: u64,
    pub(crate) post_value: u64,
}

pub trait RISCVInstruction: Sized + Copy {
    const MASK: u32;
    const MATCH: u32;

    type Format: InstructionFormat;
    type RAMAccess: Default;

    fn operands(&self) -> &Self::Format;
    fn new(word: u32, address: u64) -> Self;

    fn execute(&self, cpu: &mut Cpu, memory_state: &mut Self::RAMAccess);
    fn trace(&self, cpu: &mut Cpu) -> RISCVCycle<Self> {
        let mut cycle: RISCVCycle<Self> = RISCVCycle {
            instruction: *self,
            register_state: Default::default(),
            memory_state: Default::default(),
        };
        self.operands()
            .capture_pre_execution_state(&mut cycle.register_state, cpu);
        self.execute(cpu, &mut cycle.memory_state);
        self.operands()
            .capture_post_execution_state(&mut cycle.register_state, cpu);
        cycle
    }
}

#[derive(Default)]
pub struct RISCVCycle<T: RISCVInstruction> {
    pub instruction: T,
    pub register_state: <T::Format as InstructionFormat>::RegisterState,
    pub memory_state: T::RAMAccess,
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
    FENCE(FENCE<32>),
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
    FENCE(RISCVCycle<FENCE<32>>),
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
            RV32IMCycle::ADD(cycle) => cycle.instruction.into(),
            RV32IMCycle::ADDI(cycle) => cycle.instruction.into(),
            RV32IMCycle::AND(cycle) => cycle.instruction.into(),
            RV32IMCycle::ANDI(cycle) => cycle.instruction.into(),
            RV32IMCycle::AUIPC(cycle) => cycle.instruction.into(),
            RV32IMCycle::BEQ(cycle) => cycle.instruction.into(),
            RV32IMCycle::BGE(cycle) => cycle.instruction.into(),
            RV32IMCycle::BGEU(cycle) => cycle.instruction.into(),
            RV32IMCycle::BLT(cycle) => cycle.instruction.into(),
            RV32IMCycle::BLTU(cycle) => cycle.instruction.into(),
            RV32IMCycle::BNE(cycle) => cycle.instruction.into(),
            RV32IMCycle::DIV(cycle) => cycle.instruction.into(),
            RV32IMCycle::DIVU(cycle) => cycle.instruction.into(),
            RV32IMCycle::FENCE(cycle) => cycle.instruction.into(),
            RV32IMCycle::JAL(cycle) => cycle.instruction.into(),
            RV32IMCycle::JALR(cycle) => cycle.instruction.into(),
            RV32IMCycle::LB(cycle) => cycle.instruction.into(),
            RV32IMCycle::LBU(cycle) => cycle.instruction.into(),
            RV32IMCycle::LH(cycle) => cycle.instruction.into(),
            RV32IMCycle::LHU(cycle) => cycle.instruction.into(),
            RV32IMCycle::LUI(cycle) => cycle.instruction.into(),
            RV32IMCycle::LW(cycle) => cycle.instruction.into(),
            RV32IMCycle::MUL(cycle) => cycle.instruction.into(),
            RV32IMCycle::MULH(cycle) => cycle.instruction.into(),
            RV32IMCycle::MULHSU(cycle) => cycle.instruction.into(),
            RV32IMCycle::MULHU(cycle) => cycle.instruction.into(),
            RV32IMCycle::OR(cycle) => cycle.instruction.into(),
            RV32IMCycle::ORI(cycle) => cycle.instruction.into(),
            RV32IMCycle::REM(cycle) => cycle.instruction.into(),
            RV32IMCycle::REMU(cycle) => cycle.instruction.into(),
            RV32IMCycle::SB(cycle) => cycle.instruction.into(),
            RV32IMCycle::SH(cycle) => cycle.instruction.into(),
            RV32IMCycle::SLL(cycle) => cycle.instruction.into(),
            RV32IMCycle::SLLI(cycle) => cycle.instruction.into(),
            RV32IMCycle::SLT(cycle) => cycle.instruction.into(),
            RV32IMCycle::SLTI(cycle) => cycle.instruction.into(),
            RV32IMCycle::SLTIU(cycle) => cycle.instruction.into(),
            RV32IMCycle::SLTU(cycle) => cycle.instruction.into(),
            RV32IMCycle::SRA(cycle) => cycle.instruction.into(),
            RV32IMCycle::SRAI(cycle) => cycle.instruction.into(),
            RV32IMCycle::SRL(cycle) => cycle.instruction.into(),
            RV32IMCycle::SRLI(cycle) => cycle.instruction.into(),
            RV32IMCycle::SUB(cycle) => cycle.instruction.into(),
            RV32IMCycle::SW(cycle) => cycle.instruction.into(),
            RV32IMCycle::XOR(cycle) => cycle.instruction.into(),
            RV32IMCycle::XORI(cycle) => cycle.instruction.into(),
        }
    }
}
