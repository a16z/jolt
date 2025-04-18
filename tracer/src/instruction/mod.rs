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
    UNIMPL,
}

impl RV32IMInstruction {
    pub fn decode(instr: u32, address: u64) -> Result<Self, &'static str> {
        let opcode = instr & 0x7f;
        match opcode {
            0b0110111 => {
                // LUI: U-type => [imm(31:12), rd, opcode]
                Ok(LUI::new(instr, address).into())
            }
            0b0010111 => {
                // AUIPC: U-type => [imm(31:12), rd, opcode]
                Ok(AUIPC::new(instr, address).into())
            }
            0b1101111 => {
                // JAL: UJ-type instruction.
                Ok(JAL::new(instr, address).into())
            }
            0b1100111 => {
                // JALR: I-type, where funct3 must be 0.
                let funct3 = (instr >> 12) & 0x7;
                if funct3 != 0 {
                    return Err("Invalid funct3 for JALR");
                }
                Ok(JALR::new(instr, address).into())
            }
            0b1100011 => {
                // Branch instructions (SB-type): BEQ, BNE, BLT, BGE, BLTU, BGEU.
                match (instr >> 12) & 0x7 {
                    0b000 => Ok(BEQ::new(instr, address).into()),
                    0b001 => Ok(BNE::new(instr, address).into()),
                    0b100 => Ok(BLT::new(instr, address).into()),
                    0b101 => Ok(BGE::new(instr, address).into()),
                    0b110 => Ok(BLTU::new(instr, address).into()),
                    0b111 => Ok(BGEU::new(instr, address).into()),
                    _ => Err("Invalid branch funct3"),
                }
            }
            0b0000011 => {
                // Load instructions (I-type): LB, LH, LW, LBU, LHU.
                match (instr >> 12) & 0x7 {
                    0b000 => Ok(LB::new(instr, address).into()),
                    0b001 => Ok(LH::new(instr, address).into()),
                    0b010 => Ok(LW::new(instr, address).into()),
                    0b100 => Ok(LBU::new(instr, address).into()),
                    0b101 => Ok(LHU::new(instr, address).into()),
                    _ => Err("Invalid load funct3"),
                }
            }
            0b0100011 => {
                // Store instructions (S-type): SB, SH, SW.
                match (instr >> 12) & 0x7 {
                    0b000 => Ok(SB::new(instr, address).into()),
                    0b001 => Ok(SH::new(instr, address).into()),
                    0b010 => Ok(SW::new(instr, address).into()),
                    _ => Err("Invalid store funct3"),
                }
            }
            0b0010011 => {
                // I-type arithmetic instructions: ADDI, SLTI, SLTIU, XORI, ORI, ANDI,
                // and also shift-immediate instructions SLLI, SRLI, SRAI.
                let funct3 = (instr >> 12) & 0x7;
                let funct7 = (instr >> 25) & 0x7f;
                if funct3 == 0b001 {
                    // SLLI uses shamt and expects funct7 == 0.
                    if funct7 == 0 {
                        Ok(SLLI::new(instr, address).into())
                    } else {
                        Err("Invalid funct7 for SLLI")
                    }
                } else if funct3 == 0b101 {
                    if funct7 == 0b0000000 {
                        Ok(SRLI::new(instr, address).into())
                    } else if funct7 == 0b0100000 {
                        Ok(SRAI::new(instr, address).into())
                    } else {
                        Err("Invalid ALU shift funct7")
                    }
                } else {
                    match funct3 {
                        0b000 => Ok(ADDI::new(instr, address).into()),
                        0b010 => Ok(SLTI::new(instr, address).into()),
                        0b011 => Ok(SLTIU::new(instr, address).into()),
                        0b100 => Ok(XORI::new(instr, address).into()),
                        0b110 => Ok(ORI::new(instr, address).into()),
                        0b111 => Ok(ANDI::new(instr, address).into()),
                        _ => Err("Invalid I-type ALU funct3"),
                    }
                }
            }
            0b0110011 => {
                // R-type arithmetic instructions.
                let funct3 = (instr >> 12) & 0x7;
                let funct7 = (instr >> 25) & 0x7f;
                match (funct3, funct7) {
                    (0b000, 0b0000000) => Ok(ADD::new(instr, address).into()),
                    (0b000, 0b0100000) => Ok(SUB::new(instr, address).into()),
                    (0b001, 0b0000000) => Ok(SLL::new(instr, address).into()),
                    (0b010, 0b0000000) => Ok(SLT::new(instr, address).into()),
                    (0b011, 0b0000000) => Ok(SLTU::new(instr, address).into()),
                    (0b100, 0b0000000) => Ok(XOR::new(instr, address).into()),
                    (0b101, 0b0000000) => Ok(SRL::new(instr, address).into()),
                    (0b101, 0b0100000) => Ok(SRA::new(instr, address).into()),
                    (0b110, 0b0000000) => Ok(OR::new(instr, address).into()),
                    (0b111, 0b0000000) => Ok(AND::new(instr, address).into()),
                    // RV32M extension
                    (0b000, 0b0000001) => Ok(MUL::new(instr, address).into()),
                    (0b001, 0b0000001) => Ok(MULH::new(instr, address).into()),
                    (0b010, 0b0000001) => Ok(MULHSU::new(instr, address).into()),
                    (0b011, 0b0000001) => Ok(MULHU::new(instr, address).into()),
                    (0b100, 0b0000001) => Ok(DIV::new(instr, address).into()),
                    (0b101, 0b0000001) => Ok(DIVU::new(instr, address).into()),
                    (0b110, 0b0000001) => Ok(REM::new(instr, address).into()),
                    (0b111, 0b0000001) => Ok(REMU::new(instr, address).into()),
                    _ => Err("Invalid R-type arithmetic instruction"),
                }
            }
            0b0001111 => {
                // FENCE: I-type; the immediate encodes "pred" and "succ" flags.
                Ok(FENCE::new(instr, address).into())
            }
            0b1110011 => {
                // SYSTEM instructions: ECALL/EBREAK (I-type)
                Err("Unsupported SYSTEM instruction")
            }
            _ => Err("Unknown opcode"),
        }
    }

    pub fn trace(&self, cpu: &mut Cpu) -> RV32IMCycle {
        match self {
            RV32IMInstruction::ADD(add) => add.trace(cpu).into(),
            RV32IMInstruction::ADDI(addi) => addi.trace(cpu).into(),
            RV32IMInstruction::AND(and) => and.trace(cpu).into(),
            RV32IMInstruction::ANDI(andi) => andi.trace(cpu).into(),
            RV32IMInstruction::AUIPC(auipc) => auipc.trace(cpu).into(),
            RV32IMInstruction::BEQ(beq) => beq.trace(cpu).into(),
            RV32IMInstruction::BGE(bge) => bge.trace(cpu).into(),
            RV32IMInstruction::BGEU(bgeu) => bgeu.trace(cpu).into(),
            RV32IMInstruction::BLT(blt) => blt.trace(cpu).into(),
            RV32IMInstruction::BLTU(bltu) => bltu.trace(cpu).into(),
            RV32IMInstruction::BNE(bne) => bne.trace(cpu).into(),
            RV32IMInstruction::DIV(div) => div.trace(cpu).into(),
            RV32IMInstruction::DIVU(divu) => divu.trace(cpu).into(),
            RV32IMInstruction::FENCE(fence) => fence.trace(cpu).into(),
            RV32IMInstruction::JAL(jal) => jal.trace(cpu).into(),
            RV32IMInstruction::JALR(jalr) => jalr.trace(cpu).into(),
            RV32IMInstruction::LB(lb) => lb.trace(cpu).into(),
            RV32IMInstruction::LBU(lbu) => lbu.trace(cpu).into(),
            RV32IMInstruction::LH(lh) => lh.trace(cpu).into(),
            RV32IMInstruction::LHU(lhu) => lhu.trace(cpu).into(),
            RV32IMInstruction::LUI(lui) => lui.trace(cpu).into(),
            RV32IMInstruction::LW(lw) => lw.trace(cpu).into(),
            RV32IMInstruction::MUL(mul) => mul.trace(cpu).into(),
            RV32IMInstruction::MULH(mulh) => mulh.trace(cpu).into(),
            RV32IMInstruction::MULHSU(mulhsu) => mulhsu.trace(cpu).into(),
            RV32IMInstruction::MULHU(mulhu) => mulhu.trace(cpu).into(),
            RV32IMInstruction::OR(or) => or.trace(cpu).into(),
            RV32IMInstruction::ORI(ori) => ori.trace(cpu).into(),
            RV32IMInstruction::REM(rem) => rem.trace(cpu).into(),
            RV32IMInstruction::REMU(remu) => remu.trace(cpu).into(),
            RV32IMInstruction::SB(sb) => sb.trace(cpu).into(),
            RV32IMInstruction::SH(sh) => sh.trace(cpu).into(),
            RV32IMInstruction::SLL(sll) => sll.trace(cpu).into(),
            RV32IMInstruction::SLLI(slli) => slli.trace(cpu).into(),
            RV32IMInstruction::SLT(slt) => slt.trace(cpu).into(),
            RV32IMInstruction::SLTI(slti) => slti.trace(cpu).into(),
            RV32IMInstruction::SLTIU(sltiu) => sltiu.trace(cpu).into(),
            RV32IMInstruction::SLTU(sltu) => sltu.trace(cpu).into(),
            RV32IMInstruction::SRA(sra) => sra.trace(cpu).into(),
            RV32IMInstruction::SRAI(srai) => srai.trace(cpu).into(),
            RV32IMInstruction::SRL(srl) => srl.trace(cpu).into(),
            RV32IMInstruction::SRLI(srli) => srli.trace(cpu).into(),
            RV32IMInstruction::SUB(sub) => sub.trace(cpu).into(),
            RV32IMInstruction::SW(sw) => sw.trace(cpu).into(),
            RV32IMInstruction::XOR(xor) => xor.trace(cpu).into(),
            RV32IMInstruction::XORI(xori) => xori.trace(cpu).into(),
            RV32IMInstruction::UNIMPL => {
                unimplemented!("UNIMPL")
            }
        }
    }
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
