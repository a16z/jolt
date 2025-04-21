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
use strum_macros::IntoStaticStr;
use sub::SUB;
use sw::SW;
use xor::XOR;
use xori::XORI;

use virtual_advice::VirtualAdvice;
use virtual_assert_eq::VirtualAssertEQ;
use virtual_assert_halfword_alignment::VirtualAssertHalfwordAlignment;
use virtual_assert_lte::VirtualAssertLTE;
use virtual_assert_valid_div0::VirtualAssertValidDiv0;
use virtual_assert_valid_signed_remainder::VirtualAssertValidSignedRemainder;
use virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder;
use virtual_move::VirtualMove;
use virtual_movsign::VirtualMovsign;
use virtual_pow2::VirtualPow2;
use virtual_pow2i::VirtualPow2I;
use virtual_shift_right_bitmask::VirtualShiftRightBitmask;
use virtual_shift_right_bitmaski::VirtualShiftRightBitmaskI;
use virtual_sra::VirtualSRA;
use virtual_srl::VirtualSRL;

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
pub mod virtual_advice;
pub mod virtual_assert_eq;
pub mod virtual_assert_halfword_alignment;
pub mod virtual_assert_lte;
pub mod virtual_assert_valid_div0;
pub mod virtual_assert_valid_signed_remainder;
pub mod virtual_assert_valid_unsigned_remainder;
pub mod virtual_move;
pub mod virtual_movsign;
pub mod virtual_pow2;
pub mod virtual_pow2i;
pub mod virtual_shift_right_bitmask;
pub mod virtual_shift_right_bitmaski;
pub mod virtual_sra;
pub mod virtual_srl;
pub mod xor;
pub mod xori;

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct MemoryRead {
    pub(crate) address: u64,
    pub(crate) value: u64,
}

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct MemoryWrite {
    pub(crate) address: u64,
    pub(crate) pre_value: u64,
    pub(crate) post_value: u64,
}

pub trait RISCVInstruction: Sized + Copy + Into<RV32IMInstruction> {
    const MASK: u32;
    const MATCH: u32;

    type Format: InstructionFormat;
    type RAMAccess: Default;

    fn operands(&self) -> &Self::Format;
    fn new(word: u32, address: u64) -> Self;

    fn execute(&self, cpu: &mut Cpu, ram_access: &mut Self::RAMAccess);
}

pub trait RISCVTrace: RISCVInstruction
where
    RISCVCycle<Self>: Into<RV32IMCycle>,
{
    fn trace(&self, cpu: &mut Cpu) {
        let mut cycle: RISCVCycle<Self> = RISCVCycle {
            instruction: *self,
            register_state: Default::default(),
            ram_access: Default::default(),
        };
        self.operands()
            .capture_pre_execution_state(&mut cycle.register_state, cpu);
        self.execute(cpu, &mut cycle.ram_access);
        self.operands()
            .capture_post_execution_state(&mut cycle.register_state, cpu);
        cpu.trace.push(cycle.into());
    }
}

pub trait VirtualInstructionSequence: RISCVInstruction {
    fn virtual_sequence(&self) -> Vec<RV32IMInstruction>;
}

#[derive(Debug, From, Clone, Serialize, Deserialize)]
pub enum RV32IMInstruction {
    NoOp,
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
    FENCE(FENCE),
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
    // Virtual
    UNIMPL,
    Advice(VirtualAdvice),
    AssertEQ(VirtualAssertEQ),
    AssertHalfwordAlignment(VirtualAssertHalfwordAlignment),
    AssertLTE(VirtualAssertLTE),
    AssertValidDiv0(VirtualAssertValidDiv0),
    AssertValidSignedRemainder(VirtualAssertValidSignedRemainder),
    AssertValidUnsignedRemainder(VirtualAssertValidUnsignedRemainder),
    Move(VirtualMove),
    Movsign(VirtualMovsign),
    Pow2(VirtualPow2),
    Pow2I(VirtualPow2I),
    ShiftRightBitmask(VirtualShiftRightBitmask),
    ShiftRightBitmaskI(VirtualShiftRightBitmaskI),
    VirtualSRA(VirtualSRA),
    VirtualSRL(VirtualSRL),
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

    pub fn trace(&self, cpu: &mut Cpu) {
        match self {
            RV32IMInstruction::ADD(add) => add.trace(cpu),
            RV32IMInstruction::ADDI(addi) => addi.trace(cpu),
            RV32IMInstruction::AND(and) => and.trace(cpu),
            RV32IMInstruction::ANDI(andi) => andi.trace(cpu),
            RV32IMInstruction::AUIPC(auipc) => auipc.trace(cpu),
            RV32IMInstruction::BEQ(beq) => beq.trace(cpu),
            RV32IMInstruction::BGE(bge) => bge.trace(cpu),
            RV32IMInstruction::BGEU(bgeu) => bgeu.trace(cpu),
            RV32IMInstruction::BLT(blt) => blt.trace(cpu),
            RV32IMInstruction::BLTU(bltu) => bltu.trace(cpu),
            RV32IMInstruction::BNE(bne) => bne.trace(cpu),
            RV32IMInstruction::DIV(div) => div.trace(cpu),
            RV32IMInstruction::DIVU(divu) => divu.trace(cpu),
            RV32IMInstruction::FENCE(fence) => fence.trace(cpu),
            RV32IMInstruction::JAL(jal) => jal.trace(cpu),
            RV32IMInstruction::JALR(jalr) => jalr.trace(cpu),
            RV32IMInstruction::LB(lb) => lb.trace(cpu),
            RV32IMInstruction::LBU(lbu) => lbu.trace(cpu),
            RV32IMInstruction::LH(lh) => lh.trace(cpu),
            RV32IMInstruction::LHU(lhu) => lhu.trace(cpu),
            RV32IMInstruction::LUI(lui) => lui.trace(cpu),
            RV32IMInstruction::LW(lw) => lw.trace(cpu),
            RV32IMInstruction::MUL(mul) => mul.trace(cpu),
            RV32IMInstruction::MULH(mulh) => mulh.trace(cpu),
            RV32IMInstruction::MULHSU(mulhsu) => mulhsu.trace(cpu),
            RV32IMInstruction::MULHU(mulhu) => mulhu.trace(cpu),
            RV32IMInstruction::OR(or) => or.trace(cpu),
            RV32IMInstruction::ORI(ori) => ori.trace(cpu),
            RV32IMInstruction::REM(rem) => rem.trace(cpu),
            RV32IMInstruction::REMU(remu) => remu.trace(cpu),
            RV32IMInstruction::SB(sb) => sb.trace(cpu),
            RV32IMInstruction::SH(sh) => sh.trace(cpu),
            RV32IMInstruction::SLL(sll) => sll.trace(cpu),
            RV32IMInstruction::SLLI(slli) => slli.trace(cpu),
            RV32IMInstruction::SLT(slt) => slt.trace(cpu),
            RV32IMInstruction::SLTI(slti) => slti.trace(cpu),
            RV32IMInstruction::SLTIU(sltiu) => sltiu.trace(cpu),
            RV32IMInstruction::SLTU(sltu) => sltu.trace(cpu),
            RV32IMInstruction::SRA(sra) => sra.trace(cpu),
            RV32IMInstruction::SRAI(srai) => srai.trace(cpu),
            RV32IMInstruction::SRL(srl) => srl.trace(cpu),
            RV32IMInstruction::SRLI(srli) => srli.trace(cpu),
            RV32IMInstruction::SUB(sub) => sub.trace(cpu),
            RV32IMInstruction::SW(sw) => sw.trace(cpu),
            RV32IMInstruction::XOR(xor) => xor.trace(cpu),
            RV32IMInstruction::XORI(xori) => xori.trace(cpu),
            _ => panic!("Unexpected instruction {:?}", self),
        };
    }
}

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct RISCVCycle<T: RISCVInstruction> {
    pub instruction: T,
    pub register_state: <T::Format as InstructionFormat>::RegisterState,
    pub ram_access: T::RAMAccess,
}

#[derive(From, Clone, Serialize, Deserialize, IntoStaticStr)]
pub enum RV32IMCycle {
    NoOp,
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
    FENCE(RISCVCycle<FENCE>),
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
    // Virtual
    Advice(RISCVCycle<VirtualAdvice>),
    AssertEQ(RISCVCycle<VirtualAssertEQ>),
    AssertHalfwordAlignment(RISCVCycle<VirtualAssertHalfwordAlignment>),
    AssertLTE(RISCVCycle<VirtualAssertLTE>),
    AssertValidDiv0(RISCVCycle<VirtualAssertValidDiv0>),
    AssertValidSignedRemainder(RISCVCycle<VirtualAssertValidSignedRemainder>),
    AssertValidUnsignedRemainder(RISCVCycle<VirtualAssertValidUnsignedRemainder>),
    Move(RISCVCycle<VirtualMove>),
    Movsign(RISCVCycle<VirtualMovsign>),
    Pow2(RISCVCycle<VirtualPow2>),
    Pow2I(RISCVCycle<VirtualPow2I>),
    ShiftRightBitmask(RISCVCycle<VirtualShiftRightBitmask>),
    ShiftRightBitmaskI(RISCVCycle<VirtualShiftRightBitmaskI>),
    VirtualSRA(RISCVCycle<VirtualSRA>),
    VirtualSRL(RISCVCycle<VirtualSRL>),
}
