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
use rand::{rngs::StdRng, RngCore};
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
use strum_macros::{EnumCount as EnumCountMacro, EnumIter, IntoStaticStr};
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
use format::{InstructionFormat, InstructionRegisterState};

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

#[cfg(test)]
pub mod test;

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize)]
pub struct RAMRead {
    pub address: u64,
    pub value: u64,
}

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize)]
pub struct RAMWrite {
    pub address: u64,
    pub pre_value: u64,
    pub post_value: u64,
}

pub enum RAMAccess {
    Read(RAMRead),
    Write(RAMWrite),
    NoOp,
}

impl RAMAccess {
    pub fn address(&self) -> usize {
        match self {
            RAMAccess::Read(read) => read.address as usize,
            RAMAccess::Write(write) => write.address as usize,
            RAMAccess::NoOp => 0,
        }
    }
}

impl From<RAMRead> for RAMAccess {
    fn from(read: RAMRead) -> Self {
        Self::Read(read)
    }
}

impl From<RAMWrite> for RAMAccess {
    fn from(write: RAMWrite) -> Self {
        Self::Write(write)
    }
}

impl From<()> for RAMAccess {
    fn from(_: ()) -> Self {
        Self::NoOp
    }
}

pub trait RISCVInstruction: std::fmt::Debug + Sized + Copy + Into<RV32IMInstruction> {
    const MASK: u32;
    const MATCH: u32;

    type Format: InstructionFormat;
    type RAMAccess: Default + Into<RAMAccess> + Copy + std::fmt::Debug;

    fn operands(&self) -> &Self::Format;
    fn new(word: u32, address: u64, validate: bool) -> Self;
    fn random(rng: &mut StdRng) -> Self {
        Self::new(rng.next_u32(), rng.next_u64(), false)
    }

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

#[derive(Debug, IntoStaticStr, From, Clone, Serialize, Deserialize)]
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
                Ok(LUI::new(instr, address, true).into())
            }
            0b0010111 => {
                // AUIPC: U-type => [imm(31:12), rd, opcode]
                Ok(AUIPC::new(instr, address, true).into())
            }
            0b1101111 => {
                // JAL: UJ-type instruction.
                Ok(JAL::new(instr, address, true).into())
            }
            0b1100111 => {
                // JALR: I-type, where funct3 must be 0.
                let funct3 = (instr >> 12) & 0x7;
                if funct3 != 0 {
                    return Err("Invalid funct3 for JALR");
                }
                Ok(JALR::new(instr, address, true).into())
            }
            0b1100011 => {
                // Branch instructions (SB-type): BEQ, BNE, BLT, BGE, BLTU, BGEU.
                match (instr >> 12) & 0x7 {
                    0b000 => Ok(BEQ::new(instr, address, true).into()),
                    0b001 => Ok(BNE::new(instr, address, true).into()),
                    0b100 => Ok(BLT::new(instr, address, true).into()),
                    0b101 => Ok(BGE::new(instr, address, true).into()),
                    0b110 => Ok(BLTU::new(instr, address, true).into()),
                    0b111 => Ok(BGEU::new(instr, address, true).into()),
                    _ => Err("Invalid branch funct3"),
                }
            }
            0b0000011 => {
                // Load instructions (I-type): LB, LH, LW, LBU, LHU.
                match (instr >> 12) & 0x7 {
                    0b000 => Ok(LB::new(instr, address, true).into()),
                    0b001 => Ok(LH::new(instr, address, true).into()),
                    0b010 => Ok(LW::new(instr, address, true).into()),
                    0b100 => Ok(LBU::new(instr, address, true).into()),
                    0b101 => Ok(LHU::new(instr, address, true).into()),
                    _ => Err("Invalid load funct3"),
                }
            }
            0b0100011 => {
                // Store instructions (S-type): SB, SH, SW.
                match (instr >> 12) & 0x7 {
                    0b000 => Ok(SB::new(instr, address, true).into()),
                    0b001 => Ok(SH::new(instr, address, true).into()),
                    0b010 => Ok(SW::new(instr, address, true).into()),
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
                        Ok(SLLI::new(instr, address, true).into())
                    } else {
                        Err("Invalid funct7 for SLLI")
                    }
                } else if funct3 == 0b101 {
                    if funct7 == 0b0000000 {
                        Ok(SRLI::new(instr, address, true).into())
                    } else if funct7 == 0b0100000 {
                        Ok(SRAI::new(instr, address, true).into())
                    } else {
                        Err("Invalid ALU shift funct7")
                    }
                } else {
                    match funct3 {
                        0b000 => Ok(ADDI::new(instr, address, true).into()),
                        0b010 => Ok(SLTI::new(instr, address, true).into()),
                        0b011 => Ok(SLTIU::new(instr, address, true).into()),
                        0b100 => Ok(XORI::new(instr, address, true).into()),
                        0b110 => Ok(ORI::new(instr, address, true).into()),
                        0b111 => Ok(ANDI::new(instr, address, true).into()),
                        _ => Err("Invalid I-type ALU funct3"),
                    }
                }
            }
            0b0110011 => {
                // R-type arithmetic instructions.
                let funct3 = (instr >> 12) & 0x7;
                let funct7 = (instr >> 25) & 0x7f;
                match (funct3, funct7) {
                    (0b000, 0b0000000) => Ok(ADD::new(instr, address, true).into()),
                    (0b000, 0b0100000) => Ok(SUB::new(instr, address, true).into()),
                    (0b001, 0b0000000) => Ok(SLL::new(instr, address, true).into()),
                    (0b010, 0b0000000) => Ok(SLT::new(instr, address, true).into()),
                    (0b011, 0b0000000) => Ok(SLTU::new(instr, address, true).into()),
                    (0b100, 0b0000000) => Ok(XOR::new(instr, address, true).into()),
                    (0b101, 0b0000000) => Ok(SRL::new(instr, address, true).into()),
                    (0b101, 0b0100000) => Ok(SRA::new(instr, address, true).into()),
                    (0b110, 0b0000000) => Ok(OR::new(instr, address, true).into()),
                    (0b111, 0b0000000) => Ok(AND::new(instr, address, true).into()),
                    // RV32M extension
                    (0b000, 0b0000001) => Ok(MUL::new(instr, address, true).into()),
                    (0b001, 0b0000001) => Ok(MULH::new(instr, address, true).into()),
                    (0b010, 0b0000001) => Ok(MULHSU::new(instr, address, true).into()),
                    (0b011, 0b0000001) => Ok(MULHU::new(instr, address, true).into()),
                    (0b100, 0b0000001) => Ok(DIV::new(instr, address, true).into()),
                    (0b101, 0b0000001) => Ok(DIVU::new(instr, address, true).into()),
                    (0b110, 0b0000001) => Ok(REM::new(instr, address, true).into()),
                    (0b111, 0b0000001) => Ok(REMU::new(instr, address, true).into()),
                    _ => Err("Invalid R-type arithmetic instruction"),
                }
            }
            0b0001111 => {
                // FENCE: I-type; the immediate encodes "pred" and "succ" flags.
                Ok(FENCE::new(instr, address, true).into())
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
            RV32IMInstruction::ADD(instr) => instr.trace(cpu),
            RV32IMInstruction::ADDI(instr) => instr.trace(cpu),
            RV32IMInstruction::AND(instr) => instr.trace(cpu),
            RV32IMInstruction::ANDI(instr) => instr.trace(cpu),
            RV32IMInstruction::AUIPC(instr) => instr.trace(cpu),
            RV32IMInstruction::BEQ(instr) => instr.trace(cpu),
            RV32IMInstruction::BGE(instr) => instr.trace(cpu),
            RV32IMInstruction::BGEU(instr) => instr.trace(cpu),
            RV32IMInstruction::BLT(instr) => instr.trace(cpu),
            RV32IMInstruction::BLTU(instr) => instr.trace(cpu),
            RV32IMInstruction::BNE(instr) => instr.trace(cpu),
            RV32IMInstruction::DIV(instr) => instr.trace(cpu),
            RV32IMInstruction::DIVU(instr) => instr.trace(cpu),
            RV32IMInstruction::FENCE(instr) => instr.trace(cpu),
            RV32IMInstruction::JAL(instr) => instr.trace(cpu),
            RV32IMInstruction::JALR(instr) => instr.trace(cpu),
            RV32IMInstruction::LB(instr) => instr.trace(cpu),
            RV32IMInstruction::LBU(instr) => instr.trace(cpu),
            RV32IMInstruction::LH(instr) => instr.trace(cpu),
            RV32IMInstruction::LHU(instr) => instr.trace(cpu),
            RV32IMInstruction::LUI(instr) => instr.trace(cpu),
            RV32IMInstruction::LW(instr) => instr.trace(cpu),
            RV32IMInstruction::MUL(instr) => instr.trace(cpu),
            RV32IMInstruction::MULH(instr) => instr.trace(cpu),
            RV32IMInstruction::MULHSU(instr) => instr.trace(cpu),
            RV32IMInstruction::MULHU(instr) => instr.trace(cpu),
            RV32IMInstruction::OR(instr) => instr.trace(cpu),
            RV32IMInstruction::ORI(instr) => instr.trace(cpu),
            RV32IMInstruction::REM(instr) => instr.trace(cpu),
            RV32IMInstruction::REMU(instr) => instr.trace(cpu),
            RV32IMInstruction::SB(instr) => instr.trace(cpu),
            RV32IMInstruction::SH(instr) => instr.trace(cpu),
            RV32IMInstruction::SLL(instr) => instr.trace(cpu),
            RV32IMInstruction::SLLI(instr) => instr.trace(cpu),
            RV32IMInstruction::SLT(instr) => instr.trace(cpu),
            RV32IMInstruction::SLTI(instr) => instr.trace(cpu),
            RV32IMInstruction::SLTIU(instr) => instr.trace(cpu),
            RV32IMInstruction::SLTU(instr) => instr.trace(cpu),
            RV32IMInstruction::SRA(instr) => instr.trace(cpu),
            RV32IMInstruction::SRAI(instr) => instr.trace(cpu),
            RV32IMInstruction::SRL(instr) => instr.trace(cpu),
            RV32IMInstruction::SRLI(instr) => instr.trace(cpu),
            RV32IMInstruction::SUB(instr) => instr.trace(cpu),
            RV32IMInstruction::SW(instr) => instr.trace(cpu),
            RV32IMInstruction::XOR(instr) => instr.trace(cpu),
            RV32IMInstruction::XORI(instr) => instr.trace(cpu),
            // Virtual
            RV32IMInstruction::Advice(instr) => instr.trace(cpu),
            RV32IMInstruction::AssertEQ(instr) => instr.trace(cpu),
            RV32IMInstruction::AssertHalfwordAlignment(instr) => instr.trace(cpu),
            RV32IMInstruction::AssertLTE(instr) => instr.trace(cpu),
            RV32IMInstruction::AssertValidDiv0(instr) => instr.trace(cpu),
            RV32IMInstruction::AssertValidSignedRemainder(instr) => instr.trace(cpu),
            RV32IMInstruction::AssertValidUnsignedRemainder(instr) => instr.trace(cpu),
            RV32IMInstruction::Move(instr) => instr.trace(cpu),
            RV32IMInstruction::Movsign(instr) => instr.trace(cpu),
            RV32IMInstruction::Pow2(instr) => instr.trace(cpu),
            RV32IMInstruction::Pow2I(instr) => instr.trace(cpu),
            RV32IMInstruction::ShiftRightBitmask(instr) => instr.trace(cpu),
            RV32IMInstruction::ShiftRightBitmaskI(instr) => instr.trace(cpu),
            RV32IMInstruction::VirtualSRA(instr) => instr.trace(cpu),
            RV32IMInstruction::VirtualSRL(instr) => instr.trace(cpu),
            _ => panic!("Unexpected instruction {:?}", self),
        };
    }
}

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize)]
pub struct RISCVCycle<T: RISCVInstruction> {
    pub instruction: T,
    pub register_state: <T::Format as InstructionFormat>::RegisterState,
    pub ram_access: T::RAMAccess,
}

impl<T: RISCVInstruction> RISCVCycle<T> {
    pub fn random(&self, rng: &mut StdRng) -> Self {
        let instruction = T::random(rng);
        let register_state =
            <<T::Format as InstructionFormat>::RegisterState as InstructionRegisterState>::random(
                rng,
            );
        Self {
            instruction,
            ram_access: Default::default(),
            register_state,
        }
    }
}

#[derive(
    From, Debug, Copy, Clone, Serialize, Deserialize, IntoStaticStr, EnumIter, EnumCountMacro,
)]
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

impl RV32IMCycle {
    pub fn ram_access(&self) -> RAMAccess {
        match self {
            RV32IMCycle::NoOp => RAMAccess::NoOp,
            RV32IMCycle::ADD(cycle) => cycle.ram_access.into(),
            RV32IMCycle::ADDI(cycle) => cycle.ram_access.into(),
            RV32IMCycle::AND(cycle) => cycle.ram_access.into(),
            RV32IMCycle::ANDI(cycle) => cycle.ram_access.into(),
            RV32IMCycle::AUIPC(cycle) => cycle.ram_access.into(),
            RV32IMCycle::BEQ(cycle) => cycle.ram_access.into(),
            RV32IMCycle::BGE(cycle) => cycle.ram_access.into(),
            RV32IMCycle::BGEU(cycle) => cycle.ram_access.into(),
            RV32IMCycle::BLT(cycle) => cycle.ram_access.into(),
            RV32IMCycle::BLTU(cycle) => cycle.ram_access.into(),
            RV32IMCycle::BNE(cycle) => cycle.ram_access.into(),
            RV32IMCycle::DIV(cycle) => cycle.ram_access.into(),
            RV32IMCycle::DIVU(cycle) => cycle.ram_access.into(),
            RV32IMCycle::FENCE(cycle) => cycle.ram_access.into(),
            RV32IMCycle::JAL(cycle) => cycle.ram_access.into(),
            RV32IMCycle::JALR(cycle) => cycle.ram_access.into(),
            RV32IMCycle::LB(cycle) => cycle.ram_access.into(),
            RV32IMCycle::LBU(cycle) => cycle.ram_access.into(),
            RV32IMCycle::LH(cycle) => cycle.ram_access.into(),
            RV32IMCycle::LHU(cycle) => cycle.ram_access.into(),
            RV32IMCycle::LUI(cycle) => cycle.ram_access.into(),
            RV32IMCycle::LW(cycle) => cycle.ram_access.into(),
            RV32IMCycle::MUL(cycle) => cycle.ram_access.into(),
            RV32IMCycle::MULH(cycle) => cycle.ram_access.into(),
            RV32IMCycle::MULHSU(cycle) => cycle.ram_access.into(),
            RV32IMCycle::MULHU(cycle) => cycle.ram_access.into(),
            RV32IMCycle::OR(cycle) => cycle.ram_access.into(),
            RV32IMCycle::ORI(cycle) => cycle.ram_access.into(),
            RV32IMCycle::REM(cycle) => cycle.ram_access.into(),
            RV32IMCycle::REMU(cycle) => cycle.ram_access.into(),
            RV32IMCycle::SB(cycle) => cycle.ram_access.into(),
            RV32IMCycle::SH(cycle) => cycle.ram_access.into(),
            RV32IMCycle::SLL(cycle) => cycle.ram_access.into(),
            RV32IMCycle::SLLI(cycle) => cycle.ram_access.into(),
            RV32IMCycle::SLT(cycle) => cycle.ram_access.into(),
            RV32IMCycle::SLTI(cycle) => cycle.ram_access.into(),
            RV32IMCycle::SLTIU(cycle) => cycle.ram_access.into(),
            RV32IMCycle::SLTU(cycle) => cycle.ram_access.into(),
            RV32IMCycle::SRA(cycle) => cycle.ram_access.into(),
            RV32IMCycle::SRAI(cycle) => cycle.ram_access.into(),
            RV32IMCycle::SRL(cycle) => cycle.ram_access.into(),
            RV32IMCycle::SRLI(cycle) => cycle.ram_access.into(),
            RV32IMCycle::SUB(cycle) => cycle.ram_access.into(),
            RV32IMCycle::SW(cycle) => cycle.ram_access.into(),
            RV32IMCycle::XOR(cycle) => cycle.ram_access.into(),
            RV32IMCycle::XORI(cycle) => cycle.ram_access.into(),
            RV32IMCycle::Advice(cycle) => cycle.ram_access.into(),
            RV32IMCycle::AssertEQ(cycle) => cycle.ram_access.into(),
            RV32IMCycle::AssertHalfwordAlignment(cycle) => cycle.ram_access.into(),
            RV32IMCycle::AssertLTE(cycle) => cycle.ram_access.into(),
            RV32IMCycle::AssertValidDiv0(cycle) => cycle.ram_access.into(),
            RV32IMCycle::AssertValidSignedRemainder(cycle) => cycle.ram_access.into(),
            RV32IMCycle::AssertValidUnsignedRemainder(cycle) => cycle.ram_access.into(),
            RV32IMCycle::Move(cycle) => cycle.ram_access.into(),
            RV32IMCycle::Movsign(cycle) => cycle.ram_access.into(),
            RV32IMCycle::Pow2(cycle) => cycle.ram_access.into(),
            RV32IMCycle::Pow2I(cycle) => cycle.ram_access.into(),
            RV32IMCycle::ShiftRightBitmask(cycle) => cycle.ram_access.into(),
            RV32IMCycle::ShiftRightBitmaskI(cycle) => cycle.ram_access.into(),
            RV32IMCycle::VirtualSRA(cycle) => cycle.ram_access.into(),
            RV32IMCycle::VirtualSRL(cycle) => cycle.ram_access.into(),
        }
    }

    pub fn rs1_read(&self) -> (usize, u64) {
        match self {
            RV32IMCycle::NoOp => (0, 0),
            RV32IMCycle::ADD(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::ADDI(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::AND(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::ANDI(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::AUIPC(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::BEQ(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::BGE(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::BGEU(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::BLT(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::BLTU(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::BNE(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::DIV(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::DIVU(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::FENCE(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::JAL(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::JALR(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::LB(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::LBU(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::LH(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::LHU(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::LUI(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::LW(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::MUL(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::MULH(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::MULHSU(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::MULHU(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::OR(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::ORI(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::REM(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::REMU(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::SB(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::SH(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::SLL(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::SLLI(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::SLT(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::SLTI(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::SLTIU(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::SLTU(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::SRA(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::SRAI(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::SRL(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::SRLI(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::SUB(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::SW(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::XOR(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::XORI(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::Advice(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::AssertEQ(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::AssertHalfwordAlignment(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::AssertLTE(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::AssertValidDiv0(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::AssertValidSignedRemainder(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::AssertValidUnsignedRemainder(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::Move(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::Movsign(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::Pow2(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::Pow2I(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::ShiftRightBitmask(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::ShiftRightBitmaskI(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::VirtualSRA(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
            RV32IMCycle::VirtualSRL(cycle) => (
                cycle.instruction.operands.rs1(),
                cycle.register_state.rs1_value(),
            ),
        }
    }

    pub fn rs2_read(&self) -> (usize, u64) {
        match self {
            RV32IMCycle::NoOp => (0, 0),
            RV32IMCycle::ADD(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::ADDI(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::AND(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::ANDI(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::AUIPC(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::BEQ(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::BGE(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::BGEU(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::BLT(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::BLTU(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::BNE(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::DIV(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::DIVU(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::FENCE(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::JAL(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::JALR(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::LB(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::LBU(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::LH(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::LHU(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::LUI(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::LW(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::MUL(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::MULH(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::MULHSU(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::MULHU(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::OR(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::ORI(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::REM(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::REMU(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::SB(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::SH(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::SLL(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::SLLI(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::SLT(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::SLTI(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::SLTIU(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::SLTU(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::SRA(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::SRAI(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::SRL(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::SRLI(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::SUB(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::SW(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::XOR(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::XORI(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::Advice(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::AssertEQ(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::AssertHalfwordAlignment(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::AssertLTE(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::AssertValidDiv0(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::AssertValidSignedRemainder(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::AssertValidUnsignedRemainder(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::Move(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::Movsign(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::Pow2(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::Pow2I(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::ShiftRightBitmask(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::ShiftRightBitmaskI(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::VirtualSRA(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
            RV32IMCycle::VirtualSRL(cycle) => (
                cycle.instruction.operands.rs2(),
                cycle.register_state.rs2_value(),
            ),
        }
    }

    pub fn rd_write(&self) -> (usize, u64, u64) {
        match self {
            RV32IMCycle::NoOp => (0, 0, 0),
            RV32IMCycle::ADD(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::ADDI(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::AND(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::ANDI(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::AUIPC(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::BEQ(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::BGE(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::BGEU(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::BLT(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::BLTU(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::BNE(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::DIV(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::DIVU(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::FENCE(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::JAL(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::JALR(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::LB(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::LBU(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::LH(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::LHU(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::LUI(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::LW(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::MUL(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::MULH(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::MULHSU(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::MULHU(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::OR(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::ORI(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::REM(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::REMU(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::SB(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::SH(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::SLL(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::SLLI(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::SLT(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::SLTI(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::SLTIU(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::SLTU(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::SRA(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::SRAI(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::SRL(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::SRLI(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::SUB(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::SW(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::XOR(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::XORI(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::Advice(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::AssertEQ(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::AssertHalfwordAlignment(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::AssertLTE(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::AssertValidDiv0(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::AssertValidSignedRemainder(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::AssertValidUnsignedRemainder(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::Move(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::Movsign(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::Pow2(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::Pow2I(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::ShiftRightBitmask(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::ShiftRightBitmaskI(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::VirtualSRA(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
            RV32IMCycle::VirtualSRL(cycle) => (
                cycle.instruction.operands.rd(),
                cycle.register_state.rd_values().0,
                cycle.register_state.rd_values().1,
            ),
        }
    }
}
