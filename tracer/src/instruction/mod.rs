#![allow(clippy::upper_case_acronyms)]
use add::ADD;
use addi::ADDI;
use addiw::ADDIW;
use addw::ADDW;
use amoaddd::AMOADDD;
use amoaddw::AMOADDW;
use amoandd::AMOANDD;
use amoandw::AMOANDW;
use amomaxd::AMOMAXD;
use amomaxud::AMOMAXUD;
use amomaxuw::AMOMAXUW;
use amomaxw::AMOMAXW;
use amomind::AMOMIND;
use amominud::AMOMINUD;
use amominuw::AMOMINUW;
use amominw::AMOMINW;
use amoord::AMOORD;
use amoorw::AMOORW;
use amoswapd::AMOSWAPD;
use amoswapw::AMOSWAPW;
use amoxord::AMOXORD;
use amoxorw::AMOXORW;
use and::AND;
use andi::ANDI;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use auipc::AUIPC;
use beq::BEQ;
use bge::BGE;
use bgeu::BGEU;
use blt::BLT;
use bltu::BLTU;
use bne::BNE;
use div::DIV;
use divu::DIVU;
use divuw::DIVUW;
use divw::DIVW;
use ecall::ECALL;
use fence::FENCE;
use jal::JAL;
use jalr::JALR;
use lb::LB;
use lbu::LBU;
use ld::LD;
use lh::LH;
use lhu::LHU;
use lrd::LRD;
use lrw::LRW;
use lui::LUI;
use lw::LW;
use lwu::LWU;
use mul::MUL;
use mulh::MULH;
use mulhsu::MULHSU;
use mulhu::MULHU;
use mulw::MULW;
use or::OR;
use ori::ORI;
use rand::{rngs::StdRng, RngCore};
use rem::REM;
use remu::REMU;
use remuw::REMUW;
use remw::REMW;
use sb::SB;
use scd::SCD;
use scw::SCW;
use sd::SD;
use serde::{Deserialize, Serialize};
use sh::SH;
use sll::SLL;
use slli::SLLI;
use slliw::SLLIW;
use sllw::SLLW;
use slt::SLT;
use slti::SLTI;
use sltiu::SLTIU;
use sltu::SLTU;
use sra::SRA;
use srai::SRAI;
use sraiw::SRAIW;
use sraw::SRAW;
use srl::SRL;
use srli::SRLI;
use srliw::SRLIW;
use srlw::SRLW;
use strum_macros::{EnumCount as EnumCountMacro, EnumIter, IntoStaticStr};
use sub::SUB;
use subw::SUBW;
use sw::SW;
use xor::XOR;
use xori::XORI;

use inline_sha256::sha256::SHA256;
use inline_sha256::sha256init::SHA256INIT;
use virtual_advice::VirtualAdvice;
use virtual_assert_eq::VirtualAssertEQ;
use virtual_assert_halfword_alignment::VirtualAssertHalfwordAlignment;
use virtual_assert_lte::VirtualAssertLTE;
use virtual_assert_valid_div0::VirtualAssertValidDiv0;
use virtual_assert_valid_signed_remainder::VirtualAssertValidSignedRemainder;
use virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder;
use virtual_assert_word_alignment::VirtualAssertWordAlignment;
use virtual_change_divisor::VirtualChangeDivisor;
use virtual_change_divisor_w::VirtualChangeDivisorW;
use virtual_extend::VirtualExtend;
use virtual_lw::VirtualLW;
use virtual_move::VirtualMove;
use virtual_movsign::VirtualMovsign;
use virtual_muli::VirtualMULI;
use virtual_pow2::VirtualPow2;
use virtual_pow2_w::VirtualPow2W;
use virtual_pow2i::VirtualPow2I;
use virtual_pow2i_w::VirtualPow2IW;
use virtual_rotri::VirtualROTRI;
use virtual_shift_right_bitmask::VirtualShiftRightBitmask;
use virtual_shift_right_bitmaski::VirtualShiftRightBitmaskI;
use virtual_sign_extend::VirtualSignExtend;
use virtual_sra::VirtualSRA;
use virtual_srai::VirtualSRAI;
use virtual_srl::VirtualSRL;
use virtual_srli::VirtualSRLI;
use virtual_sw::VirtualSW;

use crate::emulator::cpu::Cpu;
use derive_more::From;
use format::{InstructionFormat, InstructionRegisterState, NormalizedOperands};

pub mod format;

pub mod instruction_macros;

pub mod add;
pub mod addi;
pub mod addiw;
pub mod addw;
pub mod amoaddd;
pub mod amoaddw;
pub mod amoandd;
pub mod amoandw;
pub mod amomaxd;
pub mod amomaxud;
pub mod amomaxuw;
pub mod amomaxw;
pub mod amomind;
pub mod amominud;
pub mod amominuw;
pub mod amominw;
pub mod amoord;
pub mod amoorw;
pub mod amoswapd;
pub mod amoswapw;
pub mod amoxord;
pub mod amoxorw;
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
pub mod ecall;
pub mod fence;
pub mod inline_sha256;
pub mod jal;
pub mod jalr;
pub mod lb;
pub mod lbu;
pub mod ld;
pub mod lh;
pub mod lhu;
pub mod lrd;
pub mod lrw;
pub mod lui;
pub mod lw;
pub mod lwu;
pub mod mul;
pub mod mulh;
pub mod mulhsu;
pub mod mulhu;
pub mod or;
pub mod ori;
pub mod rem;
pub mod remu;
pub mod sb;
pub mod scd;
pub mod scw;
pub mod sd;
pub mod sh;
pub mod sll;
pub mod slli;
pub mod slliw;
pub mod sllw;
pub mod slt;
pub mod slti;
pub mod sltiu;
pub mod sltu;
pub mod sra;
pub mod srai;
pub mod sraiw;
pub mod sraw;
pub mod srl;
pub mod srli;
pub mod srliw;
pub mod srlw;
pub mod sub;
pub mod subw;
pub mod sw;
pub mod virtual_advice;
pub mod virtual_assert_eq;
pub mod virtual_assert_halfword_alignment;
pub mod virtual_assert_lte;
pub mod virtual_assert_valid_div0;
pub mod virtual_assert_valid_signed_remainder;
pub mod virtual_assert_valid_unsigned_remainder;
pub mod virtual_assert_word_alignment;
pub mod virtual_change_divisor;
pub mod virtual_change_divisor_w;
pub mod virtual_extend;
pub mod virtual_lw;
pub mod virtual_move;
pub mod virtual_movsign;
pub mod virtual_muli;
pub mod virtual_pow2;
pub mod virtual_pow2_w;
pub mod virtual_pow2i;
pub mod virtual_pow2i_w;
pub mod virtual_rotri;
pub mod virtual_shift_right_bitmask;
pub mod virtual_shift_right_bitmaski;
pub mod virtual_sign_extend;
pub mod virtual_sra;
pub mod virtual_srai;
pub mod virtual_srl;
pub mod virtual_srli;
pub mod virtual_sw;
pub mod xor;
pub mod xori;

pub mod divuw;
pub mod divw;
pub mod mulw;
pub mod remuw;
pub mod remw;

#[cfg(test)]
pub mod test;

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RAMRead {
    pub address: u64,
    pub value: u64,
}

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RAMWrite {
    pub address: u64,
    pub pre_value: u64,
    pub post_value: u64,
}

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RAMAtomic {
    pub read: RAMRead,
    pub write: RAMWrite,
}

pub enum RAMAccess {
    Read(RAMRead),
    Write(RAMWrite),
    Atomic(RAMAtomic),
    NoOp,
}

impl RAMAccess {
    pub fn address(&self) -> usize {
        match self {
            RAMAccess::Read(read) => read.address as usize,
            RAMAccess::Write(write) => write.address as usize,
            RAMAccess::NoOp => 0,
            RAMAccess::Atomic(atomic) => atomic.read.address as usize,
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

impl From<RAMAtomic> for RAMAccess {
    fn from(atomic: RAMAtomic) -> Self {
        Self::Atomic(atomic)
    }
}

#[derive(Default)]
pub struct NormalizedInstruction {
    pub address: usize,
    pub operands: NormalizedOperands,
    pub virtual_sequence_remaining: Option<usize>,
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
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let mut cycle: RISCVCycle<Self> = RISCVCycle {
            instruction: *self,
            register_state: Default::default(),
            ram_access: Default::default(),
        };
        //self.operands()
        //    .capture_pre_execution_state(&mut cycle.register_state, cpu);
        self.execute(cpu, &mut cycle.ram_access);
        //self.operands()
        //    .capture_post_execution_state(&mut cycle.register_state, cpu);
        //if let Some(trace_vec) = trace {
        //    trace_vec.push(cycle.into());
        //}
    }
}

pub trait VirtualInstructionSequence: RISCVInstruction {
    fn virtual_sequence(&self, is_32: bool) -> Vec<RV32IMInstruction>;
}

macro_rules! define_rv32im_enums {
    (
        instructions: [$($instr:ident),* $(,)?]
    ) => {
        #[derive(Debug, IntoStaticStr, From, Clone, Serialize, Deserialize)]
        pub enum RV32IMInstruction {
            /// No-operation instruction (address)
            NoOp(usize),
            UNIMPL,
            $(
                $instr($instr),
            )*
        }

        #[derive(
            From, Debug, Copy, Clone, Serialize, Deserialize, IntoStaticStr, EnumIter, EnumCountMacro, PartialEq
        )]
        pub enum RV32IMCycle {
            /// No-operation cycle (address)
            NoOp(usize),
            $(
                $instr(RISCVCycle<$instr>),
            )*
        }

        impl RV32IMCycle {
            pub fn ram_access(&self) -> RAMAccess {
                match self {
                    RV32IMCycle::NoOp(_) => RAMAccess::NoOp,
                    $(
                        RV32IMCycle::$instr(cycle) => cycle.ram_access.into(),
                    )*
                }
            }

            pub fn rs1_read(&self) -> (usize, u64) {
                match self {
                    RV32IMCycle::NoOp(_) => (0, 0),
                    $(
                        RV32IMCycle::$instr(cycle) => (
                            cycle.instruction.operands.normalize().rs1,
                            cycle.register_state.rs1_value(),
                        ),
                    )*
                }
            }

            pub fn rs2_read(&self) -> (usize, u64) {
                match self {
                    RV32IMCycle::NoOp(_) => (0, 0),
                    $(
                        RV32IMCycle::$instr(cycle) => (
                            cycle.instruction.operands.normalize().rs2,
                            cycle.register_state.rs2_value(),
                        ),
                    )*
                }
            }

            pub fn rd_write(&self) -> (usize, u64, u64) {
                match self {
                    RV32IMCycle::NoOp(_) => (0, 0, 0),
                    $(
                        RV32IMCycle::$instr(cycle) => (
                            cycle.instruction.operands.normalize().rd,
                            cycle.register_state.rd_values().0,
                            cycle.register_state.rd_values().1,
                        ),
                    )*
                }
            }

            pub fn instruction(&self) -> RV32IMInstruction {
                match self {
                    RV32IMCycle::NoOp(address) => RV32IMInstruction::NoOp(*address),
                    $(
                        RV32IMCycle::$instr(cycle) => cycle.instruction.into(),
                    )*
                }
            }
        }

        impl RV32IMInstruction {
            pub fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
                match self {
                    RV32IMInstruction::NoOp(_) => panic!("Unsupported instruction: {:?}", self),
                    RV32IMInstruction::UNIMPL => panic!("Unsupported instruction: {:?}", self),
                    $(
                        RV32IMInstruction::$instr(instr) => instr.trace(cpu, trace),
                    )*
                }
            }

            pub fn execute(&self, cpu: &mut Cpu) {
                match self {
                    RV32IMInstruction::NoOp(_) => panic!("Unsupported instruction: {:?}", self),
                    RV32IMInstruction::UNIMPL => panic!("Unsupported instruction: {:?}", self),
                    $(
                        RV32IMInstruction::$instr(instr) => {
                            let mut cycle: RISCVCycle<$instr> = RISCVCycle {
                                instruction: *instr,
                                register_state: Default::default(),
                                ram_access: Default::default(),
                            };
                            instr.execute(cpu, &mut cycle.ram_access);
                        }
                    )*
                }
            }

            pub fn normalize(&self) -> NormalizedInstruction {
                match self {
                    RV32IMInstruction::NoOp(address) => {
                        NormalizedInstruction {
                            address: *address,
                            ..Default::default()
                        }
                    },
                    RV32IMInstruction::UNIMPL => Default::default(),
                    $(
                        RV32IMInstruction::$instr(instr) => NormalizedInstruction {
                            address: instr.address as usize,
                            operands: instr.operands.normalize(),
                            virtual_sequence_remaining: instr.virtual_sequence_remaining,
                        },
                    )*
                }
            }

            pub fn set_virtual_sequence_remaining(&mut self, remaining: Option<usize>) {
                match self {
                    RV32IMInstruction::NoOp(_) => (),
                    RV32IMInstruction::UNIMPL => (),
                    $(
                        RV32IMInstruction::$instr(instr) => {instr.virtual_sequence_remaining = remaining;}
                    )*
                }
            }
        }
    };
}

define_rv32im_enums! {
    instructions: [
        ADD, ADDI, AND, ANDI, AUIPC, BEQ, BGE, BGEU, BLT, BLTU, BNE, DIV, DIVU,
        ECALL, FENCE, JAL, JALR, LB, LBU, LD, LH, LHU, LUI, LW, MUL, MULH, MULHSU,
        MULHU, OR, ORI, REM, REMU, SB, SD, SH, SLL, SLLI, SLT, SLTI, SLTIU, SLTU,
        SRA, SRAI, SRL, SRLI, SUB, SW, XOR, XORI,
        // RV64I
        ADDIW, SLLIW, SRLIW, SRAIW, ADDW, SUBW, SLLW, SRLW, SRAW, LWU,
        // RV64M
        DIVUW, DIVW, MULW, REMUW, REMW,
        // RV32A (Atomic Memory Operations)
        LRW, SCW, AMOSWAPW, AMOADDW, AMOANDW, AMOORW, AMOXORW, AMOMINW, AMOMAXW, AMOMINUW, AMOMAXUW,
        // RV64A (Atomic Memory Operations)
        LRD, SCD, AMOSWAPD, AMOADDD, AMOANDD, AMOORD, AMOXORD, AMOMIND, AMOMAXD, AMOMINUD, AMOMAXUD,
        // Virtual
        VirtualAdvice, VirtualAssertEQ, VirtualAssertHalfwordAlignment, VirtualAssertWordAlignment, VirtualAssertLTE,
        VirtualAssertValidDiv0, VirtualAssertValidSignedRemainder, VirtualAssertValidUnsignedRemainder,
        VirtualChangeDivisor, VirtualChangeDivisorW, VirtualLW,VirtualSW,VirtualExtend,
        VirtualSignExtend,VirtualPow2W, VirtualPow2IW,
        VirtualMove, VirtualMovsign, VirtualMULI, VirtualPow2, VirtualPow2I, VirtualROTRI,
        VirtualShiftRightBitmask, VirtualShiftRightBitmaskI,
        VirtualSRA, VirtualSRAI, VirtualSRL, VirtualSRLI,
        // Extension
        SHA256, SHA256INIT,
    ]
}

impl CanonicalSerialize for RV32IMInstruction {
    fn serialize_with_mode<W: ark_serialize::Write>(
        &self,
        mut writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        let bytes = serde_json::to_vec(self).map_err(|_| SerializationError::InvalidData)?;
        let len: u64 = bytes.len() as u64;
        len.serialize_with_mode(&mut writer, _compress)?;
        writer
            .write_all(&bytes)
            .map_err(|_| SerializationError::InvalidData)?;
        Ok(())
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        let bytes = serde_json::to_vec(self).expect("serialization failed");
        bytes.len() + 8 // 8 bytes for length
    }
}

impl CanonicalDeserialize for RV32IMInstruction {
    fn deserialize_with_mode<R: ark_serialize::Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let len = u64::deserialize_with_mode(&mut reader, compress, validate)?;
        let mut bytes = vec![0u8; len as usize];
        reader
            .read_exact(&mut bytes)
            .map_err(|_| SerializationError::InvalidData)?;
        serde_json::from_slice(&bytes).map_err(|e| {
            println!("Deserialization error: {e}");
            SerializationError::InvalidData
        })
    }
}

impl Valid for RV32IMInstruction {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl RV32IMInstruction {
    pub fn is_real(&self) -> bool {
        // ignore no-op
        if matches!(self, RV32IMInstruction::NoOp(_)) {
            return false;
        }

        match self.normalize().virtual_sequence_remaining {
            None => true,     // ordinary instruction
            Some(0) => true,  // "anchor" of a virtual sequence
            Some(_) => false, // helper within the sequence
        }
    }

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
                // Load instructions (I-type): LB, LH, LW, LBU, LHU, LD, LWU.
                match (instr >> 12) & 0x7 {
                    0b000 => Ok(LB::new(instr, address, true).into()),
                    0b001 => Ok(LH::new(instr, address, true).into()),
                    0b010 => Ok(LW::new(instr, address, true).into()),
                    0b011 => Ok(LD::new(instr, address, true).into()),
                    0b100 => Ok(LBU::new(instr, address, true).into()),
                    0b101 => Ok(LHU::new(instr, address, true).into()),
                    0b110 => Ok(LWU::new(instr, address, true).into()),
                    _ => Err("Invalid load funct3"),
                }
            }
            0b0100011 => {
                // Store instructions (S-type): SB, SH, SW.
                match (instr >> 12) & 0x7 {
                    0b000 => Ok(SB::new(instr, address, true).into()),
                    0b001 => Ok(SH::new(instr, address, true).into()),
                    0b010 => Ok(SW::new(instr, address, true).into()),
                    0b011 => Ok(SD::new(instr, address, true).into()),
                    _ => Err("Invalid store funct3"),
                }
            }
            0b0010011 => {
                // I-type arithmetic instructions: ADDI, SLTI, SLTIU, XORI, ORI, ANDI,
                // and also shift-immediate instructions SLLI, SRLI, SRAI.
                let funct3 = (instr >> 12) & 0x7;
                let funct6 = (instr >> 26) & 0x3f;
                if funct3 == 0b001 {
                    // SLLI uses shamt and expects funct6 == 0.
                    if funct6 == 0 {
                        Ok(SLLI::new(instr, address, true).into())
                    } else {
                        Err("Invalid funct7 for SLLI")
                    }
                } else if funct3 == 0b101 {
                    if funct6 == 0b000000 {
                        Ok(SRLI::new(instr, address, true).into())
                    } else if funct6 == 0b010000 {
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
            0b0011011 => {
                // RV64I I-type arithmetic instructions.
                let funct3 = (instr >> 12) & 0x7;
                let funct7 = (instr >> 25) & 0x7f;
                match (funct3, funct7) {
                    (0b000, _) => Ok(ADDIW::new(instr, address, true).into()),
                    (0b001, 0b0000000) => Ok(SLLIW::new(instr, address, true).into()),
                    (0b101, 0b0000000) => Ok(SRLIW::new(instr, address, true).into()),
                    (0b101, 0b0100000) => Ok(SRAIW::new(instr, address, true).into()),
                    _ => Err("Invalid RV64I I-type arithmetic instruction"),
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
            0b0111011 => {
                // RV64I R-type arithmetic instructions.
                let funct3 = (instr >> 12) & 0x7;
                let funct7 = (instr >> 25) & 0x7f;
                match (funct3, funct7) {
                    (0b000, 0b0000000) => Ok(ADDW::new(instr, address, true).into()),
                    (0b000, 0b0100000) => Ok(SUBW::new(instr, address, true).into()),
                    (0b001, 0b0000000) => Ok(SLLW::new(instr, address, true).into()),
                    (0b100, 0b0000001) => Ok(DIVW::new(instr, address, true).into()),
                    (0b101, 0b0000000) => Ok(SRLW::new(instr, address, true).into()),
                    (0b101, 0b0100000) => Ok(SRAW::new(instr, address, true).into()),
                    (0b000, 0b0000001) => Ok(MULW::new(instr, address, true).into()),
                    (0b101, 0b0000001) => Ok(DIVUW::new(instr, address, true).into()),
                    (0b110, 0b0000001) => Ok(REMW::new(instr, address, true).into()),
                    (0b111, 0b0000001) => Ok(REMUW::new(instr, address, true).into()),
                    _ => Err("Invalid RV64I R-type arithmetic instruction"),
                }
            }
            0b0001111 => {
                // FENCE: I-type; the immediate encodes "pred" and "succ" flags.
                Ok(FENCE::new(instr, address, true).into())
            }
            0b0101111 => {
                // Atomic Memory Operations (A-extension): LR, SC, AMOSWAP, AMOADD, etc.
                // Only check funct3 (width) and funct5 (operation type)
                // bits [26:25] are aq/rl flags which can vary
                let funct3 = (instr >> 12) & 0x7;
                let funct5 = (instr >> 27) & 0x1f;

                match (funct3, funct5) {
                    // LR (Load Reserved)
                    (0b010, 0b00010) => Ok(LRW::new(instr, address, true).into()),
                    (0b011, 0b00010) => Ok(LRD::new(instr, address, true).into()),

                    // SC (Store Conditional)
                    (0b010, 0b00011) => Ok(SCW::new(instr, address, true).into()),
                    (0b011, 0b00011) => Ok(SCD::new(instr, address, true).into()),

                    // AMOSWAP
                    (0b010, 0b00001) => Ok(AMOSWAPW::new(instr, address, true).into()),
                    (0b011, 0b00001) => Ok(AMOSWAPD::new(instr, address, true).into()),

                    // AMOADD
                    (0b010, 0b00000) => Ok(AMOADDW::new(instr, address, true).into()),
                    (0b011, 0b00000) => Ok(AMOADDD::new(instr, address, true).into()),

                    // AMOAND
                    (0b010, 0b01100) => Ok(AMOANDW::new(instr, address, true).into()),
                    (0b011, 0b01100) => Ok(AMOANDD::new(instr, address, true).into()),

                    // AMOOR
                    (0b010, 0b01000) => Ok(AMOORW::new(instr, address, true).into()),
                    (0b011, 0b01000) => Ok(AMOORD::new(instr, address, true).into()),

                    // AMOXOR
                    (0b010, 0b00100) => Ok(AMOXORW::new(instr, address, true).into()),
                    (0b011, 0b00100) => Ok(AMOXORD::new(instr, address, true).into()),

                    // AMOMIN
                    (0b010, 0b10000) => Ok(AMOMINW::new(instr, address, true).into()),
                    (0b011, 0b10000) => Ok(AMOMIND::new(instr, address, true).into()),

                    // AMOMAX
                    (0b010, 0b10100) => Ok(AMOMAXW::new(instr, address, true).into()),
                    (0b011, 0b10100) => Ok(AMOMAXD::new(instr, address, true).into()),

                    // AMOMINU
                    (0b010, 0b11000) => Ok(AMOMINUW::new(instr, address, true).into()),
                    (0b011, 0b11000) => Ok(AMOMINUD::new(instr, address, true).into()),

                    // AMOMAXU
                    (0b010, 0b11100) => Ok(AMOMAXUW::new(instr, address, true).into()),
                    (0b011, 0b11100) => Ok(AMOMAXUD::new(instr, address, true).into()),

                    _ => {
                        eprintln!("Invalid atomic memory operation: instr=0x{instr:08x} funct3={funct3:03b} funct5={funct5:05b}");
                        Err("Invalid atomic memory operation")
                    }
                }
            }
            0b1110011 => {
                // For now this only (potentially) maps to ECALL.
                if instr == ECALL::MATCH {
                    Ok(ECALL::new(instr, address, true).into())
                } else {
                    Err("Unsupported SYSTEM instruction")
                }
            }
            // 0x0B is reserved for RISC-V extension
            // In attempt to standardize this space for precompiles and inlines,
            // each new type of operation should be placed under different funct7,
            // while funct3 should hold all necessary instructions for that operation.
            // funct7:
            // - 0x00: SHA256
            0b0001011 => {
                // Custom-0 opcode: SHA256 compression instructions
                let funct3 = (instr >> 12) & 0x7;
                let funct7 = (instr >> 25) & 0x7f;
                if funct7 == 0x00 {
                    match funct3 {
                        0x0 => Ok(SHA256::new(instr, address, true).into()),
                        0x1 => Ok(SHA256INIT::new(instr, address, true).into()),
                        _ => Err("Unknown funct3 for custom SHA256 instruction"),
                    }
                } else {
                    Err("Unknown funct7 for custom-0 opcode")
                }
            }
            _ => Err("Unknown opcode"),
        }
    }
}

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
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

impl RV32IMCycle {
    pub fn last_jalr(address: usize) -> Self {
        Self::JALR(RISCVCycle {
            instruction: JALR {
                address: address as u64,
                ..Default::default()
            },
            ..Default::default()
        })
    }
}
