use add::ADD;
use addi::ADDI;
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
use ecall::ECALL;
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

use inline_sha256::sha256::SHA256;
use inline_sha256::sha256init::SHA256INIT;
use virtual_advice::VirtualAdvice;
use virtual_assert_eq::VirtualAssertEQ;
use virtual_assert_halfword_alignment::VirtualAssertHalfwordAlignment;
use virtual_assert_lte::VirtualAssertLTE;
use virtual_assert_valid_div0::VirtualAssertValidDiv0;
use virtual_assert_valid_signed_remainder::VirtualAssertValidSignedRemainder;
use virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder;
use virtual_move::VirtualMove;
use virtual_movsign::VirtualMovsign;
use virtual_muli::VirtualMULI;
use virtual_pow2::VirtualPow2;
use virtual_pow2i::VirtualPow2I;
use virtual_rotri::VirtualROTRI;
use virtual_shift_right_bitmask::VirtualShiftRightBitmask;
use virtual_shift_right_bitmaski::VirtualShiftRightBitmaskI;
use virtual_sra::VirtualSRA;
use virtual_srai::VirtualSRAI;
use virtual_srl::VirtualSRL;
use virtual_srli::VirtualSRLI;

use crate::emulator::cpu::Cpu;
use derive_more::From;
use format::{InstructionFormat, InstructionRegisterState, NormalizedOperands};

pub mod format;

pub mod instruction_macros;

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
pub mod ecall;
pub mod fence;
pub mod inline_sha256;
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
pub mod virtual_muli;
pub mod virtual_pow2;
pub mod virtual_pow2i;
pub mod virtual_rotri;
pub mod virtual_shift_right_bitmask;
pub mod virtual_shift_right_bitmaski;
pub mod virtual_sra;
pub mod virtual_srai;
pub mod virtual_srl;
pub mod virtual_srli;
pub mod xor;
pub mod xori;

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
        self.operands()
            .capture_pre_execution_state(&mut cycle.register_state, cpu);
        self.execute(cpu, &mut cycle.ram_access);
        self.operands()
            .capture_post_execution_state(&mut cycle.register_state, cpu);
        if let Some(trace_vec) = trace {
            trace_vec.push(cycle.into());
        }
    }
}

pub trait VirtualInstructionSequence: RISCVInstruction {
    fn virtual_sequence(&self) -> Vec<RV32IMInstruction>;
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
        ECALL, FENCE, JAL, JALR, LB, LBU, LH, LHU, LUI, LW, MUL, MULH, MULHSU,
        MULHU, OR, ORI, REM, REMU, SB, SH, SLL, SLLI, SLT, SLTI, SLTIU, SLTU,
        SRA, SRAI, SRL, SRLI, SUB, SW, XOR, XORI,
        // Virtual
        VirtualAdvice, VirtualAssertEQ, VirtualAssertHalfwordAlignment, VirtualAssertLTE,
        VirtualAssertValidDiv0, VirtualAssertValidSignedRemainder, VirtualAssertValidUnsignedRemainder,
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
            Some(0) => true,  // “anchor” of a virtual sequence
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
