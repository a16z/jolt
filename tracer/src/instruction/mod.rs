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
use andn::ANDN;
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

use virtual_advice::VirtualAdvice;
use virtual_assert_eq::VirtualAssertEQ;
use virtual_assert_halfword_alignment::VirtualAssertHalfwordAlignment;
use virtual_assert_lte::VirtualAssertLTE;
use virtual_assert_mulu_no_overflow::VirtualAssertMulUNoOverflow;
use virtual_assert_valid_div0::VirtualAssertValidDiv0;
use virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder;
use virtual_assert_word_alignment::VirtualAssertWordAlignment;
use virtual_change_divisor::VirtualChangeDivisor;
use virtual_change_divisor_w::VirtualChangeDivisorW;
use virtual_lw::VirtualLW;
use virtual_movsign::VirtualMovsign;
use virtual_muli::VirtualMULI;
use virtual_pow2::VirtualPow2;
use virtual_pow2_w::VirtualPow2W;
use virtual_pow2i::VirtualPow2I;
use virtual_pow2i_w::VirtualPow2IW;
use virtual_rev8w::VirtualRev8W;
use virtual_rotri::VirtualROTRI;
use virtual_rotriw::VirtualROTRIW;
use virtual_shift_right_bitmask::VirtualShiftRightBitmask;
use virtual_shift_right_bitmaski::VirtualShiftRightBitmaskI;
use virtual_sign_extend_word::VirtualSignExtendWord;
use virtual_sra::VirtualSRA;
use virtual_srai::VirtualSRAI;
use virtual_srl::VirtualSRL;
use virtual_srli::VirtualSRLI;
use virtual_sw::VirtualSW;
use virtual_xor_rot::{VirtualXORROT16, VirtualXORROT24, VirtualXORROT32, VirtualXORROT63};
use virtual_xor_rotw::{VirtualXORROTW12, VirtualXORROTW16, VirtualXORROTW7, VirtualXORROTW8};
use virtual_zero_extend_word::VirtualZeroExtendWord;

use self::inline::INLINE;

use crate::emulator::cpu::{Cpu, Xlen};
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use derive_more::From;
use format::{InstructionFormat, InstructionRegisterState, NormalizedOperands};

pub mod format;

pub use crate::utils::instruction_macros;

pub(super) mod amo;

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
pub mod andn;
pub mod auipc;
pub mod beq;
pub mod bge;
pub mod bgeu;
pub mod blt;
pub mod bltu;
pub mod bne;
pub mod div;
pub mod divu;
pub mod divuw;
pub mod divw;
pub mod ecall;
pub mod fence;
pub mod inline;
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
pub mod mulw;
pub mod or;
pub mod ori;
pub mod rem;
pub mod remu;
pub mod remuw;
pub mod remw;
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
pub mod virtual_assert_mulu_no_overflow;
pub mod virtual_assert_valid_div0;
pub mod virtual_assert_valid_unsigned_remainder;
pub mod virtual_assert_word_alignment;
pub mod virtual_change_divisor;
pub mod virtual_change_divisor_w;
pub mod virtual_lw;
pub mod virtual_movsign;
pub mod virtual_muli;
pub mod virtual_pow2;
pub mod virtual_pow2_w;
pub mod virtual_pow2i;
pub mod virtual_pow2i_w;
pub mod virtual_rev8w;
pub mod virtual_rotri;
pub mod virtual_rotriw;
pub mod virtual_shift_right_bitmask;
pub mod virtual_shift_right_bitmaski;
pub mod virtual_sign_extend_word;
pub mod virtual_sra;
pub mod virtual_srai;
pub mod virtual_srl;
pub mod virtual_srli;
pub mod virtual_sw;
pub mod virtual_xor_rot;
pub mod virtual_xor_rotw;
pub mod virtual_zero_extend_word;
pub mod xor;
pub mod xori;

#[cfg(any(test, feature = "test-utils"))]
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
    pub virtual_sequence_remaining: Option<u16>,
    pub is_first_in_sequence: bool,
    pub is_compressed: bool,
}

pub trait RISCVInstruction:
    std::fmt::Debug
    + Sized
    + Copy
    + Into<Instruction>
    + From<NormalizedInstruction>
    + Into<NormalizedInstruction>
{
    const MASK: u32;
    const MATCH: u32;

    type Format: InstructionFormat;
    type RAMAccess: Default + Into<RAMAccess> + Copy + std::fmt::Debug;

    fn operands(&self) -> &Self::Format;
    fn new(word: u32, address: u64, validate: bool, compressed: bool) -> Self;
    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng) -> Self {
        use rand::RngCore;
        Self::new(rng.next_u32(), rng.next_u64(), false, false)
    }

    fn execute(&self, cpu: &mut Cpu, ram_access: &mut Self::RAMAccess);
}

pub trait RISCVTrace: RISCVInstruction
where
    RISCVCycle<Self>: Into<Cycle>,
{
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
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
    // Default implementation. Instructions with inline sequences will override this.
    fn inline_sequence(
        &self,
        _vr_allocator: &VirtualRegisterAllocator,
        _xlen: Xlen,
    ) -> Vec<Instruction> {
        vec![(*self).into()]
    }
}

macro_rules! define_rv32im_enums {
    (
        instructions: [$($instr:ident),* $(,)?]
    ) => {
        #[derive(Debug, IntoStaticStr, From, Clone, Serialize, Deserialize, EnumIter)]
        pub enum Instruction {
            /// No-operation instruction (address)
            NoOp,
            UNIMPL,
            $(
                $instr($instr),
            )*
            /// Inline instruction from external crates
            INLINE(INLINE),
        }

        #[derive(
            From, Debug, Copy, Clone, Serialize, Deserialize, IntoStaticStr, EnumIter, EnumCountMacro, PartialEq
        )]
        pub enum Cycle {
            /// No-operation cycle (address)
            NoOp,
            $(
                $instr(RISCVCycle<$instr>),
            )*
            INLINE(RISCVCycle<INLINE>),
        }

        impl Cycle {
            pub fn ram_access(&self) -> RAMAccess {
                match self {
                    Cycle::NoOp => RAMAccess::NoOp,
                    $(
                        Cycle::$instr(cycle) => cycle.ram_access.into(),
                    )*
                    Cycle::INLINE(cycle) => cycle.ram_access.into(),
                }
            }

            pub fn rs1_read(&self) -> Option<(u8, u64)> {
                match self {
                    Cycle::NoOp => None,
                    $(
                        Cycle::$instr(cycle) => {
                            if let Some(rs1_val) = cycle.register_state.rs1_value() {
                                Some((
                                    NormalizedOperands::from(cycle.instruction.operands).rs1.unwrap(),
                                    rs1_val,
                                ))
                            } else {
                                None
                            }
                        },
                    )*
                    Cycle::INLINE(cycle) => {
                        if let Some(rs1_val) = cycle.register_state.rs1_value() {
                            Some((
                                cycle.instruction.operands.rs1,
                                rs1_val,
                            ))
                        } else {
                            None
                        }
                    },
                }
            }

            pub fn rs2_read(&self) -> Option<(u8, u64)> {
                match self {
                    Cycle::NoOp => None,
                    $(
                        Cycle::$instr(cycle) => {
                            if let Some(rs2_val) = cycle.register_state.rs2_value() {
                                Some((
                                    NormalizedOperands::from(cycle.instruction.operands).rs2.unwrap(),
                                    rs2_val,
                                ))
                            } else {
                                None
                            }
                        },
                    )*
                    Cycle::INLINE(cycle) => {
                        if let Some(rs2_val) = cycle.register_state.rs2_value() {
                            Some((
                                cycle.instruction.operands.rs2,
                                rs2_val,
                            ))
                        } else {
                            None
                        }
                    },
                }
            }

            pub fn rd_write(&self) -> Option<(u8, u64, u64)> {
                match self {
                    Cycle::NoOp => None,
                    $(
                        Cycle::$instr(cycle) => {
                            if let Some((rd_pre_val, rd_post_val)) = cycle.register_state.rd_values() {
                                Some((
                                    NormalizedOperands::from(cycle.instruction.operands).rd.unwrap(),
                                    rd_pre_val,
                                    rd_post_val,
                                ))
                            } else {
                                None
                            }
                        },
                    )*
                    Cycle::INLINE(cycle) => {
                        if let Some((rd_pre_val, rd_post_val)) = cycle.register_state.rd_values() {
                            Some((
                                cycle.instruction.operands.rs3,
                                rd_pre_val,
                                rd_post_val,
                            ))
                        } else {
                            None
                        }
                    },
                }
            }

            pub fn instruction(&self) -> Instruction {
                match self {
                    Cycle::NoOp => Instruction::NoOp,
                    $(
                        Cycle::$instr(cycle) => cycle.instruction.into(),
                    )*
                    Cycle::INLINE(cycle) => cycle.instruction.into(),
                }
            }
        }

        impl Instruction {
            pub fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
                match self {
                    Instruction::NoOp => panic!("Unsupported instruction: {:?}", self),
                    Instruction::UNIMPL => panic!("Unsupported instruction: {:?}", self),
                    $(
                        Instruction::$instr(instr) => instr.trace(cpu, trace),
                    )*
                    Instruction::INLINE(instr) => instr.trace(cpu, trace),
                }
            }

            pub fn execute(&self, cpu: &mut Cpu) {
                match self {
                    Instruction::NoOp => panic!("Unsupported instruction: {:?}", self),
                    Instruction::UNIMPL => panic!("Unsupported instruction: {:?}", self),
                    $(
                        Instruction::$instr(instr) => {
                            let mut cycle: RISCVCycle<$instr> = RISCVCycle {
                                instruction: *instr,
                                register_state: Default::default(),
                                ram_access: Default::default(),
                            };
                            instr.execute(cpu, &mut cycle.ram_access);
                        }
                    )*
                    Instruction::INLINE(instr) => {
                        let mut cycle: RISCVCycle<INLINE> = RISCVCycle {
                            instruction: *instr,
                            register_state: Default::default(),
                            ram_access: Default::default(),
                        };
                        instr.execute(cpu, &mut cycle.ram_access);
                    }
                }
            }

            pub fn normalize(&self) -> NormalizedInstruction {
                self.into()
            }

            pub fn inline_sequence(&self, allocator: &VirtualRegisterAllocator, xlen: Xlen) -> Vec<Instruction> {
                match self {
                    Instruction::NoOp => vec![],
                    Instruction::UNIMPL => vec![],
                    $(
                        Instruction::$instr(instr) => instr.inline_sequence(allocator, xlen),
                    )*
                    Instruction::INLINE(instr) => instr.inline_sequence(allocator, xlen),
                }
            }

            pub fn set_virtual_sequence_remaining(&mut self, remaining: Option<u16>) {
                match self {
                    Instruction::NoOp => (),
                    Instruction::UNIMPL => (),
                    $(
                        Instruction::$instr(instr) => {instr.virtual_sequence_remaining = remaining;}
                    )*
                    Instruction::INLINE(instr) => {instr.virtual_sequence_remaining = remaining;}
                }
            }

            pub fn set_is_first_in_sequence(&mut self, is_first: bool) {
                match self {
                    Instruction::NoOp => (),
                    Instruction::UNIMPL => (),
                    $(
                        Instruction::$instr(instr) => {instr.is_first_in_sequence = is_first;}
                    )*
                    Instruction::INLINE(instr) => {instr.is_first_in_sequence = is_first;}
                }
            }

            pub fn set_is_compressed(&mut self, is_compressed: bool) {
                match self {
                    Instruction::NoOp => (),
                    Instruction::UNIMPL => (),
                    $(
                        Instruction::$instr(instr) => {instr.is_compressed = is_compressed;}
                    )*
                    Instruction::INLINE(instr) => {instr.is_compressed = is_compressed;}
                }
            }
        }

        impl From<&Instruction> for NormalizedInstruction {
            fn from(instr: &Instruction) -> Self {
                match instr {
                    Instruction::NoOp => Default::default(),
                    Instruction::UNIMPL => Default::default(),
                    $(
                        Instruction::$instr(instr) => NormalizedInstruction {
                            address: instr.address as usize,
                            operands: instr.operands.into(),
                            virtual_sequence_remaining: instr.virtual_sequence_remaining,
                            is_first_in_sequence: instr.is_first_in_sequence,
                            is_compressed: instr.is_compressed,
                        },
                    )*
                    Instruction::INLINE(instr) => NormalizedInstruction {
                        address: instr.address as usize,
                        operands: instr.operands.into(),
                        virtual_sequence_remaining: instr.virtual_sequence_remaining,
                        is_first_in_sequence: instr.is_first_in_sequence,
                        is_compressed: instr.is_compressed,
                    },
                }
            }
        }
    };
}

define_rv32im_enums! {
    instructions: [
        ADD, ADDI, AND, ANDI, ANDN, AUIPC, BEQ, BGE, BGEU, BLT, BLTU, BNE, DIV, DIVU,
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
        VirtualAssertValidDiv0, VirtualAssertValidUnsignedRemainder, VirtualAssertMulUNoOverflow,
        VirtualChangeDivisor, VirtualChangeDivisorW, VirtualLW,VirtualSW, VirtualZeroExtendWord,
        VirtualSignExtendWord,VirtualPow2W, VirtualPow2IW,
        VirtualMovsign, VirtualMULI, VirtualPow2, VirtualPow2I, VirtualRev8W, VirtualROTRI,
        VirtualROTRIW,
        VirtualShiftRightBitmask, VirtualShiftRightBitmaskI,
        VirtualSRA, VirtualSRAI, VirtualSRL, VirtualSRLI,
        // XORROT
        VirtualXORROT32, VirtualXORROT24, VirtualXORROT16, VirtualXORROT63,
        VirtualXORROTW16, VirtualXORROTW12, VirtualXORROTW8, VirtualXORROTW7,
    ]
}

impl CanonicalSerialize for Instruction {
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

impl CanonicalDeserialize for Instruction {
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

impl Valid for Instruction {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl Instruction {
    pub fn is_real(&self) -> bool {
        // ignore no-op
        if matches!(self, Instruction::NoOp) {
            return false;
        }

        match self.normalize().virtual_sequence_remaining {
            None => true,     // ordinary instruction
            Some(0) => true,  // "anchor" of a inline sequence
            Some(_) => false, // helper within the sequence
        }
    }

    pub fn decode(instr: u32, address: u64, compressed: bool) -> Result<Self, &'static str> {
        let opcode = instr & 0x7f;
        match opcode {
            0b0110111 => {
                // LUI: U-type => [imm(31:12), rd, opcode]
                Ok(LUI::new(instr, address, true, compressed).into())
            }
            0b0010111 => {
                // AUIPC: U-type => [imm(31:12), rd, opcode]
                Ok(AUIPC::new(instr, address, true, compressed).into())
            }
            0b1101111 => {
                // JAL: UJ-type instruction.
                Ok(JAL::new(instr, address, true, compressed).into())
            }
            0b1100111 => {
                // JALR: I-type, where funct3 must be 0.
                let funct3 = (instr >> 12) & 0x7;
                if funct3 != 0 {
                    return Err("Invalid funct3 for JALR");
                }
                Ok(JALR::new(instr, address, true, compressed).into())
            }
            0b1100011 => {
                // Branch instructions (SB-type): BEQ, BNE, BLT, BGE, BLTU, BGEU.
                match (instr >> 12) & 0x7 {
                    0b000 => Ok(BEQ::new(instr, address, true, compressed).into()),
                    0b001 => Ok(BNE::new(instr, address, true, compressed).into()),
                    0b100 => Ok(BLT::new(instr, address, true, compressed).into()),
                    0b101 => Ok(BGE::new(instr, address, true, compressed).into()),
                    0b110 => Ok(BLTU::new(instr, address, true, compressed).into()),
                    0b111 => Ok(BGEU::new(instr, address, true, compressed).into()),
                    _ => Err("Invalid branch funct3"),
                }
            }
            0b0000011 => {
                // Load instructions (I-type): LB, LH, LW, LBU, LHU, LD, LWU.
                match (instr >> 12) & 0x7 {
                    0b000 => Ok(LB::new(instr, address, true, compressed).into()),
                    0b001 => Ok(LH::new(instr, address, true, compressed).into()),
                    0b010 => Ok(LW::new(instr, address, true, compressed).into()),
                    0b011 => Ok(LD::new(instr, address, true, compressed).into()),
                    0b100 => Ok(LBU::new(instr, address, true, compressed).into()),
                    0b101 => Ok(LHU::new(instr, address, true, compressed).into()),
                    0b110 => Ok(LWU::new(instr, address, true, compressed).into()),
                    _ => Err("Invalid load funct3"),
                }
            }
            0b0100011 => {
                // Store instructions (S-type): SB, SH, SW.
                match (instr >> 12) & 0x7 {
                    0b000 => Ok(SB::new(instr, address, true, compressed).into()),
                    0b001 => Ok(SH::new(instr, address, true, compressed).into()),
                    0b010 => Ok(SW::new(instr, address, true, compressed).into()),
                    0b011 => Ok(SD::new(instr, address, true, compressed).into()),
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
                        Ok(SLLI::new(instr, address, true, compressed).into())
                    } else {
                        Err("Invalid funct7 for SLLI")
                    }
                } else if funct3 == 0b101 {
                    if funct6 == 0b000000 {
                        Ok(SRLI::new(instr, address, true, compressed).into())
                    } else if funct6 == 0b010000 {
                        Ok(SRAI::new(instr, address, true, compressed).into())
                    } else {
                        Err("Invalid ALU shift funct7")
                    }
                } else {
                    match funct3 {
                        0b000 => Ok(ADDI::new(instr, address, true, compressed).into()),
                        0b010 => Ok(SLTI::new(instr, address, true, compressed).into()),
                        0b011 => Ok(SLTIU::new(instr, address, true, compressed).into()),
                        0b100 => Ok(XORI::new(instr, address, true, compressed).into()),
                        0b110 => Ok(ORI::new(instr, address, true, compressed).into()),
                        0b111 => Ok(ANDI::new(instr, address, true, compressed).into()),
                        _ => Err("Invalid I-type ALU funct3"),
                    }
                }
            }
            0b0011011 => {
                // RV64I I-type arithmetic instructions.
                let funct3 = (instr >> 12) & 0x7;
                let funct7 = (instr >> 25) & 0x7f;
                match (funct3, funct7) {
                    (0b000, _) => Ok(ADDIW::new(instr, address, true, compressed).into()),
                    (0b001, 0b0000000) => Ok(SLLIW::new(instr, address, true, compressed).into()),
                    (0b101, 0b0000000) => Ok(SRLIW::new(instr, address, true, compressed).into()),
                    (0b101, 0b0100000) => Ok(SRAIW::new(instr, address, true, compressed).into()),
                    _ => Err("Invalid RV64I I-type arithmetic instruction"),
                }
            }
            0b0110011 => {
                // R-type arithmetic instructions.
                let funct3 = (instr >> 12) & 0x7;
                let funct7 = (instr >> 25) & 0x7f;
                match (funct3, funct7) {
                    (0b000, 0b0000000) => Ok(ADD::new(instr, address, true, compressed).into()),
                    (0b000, 0b0100000) => Ok(SUB::new(instr, address, true, compressed).into()),
                    (0b001, 0b0000000) => Ok(SLL::new(instr, address, true, compressed).into()),
                    (0b010, 0b0000000) => Ok(SLT::new(instr, address, true, compressed).into()),
                    (0b011, 0b0000000) => Ok(SLTU::new(instr, address, true, compressed).into()),
                    (0b100, 0b0000000) => Ok(XOR::new(instr, address, true, compressed).into()),
                    (0b101, 0b0000000) => Ok(SRL::new(instr, address, true, compressed).into()),
                    (0b101, 0b0100000) => Ok(SRA::new(instr, address, true, compressed).into()),
                    (0b110, 0b0000000) => Ok(OR::new(instr, address, true, compressed).into()),
                    (0b111, 0b0000000) => Ok(AND::new(instr, address, true, compressed).into()),

                    // RV32M extension
                    (0b000, 0b0000001) => Ok(MUL::new(instr, address, true, compressed).into()),
                    (0b001, 0b0000001) => Ok(MULH::new(instr, address, true, compressed).into()),
                    (0b010, 0b0000001) => Ok(MULHSU::new(instr, address, true, compressed).into()),
                    (0b011, 0b0000001) => Ok(MULHU::new(instr, address, true, compressed).into()),
                    (0b100, 0b0000001) => Ok(DIV::new(instr, address, true, compressed).into()),
                    (0b101, 0b0000001) => Ok(DIVU::new(instr, address, true, compressed).into()),
                    (0b110, 0b0000001) => Ok(REM::new(instr, address, true, compressed).into()),
                    (0b111, 0b0000001) => Ok(REMU::new(instr, address, true, compressed).into()),
                    _ => Err("Invalid R-type arithmetic instruction"),
                }
            }
            0b0111011 => {
                // RV64I R-type arithmetic instructions.
                let funct3 = (instr >> 12) & 0x7;
                let funct7 = (instr >> 25) & 0x7f;
                match (funct3, funct7) {
                    (0b000, 0b0000000) => Ok(ADDW::new(instr, address, true, compressed).into()),
                    (0b000, 0b0100000) => Ok(SUBW::new(instr, address, true, compressed).into()),
                    (0b001, 0b0000000) => Ok(SLLW::new(instr, address, true, compressed).into()),
                    (0b100, 0b0000001) => Ok(DIVW::new(instr, address, true, compressed).into()),
                    (0b101, 0b0000000) => Ok(SRLW::new(instr, address, true, compressed).into()),
                    (0b101, 0b0100000) => Ok(SRAW::new(instr, address, true, compressed).into()),
                    (0b000, 0b0000001) => Ok(MULW::new(instr, address, true, compressed).into()),
                    (0b101, 0b0000001) => Ok(DIVUW::new(instr, address, true, compressed).into()),
                    (0b110, 0b0000001) => Ok(REMW::new(instr, address, true, compressed).into()),
                    (0b111, 0b0000001) => Ok(REMUW::new(instr, address, true, compressed).into()),
                    _ => Err("Invalid RV64I R-type arithmetic instruction"),
                }
            }
            0b0001111 => {
                // FENCE: I-type; the immediate encodes "pred" and "succ" flags.
                Ok(FENCE::new(instr, address, true, compressed).into())
            }
            0b0101111 => {
                // Atomic Memory Operations (A-extension): LR, SC, AMOSWAP, AMOADD, etc.
                // Only check funct3 (width) and funct5 (operation type)
                // bits [26:25] are aq/rl flags which can vary
                let funct3 = (instr >> 12) & 0x7;
                let funct5 = (instr >> 27) & 0x1f;

                match (funct3, funct5) {
                    // LR (Load Reserved)
                    (0b010, 0b00010) => Ok(LRW::new(instr, address, true, compressed).into()),
                    (0b011, 0b00010) => Ok(LRD::new(instr, address, true, compressed).into()),

                    // SC (Store Conditional)
                    (0b010, 0b00011) => Ok(SCW::new(instr, address, true, compressed).into()),
                    (0b011, 0b00011) => Ok(SCD::new(instr, address, true, compressed).into()),

                    // AMOSWAP
                    (0b010, 0b00001) => Ok(AMOSWAPW::new(instr, address, true, compressed).into()),
                    (0b011, 0b00001) => Ok(AMOSWAPD::new(instr, address, true, compressed).into()),

                    // AMOADD
                    (0b010, 0b00000) => Ok(AMOADDW::new(instr, address, true, compressed).into()),
                    (0b011, 0b00000) => Ok(AMOADDD::new(instr, address, true, compressed).into()),

                    // AMOAND
                    (0b010, 0b01100) => Ok(AMOANDW::new(instr, address, true, compressed).into()),
                    (0b011, 0b01100) => Ok(AMOANDD::new(instr, address, true, compressed).into()),

                    // AMOOR
                    (0b010, 0b01000) => Ok(AMOORW::new(instr, address, true, compressed).into()),
                    (0b011, 0b01000) => Ok(AMOORD::new(instr, address, true, compressed).into()),

                    // AMOXOR
                    (0b010, 0b00100) => Ok(AMOXORW::new(instr, address, true, compressed).into()),
                    (0b011, 0b00100) => Ok(AMOXORD::new(instr, address, true, compressed).into()),

                    // AMOMIN
                    (0b010, 0b10000) => Ok(AMOMINW::new(instr, address, true, compressed).into()),
                    (0b011, 0b10000) => Ok(AMOMIND::new(instr, address, true, compressed).into()),

                    // AMOMAX
                    (0b010, 0b10100) => Ok(AMOMAXW::new(instr, address, true, compressed).into()),
                    (0b011, 0b10100) => Ok(AMOMAXD::new(instr, address, true, compressed).into()),

                    // AMOMINU
                    (0b010, 0b11000) => Ok(AMOMINUW::new(instr, address, true, compressed).into()),
                    (0b011, 0b11000) => Ok(AMOMINUD::new(instr, address, true, compressed).into()),

                    // AMOMAXU
                    (0b010, 0b11100) => Ok(AMOMAXUW::new(instr, address, true, compressed).into()),
                    (0b011, 0b11100) => Ok(AMOMAXUD::new(instr, address, true, compressed).into()),

                    _ => {
                        eprintln!("Invalid atomic memory operation: instr=0x{instr:08x} funct3={funct3:03b} funct5={funct5:05b}");
                        Err("Invalid atomic memory operation")
                    }
                }
            }
            0b1110011 => {
                // For now this only (potentially) maps to ECALL.
                if instr == ECALL::MATCH {
                    Ok(ECALL::new(instr, address, true, compressed).into())
                } else {
                    Err("Unsupported SYSTEM instruction")
                }
            }
            // 0x0B is reserved for inlines supported by Jolt in jolt-inlines crate.
            // In attempt to standardize this space for precompiles and inlines,
            // each new type of operation should be placed under different funct7,
            // while funct3 should hold all necessary instructions for that operation.
            // funct7:
            // - 0x00: SHA256
            // - 0x01: Keccak256
            0b0001011 => Ok(INLINE::new(instr, address, false, compressed).into()),
            // 0x2B is reserved for external inlines
            0b0101011 => Ok(INLINE::new(instr, address, false, compressed).into()),
            // 0x5B is reserved for virtual instructions.
            0b1011011 => {
                let funct3 = (instr >> 12) & 0x7;
                match funct3 {
                    0b000 => Ok(VirtualRev8W::new(instr, address, true, compressed).into()),
                    0b001 => Ok(VirtualAssertEQ::new(instr, address, true, compressed).into()),
                    _ => Err("Invalid virtual instruction"),
                }
            }
            _ => Err("Unknown opcode"),
        }
    }
}

// @TODO: Optimize
pub fn uncompress_instruction(halfword: u32, xlen: Xlen) -> u32 {
    let op = halfword & 0x3; // [1:0]
    let funct3 = (halfword >> 13) & 0x7; // [15:13]

    match op {
        0 => match funct3 {
            0 => {
                // C.ADDI4SPN
                // addi rd+8, x2, nzuimm
                let rd = (halfword >> 2) & 0x7; // [4:2]
                let nzuimm = ((halfword >> 7) & 0x30) | // nzuimm[5:4] <= [12:11]
                        ((halfword >> 1) & 0x3c0) | // nzuimm{9:6] <= [10:7]
                        ((halfword >> 4) & 0x4) | // nzuimm[2] <= [6]
                        ((halfword >> 2) & 0x8); // nzuimm[3] <= [5]
                                                 // nzuimm == 0 is reserved instruction
                if nzuimm != 0 {
                    return (nzuimm << 20) | (2 << 15) | ((rd + 8) << 7) | 0x13;
                }
            }
            1 => {
                // @TODO: Support C.LQ for 128-bit
                // C.FLD for 32, 64-bit
                // fld rd+8, offset(rs1+8)
                let rd = (halfword >> 2) & 0x7; // [4:2]
                let rs1 = (halfword >> 7) & 0x7; // [9:7]
                let offset = ((halfword >> 7) & 0x38) | // offset[5:3] <= [12:10]
                        ((halfword << 1) & 0xc0); // offset[7:6] <= [6:5]
                return (offset << 20) | ((rs1 + 8) << 15) | (3 << 12) | ((rd + 8) << 7) | 0x7;
            }
            2 => {
                // C.LW
                // lw rd+8, offset(rs1+8)
                let rs1 = (halfword >> 7) & 0x7; // [9:7]
                let rd = (halfword >> 2) & 0x7; // [4:2]
                let offset = ((halfword >> 7) & 0x38) | // offset[5:3] <= [12:10]
                        ((halfword >> 4) & 0x4) | // offset[2] <= [6]
                        ((halfword << 1) & 0x40); // offset[6] <= [5]
                return (offset << 20) | ((rs1 + 8) << 15) | (2 << 12) | ((rd + 8) << 7) | 0x3;
            }
            3 => {
                // @TODO: Support C.FLW in 32-bit mode
                // C.LD in 64-bit mode
                // ld rd+8, offset(rs1+8)
                let rs1 = (halfword >> 7) & 0x7; // [9:7]
                let rd = (halfword >> 2) & 0x7; // [4:2]
                let offset = ((halfword >> 7) & 0x38) | // offset[5:3] <= [12:10]
                        ((halfword << 1) & 0xc0); // offset[7:6] <= [6:5]
                return (offset << 20) | ((rs1 + 8) << 15) | (3 << 12) | ((rd + 8) << 7) | 0x3;
            }
            4 => {
                // Reserved
            }
            5 => {
                // C.FSD
                // fsd rs2+8, offset(rs1+8)
                let rs1 = (halfword >> 7) & 0x7; // [9:7]
                let rs2 = (halfword >> 2) & 0x7; // [4:2]
                let offset = ((halfword >> 7) & 0x38) | // uimm[5:3] <= [12:10]
                        ((halfword << 1) & 0xc0); // uimm[7:6] <= [6:5]
                let imm11_5 = (offset >> 5) & 0x7f;
                let imm4_0 = offset & 0x1f;
                return (imm11_5 << 25)
                    | ((rs2 + 8) << 20)
                    | ((rs1 + 8) << 15)
                    | (3 << 12)
                    | (imm4_0 << 7)
                    | 0x27;
            }
            6 => {
                // C.SW
                // sw rs2+8, offset(rs1+8)
                let rs1 = (halfword >> 7) & 0x7; // [9:7]
                let rs2 = (halfword >> 2) & 0x7; // [4:2]
                let offset = ((halfword >> 7) & 0x38) | // offset[5:3] <= [12:10]
                        ((halfword << 1) & 0x40) | // offset[6] <= [5]
                        ((halfword >> 4) & 0x4); // offset[2] <= [6]
                let imm11_5 = (offset >> 5) & 0x7f;
                let imm4_0 = offset & 0x1f;
                return (imm11_5 << 25)
                    | ((rs2 + 8) << 20)
                    | ((rs1 + 8) << 15)
                    | (2 << 12)
                    | (imm4_0 << 7)
                    | 0x23;
            }
            7 => {
                // @TODO: Support C.FSW in 32-bit mode
                // C.SD
                // sd rs2+8, offset(rs1+8)
                let rs1 = (halfword >> 7) & 0x7; // [9:7]
                let rs2 = (halfword >> 2) & 0x7; // [4:2]
                let offset = ((halfword >> 7) & 0x38) | // uimm[5:3] <= [12:10]
                        ((halfword << 1) & 0xc0); // uimm[7:6] <= [6:5]
                let imm11_5 = (offset >> 5) & 0x7f;
                let imm4_0 = offset & 0x1f;
                return (imm11_5 << 25)
                    | ((rs2 + 8) << 20)
                    | ((rs1 + 8) << 15)
                    | (3 << 12)
                    | (imm4_0 << 7)
                    | 0x23;
            }
            _ => {} // Not happens
        },
        1 => {
            match funct3 {
                0 => {
                    // C.ADDI
                    let r = (halfword >> 7) & 0x1f; // [11:7]
                    let imm = match halfword & 0x1000 {
                            0x1000 => 0xffffffc0,
                            _ => 0
                        } | // imm[31:6] <= [12]
                        ((halfword >> 7) & 0x20) | // imm[5] <= [12]
                        ((halfword >> 2) & 0x1f); // imm[4:0] <= [6:2]

                    match (r, imm) {
                        (0, 0) => {
                            // NOP
                            return 0x13;
                        }
                        (0, _) => {
                            // HINT
                            return 0x13;
                        }
                        (_, 0) => {
                            // HINT
                            return 0x13;
                        }
                        (_, _) => {
                            return (imm << 20) | (r << 15) | (r << 7) | 0x13;
                        }
                    }
                }
                1 => {
                    match xlen {
                        Xlen::Bit32 => {
                            // C.JAL (RV32C only)
                            // jal x1, offset
                            let offset = match halfword & 0x1000 {
                                    0x1000 => 0xfffff000,
                                    _ => 0
                                } | // offset[31:12] <= [12]
                                ((halfword >> 1) & 0x800) | // offset[11] <= [12]
                                ((halfword >> 7) & 0x10) | // offset[4] <= [11]
                                ((halfword >> 1) & 0x300) | // offset[9:8] <= [10:9]
                                ((halfword << 2) & 0x400) | // offset[10] <= [8]
                                ((halfword >> 1) & 0x40) | // offset[6] <= [7]
                                ((halfword << 1) & 0x80) | // offset[7] <= [6]
                                ((halfword >> 2) & 0xe) | // offset[3:1] <= [5:3]
                                ((halfword << 3) & 0x20); // offset[5] <= [2]
                            let imm = ((offset >> 1) & 0x80000) | // imm[19] <= offset[20]
                                    ((offset << 8) & 0x7fe00) | // imm[18:9] <= offset[10:1]
                                    ((offset >> 3) & 0x100) | // imm[8] <= offset[11]
                                    ((offset >> 12) & 0xff); // imm[7:0] <= offset[19:12]
                            return (imm << 12) | (1 << 7) | 0x6f;
                        }
                        Xlen::Bit64 => {
                            // C.ADDIW (RV64C only)
                            let r = (halfword >> 7) & 0x1f;
                            let imm = match halfword & 0x1000 {
                            0x1000 => 0xffffffc0,
                            _ => 0
                        } | // imm[31:6] <= [12]
                        ((halfword >> 7) & 0x20) | // imm[5] <= [12]
                        ((halfword >> 2) & 0x1f); // imm[4:0] <= [6:2]
                            if r == 0 {
                                // Reserved
                            } else if imm == 0 {
                                // sext.w rd
                                return (r << 15) | (r << 7) | 0x1b;
                            } else {
                                // addiw r, r, imm
                                return (imm << 20) | (r << 15) | (r << 7) | 0x1b;
                            }
                        }
                    }
                }
                2 => {
                    // C.LI
                    let r = (halfword >> 7) & 0x1f;
                    let imm = match halfword & 0x1000 {
                            0x1000 => 0xffffffc0,
                            _ => 0
                        } | // imm[31:6] <= [12]
                        ((halfword >> 7) & 0x20) | // imm[5] <= [12]
                        ((halfword >> 2) & 0x1f); // imm[4:0] <= [6:2]
                    if r != 0 {
                        // addi rd, x0, imm
                        return (imm << 20) | (r << 7) | 0x13;
                    } else {
                        // HINT
                        return 0x13;
                    }
                }
                3 => {
                    let r = (halfword >> 7) & 0x1f; // [11:7]
                    if r == 2 {
                        // C.ADDI16SP
                        // addi r, r, nzimm
                        let imm = match halfword & 0x1000 {
                                0x1000 => 0xfffffc00,
                                _ => 0
                            } | // imm[31:10] <= [12]
                            ((halfword >> 3) & 0x200) | // imm[9] <= [12]
                            ((halfword >> 2) & 0x10) | // imm[4] <= [6]
                            ((halfword << 1) & 0x40) | // imm[6] <= [5]
                            ((halfword << 4) & 0x180) | // imm[8:7] <= [4:3]
                            ((halfword << 3) & 0x20); // imm[5] <= [2]
                        if imm != 0 {
                            return (imm << 20) | (r << 15) | (r << 7) | 0x13;
                        }
                        // imm == 0 is for reserved instruction
                    }
                    if r != 0 && r != 2 {
                        // C.LUI
                        // lui r, nzimm
                        let nzimm = match halfword & 0x1000 {
                                0x1000 => 0xfffc0000,
                                _ => 0
                            } | // nzimm[31:18] <= [12]
                            ((halfword << 5) & 0x20000) | // nzimm[17] <= [12]
                            ((halfword << 10) & 0x1f000); // nzimm[16:12] <= [6:2]
                        if nzimm != 0 {
                            return nzimm | (r << 7) | 0x37;
                        }
                        // nzimm == 0 is for reserved instruction
                    }
                    if r == 0 {
                        // NOP
                        return 0x13;
                    }
                }
                4 => {
                    let funct2 = (halfword >> 10) & 0x3; // [11:10]
                    match funct2 {
                        0 => {
                            // C.SRLI
                            // c.srli rs1+8, rs1+8, shamt
                            let shamt = ((halfword >> 7) & 0x20) | // shamt[5] <= [12]
                                    ((halfword >> 2) & 0x1f); // shamt[4:0] <= [6:2]
                            let rs1 = (halfword >> 7) & 0x7; // [9:7]
                            return (shamt << 20)
                                | ((rs1 + 8) << 15)
                                | (5 << 12)
                                | ((rs1 + 8) << 7)
                                | 0x13;
                        }
                        1 => {
                            // C.SRAI
                            // srai rs1+8, rs1+8, shamt
                            let shamt = ((halfword >> 7) & 0x20) | // shamt[5] <= [12]
                                    ((halfword >> 2) & 0x1f); // shamt[4:0] <= [6:2]
                            let rs1 = (halfword >> 7) & 0x7; // [9:7]
                            return (0x20 << 25)
                                | (shamt << 20)
                                | ((rs1 + 8) << 15)
                                | (5 << 12)
                                | ((rs1 + 8) << 7)
                                | 0x13;
                        }
                        2 => {
                            // C.ANDI
                            // andi, r+8, r+8, imm
                            let r = (halfword >> 7) & 0x7; // [9:7]
                            let imm = match halfword & 0x1000 {
                                    0x1000 => 0xffffffc0,
                                    _ => 0
                                } | // imm[31:6] <= [12]
                                ((halfword >> 7) & 0x20) | // imm[5] <= [12]
                                ((halfword >> 2) & 0x1f); // imm[4:0] <= [6:2]
                            return (imm << 20)
                                | ((r + 8) << 15)
                                | (7 << 12)
                                | ((r + 8) << 7)
                                | 0x13;
                        }
                        3 => {
                            let funct1 = (halfword >> 12) & 1; // [12]
                            let funct2_2 = (halfword >> 5) & 0x3; // [6:5]
                            let rs1 = (halfword >> 7) & 0x7;
                            let rs2 = (halfword >> 2) & 0x7;
                            match funct1 {
                                0 => match funct2_2 {
                                    0 => {
                                        // C.SUB
                                        // sub rs1+8, rs1+8, rs2+8
                                        return (0x20 << 25)
                                            | ((rs2 + 8) << 20)
                                            | ((rs1 + 8) << 15)
                                            | ((rs1 + 8) << 7)
                                            | 0x33;
                                    }
                                    1 => {
                                        // C.XOR
                                        // xor rs1+8, rs1+8, rs2+8
                                        return ((rs2 + 8) << 20)
                                            | ((rs1 + 8) << 15)
                                            | (4 << 12)
                                            | ((rs1 + 8) << 7)
                                            | 0x33;
                                    }
                                    2 => {
                                        // C.OR
                                        // or rs1+8, rs1+8, rs2+8
                                        return ((rs2 + 8) << 20)
                                            | ((rs1 + 8) << 15)
                                            | (6 << 12)
                                            | ((rs1 + 8) << 7)
                                            | 0x33;
                                    }
                                    3 => {
                                        // C.AND
                                        // and rs1+8, rs1+8, rs2+8
                                        return ((rs2 + 8) << 20)
                                            | ((rs1 + 8) << 15)
                                            | (7 << 12)
                                            | ((rs1 + 8) << 7)
                                            | 0x33;
                                    }
                                    _ => {} // Not happens
                                },
                                1 => match funct2_2 {
                                    0 => {
                                        // C.SUBW
                                        // subw r1+8, r1+8, r2+8
                                        return (0x20 << 25)
                                            | ((rs2 + 8) << 20)
                                            | ((rs1 + 8) << 15)
                                            | ((rs1 + 8) << 7)
                                            | 0x3b;
                                    }
                                    1 => {
                                        // C.ADDW
                                        // addw r1+8, r1+8, r2+8
                                        return ((rs2 + 8) << 20)
                                            | ((rs1 + 8) << 15)
                                            | ((rs1 + 8) << 7)
                                            | 0x3b;
                                    }
                                    2 => {
                                        // Reserved
                                    }
                                    3 => {
                                        // Reserved
                                    }
                                    _ => {} // Not happens
                                },
                                _ => {} // No happens
                            };
                        }
                        _ => {} // not happens
                    };
                }
                5 => {
                    // C.J
                    // jal x0, imm
                    let offset = match halfword & 0x1000 {
                                0x1000 => 0xfffff000,
                                _ => 0
                            } | // offset[31:12] <= [12]
                            ((halfword >> 1) & 0x800) | // offset[11] <= [12]
                            ((halfword >> 7) & 0x10) | // offset[4] <= [11]
                            ((halfword >> 1) & 0x300) | // offset[9:8] <= [10:9]
                            ((halfword << 2) & 0x400) | // offset[10] <= [8]
                            ((halfword >> 1) & 0x40) | // offset[6] <= [7]
                            ((halfword << 1) & 0x80) | // offset[7] <= [6]
                            ((halfword >> 2) & 0xe) | // offset[3:1] <= [5:3]
                            ((halfword << 3) & 0x20); // offset[5] <= [2]
                    let imm = ((offset >> 1) & 0x80000) | // imm[19] <= offset[20]
                            ((offset << 8) & 0x7fe00) | // imm[18:9] <= offset[10:1]
                            ((offset >> 3) & 0x100) | // imm[8] <= offset[11]
                            ((offset >> 12) & 0xff); // imm[7:0] <= offset[19:12]
                    return (imm << 12) | 0x6f;
                }
                6 => {
                    // C.BEQZ
                    // beq r+8, x0, offset
                    let r = (halfword >> 7) & 0x7;
                    let offset = match halfword & 0x1000 {
                                0x1000 => 0xfffffe00,
                                _ => 0
                            } | // offset[31:9] <= [12]
                            ((halfword >> 4) & 0x100) | // offset[8] <= [12]
                            ((halfword >> 7) & 0x18) | // offset[4:3] <= [11:10]
                            ((halfword << 1) & 0xc0) | // offset[7:6] <= [6:5]
                            ((halfword >> 2) & 0x6) | // offset[2:1] <= [4:3]
                            ((halfword << 3) & 0x20); // offset[5] <= [2]
                    let imm2 = ((offset >> 6) & 0x40) | // imm2[6] <= [12]
                            ((offset >> 5) & 0x3f); // imm2[5:0] <= [10:5]
                    let imm1 = (offset & 0x1e) | // imm1[4:1] <= [4:1]
                            ((offset >> 11) & 0x1); // imm1[0] <= [11]
                    return (imm2 << 25) | ((r + 8) << 20) | (imm1 << 7) | 0x63;
                }
                7 => {
                    // C.BNEZ
                    // bne r+8, x0, offset
                    let r = (halfword >> 7) & 0x7;
                    let offset = match halfword & 0x1000 {
                                0x1000 => 0xfffffe00,
                                _ => 0
                            } | // offset[31:9] <= [12]
                            ((halfword >> 4) & 0x100) | // offset[8] <= [12]
                            ((halfword >> 7) & 0x18) | // offset[4:3] <= [11:10]
                            ((halfword << 1) & 0xc0) | // offset[7:6] <= [6:5]
                            ((halfword >> 2) & 0x6) | // offset[2:1] <= [4:3]
                            ((halfword << 3) & 0x20); // offset[5] <= [2]
                    let imm2 = ((offset >> 6) & 0x40) | // imm2[6] <= [12]
                            ((offset >> 5) & 0x3f); // imm2[5:0] <= [10:5]
                    let imm1 = (offset & 0x1e) | // imm1[4:1] <= [4:1]
                            ((offset >> 11) & 0x1); // imm1[0] <= [11]
                    return (imm2 << 25) | ((r + 8) << 20) | (1 << 12) | (imm1 << 7) | 0x63;
                }
                _ => {} // No happens
            };
        }
        2 => {
            match funct3 {
                0 => {
                    // C.SLLI
                    // slli r, r, shamt
                    let r = (halfword >> 7) & 0x1f;
                    let shamt = ((halfword >> 7) & 0x20) | // imm[5] <= [12]
                            ((halfword >> 2) & 0x1f); // imm[4:0] <= [6:2]
                    if r != 0 {
                        return (shamt << 20) | (r << 15) | (1 << 12) | (r << 7) | 0x13;
                    }
                    // r == 0 is reserved instruction?
                }
                1 => {
                    // C.FLDSP
                    // fld rd, offset(x2)
                    let rd = (halfword >> 7) & 0x1f;
                    let offset = ((halfword >> 7) & 0x20) | // offset[5] <= [12]
                            ((halfword >> 2) & 0x18) | // offset[4:3] <= [6:5]
                            ((halfword << 4) & 0x1c0); // offset[8:6] <= [4:2]
                    if rd != 0 {
                        return (offset << 20) | (2 << 15) | (3 << 12) | (rd << 7) | 0x7;
                    }
                    // rd == 0 is reserved instruction
                }
                2 => {
                    // C.LWSP
                    // lw r, offset(x2)
                    let r = (halfword >> 7) & 0x1f;
                    let offset = ((halfword >> 7) & 0x20) | // offset[5] <= [12]
                            ((halfword >> 2) & 0x1c) | // offset[4:2] <= [6:4]
                            ((halfword << 4) & 0xc0); // offset[7:6] <= [3:2]
                    if r != 0 {
                        return (offset << 20) | (2 << 15) | (2 << 12) | (r << 7) | 0x3;
                    }
                    // r == 0 is reserved instruction
                }
                3 => {
                    // @TODO: Support C.FLWSP in 32-bit mode
                    // C.LDSP
                    // ld rd, offset(x2)
                    let rd = (halfword >> 7) & 0x1f;
                    let offset = ((halfword >> 7) & 0x20) | // offset[5] <= [12]
                            ((halfword >> 2) & 0x18) | // offset[4:3] <= [6:5]
                            ((halfword << 4) & 0x1c0); // offset[8:6] <= [4:2]
                    if rd != 0 {
                        return (offset << 20) | (2 << 15) | (3 << 12) | (rd << 7) | 0x3;
                    }
                    // rd == 0 is reserved instruction
                }
                4 => {
                    let funct1 = (halfword >> 12) & 1; // [12]
                    let rs1 = (halfword >> 7) & 0x1f; // [11:7]
                    let rs2 = (halfword >> 2) & 0x1f; // [6:2]
                    match funct1 {
                        0 => {
                            // C.MV
                            match (rs1, rs2) {
                                (0, 0) => {
                                    // Reserved
                                }
                                (r, 0) if r != 0 => {
                                    // C.JR: jalr x0, 0(rs1)
                                    return (rs1 << 15) | 0x67;
                                }
                                (0, r2) if r2 != 0 => {
                                    // HINT
                                    return 0x13;
                                }
                                (rd, rs2) => {
                                    // add rd, x0, rs2
                                    return (rs2 << 20) | (rd << 7) | 0x33;
                                }
                            }
                        }
                        1 => {
                            // C.ADD
                            match (rs1, rs2) {
                                (0, 0) => {
                                    // C.EBREAK
                                    // ebreak
                                    return 0x00100073;
                                }
                                (rs1, 0) if rs1 != 0 => {
                                    // C.JALR
                                    // jalr x1, 0(rs1)
                                    return (rs1 << 15) | (1 << 7) | 0x67;
                                }
                                (0, rs2) if rs2 != 0 => {
                                    // HINT
                                    return 0x13;
                                }
                                (rs1, rs2) => {
                                    // C.ADD
                                    // add rs1, rs1, rs2
                                    return (rs2 << 20) | (rs1 << 15) | (rs1 << 7) | 0x33;
                                }
                            }
                        }
                        _ => {} // Not happens
                    };
                }
                5 => {
                    // @TODO: Implement
                    // C.FSDSP
                    // fsd rs2, offset(x2)
                    let rs2 = (halfword >> 2) & 0x1f; // [6:2]
                    let offset = ((halfword >> 7) & 0x38) | // offset[5:3] <= [12:10]
                            ((halfword >> 1) & 0x1c0); // offset[8:6] <= [9:7]
                    let imm11_5 = (offset >> 5) & 0x3f;
                    let imm4_0 = offset & 0x1f;
                    return (imm11_5 << 25)
                        | (rs2 << 20)
                        | (2 << 15)
                        | (3 << 12)
                        | (imm4_0 << 7)
                        | 0x27;
                }
                6 => {
                    // C.SWSP
                    // sw rs2, offset(x2)
                    let rs2 = (halfword >> 2) & 0x1f; // [6:2]
                    let offset = ((halfword >> 7) & 0x3c) | // offset[5:2] <= [12:9]
                            ((halfword >> 1) & 0xc0); // offset[7:6] <= [8:7]
                    let imm11_5 = (offset >> 5) & 0x3f;
                    let imm4_0 = offset & 0x1f;
                    return (imm11_5 << 25)
                        | (rs2 << 20)
                        | (2 << 15)
                        | (2 << 12)
                        | (imm4_0 << 7)
                        | 0x23;
                }
                7 => {
                    // @TODO: Support C.FSWSP in 32-bit mode
                    // C.SDSP
                    // sd rs, offset(x2)
                    let rs2 = (halfword >> 2) & 0x1f; // [6:2]
                    let offset = ((halfword >> 7) & 0x38) | // offset[5:3] <= [12:10]
                            ((halfword >> 1) & 0x1c0); // offset[8:6] <= [9:7]
                    let imm11_5 = (offset >> 5) & 0x3f;
                    let imm4_0 = offset & 0x1f;
                    return (imm11_5 << 25)
                        | (rs2 << 20)
                        | (2 << 15)
                        | (3 << 12)
                        | (imm4_0 << 7)
                        | 0x23;
                }
                _ => {} // Not happens
            };
        }
        _ => {} // Not happens
    };
    0xffffffff // Return invalid value
}

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RISCVCycle<T: RISCVInstruction> {
    pub instruction: T,
    pub register_state: <T::Format as InstructionFormat>::RegisterState,
    pub ram_access: T::RAMAccess,
}

impl<T: RISCVInstruction> RISCVCycle<T> {
    #[cfg(any(feature = "test-utils", test))]
    pub fn random(&self, rng: &mut rand::rngs::StdRng) -> Self {
        let instruction = T::random(rng);
        let register_state =
            <<T::Format as InstructionFormat>::RegisterState as InstructionRegisterState>::random(
                rng,
                &Into::<NormalizedInstruction>::into(instruction).operands,
            );
        Self {
            instruction,
            ram_access: Default::default(),
            register_state,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // Check that the size of Cycle is as expected.
    fn rv32im_cycle_size() {
        let size = size_of::<Cycle>();
        let expected = 96;
        assert_eq!(
            size, expected,
            "Cycle size should be {expected} bytes, but is {size} bytes"
        );
    }
}
