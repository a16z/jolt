use add::ADDInstruction;
use and::ANDInstruction;
use beq::BEQInstruction;
use bge::BGEInstruction;
use bgeu::BGEUInstruction;
use bne::BNEInstruction;
use enum_dispatch::enum_dispatch;
use mul::MULInstruction;
use mulhu::MULHUInstruction;
use mulu::MULUInstruction;
use or::ORInstruction;
use prefixes::PrefixEval;
use rand::rngs::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use slt::SLTInstruction;
use sltu::SLTUInstruction;
use std::marker::Sync;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};
use sub::SUBInstruction;
use suffixes::{SuffixEval, Suffixes};
use tracer::{ELFInstruction, RVTraceRow, RegisterState, RV32IM};
use virtual_advice::ADVICEInstruction;
use virtual_assert_halfword_alignment::AssertHalfwordAlignmentInstruction;
use virtual_assert_lte::ASSERTLTEInstruction;
use virtual_assert_valid_div0::AssertValidDiv0Instruction;
use virtual_assert_valid_signed_remainder::AssertValidSignedRemainderInstruction;
use virtual_assert_valid_unsigned_remainder::AssertValidUnsignedRemainderInstruction;
use virtual_move::MOVEInstruction;
use virtual_movsign::MOVSIGNInstruction;
use virtual_pow2::POW2Instruction;
use virtual_right_shift_padding::RightShiftPaddingInstruction;
use xor::XORInstruction;

use crate::field::JoltField;
use crate::subprotocols::sparse_dense_shout::PrefixSuffixDecomposition;
use crate::utils::interleave_bits;
use std::fmt::Debug;

#[enum_dispatch]
pub trait JoltInstruction: Clone + Debug + Send + Sync + Serialize {
    /// Converts this instruction's operands into a lookup index (as used in sparse-dense Shout).
    /// By default, interleaves the two bits of the two operands together.
    fn to_lookup_index(&self) -> u64 {
        let (x, y) = self.operands();
        interleave_bits(x as u32, y as u32)
    }

    /// Materializes the entire lookup table for this instruction (assuming an 8-bit word size).
    #[cfg(test)]
    fn materialize(&self) -> Vec<u64> {
        (0..1 << 16).map(|i| self.materialize_entry(i)).collect()
    }

    /// Materialize the entry at the given `index` in the lookup table for this instruction.
    fn materialize_entry(&self, index: u64) -> u64;

    /// Evaluates the MLE of this lookup table on the given point `r`.
    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F;

    /// Returns a tuple of the instruction's operands. If the instruction has only one operand,
    /// one of the tuple values will be 0.
    fn operands(&self) -> (u64, u64);

    /// Computes the output lookup entry for this instruction as a u64.
    fn lookup_entry(&self) -> u64;

    fn random(&self, rng: &mut StdRng) -> Self;
}

pub trait JoltInstructionSet:
    JoltInstruction + IntoEnumIterator + EnumCount + for<'a> TryFrom<&'a ELFInstruction> + Send + Sync
{
    fn enum_index(instruction: &Self) -> usize {
        // Discriminant: https://doc.rust-lang.org/reference/items/enumerations.html#pointer-casting
        let byte = unsafe { *(instruction as *const Self as *const u8) };
        byte as usize
    }
}

pub trait VirtualInstructionSequence {
    const SEQUENCE_LENGTH: usize;
    fn virtual_sequence(instruction: ELFInstruction) -> Vec<ELFInstruction> {
        let dummy_trace_row = RVTraceRow {
            instruction,
            register_state: RegisterState {
                rs1_val: Some(0),
                rs2_val: Some(0),
                rd_post_val: Some(0),
            },
            memory_state: None,
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        };
        Self::virtual_trace(dummy_trace_row)
            .into_iter()
            .map(|trace_row| trace_row.instruction)
            .collect()
    }
    fn virtual_trace(trace_row: RVTraceRow) -> Vec<RVTraceRow>;
    fn sequence_output(x: u64, y: u64) -> u64;
}

pub mod prefixes;
pub mod suffixes;

pub mod add;
pub mod and;
pub mod beq;
pub mod bge;
pub mod bgeu;
pub mod bne;
pub mod div;
pub mod divu;
pub mod lb;
pub mod lbu;
pub mod lh;
pub mod lhu;
pub mod mul;
pub mod mulh;
pub mod mulhsu;
pub mod mulhu;
pub mod mulu;
pub mod or;
pub mod rem;
pub mod remu;
pub mod sb;
pub mod sh;
pub mod sll;
pub mod slt;
pub mod sltu;
pub mod sra;
pub mod srl;
pub mod sub;
pub mod virtual_advice;
pub mod virtual_assert_halfword_alignment;
pub mod virtual_assert_lte;
pub mod virtual_assert_valid_div0;
pub mod virtual_assert_valid_signed_remainder;
pub mod virtual_assert_valid_unsigned_remainder;
pub mod virtual_move;
pub mod virtual_movsign;
pub mod virtual_pow2;
pub mod virtual_right_shift_padding;
pub mod xor;

#[cfg(test)]
pub mod test;

#[derive(Copy, Clone, Debug, Serialize, Deserialize, EnumIter, EnumCountMacro)]
#[repr(u8)]
pub enum LookupTables<const WORD_SIZE: usize> {
    Add(ADDInstruction<WORD_SIZE>),
    Sub(SUBInstruction<WORD_SIZE>),
    And(ANDInstruction<WORD_SIZE>),
    Or(ORInstruction<WORD_SIZE>),
    Xor(XORInstruction<WORD_SIZE>),
    Beq(BEQInstruction<WORD_SIZE>),
    Bge(BGEInstruction<WORD_SIZE>),
    Bgeu(BGEUInstruction<WORD_SIZE>),
    Bne(BNEInstruction<WORD_SIZE>),
    Slt(SLTInstruction<WORD_SIZE>),
    Sltu(SLTUInstruction<WORD_SIZE>),
    Move(MOVEInstruction<WORD_SIZE>),
    Movsign(MOVSIGNInstruction<WORD_SIZE>),
    Mul(MULInstruction<WORD_SIZE>),
    Mulu(MULUInstruction<WORD_SIZE>),
    Mulhu(MULHUInstruction<WORD_SIZE>),
    Advice(ADVICEInstruction<WORD_SIZE>),
    AssertLte(ASSERTLTEInstruction<WORD_SIZE>),
    AssertValidSignedRemainder(AssertValidSignedRemainderInstruction<WORD_SIZE>),
    AssertValidUnsignedRemainder(AssertValidUnsignedRemainderInstruction<WORD_SIZE>),
    AssertValidDiv0(AssertValidDiv0Instruction<WORD_SIZE>),
    AssertHalfwordAlignment(AssertHalfwordAlignmentInstruction<WORD_SIZE>),
    Pow2(POW2Instruction<WORD_SIZE>),
    RightShiftPadding(RightShiftPaddingInstruction<WORD_SIZE>),
}

impl<const WORD_SIZE: usize> TryFrom<&RVTraceRow> for LookupTables<WORD_SIZE> {
    type Error = &'static str;

    fn try_from(row: &RVTraceRow) -> Result<Self, Self::Error> {
        match row.instruction.opcode {
            RV32IM::ADD => Ok(Self::Add(ADDInstruction(
                row.register_state.rs1_val.unwrap(),
                row.register_state.rs2_val.unwrap(),
            ))),
            RV32IM::SUB => Ok(Self::Sub(SUBInstruction(
                row.register_state.rs1_val.unwrap(),
                row.register_state.rs2_val.unwrap(),
            ))),
            RV32IM::XOR => Ok(Self::Xor(XORInstruction(
                row.register_state.rs1_val.unwrap(),
                row.register_state.rs2_val.unwrap(),
            ))),
            RV32IM::OR => Ok(Self::Or(ORInstruction(
                row.register_state.rs1_val.unwrap(),
                row.register_state.rs2_val.unwrap(),
            ))),
            RV32IM::AND => Ok(Self::And(ANDInstruction(
                row.register_state.rs1_val.unwrap(),
                row.register_state.rs2_val.unwrap(),
            ))),
            RV32IM::SLT => Ok(Self::Slt(SLTInstruction(
                row.register_state.rs1_val.unwrap(),
                row.register_state.rs2_val.unwrap(),
            ))),
            RV32IM::SLTU => Ok(Self::Sltu(SLTUInstruction(
                row.register_state.rs1_val.unwrap(),
                row.register_state.rs2_val.unwrap(),
            ))),
            RV32IM::ADDI => Ok(Self::Add(ADDInstruction(
                row.register_state.rs1_val.unwrap(),
                row.imm_u32() as u64,
            ))),
            RV32IM::XORI => Ok(Self::Xor(XORInstruction(
                row.register_state.rs1_val.unwrap(),
                row.imm_u32() as u64,
            ))),
            RV32IM::ORI => Ok(Self::Or(ORInstruction(
                row.register_state.rs1_val.unwrap(),
                row.imm_u32() as u64,
            ))),
            RV32IM::ANDI => Ok(Self::And(ANDInstruction(
                row.register_state.rs1_val.unwrap(),
                row.imm_u32() as u64,
            ))),
            RV32IM::SLTI => Ok(Self::Slt(SLTInstruction(
                row.register_state.rs1_val.unwrap(),
                row.imm_u32() as u64,
            ))),
            RV32IM::SLTIU => Ok(Self::Sltu(SLTUInstruction(
                row.register_state.rs1_val.unwrap(),
                row.imm_u32() as u64,
            ))),
            RV32IM::BEQ => Ok(Self::Beq(BEQInstruction(
                row.register_state.rs1_val.unwrap(),
                row.register_state.rs2_val.unwrap(),
            ))),
            RV32IM::BNE => Ok(Self::Bne(BNEInstruction(
                row.register_state.rs1_val.unwrap(),
                row.register_state.rs2_val.unwrap(),
            ))),
            RV32IM::BLT => Ok(Self::Slt(SLTInstruction(
                row.register_state.rs1_val.unwrap(),
                row.register_state.rs2_val.unwrap(),
            ))),
            RV32IM::BLTU => Ok(Self::Sltu(SLTUInstruction(
                row.register_state.rs1_val.unwrap(),
                row.register_state.rs2_val.unwrap(),
            ))),
            RV32IM::BGE => Ok(Self::Bge(BGEInstruction(
                row.register_state.rs1_val.unwrap(),
                row.register_state.rs2_val.unwrap(),
            ))),
            RV32IM::BGEU => Ok(Self::Bgeu(BGEUInstruction(
                row.register_state.rs1_val.unwrap(),
                row.register_state.rs2_val.unwrap(),
            ))),
            RV32IM::JAL => Ok(Self::Add(ADDInstruction(
                row.instruction.address,
                row.imm_u32() as u64,
            ))),
            RV32IM::JALR => Ok(Self::Add(ADDInstruction(
                row.register_state.rs1_val.unwrap(),
                row.imm_u32() as u64,
            ))),
            RV32IM::AUIPC => Ok(Self::Add(ADDInstruction(
                row.instruction.address,
                row.imm_u32() as u64,
            ))),
            RV32IM::LUI => Ok(Self::Advice(ADVICEInstruction(row.imm_u32() as u64))),
            RV32IM::MUL => Ok(Self::Mul(MULInstruction(
                row.register_state.rs1_val.unwrap(),
                row.register_state.rs2_val.unwrap(),
            ))),
            RV32IM::MULU => Ok(Self::Mulu(MULUInstruction(
                row.register_state.rs1_val.unwrap(),
                row.register_state.rs2_val.unwrap(),
            ))),
            RV32IM::MULHU => Ok(Self::Mulhu(MULHUInstruction(
                row.register_state.rs1_val.unwrap(),
                row.register_state.rs2_val.unwrap(),
            ))),
            RV32IM::VIRTUAL_ADVICE => {
                Ok(Self::Advice(ADVICEInstruction(row.advice_value.unwrap())))
            }
            RV32IM::VIRTUAL_MOVE => Ok(Self::Move(MOVEInstruction(
                row.register_state.rs1_val.unwrap(),
            ))),
            RV32IM::VIRTUAL_MOVSIGN => Ok(Self::Movsign(MOVSIGNInstruction(
                row.register_state.rs1_val.unwrap(),
            ))),
            RV32IM::VIRTUAL_ASSERT_EQ => Ok(Self::Beq(BEQInstruction(
                row.register_state.rs1_val.unwrap(),
                row.register_state.rs2_val.unwrap(),
            ))),
            RV32IM::VIRTUAL_ASSERT_LTE => Ok(Self::AssertLte(ASSERTLTEInstruction(
                row.register_state.rs1_val.unwrap(),
                row.register_state.rs2_val.unwrap(),
            ))),
            RV32IM::VIRTUAL_ASSERT_VALID_UNSIGNED_REMAINDER => Ok(
                Self::AssertValidUnsignedRemainder(AssertValidUnsignedRemainderInstruction(
                    row.register_state.rs1_val.unwrap(),
                    row.register_state.rs2_val.unwrap(),
                )),
            ),
            RV32IM::VIRTUAL_ASSERT_VALID_SIGNED_REMAINDER => Ok(Self::AssertValidSignedRemainder(
                AssertValidSignedRemainderInstruction(
                    row.register_state.rs1_val.unwrap(),
                    row.register_state.rs2_val.unwrap(),
                ),
            )),
            RV32IM::VIRTUAL_ASSERT_VALID_DIV0 => {
                Ok(Self::AssertValidDiv0(AssertValidDiv0Instruction(
                    row.register_state.rs1_val.unwrap(),
                    row.register_state.rs2_val.unwrap(),
                )))
            }
            RV32IM::VIRTUAL_ASSERT_HALFWORD_ALIGNMENT => Ok(Self::AssertHalfwordAlignment(
                AssertHalfwordAlignmentInstruction::<WORD_SIZE>(
                    row.register_state.rs1_val.unwrap(),
                    row.imm_u32() as u64,
                ),
            )),
            RV32IM::VIRTUAL_POW2 => Ok(Self::Pow2(POW2Instruction::<WORD_SIZE>(
                row.register_state.rs1_val.unwrap(),
            ))),
            RV32IM::VIRTUAL_POW2I => Ok(Self::Pow2(POW2Instruction::<WORD_SIZE>(row.imm_u64()))),
            RV32IM::VIRTUAL_SRA_PAD => Ok(Self::RightShiftPadding(RightShiftPaddingInstruction::<
                WORD_SIZE,
            >(
                row.register_state.rs1_val.unwrap(),
            ))),
            RV32IM::VIRTUAL_SRA_PADI => {
                Ok(Self::RightShiftPadding(RightShiftPaddingInstruction::<
                    WORD_SIZE,
                >(row.imm_u64())))
            }

            _ => Err("No corresponding RV32I instruction"),
        }
    }
}

impl<const WORD_SIZE: usize> LookupTables<WORD_SIZE> {
    pub fn enum_index(instruction: &Self) -> usize {
        // Discriminant: https://doc.rust-lang.org/reference/items/enumerations.html#pointer-casting
        let byte = unsafe { *(instruction as *const Self as *const u8) };
        byte as usize
    }

    pub fn to_lookup_index(&self) -> u64 {
        match self {
            LookupTables::Add(instr) => instr.to_lookup_index(),
            LookupTables::Sub(instr) => instr.to_lookup_index(),
            LookupTables::And(instr) => instr.to_lookup_index(),
            LookupTables::Or(instr) => instr.to_lookup_index(),
            LookupTables::Xor(instr) => instr.to_lookup_index(),
            LookupTables::Beq(instr) => instr.to_lookup_index(),
            LookupTables::Bge(instr) => instr.to_lookup_index(),
            LookupTables::Bgeu(instr) => instr.to_lookup_index(),
            LookupTables::Bne(instr) => instr.to_lookup_index(),
            LookupTables::Slt(instr) => instr.to_lookup_index(),
            LookupTables::Sltu(instr) => instr.to_lookup_index(),
            LookupTables::Move(instr) => instr.to_lookup_index(),
            LookupTables::Movsign(instr) => instr.to_lookup_index(),
            LookupTables::Mul(instr) => instr.to_lookup_index(),
            LookupTables::Mulu(instr) => instr.to_lookup_index(),
            LookupTables::Mulhu(instr) => instr.to_lookup_index(),
            LookupTables::Advice(instr) => instr.to_lookup_index(),
            LookupTables::AssertLte(instr) => instr.to_lookup_index(),
            LookupTables::AssertValidSignedRemainder(instr) => instr.to_lookup_index(),
            LookupTables::AssertValidUnsignedRemainder(instr) => instr.to_lookup_index(),
            LookupTables::AssertValidDiv0(instr) => instr.to_lookup_index(),
            LookupTables::AssertHalfwordAlignment(instr) => instr.to_lookup_index(),
            LookupTables::Pow2(instr) => instr.to_lookup_index(),
            LookupTables::RightShiftPadding(instr) => instr.to_lookup_index(),
        }
    }

    #[cfg(test)]
    pub fn materialize(&self) -> Vec<u64> {
        match self {
            LookupTables::Add(instr) => instr.materialize(),
            LookupTables::Sub(instr) => instr.materialize(),
            LookupTables::And(instr) => instr.materialize(),
            LookupTables::Or(instr) => instr.materialize(),
            LookupTables::Xor(instr) => instr.materialize(),
            LookupTables::Beq(instr) => instr.materialize(),
            LookupTables::Bge(instr) => instr.materialize(),
            LookupTables::Bgeu(instr) => instr.materialize(),
            LookupTables::Bne(instr) => instr.materialize(),
            LookupTables::Slt(instr) => instr.materialize(),
            LookupTables::Sltu(instr) => instr.materialize(),
            LookupTables::Move(instr) => instr.materialize(),
            LookupTables::Movsign(instr) => instr.materialize(),
            LookupTables::Mul(instr) => instr.materialize(),
            LookupTables::Mulu(instr) => instr.materialize(),
            LookupTables::Mulhu(instr) => instr.materialize(),
            LookupTables::Advice(instr) => instr.materialize(),
            LookupTables::AssertLte(instr) => instr.materialize(),
            LookupTables::AssertValidSignedRemainder(instr) => instr.materialize(),
            LookupTables::AssertValidUnsignedRemainder(instr) => instr.materialize(),
            LookupTables::AssertValidDiv0(instr) => instr.materialize(),
            LookupTables::AssertHalfwordAlignment(instr) => instr.materialize(),
            LookupTables::Pow2(instr) => instr.materialize(),
            LookupTables::RightShiftPadding(instr) => instr.materialize(),
        }
    }

    pub fn materialize_entry(&self, index: u64) -> u64 {
        match self {
            LookupTables::Add(instr) => instr.materialize_entry(index),
            LookupTables::Sub(instr) => instr.materialize_entry(index),
            LookupTables::And(instr) => instr.materialize_entry(index),
            LookupTables::Or(instr) => instr.materialize_entry(index),
            LookupTables::Xor(instr) => instr.materialize_entry(index),
            LookupTables::Beq(instr) => instr.materialize_entry(index),
            LookupTables::Bge(instr) => instr.materialize_entry(index),
            LookupTables::Bgeu(instr) => instr.materialize_entry(index),
            LookupTables::Bne(instr) => instr.materialize_entry(index),
            LookupTables::Slt(instr) => instr.materialize_entry(index),
            LookupTables::Sltu(instr) => instr.materialize_entry(index),
            LookupTables::Move(instr) => instr.materialize_entry(index),
            LookupTables::Movsign(instr) => instr.materialize_entry(index),
            LookupTables::Mul(instr) => instr.materialize_entry(index),
            LookupTables::Mulu(instr) => instr.materialize_entry(index),
            LookupTables::Mulhu(instr) => instr.materialize_entry(index),
            LookupTables::Advice(instr) => instr.materialize_entry(index),
            LookupTables::AssertLte(instr) => instr.materialize_entry(index),
            LookupTables::AssertValidSignedRemainder(instr) => instr.materialize_entry(index),
            LookupTables::AssertValidUnsignedRemainder(instr) => instr.materialize_entry(index),
            LookupTables::AssertValidDiv0(instr) => instr.materialize_entry(index),
            LookupTables::AssertHalfwordAlignment(instr) => instr.materialize_entry(index),
            LookupTables::Pow2(instr) => instr.materialize_entry(index),
            LookupTables::RightShiftPadding(instr) => instr.materialize_entry(index),
        }
    }

    pub fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        match self {
            LookupTables::Add(instr) => instr.evaluate_mle(r),
            LookupTables::Sub(instr) => instr.evaluate_mle(r),
            LookupTables::And(instr) => instr.evaluate_mle(r),
            LookupTables::Or(instr) => instr.evaluate_mle(r),
            LookupTables::Xor(instr) => instr.evaluate_mle(r),
            LookupTables::Beq(instr) => instr.evaluate_mle(r),
            LookupTables::Bge(instr) => instr.evaluate_mle(r),
            LookupTables::Bgeu(instr) => instr.evaluate_mle(r),
            LookupTables::Bne(instr) => instr.evaluate_mle(r),
            LookupTables::Slt(instr) => instr.evaluate_mle(r),
            LookupTables::Sltu(instr) => instr.evaluate_mle(r),
            LookupTables::Move(instr) => instr.evaluate_mle(r),
            LookupTables::Movsign(instr) => instr.evaluate_mle(r),
            LookupTables::Mul(instr) => instr.evaluate_mle(r),
            LookupTables::Mulu(instr) => instr.evaluate_mle(r),
            LookupTables::Mulhu(instr) => instr.evaluate_mle(r),
            LookupTables::Advice(instr) => instr.evaluate_mle(r),
            LookupTables::AssertLte(instr) => instr.evaluate_mle(r),
            LookupTables::AssertValidSignedRemainder(instr) => instr.evaluate_mle(r),
            LookupTables::AssertValidUnsignedRemainder(instr) => instr.evaluate_mle(r),
            LookupTables::AssertValidDiv0(instr) => instr.evaluate_mle(r),
            LookupTables::AssertHalfwordAlignment(instr) => instr.evaluate_mle(r),
            LookupTables::Pow2(instr) => instr.evaluate_mle(r),
            LookupTables::RightShiftPadding(instr) => instr.evaluate_mle(r),
        }
    }

    pub fn random(rng: &mut StdRng, instruction: Option<Self>) -> Self {
        let instruction = instruction.unwrap_or_else(|| {
            let index = rng.next_u64() as usize % Self::COUNT;
            Self::iter()
                .enumerate()
                .filter(|(i, _)| *i == index)
                .map(|(_, x)| x)
                .next()
                .unwrap()
        });
        match instruction {
            LookupTables::Add(instr) => LookupTables::Add(instr.random(rng)),
            LookupTables::Sub(instr) => LookupTables::Sub(instr.random(rng)),
            LookupTables::And(instr) => LookupTables::And(instr.random(rng)),
            LookupTables::Or(instr) => LookupTables::Or(instr.random(rng)),
            LookupTables::Xor(instr) => LookupTables::Xor(instr.random(rng)),
            LookupTables::Beq(instr) => LookupTables::Beq(instr.random(rng)),
            LookupTables::Bge(instr) => LookupTables::Bge(instr.random(rng)),
            LookupTables::Bgeu(instr) => LookupTables::Bgeu(instr.random(rng)),
            LookupTables::Bne(instr) => LookupTables::Bne(instr.random(rng)),
            LookupTables::Slt(instr) => LookupTables::Slt(instr.random(rng)),
            LookupTables::Sltu(instr) => LookupTables::Sltu(instr.random(rng)),
            LookupTables::Move(instr) => LookupTables::Move(instr.random(rng)),
            LookupTables::Movsign(instr) => LookupTables::Movsign(instr.random(rng)),
            LookupTables::Mul(instr) => LookupTables::Mul(instr.random(rng)),
            LookupTables::Mulu(instr) => LookupTables::Mulu(instr.random(rng)),
            LookupTables::Mulhu(instr) => LookupTables::Mulhu(instr.random(rng)),
            LookupTables::Advice(instr) => LookupTables::Advice(instr.random(rng)),
            LookupTables::AssertLte(instr) => LookupTables::AssertLte(instr.random(rng)),
            LookupTables::AssertValidSignedRemainder(instr) => {
                LookupTables::AssertValidSignedRemainder(instr.random(rng))
            }
            LookupTables::AssertValidUnsignedRemainder(instr) => {
                LookupTables::AssertValidUnsignedRemainder(instr.random(rng))
            }
            LookupTables::AssertValidDiv0(instr) => {
                LookupTables::AssertValidDiv0(instr.random(rng))
            }
            LookupTables::AssertHalfwordAlignment(instr) => {
                LookupTables::AssertHalfwordAlignment(instr.random(rng))
            }
            LookupTables::Pow2(instr) => LookupTables::Pow2(instr.random(rng)),
            LookupTables::RightShiftPadding(instr) => {
                LookupTables::RightShiftPadding(instr.random(rng))
            }
        }
    }

    pub fn suffixes(&self) -> Vec<Suffixes> {
        match self {
            LookupTables::Add(instr) => instr.suffixes(),
            LookupTables::Sub(instr) => instr.suffixes(),
            LookupTables::And(instr) => instr.suffixes(),
            LookupTables::Or(instr) => instr.suffixes(),
            LookupTables::Xor(instr) => instr.suffixes(),
            LookupTables::Beq(instr) => instr.suffixes(),
            LookupTables::Bge(instr) => instr.suffixes(),
            LookupTables::Bgeu(instr) => instr.suffixes(),
            LookupTables::Bne(instr) => instr.suffixes(),
            LookupTables::Slt(instr) => instr.suffixes(),
            LookupTables::Sltu(instr) => instr.suffixes(),
            LookupTables::Move(instr) => instr.suffixes(),
            LookupTables::Movsign(instr) => instr.suffixes(),
            LookupTables::Mul(instr) => instr.suffixes(),
            LookupTables::Mulu(instr) => instr.suffixes(),
            LookupTables::Mulhu(instr) => instr.suffixes(),
            LookupTables::Advice(instr) => instr.suffixes(),
            LookupTables::AssertLte(instr) => instr.suffixes(),
            LookupTables::AssertValidSignedRemainder(instr) => instr.suffixes(),
            LookupTables::AssertValidUnsignedRemainder(instr) => instr.suffixes(),
            LookupTables::AssertValidDiv0(instr) => instr.suffixes(),
            LookupTables::AssertHalfwordAlignment(instr) => instr.suffixes(),
            LookupTables::Pow2(instr) => instr.suffixes(),
            LookupTables::RightShiftPadding(instr) => instr.suffixes(),
        }
    }

    pub fn combine<F: JoltField>(
        &self,
        prefixes: &[PrefixEval<F>],
        suffixes: &[SuffixEval<F>],
    ) -> F {
        match self {
            LookupTables::Add(instr) => instr.combine(prefixes, suffixes),
            LookupTables::Sub(instr) => instr.combine(prefixes, suffixes),
            LookupTables::And(instr) => instr.combine(prefixes, suffixes),
            LookupTables::Or(instr) => instr.combine(prefixes, suffixes),
            LookupTables::Xor(instr) => instr.combine(prefixes, suffixes),
            LookupTables::Beq(instr) => instr.combine(prefixes, suffixes),
            LookupTables::Bge(instr) => instr.combine(prefixes, suffixes),
            LookupTables::Bgeu(instr) => instr.combine(prefixes, suffixes),
            LookupTables::Bne(instr) => instr.combine(prefixes, suffixes),
            LookupTables::Slt(instr) => instr.combine(prefixes, suffixes),
            LookupTables::Sltu(instr) => instr.combine(prefixes, suffixes),
            LookupTables::Move(instr) => instr.combine(prefixes, suffixes),
            LookupTables::Movsign(instr) => instr.combine(prefixes, suffixes),
            LookupTables::Mul(instr) => instr.combine(prefixes, suffixes),
            LookupTables::Mulu(instr) => instr.combine(prefixes, suffixes),
            LookupTables::Mulhu(instr) => instr.combine(prefixes, suffixes),
            LookupTables::Advice(instr) => instr.combine(prefixes, suffixes),
            LookupTables::AssertLte(instr) => instr.combine(prefixes, suffixes),
            LookupTables::AssertValidSignedRemainder(instr) => instr.combine(prefixes, suffixes),
            LookupTables::AssertValidUnsignedRemainder(instr) => instr.combine(prefixes, suffixes),
            LookupTables::AssertValidDiv0(instr) => instr.combine(prefixes, suffixes),
            LookupTables::AssertHalfwordAlignment(instr) => instr.combine(prefixes, suffixes),
            LookupTables::Pow2(instr) => instr.combine(prefixes, suffixes),
            LookupTables::RightShiftPadding(instr) => instr.combine(prefixes, suffixes),
        }
    }
}
