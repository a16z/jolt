use add::ADDInstruction;
use and::ANDInstruction;
use beq::BEQInstruction;
use bge::BGEInstruction;
use bgeu::BGEUInstruction;
use bne::BNEInstruction;
use enum_dispatch::enum_dispatch;
use fixedbitset::*;
use mul::MULInstruction;
use mulhu::MULHUInstruction;
use mulu::MULUInstruction;
use or::ORInstruction;
use prefixes::PrefixEval;
use rand::rngs::StdRng;
use rand::RngCore;
use serde::Serialize;
use slt::SLTInstruction;
use sltu::SLTUInstruction;
use std::marker::Sync;
use std::ops::Range;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};
use sub::SUBInstruction;
use suffixes::{SuffixEval, Suffixes};
use tracer::{RVTraceRow, RegisterState};
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
use crate::jolt::subtable::LassoSubtable;
use crate::subprotocols::sparse_dense_shout::PrefixSuffixDecomposition;
use crate::utils::instruction_utils::chunk_operand;
use crate::utils::interleave_bits;
use common::rv_trace::ELFInstruction;
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

    /// Combines `vals` according to the instruction's "collation" polynomial `g`.
    /// If `vals` are subtable entries (as opposed to MLE evaluations), this function returns the
    /// output of the instruction. This function can also be thought of as the low-degree extension
    /// for the instruction.
    ///
    /// Params:
    /// - `vals`: Subtable entries or MLE evaluations. Assumed to be ordered
    ///   [T1_1, ..., T1_C, T2_1, ..., T2_C, ..., Tk_1, ..., Tk_C]
    ///   where T1, ..., Tk are the unique subtable types used by this instruction, in the order
    ///   given by the `subtables` method below. Note that some subtable values may be unused.
    /// - `C`: The "dimension" of the decomposition, i.e. the number of values read from each subtable.
    /// - `M`: The size of each subtable/memory.
    ///
    /// Returns: The combined value g(vals).
    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F;

    /// The degree of the `g` polynomial described by `combine_lookups`
    fn g_poly_degree(&self, C: usize) -> usize;

    /// Returns a Vec of the unique subtable types used by this instruction. For some instructions,
    /// e.g. SLL, the list of subtables depends on the dimension `C`.
    fn subtables<F: JoltField>(
        &self,
        C: usize,
        M: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)>;

    /// Converts the instruction operand(s) in their native word-sized representation into a Vec
    /// of subtable lookups indices. The returned Vec is length `C`, with elements in [0, `log_M`).
    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize>;

    /// Computes the output lookup entry for this instruction as a u64.
    fn lookup_entry(&self) -> u64;

    fn operand_chunks(&self, C: usize, log_M: usize) -> (Vec<u8>, Vec<u8>) {
        assert!(
            log_M % 2 == 0,
            "log_M must be even for operand_chunks to work"
        );
        let (left_operand, right_operand) = self.operands();
        (
            chunk_operand(left_operand, C, log_M / 2),
            chunk_operand(right_operand, C, log_M / 2),
        )
    }

    fn random(&self, rng: &mut StdRng) -> Self;

    fn slice_values<'a, F: JoltField>(&self, vals: &'a [F], C: usize, M: usize) -> Vec<&'a [F]> {
        let mut offset = 0;
        let mut slices = vec![];
        for (_, indices) in self.subtables::<F>(C, M) {
            slices.push(&vals[offset..offset + indices.len()]);
            offset += indices.len();
        }
        assert_eq!(offset, vals.len());
        slices
    }
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

#[derive(Clone)]
pub struct SubtableIndices {
    bitset: FixedBitSet,
}

impl SubtableIndices {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bitset: FixedBitSet::with_capacity(capacity),
        }
    }

    pub fn union_with(&mut self, other: &Self) {
        self.bitset.union_with(&other.bitset);
    }

    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.bitset.ones()
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.bitset.count_ones(..)
    }

    pub fn contains(&self, index: usize) -> bool {
        self.bitset.contains(index)
    }
}

impl From<usize> for SubtableIndices {
    fn from(index: usize) -> Self {
        let mut bitset = FixedBitSet::new();
        bitset.grow_and_insert(index);
        Self { bitset }
    }
}

impl From<Range<usize>> for SubtableIndices {
    fn from(range: Range<usize>) -> Self {
        let bitset = FixedBitSet::from_iter(range);
        Self { bitset }
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
pub mod sll_virtual_sequence;
pub mod slt;
pub mod sltu;
pub mod sra;
pub mod sra_virtual_sequence;
pub mod srl;
pub mod srl_virtual_sequence;
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

#[derive(Copy, Clone, Debug, Serialize, EnumIter, EnumCountMacro)]
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
