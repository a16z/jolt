use rand::rngs::StdRng;
use tracer::instruction::{RISCVCycle, RISCVInstruction, RV32IMCycle};

use crate::utils::interleave_bits;

use super::lookup_table::LookupTables;

pub trait InstructionLookup<const WORD_SIZE: usize> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>>;

    /// Converts this instruction's operands into a lookup index (as used in sparse-dense Shout).
    /// By default, interleaves the two bits of the two operands together.
    fn to_lookup_index(&self) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::to_lookup_query(self);
        interleave_bits(x as u32, y as u32)
    }

    /// Returns a tuple of the instruction's operands. If the instruction has only one operand,
    /// one of the tuple values will be 0.
    fn to_lookup_query(&self) -> (u64, u64);

    /// Computes the output lookup entry for this instruction as a u64.
    fn to_lookup_output(&self) -> u64;

    // fn random(&self, rng: &mut StdRng) -> RISCVCycle<Self>;
}

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for RV32IMCycle {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        match self {
            RV32IMCycle::NoOp => None,
            RV32IMCycle::ADD(cycle) => cycle.lookup_table(),
            RV32IMCycle::ADDI(cycle) => cycle.lookup_table(),
            RV32IMCycle::AND(cycle) => cycle.lookup_table(),
            RV32IMCycle::ANDI(cycle) => cycle.lookup_table(),
            RV32IMCycle::AUIPC(cycle) => cycle.lookup_table(),
            RV32IMCycle::BEQ(cycle) => cycle.lookup_table(),
            RV32IMCycle::BGE(cycle) => cycle.lookup_table(),
            RV32IMCycle::BGEU(cycle) => cycle.lookup_table(),
            RV32IMCycle::BLT(cycle) => cycle.lookup_table(),
            RV32IMCycle::BLTU(cycle) => cycle.lookup_table(),
            RV32IMCycle::BNE(cycle) => cycle.lookup_table(),
            RV32IMCycle::FENCE(cycle) => cycle.lookup_table(),
            RV32IMCycle::JAL(cycle) => cycle.lookup_table(),
            RV32IMCycle::JALR(cycle) => cycle.lookup_table(),
            RV32IMCycle::LUI(cycle) => cycle.lookup_table(),
            RV32IMCycle::LW(cycle) => cycle.lookup_table(),
            RV32IMCycle::MUL(cycle) => cycle.lookup_table(),
            RV32IMCycle::MULHU(cycle) => cycle.lookup_table(),
            RV32IMCycle::OR(cycle) => cycle.lookup_table(),
            RV32IMCycle::ORI(cycle) => cycle.lookup_table(),
            RV32IMCycle::SLT(cycle) => cycle.lookup_table(),
            RV32IMCycle::SLTI(cycle) => cycle.lookup_table(),
            RV32IMCycle::SLTIU(cycle) => cycle.lookup_table(),
            RV32IMCycle::SLTU(cycle) => cycle.lookup_table(),
            RV32IMCycle::SUB(cycle) => cycle.lookup_table(),
            RV32IMCycle::SW(cycle) => cycle.lookup_table(),
            RV32IMCycle::XOR(cycle) => cycle.lookup_table(),
            RV32IMCycle::XORI(cycle) => cycle.lookup_table(),
            RV32IMCycle::Advice(cycle) => cycle.lookup_table(),
            RV32IMCycle::AssertEQ(cycle) => cycle.lookup_table(),
            RV32IMCycle::AssertHalfwordAlignment(cycle) => cycle.lookup_table(),
            RV32IMCycle::AssertLTE(cycle) => cycle.lookup_table(),
            RV32IMCycle::AssertValidDiv0(cycle) => cycle.lookup_table(),
            RV32IMCycle::AssertValidSignedRemainder(cycle) => cycle.lookup_table(),
            RV32IMCycle::AssertValidUnsignedRemainder(cycle) => cycle.lookup_table(),
            RV32IMCycle::Move(cycle) => cycle.lookup_table(),
            RV32IMCycle::Movsign(cycle) => cycle.lookup_table(),
            RV32IMCycle::Pow2(cycle) => cycle.lookup_table(),
            RV32IMCycle::Pow2I(cycle) => cycle.lookup_table(),
            RV32IMCycle::ShiftRightBitmask(cycle) => cycle.lookup_table(),
            RV32IMCycle::ShiftRightBitmaskI(cycle) => cycle.lookup_table(),
            RV32IMCycle::VirtualSRA(cycle) => cycle.lookup_table(),
            RV32IMCycle::VirtualSRL(cycle) => cycle.lookup_table(),
            _ => panic!("Unexpected instruction {:?}", self),
        }
    }

    fn to_lookup_query(&self) -> (u64, u64) {
        match self {
            RV32IMCycle::NoOp => (0, 0),
            RV32IMCycle::ADD(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::ADDI(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::AND(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::ANDI(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::AUIPC(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::BEQ(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::BGE(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::BGEU(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::BLT(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::BLTU(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::BNE(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::FENCE(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::JAL(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::JALR(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::LUI(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::LW(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::MUL(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::MULHU(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::OR(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::ORI(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::SLT(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::SLTI(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::SLTIU(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::SLTU(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::SUB(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::SW(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::XOR(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::XORI(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::Advice(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::AssertEQ(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::AssertHalfwordAlignment(cycle) => {
                InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle)
            }
            RV32IMCycle::AssertLTE(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::AssertValidDiv0(cycle) => {
                InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle)
            }
            RV32IMCycle::AssertValidSignedRemainder(cycle) => {
                InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle)
            }
            RV32IMCycle::AssertValidUnsignedRemainder(cycle) => {
                InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle)
            }
            RV32IMCycle::Move(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::Movsign(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::Pow2(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::Pow2I(cycle) => InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::ShiftRightBitmask(cycle) => {
                InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle)
            }
            RV32IMCycle::ShiftRightBitmaskI(cycle) => {
                InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle)
            }
            RV32IMCycle::VirtualSRA(cycle) => {
                InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle)
            }
            RV32IMCycle::VirtualSRL(cycle) => {
                InstructionLookup::<WORD_SIZE>::to_lookup_query(cycle)
            }
            _ => panic!("Unexpected instruction {:?}", self),
        }
    }

    fn to_lookup_output(&self) -> u64 {
        todo!()
    }
}

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
pub mod fence;
pub mod jal;
pub mod jalr;
pub mod lui;
pub mod lw;
pub mod mul;
pub mod mulhu;
pub mod or;
pub mod ori;
pub mod slt;
pub mod slti;
pub mod sltiu;
pub mod sltu;
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
