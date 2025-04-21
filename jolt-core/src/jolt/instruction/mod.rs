use rand::rngs::StdRng;
use tracer::instruction::{RISCVCycle, RISCVInstruction};

use crate::utils::interleave_bits;

use super::lookup_table::LookupTables;

pub trait InstructionLookup<const WORD_SIZE: usize>: RISCVInstruction {
    fn lookup_table() -> Option<LookupTables<WORD_SIZE>>;

    /// Converts this instruction's operands into a lookup index (as used in sparse-dense Shout).
    /// By default, interleaves the two bits of the two operands together.
    fn to_lookup_index(cycle: &RISCVCycle<Self>) -> u64 {
        let (x, y) = InstructionLookup::<WORD_SIZE>::lookup_query(cycle);
        interleave_bits(x as u32, y as u32)
    }

    /// Returns a tuple of the instruction's operands. If the instruction has only one operand,
    /// one of the tuple values will be 0.
    fn lookup_query(cycle: &RISCVCycle<Self>) -> (u64, u64);

    /// Computes the output lookup entry for this instruction as a u64.
    fn lookup_entry(cycle: &RISCVCycle<Self>) -> u64;

    // fn random(&self, rng: &mut StdRng) -> RISCVCycle<Self>;
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
