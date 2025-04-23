use strum::EnumCount;
use strum_macros::EnumCount as EnumCountMacro;
use tracer::instruction::{RV32IMCycle, RV32IMInstruction};

use crate::utils::interleave_bits;

use super::lookup_table::LookupTables;

pub trait InstructionLookup<const WORD_SIZE: usize> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>>;
}

pub trait LookupQuery<const WORD_SIZE: usize> {
    /// Converts this instruction's operands into a lookup index (as used in sparse-dense Shout).
    /// By default, interleaves the two bits of the two operands together.
    fn to_lookup_index(&self) -> u64 {
        let (x, y) = LookupQuery::<WORD_SIZE>::to_lookup_query(self);
        interleave_bits(x as u32, y as u32)
    }

    /// Returns a tuple of the instruction's operands. If the instruction has only one operand,
    /// one of the tuple values will be 0.
    fn to_lookup_query(&self) -> (u64, u64);

    /// Computes the output lookup entry for this instruction as a u64.
    #[cfg(test)]
    fn to_lookup_output(&self) -> u64;
}

/// Boolean flags used in Jolt's R1CS constraints (`opflags` in the Jolt paper).
/// Note that the flags below deviate slightly from those described in Appendix A.1
/// of the Jolt paper.
#[derive(Clone, Copy, Debug, EnumCountMacro)]
pub enum CircuitFlags {
    /// 1 if the first lookup operand is the program counter; 0 otherwise (first lookup operand is RS1 value).
    LeftOperandIsPC,
    /// 1 if the second lookup operand is `imm`; 0 otherwise (second lookup operand is RS2 value).
    RightOperandIsImm,
    AddOperands,
    SubtractOperands,
    MultiplyOperands,
    /// 1 if the instruction is a load (i.e. `LW`)
    Load,
    /// 1 if the instruction is a store (i.e. `SW`)
    Store,
    /// 1 if the instruction is a jump (i.e. `JAL`, `JALR`)
    Jump,
    /// 1 if the instruction is a branch (i.e. `BEQ`, `BNE`, etc.)
    Branch,
    /// 1 if the lookup output is to be stored in `rd` at the end of the step.
    WriteLookupOutputToRD,
    /// 1 if the instruction is "virtual", as defined in Section 6.1 of the Jolt paper.
    Virtual,
    /// 1 if the instruction is an assert, as defined in Section 6.1.1 of the Jolt paper.
    Assert,
    /// Used in virtual sequences; the program counter should be the same for the full sequence.
    DoNotUpdatePC,
}

pub const NUM_CIRCUIT_FLAGS: usize = CircuitFlags::COUNT;

pub trait InstructionFlags {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS];

    fn bitflags(&self) -> u64 {
        todo!()
    }
}

impl InstructionLookup<32> for RV32IMInstruction {
    fn lookup_table(&self) -> Option<LookupTables<32>> {
        match self {
            RV32IMInstruction::NoOp => None,
            RV32IMInstruction::UNIMPL => None,
            RV32IMInstruction::ADD(add) => add.lookup_table(),
            RV32IMInstruction::ADDI(addi) => addi.lookup_table(),
            RV32IMInstruction::AND(and) => and.lookup_table(),
            RV32IMInstruction::ANDI(andi) => andi.lookup_table(),
            RV32IMInstruction::AUIPC(auipc) => auipc.lookup_table(),
            RV32IMInstruction::BEQ(beq) => beq.lookup_table(),
            RV32IMInstruction::BGE(bge) => bge.lookup_table(),
            RV32IMInstruction::BGEU(bgeu) => bgeu.lookup_table(),
            RV32IMInstruction::BLT(blt) => blt.lookup_table(),
            RV32IMInstruction::BLTU(bltu) => bltu.lookup_table(),
            RV32IMInstruction::BNE(bne) => bne.lookup_table(),
            RV32IMInstruction::FENCE(fence) => fence.lookup_table(),
            RV32IMInstruction::JAL(jal) => jal.lookup_table(),
            RV32IMInstruction::JALR(jalr) => jalr.lookup_table(),
            RV32IMInstruction::LUI(lui) => lui.lookup_table(),
            RV32IMInstruction::LW(lw) => lw.lookup_table(),
            RV32IMInstruction::MUL(mul) => mul.lookup_table(),
            RV32IMInstruction::MULHU(mulhu) => mulhu.lookup_table(),
            RV32IMInstruction::OR(or) => or.lookup_table(),
            RV32IMInstruction::ORI(ori) => ori.lookup_table(),
            RV32IMInstruction::SLT(slt) => slt.lookup_table(),
            RV32IMInstruction::SLTI(slti) => slti.lookup_table(),
            RV32IMInstruction::SLTIU(sltiu) => sltiu.lookup_table(),
            RV32IMInstruction::SLTU(sltu) => sltu.lookup_table(),
            RV32IMInstruction::SUB(sub) => sub.lookup_table(),
            RV32IMInstruction::SW(sw) => sw.lookup_table(),
            RV32IMInstruction::XOR(xor) => xor.lookup_table(),
            RV32IMInstruction::XORI(xori) => xori.lookup_table(),
            RV32IMInstruction::Advice(instr) => instr.lookup_table(),
            RV32IMInstruction::AssertEQ(instr) => instr.lookup_table(),
            RV32IMInstruction::AssertHalfwordAlignment(instr) => instr.lookup_table(),
            RV32IMInstruction::AssertLTE(instr) => instr.lookup_table(),
            RV32IMInstruction::AssertValidDiv0(instr) => instr.lookup_table(),
            RV32IMInstruction::AssertValidSignedRemainder(instr) => instr.lookup_table(),
            RV32IMInstruction::AssertValidUnsignedRemainder(instr) => instr.lookup_table(),
            RV32IMInstruction::Move(instr) => instr.lookup_table(),
            RV32IMInstruction::Movsign(instr) => instr.lookup_table(),
            RV32IMInstruction::Pow2(instr) => instr.lookup_table(),
            RV32IMInstruction::Pow2I(instr) => instr.lookup_table(),
            RV32IMInstruction::ShiftRightBitmask(instr) => instr.lookup_table(),
            RV32IMInstruction::ShiftRightBitmaskI(instr) => instr.lookup_table(),
            RV32IMInstruction::VirtualSRA(instr) => instr.lookup_table(),
            RV32IMInstruction::VirtualSRL(instr) => instr.lookup_table(),
            _ => panic!("Unexpected instruction {:?}", self),
        }
    }
}

impl InstructionFlags for RV32IMInstruction {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        match self {
            RV32IMInstruction::NoOp => [false; NUM_CIRCUIT_FLAGS],
            RV32IMInstruction::UNIMPL => [false; NUM_CIRCUIT_FLAGS],
            RV32IMInstruction::ADD(add) => add.circuit_flags(),
            RV32IMInstruction::ADDI(addi) => addi.circuit_flags(),
            RV32IMInstruction::AND(and) => and.circuit_flags(),
            RV32IMInstruction::ANDI(andi) => andi.circuit_flags(),
            RV32IMInstruction::AUIPC(auipc) => auipc.circuit_flags(),
            RV32IMInstruction::BEQ(beq) => beq.circuit_flags(),
            RV32IMInstruction::BGE(bge) => bge.circuit_flags(),
            RV32IMInstruction::BGEU(bgeu) => bgeu.circuit_flags(),
            RV32IMInstruction::BLT(blt) => blt.circuit_flags(),
            RV32IMInstruction::BLTU(bltu) => bltu.circuit_flags(),
            RV32IMInstruction::BNE(bne) => bne.circuit_flags(),
            RV32IMInstruction::FENCE(fence) => fence.circuit_flags(),
            RV32IMInstruction::JAL(jal) => jal.circuit_flags(),
            RV32IMInstruction::JALR(jalr) => jalr.circuit_flags(),
            RV32IMInstruction::LUI(lui) => lui.circuit_flags(),
            RV32IMInstruction::LW(lw) => lw.circuit_flags(),
            RV32IMInstruction::MUL(mul) => mul.circuit_flags(),
            RV32IMInstruction::MULHU(mulhu) => mulhu.circuit_flags(),
            RV32IMInstruction::OR(or) => or.circuit_flags(),
            RV32IMInstruction::ORI(ori) => ori.circuit_flags(),
            RV32IMInstruction::SLT(slt) => slt.circuit_flags(),
            RV32IMInstruction::SLTI(slti) => slti.circuit_flags(),
            RV32IMInstruction::SLTIU(sltiu) => sltiu.circuit_flags(),
            RV32IMInstruction::SLTU(sltu) => sltu.circuit_flags(),
            RV32IMInstruction::SUB(sub) => sub.circuit_flags(),
            RV32IMInstruction::SW(sw) => sw.circuit_flags(),
            RV32IMInstruction::XOR(xor) => xor.circuit_flags(),
            RV32IMInstruction::XORI(xori) => xori.circuit_flags(),
            RV32IMInstruction::Advice(instr) => instr.circuit_flags(),
            RV32IMInstruction::AssertEQ(instr) => instr.circuit_flags(),
            RV32IMInstruction::AssertHalfwordAlignment(instr) => instr.circuit_flags(),
            RV32IMInstruction::AssertLTE(instr) => instr.circuit_flags(),
            RV32IMInstruction::AssertValidDiv0(instr) => instr.circuit_flags(),
            RV32IMInstruction::AssertValidSignedRemainder(instr) => instr.circuit_flags(),
            RV32IMInstruction::AssertValidUnsignedRemainder(instr) => instr.circuit_flags(),
            RV32IMInstruction::Move(instr) => instr.circuit_flags(),
            RV32IMInstruction::Movsign(instr) => instr.circuit_flags(),
            RV32IMInstruction::Pow2(instr) => instr.circuit_flags(),
            RV32IMInstruction::Pow2I(instr) => instr.circuit_flags(),
            RV32IMInstruction::ShiftRightBitmask(instr) => instr.circuit_flags(),
            RV32IMInstruction::ShiftRightBitmaskI(instr) => instr.circuit_flags(),
            RV32IMInstruction::VirtualSRA(instr) => instr.circuit_flags(),
            RV32IMInstruction::VirtualSRL(instr) => instr.circuit_flags(),
            _ => panic!("Unexpected instruction {:?}", self),
        }
    }
}

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for RV32IMCycle {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        match self {
            RV32IMCycle::NoOp => None,
            RV32IMCycle::ADD(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::ADDI(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::AND(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::ANDI(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::AUIPC(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::BEQ(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::BGE(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::BGEU(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::BLT(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::BLTU(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::BNE(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::FENCE(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::JAL(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::JALR(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::LUI(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::LW(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::MUL(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::MULHU(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::OR(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::ORI(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::SLT(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::SLTI(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::SLTIU(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::SLTU(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::SUB(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::SW(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::XOR(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::XORI(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::Advice(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::AssertEQ(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::AssertHalfwordAlignment(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::AssertLTE(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::AssertValidDiv0(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::AssertValidSignedRemainder(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::AssertValidUnsignedRemainder(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::Move(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::Movsign(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::Pow2(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::Pow2I(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::ShiftRightBitmask(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::ShiftRightBitmaskI(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::VirtualSRA(cycle) => cycle.instruction.lookup_table(),
            RV32IMCycle::VirtualSRL(cycle) => cycle.instruction.lookup_table(),
            _ => panic!("Unexpected instruction {:?}", self),
        }
    }
}

impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for RV32IMCycle {
    fn to_lookup_query(&self) -> (u64, u64) {
        match self {
            RV32IMCycle::NoOp => (0, 0),
            RV32IMCycle::ADD(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::ADDI(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::AND(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::ANDI(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::AUIPC(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::BEQ(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::BGE(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::BGEU(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::BLT(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::BLTU(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::BNE(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::FENCE(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::JAL(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::JALR(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::LUI(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::LW(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::MUL(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::MULHU(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::OR(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::ORI(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::SLT(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::SLTI(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::SLTIU(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::SLTU(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::SUB(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::SW(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::XOR(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::XORI(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::Advice(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::AssertEQ(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::AssertHalfwordAlignment(cycle) => {
                LookupQuery::<WORD_SIZE>::to_lookup_query(cycle)
            }
            RV32IMCycle::AssertLTE(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::AssertValidDiv0(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::AssertValidSignedRemainder(cycle) => {
                LookupQuery::<WORD_SIZE>::to_lookup_query(cycle)
            }
            RV32IMCycle::AssertValidUnsignedRemainder(cycle) => {
                LookupQuery::<WORD_SIZE>::to_lookup_query(cycle)
            }
            RV32IMCycle::Move(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::Movsign(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::Pow2(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::Pow2I(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::ShiftRightBitmask(cycle) => {
                LookupQuery::<WORD_SIZE>::to_lookup_query(cycle)
            }
            RV32IMCycle::ShiftRightBitmaskI(cycle) => {
                LookupQuery::<WORD_SIZE>::to_lookup_query(cycle)
            }
            RV32IMCycle::VirtualSRA(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            RV32IMCycle::VirtualSRL(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_query(cycle),
            _ => panic!("Unexpected instruction {:?}", self),
        }
    }

    #[cfg(test)]
    fn to_lookup_output(&self) -> u64 {
        unimplemented!("unused")
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

#[cfg(test)]
pub mod test;
