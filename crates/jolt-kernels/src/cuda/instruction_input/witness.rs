#[cfg(test)]
use jolt_riscv::InstructionFlags as InstructionFlagKind;
#[cfg(test)]
use jolt_witness::witnesses::{Imm, InstructionFlag, Rs1Value, Rs2Value, UnexpandedPc};
#[cfg(test)]
use jolt_witness::WitnessBundle;

pub const COLUMNS: usize = 8;

pub const LEFT_IS_RS1_COLUMN: usize = 0;
pub const RS1_VALUE_COLUMN: usize = 1;
pub const LEFT_IS_PC_COLUMN: usize = 2;
pub const UNEXPANDED_PC_COLUMN: usize = 3;
pub const RIGHT_IS_RS2_COLUMN: usize = 4;
pub const RS2_VALUE_COLUMN: usize = 5;
pub const RIGHT_IS_IMM_COLUMN: usize = 6;
pub const IMM_COLUMN: usize = 7;

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, WitnessBundle)]
pub struct InstructionInputWitness {
    #[opening(InstructionFlags(InstructionFlagKind::LeftOperandIsRs1Value))]
    pub left_operand_is_rs1: InstructionFlag,
    #[opening(Rs1Value)]
    pub rs1_value: Rs1Value,
    #[opening(InstructionFlags(InstructionFlagKind::LeftOperandIsPC))]
    pub left_operand_is_pc: InstructionFlag,
    #[opening(UnexpandedPC)]
    pub unexpanded_pc: UnexpandedPc,
    #[opening(InstructionFlags(InstructionFlagKind::RightOperandIsRs2Value))]
    pub right_operand_is_rs2: InstructionFlag,
    #[opening(Rs2Value)]
    pub rs2_value: Rs2Value,
    #[opening(InstructionFlags(InstructionFlagKind::RightOperandIsImm))]
    pub right_operand_is_imm: InstructionFlag,
    #[opening(Imm)]
    pub imm: Imm,
}
