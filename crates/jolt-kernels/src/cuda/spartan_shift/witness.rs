use jolt_riscv::{CircuitFlags, InstructionFlags as InstructionFlagKind};
use jolt_witness::witnesses::{InstructionFlag, OpFlag, Pc, UnexpandedPc};
use jolt_witness::WitnessBundle;

#[derive(Clone, Copy, Debug, PartialEq, Eq, WitnessBundle)]
pub struct SpartanShiftWitness {
    pub unexpanded_pc: UnexpandedPc,
    pub pc: Pc,
    #[opening(OpFlags(CircuitFlags::VirtualInstruction))]
    pub virtual_instruction: OpFlag,
    #[opening(OpFlags(CircuitFlags::IsFirstInSequence))]
    pub is_first_in_sequence: OpFlag,
    #[opening(InstructionFlags(InstructionFlagKind::IsNoop))]
    pub is_noop: InstructionFlag,
}
