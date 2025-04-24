use common::constants::REGISTER_COUNT;

use crate::{field::JoltField, jolt::instruction::CircuitFlags};

use super::{
    builder::{CombinedUniformBuilder, OffsetEqConstraint, R1CSBuilder},
    inputs::JoltR1CSInputs,
};

pub const PC_START_ADDRESS: i64 = 0x80000000;
const PC_NOOP_SHIFT: i64 = 4;

pub trait R1CSConstraints<F: JoltField> {
    fn construct_constraints(
        padded_trace_length: usize,
        memory_start: u64,
    ) -> CombinedUniformBuilder<F> {
        let mut uniform_builder = R1CSBuilder::new();
        Self::uniform_constraints(&mut uniform_builder, memory_start);
        let cross_step_constraints = Self::cross_step_constraints();

        CombinedUniformBuilder::construct(
            uniform_builder,
            padded_trace_length,
            cross_step_constraints,
        )
    }
    /// Constructs Jolt's uniform constraints.
    /// Uniform constraints are constraints that hold for each step of
    /// the execution trace.
    fn uniform_constraints(builder: &mut R1CSBuilder, memory_start: u64);
    /// Construct's Jolt's cross-step constraints.
    /// Cross-step constraints are constraints whose inputs involve witness
    /// values from multiple steps of the execution trace.
    /// Currently, all of Jolt's cross-step constraints are of the form
    ///     if condition { some constraint on steps i and i+1 }
    /// This structure is captured in `OffsetEqConstraint`.
    fn cross_step_constraints() -> Vec<OffsetEqConstraint>;
}

pub struct JoltRV32IMConstraints;
impl<F: JoltField> R1CSConstraints<F> for JoltRV32IMConstraints {
    fn uniform_constraints(cs: &mut R1CSBuilder, memory_start: u64) {
        cs.constrain_if_else(
            JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsPC),
            JoltR1CSInputs::RealInstructionAddress - PC_NOOP_SHIFT,
            JoltR1CSInputs::Rs1Value,
            JoltR1CSInputs::LeftInstructionInput,
        );

        cs.constrain_if_else(
            JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsImm),
            JoltR1CSInputs::Imm,
            JoltR1CSInputs::Rs2Value,
            JoltR1CSInputs::RightInstructionInput,
        );

        let is_load_or_store = JoltR1CSInputs::OpFlags(CircuitFlags::Load)
            + JoltR1CSInputs::OpFlags(CircuitFlags::Store);
        let memory_start: i64 = memory_start.try_into().unwrap();
        cs.constrain_eq_conditional(
            is_load_or_store,
            JoltR1CSInputs::Rs1Value + JoltR1CSInputs::Imm,
            4 * JoltR1CSInputs::RamAddress + memory_start - 4 * REGISTER_COUNT as i64,
        );

        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::Load),
            JoltR1CSInputs::RamReadValue,
            JoltR1CSInputs::RamWriteValue,
        );

        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::Load),
            JoltR1CSInputs::RamReadValue,
            JoltR1CSInputs::RdWriteValue,
        );

        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::Store),
            JoltR1CSInputs::Rs2Value,
            JoltR1CSInputs::RamWriteValue,
        );

        cs.constrain_if_else(
            JoltR1CSInputs::OpFlags(CircuitFlags::SingleOperandLookup),
            0,
            JoltR1CSInputs::LeftInstructionInput,
            JoltR1CSInputs::LeftLookupOperand,
        );

        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands),
            JoltR1CSInputs::RightLookupOperand,
            JoltR1CSInputs::LeftInstructionInput + JoltR1CSInputs::RightInstructionInput,
        );

        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands),
            JoltR1CSInputs::RightLookupOperand,
            // Converts from unsigned to twos-complement representation
            JoltR1CSInputs::LeftInstructionInput - JoltR1CSInputs::RightInstructionInput
                + (0xffffffffi64 + 1),
        );

        cs.constrain_prod(
            JoltR1CSInputs::Rs1Value,
            JoltR1CSInputs::Rs2Value,
            JoltR1CSInputs::Product,
        );

        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands),
            JoltR1CSInputs::RightLookupOperand,
            JoltR1CSInputs::Product,
        );

        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::Assert),
            JoltR1CSInputs::LookupOutput,
            1,
        );

        // if (rd != 0 && update_rd_with_lookup_output == 1) constrain(rd_val == LookupOutput)
        cs.constrain_prod(
            JoltR1CSInputs::Rd,
            JoltR1CSInputs::OpFlags(CircuitFlags::WriteLookupOutputToRD),
            JoltR1CSInputs::WriteLookupOutputToRD,
        );

        cs.constrain_eq_conditional(
            JoltR1CSInputs::WriteLookupOutputToRD,
            JoltR1CSInputs::RdWriteValue,
            JoltR1CSInputs::LookupOutput,
        );

        // if (rd != 0 && is_jump_instr == 1) constrain(rd_val == 4 * PC)
        cs.constrain_prod(
            JoltR1CSInputs::Rd,
            JoltR1CSInputs::OpFlags(CircuitFlags::Jump),
            JoltR1CSInputs::WritePCtoRD,
        );

        cs.constrain_eq_conditional(
            JoltR1CSInputs::WritePCtoRD,
            JoltR1CSInputs::RealInstructionAddress,
            JoltR1CSInputs::RdWriteValue,
        );

        cs.constrain_if_else(
            JoltR1CSInputs::OpFlags(CircuitFlags::Jump),
            JoltR1CSInputs::LookupOutput + 4,
            JoltR1CSInputs::RealInstructionAddress + 4
                - 4 * JoltR1CSInputs::OpFlags(CircuitFlags::DoNotUpdatePC),
            JoltR1CSInputs::NextPCJump,
        );

        cs.constrain_prod(
            JoltR1CSInputs::OpFlags(CircuitFlags::Branch),
            JoltR1CSInputs::LookupOutput,
            JoltR1CSInputs::ShouldBranch,
        );

        cs.constrain_if_else(
            JoltR1CSInputs::ShouldBranch,
            JoltR1CSInputs::RealInstructionAddress + JoltR1CSInputs::Imm,
            JoltR1CSInputs::NextPCJump,
            JoltR1CSInputs::NextPC,
        );
    }

    fn cross_step_constraints() -> Vec<OffsetEqConstraint> {
        // If the next instruction's ELF address is not zero (i.e. it's
        // not padding), then check the PC update.
        let pc_constraint = OffsetEqConstraint::new(
            (JoltR1CSInputs::RealInstructionAddress, true),
            (JoltR1CSInputs::NextPC, false),
            (JoltR1CSInputs::RealInstructionAddress, true),
        );

        // If the current instruction is virtual, check that the next instruction
        // in the trace is the next instruction in bytecode. Virtual sequences
        // do not involve jumps or branches, so this should always hold,
        // EXCEPT if we encounter a virtual instruction followed by a padding
        // instruction. But that should never happen because the execution
        // trace should always end with some return handling, which shouldn't involve
        // any virtual sequences.
        let virtual_sequence_constraint = OffsetEqConstraint::new(
            (JoltR1CSInputs::OpFlags(CircuitFlags::Virtual), false),
            (JoltR1CSInputs::VirutalInstructionAddress, true),
            (JoltR1CSInputs::VirutalInstructionAddress + 1, false),
        );

        vec![pc_constraint, virtual_sequence_constraint]
    }
}
