use crate::{
    assert_static_aux_index, field::JoltField, impl_r1cs_input_lc_conversions, input_range,
    jolt::vm::rv32i_vm::C,
};

use super::{
    builder::{CombinedUniformBuilder, OffsetEqConstraint, R1CSBuilder, R1CSConstraintBuilder},
    ops::{ConstraintInput, Variable},
};

pub fn construct_jolt_constraints<F: JoltField>(
    padded_trace_length: usize,
    memory_start: u64,
) -> CombinedUniformBuilder<F, JoltIn> {
    let mut uniform_builder = R1CSBuilder::<F, JoltIn>::new();
    let constraints = UniformJoltConstraints::new(memory_start);
    constraints.build_constraints(&mut uniform_builder);

    // If the next instruction's ELF address is not zero (i.e. it's
    // not padding), then check the PC update.
    let pc_constraint = OffsetEqConstraint::new(
        (JoltIn::Bytecode_ELFAddress, true),
        (Variable::Auxiliary(NEXT_PC), false),
        (4 * JoltIn::Bytecode_ELFAddress + PC_START_ADDRESS, true),
    );

    // If the current instruction is virtual, check that the next instruction
    // in the trace is the next instruction in bytecode. Virtual sequences
    // do not involve jumps or branches, so this should always hold,
    // EXCEPT if we encounter a virtual instruction followed by a padding
    // instruction. But that should never happen because the execution
    // trace should always end with some return handling, which shouldn't involve
    // any virtual sequences.

    let virtual_sequence_constraint = OffsetEqConstraint::new(
        (JoltIn::OpFlags_IsVirtualInstruction, false),
        (JoltIn::Bytecode_A, true),
        (JoltIn::Bytecode_A + 1, false),
    );

    CombinedUniformBuilder::construct(
        uniform_builder,
        padded_trace_length,
        vec![pc_constraint, virtual_sequence_constraint],
    )
}

// TODO(#377): Dedupe OpFlags / CircuitFlags
// TODO(#378): Explicit unit test for comparing OpFlags and InstructionFlags
#[allow(non_camel_case_types)]
#[derive(
    strum_macros::EnumIter,
    strum_macros::EnumCount,
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Hash,
    Ord,
)]
#[repr(usize)]
pub enum JoltIn {
    PcIn,

    Bytecode_A, // Virtual address
    // Bytecode_V
    Bytecode_ELFAddress,
    Bytecode_Bitflags,
    Bytecode_RS1,
    Bytecode_RS2,
    Bytecode_RD,
    Bytecode_Imm,

    RAM_A,
    // Ram_V
    RS1_Read,
    RS2_Read,
    RD_Read,
    RAM_Read_Byte0,
    RAM_Read_Byte1,
    RAM_Read_Byte2,
    RAM_Read_Byte3,
    RD_Write,
    RAM_Write_Byte0,
    RAM_Write_Byte1,
    RAM_Write_Byte2,
    RAM_Write_Byte3,

    ChunksX_0,
    ChunksX_1,
    ChunksX_2,
    ChunksX_3,

    ChunksY_0,
    ChunksY_1,
    ChunksY_2,
    ChunksY_3,

    ChunksQ_0,
    ChunksQ_1,
    ChunksQ_2,
    ChunksQ_3,

    LookupOutput,

    // Should match rv_trace.to_circuit_flags()
    OpFlags_IsPC,
    OpFlags_IsImm,
    OpFlags_IsLoad,
    OpFlags_IsStore,
    OpFlags_IsJmp,
    OpFlags_IsBranch,
    OpFlags_LookupOutToRd,
    OpFlags_SignImm,
    OpFlags_IsConcat,
    OpFlags_IsVirtualInstruction,
    OpFlags_IsAssert,
    OpFlags_DoNotUpdatePC,

    // Instruction Flags
    // Should match JoltInstructionSet
    IF_Add,
    IF_Sub,
    IF_And,
    IF_Or,
    IF_Xor,
    IF_Lb,
    IF_Lh,
    IF_Sb,
    IF_Sh,
    IF_Sw,
    IF_Beq,
    IF_Bge,
    IF_Bgeu,
    IF_Bne,
    IF_Slt,
    IF_Sltu,
    IF_Sll,
    IF_Sra,
    IF_Srl,
    IF_Movsign,
    IF_Mul,
    IF_MulU,
    IF_MulHu,
    IF_Virt_Advice,
    IF_Virt_Move,
    IF_Virt_Assert_LTE,
    IF_Virt_Assert_VALID_SIGNED_REMAINDER,
    IF_Virt_Assert_VALID_UNSIGNED_REMAINDER,
    IF_Virt_Assert_VALID_DIV0,
}
impl_r1cs_input_lc_conversions!(JoltIn);
impl ConstraintInput for JoltIn {}

pub const PC_START_ADDRESS: i64 = 0x80000000;
const PC_NOOP_SHIFT: i64 = 4;
const LOG_M: usize = 16;
const OPERAND_SIZE: usize = LOG_M / 2;
pub const NEXT_PC: usize = 12;

pub struct UniformJoltConstraints {
    memory_start: u64,
}

impl UniformJoltConstraints {
    pub fn new(memory_start: u64) -> Self {
        Self { memory_start }
    }
}

impl<F: JoltField> R1CSConstraintBuilder<F> for UniformJoltConstraints {
    type Inputs = JoltIn;
    fn build_constraints(&self, cs: &mut R1CSBuilder<F, Self::Inputs>) {
        let flags = input_range!(JoltIn::OpFlags_IsPC, JoltIn::IF_Virt_Assert_VALID_DIV0);
        for flag in flags {
            cs.constrain_binary(flag);
        }

        cs.constrain_eq(JoltIn::PcIn, JoltIn::Bytecode_A);

        cs.constrain_pack_be(flags.to_vec(), JoltIn::Bytecode_Bitflags, 1);

        let real_pc = 4i64 * JoltIn::Bytecode_ELFAddress + (PC_START_ADDRESS - PC_NOOP_SHIFT);
        let x = cs.allocate_if_else(JoltIn::OpFlags_IsPC, real_pc, JoltIn::RS1_Read);
        let y = cs.allocate_if_else(
            JoltIn::OpFlags_IsImm,
            JoltIn::Bytecode_Imm,
            JoltIn::RS2_Read,
        );

        // Converts from unsigned to twos-complement representation
        let signed_output = JoltIn::Bytecode_Imm - (0xffffffffi64 + 1i64);
        let imm_signed =
            cs.allocate_if_else(JoltIn::OpFlags_SignImm, signed_output, JoltIn::Bytecode_Imm);

        let is_load_or_store = JoltIn::OpFlags_IsLoad + JoltIn::OpFlags_IsStore;
        let memory_start: i64 = self.memory_start.try_into().unwrap();
        cs.constrain_eq_conditional(
            is_load_or_store,
            JoltIn::RS1_Read + imm_signed,
            JoltIn::RAM_A + memory_start,
        );

        cs.constrain_eq_conditional(
            JoltIn::OpFlags_IsLoad,
            JoltIn::RAM_Read_Byte0,
            JoltIn::RAM_Write_Byte0,
        );
        cs.constrain_eq_conditional(
            JoltIn::OpFlags_IsLoad,
            JoltIn::RAM_Read_Byte1,
            JoltIn::RAM_Write_Byte1,
        );
        cs.constrain_eq_conditional(
            JoltIn::OpFlags_IsLoad,
            JoltIn::RAM_Read_Byte2,
            JoltIn::RAM_Write_Byte2,
        );
        cs.constrain_eq_conditional(
            JoltIn::OpFlags_IsLoad,
            JoltIn::RAM_Read_Byte3,
            JoltIn::RAM_Write_Byte3,
        );

        let ram_writes = input_range!(JoltIn::RAM_Write_Byte0, JoltIn::RAM_Write_Byte3);
        let packed_load_store = R1CSBuilder::<F, JoltIn>::pack_le(ram_writes.to_vec(), 8);
        cs.constrain_eq_conditional(
            JoltIn::OpFlags_IsStore,
            packed_load_store.clone(),
            JoltIn::LookupOutput,
        );

        let packed_query = R1CSBuilder::<F, JoltIn>::pack_be(
            input_range!(JoltIn::ChunksQ_0, JoltIn::ChunksQ_3).to_vec(),
            LOG_M,
        );

        cs.constrain_eq_conditional(JoltIn::IF_Add, packed_query.clone(), x + y);
        // Converts from unsigned to twos-complement representation
        cs.constrain_eq_conditional(
            JoltIn::IF_Sub,
            packed_query.clone(),
            x - y + (0xffffffffi64 + 1),
        );
        let is_mul = JoltIn::IF_Mul + JoltIn::IF_MulU + JoltIn::IF_MulHu;
        let product = cs.allocate_prod(x, y);
        cs.constrain_eq_conditional(is_mul, packed_query.clone(), product);
        cs.constrain_eq_conditional(
            JoltIn::IF_Movsign + JoltIn::IF_Virt_Move,
            packed_query.clone(),
            x,
        );
        cs.constrain_eq_conditional(
            JoltIn::OpFlags_IsLoad,
            packed_query.clone(),
            packed_load_store,
        );
        cs.constrain_eq_conditional(JoltIn::OpFlags_IsStore, packed_query, JoltIn::RS2_Read);

        cs.constrain_eq_conditional(JoltIn::OpFlags_IsAssert, JoltIn::LookupOutput, 1);

        let chunked_x = R1CSBuilder::<F, JoltIn>::pack_be(
            input_range!(JoltIn::ChunksX_0, JoltIn::ChunksX_3).to_vec(),
            OPERAND_SIZE,
        );
        let chunked_y = R1CSBuilder::<F, JoltIn>::pack_be(
            input_range!(JoltIn::ChunksY_0, JoltIn::ChunksY_3).to_vec(),
            OPERAND_SIZE,
        );
        cs.constrain_eq_conditional(JoltIn::OpFlags_IsConcat, chunked_x, x);
        cs.constrain_eq_conditional(JoltIn::OpFlags_IsConcat, chunked_y, y);

        // if is_shift ? chunks_query[i] == zip(chunks_x[i], chunks_y[C-1]) : chunks_query[i] == zip(chunks_x[i], chunks_y[i])
        let is_shift = JoltIn::IF_Sll + JoltIn::IF_Srl + JoltIn::IF_Sra;
        let chunks_x = input_range!(JoltIn::ChunksX_0, JoltIn::ChunksX_3);
        let chunks_y = input_range!(JoltIn::ChunksY_0, JoltIn::ChunksY_3);
        let chunks_query = input_range!(JoltIn::ChunksQ_0, JoltIn::ChunksQ_3);
        for i in 0..C {
            let relevant_chunk_y =
                cs.allocate_if_else(is_shift.clone(), chunks_y[C - 1], chunks_y[i]);
            cs.constrain_eq_conditional(
                JoltIn::OpFlags_IsConcat,
                chunks_query[i],
                (1i64 << 8) * chunks_x[i] + relevant_chunk_y,
            );
        }

        // if (rd != 0 && update_rd_with_lookup_output == 1) constrain(rd_val == LookupOutput)
        // if (rd != 0 && is_jump_instr == 1) constrain(rd_val == 4 * PC)
        let rd_nonzero_and_lookup_to_rd =
            cs.allocate_prod(JoltIn::Bytecode_RD, JoltIn::OpFlags_LookupOutToRd);
        cs.constrain_eq_conditional(
            rd_nonzero_and_lookup_to_rd,
            JoltIn::RD_Write,
            JoltIn::LookupOutput,
        );
        let rd_nonzero_and_jmp = cs.allocate_prod(JoltIn::Bytecode_RD, JoltIn::OpFlags_IsJmp);
        let lhs = JoltIn::Bytecode_ELFAddress + (PC_START_ADDRESS - PC_NOOP_SHIFT);
        let rhs = JoltIn::RD_Write;
        cs.constrain_eq_conditional(rd_nonzero_and_jmp, lhs, rhs);

        let next_pc_jump = cs.allocate_if_else(
            JoltIn::OpFlags_IsJmp,
            JoltIn::LookupOutput + 4,
            4 * JoltIn::Bytecode_ELFAddress + PC_START_ADDRESS + 4
                - 4 * JoltIn::OpFlags_DoNotUpdatePC,
        );

        let should_branch = cs.allocate_prod(JoltIn::OpFlags_IsBranch, JoltIn::LookupOutput);
        let next_pc = cs.allocate_if_else(
            should_branch,
            4 * JoltIn::Bytecode_ELFAddress + PC_START_ADDRESS + imm_signed,
            next_pc_jump,
        );

        assert_static_aux_index!(next_pc, NEXT_PC);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{jolt::vm::rv32i_vm::RV32I, r1cs::builder::CombinedUniformBuilder};

    use ark_bn254::Fr;
    use ark_std::Zero;
    use strum::EnumCount;

    #[test]
    fn instruction_flags_length() {
        assert_eq!(
            input_range!(JoltIn::IF_Add, JoltIn::IF_Virt_Assert_VALID_DIV0).len(),
            RV32I::COUNT
        );
    }

    #[test]
    fn single_instruction_jolt() {
        let mut uniform_builder = R1CSBuilder::<Fr, JoltIn>::new();

        let jolt_constraints = UniformJoltConstraints::new(0);
        jolt_constraints.build_constraints(&mut uniform_builder);

        let num_steps = 1;
        let combined_builder =
            CombinedUniformBuilder::construct(uniform_builder, num_steps, vec![]);
        let mut inputs = vec![vec![Fr::zero(); num_steps]; JoltIn::COUNT];

        // ADD instruction
        inputs[JoltIn::PcIn as usize][0] = Fr::from(10);
        inputs[JoltIn::Bytecode_A as usize][0] = Fr::from(10);
        inputs[JoltIn::Bytecode_Bitflags as usize][0] = Fr::from(0);
        inputs[JoltIn::Bytecode_RS1 as usize][0] = Fr::from(2);
        inputs[JoltIn::Bytecode_RS2 as usize][0] = Fr::from(3);
        inputs[JoltIn::Bytecode_RD as usize][0] = Fr::from(4);

        inputs[JoltIn::RD_Read as usize][0] = Fr::from(0);
        inputs[JoltIn::RS1_Read as usize][0] = Fr::from(100);
        inputs[JoltIn::RS2_Read as usize][0] = Fr::from(200);
        inputs[JoltIn::RD_Write as usize][0] = Fr::from(300);
        // remainder RAM == 0

        // rv_trace::to_circuit_flags
        // all zero for ADD
        inputs[JoltIn::OpFlags_IsPC as usize][0] = Fr::zero(); // first_operand = rs1
        inputs[JoltIn::OpFlags_IsImm as usize][0] = Fr::zero(); // second_operand = rs2 => immediate

        let aux = combined_builder.compute_aux(&inputs);

        let (az, bz, cz) = combined_builder.compute_spartan_Az_Bz_Cz(&inputs, &aux);
        combined_builder.assert_valid(&az, &bz, &cz);
    }
}
