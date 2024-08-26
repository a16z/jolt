use common::{constants::RAM_OPS_PER_INSTRUCTION, rv_trace::CircuitFlags};
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

use crate::{
    field::JoltField,
    impl_r1cs_input_lc_conversions,
    jolt::{
        instruction::{
            add::ADDInstruction, mul::MULInstruction, mulhu::MULHUInstruction,
            mulu::MULUInstruction, sll::SLLInstruction, sra::SRAInstruction, srl::SRLInstruction,
            sub::SUBInstruction, virtual_move::MOVEInstruction,
            virtual_movsign::MOVSIGNInstruction,
        },
        vm::{rv32i_vm::RV32I, JoltPolynomials},
    },
    poly::{commitment::commitment_scheme::CommitmentScheme, dense_mlpoly::DensePolynomial},
};

use super::{
    builder::{CombinedUniformBuilder, OffsetEqConstraint, R1CSBuilder, R1CSConstraintBuilder},
    ops::{ConstraintInput, Variable},
};

pub fn construct_jolt_constraints<const C: usize, F: JoltField, I: ConstraintInput>(
    padded_trace_length: usize,
    memory_start: u64,
) -> CombinedUniformBuilder<C, F, I> {
    let mut uniform_builder = R1CSBuilder::<C, F, I>::new();
    let constraints = UniformJoltConstraints::new(memory_start);
    constraints.build_constraints(&mut uniform_builder);

    // If the next instruction's ELF address is not zero (i.e. it's
    // not padding), then check the PC update.
    let pc_constraint = OffsetEqConstraint::new(
        (JoltIn::Bytecode_ELFAddress, true),
        (JoltIn::Aux(AuxVariable::NextPC), false),
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
        (JoltIn::OpFlags(CircuitFlags::Virtual), false),
        (JoltIn::Bytecode_A, true),
        (JoltIn::Bytecode_A + 1, false),
    );

    CombinedUniformBuilder::construct(
        uniform_builder,
        padded_trace_length,
        vec![pc_constraint, virtual_sequence_constraint],
    )
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Hash, Ord, EnumIter)]
pub enum JoltIn {
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
    RAM_Read(usize),
    RD_Write,
    RAM_Write(usize),

    ChunksQuery(usize),
    LookupOutput,
    ChunksX(usize),
    ChunksY(usize),

    OpFlags(CircuitFlags),
    InstructionFlags(RV32I),
    Aux(AuxVariable),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Hash, Ord, Default, EnumIter)]
enum AuxVariable {
    #[default] // Need a default so that we can derive EnumIter on `JoltIn`
    LeftLookupOperand,
    RightLookupOperand,
    ImmSigned,
    Product,
    RelevantYChunk(usize),
    WriteLookupOutputToRD,
    WritePCtoRD,
    NextPCJump,
    ShouldBranch,
    NextPC,
}

impl_r1cs_input_lc_conversions!(JoltIn, 4);
impl ConstraintInput for JoltIn {
    fn flatten<const C: usize>() -> Vec<Self> {
        JoltIn::iter()
            .flat_map(|variant| match variant {
                Self::RAM_Read(_) => (0..RAM_OPS_PER_INSTRUCTION)
                    .into_iter()
                    .map(|i| Self::RAM_Read(i))
                    .collect(),
                Self::RAM_Write(_) => (0..RAM_OPS_PER_INSTRUCTION)
                    .into_iter()
                    .map(|i| Self::RAM_Write(i))
                    .collect(),
                Self::ChunksQuery(_) => (0..C).into_iter().map(|i| Self::ChunksQuery(i)).collect(),
                Self::ChunksX(_) => (0..C).into_iter().map(|i| Self::ChunksX(i)).collect(),
                Self::ChunksY(_) => (0..C).into_iter().map(|i| Self::ChunksY(i)).collect(),
                Self::OpFlags(_) => CircuitFlags::iter()
                    .map(|flag| Self::OpFlags(flag))
                    .collect(),
                Self::InstructionFlags(_) => RV32I::iter()
                    .map(|flag| Self::InstructionFlags(flag))
                    .collect(),
                Self::Aux(_) => AuxVariable::iter()
                    .flat_map(|aux| match aux {
                        AuxVariable::RelevantYChunk(_) => (0..C)
                            .into_iter()
                            .map(|i| Self::Aux(AuxVariable::RelevantYChunk(i)))
                            .collect(),
                        _ => vec![Self::Aux(aux)],
                    })
                    .collect(),
                _ => vec![variant],
            })
            .collect()
    }
    fn get_poly_ref<'a, F: JoltField, PCS: CommitmentScheme<Field = F>>(
        &self,
        jolt_polynomials: &'a JoltPolynomials<F, PCS>,
    ) -> &'a DensePolynomial<F> {
        todo!()
    }

    fn get_poly_ref_mut<F: JoltField, PCS: CommitmentScheme<Field = F>>(
        &self,
        jolt_polynomials: &mut JoltPolynomials<F, PCS>,
    ) -> &mut DensePolynomial<F> {
        todo!();
    }
}

pub const PC_START_ADDRESS: i64 = 0x80000000;
const PC_NOOP_SHIFT: i64 = 4;
const LOG_M: usize = 16;
const OPERAND_SIZE: usize = LOG_M / 2;

pub struct UniformJoltConstraints {
    memory_start: u64,
}

impl UniformJoltConstraints {
    pub fn new(memory_start: u64) -> Self {
        Self { memory_start }
    }
}

impl<const C: usize, F: JoltField> R1CSConstraintBuilder<C, F, JoltIn> for UniformJoltConstraints {
    type Inputs = JoltIn;
    fn build_constraints(&self, cs: &mut R1CSBuilder<C, F, JoltIn>) {
        for flag in RV32I::iter() {
            cs.constrain_binary(JoltIn::InstructionFlags(flag));
        }
        for flag in CircuitFlags::iter() {
            cs.constrain_binary(JoltIn::OpFlags(flag));
        }

        let flags = RV32I::iter()
            .map(|flag| JoltIn::InstructionFlags(flag).into())
            .chain(CircuitFlags::iter().map(|flag| JoltIn::OpFlags(flag).into()))
            .collect();
        cs.constrain_pack_be(flags, JoltIn::Bytecode_Bitflags, 1);

        let real_pc = 4i64 * JoltIn::Bytecode_ELFAddress + (PC_START_ADDRESS - PC_NOOP_SHIFT);
        let x = cs.allocate_if_else(
            JoltIn::Aux(AuxVariable::LeftLookupOperand),
            JoltIn::OpFlags(CircuitFlags::RS1IsPC),
            real_pc,
            JoltIn::RS1_Read,
        );
        let y = cs.allocate_if_else(
            JoltIn::Aux(AuxVariable::RightLookupOperand),
            JoltIn::OpFlags(CircuitFlags::RS2IsImm),
            JoltIn::Bytecode_Imm,
            JoltIn::RS2_Read,
        );

        // Converts from unsigned to twos-complement representation
        let signed_output = JoltIn::Bytecode_Imm - (0xffffffffi64 + 1i64);
        let imm_signed = cs.allocate_if_else(
            JoltIn::Aux(AuxVariable::ImmSigned),
            JoltIn::OpFlags(CircuitFlags::ImmSignBit),
            signed_output,
            JoltIn::Bytecode_Imm,
        );

        let is_load_or_store =
            JoltIn::OpFlags(CircuitFlags::Load) + JoltIn::OpFlags(CircuitFlags::Store);
        let memory_start: i64 = self.memory_start.try_into().unwrap();
        cs.constrain_eq_conditional(
            is_load_or_store,
            JoltIn::RS1_Read + imm_signed,
            JoltIn::RAM_A + memory_start,
        );

        for i in 0..RAM_OPS_PER_INSTRUCTION {
            cs.constrain_eq_conditional(
                JoltIn::OpFlags(CircuitFlags::Load),
                JoltIn::RAM_Read(i),
                JoltIn::RAM_Write(i),
            );
        }

        let ram_writes = (0..RAM_OPS_PER_INSTRUCTION)
            .into_iter()
            .map(|i| Variable::Input(JoltIn::RAM_Write(i).to_index::<C>()))
            .collect();
        let packed_load_store = R1CSBuilder::<C, F, JoltIn>::pack_le(ram_writes, 8);
        cs.constrain_eq_conditional(
            JoltIn::OpFlags(CircuitFlags::Store),
            packed_load_store.clone(),
            JoltIn::LookupOutput,
        );

        let query_chunks: Vec<Variable> = (0..C)
            .into_iter()
            .map(|i| Variable::Input(JoltIn::ChunksQuery(i).to_index::<C>()))
            .collect();
        let packed_query = R1CSBuilder::<C, F, JoltIn>::pack_be(query_chunks.clone(), LOG_M);

        cs.constrain_eq_conditional(
            JoltIn::InstructionFlags(ADDInstruction::default().into()),
            packed_query.clone(),
            x + y,
        );
        // Converts from unsigned to twos-complement representation
        cs.constrain_eq_conditional(
            JoltIn::InstructionFlags(SUBInstruction::default().into()),
            packed_query.clone(),
            x - y + (0xffffffffi64 + 1),
        );
        let is_mul = JoltIn::InstructionFlags(MULInstruction::default().into())
            + JoltIn::InstructionFlags(MULUInstruction::default().into())
            + JoltIn::InstructionFlags(MULHUInstruction::default().into());
        let product = cs.allocate_prod(JoltIn::Aux(AuxVariable::Product), x, y);
        cs.constrain_eq_conditional(is_mul, packed_query.clone(), product);
        cs.constrain_eq_conditional(
            JoltIn::InstructionFlags(MOVSIGNInstruction::default().into())
                + JoltIn::InstructionFlags(MOVEInstruction::default().into()),
            packed_query.clone(),
            x,
        );
        cs.constrain_eq_conditional(
            JoltIn::OpFlags(CircuitFlags::Load),
            packed_query.clone(),
            packed_load_store,
        );
        cs.constrain_eq_conditional(
            JoltIn::OpFlags(CircuitFlags::Store),
            packed_query,
            JoltIn::RS2_Read,
        );

        cs.constrain_eq_conditional(
            JoltIn::OpFlags(CircuitFlags::Assert),
            JoltIn::LookupOutput,
            1,
        );

        let x_chunks: Vec<Variable> = (0..C)
            .into_iter()
            .map(|i| Variable::Input(JoltIn::ChunksX(i).to_index::<C>()))
            .collect();
        let y_chunks: Vec<Variable> = (0..C)
            .into_iter()
            .map(|i| Variable::Input(JoltIn::ChunksY(i).to_index::<C>()))
            .collect();
        let x_concat = R1CSBuilder::<C, F, JoltIn>::pack_be(x_chunks.clone(), OPERAND_SIZE);
        let y_concat = R1CSBuilder::<C, F, JoltIn>::pack_be(y_chunks.clone(), OPERAND_SIZE);
        cs.constrain_eq_conditional(
            JoltIn::OpFlags(CircuitFlags::ConcatLookupQueryChunks),
            x_concat,
            x,
        );
        cs.constrain_eq_conditional(
            JoltIn::OpFlags(CircuitFlags::ConcatLookupQueryChunks),
            y_concat,
            y,
        );

        // if is_shift ? chunks_query[i] == zip(chunks_x[i], chunks_y[C-1]) : chunks_query[i] == zip(chunks_x[i], chunks_y[i])
        let is_shift = JoltIn::InstructionFlags(SLLInstruction::default().into())
            + JoltIn::InstructionFlags(SRLInstruction::default().into())
            + JoltIn::InstructionFlags(SRAInstruction::default().into());
        for i in 0..C {
            let relevant_chunk_y = cs.allocate_if_else(
                JoltIn::Aux(AuxVariable::RelevantYChunk(i)),
                is_shift.clone(),
                y_chunks[C - 1],
                y_chunks[i],
            );
            cs.constrain_eq_conditional(
                JoltIn::OpFlags(CircuitFlags::ConcatLookupQueryChunks),
                query_chunks[i],
                x_chunks[i] * (1i64 << 8) + relevant_chunk_y,
            );
        }

        // if (rd != 0 && update_rd_with_lookup_output == 1) constrain(rd_val == LookupOutput)
        // if (rd != 0 && is_jump_instr == 1) constrain(rd_val == 4 * PC)
        let rd_nonzero_and_lookup_to_rd = cs.allocate_prod(
            JoltIn::Aux(AuxVariable::WriteLookupOutputToRD),
            JoltIn::Bytecode_RD,
            JoltIn::OpFlags(CircuitFlags::WriteLookupOutputToRD),
        );
        cs.constrain_eq_conditional(
            rd_nonzero_and_lookup_to_rd,
            JoltIn::RD_Write,
            JoltIn::LookupOutput,
        );
        let rd_nonzero_and_jmp = cs.allocate_prod(
            JoltIn::Aux(AuxVariable::WritePCtoRD),
            JoltIn::Bytecode_RD,
            JoltIn::OpFlags(CircuitFlags::Jump),
        );
        let lhs = JoltIn::Bytecode_ELFAddress + (PC_START_ADDRESS - PC_NOOP_SHIFT);
        let rhs = JoltIn::RD_Write;
        cs.constrain_eq_conditional(rd_nonzero_and_jmp, lhs, rhs);

        let next_pc_jump = cs.allocate_if_else(
            JoltIn::Aux(AuxVariable::NextPCJump),
            JoltIn::OpFlags(CircuitFlags::Jump),
            JoltIn::LookupOutput + 4,
            4 * JoltIn::Bytecode_ELFAddress + PC_START_ADDRESS + 4
                - 4 * JoltIn::OpFlags(CircuitFlags::DoNotUpdatePC),
        );

        let should_branch = cs.allocate_prod(
            JoltIn::Aux(AuxVariable::ShouldBranch),
            JoltIn::OpFlags(CircuitFlags::Branch),
            JoltIn::LookupOutput,
        );
        let _next_pc = cs.allocate_if_else(
            JoltIn::Aux(AuxVariable::NextPC),
            should_branch,
            4 * JoltIn::Bytecode_ELFAddress + PC_START_ADDRESS + imm_signed,
            next_pc_jump,
        );
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
