use crate::{
    assert_static_aux_index,
    field::JoltField,
    impl_r1cs_input_lc_conversions,
    input_range,
    jolt::vm::{ read_write_memory, rv32i_vm::C },
    r1cs::ops::Term,
};

use super::{
    builder::{ R1CSBuilder, R1CSConstraintBuilder },
    ops::{ ConstraintInput, Variable, LC },
};

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
    Ord
)]
#[repr(usize)]
pub enum JoltIn {
    PcIn,

    Bytecode_A, // Virtual address
    // Bytecode_V
    Bytecode_ELFAddress,
    Bytecode_Opcode,
    Bytecode_RS1,
    Bytecode_RS2,
    Bytecode_RD,
    Bytecode_Imm,

    RAM_A,
    // Ram_V
    RAM_Read_RS1,
    RAM_Read_RS2,
    RAM_Read_RD, // TODO(sragss): Appears to be unused?
    RAM_Read_Byte0,
    RAM_Read_Byte1,
    RAM_Read_Byte2,
    RAM_Read_Byte3,
    RAM_Write_RD,
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
    OpFlags_IsRs1Rs2,
    OpFlags_IsImm,
    OpFlags_IsLbu,
    OpFlags_IsLhu,
    OpFlags_IsLw,
    OpFlags_IsLb,
    OpFlags_IsLh,
    OpFlags_IsSb,
    OpFlags_IsSh,
    OpFlags_IsSw,
    OpFlags_IsJmp,
    OpFlags_IsBranch,
    OpFlags_LookupOutToRd,
    OpFlags_SignImm,
    OpFlags_IsConcat,
    OpFlags_IsVirtualSequence,
    OpFlags_IsVirtual,

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

    // Remainder
    Remainder,
}
impl_r1cs_input_lc_conversions!(JoltIn);
impl ConstraintInput for JoltIn {}

pub const PC_START_ADDRESS: i64 = 0x80000000;
const PC_NOOP_SHIFT: i64 = 4;
const LOG_M: usize = 16;
const OPERAND_SIZE: usize = LOG_M / 2;
pub const PC_BRANCH_AUX_INDEX: usize = 15;

pub struct JoltConstraints {
    memory_start: u64,
}

impl JoltConstraints {
    pub fn new(memory_start: u64) -> Self {
        Self { memory_start }
    }
}

impl<F: JoltField> R1CSConstraintBuilder<F> for JoltConstraints {
    type Inputs = JoltIn;
    fn build_constraints(&self, cs: &mut R1CSBuilder<F, Self::Inputs>) {
        let flags = input_range!(JoltIn::OpFlags_IsRs1Rs2, JoltIn::IF_MulHu);
        for flag in flags {
            cs.constrain_binary(flag);
        }

        cs.constrain_eq(JoltIn::PcIn, JoltIn::Bytecode_A);

        cs.constrain_pack_be(flags.to_vec(), JoltIn::Bytecode_Opcode, 1);

        let real_pc = LC::sum2(4i64 * JoltIn::PcIn, PC_START_ADDRESS - PC_NOOP_SHIFT);
        let x = cs.allocate_if_else(JoltIn::OpFlags_IsRs1Rs2, real_pc, JoltIn::RAM_Read_RS1);
        let y = cs.allocate_if_else(
            JoltIn::OpFlags_IsImm,
            JoltIn::Bytecode_Imm,
            JoltIn::RAM_Read_RS2
        );

        // Converts from unsigned to twos-complement representation
        let signed_output = LC::sub2(JoltIn::Bytecode_Imm, 0xffffffffi64 + 1i64);
        let imm_signed = cs.allocate_if_else(
            JoltIn::OpFlags_SignImm,
            signed_output,
            JoltIn::Bytecode_Imm
        );

        let packed_query = cs.allocate_pack_be(
            input_range!(JoltIn::ChunksQ_0, JoltIn::ChunksQ_3).to_vec(),
            LOG_M
        );

        cs.constrain_eq_conditional(JoltIn::IF_Add, packed_query, x + y);
        // Converts from unsigned to twos-complement representation
        cs.constrain_eq_conditional(JoltIn::IF_Sub, packed_query, x - y + (0xffffffffi64 + 1));

        // CONSTRAINT - actual_address - 4 * load_store_address - remainder = 0 (TBA)
        
        // CONSTRAINT - (LB_flag + LBU_flag + SB_flag) [remainder*(remainder -1)*(remainder -2)*(remainder-3)] + (LH_flag + LHU_flag + SH_flag) [remainder*(remainder -2)] + (LW_flag + SW_flag)*remainder  = 0

        // remainder * (remainder - 2) -> remainder02
        let remainder = cs.create_remainder_term_with_value(0);
        let remainder_minus_2_term = cs.create_remainder_term_with_value(-2);
        let remainder02 = cs.allocate_prod(
            remainder,
            remainder_minus_2_term
        );

        // (remainder - 1) * (remainder - 3) -> remainder13
        let remainder_minus_1_term = cs.create_remainder_term_with_value(-1);
        let remainder_minus_3_term = cs.create_remainder_term_with_value(-3);
        let remainder13 = cs.allocate_prod(remainder_minus_1_term, remainder_minus_3_term);

        // remainder * (remainder - 2) * (remainder - 1) * (remainder - 3) -> remainder0123
        let remainder0123 = cs.allocate_prod(remainder02, remainder13);

        let remainder012 = cs.allocate_prod(remainder02, remainder_minus_1_term);
        let remainder023 = cs.allocate_prod(remainder02, remainder_minus_3_term);
        let remainder013 = cs.allocate_prod(remainder, remainder13);
        let remainder123 = cs.allocate_prod(remainder_minus_2_term, remainder13);

        // (LB_flag + LBU_flag + SB_flag) [remainder*(remainder -1)*(remainder -2)*(remainder-3)] -> product4
        let lb_lbu_sb_sum = LC::sum_any(
            vec![
                JoltIn::OpFlags_IsLb,
                JoltIn::OpFlags_IsLbu,
                JoltIn::OpFlags_IsSb
            ]
        );
        let product4 = cs.allocate_prod(remainder0123, lb_lbu_sb_sum);

        // (LH_flag + LHU_flag + SH_flag) [remainder*(remainder -2)] -> product5
        let lh_lhu_sh_sum = LC::sum3(
            JoltIn::OpFlags_IsLh,
            JoltIn::OpFlags_IsLhu,
            JoltIn::OpFlags_IsSh
        );
        let product5 = cs.allocate_prod(remainder02, lh_lhu_sh_sum);

        // (LW_flag + SW_flag)*remainder -> product6
        let lw_sw_sum = LC::sum2(JoltIn::OpFlags_IsLw, JoltIn::OpFlags_IsSw);
        let product6 = cs.allocate_prod(remainder, lw_sw_sum);

        // product4 + product5 + product6 = 0
        let sum = LC::new(vec![product4.into(), product5.into(), product6.into()]);
        cs.constrain_eq_zero(sum);


        // CONSTRAINT - actual_address is computed correctly using rs1_val, and imm_extension
        let flag_0_or_1_condition = LC::sum_any(
            vec![
                JoltIn::OpFlags_IsLb,
                JoltIn::OpFlags_IsLbu,
                JoltIn::OpFlags_IsLh,
                JoltIn::OpFlags_IsLhu,
                JoltIn::OpFlags_IsLw,
                JoltIn::OpFlags_IsSb,
                JoltIn::OpFlags_IsSh,
                JoltIn::OpFlags_IsSw,
            ]
        );
        // (TODO: Vineet) Check this later
        // let memory_start: i64 = self.memory_start.try_into().unwrap();
        cs.constrain_eq_conditional(
            flag_0_or_1_condition,
            LC::sum2(JoltIn::RAM_Read_RS1, imm_signed),
            JoltIn::RAM_A
        );

        // LOAD CONSTRAINT a
        // For the load instructions, we have that the four bytes read at 
        // index load_store_address of memory is the same as written

        let all_load_flags = LC::sum_any(
            vec![
                JoltIn::OpFlags_IsLb,
                JoltIn::OpFlags_IsLbu,
                JoltIn::OpFlags_IsLh,
                JoltIn::OpFlags_IsLhu,
                JoltIn::OpFlags_IsLw,
            ]
        );
        
        cs.constrain_eq_conditional(
            all_load_flags.clone(),
            JoltIn::RAM_Read_Byte0,
            JoltIn::RAM_Write_Byte0,
        );
        cs.constrain_eq_conditional(
            all_load_flags.clone(),
            JoltIn::RAM_Read_Byte1,
            JoltIn::RAM_Write_Byte1,
        );
        cs.constrain_eq_conditional(
            all_load_flags.clone(),
            JoltIn::RAM_Read_Byte2,
            JoltIn::RAM_Write_Byte2,
        );
        cs.constrain_eq_conditional(
            all_load_flags,
            JoltIn::RAM_Read_Byte3,
            JoltIn::RAM_Write_Byte3,
        );

        // LOAD CONSTRAINT b-1
        // (JoltIn::OpFlags_IsLb) [ (memory_read[0] - combined_z_chunks) *
        //                  (remainder - 1) * (remainder - 2) * (remainder - 3) +
        //                  (memory_read[1] - combined_z_chunks) * remainder * (remainder - 2) * (remainder - 3) +
        //                  (memory_read[2] - combined_z_chunks) * remainder * (remainder - 1) * (remainder - 3) +
        //                  (memory_read[3] - combined_z_chunks) * remainder * (remainder - 1) * (remainder - 2)
        //                  ] = 0

        let term0 = cs.allocate_prod(LC::sub2(JoltIn::RAM_Read_Byte0, packed_query), remainder123);
        let term1 = cs.allocate_prod(LC::sub2(JoltIn::RAM_Read_Byte1, packed_query), remainder023);
        let term2 = cs.allocate_prod(LC::sub2(JoltIn::RAM_Read_Byte2, packed_query), remainder013);
        let term3 = cs.allocate_prod(LC::sub2(JoltIn::RAM_Read_Byte3, packed_query), remainder012);
        let lb_lbu_flag = LC::sum2(JoltIn::OpFlags_IsLb, JoltIn::OpFlags_IsLbu);
        let term = LC::new(vec![term0.into(), term1.into(), term2.into(), term3.into()]);
        cs.constrain_prod(lb_lbu_flag, term, LC::zero());

        // LOAD CONSTRAINT b-2
        // (LH_flag) [ (memory_read[0] + 2^{8}memory_read[1] - combined_z_chunks) * (remainder - 2)  +  
        //                  (memory_read[2] + 2^{8}*memory_read[3] - combined_z_chunks) * remainder 
        //                ] = 0
        let lc1 = LC::new(
            vec![
                cs.create_memory_term(JoltIn::RAM_Read_Byte0, 1),
                cs.create_memory_term(JoltIn::RAM_Read_Byte1, 1 << 8)
            ]
        );
        let lc2 = LC::new(
            vec![
                cs.create_memory_term(JoltIn::RAM_Read_Byte2, 1),
                cs.create_memory_term(JoltIn::RAM_Read_Byte3, 1 << 8)
            ]
        );
        let term00 = cs.allocate_prod(LC::sub2(lc1, packed_query), remainder_minus_2_term);
        let term01 = cs.allocate_prod(LC::sub2(lc2, packed_query), JoltIn::Remainder);
        let lh_lhu_flag = LC::sum2(JoltIn::OpFlags_IsLh, JoltIn::OpFlags_IsLhu);
        let term02 = LC::new(vec![term00.into(), term01.into()]);
        cs.constrain_prod(lh_lhu_flag, term02, LC::zero());

        // LOAD CONSTRAINT b-3
        // (LW_flag) [ memory_read[0] + 2^{8}memory_read[1] + 2^{16}memory_read[2] + 
        //                  2^{24}memory_read[3]  - combined_z_chunks) ] = 0

        let lc3 = LC::new(
            vec![
                cs.create_memory_term(JoltIn::RAM_Read_Byte0, 1),
                cs.create_memory_term(JoltIn::RAM_Read_Byte1, 1 << 8),
                cs.create_memory_term(JoltIn::RAM_Read_Byte2, 1 << 16),
                cs.create_memory_term(JoltIn::RAM_Read_Byte3, 1 << 24)
            ]
        );
        cs.constrain_prod(JoltIn::OpFlags_IsLw, LC::sub2(lc3.clone(), packed_query), LC::zero());

        // LOAD CONSTRAINT c
        // Constraint to check rd is updated with lookup output 
        // check this constraint later
        let rd_nonzero_and_lookup_to_rd =
            cs.allocate_prod(JoltIn::Bytecode_RD, JoltIn::OpFlags_LookupOutToRd);
        cs.constrain_eq_conditional(
            rd_nonzero_and_lookup_to_rd,
            JoltIn::RAM_Write_RD,
            JoltIn::LookupOutput,
        );

        // STORE CONSTRAINT a
        // (SB_flag + SH_flag + SW_flag) [ rs2_val - combined_z_chunks] +  
        // (LBU_flag + LHU_flag)[
        //              memory_read[0] + 2^{8}memory_read[1] + 
        //              2^{16}memory_read[2] + 2^{24}memory_read[3] - 
        //              combined_z_chunks
        //           ] = 0

        let all_store_flags = LC::sum3(JoltIn::OpFlags_IsSb, JoltIn::OpFlags_IsSh, JoltIn::OpFlags_IsSw);
        let term00 = cs.allocate_prod(all_store_flags, LC::sub2(JoltIn::RAM_Read_RS2, packed_query));

        let lbu_lhu_flag = LC::sum2(JoltIn::OpFlags_IsLbu, JoltIn::OpFlags_IsLhu);
        let term01 = cs.allocate_prod(lbu_lhu_flag, LC::sub2(lc3, packed_query));
        let sum = LC::new(vec![term00.into(), term01.into()]);
        cs.constrain_eq_zero(sum);

        // STORE CONSTRAINT b-1
        // (SB_flag) [ 
        //           (memory_write[0]  - lookup_output) *  (remainder - 1) (remainder - 2) * (remainder - 3) +  
        //           (memory_write[1] - lookup_output) * remainder * (remainder - 2) * (remainder - 3) + 
        //           (memory_write[2] - lookup_output) * remainder * (remainder - 1) * (remainder - 3) + 
        //           (memory_write[3] - lookup_output) * remainder * (remainder - 1) * (remainder - 2)
        //       ] = 0
        let term0 = cs.allocate_prod(LC::sub2(JoltIn::RAM_Write_Byte0, JoltIn::LookupOutput), remainder123);
        let term1 = cs.allocate_prod(LC::sub2(JoltIn::RAM_Write_Byte1, JoltIn::LookupOutput), remainder023);
        let term2 = cs.allocate_prod(LC::sub2(JoltIn::RAM_Write_Byte2, JoltIn::LookupOutput), remainder013);
        let term3 = cs.allocate_prod(LC::sub2(JoltIn::RAM_Write_Byte3, JoltIn::LookupOutput), remainder012);
        let term = LC::new(vec![term0.into(), term1.into(), term2.into(), term3.into()]);
        cs.constrain_prod(JoltIn::OpFlags_IsSb, term, LC::zero());


        // STORE CONSTRAINT b-2
        // (SH_flag) [ 
        //           (memory_write[0] + 2^{8}memory_write[1] - lookup_output) * (remainder - 2)  +  
        //           (memory_write[2] +  2^{8}*memory_write[3] - lookup_output) * remainder 
        //     ] = 0
        let lc1 = LC::new(
            vec![
                cs.create_memory_term(JoltIn::RAM_Write_Byte0, 1),
                cs.create_memory_term(JoltIn::RAM_Write_Byte1, 1 << 8)
            ]
        );
        let lc2 = LC::new(
            vec![
                cs.create_memory_term(JoltIn::RAM_Write_Byte2, 1),
                cs.create_memory_term(JoltIn::RAM_Write_Byte3, 1 << 8)
            ]
        );
        let term00 = cs.allocate_prod(LC::sub2(lc1, JoltIn::LookupOutput), remainder_minus_2_term);
        let term01 = cs.allocate_prod(LC::sub2(lc2, JoltIn::LookupOutput), JoltIn::Remainder);
        let term02 = LC::new(vec![term00.into(), term01.into()]);
        cs.constrain_prod(JoltIn::OpFlags_IsSh, term02, LC::zero());
        
       // STORE CONSTRAINT b-3
       // (SW_flag) [
       //           memory_write[0] + 2^{8}memory_write[1] + 
       //           2^{16}memory_write[2] + 2^{24}memory_write[3]  - 
       //           combined_z_chunks) 
       //           ] = 0

       let lc4 = LC::new(
        vec![
                cs.create_memory_term(JoltIn::RAM_Write_Byte0, 1),
                cs.create_memory_term(JoltIn::RAM_Write_Byte1, 1 << 8),
                cs.create_memory_term(JoltIn::RAM_Write_Byte2, 1 << 16),
                cs.create_memory_term(JoltIn::RAM_Write_Byte3, 1 << 24)
            ]
        );
        cs.constrain_prod(JoltIn::OpFlags_IsSw, LC::sub2(lc4, packed_query), LC::zero());
       
        // TODO(sragss): Uses 2 excess constraints for condition gating. Could make constrain_pack_be_conditional... Or make everything conditional...
        let chunked_x = cs.allocate_pack_be(
            input_range!(JoltIn::ChunksX_0, JoltIn::ChunksX_3).to_vec(),
            OPERAND_SIZE
        );
        let chunked_y = cs.allocate_pack_be(
            input_range!(JoltIn::ChunksY_0, JoltIn::ChunksY_3).to_vec(),
            OPERAND_SIZE
        );
        cs.constrain_eq_conditional(JoltIn::OpFlags_IsConcat, chunked_x, x);
        cs.constrain_eq_conditional(JoltIn::OpFlags_IsConcat, chunked_y, y);

        // if is_shift ? chunks_query[i] == zip(chunks_x[i], chunks_y[C-1]) : chunks_query[i] == zip(chunks_x[i], chunks_y[i])
        let is_shift = JoltIn::IF_Sll + JoltIn::IF_Srl + JoltIn::IF_Sra;
        let chunks_x = input_range!(JoltIn::ChunksX_0, JoltIn::ChunksX_3);
        let chunks_y = input_range!(JoltIn::ChunksY_0, JoltIn::ChunksY_3);
        let chunks_query = input_range!(JoltIn::ChunksQ_0, JoltIn::ChunksQ_3);
        for i in 0..C {
            let relevant_chunk_y = cs.allocate_if_else(
                is_shift.clone(),
                chunks_y[C - 1],
                chunks_y[i]
            );
            cs.constrain_eq_conditional(
                JoltIn::OpFlags_IsConcat,
                chunks_query[i],
                (1i64 << 8) * chunks_x[i] + relevant_chunk_y
            );
        }

        // if (rd != 0 && update_rd_with_lookup_output == 1) constrain(rd_val == LookupOutput)
        // if (rd != 0 && is_jump_instr == 1) constrain(rd_val == 4 * PC)

        let rd_nonzero_and_jmp = cs.allocate_prod(JoltIn::Bytecode_RD, JoltIn::OpFlags_IsJmp);
        let lhs = LC::sum2(JoltIn::PcIn, PC_START_ADDRESS - PC_NOOP_SHIFT);
        let rhs = JoltIn::RAM_Write_RD;
        cs.constrain_eq_conditional(rd_nonzero_and_jmp, lhs, rhs);

        let branch_and_lookup_output =
            cs.allocate_prod(JoltIn::OpFlags_IsBranch, JoltIn::LookupOutput);
        let next_pc_jump = cs.allocate_if_else(
            JoltIn::OpFlags_IsJmp,
            JoltIn::LookupOutput + 4,
            4 * JoltIn::PcIn + PC_START_ADDRESS + 4,
        );

        let next_pc_jump_branch = cs.allocate_if_else(
            branch_and_lookup_output,
            4 * JoltIn::PcIn + PC_START_ADDRESS + imm_signed,
            next_pc_jump,
        );
        assert_static_aux_index!(next_pc_jump_branch, PC_BRANCH_AUX_INDEX);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::r1cs::builder::{ CombinedUniformBuilder, OffsetEqConstraint };

    use ark_bn254::Fr;
    use strum::EnumCount;

    #[test]
    fn single_instruction_jolt() {
        let mut uniform_builder = R1CSBuilder::<Fr, JoltIn>::new();

        let jolt_constraints = JoltConstraints::new(0);
        jolt_constraints.build_constraints(&mut uniform_builder);

        let num_steps = 1;
        let combined_builder = CombinedUniformBuilder::construct(
            uniform_builder,
            num_steps,
            OffsetEqConstraint::empty()
        );
        let mut inputs = vec![vec![Fr::zero(); num_steps]; JoltIn::COUNT];

        // ADD instruction
        inputs[JoltIn::PcIn as usize][0] = Fr::from(10);
        inputs[JoltIn::Bytecode_A as usize][0] = Fr::from(10);
        inputs[JoltIn::Bytecode_Opcode as usize][0] = Fr::from(0);
        inputs[JoltIn::Bytecode_RS1 as usize][0] = Fr::from(2);
        inputs[JoltIn::Bytecode_RS2 as usize][0] = Fr::from(3);
        inputs[JoltIn::Bytecode_RD as usize][0] = Fr::from(4);

        inputs[JoltIn::RAM_Read_RD as usize][0] = Fr::from(0);
        inputs[JoltIn::RAM_Read_RS1 as usize][0] = Fr::from(100);
        inputs[JoltIn::RAM_Read_RS2 as usize][0] = Fr::from(200);
        inputs[JoltIn::RAM_Write_RD as usize][0] = Fr::from(300);
        // remainder RAM == 0

        // rv_trace::to_circuit_flags
        // all zero for ADD
        inputs[JoltIn::OpFlags_IsRs1Rs2 as usize][0] = Fr::zero(); // first_operand = rs1
        inputs[JoltIn::OpFlags_IsImm as usize][0] = Fr::zero(); // second_operand = rs2 => immediate

        let aux = combined_builder.compute_aux(&inputs);
        let (az, bz, cz) = combined_builder.compute_spartan(&inputs, &aux);

        combined_builder.assert_valid(&az, &bz, &cz);
    }
}
