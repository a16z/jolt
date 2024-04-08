/// Handwritten circuit 

use ark_ff::PrimeField; 
use smallvec::{smallvec, SmallVec};
use rayon::prelude::*;
use common::{constants::{RAM_START_ADDRESS, RAM_WITNESS_OFFSET}, rv_trace::NUM_CIRCUIT_FLAGS};
use strum::EnumCount;

use crate::jolt::{instruction::{add::ADDInstruction, sll::SLLInstruction, sra::SRAInstruction, srl::SRLInstruction, sub::SUBInstruction, JoltInstructionSet}, vm::rv32i_vm::RV32I};

use super::snark::R1CSStepInputs;

/* Compiler Variables */
const C: usize = 4; 
const N_FLAGS: usize = NUM_CIRCUIT_FLAGS + RV32I::COUNT; 
const W: usize = 32;
const LOG_M: usize = 16; 
const MEMORY_ADDRESS_OFFSET: usize = (RAM_START_ADDRESS - RAM_WITNESS_OFFSET) as usize; 
const PC_START_ADDRESS: u64 = RAM_START_ADDRESS;
const PC_NOOP_SHIFT: usize = 4;
// "memreg ops per step" 
const MOPS: usize = 7;
/* End of Compiler Variables */

const L_CHUNK: usize = LOG_M/2;  
const PC_IDX: usize = 0; 

const ALL_ONES: i64 = 0xffffffff;

const ONE: (usize, i64) = (0, 1);

const STATE_LENGTH: usize = 1;

#[derive(Debug, PartialEq, Eq, Hash)]
enum InputType {
    Constant       = 0,
    InputState     = 1,
    OutputState    = 2,
    ProgARW        = 3,
    ProgVRW        = 4,
    MemregARW      = 5,
    MemregVReads   = 6,
    MemregVWrites  = 7,
    ChunksX        = 8,
    ChunksY        = 9,
    ChunksQuery    = 10,
    LookupOutput   = 11,
    OpFlags        = 12,
    InstrFlags     = 13,
}

const INPUT_SIZES: &[(InputType, usize)] = &[
    (InputType::Constant,        1), 
    (InputType::InputState,      STATE_LENGTH),
    (InputType::OutputState,     STATE_LENGTH),
    (InputType::ProgARW,         1),
    (InputType::ProgVRW,         5),
    (InputType::MemregARW,       1),
    (InputType::MemregVReads,    7),
    (InputType::MemregVWrites,   5),
    (InputType::ChunksX,         C),
    (InputType::ChunksY,         C),
    (InputType::ChunksQuery,     C),
    (InputType::LookupOutput,    1),
    (InputType::OpFlags,         NUM_CIRCUIT_FLAGS),
    (InputType::InstrFlags,      RV32I::COUNT),
];

const INPUT_OFFSETS: [usize; INPUT_SIZES.len()] = {
    let mut arr = [0; INPUT_SIZES.len()];
    let mut sum = 0;
    let mut i = 0;
    while i < INPUT_SIZES.len() {
        arr[i] = sum;
        sum += INPUT_SIZES[i].1;
        i += 1;
    }
    arr
};

const fn GET_TOTAL_LEN() -> usize {
    let mut sum = 0;
    let mut i = 0;
    while i < INPUT_SIZES.len() {
        sum += INPUT_SIZES[i].1;
        i += 1;
    }
    sum
}

const fn GET_INDEX(input_type: InputType, offset: usize) -> usize {
    INPUT_OFFSETS[input_type as usize] + offset
}

const SMALLVEC_SIZE: usize = 4;

#[derive(Debug)]
pub struct R1CSBuilder {
    pub A: Vec<(usize, usize, i64)>,
    pub B: Vec<(usize, usize, i64)>,
    pub C: Vec<(usize, usize, i64)>,
    pub num_constraints: usize,
    pub num_variables: usize,
    pub num_inputs: usize, 
    pub num_aux: usize, 
    pub num_internal: usize, 
}

fn subtract_vectors(mut x: SmallVec<[(usize, i64); SMALLVEC_SIZE]>, y: SmallVec<[(usize, i64); SMALLVEC_SIZE]>) -> SmallVec<[(usize, i64); SMALLVEC_SIZE]> {
    for (y_idx, y_coeff) in y {
        if let Some((_, x_coeff)) = x.iter_mut().find(|(x_idx, _)| *x_idx == y_idx) {
            *x_coeff -= y_coeff;
        } else {
            x.push((y_idx, -y_coeff));
        }
    }
    x
}

// big-endian
fn combine_chunks_vec(start_idx: usize, L: usize, N: usize) -> SmallVec<[(usize, i64); SMALLVEC_SIZE]> {
    let mut result = SmallVec::with_capacity(N);
    for i in 0..N {
        result.push((start_idx + i, 1 << ((N-1-i)*L)));
    }
    result
}

fn concat_constraint_vecs(mut x: SmallVec<[(usize, i64); SMALLVEC_SIZE]>, y: SmallVec<[(usize, i64); SMALLVEC_SIZE]>) -> SmallVec<[(usize, i64); SMALLVEC_SIZE]> {
    for (y_idx, y_coeff) in y {
        if let Some((_, x_coeff)) = x.iter_mut().find(|(x_idx, _)| *x_idx == y_idx) {
            *x_coeff += y_coeff;
        } else {
            x.push((y_idx, y_coeff));
        }
    }
    x
}

fn i64_to_f<F: PrimeField>(num: i64) -> F {
    // TODO(sragss): Make from_u64
    if num < 0 {
        F::zero() - F::from_u64((-num) as u64).unwrap()
    } else {
        F::from_u64(num as u64).unwrap()
    }
}

impl R1CSBuilder {
    fn new_constraint(&mut self, a: SmallVec<[(usize, i64); SMALLVEC_SIZE]>, b: SmallVec<[(usize, i64); SMALLVEC_SIZE]>, c: SmallVec<[(usize, i64); SMALLVEC_SIZE]>) {
        let row: usize = self.num_constraints; 
        let prepend_row = |vec: SmallVec<[(usize, i64); SMALLVEC_SIZE]>, matrix: &mut Vec<(usize, usize, i64)>| {
            for (idx, val) in vec {
                    matrix.push((row, idx, val));
            }
        };
        prepend_row(a, &mut self.A); 
        prepend_row(b, &mut self.B); 
        prepend_row(c, &mut self.C); 
        self.num_constraints += 1;
    }

    fn assign_aux(&mut self) -> usize {
        let idx = self.num_variables; 
        self.num_aux += 1;
        self.num_variables += 1;
        self.num_internal += 1; 
        idx
    }   

    fn constr_abc(&mut self, a: SmallVec<[(usize, i64); SMALLVEC_SIZE]>, b: SmallVec<[(usize, i64); SMALLVEC_SIZE]>, c: SmallVec<[(usize, i64); SMALLVEC_SIZE]>) {
        self.new_constraint(a, b, c); 
    }

    // Combines the L-bit wire values of [start_idx, ..., start_idx + N - 1] into a single value 
    // and constraints it to be equal to the wire value at result_idx.
    // The combination is big-endian with the most significant bit at start_idx.
    fn combine_constraint(&mut self, start_idx: usize, L: usize, N: usize, result_idx: usize) {
        let mut constraint_A = SmallVec::with_capacity(N);
        let constraint_B = smallvec![ONE];
        let constraint_C = smallvec![(result_idx, 1)];

        for i in 0..N {
            constraint_A.push((start_idx + i, 1 << ((N-1-i)*L)));
        }

        // Here the result_idx is assumed to already be assigned

        self.new_constraint(constraint_A, constraint_B, constraint_C); 
    }

    fn combine_le(&mut self, start_idx: usize, L: usize, N: usize) -> usize {
        let result_idx = Self::assign_aux(self);

        let mut constraint_A = SmallVec::with_capacity(N);
        let constraint_B = smallvec![ONE];
        let constraint_C = smallvec![(result_idx, 1)];

        for i in 0..N {
            constraint_A.push((start_idx + i, 1 << (i*L)));
        }

        self.new_constraint(constraint_A, constraint_B, constraint_C); 
        result_idx
    }

    fn combine_be(&mut self, start_idx: usize, L: usize, N: usize) -> usize {
        let result_idx = Self::assign_aux(self);

        let mut constraint_A = SmallVec::with_capacity(N);
        let constraint_B = smallvec![ONE];
        let constraint_C = smallvec![(result_idx, 1)];

        for i in 0..N {
            constraint_A.push((start_idx + i, 1 << ((N-1-i)*L)));
        }

        self.new_constraint(constraint_A, constraint_B, constraint_C); 
        result_idx
    }

    fn multiply(&mut self, x: SmallVec<[(usize, i64); SMALLVEC_SIZE]>, y: SmallVec<[(usize, i64); SMALLVEC_SIZE]>) -> usize {
        let xy_idx = Self::assign_aux(self); 

        R1CSBuilder::constr_abc(self, 
            x, 
            y, 
            smallvec![(xy_idx, 1)], 
        ); 

        xy_idx 
    }

    fn if_else(&mut self, choice: SmallVec<[(usize, i64); SMALLVEC_SIZE]>, x: SmallVec<[(usize, i64); SMALLVEC_SIZE]>, y: SmallVec<[(usize, i64); SMALLVEC_SIZE]>) -> usize {
        let z_idx = Self::assign_aux(self);

        let y_minus_x = subtract_vectors(y, x.clone()); 
        let z_minus_x = subtract_vectors(smallvec![(z_idx, 1)], x);

        self.new_constraint(choice, y_minus_x, z_minus_x); 
        z_idx 
    }

    fn if_else_simple(&mut self, choice: usize, x: usize, y: usize) -> usize {
        Self::if_else(self, smallvec![(choice, 1)], smallvec![(x, 1)], smallvec![(y, 1)])  
    }

    // The left side is an lc but the right is a single index
    fn eq_simple(&mut self, left_idx: usize, right_idx: usize) {
        let constraint_A = smallvec![(left_idx, 1)];
        let constraint_B = smallvec![ONE];
        let constraint_C = smallvec![(right_idx, 1)];

        self.new_constraint(constraint_A, constraint_B, constraint_C); 
    }

    // The left side is an lc but the right is a single index
    fn eq(&mut self, left: SmallVec<[(usize, i64); SMALLVEC_SIZE]>, right_idx: usize, assign: bool) {
        let constraint_A = left;
        let constraint_B = smallvec![ONE];
        let constraint_C = smallvec![(right_idx, 1)];

        self.new_constraint(constraint_A, constraint_B, constraint_C); 
    }

    fn constr_prod_0(&mut self, x: SmallVec<[(usize, i64); SMALLVEC_SIZE]>, y: SmallVec<[(usize, i64); SMALLVEC_SIZE]>, z: SmallVec<[(usize, i64); SMALLVEC_SIZE]>) {
        let xy_idx = Self::assign_aux(self); 

        R1CSBuilder::constr_abc(self, 
            x, 
            y, 
            smallvec![(xy_idx, 1)], 
        ); 

        R1CSBuilder::constr_abc(self, 
            smallvec![(xy_idx, 1)], 
            z, 
            smallvec![], 
        ); 
    }

    pub fn default() -> Self {
        R1CSBuilder {
            A: Vec::with_capacity(100),
            B: Vec::with_capacity(100),
            C: Vec::with_capacity(100),
            num_constraints: 0,
            num_variables: GET_TOTAL_LEN(), // includes ("constant", 1) and ("output_state", ..)
            num_inputs: 0, // technically inputs are also aux, so keep this 0
            num_aux: GET_TOTAL_LEN()-1, // dont' include the constant  
            num_internal: 0, 
        }
    }

    pub fn get_matrices(instance: &mut R1CSBuilder) {
        // Parse the input indices 
        let op_flags_packed= GET_INDEX(InputType::ProgVRW, 0);
        let rd = GET_INDEX(InputType::ProgVRW, 1);
        let rs1 = GET_INDEX(InputType::ProgVRW, 2);
        let rs2 = GET_INDEX(InputType::ProgVRW, 3);
        let immediate = GET_INDEX(InputType::ProgVRW, 4);

        let is_load_instr: usize = GET_INDEX(InputType::OpFlags, 2);
        let is_store_instr: usize = GET_INDEX(InputType::OpFlags, 3);
        let is_jump_instr: usize = GET_INDEX(InputType::OpFlags, 4);
        let is_branch_instr: usize = GET_INDEX(InputType::OpFlags, 5);
        let if_update_rd_with_lookup_output: usize = GET_INDEX(InputType::OpFlags, 6);
        let sign_imm_flag: usize = GET_INDEX(InputType::OpFlags, 7);
        let is_concat: usize = GET_INDEX(InputType::OpFlags, 8);
        let is_add_instr: usize = GET_INDEX(
            InputType::InstrFlags, 
            RV32I::enum_index(&RV32I::ADD(ADDInstruction::default()))
        );
        let is_sub_instr: usize = GET_INDEX(
            InputType::InstrFlags, 
            RV32I::enum_index(&RV32I::SUB(SUBInstruction::default()))
        );
        let is_shift_instr = smallvec![
            (
                GET_INDEX(InputType::InstrFlags, RV32I::enum_index(&RV32I::SLL(SLLInstruction::default()))),
                1
            ),
            (
                GET_INDEX(InputType::InstrFlags, RV32I::enum_index(&RV32I::SRL(SRLInstruction::default()))),
                1
            ),
            (
                GET_INDEX(InputType::InstrFlags, RV32I::enum_index(&RV32I::SRA(SRAInstruction::default()))),
                1
            ),
        ];

        let PC = GET_INDEX(InputType::InputState, PC_IDX); 

        // Constraints: binary flags checks
        for i in 0..NUM_CIRCUIT_FLAGS {
            R1CSBuilder::constr_abc(instance, 
                smallvec![(GET_INDEX(InputType::OpFlags, i), 1)], 
                smallvec![(GET_INDEX(InputType::OpFlags, i), -1), (0, 1)], 
                smallvec![], 
            );   
        }
        for i in 0..RV32I::COUNT {
            R1CSBuilder::constr_abc(instance, 
                smallvec![(GET_INDEX(InputType::InstrFlags, i), 1)], 
                smallvec![(GET_INDEX(InputType::InstrFlags, i), -1), (0, 1)], 
                smallvec![], 
            );   
        }

        // Constraint: relation between PC and prog_a_rw
        R1CSBuilder::constr_abc(instance, 
            smallvec![(GET_INDEX(InputType::ProgARW, 0), 1), (PC, -1)], 
            smallvec![(PC, 1)], 
            smallvec![], 
        ); 

        // Combine flag_bits and check that they equal op_flags_packed. 
        R1CSBuilder::combine_constraint(instance,
            GET_INDEX(InputType::OpFlags, 0), 
            1, 
            N_FLAGS, 
            op_flags_packed
        );

        let rs1_val = GET_INDEX(InputType::MemregVReads, 0);
        let rs2_val = GET_INDEX(InputType::MemregVReads, 1);

        /*
            signal mem_v_bytes[MOPS()-3] <== subarray(1, MOPS()-3, MOPS())(memreg_v_writes);
            signal load_or_store_value <== combine_chunks_le(MOPS()-3, 8)(mem_v_bytes); 
        */
        let load_or_store_value = R1CSBuilder::combine_le(instance, GET_INDEX(InputType::MemregVWrites, 1), 8, MOPS-3); 

        /*
            signal x <== if_else()([op_flags[0], rs1_val, PC]); // TODO: change this for virtual instructions
            signal y <== if_else()([op_flags[1], rs2_val, immediate]);
         */
        // let x = R1CSBuilder::if_else_simple(instance, GET_INDEX(InputType::OpFlags, 0), rs1_val, PC);
        let x = R1CSBuilder::if_else(instance, 
            smallvec![(GET_INDEX(InputType::OpFlags, 0), 1)], 
            smallvec![(rs1_val, 1)], 
            smallvec![(PC, 4), (0, PC_START_ADDRESS as i64 -1 * PC_NOOP_SHIFT as i64)], 
        );
        let y = R1CSBuilder::if_else_simple(instance, GET_INDEX(InputType::OpFlags, 1), rs2_val, immediate);

        /* 
            signal immediate_signed <== if_else()([sign_imm_flag, immediate, -ALL_ONES() + immediate - 1]);
            (is_load_instr + is_store_instr) * ((rs1_val + immediate_signed) - (memreg_a_rw[3] + MEMORY_ADDRESS_OFFSET())) === 0;
        */
        let immediate_signed = R1CSBuilder::if_else(instance, smallvec![(sign_imm_flag, 1)], smallvec![(immediate, 1)], smallvec![(immediate, 1), (0, -ALL_ONES - 1)]);
        R1CSBuilder::constr_abc(instance, 
            smallvec![(is_load_instr, 1), (is_store_instr, 1)], 
            smallvec![(rs1_val, 1), (immediate_signed, 1), (GET_INDEX(InputType::MemregARW, 0), -1), (0, -1 * MEMORY_ADDRESS_OFFSET as i64)], 
            smallvec![]
        ); 

        /*
            for (var i=0; i<MOPS()-3; i++) {
                (memreg_v_reads[3+i] - memreg_v_writes[1+i]) * is_load_instr === 0;
            }
        */
        for i in 0..MOPS-3 {
            R1CSBuilder::constr_abc(instance, 
                smallvec![(is_load_instr, 1)], 
                smallvec![(GET_INDEX(InputType::MemregVReads, 3+i), 1), (GET_INDEX(InputType::MemregVWrites, 1+i), -1)], 
                smallvec![]
            );
        }

        // is_store_instr * (load_or_store_value - rs2_val) === 0;
        R1CSBuilder::constr_abc(instance, 
            smallvec![(is_store_instr, 1)], 
            smallvec![(load_or_store_value, 1), (GET_INDEX(InputType::LookupOutput, 0), -1)], 
            smallvec![]
        );


        /* Create the lookup query 
        - First, obtain combined_z_chunks (which should be the query)
        - Verify that the query is structured correctly, based on the instruction.
        - Then verify that the chunks of x, y, z are correct. 

        Constraints to check correctness of chunks_query 
            If NOT a concat query: chunks_query === chunks_z 
            If its a concat query: then chunks_query === zip(chunks_x, chunks_y)
        For concat, queries the tests are done later. 

            signal combined_z_chunks <== combine_chunks(C(), LOG_M())(chunks_query);
            is_add_instr * (combined_z_chunks - (x + y)) === 0; 
            is_sub_instr * (combined_z_chunks - (x + (ALL_ONES() - y + 1))) === 0; 
        */

        let combined_z_chunks = R1CSBuilder::combine_be(instance, 
            GET_INDEX(InputType::ChunksQuery, 0), 
            LOG_M,
            C, 
        );
        R1CSBuilder::constr_abc(instance,
            smallvec![(is_add_instr, 1)], 
            smallvec![(combined_z_chunks, 1), (x, -1), (y, -1)],
            smallvec![]
        ); 
        R1CSBuilder::constr_abc(instance,
            smallvec![(is_sub_instr, 1)], 
            smallvec![(combined_z_chunks, 1), (x, -1), (y, 1), (0, -1 * (ALL_ONES + 1))],
            smallvec![]
        ); 
        R1CSBuilder::constr_abc(instance,
            smallvec![(is_load_instr, 1)], 
            smallvec![(combined_z_chunks, 1), (load_or_store_value, -1)],
            smallvec![]
        ); 
        R1CSBuilder::constr_abc(instance,
            smallvec![(is_store_instr, 1)], 
            smallvec![(combined_z_chunks, 1), (rs2_val, -1)],
            smallvec![]
        ); 

        /* Verify the chunks of x and y for concat instructions. 
            signal combined_x_chunks <== combine_chunks(C(), L_CHUNK())(chunks_x);
            (combined_x_chunks - x) * is_concat === 0; 

            Note that a wire value need not be assigned for combined x_chunks. 
            Repeat for y. 
        */
        R1CSBuilder::constr_abc(
            instance,  
            concat_constraint_vecs(
                combine_chunks_vec(GET_INDEX(InputType::ChunksX, 0), L_CHUNK, C), 
                smallvec![(x, -1)]
            ), 
            smallvec![(is_concat, 1)],
            smallvec![], 
        ); 
        R1CSBuilder::constr_abc(
            instance,  
            concat_constraint_vecs(
                combine_chunks_vec(GET_INDEX(InputType::ChunksY, 0), L_CHUNK, C),
                smallvec![(y, -1)]
            ), 
            smallvec![(is_concat, 1)],
            smallvec![], 
        ); 

        /* Concat query construction: 
            signal chunk_y_used[C()]; 
            for (var i=0; i<C(); i++) {
                chunk_y_used[i] <== if_else()([is_shift, chunks_y[i], chunks_y[C()-1]]);
                (chunks_query[i] - (chunk_y_used[i] + chunks_x[i] * 2**L_CHUNK())) * is_concat === 0;
            }  
        */
        for i in 0..C {
            let chunk_y_used_i = R1CSBuilder::if_else(
                instance,
                is_shift_instr.clone(),
                smallvec![(GET_INDEX(InputType::ChunksY, i), 1)], 
                smallvec![(GET_INDEX(InputType::ChunksY, C-1), 1)]
            );
            R1CSBuilder::constr_abc(
                instance,  
                smallvec![(GET_INDEX(InputType::ChunksQuery, i), 1), (chunk_y_used_i, -1), (GET_INDEX(InputType::ChunksX, i), (1<<L_CHUNK)*-1)],
                smallvec![(is_concat, 1)],
                smallvec![], 
            ); 
        }

        // TODO: handle case when C() doesn't divide W() 
        // Maybe like this: var idx_ms_chunk = C()-1; (chunks_query[idx_ms_chunk] - (chunks_x[idx_ms_chunk] + chunks_y[idx_ms_chunk] * 2**(L_MS_CHUNK()))) * is_concat === 0;

        /* Constraints for storing value in register rd.
            // lui doesn't need a lookup and simply requires the lookup_output to be set to immediate 
            // so it can be stored in the destination register. 

            signal rd_val <== memreg_v_writes[2];
            is_load_instr * (rd_val - load_or_store_value) === 0;

            component rd_test_lookup = prodZeroTest(3);
            rd_test_lookup.in <== [rd, if_update_rd_with_lookup_output, (rd_val - lookup_output)]; 
            component rd_test_jump = prodZeroTest(3); 
            rd_test_jump.in <== [rd, is_jump_instr, (rd_val - (prog_a_rw + 4))]; 
        */
        let rd_val = GET_INDEX(InputType::MemregVWrites, 0);
        R1CSBuilder::constr_prod_0(
            instance, 
            smallvec![(rd, 1)], 
            smallvec![(if_update_rd_with_lookup_output, 1)], 
            smallvec![(rd_val, 1), (GET_INDEX(InputType::LookupOutput, 0), -1)], 
        );
        R1CSBuilder::constr_prod_0(
            instance, 
            smallvec![(rd, 1)], 
            smallvec![(is_jump_instr, 1)], 
            smallvec![(rd_val, -1), (PC, 4), (0, PC_START_ADDRESS as i64 + 4 - PC_NOOP_SHIFT as i64)], // NOTE: the PC value is shifted by +4 already after pre-pending no-op
        ); 
        
        /*  Set next PC 
            signal next_pc_j <== if_else()([
                is_jump_instr,  
                input_state[PC_IDX()] + 4, 
                lookup_output
            ]);
            signal next_pc_j_b <== if_else()([
                is_branch_instr * lookup_output, 
                next_pc_j, 
                input_state[PC_IDX()] + immediate_signed
            ]);
            output_state[PC_IDX()] <== next_pc_j_b;
        */
        let is_branch_times_lookup_output = R1CSBuilder::multiply(instance,
            smallvec![(is_branch_instr, 1)], 
            smallvec![(GET_INDEX(InputType::LookupOutput, 0), 1)]
        ); 
        let next_pc_j = R1CSBuilder::if_else(instance, 
            smallvec![(is_jump_instr, 1)], 
            smallvec![(PC, 4), (0, PC_START_ADDRESS as i64 + 4)], 
            smallvec![(GET_INDEX(InputType::LookupOutput, 0), 1), (0, 4)] // NOTE: +4 because jump instruction outputs are to the original addresses unshifted by no-ops
        );
        let next_pc_j_b = R1CSBuilder::if_else(instance, 
            smallvec![(is_branch_times_lookup_output, 1)], 
            smallvec![(next_pc_j, 1)], 
            smallvec![(PC, 4), (0, PC_START_ADDRESS as i64), (immediate_signed, 1)]
        );

        R1CSBuilder::constr_abc(instance, 
            smallvec![(next_pc_j_b, -1), (GET_INDEX(InputType::OutputState, PC_IDX), 4), (0, PC_START_ADDRESS as i64)], 
            smallvec![(GET_INDEX(InputType::OutputState, PC_IDX), 1)], 
            smallvec![], 
        );

        R1CSBuilder::move_constant_to_end(instance);
    }

    /// Returns (aux, pc_next, pc)
    pub fn calculate_aux<F: PrimeField>(inputs: R1CSStepInputs<F>, num_aux: usize) -> (Vec<F>, F) {
        let four = F::from_u64(4).unwrap();
        // Index of values within the input vector variables
        const RD: usize = 1; 
        // Within circuit_flags 
        const IMM: usize = 4; 
        const IS_JUMP: usize = 4; 
        const IS_BRANCH: usize = 5; 
        const IF_UPDATE_RD_WITH_LOOKUP_OUTPUT: usize = 6; 
        const SIGN_IMM_FLAG: usize = 7; 

        let mut aux: Vec<F> = Vec::with_capacity(num_aux);

        // 1. let load_or_store_value = R1CSBuilder::combine_le(instance, GET_INDEX(InputType::MemregVWrites, 1), 8, MOPS-3);
        aux.push({
            let mut val = F::zero(); 
            const L: usize = 8;
            const N: usize = MOPS-3;
            for i in 0..N {
                val += inputs.memreg_v_writes[1 + i] * F::from_u64(1u64<<(i*L)).unwrap();
            }
            val 
        });

        // 2. let x = R1CSBuilder::if_else_simple(instance, GET_INDEX(InputType::OpFlags, 0), rs1_val, PC);
        aux.push(
            if inputs.circuit_flags_bits[0].is_zero() {
                inputs.memreg_v_reads[0]
            } else {
                inputs.input_pc * four + F::from_u64(PC_START_ADDRESS).unwrap() - F::from_u64(PC_NOOP_SHIFT as u64).unwrap() 
            }
        );

        // 3. let _y = R1CSBuilder::if_else_simple(instance, GET_INDEX(InputType::OpFlags, 1), rs2_val, immediate);
        aux.push(
            if inputs.circuit_flags_bits[1].is_zero() {
                inputs.memreg_v_reads[1]
            } else {
                inputs.bytecode_v[IMM]
            }
        );

        // 4. let immediate_signed = R1CSBuilder::if_else(instance, smallvec![(sign_imm_flag, 1)], smallvec![(immediate, 1)], smallvec![(immediate, 1), (0, -ALL_ONES - 1)]);
        let imm_signed_index = aux.len();
        aux.push(
            if inputs.circuit_flags_bits[SIGN_IMM_FLAG].is_zero() {
                inputs.bytecode_v[IMM]
            } else {
                inputs.bytecode_v[IMM] - F::from_u64((ALL_ONES+1) as u64).unwrap()
            }
        );

        // 5. let combined_z_chunks = R1CSBuilder::combine_be(instance, GET_INDEX(InputType::ChunksQuery, 0), LOG_M, C);
        aux.push({
            let mut val = F::zero();
            const L: usize = LOG_M;
            const N: usize = C;
            for i in 0..N {
                val += inputs.chunks_query[i] * F::from_u64(1u64<<((N-1-i)*L)).unwrap();
            }
            val 
        });

        // 6-9. let chunk_y_used_i = R1CSBuilder::if_else_simple(&mut instance, is_shift, GET_INDEX(InputType::ChunksY, i), GET_INDEX(InputType::ChunksY, C-1));
        let is_shift = 
            inputs.instruction_flags_bits[RV32I::enum_index(&RV32I::SLL(SLLInstruction::default()))].is_one() 
            || inputs.instruction_flags_bits[RV32I::enum_index(&RV32I::SRL(SRLInstruction::default()))].is_one() 
            || inputs.instruction_flags_bits[RV32I::enum_index(&RV32I::SRA(SRAInstruction::default()))].is_one();
        for i in 0..C {
            aux.push(
                if is_shift {
                    inputs.chunks_y[C-1]
                } else {
                    inputs.chunks_y[i]
                }
            );
        }

        // 10. R1CSBuilder::constr_prod_0(smallvec![(rd, 1)], smallvec![(if_update_rd_with_lookup_output, 1)], smallvec![(rd_val, 1), (GET_INDEX(InputType::LookupOutput, 0), -1)], );
        aux.push(inputs.bytecode_v[RD] * inputs.circuit_flags_bits[IF_UPDATE_RD_WITH_LOOKUP_OUTPUT]); 

        // 11. constr_prod_0[is_jump_instr, rd, rd_val, prog_a_rw, 4]
        aux.push(inputs.bytecode_v[RD] * inputs.circuit_flags_bits[IS_JUMP]); 

        // 12. let is_branch_times_lookup_output = R1CSBuilder::multiply(instance, smallvec![(is_branch_instr, 1)], smallvec![(GET_INDEX(InputType::LookupOutput, 0), 1)]); 
        let is_branch_times_lookup_output = aux.len();
        aux.push(inputs.circuit_flags_bits[IS_BRANCH] * inputs.lookup_outputs[0]);

        // 13. let next_pc_j = R1CSBuilder::if_else(instance, smallvec![(is_jump_instr, 1)], smallvec![(PC, 1), (0, 4)], smallvec![(GET_INDEX(InputType::LookupOutput, 0), 1)]);
        let next_pc_j = aux.len();
        aux.push(
            if inputs.circuit_flags_bits[IS_JUMP].is_zero() {
                inputs.input_pc * four + F::from_u64(RAM_START_ADDRESS).unwrap() + F::from_u64(4).unwrap()
            } else {
                inputs.lookup_outputs[0] + four
            }
        );

        // 14. let next_pc_j_b = R1CSBuilder::if_else(instance, smallvec![(is_branch_times_lookup_output, 1)], smallvec![(next_pc_j, 1)], smallvec![(PC, 1), (immediate_signed, 1)]);
        let next_pc_j_b = aux.len();
        aux.push(
            if aux[is_branch_times_lookup_output].is_zero() {
                aux[next_pc_j]
            } else {
                inputs.input_pc * four + F::from_u64(PC_START_ADDRESS as u64).unwrap() + aux[imm_signed_index]
            }
        );

        let pc_next = aux[next_pc_j_b];
        (aux, pc_next)
    }

    fn move_constant_to_end(&mut self) {
        let modify_matrix= |mat: &mut Vec<(usize, usize, i64)>| {
            for &mut (_, ref mut value, _) in mat {
                if *value != 0 {
                    *value -= 1;
                } else {
                    *value = self.num_variables-1;
                }
            }
        };
        
        let _ = rayon::join(|| modify_matrix(&mut self.A),
        || rayon::join(|| modify_matrix(&mut self.B),
                       || modify_matrix(&mut self.C)));
    }

    #[tracing::instrument(skip_all, name = "Shape::convert_to_field")]
    pub fn convert_to_field<F: PrimeField>(&self) -> (Vec<(usize, usize, F)>, Vec<(usize, usize, F)>, Vec<(usize, usize, F)>) {
        (
            self.A.par_iter().map(|(row, idx, val)| (*row, *idx, i64_to_f::<F>(*val))).collect(),
            self.B.par_iter().map(|(row, idx, val)| (*row, *idx, i64_to_f::<F>(*val))).collect(),
            self.C.par_iter().map(|(row, idx, val)| (*row, *idx, i64_to_f::<F>(*val))).collect(),
        )
    }
}