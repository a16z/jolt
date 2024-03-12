// Handwritten circuit 

use circom_scotia::r1cs::R1CS;

/* Compiler Variables */
const C: usize = 4; 
const N_FLAGS: usize = 17; 
const W: usize = 32;
const LOG_M: usize = 16; 
const PROG_START_ADDR: usize = 2147483664;
const RAM_START_ADDRESS: usize = 0x80000000; 
const MEMORY_ADDRESS_OFFSET: usize = 0x80000000 - 0x20; 
/* End of Compiler Variables */

const L_CHUNK: usize = LOG_M/2;  
// "memreg ops per step" 
const MOPS: usize = 7;

const PC_IDX: usize = 1; 

const ONE: (usize, i64) = (0, 1);

const INPUTS: &[(&str, usize)] = &[
    ("output_state", 2),
    ("input_state", 2),
    ("prog_a_rw", 1),
    ("prog_v_rw", 6),
    ("memreg_a_rw", 7),
    ("memreg_v_reads", 7),
    ("memreg_v_writes", 7),
    ("chunks_x", C),
    ("chunks_y", C),
    ("chunks_query", C),
    ("lookup_output", 1),
    ("op_flags", N_FLAGS),
];

fn GET_INDEX(name: &str, offset: usize) -> Option<usize> {
    let mut total = 0;
    for &(input_name, value) in INPUTS {
        if input_name == name {
            return Some(total + offset);
        }
        total += value;
    }
    None
}

#[derive(Debug)]
pub struct R1CSInstance {
    A: Vec<Vec<(usize, i64)>>,
    B: Vec<Vec<(usize, i64)>>,
    C: Vec<Vec<(usize, i64)>>,
    num_constraints: usize,
    num_variables: usize,
    num_inputs: usize, 
    num_aux: usize, 
}

fn subtract_vectors(x: Vec<(usize, i64)>, mut y: Vec<(usize, i64)>) -> Vec<(usize, i64)> {
    y.extend(x.into_iter().map(|(idx, coeff)| (idx, -coeff)));
    y
}

impl R1CSInstance {

    fn assign_aux(&mut self) -> usize {
        let idx = self.num_aux; 
        self.num_aux += 1;
        idx
    }   

    fn combine_eq(&mut self, start_idx: usize, L: usize, N: usize, result_idx: usize) {
        let mut constraint_A = vec![];
        let mut constraint_B = vec![];
        let mut constraint_C = vec![];

        for i in 0..N {
            constraint_A.push((start_idx + i, 1 << ((N-1-i)*L)));
        }
        constraint_B.push(ONE); 
        constraint_C.push((result_idx, 1));

        self.A.push(constraint_A); 
        self.B.push(constraint_B);
        self.C.push(constraint_C);

        self.num_constraints += 1;
    }

    fn eq(&mut self, left_idx: usize, left_coeff: i64, left_const: i64, right_idx: usize, right_coeff: i64, right_const: i64) {
        let mut constraint_A = vec![];
        let mut constraint_B = vec![];
        let mut constraint_C = vec![];

        constraint_A.push((left_idx, left_coeff)); 
        constraint_A.push((0, left_const)); 
        constraint_B.push(ONE); 
        constraint_C.push((right_idx, right_coeff));
        constraint_A.push((0, right_const)); 

        self.A.push(constraint_A); 
        self.B.push(constraint_B);
        self.C.push(constraint_C);

        self.num_constraints += 1;
    }

    fn constr_if_else(&mut self, choice: Vec<(usize, i64)>, x: Vec<(usize, i64)>, y: Vec<(usize, i64)>) -> usize {
        let z_idx = self.num_aux; 
        self.num_aux+=1;

        let y_minus_x = subtract_vectors(y, x.clone()); 
        let z_minus_x = subtract_vectors(vec![(z_idx, 1)], x);

        self.A.push(choice); 
        self.B.push(y_minus_x); 
        self.C.push(z_minus_x); 

        self.num_constraints += 1;
        z_idx 
    }

    fn eq_simple(&mut self, left_idx: usize, right_idx: usize) {
        Self::eq(self, left_idx, 1, 0, right_idx, 1, 0);
    }

    pub fn get_matrices() -> Option<Self> {
        let mut instance = R1CSInstance {
            A: vec![],
            B: vec![],
            C: vec![],
            num_constraints: 0,
            num_variables: 1,
            num_inputs: 0,
            num_aux: 0,
        };

        let opcode = GET_INDEX("prog_v_rw", 0)?;
        let rs1 = GET_INDEX("prog_v_rw", 1)?;
        let rs2 = GET_INDEX("prog_v_rw", 2)?;
        let rd = GET_INDEX("prog_v_rw", 3)?;
        let immediate_before_processing = GET_INDEX("prog_v_rw", 4)?;
        let op_flags_packed = GET_INDEX("prog_v_rw", 5)?;
        let PC = GET_INDEX("input_state", PC_IDX)?; 

        let is_load_instr: usize = GET_INDEX("op_flags", 2)?;
        let is_store_instr: usize = GET_INDEX("op_flags", 3)?;
        let is_jump_instr: usize = GET_INDEX("op_flags", 4)?;
        let is_branch_instr: usize = GET_INDEX("op_flags", 5)?;
        let if_update_rd_with_lookup_output: usize = GET_INDEX("op_flags", 6)?;
        let is_add_instr: usize = GET_INDEX("op_flags", 7)?;
        let is_sub_instr: usize = GET_INDEX("op_flags", 8)?;
        let is_mul_instr: usize = GET_INDEX("op_flags", 9)?;
        let is_advice_instr: usize = GET_INDEX("op_flags", 10)?;
        let is_assert_false_instr: usize = GET_INDEX("op_flags", 11)?;
        let is_assert_true_instr: usize = GET_INDEX("op_flags", 12)?;
        let sign_imm_flag: usize = GET_INDEX("op_flags", 13)?;
        let is_concat: usize = GET_INDEX("op_flags", 14)?;
        let is_lui_auipc: usize = GET_INDEX("op_flags", 15)?;
        let is_shift: usize = GET_INDEX("op_flags", 16)?;

        // Constraint 1: relation between PC and prog_a_rw
        // TODO(arasuarun)

        // Constraint: combine flag_bits and check that they equal op_flags_packed 
        R1CSInstance::combine_eq(&mut instance, GET_INDEX("op_flags", 0)?, 1, N_FLAGS, op_flags_packed);

        // TODO: get immediate 

        let immediate: usize = 0; 

        // Constraint: rs1 === memreg_a_rw[0];
        R1CSInstance::eq_simple(&mut instance, rs1, GET_INDEX("memreg_a_rw", 0)?); 
        // Constraint: memreg_v_reads[0] === memreg_v_writes[0];
        R1CSInstance::eq_simple(&mut instance, GET_INDEX("memreg_v_reads", 0)?, GET_INDEX("memreg_v_writes", 0)?); 
        let rs1_val = GET_INDEX("memreg_v_reads", 0)?;

        // Constraint: rs2 === memreg_a_rw[1];
        R1CSInstance::eq_simple(&mut instance, rs2, GET_INDEX("memreg_a_rw", 1)?);
        // Constraint: memreg_v_reads[1] === memreg_v_writes[1];
        R1CSInstance::eq_simple(&mut instance, GET_INDEX("memreg_v_reads", 1)?, GET_INDEX("memreg_v_writes", 1)?);
        let rs2_val = GET_INDEX("memreg_v_reads", 1)?;

        // Constraint: rd === memreg_a_rw[2];
        R1CSInstance::eq_simple(&mut instance, rd, GET_INDEX("memreg_a_rw", 2)?);

        //  signal x <== if_else()([op_flags[0], rs1_val, PC]); // TODO: change this for virtual instructions
        let x = R1CSInstance::constr_if_else(&mut instance, vec![(GET_INDEX("op_flags", 0)?, 1)], vec![(rs1_val, 1)], vec![(PC, 1)]);
        // signal _y <== if_else()([op_flags[1], rs2_val, immediate]);
        let _y = R1CSInstance::constr_if_else(&mut instance, vec![(GET_INDEX("op_flags", 1)?, 1)], vec![(rs2_val, 1)], vec![(immediate, 1)]);
        // signal y <== if_else()([1-is_advice_instr, lookup_output, _y]);
        let y = R1CSInstance::constr_if_else(&mut instance, vec![(is_advice_instr,-1), ONE], vec![(GET_INDEX("lookup_output", 0)?, 1)], vec![(_y, 1)]); 

        let load_or_store_value = R1CSInstance::assign_aux(&mut instance); 
        // TODO:     signal load_or_store_value <== combine_chunks_le(MOPS()-3, 8)(mem_v_bytes); 

        /* 
        signal is_load_store_instr <== is_load_instr + is_store_instr;
    signal immediate_absolute <== if_else()([sign_imm_flag, immediate, ALL_ONES() - immediate + 1]);
    signal sign_of_immediate <== 1-2*sign_imm_flag;
    signal immediate_signed <== sign_of_immediate * immediate_absolute;
    signal _load_store_addr <== rs1_val + immediate_signed;
    signal load_store_addr <== is_load_store_instr * _load_store_addr;
         */


        Some(instance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_matrices() {
        let instance = R1CSInstance::get_matrices();
        println!("{:?}", instance.unwrap());

    }
}