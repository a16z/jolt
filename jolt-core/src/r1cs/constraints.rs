/// This file generates R1CS matrices and witness vectors for the Jolt circuit.
/// Its syntax is based on circom.
/// As the constraint system involved in Jolt is very simple, it's easy to generate the matrices directly
/// and avoids the need for using the circom library.
use crate::poly::field::JoltField;
use common::{constants::RAM_START_ADDRESS, rv_trace::NUM_CIRCUIT_FLAGS};
use rayon::prelude::*;
use smallvec::{smallvec, SmallVec};
use strum::EnumCount;

use crate::jolt::{
    instruction::{
        add::ADDInstruction, sll::SLLInstruction, sra::SRAInstruction, srl::SRLInstruction,
        sub::SUBInstruction, JoltInstructionSet,
    },
    vm::rv32i_vm::RV32I,
};

use super::snark::R1CSStepInputs;

/* Compiler Variables */
const C: usize = 4;
const N_FLAGS: usize = NUM_CIRCUIT_FLAGS + RV32I::COUNT;
const LOG_M: usize = 16;
const PC_START_ADDRESS: u64 = RAM_START_ADDRESS;
const MOPS: usize = 7; // "memory ops per step"
const PC_NOOP_SHIFT: usize = 4;
/* End of Compiler Variables */

const L_CHUNK: usize = LOG_M / 2;
const ALL_ONES: i64 = 0xffffffff;

const ONE: (usize, i64) = (0, 1);

const STATE_LENGTH: usize = 1;

#[derive(Debug, PartialEq, Eq, Hash)]
enum InputType {
    Constant = 0,
    InputState = 1,
    OutputState = 2,
    ProgARW = 3,
    ProgVRW = 4,
    MemregARW = 5,
    MemregVReads = 6,
    MemregVWrites = 7,
    ChunksX = 8,
    ChunksY = 9,
    ChunksQuery = 10,
    LookupOutput = 11,
    OpFlags = 12,
    InstrFlags = 13,
}

const INPUT_SIZES: &[(InputType, usize)] = &[
    (InputType::Constant, 1),
    (InputType::InputState, STATE_LENGTH),
    (InputType::OutputState, STATE_LENGTH),
    (InputType::ProgARW, 1),
    (InputType::ProgVRW, 6),
    (InputType::MemregARW, 1),
    (InputType::MemregVReads, 7),
    (InputType::MemregVWrites, 5),
    (InputType::ChunksX, C),
    (InputType::ChunksY, C),
    (InputType::ChunksQuery, C),
    (InputType::LookupOutput, 1),
    (InputType::OpFlags, NUM_CIRCUIT_FLAGS),
    (InputType::InstrFlags, RV32I::COUNT),
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
    pub num_internal: usize, // aux that isn't inputs
}

fn subtract_vectors(
    mut x: SmallVec<[(usize, i64); SMALLVEC_SIZE]>,
    y: SmallVec<[(usize, i64); SMALLVEC_SIZE]>,
) -> SmallVec<[(usize, i64); SMALLVEC_SIZE]> {
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
fn combine_chunks_vec(
    start_idx: usize,
    L: usize,
    N: usize,
) -> SmallVec<[(usize, i64); SMALLVEC_SIZE]> {
    let mut result = SmallVec::with_capacity(N);
    for i in 0..N {
        result.push((start_idx + i, 1 << ((N - 1 - i) * L)));
    }
    result
}

fn concat_constraint_vecs(
    mut x: SmallVec<[(usize, i64); SMALLVEC_SIZE]>,
    y: SmallVec<[(usize, i64); SMALLVEC_SIZE]>,
) -> SmallVec<[(usize, i64); SMALLVEC_SIZE]> {
    for (y_idx, y_coeff) in y {
        if let Some((_, x_coeff)) = x.iter_mut().find(|(x_idx, _)| *x_idx == y_idx) {
            *x_coeff += y_coeff;
        } else {
            x.push((y_idx, y_coeff));
        }
    }
    x
}

fn i64_to_f<F: JoltField>(num: i64) -> F {
    if num < 0 {
        F::zero() - F::from_u64((-num) as u64).unwrap()
    } else {
        F::from_u64(num as u64).unwrap()
    }
}

impl Default for R1CSBuilder {
    fn default() -> Self {
        R1CSBuilder {
            A: Vec::with_capacity(100),
            B: Vec::with_capacity(100),
            C: Vec::with_capacity(100),
            num_constraints: 0,
            num_variables: GET_TOTAL_LEN(), // includes ("constant", 1) and ("output_state", ..)
            num_inputs: 0,                  // technically inputs are also aux, so keep this 0
            num_aux: GET_TOTAL_LEN() - 1,   // dont' include the constant
            num_internal: 0,
        }
    }
}

impl R1CSBuilder {
    fn new_constraint(
        &mut self,
        a: SmallVec<[(usize, i64); SMALLVEC_SIZE]>,
        b: SmallVec<[(usize, i64); SMALLVEC_SIZE]>,
        c: SmallVec<[(usize, i64); SMALLVEC_SIZE]>,
    ) {
        let row: usize = self.num_constraints;
        let prepend_row = |vec: SmallVec<[(usize, i64); SMALLVEC_SIZE]>,
                           matrix: &mut Vec<(usize, usize, i64)>| {
            for (idx, val) in vec {
                matrix.push((row, idx, val));
            }
        };
        prepend_row(a, &mut self.A);
        prepend_row(b, &mut self.B);
        prepend_row(c, &mut self.C);
        self.num_constraints += 1;
    }

    // Creates an auxiliary variable by allocating a "wire", which is just an index in the witness vector.
    fn assign_aux(&mut self) -> usize {
        let idx = self.num_variables;
        self.num_aux += 1;
        self.num_variables += 1;
        self.num_internal += 1;
        idx
    }

    /******* Constraint generation functions *******/

    fn constr_abc(
        &mut self,
        a: SmallVec<[(usize, i64); SMALLVEC_SIZE]>,
        b: SmallVec<[(usize, i64); SMALLVEC_SIZE]>,
        c: SmallVec<[(usize, i64); SMALLVEC_SIZE]>,
    ) {
        self.new_constraint(a, b, c);
    }

    // Packs the L-bit wire values of [start_idx, ..., start_idx + N - 1] into a single value
    // and constraints it to be equal to the wire value at result_idx.
    // The combination is big-endian with the most significant bit at start_idx.
    fn combine_be_existing(&mut self, start_idx: usize, L: usize, N: usize, result_idx: usize) {
        let mut constraint_A = SmallVec::with_capacity(N);
        let constraint_B = smallvec![ONE];
        let constraint_C = smallvec![(result_idx, 1)];

        for i in 0..N {
            constraint_A.push((start_idx + i, 1 << ((N - 1 - i) * L)));
        }

        self.new_constraint(constraint_A, constraint_B, constraint_C);
    }

    /* Packs the L-bit wires of [start_idx, ..., start_idx + N - 1] into a single value, big-endian order,
    and creates a new wire value for the result.
     */
    fn combine_be(&mut self, start_idx: usize, L: usize, N: usize) -> usize {
        let result_idx = Self::assign_aux(self);

        let mut constraint_A = SmallVec::with_capacity(N);
        let constraint_B = smallvec![ONE];
        let constraint_C = smallvec![(result_idx, 1)];

        for i in 0..N {
            constraint_A.push((start_idx + i, 1 << ((N - 1 - i) * L)));
        }

        self.new_constraint(constraint_A, constraint_B, constraint_C);
        result_idx
    }

    // Same as above, but little-endian
    fn combine_le(&mut self, start_idx: usize, L: usize, N: usize) -> usize {
        let result_idx = Self::assign_aux(self);

        let mut constraint_A = SmallVec::with_capacity(N);
        let constraint_B = smallvec![ONE];
        let constraint_C = smallvec![(result_idx, 1)];

        for i in 0..N {
            constraint_A.push((start_idx + i, 1 << (i * L)));
        }

        self.new_constraint(constraint_A, constraint_B, constraint_C);
        result_idx
    }

    // Multiples the values denoted by lc x and lc y and assigns the result to a new wire.
    fn multiply(
        &mut self,
        x: SmallVec<[(usize, i64); SMALLVEC_SIZE]>,
        y: SmallVec<[(usize, i64); SMALLVEC_SIZE]>,
    ) -> usize {
        let xy_idx = Self::assign_aux(self);

        R1CSBuilder::constr_abc(self, x, y, smallvec![(xy_idx, 1)]);

        xy_idx
    }

    // Assigns a new wire to be lc x is choice = 0 and lc y if choice = 1
    fn if_else(
        &mut self,
        choice: SmallVec<[(usize, i64); SMALLVEC_SIZE]>,
        x: SmallVec<[(usize, i64); SMALLVEC_SIZE]>,
        y: SmallVec<[(usize, i64); SMALLVEC_SIZE]>,
    ) -> usize {
        let z_idx = Self::assign_aux(self);

        let y_minus_x = subtract_vectors(y, x.clone());
        let z_minus_x = subtract_vectors(smallvec![(z_idx, 1)], x);

        self.new_constraint(choice, y_minus_x, z_minus_x);
        z_idx
    }

    // A simpler version of if_else with indices instead of lc.
    fn if_else_simple(&mut self, choice: usize, x: usize, y: usize) -> usize {
        Self::if_else(
            self,
            smallvec![(choice, 1)],
            smallvec![(x, 1)],
            smallvec![(y, 1)],
        )
    }

    // Constrains the products of three lcs to be 0.
    fn constr_prod_0(
        &mut self,
        x: SmallVec<[(usize, i64); SMALLVEC_SIZE]>,
        y: SmallVec<[(usize, i64); SMALLVEC_SIZE]>,
        z: SmallVec<[(usize, i64); SMALLVEC_SIZE]>,
    ) {
        let xy_idx = Self::assign_aux(self);

        R1CSBuilder::constr_abc(self, x, y, smallvec![(xy_idx, 1)]);

        R1CSBuilder::constr_abc(self, smallvec![(xy_idx, 1)], z, smallvec![]);
    }

    /******* End of constraint generation functions *******/

    /* This is the main function that generates the Jolt R1CS constraint matrices.
     */
    pub fn jolt_r1cs_matrices(instance: &mut R1CSBuilder, memory_start: u64) {
        // Obtain the indices of various inputs to the circuit.
        let PC_mapped = GET_INDEX(InputType::InputState, 0);
        let op_flags_packed = GET_INDEX(InputType::ProgVRW, 1);
        let rd = GET_INDEX(InputType::ProgVRW, 2);
        let _rs1 = GET_INDEX(InputType::ProgVRW, 3);
        let _rs2 = GET_INDEX(InputType::ProgVRW, 4);
        let immediate = GET_INDEX(InputType::ProgVRW, 5);

        // Indices of flags.
        let is_load_instr: usize = GET_INDEX(InputType::OpFlags, 2);
        let is_store_instr: usize = GET_INDEX(InputType::OpFlags, 3);
        let is_jump_instr: usize = GET_INDEX(InputType::OpFlags, 4);
        let is_branch_instr: usize = GET_INDEX(InputType::OpFlags, 5);
        let if_update_rd_with_lookup_output: usize = GET_INDEX(InputType::OpFlags, 6);
        let sign_imm_flag: usize = GET_INDEX(InputType::OpFlags, 7);
        let is_concat: usize = GET_INDEX(InputType::OpFlags, 8);

        // These flags indicate the type of lookup employed and are obtained using the instruction flags.
        let is_add_instr: usize = GET_INDEX(
            InputType::InstrFlags,
            RV32I::enum_index(&RV32I::ADD(ADDInstruction::default())),
        );
        let is_sub_instr: usize = GET_INDEX(
            InputType::InstrFlags,
            RV32I::enum_index(&RV32I::SUB(SUBInstruction::default())),
        );
        let is_shift_instr = smallvec![
            (
                GET_INDEX(
                    InputType::InstrFlags,
                    RV32I::enum_index(&RV32I::SLL(SLLInstruction::default()))
                ),
                1
            ),
            (
                GET_INDEX(
                    InputType::InstrFlags,
                    RV32I::enum_index(&RV32I::SRL(SRLInstruction::default()))
                ),
                1
            ),
            (
                GET_INDEX(
                    InputType::InstrFlags,
                    RV32I::enum_index(&RV32I::SRA(SRAInstruction::default()))
                ),
                1
            ),
        ];

        // Constraints: binary checks for the input circuit and instruction flags
        for i in 0..NUM_CIRCUIT_FLAGS {
            R1CSBuilder::constr_abc(
                instance,
                smallvec![(GET_INDEX(InputType::OpFlags, i), 1)],
                smallvec![(GET_INDEX(InputType::OpFlags, i), -1), (0, 1)],
                smallvec![],
            );
        }
        for i in 0..RV32I::COUNT {
            R1CSBuilder::constr_abc(
                instance,
                smallvec![(GET_INDEX(InputType::InstrFlags, i), 1)],
                smallvec![(GET_INDEX(InputType::InstrFlags, i), -1), (0, 1)],
                smallvec![],
            );
        }

        // Constraint: ensure that the bytecode read address (prog_v_rw) is the same as the input PC.
        R1CSBuilder::constr_abc(
            instance,
            smallvec![(GET_INDEX(InputType::ProgVRW, 0), 1), (PC_mapped, -1)],
            smallvec![(PC_mapped, 1)],
            smallvec![],
        );

        // Constraint: combine flag_bits and check that they equal op_flags_packed.
        R1CSBuilder::combine_be_existing(
            instance,
            GET_INDEX(InputType::OpFlags, 0),
            1,
            N_FLAGS,
            op_flags_packed,
        );

        let rs1_val = GET_INDEX(InputType::MemregVReads, 0);
        let rs2_val = GET_INDEX(InputType::MemregVReads, 1);

        // Constraint: combine the bytes read/store to/from memory into a single W-bit value.
        let load_or_store_value = R1CSBuilder::combine_le(
            instance,
            GET_INDEX(InputType::MemregVWrites, 1),
            8,
            MOPS - 3,
        );

        /* Constraints: obtain the two operands for this instruction.
           x is either rs1_val or the PC
           y is either rs2_val or immediate
        */
        let x = R1CSBuilder::if_else(
            instance,
            smallvec![(GET_INDEX(InputType::OpFlags, 0), 1)],
            smallvec![(rs1_val, 1)],
            smallvec![
                (PC_mapped, 4),
                (0, PC_START_ADDRESS as i64 - PC_NOOP_SHIFT as i64)
            ],
        );
        let y = R1CSBuilder::if_else_simple(
            instance,
            GET_INDEX(InputType::OpFlags, 1),
            rs2_val,
            immediate,
        );

        // Constraint: compute immediate_signed which is immediate or -(ALL_ONES() + immediate - 1) depending on the sign flag.
        let immediate_signed = R1CSBuilder::if_else(
            instance,
            smallvec![(sign_imm_flag, 1)],
            smallvec![(immediate, 1)],
            smallvec![(immediate, 1), (0, -ALL_ONES - 1)],
        );

        // Constraint: memreg_a_rw[0] (the first byte involved) is rs1_val + immediate_signed
        R1CSBuilder::constr_abc(
            instance,
            smallvec![(is_load_instr, 1), (is_store_instr, 1)],
            smallvec![
                (rs1_val, 1),
                (immediate_signed, 1),
                (GET_INDEX(InputType::MemregARW, 0), -1),
                (0, -(memory_start as i64))
            ],
            smallvec![],
        );

        // Constraints: loads are reads, so the value written back is the same.
        for i in 0..MOPS - 3 {
            R1CSBuilder::constr_abc(
                instance,
                smallvec![(is_load_instr, 1)],
                smallvec![
                    (GET_INDEX(InputType::MemregVReads, 3 + i), 1),
                    (GET_INDEX(InputType::MemregVWrites, 1 + i), -1)
                ],
                smallvec![],
            );
        }

        // Constriants: for stores, the value written is the lookup output.
        R1CSBuilder::constr_abc(
            instance,
            smallvec![(is_store_instr, 1)],
            smallvec![
                (load_or_store_value, 1),
                (GET_INDEX(InputType::LookupOutput, 0), -1)
            ],
            smallvec![],
        );

        /* Create the lookup query (z = query)
        - First, obtain combined_z_chunks
        - Verify that the z is structured correctly, based on the instruction.
        - Then verify that the chunks of x, y, z are correct.

        Constraints to check correctness of chunks_z
            If NOT a concat query: chunks_query === chunks_z

            adds: query x+y
            subs: query x + (ALL_ONES() - y + 1)
            loads: query load_or_store_value
            stores: query rs2_val

            If its a concat query: then chunks_query === zip(chunks_x, chunks_y)
        */

        let combined_z_chunks =
            R1CSBuilder::combine_be(instance, GET_INDEX(InputType::ChunksQuery, 0), LOG_M, C);
        R1CSBuilder::constr_abc(
            instance,
            smallvec![(is_add_instr, 1)],
            smallvec![(combined_z_chunks, 1), (x, -1), (y, -1)],
            smallvec![],
        );
        R1CSBuilder::constr_abc(
            instance,
            smallvec![(is_sub_instr, 1)],
            smallvec![
                (combined_z_chunks, 1),
                (x, -1),
                (y, 1),
                (0, -(ALL_ONES + 1))
            ],
            smallvec![],
        );
        R1CSBuilder::constr_abc(
            instance,
            smallvec![(is_load_instr, 1)],
            smallvec![(combined_z_chunks, 1), (load_or_store_value, -1)],
            smallvec![],
        );
        R1CSBuilder::constr_abc(
            instance,
            smallvec![(is_store_instr, 1)],
            smallvec![(combined_z_chunks, 1), (rs2_val, -1)],
            smallvec![],
        );

        // Verify the chunks of x and y for concat instructions.
        R1CSBuilder::constr_abc(
            instance,
            concat_constraint_vecs(
                combine_chunks_vec(GET_INDEX(InputType::ChunksX, 0), L_CHUNK, C),
                smallvec![(x, -1)],
            ),
            smallvec![(is_concat, 1)],
            smallvec![],
        );
        R1CSBuilder::constr_abc(
            instance,
            concat_constraint_vecs(
                combine_chunks_vec(GET_INDEX(InputType::ChunksY, 0), L_CHUNK, C),
                smallvec![(y, -1)],
            ),
            smallvec![(is_concat, 1)],
            smallvec![],
        );

        /* Very query construction for concats.
            Here, chunks_query === zip(chunks_x, chunks_y)
            However, for shifts, chunks_query === zip(chunks_x, chunks_y[C-1])
        */
        for i in 0..C {
            let chunk_y_used_i = R1CSBuilder::if_else(
                instance,
                is_shift_instr.clone(),
                smallvec![(GET_INDEX(InputType::ChunksY, i), 1)],
                smallvec![(GET_INDEX(InputType::ChunksY, C - 1), 1)],
            );
            R1CSBuilder::constr_abc(
                instance,
                smallvec![
                    (GET_INDEX(InputType::ChunksQuery, i), 1),
                    (chunk_y_used_i, -1),
                    (GET_INDEX(InputType::ChunksX, i), -(1 << L_CHUNK))
                ],
                smallvec![(is_concat, 1)],
                smallvec![],
            );
        }

        // TODO(arasuarun): handle case when C() doesn't divide W()

        /* Constraints for storing value in register rd.
        - the flag, if_update_rd_with_lookup_output is used here.
        - If the instruction is a jump, then the value stored in rd is current PC + 4
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
            smallvec![
                (rd_val, -1),
                (PC_mapped, 4),
                (0, PC_START_ADDRESS as i64 + 4 - PC_NOOP_SHIFT as i64)
            ], // NOTE: the PC value is shifted by +4 already after pre-pending no-op
        );

        /*  Constraints for setting the next PC.
            - Default: increment by 4
            - Jump: set PC to lookup output
            - Branch: PC + immediate_signed if the lookup output is 1
        */
        let is_branch_times_lookup_output = R1CSBuilder::multiply(
            instance,
            smallvec![(is_branch_instr, 1)],
            smallvec![(GET_INDEX(InputType::LookupOutput, 0), 1)],
        );
        let next_pc_j = R1CSBuilder::if_else(
            instance,
            smallvec![(is_jump_instr, 1)],
            smallvec![(PC_mapped, 4), (0, PC_START_ADDRESS as i64 + 4)],
            smallvec![(GET_INDEX(InputType::LookupOutput, 0), 1), (0, 4)], // NOTE: +4 because jump instruction outputs are to the original addresses unshifted by no-ops
        );
        let next_pc_j_b = R1CSBuilder::if_else(
            instance,
            smallvec![(is_branch_times_lookup_output, 1)],
            smallvec![(next_pc_j, 1)],
            smallvec![
                (PC_mapped, 4),
                (0, PC_START_ADDRESS as i64),
                (immediate_signed, 1)
            ],
        );

        // Constraint: check the claimed output PC value, except when it is set to 0 (as is for the padded parts of the trace)
        R1CSBuilder::constr_abc(
            instance,
            smallvec![
                (next_pc_j_b, -1),
                (GET_INDEX(InputType::OutputState, 0), 4),
                (0, PC_START_ADDRESS as i64)
            ],
            smallvec![(GET_INDEX(InputType::OutputState, 0), 1)],
            smallvec![],
        );

        R1CSBuilder::move_constant_to_end(instance);
    }

    /* Given the inputs to a step of the Jolt circuit, this function returns the internal
       "auxiliary" wires values.
       The wires are built sequentially, indicating the constraint that creates it in the comments.
    */
    pub fn calculate_jolt_aux<F: JoltField>(inputs: R1CSStepInputs<F>, num_aux: usize) -> Vec<F> {
        let four = F::from_u64(4).unwrap();

        // Indices of values within their respective input vector variables.
        const RD: usize = 2;
        const IMM: usize = 5;
        const IS_JUMP: usize = 4;
        const IS_BRANCH: usize = 5;
        const IF_UPDATE_RD_WITH_LOOKUP_OUTPUT: usize = 6;
        const SIGN_IMM_FLAG: usize = 7;

        let mut aux: Vec<F> = Vec::with_capacity(num_aux);

        // 1. let load_or_store_value = R1CSBuilder::combine_le(instance, GET_INDEX(InputType::MemregVWrites, 1), 8, MOPS-3);
        aux.push({
            let mut val = F::zero();
            const L: usize = 8;
            const N: usize = MOPS - 3;
            for i in 0..N {
                val += inputs.memreg_v_writes[1 + i] * F::from_u64(1u64 << (i * L)).unwrap();
            }
            val
        });

        // 2. let x = R1CSBuilder::if_else_simple(instance, GET_INDEX(InputType::OpFlags, 0), rs1_val, PC);
        aux.push(if inputs.circuit_flags_bits[0].is_zero() {
            inputs.memreg_v_reads[0]
        } else {
            inputs.input_pc * four + F::from_u64(PC_START_ADDRESS).unwrap()
                - F::from_u64(PC_NOOP_SHIFT as u64).unwrap()
        });

        // 3. let _y = R1CSBuilder::if_else_simple(instance, GET_INDEX(InputType::OpFlags, 1), rs2_val, immediate);
        aux.push(if inputs.circuit_flags_bits[1].is_zero() {
            inputs.memreg_v_reads[1]
        } else {
            inputs.bytecode_v[IMM]
        });

        // 4. let immediate_signed = R1CSBuilder::if_else(instance, smallvec![(sign_imm_flag, 1)], smallvec![(immediate, 1)], smallvec![(immediate, 1), (0, -ALL_ONES - 1)]);
        let imm_signed_index = aux.len();
        aux.push(if inputs.circuit_flags_bits[SIGN_IMM_FLAG].is_zero() {
            inputs.bytecode_v[IMM]
        } else {
            inputs.bytecode_v[IMM] - F::from_u64((ALL_ONES + 1) as u64).unwrap()
        });

        // 5. let combined_z_chunks = R1CSBuilder::combine_be(instance, GET_INDEX(InputType::ChunksQuery, 0), LOG_M, C);
        aux.push({
            let mut val = F::zero();
            const L: usize = LOG_M;
            const N: usize = C;
            for i in 0..N {
                val += inputs.chunks_query[i] * F::from_u64(1u64 << ((N - 1 - i) * L)).unwrap();
            }
            val
        });

        // 6-9. let chunk_y_used_i = R1CSBuilder::if_else_simple(&mut instance, is_shift, GET_INDEX(InputType::ChunksY, i), GET_INDEX(InputType::ChunksY, C-1));
        let is_shift = inputs.instruction_flags_bits
            [RV32I::enum_index(&RV32I::SLL(SLLInstruction::default()))]
        .is_one()
            || inputs.instruction_flags_bits
                [RV32I::enum_index(&RV32I::SRL(SRLInstruction::default()))]
            .is_one()
            || inputs.instruction_flags_bits
                [RV32I::enum_index(&RV32I::SRA(SRAInstruction::default()))]
            .is_one();
        for i in 0..C {
            aux.push(if is_shift {
                inputs.chunks_y[C - 1]
            } else {
                inputs.chunks_y[i]
            });
        }

        // 10. R1CSBuilder::constr_prod_0(smallvec![(rd, 1)], smallvec![(if_update_rd_with_lookup_output, 1)], smallvec![(rd_val, 1), (GET_INDEX(InputType::LookupOutput, 0), -1)], );
        aux.push(
            inputs.bytecode_v[RD] * inputs.circuit_flags_bits[IF_UPDATE_RD_WITH_LOOKUP_OUTPUT],
        );

        // 11. constr_prod_0[is_jump_instr, rd, rd_val, prog_a_rw, 4]
        aux.push(inputs.bytecode_v[RD] * inputs.circuit_flags_bits[IS_JUMP]);

        // 12. let is_branch_times_lookup_output = R1CSBuilder::multiply(instance, smallvec![(is_branch_instr, 1)], smallvec![(GET_INDEX(InputType::LookupOutput, 0), 1)]);
        let is_branch_times_lookup_output = aux.len();
        aux.push(inputs.circuit_flags_bits[IS_BRANCH] * inputs.lookup_outputs[0]);

        // 13. let next_pc_j = R1CSBuilder::if_else(instance, smallvec![(is_jump_instr, 1)], smallvec![(PC_mapped, 4), (0, PC_START_ADDRESS as i64 + 4)], smallvec![(GET_INDEX(InputType::LookupOutput, 0), 1), (0, 4)] // NOTE: +4 because jump instruction outputs are to the original addresses unshifted by no-ops);
        let next_pc_j = aux.len();
        aux.push(if inputs.circuit_flags_bits[IS_JUMP].is_zero() {
            inputs.input_pc * four
                + F::from_u64(PC_START_ADDRESS).unwrap()
                + F::from_u64(4).unwrap()
        } else {
            inputs.lookup_outputs[0] + four
        });

        // 14. let next_pc_j_b = R1CSBuilder::if_else(instance, smallvec![(is_branch_times_lookup_output, 1)], smallvec![(next_pc_j, 1)], smallvec![(PC_mapped, 4), (0, PC_START_ADDRESS as i64), (immediate_signed, 1)]);
        let _next_pc_j_b = aux.len();
        aux.push(if aux[is_branch_times_lookup_output].is_zero() {
            aux[next_pc_j]
        } else {
            inputs.input_pc * four + F::from_u64(PC_START_ADDRESS).unwrap() + aux[imm_signed_index]
        });

        aux
    }

    /* In Spartan, the constant variable is assumed to be at the end of the witness vector z.
    The jolt_r1cs_matrices() function builds the r1cs matrices assuming that the constant is at the beginning.
    This function then moves the constant to the end, adjusting the matrices accordingly.
    */
    fn move_constant_to_end(&mut self) {
        let modify_matrix = |mat: &mut Vec<(usize, usize, i64)>| {
            for &mut (_, ref mut value, _) in mat {
                if *value != 0 {
                    *value -= 1;
                } else {
                    *value = self.num_variables - 1;
                }
            }
        };

        let _ = rayon::join(
            || modify_matrix(&mut self.A),
            || rayon::join(|| modify_matrix(&mut self.B), || modify_matrix(&mut self.C)),
        );
    }

    /* Converts the i64 coefficients to field elements. */
    #[allow(clippy::type_complexity)]
    #[tracing::instrument(skip_all, name = "Shape::convert_to_field")]
    pub fn convert_to_field<F: JoltField>(
        &self,
    ) -> (
        Vec<(usize, usize, F)>,
        Vec<(usize, usize, F)>,
        Vec<(usize, usize, F)>,
    ) {
        (
            self.A
                .par_iter()
                .map(|(row, idx, val)| (*row, *idx, i64_to_f::<F>(*val)))
                .collect(),
            self.B
                .par_iter()
                .map(|(row, idx, val)| (*row, *idx, i64_to_f::<F>(*val)))
                .collect(),
            self.C
                .par_iter()
                .map(|(row, idx, val)| (*row, *idx, i64_to_f::<F>(*val)))
                .collect(),
        )
    }
}
