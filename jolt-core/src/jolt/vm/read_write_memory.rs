use crate::field::JoltField;
use crate::jolt::instruction::JoltInstructionSet;
use core::iter::zip;
use itertools::Itertools;
use rand::rngs::StdRng;
use rand::RngCore;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
#[cfg(test)]
use std::collections::HashSet;
use std::marker::PhantomData;
#[cfg(test)]
use std::sync::{Arc, Mutex};

use crate::poly::commitment::commitment_scheme::{BatchType, CommitShape, CommitmentScheme};
use crate::utils::transcript::AppendToTranscript;
use crate::{
    lasso::memory_checking::{
        MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier, MultisetHashes,
        NoPreprocessing,
    },
    poly::{
        dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial, identity_poly::IdentityPolynomial,
        structured_poly::StructuredOpeningProof,
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{errors::ProofVerifyError, math::Math, mul_0_optimized, transcript::ProofTranscript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::{
    memory_address_to_witness_index, BYTES_PER_INSTRUCTION, MEMORY_OPS_PER_INSTRUCTION,
    RAM_OPS_PER_INSTRUCTION, RAM_START_ADDRESS, RAM_WORD_OPS_PER_INSTRUCTION, REGISTER_COUNT,
    REG_OPS_PER_INSTRUCTION, WORD_BYTES,
};
use common::rv_trace::{JoltDevice, MemoryLayout, MemoryOp};

use super::JoltTraceStep;
use super::{timestamp_range_check::TimestampValidityProof, JoltCommitments, JoltPolynomials};

pub fn random_memory_trace<F: JoltField>(
    memory_init: &Vec<(u64, u8)>,
    max_memory_address: usize,
    m: usize,
    rng: &mut StdRng,
) -> (
    Vec<[MemoryOp; MEMORY_OPS_PER_INSTRUCTION]>,
    [DensePolynomial<F>; 5],
) {
    let mut memory: Vec<u64> = vec![0; max_memory_address];
    for (addr, byte) in memory_init {
        let remapped_addr = addr - RAM_START_ADDRESS + REGISTER_COUNT;
        memory[remapped_addr as usize] = *byte as u64;
    }

    let m = m.next_power_of_two();
    let mut memory_trace = Vec::with_capacity(m);
    let mut load_store_flags: [Vec<u64>; 5] = std::array::from_fn(|_| Vec::with_capacity(m));

    for _ in 0..m {
        let mut ops: [MemoryOp; MEMORY_OPS_PER_INSTRUCTION] =
            std::array::from_fn(|_| MemoryOp::noop_read());

        let rs1 = rng.next_u64() % REGISTER_COUNT;
        ops[RS1] = MemoryOp::Read(rs1);

        let rs2 = rng.next_u64() % REGISTER_COUNT;
        ops[RS2] = MemoryOp::Read(rs2);

        // Don't write to the zero register
        let rd = (rng.next_u64() % (REGISTER_COUNT - 1)) + 1;
        // Registers are 32 bits
        let register_value = rng.next_u32() as u64;
        ops[RD] = MemoryOp::Write(rd, register_value);
        memory[rd as usize] = register_value;

        let ram_rng = rng.next_u32();
        if ram_rng % 3 == 0 {
            // LOAD
            let remapped_address = REGISTER_COUNT
                + (rng.next_u64() % ((max_memory_address as u64) - REGISTER_COUNT - 4));
            let ram_address = remapped_address - REGISTER_COUNT + RAM_START_ADDRESS;

            let load_rng = rng.next_u32();
            if load_rng % 3 == 0 {
                // LB
                ops[3] = MemoryOp::Read(ram_address);
                for i in 1..4 {
                    ops[i + 3] = MemoryOp::noop_read();
                }
                for (i, flag) in load_store_flags.iter_mut().enumerate() {
                    flag.push(if i == 0 { 1 } else { 0 });
                }
            } else if load_rng % 3 == 1 {
                // LH
                for i in 0..2 {
                    ops[i + 3] = MemoryOp::Read(ram_address + (i as u64));
                }
                for i in 2..4 {
                    ops[i + 3] = MemoryOp::noop_read();
                }
                for (i, flag) in load_store_flags.iter_mut().enumerate() {
                    flag.push(if i == 1 { 1 } else { 0 });
                }
            } else {
                // LW
                for i in 0..4 {
                    ops[i + 3] = MemoryOp::Read(ram_address + (i as u64));
                }
                for (i, flag) in load_store_flags.iter_mut().enumerate() {
                    flag.push(if i == 4 { 1 } else { 0 });
                }
            }
        } else if ram_rng % 3 == 1 {
            // STORE
            let remapped_address = REGISTER_COUNT
                + (rng.next_u64() % ((max_memory_address as u64) - REGISTER_COUNT - 4));
            let ram_address = remapped_address - REGISTER_COUNT + RAM_START_ADDRESS;
            let store_rng = rng.next_u32();
            if store_rng % 3 == 0 {
                // SB
                // RAM is byte-addressable, so values are a single byte
                let ram_value = rng.next_u64() & 0xff;
                ops[3] = MemoryOp::Write(ram_address, ram_value);
                memory[remapped_address as usize] = ram_value;
                for i in 1..4 {
                    ops[i + 3] = MemoryOp::noop_read();
                }
                for (i, flag) in load_store_flags.iter_mut().enumerate() {
                    flag.push(if i == 2 { 1 } else { 0 });
                }
            } else if store_rng % 3 == 1 {
                // SH
                for i in 0..2 {
                    // RAM is byte-addressable, so values are a single byte
                    let ram_value = rng.next_u64() & 0xff;
                    ops[i + 3] = MemoryOp::Write(ram_address + (i as u64), ram_value);
                    memory[i + (remapped_address as usize)] = ram_value;
                }
                for i in 2..4 {
                    ops[i + 3] = MemoryOp::noop_read();
                }
                for (i, flag) in load_store_flags.iter_mut().enumerate() {
                    flag.push(if i == 3 { 1 } else { 0 });
                }
            } else {
                // SW
                for i in 0..4 {
                    // RAM is byte-addressable, so values are a single byte
                    let ram_value = rng.next_u64() & 0xff;
                    ops[i + 3] = MemoryOp::Write(ram_address + (i as u64), ram_value);
                    memory[i + (remapped_address as usize)] = ram_value;
                }
                for (i, flag) in load_store_flags.iter_mut().enumerate() {
                    flag.push(if i == 4 { 1 } else { 0 });
                }
            }
        } else {
            for i in 0..4 {
                ops[i + 3] = MemoryOp::noop_read();
            }
            for flag in load_store_flags.iter_mut() {
                flag.push(0);
            }
        }

        memory_trace.push(ops);
    }

    (
        memory_trace,
        load_store_flags
            .iter()
            .map(|bitvector| DensePolynomial::from_u64(bitvector))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap(),
    )
}

#[derive(Clone)]
pub struct ReadWriteMemoryPreprocessing {
    min_bytecode_address: u64,
    pub bytecode_bytes: Vec<u8>,
    // HACK: The verifier will populate this field by copying it
    // over from the `ReadWriteMemoryProof`. Having `program_io` in
    // this preprocessing struct allows the verifier to access it
    // to compute the v_init and v_final openings, with no impact
    // on existing function signatures.
    pub program_io: Option<JoltDevice>,
}

impl ReadWriteMemoryPreprocessing {
    #[tracing::instrument(skip_all, name = "ReadWriteMemoryPreprocessing::preprocess")]
    pub fn preprocess(memory_init: Vec<(u64, u8)>) -> Self {
        let min_bytecode_address = memory_init
            .iter()
            .map(|(address, _)| *address)
            .min()
            .unwrap_or(0);

        let max_bytecode_address = memory_init
            .iter()
            .map(|(address, _)| *address)
            .max()
            .unwrap_or(0)
            + ((BYTES_PER_INSTRUCTION as u64) - 1); // For RV32I, instructions occupy 4 bytes, so the max bytecode address is the max instruction address + 3

        let mut bytecode_bytes =
            vec![0u8; (max_bytecode_address - min_bytecode_address + 1) as usize];
        for (address, byte) in memory_init.iter() {
            let remapped_index = (address - min_bytecode_address) as usize;
            bytecode_bytes[remapped_index] = *byte;
        }

        Self {
            min_bytecode_address,
            bytecode_bytes,
            program_io: None,
        }
    }
}

fn remap_address(a: u64, memory_layout: &MemoryLayout) -> u64 {
    if a >= memory_layout.input_start {
        memory_address_to_witness_index(a, memory_layout.ram_witness_offset) as u64
    } else if a < REGISTER_COUNT {
        // If a < REGISTER_COUNT, it is one of the registers and doesn't
        // need to be remapped
        a
    } else {
        panic!("Unexpected address {}", a)
    }
}

fn remap_address_index(remapped_a: u64) -> usize {
    (remapped_a - REGISTER_COUNT) as usize
}

const RS1: usize = 0;
const RS2: usize = 1;
const RD: usize = 2;
const RAM_1: usize = 3;
const RAM_2: usize = 4;
const RAM_3: usize = 5;
const RAM_4: usize = 6;
const RAM_1_INDEX: usize = RAM_1 - 3;
const RAM_2_INDEX: usize = RAM_2 - 3;
const RAM_3_INDEX: usize = RAM_3 - 3;
const RAM_4_INDEX: usize = RAM_4 - 3;

pub struct ReadWriteMemory<F, C>
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
{
    _group: PhantomData<C>,
    /// Size of entire address space (i.e. registers + IO + RAM)
    memory_size: usize,
    /// MLE of initial memory values. RAM is initialized to contain the program bytecode and inputs.
    pub v_init_reg: DensePolynomial<F>,
    pub v_init_ram: [DensePolynomial<F>; WORD_BYTES],

    /// MLE of read/write addresses. For offline memory checking, each read is paired with a "virtual" write
    /// and vice versa, so the read addresses and write addresses are the same.
    pub a_ram: DensePolynomial<F>,
    /// MLE of the read values.
    pub v_read_reg: [DensePolynomial<F>; REG_OPS_PER_INSTRUCTION],
    pub v_read_ram: [DensePolynomial<F>; WORD_BYTES],
    /// MLE of the write values.
    pub v_write_rd: DensePolynomial<F>,
    pub v_write_ram: [DensePolynomial<F>; WORD_BYTES],
    /// MLE of the final memory state.
    pub v_final_reg: DensePolynomial<F>,
    pub v_final_ram: [DensePolynomial<F>; WORD_BYTES],

    /// MLE of the read timestamps.
    pub t_read_reg: [DensePolynomial<F>; REG_OPS_PER_INSTRUCTION],
    pub t_read_ram: DensePolynomial<F>,
    /// MLE of the write timestamps.
    pub t_write_ram: DensePolynomial<F>,
    /// MLE of the final timestamps.
    pub t_final_reg: DensePolynomial<F>,
    pub t_final_ram: DensePolynomial<F>,
    pub remainder: DensePolynomial<F>,
}

fn merge_vec_array(
    mut reg_arr: [Vec<u64>; REG_OPS_PER_INSTRUCTION],
    mut ram_arr: [Vec<u64>; RAM_OPS_PER_INSTRUCTION],
    memory_trace_len: usize,
) -> [Vec<u64>; MEMORY_OPS_PER_INSTRUCTION] {
    let mut merged_arr: [Vec<u64>; MEMORY_OPS_PER_INSTRUCTION] =
        std::array::from_fn(|_| Vec::with_capacity(memory_trace_len));

    merged_arr.iter_mut().enumerate().for_each(|(i, v)| {
        if i < REG_OPS_PER_INSTRUCTION {
            *v = std::mem::take(&mut reg_arr[i]);
        } else {
            *v = std::mem::take(&mut ram_arr[i - REG_OPS_PER_INSTRUCTION]);
        }
    });

    merged_arr
}

fn map_to_polys<F: JoltField, const N: usize>(vals: &[Vec<u64>; N]) -> [DensePolynomial<F>; N] {
    vals.par_iter()
        .map(|vals| DensePolynomial::from_u64(vals))
        .collect::<Vec<DensePolynomial<F>>>()
        .try_into()
        .unwrap()
}

#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    pub bytes: [u64; WORD_BYTES],
}

impl Word {
    fn new(bytes: [u64; WORD_BYTES]) -> Self {
        Word { bytes }
    }

    fn get_bytes_array_from_word_vec(word_vector: &Vec<Word>) -> [Vec<u64>; WORD_BYTES] {
        let input_vector_word: Vec<Vec<u64>> = (0..WORD_BYTES)
            .map(|i| word_vector.iter().map(|row| row.bytes[i]).collect())
            .collect();

        let mut output_array: [Vec<u64>; WORD_BYTES] = Default::default();

        for (i, row) in input_vector_word.iter().enumerate() {
            output_array[i] = row.clone();
        }

        output_array
    }

    fn get_bytes_array_from_u64_vec(input_vector: Vec<u64>) -> [Vec<u64>; WORD_BYTES] {
        let input_vector: Vec<Vec<u64>> = input_vector
            .chunks(WORD_BYTES)
            .map(|chunk| chunk.to_vec())
            .collect();

        // now each row in input_vector will be a vector of bytes corresponding to the particular byte index in the word
        // for example, the first row will be a vector of first bytes from all the words in the memory
        let input_vector_word: Vec<Vec<u64>> = (0..WORD_BYTES)
            .map(|i| input_vector.iter().map(|row| row[i]).collect())
            .collect();

        let mut output_array: [Vec<u64>; WORD_BYTES] = Default::default();

        for (i, row) in input_vector_word.iter().enumerate() {
            output_array[i] = row.clone();
        }

        output_array
    }
}

fn pad_zeros(vector: &mut Vec<u64>) {
    let next_power_of_two_of_length = vector.len().next_power_of_two();
    for _ in vector.len()..next_power_of_two_of_length {
        vector.push(0);
    }
}

impl<F: JoltField, C: CommitmentScheme<Field = F>> ReadWriteMemory<F, C> {
    #[tracing::instrument(skip_all, name = "ReadWriteMemory::new")]
    pub fn new<InstructionSet: JoltInstructionSet>(
        program_io: &JoltDevice,
        load_store_flags: &[DensePolynomial<F>],
        preprocessing: &ReadWriteMemoryPreprocessing,
        trace: &Vec<JoltTraceStep<InstructionSet>>,
    ) -> (Self, [Vec<u64>; REG_OPS_PER_INSTRUCTION], Vec<u64>) {
        assert!(program_io.inputs.len() <= (program_io.memory_layout.max_input_size as usize));
        assert!(program_io.outputs.len() <= (program_io.memory_layout.max_output_size as usize));

        let m = trace.len();
        assert!(m.is_power_of_two());

        let max_trace_address = trace
            .iter()
            .flat_map(|step| {
                step.memory_ops.iter().map(|op| match op {
                    MemoryOp::Read(a) => remap_address(*a, &program_io.memory_layout),
                    MemoryOp::Write(a, _) => remap_address(*a, &program_io.memory_layout),
                })
            })
            .max()
            .unwrap_or(0);

        let memory_size = (program_io.memory_layout.ram_witness_offset + max_trace_address)
            .next_power_of_two() as usize;
        let mut v_init: Vec<u64> = vec![0; memory_size];
        // Copy bytecode
        let mut v_init_index = memory_address_to_witness_index(
            preprocessing.min_bytecode_address,
            program_io.memory_layout.ram_witness_offset,
        );
        for byte in preprocessing.bytecode_bytes.iter() {
            v_init[v_init_index] = *byte as u64;
            v_init_index += 1;
        }
        // Copy input bytes
        v_init_index = memory_address_to_witness_index(
            program_io.memory_layout.input_start,
            program_io.memory_layout.ram_witness_offset,
        );
        for byte in program_io.inputs.iter() {
            v_init[v_init_index] = *byte as u64;
            v_init_index += 1;
        }

        #[cfg(test)]
        let mut init_tuples: HashSet<(u64, u64, u64)> = HashSet::new();
        #[cfg(test)]
        {
            for (a, v) in v_init.iter().enumerate() {
                init_tuples.insert((a as u64, *v, 0u64));
            }
        }
        #[cfg(test)]
        let read_tuples: Arc<Mutex<HashSet<(u64, u64, u64)>>> =
            Arc::new(Mutex::new(HashSet::new()));
        #[cfg(test)]
        let write_tuples: Arc<Mutex<HashSet<(u64, u64, u64)>>> =
            Arc::new(Mutex::new(HashSet::new()));

        let (memory_trace_reg, memory_trace_ram): (Vec<Vec<MemoryOp>>, Vec<Vec<MemoryOp>>) = trace
            .into_par_iter()
            .map(|step| {
                let (reg, ram) = step.memory_ops.split_at(3);
                (reg.to_vec(), ram.to_vec())
            })
            .unzip();

        let remainder_vec: Vec<u64> = trace.into_par_iter().map(|step| step.remainder).collect();

        let reg_count = REGISTER_COUNT as usize;

        // split v_init into v_init_reg and v_init_ram
        let v_init_reg = v_init[..reg_count].to_vec();
        let mut v_init_ram = v_init[reg_count..].to_vec();
        pad_zeros(&mut v_init_ram);

        // split v_final into v_final_reg and v_final_ram
        let mut v_final_reg = v_init[..reg_count].to_vec();
        let mut v_final_ram = v_init[reg_count..].to_vec();
        pad_zeros(&mut v_final_ram);

        let mut t_final_reg = vec![0; reg_count];
        let mut t_final_ram = vec![0; (memory_size - reg_count).next_power_of_two() / 4];

        let mut v_read_reg: [Vec<u64>; REG_OPS_PER_INSTRUCTION] =
            std::array::from_fn(|_| Vec::with_capacity(m));
        // change for word-addressable
        let mut v_read_ram: Vec<Word> = Vec::with_capacity(m);

        let mut t_read_reg: [Vec<u64>; REG_OPS_PER_INSTRUCTION] =
            std::array::from_fn(|_| Vec::with_capacity(m));
        let mut t_read_ram: Vec<u64> = Vec::with_capacity(m);

        // REG only
        let mut v_write_rd: Vec<u64> = Vec::with_capacity(m);
        // RAM only
        let mut a_ram: Vec<u64> = Vec::with_capacity(m);
        let mut v_write_ram: Vec<Word> = Vec::with_capacity(m);

        let mut t_write_ram: Vec<u64> = Vec::with_capacity(m);

        #[cfg(test)]
        let r_tuples_ram = read_tuples.clone();
        #[cfg(test)]
        let w_tuples_ram = write_tuples.clone();
        #[cfg(test)]
        let r_tuples_reg = read_tuples.clone();
        #[cfg(test)]
        let w_tuples_reg = write_tuples.clone();

        let span = tracing::span!(tracing::Level::DEBUG, "memory_trace_processing");
        let _enter = span.enter();

        let result = rayon::join(
            move || {
                let span = tracing::span!(tracing::Level::DEBUG, "ram_trace_processing");
                let _enter = span.enter();

                let lbu_flag = &load_store_flags[0];
                let lhu_flag = &load_store_flags[1];
                let lw_flag = &load_store_flags[2];
                let lb_flag = &load_store_flags[3];
                let lh_flag = &load_store_flags[4];
                let sb_flag = &load_store_flags[5];
                let sh_flag = &load_store_flags[6];
                let sw_flag = &load_store_flags[7];

                for (i, step) in memory_trace_ram.iter().enumerate() {
                    let timestamp = i as u64;

                    #[allow(unused_assignments)]
                    let mut ram_word_address = 0;
                    let mut is_v_write_ram = false;

                    // If it is a load flag
                    if lb_flag[i].is_one()
                        || lh_flag[i].is_one()
                        || lbu_flag[i].is_one()
                        || lhu_flag[i].is_one()
                        || lw_flag[i].is_one()
                    {
                        match step[..] {
                            [MemoryOp::Read(a0), MemoryOp::Read(_a1), MemoryOp::Read(_a2), MemoryOp::Read(_a3)] =>
                            {
                                let remapped_a = remap_address(a0, &program_io.memory_layout);
                                let remapped_a_index = if remapped_a == 0 {
                                    0
                                } else {
                                    remap_address_index(remapped_a)
                                };
                                a_ram.push((remapped_a_index / WORD_BYTES).try_into().unwrap());

                                // construct word using v_final_ram and remapped index
                                let v_word = Word::new(
                                    v_final_ram[remapped_a_index..remapped_a_index + WORD_BYTES]
                                        .try_into()
                                        .unwrap(),
                                );

                                // push the word into both v_read_ram and v_write_ram as it is a load instruction
                                v_read_ram.push(v_word.clone());
                                v_write_ram.push(v_word);

                                t_read_ram.push(t_final_ram[remapped_a_index / WORD_BYTES]);
                                t_write_ram.push(timestamp + 1);

                                // reset t_final_ram to timestamp + 1
                                t_final_ram[remapped_a_index / WORD_BYTES] = timestamp + 1;
                            }
                            _ => {
                                panic!("Invalid memory ops for a load instruction");
                            }
                        }
                    } else if sw_flag[i].is_one() {
                        match step[..] {
                            [MemoryOp::Write(a0, v0_new), MemoryOp::Write(_a1, v1_new), MemoryOp::Write(_a2, v2_new), MemoryOp::Write(_a3, v3_new)] =>
                            {
                                let remapped_a = remap_address(a0, &program_io.memory_layout);
                                let remapped_a_index = if remapped_a == 0 {
                                    0
                                } else {
                                    remap_address_index(remapped_a)
                                };
                                a_ram.push((remapped_a_index / WORD_BYTES).try_into().unwrap());

                                // construct old_word using v_final_ram and push it into v_read_ram
                                let v_old_word = Word::new(
                                    v_final_ram[remapped_a_index..remapped_a_index + WORD_BYTES]
                                        .try_into()
                                        .unwrap(),
                                );
                                v_read_ram.push(v_old_word);

                                // construct new_word using v_new
                                let v_new_word = Word::new([v0_new, v1_new, v2_new, v3_new]);
                                v_write_ram.push(v_new_word.clone());

                                t_read_ram.push(t_final_ram[remapped_a_index / WORD_BYTES]);
                                t_write_ram.push(timestamp + 1);

                                // change v_final_ram and t_final_ram
                                for index in 0..WORD_BYTES {
                                    v_final_ram[remapped_a_index + index] = v_new_word.bytes[index];
                                }
                                t_final_ram[remapped_a_index / WORD_BYTES] = timestamp + 1;
                            }
                            _ => {
                                panic!("Invalid memory ops for a store word instruction");
                            }
                        }
                    } else if sh_flag[i].is_one() {
                        match step[..] {
                            [MemoryOp::Write(a0, v0_new), MemoryOp::Write(_a1, v1_new), MemoryOp::Read(_a2), MemoryOp::Read(_a3)] =>
                            {
                                let remapped_a = remap_address(a0, &program_io.memory_layout);
                                let remapped_a_index = if remapped_a == 0 {
                                    0
                                } else {
                                    remap_address_index(remapped_a)
                                };
                                a_ram.push((remapped_a_index / WORD_BYTES).try_into().unwrap());

                                // construct old_word using v_final_ram and push it into v_read_ram
                                let v_old_word = Word::new(
                                    v_final_ram[remapped_a_index..remapped_a_index + WORD_BYTES]
                                        .try_into()
                                        .unwrap(),
                                );
                                v_read_ram.push(v_old_word.clone());

                                // construct new_word using v_new
                                let v_new_word = Word::new([
                                    v0_new,
                                    v1_new,
                                    v_old_word.bytes[2],
                                    v_old_word.bytes[3],
                                ]);
                                v_write_ram.push(v_new_word.clone());

                                t_read_ram.push(t_final_ram[remapped_a_index / WORD_BYTES]);
                                t_write_ram.push(timestamp + 1);

                                // change v_final_ram and t_final_ram
                                for index in 0..WORD_BYTES {
                                    v_final_ram[remapped_a_index + index] = v_new_word.bytes[index];
                                }
                                t_final_ram[remapped_a_index / WORD_BYTES] = timestamp + 1;
                            }
                            [MemoryOp::Read(a0), MemoryOp::Read(_a1), MemoryOp::Write(_a2, v2_new), MemoryOp::Write(_a3, v3_new)] =>
                            {
                                let remapped_a = remap_address(a0, &program_io.memory_layout);
                                let remapped_a_index = if remapped_a == 0 {
                                    0
                                } else {
                                    remap_address_index(remapped_a)
                                };
                                a_ram.push((remapped_a_index / WORD_BYTES).try_into().unwrap());

                                let v_old_word = Word::new(
                                    v_final_ram[remapped_a_index..remapped_a_index + WORD_BYTES]
                                        .try_into()
                                        .unwrap(),
                                );
                                v_read_ram.push(v_old_word.clone());

                                let v_new_word = Word::new([
                                    v_old_word.bytes[0],
                                    v_old_word.bytes[1],
                                    v2_new,
                                    v3_new,
                                ]);
                                v_write_ram.push(v_new_word.clone());

                                t_read_ram.push(t_final_ram[remapped_a_index / WORD_BYTES]);
                                t_write_ram.push(timestamp + 1);

                                for index in 0..WORD_BYTES {
                                    v_final_ram[remapped_a_index + index] = v_new_word.bytes[index];
                                }
                                t_final_ram[remapped_a_index / WORD_BYTES] = timestamp + 1;
                            }
                            _ => {
                                panic!("Invalid memory ops for a store half instruction");
                                // Change later
                            }
                        }
                    } else if sb_flag[i].is_one() {
                        match step[..] {
                            [MemoryOp::Write(a0, v0_new), MemoryOp::Read(_a1), MemoryOp::Read(_a2), MemoryOp::Read(_a3)] =>
                            {
                                let remapped_a = remap_address(a0, &program_io.memory_layout);
                                let remapped_a_index = if remapped_a == 0 {
                                    0
                                } else {
                                    remap_address_index(remapped_a)
                                };
                                a_ram.push((remapped_a_index / WORD_BYTES).try_into().unwrap());

                                let v_old_word = Word::new(
                                    v_final_ram[remapped_a_index..remapped_a_index + WORD_BYTES]
                                        .try_into()
                                        .unwrap(),
                                );
                                v_read_ram.push(v_old_word.clone());

                                let v_new_word = Word::new([
                                    v0_new,
                                    v_old_word.bytes[1],
                                    v_old_word.bytes[2],
                                    v_old_word.bytes[3],
                                ]);
                                v_write_ram.push(v_new_word.clone());

                                t_read_ram.push(t_final_ram[remapped_a_index / WORD_BYTES]);
                                t_write_ram.push(timestamp + 1);

                                for index in 0..WORD_BYTES {
                                    v_final_ram[remapped_a_index + index] = v_new_word.bytes[index];
                                }
                                t_final_ram[remapped_a_index / WORD_BYTES] = timestamp + 1;
                            }
                            [MemoryOp::Read(a0), MemoryOp::Write(_a1, v1_new), MemoryOp::Read(_a2), MemoryOp::Read(_a3)] =>
                            {
                                let remapped_a = remap_address(a0, &program_io.memory_layout);
                                let remapped_a_index = if remapped_a == 0 {
                                    0
                                } else {
                                    remap_address_index(remapped_a)
                                };
                                a_ram.push((remapped_a_index / WORD_BYTES).try_into().unwrap());

                                let v_old_word = Word::new(
                                    v_final_ram[remapped_a_index..remapped_a_index + WORD_BYTES]
                                        .try_into()
                                        .unwrap(),
                                );
                                v_read_ram.push(v_old_word.clone());

                                let v_new_word = Word::new([
                                    v_old_word.bytes[0],
                                    v1_new,
                                    v_old_word.bytes[2],
                                    v_old_word.bytes[3],
                                ]);
                                v_write_ram.push(v_new_word.clone());

                                t_read_ram.push(t_final_ram[remapped_a_index / WORD_BYTES]);
                                t_write_ram.push(timestamp + 1);

                                for index in 0..WORD_BYTES {
                                    v_final_ram[remapped_a_index + index] = v_new_word.bytes[index];
                                }
                                t_final_ram[remapped_a_index / WORD_BYTES] = timestamp + 1;
                            }
                            [MemoryOp::Read(a0), MemoryOp::Read(_a1), MemoryOp::Write(_a2, v2_new), MemoryOp::Read(_a3)] =>
                            {
                                let remapped_a = remap_address(a0, &program_io.memory_layout);
                                let remapped_a_index = if remapped_a == 0 {
                                    0
                                } else {
                                    remap_address_index(remapped_a)
                                };
                                a_ram.push((remapped_a_index / WORD_BYTES).try_into().unwrap());

                                let v_old_word = Word::new(
                                    v_final_ram[remapped_a_index..remapped_a_index + WORD_BYTES]
                                        .try_into()
                                        .unwrap(),
                                );
                                v_read_ram.push(v_old_word.clone());

                                let v_new_word = Word::new([
                                    v_old_word.bytes[0],
                                    v_old_word.bytes[1],
                                    v2_new,
                                    v_old_word.bytes[3],
                                ]);
                                v_write_ram.push(v_new_word.clone());

                                t_read_ram.push(t_final_ram[remapped_a_index / WORD_BYTES]);
                                t_write_ram.push(timestamp + 1);

                                for index in 0..WORD_BYTES {
                                    v_final_ram[remapped_a_index + index] = v_new_word.bytes[index];
                                }
                                t_final_ram[remapped_a_index / WORD_BYTES] = timestamp + 1;
                            }
                            [MemoryOp::Read(a0), MemoryOp::Read(_a1), MemoryOp::Read(_a2), MemoryOp::Write(_a3, v3_new)] =>
                            {
                                let remapped_a = remap_address(a0, &program_io.memory_layout);
                                let remapped_a_index = if remapped_a == 0 {
                                    0
                                } else {
                                    remap_address_index(remapped_a)
                                };
                                a_ram.push((remapped_a_index / WORD_BYTES).try_into().unwrap());

                                let v_old_word = Word::new(
                                    v_final_ram[remapped_a_index..remapped_a_index + WORD_BYTES]
                                        .try_into()
                                        .unwrap(),
                                );
                                v_read_ram.push(v_old_word.clone());

                                let v_new_word = Word::new([
                                    v_old_word.bytes[0],
                                    v_old_word.bytes[1],
                                    v_old_word.bytes[2],
                                    v3_new,
                                ]);
                                v_write_ram.push(v_new_word.clone());

                                t_read_ram.push(t_final_ram[remapped_a_index / WORD_BYTES]);
                                t_write_ram.push(timestamp + 1);

                                for index in 0..WORD_BYTES {
                                    v_final_ram[remapped_a_index + index] = v_new_word.bytes[index];
                                }
                                t_final_ram[remapped_a_index / WORD_BYTES] = timestamp + 1;
                            }
                            _ => {
                                panic!("Invalid memory ops for a store byte instruction");
                            }
                        }
                    } else {
                        let remapped_a_index = 0;
                        a_ram.push((remapped_a_index / WORD_BYTES).try_into().unwrap());

                        // construct word using v_final_ram and remapped index
                        let v_word = Word::new(
                            v_final_ram[remapped_a_index..remapped_a_index + WORD_BYTES]
                                .try_into()
                                .unwrap(),
                        );

                        // push the word into both v_read_ram and v_write_ram as it is a load instruction
                        v_read_ram.push(v_word.clone());
                        v_write_ram.push(v_word);

                        t_read_ram.push(t_final_ram[remapped_a_index / WORD_BYTES]);
                        t_write_ram.push(timestamp + 1);

                        // reset t_final_ram to timestamp + 1
                        t_final_ram[remapped_a_index / WORD_BYTES] = timestamp + 1;
                    }
                }

                drop(_enter);
                drop(span);

                (
                    v_final_ram,
                    t_final_ram,
                    v_read_ram,
                    t_read_ram,
                    v_write_ram,
                    t_write_ram,
                    a_ram,
                )
            },
            move || {
                let span = tracing::span!(tracing::Level::DEBUG, "register_trace_processing");
                let _enter = span.enter();

                for (i, step) in memory_trace_reg.iter().enumerate() {
                    let timestamp = i as u64;

                    match step[RS1] {
                        MemoryOp::Read(a) => {
                            assert!(a < REGISTER_COUNT);
                            let v = v_final_reg[a as usize];

                            #[cfg(test)]
                            {
                                r_tuples_reg.lock().unwrap().insert((
                                    a,
                                    v,
                                    t_final_reg[a as usize],
                                ));
                                w_tuples_reg.lock().unwrap().insert((a, v, timestamp));
                            }

                            v_read_reg[RS1].push(v);
                            t_read_reg[RS1].push(t_final_reg[a as usize]);
                            t_final_reg[a as usize] = timestamp;
                        }
                        MemoryOp::Write(a, v) => {
                            panic!("Unexpected rs1 MemoryOp::Write({}, {})", a, v);
                        }
                    }

                    match step[RS2] {
                        MemoryOp::Read(a) => {
                            assert!(a < REGISTER_COUNT);
                            let v = v_final_reg[a as usize];

                            #[cfg(test)]
                            {
                                r_tuples_reg.lock().unwrap().insert((
                                    a,
                                    v,
                                    t_final_reg[a as usize],
                                ));
                                w_tuples_reg.lock().unwrap().insert((a, v, timestamp));
                            }

                            v_read_reg[RS2].push(v);
                            t_read_reg[RS2].push(t_final_reg[a as usize]);
                            t_final_reg[a as usize] = timestamp;
                        }
                        MemoryOp::Write(a, v) => {
                            panic!("Unexpected rs2 MemoryOp::Write({}, {})", a, v);
                        }
                    }

                    match step[RD] {
                        MemoryOp::Read(a) => {
                            panic!("Unexpected rd MemoryOp::Read({})", a);
                        }
                        MemoryOp::Write(a, v_new) => {
                            assert!(a < REGISTER_COUNT);
                            let v_old = v_final_reg[a as usize];

                            #[cfg(test)]
                            {
                                r_tuples_reg.lock().unwrap().insert((
                                    a,
                                    v_old,
                                    t_final_reg[a as usize],
                                ));
                                w_tuples_reg
                                    .lock()
                                    .unwrap()
                                    .insert((a, v_new, timestamp + 1));
                            }

                            v_read_reg[RD].push(v_old);
                            t_read_reg[RD].push(t_final_reg[a as usize]);
                            v_write_rd.push(v_new);
                            v_final_reg[a as usize] = v_new;
                            t_final_reg[a as usize] = timestamp + 1;
                        }
                    };
                }

                drop(_enter);
                drop(span);

                (v_final_reg, t_final_reg, v_read_reg, t_read_reg, v_write_rd)
            },
        );
        drop(_enter);
        drop(span);

        let (v_final_reg, t_final_reg, v_read_reg, t_read_reg, v_write_rd) = result.1;
        let (v_final_ram, t_final_ram, v_read_ram, t_read_ram, v_write_ram, t_write_ram, a_ram) =
            result.0;

        let v_init_ram = Word::get_bytes_array_from_u64_vec(v_init_ram);
        let v_final_ram = Word::get_bytes_array_from_u64_vec(v_final_ram);

        let (
            [a_ram, v_write_rd, v_init_reg, mut v_final_reg, mut t_final_reg, t_final_ram],
            v_init_ram,
            v_final_ram,
            v_read_reg,
            v_read_ram,
            v_write_ram,
            t_read_reg_polys,
            t_read_ram_polys,
            t_write_ram,
            remainder,
        ): (
            [DensePolynomial<F>; 6],
            [DensePolynomial<F>; WORD_BYTES],
            [DensePolynomial<F>; WORD_BYTES],
            [DensePolynomial<F>; REG_OPS_PER_INSTRUCTION],
            [DensePolynomial<F>; WORD_BYTES],
            [DensePolynomial<F>; WORD_BYTES],
            [DensePolynomial<F>; REG_OPS_PER_INSTRUCTION],
            DensePolynomial<F>,
            DensePolynomial<F>,
            DensePolynomial<F>,
        ) = common::par_join_10!(
            || map_to_polys(&[
                a_ram,
                v_write_rd,
                v_init_reg,
                v_final_reg,
                t_final_reg,
                t_final_ram
            ]),
            || map_to_polys(&v_init_ram),
            || map_to_polys(&v_final_ram),
            || map_to_polys(&[
                v_read_reg[0].clone(),
                v_read_reg[1].clone(),
                v_read_reg[2].clone()
            ]),
            || map_to_polys(&Word::get_bytes_array_from_word_vec(&v_read_ram)),
            || map_to_polys(&Word::get_bytes_array_from_word_vec(&v_write_ram)),
            || map_to_polys(&[
                t_read_reg[0].clone(),
                t_read_reg[1].clone(),
                t_read_reg[2].clone()
            ]),
            || map_to_polys(&[t_read_ram.clone()])[0].clone(),
            || map_to_polys(&[t_write_ram])[0].clone(),
            || map_to_polys(&[remainder_vec])[0].clone()
        );

        v_final_reg.padded_to_length(v_final_ram[0].len());
        t_final_reg.padded_to_length(t_final_ram.len());

        (
            Self {
                _group: PhantomData,
                memory_size,
                v_init_reg,
                v_init_ram,
                a_ram,
                v_read_reg,
                v_read_ram,
                v_write_rd,
                v_write_ram,
                v_final_reg,
                v_final_ram,
                t_read_reg: t_read_reg_polys,
                t_read_ram: t_read_ram_polys,
                t_write_ram,
                t_final_reg,
                t_final_ram,
                remainder,
            },
            t_read_reg,
            t_read_ram,
        )
    }

    #[tracing::instrument(skip_all, name = "ReadWriteMemory::get_polys_r1cs")]
    pub fn get_polys_r1cs<'a>(&'a self) -> (&'a [F], Vec<&'a F>, Vec<&'a F>) {
        let (a_polys, (v_read_polys, v_write_polys)) = rayon::join(
            || self.a_ram.evals_ref(),
            || {
                rayon::join(
                    || {
                        self.v_read_reg
                            .par_iter()
                            .chain(self.v_read_ram.par_iter())
                            .flat_map(|poly| poly.evals_ref().par_iter())
                            .collect::<Vec<_>>()
                    },
                    || {
                        [&self.v_write_rd]
                            .into_par_iter()
                            .chain(self.v_write_ram.par_iter())
                            .flat_map(|poly| poly.evals_ref().par_iter())
                            .collect::<Vec<_>>()
                    },
                )
            },
        );

        (a_polys, v_read_polys, v_write_polys)
    }

    /// Computes the shape of all commitments.
    pub fn commitment_shapes(
        max_memory_address: usize,
        max_trace_length: usize,
    ) -> Vec<CommitShape> {
        let max_memory_address = max_memory_address.next_power_of_two();
        let max_trace_length = max_trace_length.next_power_of_two();

        // { rs1, rs2, rd, ram_byte_1, ram_byte_2, ram_byte_3, ram_byte_4 }
        let t_read_write_len = (max_trace_length
            * (REG_OPS_PER_INSTRUCTION + RAM_WORD_OPS_PER_INSTRUCTION))
            .next_power_of_two();
        let t_read_write_shape = CommitShape::new(t_read_write_len, BatchType::Big);

        // { a_ram, v_read, v_write_rd, v_write_ram }
        let r1cs_shape = CommitShape::new(max_trace_length, BatchType::Big);
        // v_final, t_final
        let init_final_len = max_memory_address.next_power_of_two();
        let init_final_shape = CommitShape::new(init_final_len, BatchType::Small);

        vec![t_read_write_shape, r1cs_shape, init_final_shape]
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct MemoryCommitment<C: CommitmentScheme> {
    pub trace_commitments: Vec<C::Commitment>,
    pub v_final_reg_commitment: C::Commitment,
    pub t_final_reg_commitment: C::Commitment,
    pub v_final_ram_commitment: Vec<C::Commitment>,
    pub t_final_ram_commitment: C::Commitment,
}

impl<C: CommitmentScheme> AppendToTranscript for MemoryCommitment<C> {
    fn append_to_transcript(&self, label: &'static [u8], transcript: &mut ProofTranscript) {
        transcript.append_message(label, b"MemoryCommitment_begin");
        for commitment in &self.trace_commitments {
            commitment.append_to_transcript(b"trace_commit", transcript);
        }
        self.v_final_reg_commitment
            .append_to_transcript(b"v_final_commit", transcript);
        self.t_final_reg_commitment
            .append_to_transcript(b"t_final_commit", transcript);
        transcript.append_message(label, b"MemoryCommitment_end");
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct MemoryReadWriteOpenings<F, C>
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
{
    /// Evaluation of the a_read_write polynomial at the opening point.
    pub a_read_write_opening: [F; REG_OPS_PER_INSTRUCTION + RAM_WORD_OPS_PER_INSTRUCTION],
    /// Evaluation of the v_read polynomial at the opening point.
    pub v_read_opening: [F; MEMORY_OPS_PER_INSTRUCTION],
    /// Evaluation of the v_write polynomial at the opening point.
    pub v_write_opening: [F; 5],
    /// Evaluation of the t_read polynomial at the opening point.
    pub t_read_opening: [F; REG_OPS_PER_INSTRUCTION + RAM_WORD_OPS_PER_INSTRUCTION],
    /// Evaluation of the t_write_ram polynomial at the opening point.
    pub t_write_ram_opening: [F; 1],
    pub identity_poly_opening: Option<F>,
}

impl<F, C> StructuredOpeningProof<F, C, JoltPolynomials<F, C>> for MemoryReadWriteOpenings<F, C>
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
{
    type Proof = C::BatchedProof;

    #[tracing::instrument(skip_all, name = "MemoryReadWriteOpenings::open")]
    fn open(polynomials: &JoltPolynomials<F, C>, opening_point: &[F]) -> Self {
        let chis = EqPolynomial::evals(opening_point);
        let mut openings = [
            &polynomials.bytecode.v_read_write[2], // rd
            &polynomials.bytecode.v_read_write[3], // rs1
            &polynomials.bytecode.v_read_write[4], // rs2
            &polynomials.read_write_memory.a_ram,
        ]
        .into_par_iter()
        .chain(polynomials.read_write_memory.v_read_reg.par_iter())
        .chain(polynomials.read_write_memory.v_read_ram.par_iter())
        .chain([&polynomials.read_write_memory.v_write_rd].into_par_iter())
        .chain(polynomials.read_write_memory.v_write_ram.par_iter())
        .chain(polynomials.read_write_memory.t_read_reg.par_iter())
        .chain([&polynomials.read_write_memory.t_read_ram].into_par_iter())
        .chain([&polynomials.read_write_memory.t_write_ram].into_par_iter())
        .map(|poly| poly.evaluate_at_chi(&chis))
        .collect::<Vec<F>>()
        .into_iter();

        let a_read_write_opening = openings.next_chunk().unwrap();
        let v_read_opening = openings.next_chunk().unwrap();
        let v_write_opening = openings.next_chunk().unwrap();
        let t_read_opening = openings.next_chunk().unwrap();
        let t_write_ram_opening = openings.next_chunk().unwrap();

        Self {
            a_read_write_opening,
            v_read_opening,
            v_write_opening,
            t_read_opening,
            t_write_ram_opening,
            identity_poly_opening: None,
        }
    }

    #[tracing::instrument(skip_all, name = "MemoryReadWriteOpenings::-ings")]
    fn prove_openings(
        generators: &C::Setup,
        polynomials: &JoltPolynomials<F, C>,
        opening_point: &[F],
        openings: &Self,
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        let read_write_polys = [
            &polynomials.bytecode.v_read_write[2], // rd
            &polynomials.bytecode.v_read_write[3], // rs1
            &polynomials.bytecode.v_read_write[4], // rs2
            &polynomials.read_write_memory.a_ram,
        ]
        .into_iter()
        .chain(polynomials.read_write_memory.v_read_reg.iter())
        .chain(polynomials.read_write_memory.v_read_ram.iter())
        .chain([&polynomials.read_write_memory.v_write_rd].into_iter())
        .chain(polynomials.read_write_memory.v_write_ram.iter())
        .chain(polynomials.read_write_memory.t_read_reg.iter())
        .chain([&polynomials.read_write_memory.t_read_ram].into_iter())
        .chain([&polynomials.read_write_memory.t_write_ram].into_iter())
        .collect::<Vec<_>>();
        let read_write_openings = openings
            .a_read_write_opening
            .into_iter()
            .chain(openings.v_read_opening.into_iter())
            .chain(openings.v_write_opening.into_iter())
            .chain(openings.t_read_opening.into_iter())
            .chain(openings.t_write_ram_opening.into_iter())
            .collect::<Vec<_>>();

        C::batch_prove(
            generators,
            &read_write_polys,
            opening_point,
            &read_write_openings,
            BatchType::Big,
            transcript,
        )
    }

    fn compute_verifier_openings(&mut self, _: &NoPreprocessing, opening_point: &[F]) {
        self.identity_poly_opening =
            Some(IdentityPolynomial::new(opening_point.len()).evaluate(opening_point));
    }

    fn verify_openings(
        &self,
        generators: &C::Setup,
        opening_proof: &Self::Proof,
        commitment: &JoltCommitments<C>,
        opening_point: &[F],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let openings = self
            .a_read_write_opening
            .into_iter()
            .chain(self.v_read_opening)
            .chain(self.v_write_opening)
            .chain(self.t_read_opening)
            .chain(self.t_write_ram_opening)
            .collect::<Vec<_>>();
        C::batch_verify(
            opening_proof,
            generators,
            opening_point,
            &openings,
            &commitment.bytecode.trace_commitments[4..7]
                .iter()
                .chain(
                    commitment.read_write_memory.trace_commitments
                        [..commitment.read_write_memory.trace_commitments.len() - 1]
                        .iter(),
                )
                .collect::<Vec<_>>(),
            transcript,
        )
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct MemoryInitFinalOpenings<F>
where
    F: JoltField,
{
    /// Evaluation of the a_init_final polynomial at the opening point. Computed by the verifier in `compute_verifier_openings`.
    a_init_final_reg: Option<F>,
    a_init_final_ram: Option<F>,
    /// Evaluation of the v_init polynomial at the opening point. Computed by the verifier in `compute_verifier_openings`.
    v_init_reg: Option<F>,
    v_init_ram: [Option<F>; WORD_BYTES],
    /// Evaluation of the v_final polynomial at the opening point.
    v_final_reg: F,
    v_final_ram: [F; WORD_BYTES],
    /// Evaluation of the t_final polynomial at the opening point.
    t_final_reg: F,
    t_final_ram: F,
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct MemoryInitFinalOpeningProof<F, C>
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
{
    v_t_opening_proof: C::BatchedProof,
}

impl<F, C> StructuredOpeningProof<F, C, JoltPolynomials<F, C>> for MemoryInitFinalOpenings<F>
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
{
    type Proof = MemoryInitFinalOpeningProof<F, C>;
    type Preprocessing = ReadWriteMemoryPreprocessing;

    #[tracing::instrument(skip_all, name = "MemoryInitFinalOpenings::open")]
    fn open(polynomials: &JoltPolynomials<F, C>, opening_point: &[F]) -> Self {
        let chis = EqPolynomial::evals(opening_point);

        let ((v_final_reg, v_final_ram), (t_final_reg, t_final_ram)) = rayon::join(
            || {
                rayon::join(
                    || {
                        polynomials
                            .read_write_memory
                            .v_final_reg
                            .evaluate_at_chi(&chis)
                    },
                    || {
                        let evaluations: Vec<_> = polynomials
                            .read_write_memory
                            .v_final_ram
                            .iter()
                            .take(WORD_BYTES)
                            .map(|ram| ram.evaluate_at_chi(&chis))
                            .collect();

                        evaluations
                    },
                )
            },
            || {
                rayon::join(
                    || {
                        polynomials
                            .read_write_memory
                            .t_final_reg
                            .evaluate_at_chi(&chis)
                    },
                    || {
                        polynomials
                            .read_write_memory
                            .t_final_ram
                            .evaluate_at_chi(&chis)
                    },
                )
            },
        );

        Self {
            a_init_final_reg: None,
            a_init_final_ram: None,
            v_init_reg: None,
            v_init_ram: [None, None, None, None],
            v_final_reg,
            v_final_ram: v_final_ram.try_into().unwrap(),
            t_final_reg,
            t_final_ram,
        }
    }

    #[tracing::instrument(skip_all, name = "MemoryInitFinalOpenings::prove_openings")]
    fn prove_openings(
        generators: &C::Setup,
        polynomials: &JoltPolynomials<F, C>,
        opening_point: &[F],
        openings: &Self,
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        let v_t_opening_proof = C::batch_prove(
            generators,
            &[
                &polynomials.read_write_memory.v_final_reg,
                &polynomials.read_write_memory.v_final_ram[0],
                &polynomials.read_write_memory.v_final_ram[1],
                &polynomials.read_write_memory.v_final_ram[2],
                &polynomials.read_write_memory.v_final_ram[3],
                &polynomials.read_write_memory.t_final_reg,
                &polynomials.read_write_memory.t_final_ram,
            ],
            &opening_point,
            &[
                openings.v_final_reg,
                openings.v_final_ram[0],
                openings.v_final_ram[1],
                openings.v_final_ram[2],
                openings.v_final_ram[3],
                openings.t_final_reg,
                openings.t_final_ram,
            ],
            BatchType::Small,
            transcript,
        );

        Self::Proof { v_t_opening_proof }
    }

    fn compute_verifier_openings(
        &mut self,
        preprocessing: &Self::Preprocessing,
        opening_point: &[F],
    ) {
        self.a_init_final_reg =
            Some(IdentityPolynomial::new(opening_point.len()).evaluate(opening_point));

        self.a_init_final_ram =
            Some(IdentityPolynomial::new(opening_point.len()).evaluate(opening_point));

        let memory_layout = &preprocessing.program_io.as_ref().unwrap().memory_layout;

        // TODO(moodlezoup): Compute opening without instantiating v_init polynomial itself
        let memory_size = opening_point.len().pow2();
        let mut v_init: [Vec<u64>; 4] = [
            vec![0; memory_size],
            vec![0; memory_size],
            vec![0; memory_size],
            vec![0; memory_size],
        ];
        // Copy bytecode
        let mut v_init_index = memory_address_to_witness_index(
            preprocessing.min_bytecode_address,
            memory_layout.ram_witness_offset,
        );
        v_init_index -= REGISTER_COUNT as usize;
        for byte in preprocessing.bytecode_bytes.iter() {
            let quotient = v_init_index / WORD_BYTES;
            let remainder = v_init_index % WORD_BYTES;
            v_init[remainder][quotient] = *byte as u64;
            v_init_index += 1;
        }
        // Copy input bytes
        v_init_index = memory_address_to_witness_index(
            memory_layout.input_start,
            memory_layout.ram_witness_offset,
        );
        v_init_index -= REGISTER_COUNT as usize;
        for byte in preprocessing.program_io.as_ref().unwrap().inputs.iter() {
            let quotient = v_init_index / 4;
            let remainder = v_init_index % 4;
            v_init[remainder][quotient] = *byte as u64;
            v_init_index += 1;
        }

        self.v_init_reg = Some(F::zero());
        for (i, value) in v_init.iter().take(4).enumerate() {
            self.v_init_ram[i] = Some(DensePolynomial::from_u64(value).evaluate(&opening_point));
        }
    }

    fn verify_openings(
        &self,
        generators: &C::Setup,
        opening_proof: &Self::Proof,
        commitment: &JoltCommitments<C>,
        opening_point: &[F],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        // println!("The length of v_final_reg_commitments is {:?}", commitment.read_write_memory.v_final_reg_commitment;

        C::batch_verify(
            &opening_proof.v_t_opening_proof,
            generators,
            opening_point,
            &[
                self.v_final_reg,
                self.v_final_ram[0],
                self.v_final_ram[1],
                self.v_final_ram[2],
                self.v_final_ram[3],
                self.t_final_reg,
                self.t_final_ram,
            ],
            &[
                &commitment.read_write_memory.v_final_reg_commitment,
                &commitment.read_write_memory.v_final_ram_commitment[0],
                &commitment.read_write_memory.v_final_ram_commitment[1],
                &commitment.read_write_memory.v_final_ram_commitment[2],
                &commitment.read_write_memory.v_final_ram_commitment[3],
                &commitment.read_write_memory.t_final_reg_commitment,
                &commitment.read_write_memory.t_final_ram_commitment,
            ],
            transcript,
        )?;
        Ok(())
    }
}

impl<F, C> MemoryCheckingProver<F, C, JoltPolynomials<F, C>> for ReadWriteMemoryProof<F, C>
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
{
    type Preprocessing = ReadWriteMemoryPreprocessing;
    type ReadWriteOpenings = MemoryReadWriteOpenings<F, C>;
    type InitFinalOpenings = MemoryInitFinalOpenings<F>;

    // (a, v, t)
    type MemoryTuple = (F, F, F);

    fn fingerprint(inputs: &(F, F, F), gamma: &F, tau: &F) -> F {
        let (a, v, t) = *inputs;
        t * gamma.square() + v * *gamma + a - *tau
    }

    #[tracing::instrument(skip_all, name = "ReadWriteMemory::compute_leaves")]
    fn compute_leaves(
        _: &Self::Preprocessing,
        polynomials: &JoltPolynomials<F, C>,
        gamma: &F,
        tau: &F,
    ) -> (Vec<Vec<F>>, Vec<Vec<F>>) {
        let mut gamma_powers = [F::zero(); 6];
        let mut power = F::one();
        for index in 0..6 {
            gamma_powers[index] = power;
            power *= *gamma;
        }

        let num_ops = polynomials.read_write_memory.a_ram.len();

        let read_write_leaves = (0..REG_OPS_PER_INSTRUCTION + RAM_WORD_OPS_PER_INSTRUCTION)
            .into_par_iter()
            .flat_map(|i| {
                let read_fingerprints = (0..num_ops)
                    .into_par_iter()
                    .map(|j| {
                        let leaf = match i {
                            RS1 => {
                                polynomials.read_write_memory.t_read_reg[i][j] * gamma_powers[2]
                                    + mul_0_optimized(
                                        &polynomials.read_write_memory.v_read_reg[i][j],
                                        &gamma_powers[1],
                                    )
                                    + polynomials.bytecode.v_read_write[3][j]
                                    - *tau
                            }
                            RS2 => {
                                polynomials.read_write_memory.t_read_reg[i][j] * gamma_powers[2]
                                    + mul_0_optimized(
                                        &polynomials.read_write_memory.v_read_reg[i][j],
                                        &gamma_powers[1],
                                    )
                                    + polynomials.bytecode.v_read_write[4][j]
                                    - *tau
                            }
                            RD => {
                                polynomials.read_write_memory.t_read_reg[i][j] * gamma_powers[2]
                                    + mul_0_optimized(
                                        &polynomials.read_write_memory.v_read_reg[i][j],
                                        &gamma_powers[1],
                                    )
                                    + polynomials.bytecode.v_read_write[2][j]
                                    - *tau
                            }
                            _ => {
                                polynomials.read_write_memory.a_ram[j]
                                    + polynomials.read_write_memory.t_read_ram[j] * gamma_powers[5]
                                    + (0..WORD_BYTES)
                                        .map(|i| {
                                            mul_0_optimized(
                                                &polynomials.read_write_memory.v_read_ram[i][j],
                                                &gamma_powers[i + 1],
                                            )
                                        })
                                        .sum()
                                    - *tau
                            }
                        };
                        leaf
                    })
                    .collect();

                let write_fingerprints: Vec<F> = (0..num_ops)
                    .into_par_iter()
                    .map(|j| {
                        let leaf = match i {
                            RS1 => {
                                F::from_u64(j as u64).unwrap() * gamma_powers[2]
                                    + mul_0_optimized(
                                        &polynomials.read_write_memory.v_read_reg[i][j],
                                        &gamma_powers[1],
                                    )
                                    + polynomials.bytecode.v_read_write[3][j]
                                    - *tau
                            }
                            RS2 => {
                                F::from_u64(j as u64).unwrap() * gamma_powers[2]
                                    + mul_0_optimized(
                                        &polynomials.read_write_memory.v_read_reg[i][j],
                                        &gamma_powers[1],
                                    )
                                    + polynomials.bytecode.v_read_write[4][j]
                                    - *tau
                            }
                            RD => {
                                F::from_u64((j + 1) as u64).unwrap() * gamma_powers[2]
                                    + mul_0_optimized(
                                        &polynomials.read_write_memory.v_write_rd[j],
                                        &gamma_powers[1],
                                    )
                                    + polynomials.bytecode.v_read_write[2][j]
                                    - *tau
                            }
                            _ => {
                                polynomials.read_write_memory.a_ram[j]
                                    + polynomials.read_write_memory.t_write_ram[j] * gamma_powers[5]
                                    + (0..WORD_BYTES)
                                        .map(|i| {
                                            mul_0_optimized(
                                                &polynomials.read_write_memory.v_write_ram[i][j],
                                                &gamma_powers[i + 1],
                                            )
                                        })
                                        .sum()
                                    - *tau
                            }
                        };

                        leaf
                    })
                    .collect();

                [read_fingerprints, write_fingerprints]
            })
            .collect();

        let init_fingerprints_reg: Vec<F> = (0..((polynomials.read_write_memory.memory_size
            - (REGISTER_COUNT as usize))
            / WORD_BYTES)
            .next_power_of_two())
            .into_par_iter()
            .map(|i| {
                if i < (REGISTER_COUNT as usize) {
                    /* 0 * gamma^2 + */
                    mul_0_optimized(&polynomials.read_write_memory.v_init_reg[i as usize], gamma)
                        + F::from_u64(i as u64).unwrap()
                        - *tau
                } else {
                    F::from_u64(i as u64).unwrap() - *tau
                }
            })
            .collect();
        let init_fingerprints_memory: Vec<F> = (0..((polynomials.read_write_memory.memory_size
            - (REGISTER_COUNT as usize))
            / WORD_BYTES)
            .next_power_of_two())
            .into_par_iter()
            .map(|i|
                    /* 0 * gamma^2 + */
                    F::from_u64(i as u64).unwrap() +
                    (0..WORD_BYTES)
                        .map(|j|
                            mul_0_optimized(
                                &polynomials.read_write_memory.v_init_ram[j][i as usize],
                                &gamma_powers[j + 1]
                            )
                        )
                        .sum() -
                    *tau)
            .collect();

        let final_fingerprints_reg: Vec<F> = (0..((polynomials.read_write_memory.memory_size
            - (REGISTER_COUNT as usize))
            / WORD_BYTES)
            .next_power_of_two())
            .into_par_iter()
            .map(|i| {
                if i < (REGISTER_COUNT as usize) {
                    /* 0 * gamma^2 + */
                    mul_0_optimized(
                        &polynomials.read_write_memory.t_final_reg[i as usize],
                        &gamma_powers[2],
                    ) + mul_0_optimized(
                        &polynomials.read_write_memory.v_final_reg[i as usize],
                        &gamma_powers[1],
                    ) + F::from_u64(i as u64).unwrap()
                        - *tau
                } else {
                    F::from_u64(i as u64).unwrap() - *tau
                }
            })
            .collect();
        let final_fingerprints_memory: Vec<F> = (0..((polynomials.read_write_memory.memory_size
            - (REGISTER_COUNT as usize))
            / WORD_BYTES)
            .next_power_of_two())
            .into_par_iter()
            .map(|i| {
                mul_0_optimized(
                    &polynomials.read_write_memory.t_final_ram[i],
                    &gamma_powers[5],
                ) + (0..WORD_BYTES)
                    .map(|j| {
                        mul_0_optimized(
                            &polynomials.read_write_memory.v_final_ram[j][i as usize],
                            &gamma_powers[j + 1],
                        )
                    })
                    .sum()
                    + F::from_u64(i as u64).unwrap()
                    - *tau
            })
            .collect();

        (
            read_write_leaves,
            vec![
                init_fingerprints_reg,
                final_fingerprints_reg,
                init_fingerprints_memory,
                final_fingerprints_memory,
            ],
        )
    }

    fn uninterleave_hashes(
        _preprocessing: &Self::Preprocessing,
        read_write_hashes: Vec<F>,
        init_final_hashes: Vec<F>,
    ) -> MultisetHashes<F> {
        assert_eq!(
            read_write_hashes.len(),
            2 * (REG_OPS_PER_INSTRUCTION + RAM_WORD_OPS_PER_INSTRUCTION)
        );
        let mut read_hashes =
            Vec::with_capacity(REG_OPS_PER_INSTRUCTION + RAM_WORD_OPS_PER_INSTRUCTION);
        let mut write_hashes =
            Vec::with_capacity(REG_OPS_PER_INSTRUCTION + RAM_WORD_OPS_PER_INSTRUCTION);
        for i in 0..REG_OPS_PER_INSTRUCTION + RAM_WORD_OPS_PER_INSTRUCTION {
            read_hashes.push(read_write_hashes[2 * i]);
            write_hashes.push(read_write_hashes[2 * i + 1]);
        }

        assert_eq!(init_final_hashes.len(), 4);
        let init_hash = vec![init_final_hashes[0], init_final_hashes[2]];
        let final_hash = vec![init_final_hashes[1], init_final_hashes[3]];

        MultisetHashes {
            read_hashes,
            write_hashes,
            init_hashes: init_hash,
            final_hashes: final_hash,
        }
    }

    fn check_multiset_equality(
        _preprocessing: &Self::Preprocessing,
        multiset_hashes: &MultisetHashes<F>,
    ) {
        assert_eq!(
            multiset_hashes.read_hashes.len(),
            REG_OPS_PER_INSTRUCTION + RAM_WORD_OPS_PER_INSTRUCTION
        );
        assert_eq!(
            multiset_hashes.write_hashes.len(),
            REG_OPS_PER_INSTRUCTION + RAM_WORD_OPS_PER_INSTRUCTION
        );
        assert_eq!(multiset_hashes.init_hashes.len(), 2);
        assert_eq!(multiset_hashes.final_hashes.len(), 2);

        let read_hash: F = multiset_hashes.read_hashes.iter().product();
        let write_hash: F = multiset_hashes.write_hashes.iter().product();
        let init_hash: F = multiset_hashes.init_hashes.iter().product();
        let final_hash: F = multiset_hashes.final_hashes.iter().product();

        assert_eq!(
            init_hash * write_hash,
            final_hash * read_hash,
            "Multiset hashes don't match"
        );
    }

    fn protocol_name() -> &'static [u8] {
        b"Registers/RAM memory checking"
    }
}

impl<F, C> MemoryCheckingVerifier<F, C, JoltPolynomials<F, C>> for ReadWriteMemoryProof<F, C>
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
{
    fn read_tuples(
        &_: &Self::Preprocessing,
        openings: &Self::ReadWriteOpenings,
    ) -> Vec<Self::MemoryTuple> {
        (0..REG_OPS_PER_INSTRUCTION)
            .map(|i| {
                let a = match i {
                    RD => openings.a_read_write_opening[0],
                    RS1 => openings.a_read_write_opening[1],
                    RS2 => openings.a_read_write_opening[2],
                    _ => panic!("Error in index"),
                };
                (a, openings.v_read_opening[i], openings.t_read_opening[i])
            })
            .collect()
    }
    fn write_tuples(
        &_: &Self::Preprocessing,
        openings: &Self::ReadWriteOpenings,
    ) -> Vec<Self::MemoryTuple> {
        (0..REG_OPS_PER_INSTRUCTION)
            .map(|i| {
                let a = match i {
                    RD => openings.a_read_write_opening[0],
                    RS1 => openings.a_read_write_opening[1],
                    RS2 => openings.a_read_write_opening[2],
                    _ => panic!("Error in index"),
                };
                let v = if i == RS1 || i == RS2 {
                    // For rs1 and rs2, v_write = v_read
                    openings.v_read_opening[i]
                } else {
                    openings.v_write_opening[i - 2]
                };
                let t = if i == RS1 || i == RS2 {
                    openings.identity_poly_opening.unwrap()
                } else {
                    openings.identity_poly_opening.unwrap() + F::one()
                };
                (a, v, t)
            })
            .collect()
    }
    fn init_tuples(
        &_: &Self::Preprocessing,
        openings: &Self::InitFinalOpenings,
    ) -> Vec<Self::MemoryTuple> {
        vec![(
            openings.a_init_final_reg.unwrap(),
            openings.v_init_reg.unwrap(),
            F::zero(),
        )]
    }
    fn final_tuples(
        &_: &Self::Preprocessing,
        openings: &Self::InitFinalOpenings,
    ) -> Vec<Self::MemoryTuple> {
        vec![(
            openings.a_init_final_reg.unwrap(),
            openings.v_final_reg,
            openings.t_final_reg,
        )]
    }

    // Redefining Check Fingerprint
    fn check_fingerprints(
        preprocessing: &Self::Preprocessing,
        claims_read_write: Vec<F>,
        claims_init_final: Vec<F>,
        read_write_openings: &Self::ReadWriteOpenings,
        init_final_openings: &Self::InitFinalOpenings,
        gamma: &F,
        tau: &F,
    ) {
        let mut gamma_powers = [F::zero(); 6];
        let mut power = F::one();
        for index in 0..6 {
            gamma_powers[index] = power;
            power *= *gamma;
        }

        let mut read_hashes: Vec<_> = Self::read_tuples(preprocessing, &read_write_openings)
            .iter()
            .map(|tuple| Self::fingerprint(tuple, gamma, tau))
            .collect();

        // Collecting read value corresponding to memory

        let mut v = read_write_openings.v_read_opening[3];
        for i in 1..4 {
            v += gamma_powers[i] * read_write_openings.v_read_opening[3 + i];
        }
        let t = read_write_openings.t_read_opening[3];
        let a = read_write_openings.a_read_write_opening[3];

        let fingerprint = t * gamma_powers[5] + v * gamma_powers[1] + a - *tau;
        read_hashes.push(fingerprint);

        let mut write_hashes: Vec<_> = Self::write_tuples(preprocessing, &read_write_openings)
            .iter()
            .map(|tuple| Self::fingerprint(tuple, gamma, tau))
            .collect();

        // Collecting value corresponding to memory
        let mut v = read_write_openings.v_write_opening[1];
        for i in 1..4 {
            v += gamma_powers[i] * read_write_openings.v_write_opening[1 + i];
        }

        let t = read_write_openings.t_write_ram_opening[0];
        let a = read_write_openings.a_read_write_opening[3];

        let fingerprint = t * gamma_powers[5] + v * gamma_powers[1] + a - *tau;
        write_hashes.push(fingerprint);

        let mut init_hashes: Vec<_> = Self::init_tuples(preprocessing, init_final_openings)
            .iter()
            .map(|tuple| Self::fingerprint(tuple, gamma, tau))
            .collect();

        // Collecting memory init value
        let mut v = init_final_openings.v_init_ram[0].unwrap();
        for i in 1..4 {
            v += gamma_powers[i] * init_final_openings.v_init_ram[0 + i].unwrap();
        }

        let a = init_final_openings.a_init_final_ram.unwrap();
        let fingerprint = v * gamma_powers[1] + a - *tau;
        init_hashes.push(fingerprint);

        let mut final_hashes: Vec<_> = Self::final_tuples(preprocessing, init_final_openings)
            .iter()
            .map(|tuple| Self::fingerprint(tuple, gamma, tau))
            .collect();

        // Collecting memory final value

        let mut v = init_final_openings.v_final_ram[0];
        for i in 1..4 {
            v += gamma_powers[i] * init_final_openings.v_final_ram[0 + i];
        }
        let t = init_final_openings.t_final_ram;
        let a = init_final_openings.a_init_final_ram.unwrap();

        let fingerprint = t * gamma_powers[5] + v * gamma_powers[1] + a - *tau;
        final_hashes.push(fingerprint);

        assert_eq!(
            read_hashes.len() + write_hashes.len(),
            claims_read_write.len()
        );
        assert_eq!(
            init_hashes.len() + final_hashes.len(),
            claims_init_final.len()
        );

        let multiset_hashes = MultisetHashes {
            read_hashes,
            write_hashes,
            init_hashes,
            final_hashes,
        };
        let (read_write_hashes, init_final_hashes) =
            Self::interleave_hashes(preprocessing, &multiset_hashes);

        for (claim, fingerprint) in zip(claims_read_write, read_write_hashes) {
            assert_eq!(claim, fingerprint);
        }
        for (claim, fingerprint) in zip(claims_init_final, init_final_hashes) {
            assert_eq!(claim, fingerprint);
        }
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct OutputSumcheckProof<F, C>
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
{
    num_rounds: usize,
    /// Sumcheck proof that v_final is equal to the program outputs at the relevant indices.
    sumcheck_proof: SumcheckInstanceProof<F>,
    /// Opening of v_final at the random point chosen over the course of sumcheck
    opening: Vec<F>,
    /// Hyrax opening proof of the v_final opening
    opening_proof: C::BatchedProof,
}

impl<F, C> OutputSumcheckProof<F, C>
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
{
    fn prove_outputs(
        generators: &C::Setup,
        polynomials: &ReadWriteMemory<F, C>,
        program_io: &JoltDevice,
        transcript: &mut ProofTranscript,
    ) -> Self {
        let actual_memory_size =
            (polynomials.memory_size - (REGISTER_COUNT as usize)).next_power_of_two() / WORD_BYTES;
        let num_rounds = actual_memory_size.log_2();
        let r_eq = transcript.challenge_vector(b"output_sumcheck", num_rounds);

        let eq: DensePolynomial<F> = DensePolynomial::new(EqPolynomial::evals(&r_eq));
        //Error: need to map input_start and output_start

        let mut input_start = memory_address_to_witness_index(
            program_io.memory_layout.input_start,
            program_io.memory_layout.ram_witness_offset,
        );
        input_start = input_start - (REGISTER_COUNT as usize);
        // Assume memory size is greater than 128 so that next power of two does not change
        let io_witness_range: Vec<Vec<F>> = (0..4)
            .map(|io_witness_range_index| {
                (0..actual_memory_size as u64)
                    .map(|i| {
                        if 4 * i + io_witness_range_index >= (input_start as u64)
                            && 4 * i < program_io.memory_layout.ram_witness_offset - REGISTER_COUNT
                        {
                            F::one()
                        } else {
                            F::zero()
                        }
                    })
                    .collect()
            })
            .collect();

        let mut v_io: Vec<Vec<u64>> = vec![vec![0; actual_memory_size]; WORD_BYTES];
        // Copy input bytes
        let mut input_index = memory_address_to_witness_index(
            program_io.memory_layout.input_start,
            program_io.memory_layout.ram_witness_offset,
        );
        input_index = input_index - (REGISTER_COUNT as usize);
        let mut input_index_quotient = input_index / WORD_BYTES;
        let mut input_index_remainder = input_index % WORD_BYTES;
        for byte in program_io.inputs.iter() {
            v_io[input_index_remainder][input_index_quotient] = *byte as u64;
            input_index += 1;
            input_index_quotient = input_index / WORD_BYTES;
            input_index_remainder = input_index % WORD_BYTES;
        }
        // Copy output bytes
        let mut output_index = memory_address_to_witness_index(
            program_io.memory_layout.output_start,
            program_io.memory_layout.ram_witness_offset,
        );
        output_index = output_index - (REGISTER_COUNT as usize);
        let mut output_index_quotient = output_index / WORD_BYTES;
        let mut output_index_remainder = output_index % WORD_BYTES;
        for byte in program_io.outputs.iter() {
            v_io[output_index_remainder][output_index_quotient] = *byte as u64;
            output_index += 1;
            output_index_quotient = output_index / WORD_BYTES;
            output_index_remainder = output_index % WORD_BYTES;
        }

        // Copy panic bit
        let panic_index = memory_address_to_witness_index(
            program_io.memory_layout.panic,
            program_io.memory_layout.ram_witness_offset,
        );
        let panic_index_quotient = (panic_index - (REGISTER_COUNT as usize)) / WORD_BYTES;
        let panic_index_remainder = (panic_index - (REGISTER_COUNT as usize)) % WORD_BYTES;
        v_io[panic_index_remainder][panic_index_quotient] = program_io.panic as u64;

        let mut sumcheck_polys = vec![eq.clone()];
        (0..WORD_BYTES).for_each(|i| {
            sumcheck_polys.append(&mut vec![
                DensePolynomial::new(io_witness_range[i].clone()),
                polynomials.v_final_ram[i].clone(),
                DensePolynomial::from_u64(&v_io[i]),
            ])
        });

        //Generate random points to aggregate the sum-check across the 4 memory bytes
        let r_combiner: Vec<F> = transcript.challenge_vector(b"sumcheck_combiners", WORD_BYTES);

        // eq * io_witness_range * (v_final - v_io)
        let output_check_fn = |vals: &[F]| -> F {
            (0..WORD_BYTES).fold(F::zero(), |acc, idx| {
                acc + (r_combiner[idx]
                    * (vals[3 * idx + 1] * (vals[3 * idx + 2] - vals[3 * idx + 3])))
            }) * vals[0]
        };

        let (sumcheck_proof, r_sumcheck, sumcheck_opening) =
            SumcheckInstanceProof::<F>::prove_arbitrary::<_>(
                &F::zero(),
                num_rounds,
                &mut sumcheck_polys,
                output_check_fn,
                3,
                transcript,
            );

        let v_final_ram = (0..WORD_BYTES)
            .map(|idx| &polynomials.v_final_ram[idx])
            .collect_vec();

        // only need v_final; verifier computes the rest on its own
        let opening: Vec<_> = sumcheck_opening
            .iter()
            .enumerate()
            .filter(|(idx, _)| (idx + 1) % 3 == 0)
            .map(|(_, &value)| value)
            .collect();

        let sumcheck_opening_proof = C::batch_prove(
            generators,
            &v_final_ram,
            &r_sumcheck,
            &opening,
            BatchType::Small,
            transcript,
        );

        Self {
            num_rounds,
            sumcheck_proof,
            opening,
            opening_proof: sumcheck_opening_proof,
        }
    }

    fn verify(
        proof: &Self,
        preprocessing: &ReadWriteMemoryPreprocessing,
        generators: &C::Setup,
        commitment: &MemoryCommitment<C>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let r_eq = transcript.challenge_vector(b"output_sumcheck", proof.num_rounds);

        let memory_layout = &preprocessing.program_io.as_ref().unwrap().memory_layout;

        let nonzero_memory_size = memory_layout.ram_witness_offset as usize;
        let log_nonzero_memory_size = (nonzero_memory_size / WORD_BYTES).log_2();
        assert!(
            nonzero_memory_size.is_power_of_two(),
            "Ram witness offset must be a power of two"
        );

        let mut input_start = memory_address_to_witness_index(
            memory_layout.input_start,
            memory_layout.ram_witness_offset,
        );
        input_start = input_start - (REGISTER_COUNT as usize);
        let io_witness_range: Vec<Vec<F>> = (0..WORD_BYTES as u64)
            .map(|io_witness_range_index| {
                (0..(nonzero_memory_size / WORD_BYTES) as u64)
                    .map(|i| {
                        if 4 * i + io_witness_range_index >= (input_start as u64)
                            && 4 * i + io_witness_range_index
                                < (nonzero_memory_size as u64) - REGISTER_COUNT
                        {
                            F::one()
                        } else {
                            F::zero()
                        }
                    })
                    .collect()
            })
            .collect();

        let mut v_io = [
            vec![0; nonzero_memory_size / WORD_BYTES],
            vec![0; nonzero_memory_size / WORD_BYTES],
            vec![0; nonzero_memory_size / WORD_BYTES],
            vec![0; nonzero_memory_size / WORD_BYTES],
        ];
        // Copy input bytes
        let mut input_index = memory_address_to_witness_index(
            memory_layout.input_start,
            memory_layout.ram_witness_offset,
        );
        input_index -= REGISTER_COUNT as usize;
        for byte in preprocessing.program_io.as_ref().unwrap().inputs.iter() {
            let remainder = input_index % WORD_BYTES;
            let quotient = input_index / WORD_BYTES;

            v_io[remainder][quotient] = *byte as u64;

            input_index += 1;
        }
        // Copy output bytes
        let mut output_index = memory_address_to_witness_index(
            memory_layout.output_start,
            memory_layout.ram_witness_offset,
        );
        output_index -= REGISTER_COUNT as usize;
        for byte in preprocessing.program_io.as_ref().unwrap().outputs.iter() {
            let remainder = output_index % WORD_BYTES;
            let quotient = output_index / WORD_BYTES;
            v_io[remainder][quotient] = *byte as u64;

            output_index += 1;
        }
        // Copy panic bit
        let mut panic_address =
            memory_address_to_witness_index(memory_layout.panic, memory_layout.ram_witness_offset);
        panic_address -= REGISTER_COUNT as usize;
        let remainder = panic_address % WORD_BYTES;
        let quotient = panic_address / WORD_BYTES;
        v_io[remainder][quotient] = preprocessing.program_io.as_ref().unwrap().panic as u64;

        //Generate random points to aggregate the sum-check across the 4 memory bytes
        let r_combiner: Vec<F> = transcript.challenge_vector(b"sumcheck_combiners", WORD_BYTES);

        let (sumcheck_claim, r_sumcheck) =
            proof
                .sumcheck_proof
                .verify(F::zero(), proof.num_rounds, 3, transcript)?;
        let eq_eval = EqPolynomial::new(r_eq.to_vec()).evaluate(&r_sumcheck);

        let r_prod = r_sumcheck
            .iter()
            .take(r_sumcheck.len() - log_nonzero_memory_size)
            .fold(F::one(), |acc, &r| acc * (F::one() - r));

        let io_witness_range_eval = (0..WORD_BYTES)
            .map(|idx| {
                DensePolynomial::new(io_witness_range[idx].clone()).evaluate(
                    &r_sumcheck[r_sumcheck.len() - log_nonzero_memory_size..r_sumcheck.len()],
                ) * r_prod
            })
            .collect_vec();

        let v_io_eval = (0..WORD_BYTES)
            .map(|idx| {
                DensePolynomial::from_u64(&v_io[idx]).evaluate(
                    &r_sumcheck[r_sumcheck.len() - log_nonzero_memory_size..r_sumcheck.len()],
                ) * r_prod
            })
            .collect_vec();

        let actual_sumcheck_claim = (0..WORD_BYTES).fold(F::zero(), |acc, idx| {
            acc + r_combiner[idx]
                * io_witness_range_eval[idx]
                * (proof.opening[idx] - v_io_eval[idx])
        }) * eq_eval;

        assert_eq!(
            actual_sumcheck_claim, sumcheck_claim,
            "Output sumcheck check failed."
        );

        match C::batch_verify(
            &proof.opening_proof,
            generators,
            &r_sumcheck,
            &proof.opening,
            &commitment
                .v_final_ram_commitment
                .iter()
                .map(|value| value)
                .collect_vec(),
            transcript,
        ) {
            Ok(_) => {}
            Err(error) => {
                return Err(error);
            }
        }

        Ok(())
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct ReadWriteMemoryProof<F, C>
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
{
    pub memory_checking_proof: MemoryCheckingProof<
        F,
        C,
        JoltPolynomials<F, C>,
        MemoryReadWriteOpenings<F, C>,
        MemoryInitFinalOpenings<F>,
    >,
    pub timestamp_validity_proof: TimestampValidityProof<F, C>,
    pub output_proof: OutputSumcheckProof<F, C>,
}

impl<F, C> ReadWriteMemoryProof<F, C>
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
{
    #[tracing::instrument(skip_all, name = "ReadWriteMemoryProof::prove")]
    pub fn prove(
        generators: &C::Setup,
        preprocessing: &ReadWriteMemoryPreprocessing,
        polynomials: &JoltPolynomials<F, C>,
        program_io: &JoltDevice,
        transcript: &mut ProofTranscript,
    ) -> Self {
        let memory_checking_proof = ReadWriteMemoryProof::prove_memory_checking(
            generators,
            preprocessing,
            polynomials,
            transcript,
        );

        let output_proof = OutputSumcheckProof::prove_outputs(
            generators,
            &polynomials.read_write_memory,
            program_io,
            transcript,
        );

        let mut t_read = polynomials.read_write_memory.t_read_reg.to_vec();
        t_read.push(polynomials.read_write_memory.t_read_ram.clone());
        let timestamp_validity_proof = TimestampValidityProof::prove(
            generators,
            &polynomials.timestamp_range_check,
            &t_read,
            transcript,
        );

        Self {
            memory_checking_proof,
            output_proof,
            timestamp_validity_proof,
        }
    }

    pub fn verify(
        mut self,
        generators: &C::Setup,
        preprocessing: &ReadWriteMemoryPreprocessing,
        commitment: &JoltCommitments<C>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        ReadWriteMemoryProof::verify_memory_checking(
            preprocessing,
            generators,
            self.memory_checking_proof,
            commitment,
            transcript,
        )?;

        OutputSumcheckProof::verify(
            &self.output_proof,
            preprocessing,
            generators,
            &commitment.read_write_memory,
            transcript,
        )?;
        TimestampValidityProof::verify(
            &mut self.timestamp_validity_proof,
            generators,
            &commitment.timestamp_range_check,
            &commitment.read_write_memory,
            transcript,
        )
    }
}
