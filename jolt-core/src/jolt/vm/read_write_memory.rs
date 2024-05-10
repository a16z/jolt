use crate::poly::field::JoltField;
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
    RAM_OPS_PER_INSTRUCTION, RAM_START_ADDRESS, REGISTER_COUNT, REG_OPS_PER_INSTRUCTION,
};
use common::rv_trace::{ELFInstruction, JoltDevice, MemoryLayout, MemoryOp, RV32IM};
use common::to_ram_address;

use super::{timestamp_range_check::TimestampValidityProof, JoltCommitments, JoltPolynomials};

pub trait RandomInstruction {
    fn random(index: usize, rng: &mut StdRng) -> Self;
}

impl RandomInstruction for ELFInstruction {
    fn random(index: usize, rng: &mut StdRng) -> Self {
        Self {
            address: to_ram_address(index) as u64,
            raw: rng.next_u32(),
            // Only `address` and `raw` are used in ReadWriteMemory; the rest don't matter
            opcode: RV32IM::ADD,
            rs1: None,
            rs2: None,
            rd: None,
            imm: None,
        }
    }
}

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
        ops[RS1] = MemoryOp::Read(rs1, memory[rs1 as usize]);

        let rs2 = rng.next_u64() % REGISTER_COUNT;
        ops[RS2] = MemoryOp::Read(rs2, memory[rs2 as usize]);

        // Don't write to the zero register
        let rd = rng.next_u64() % (REGISTER_COUNT - 1) + 1;
        // Registers are 32 bits
        let register_value = rng.next_u32() as u64;
        ops[RD] = MemoryOp::Write(rd, register_value);
        memory[rd as usize] = register_value;

        let ram_rng = rng.next_u32();
        if ram_rng % 3 == 0 {
            // LOAD
            let remapped_address =
                REGISTER_COUNT + rng.next_u64() % (max_memory_address as u64 - REGISTER_COUNT - 4);
            let ram_address = remapped_address - REGISTER_COUNT + RAM_START_ADDRESS;

            let load_rng = rng.next_u32();
            if load_rng % 3 == 0 {
                // LB
                ops[3] = MemoryOp::Read(ram_address, memory[remapped_address as usize]);
                for i in 1..4 {
                    ops[i + 3] = MemoryOp::noop_read();
                }
                for (i, flag) in load_store_flags.iter_mut().enumerate() {
                    flag.push(if i == 0 { 1 } else { 0 });
                }
            } else if load_rng % 3 == 1 {
                // LH
                for i in 0..2 {
                    ops[i + 3] = MemoryOp::Read(
                        ram_address + i as u64,
                        memory[i + remapped_address as usize],
                    );
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
                    ops[i + 3] = MemoryOp::Read(
                        ram_address + i as u64,
                        memory[i + remapped_address as usize],
                    );
                }
                for (i, flag) in load_store_flags.iter_mut().enumerate() {
                    flag.push(if i == 4 { 1 } else { 0 });
                }
            }
        } else if ram_rng % 3 == 1 {
            // STORE
            let remapped_address =
                REGISTER_COUNT + rng.next_u64() % (max_memory_address as u64 - REGISTER_COUNT - 4);
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
                    ops[i + 3] = MemoryOp::Write(ram_address + i as u64, ram_value);
                    memory[i + remapped_address as usize] = ram_value;
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
                    ops[i + 3] = MemoryOp::Write(ram_address + i as u64, ram_value);
                    memory[i + remapped_address as usize] = ram_value;
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
            + (BYTES_PER_INSTRUCTION as u64 - 1); // For RV32I, instructions occupy 4 bytes, so the max bytecode address is the max instruction address + 3

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
    pub v_init: DensePolynomial<F>,
    /// MLE of read/write addresses. For offline memory checking, each read is paired with a "virtual" write
    /// and vice versa, so the read addresses and write addresses are the same.
    pub a_ram: DensePolynomial<F>,
    /// MLE of the read values.
    pub v_read: [DensePolynomial<F>; MEMORY_OPS_PER_INSTRUCTION],
    /// MLE of the write values.
    pub v_write_rd: DensePolynomial<F>,
    pub v_write_ram: [DensePolynomial<F>; 4],
    /// MLE of the final memory state.
    pub v_final: DensePolynomial<F>,
    /// MLE of the read timestamps.
    pub t_read: [DensePolynomial<F>; MEMORY_OPS_PER_INSTRUCTION],
    /// MLE of the write timestamps.
    pub t_write_ram: [DensePolynomial<F>; 4],
    /// MLE of the final timestamps.
    pub t_final: DensePolynomial<F>,
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

impl<F: JoltField, C: CommitmentScheme<Field = F>> ReadWriteMemory<F, C> {
    #[tracing::instrument(skip_all, name = "ReadWriteMemory::new")]
    pub fn new(
        program_io: &JoltDevice,
        load_store_flags: &[DensePolynomial<F>],
        preprocessing: &ReadWriteMemoryPreprocessing,
        memory_trace: Vec<[MemoryOp; MEMORY_OPS_PER_INSTRUCTION]>,
    ) -> (Self, [Vec<u64>; MEMORY_OPS_PER_INSTRUCTION]) {
        assert!(program_io.inputs.len() <= program_io.memory_layout.max_input_size as usize);
        assert!(program_io.outputs.len() <= program_io.memory_layout.max_output_size as usize);

        let m = memory_trace.len();
        assert!(m.is_power_of_two());

        let max_trace_address = memory_trace
            .iter()
            .flat_map(|step| {
                step.iter().map(|op| match op {
                    MemoryOp::Read(a, _) => remap_address(*a, &program_io.memory_layout),
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

        let (memory_trace_reg, memory_trace_ram): (Vec<Vec<MemoryOp>>, Vec<Vec<MemoryOp>>) =
            memory_trace
                .into_par_iter()
                .map(|item| {
                    let (reg, ram) = item.split_at(3);
                    (reg.to_vec(), ram.to_vec())
                })
                .unzip();

        let reg_count = REGISTER_COUNT as usize;
        let mut v_final_reg = v_init[..reg_count].to_vec();
        let mut v_final_ram = v_init[reg_count..].to_vec();
        let mut t_final_reg = vec![0; reg_count];
        let mut t_final_ram = vec![0; memory_size - reg_count];

        let mut v_read_reg: [Vec<u64>; REG_OPS_PER_INSTRUCTION] =
            std::array::from_fn(|_| Vec::with_capacity(m));
        let mut v_read_ram: [Vec<u64>; RAM_OPS_PER_INSTRUCTION] =
            std::array::from_fn(|_| Vec::with_capacity(m));

        let mut t_read_reg: [Vec<u64>; REG_OPS_PER_INSTRUCTION] =
            std::array::from_fn(|_| Vec::with_capacity(m));
        let mut t_read_ram: [Vec<u64>; RAM_OPS_PER_INSTRUCTION] =
            std::array::from_fn(|_| Vec::with_capacity(m));

        // REG only
        let mut v_write_rd: Vec<u64> = Vec::with_capacity(m);
        // RAM only
        let mut a_ram: Vec<u64> = Vec::with_capacity(m);
        let mut v_write_ram: [Vec<u64>; 4] = std::array::from_fn(|_| Vec::with_capacity(m));
        let mut t_write_ram: [Vec<u64>; 4] = std::array::from_fn(|_| Vec::with_capacity(m));

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

                let lb_flag = &load_store_flags[0];
                let lh_flag = &load_store_flags[1];
                let sb_flag = &load_store_flags[2];
                let sh_flag = &load_store_flags[3];
                let sw_flag = &load_store_flags[4];

                for (i, step) in memory_trace_ram.iter().enumerate() {
                    let timestamp = i as u64;

                    #[allow(unused_assignments)]
                    let mut ram_word_address = 0;
                    let mut is_v_write_ram = false;

                    // Only the LB/SB/LH/SH/LW/SW instructions access ≥1 byte of RAM
                    if lb_flag[i].is_one()
                        || lh_flag[i].is_one()
                        || sb_flag[i].is_one()
                        || sh_flag[i].is_one()
                        || sw_flag[i].is_one()
                    {
                        match step[RAM_1_INDEX] {
                            MemoryOp::Read(a, v) => {
                                assert!(a >= program_io.memory_layout.input_start);
                                let remapped_a = remap_address(a, &program_io.memory_layout);
                                let remapped_a_index = remap_address_index(remapped_a);
                                debug_assert_eq!(v, v_final_ram[remapped_a_index]);

                                #[cfg(test)]
                                {
                                    r_tuples_ram.lock().unwrap().insert((
                                        remapped_a,
                                        v,
                                        t_final_ram[remapped_a_index],
                                    ));
                                    w_tuples_ram
                                        .lock()
                                        .unwrap()
                                        .insert((remapped_a, v, timestamp));
                                }

                                a_ram.push(remapped_a);
                                v_read_ram[RAM_1_INDEX].push(v);
                                t_read_ram[RAM_1_INDEX].push(t_final_ram[remapped_a_index]);
                                v_write_ram[0].push(v);
                                t_write_ram[0].push(timestamp);
                                t_final_ram[remapped_a_index] = timestamp;
                                ram_word_address = a;
                            }
                            MemoryOp::Write(a, v_new) => {
                                assert!(a >= program_io.memory_layout.input_start);
                                let remapped_a = remap_address(a, &program_io.memory_layout);
                                let remapped_a_index = remap_address_index(remapped_a);
                                let v_old = v_final_ram[remapped_a_index];

                                #[cfg(test)]
                                {
                                    r_tuples_ram.lock().unwrap().insert((
                                        remapped_a,
                                        v_old,
                                        t_final_ram[remapped_a_index],
                                    ));
                                    w_tuples_ram.lock().unwrap().insert((
                                        remapped_a,
                                        v_new,
                                        timestamp + 1,
                                    ));
                                }

                                a_ram.push(remapped_a);
                                v_read_ram[RAM_1_INDEX].push(v_old);
                                t_read_ram[RAM_1_INDEX].push(t_final_ram[remapped_a_index]);
                                v_write_ram[0].push(v_new);
                                t_write_ram[0].push(timestamp + 1);
                                v_final_ram[remapped_a_index] = v_new;
                                t_final_ram[remapped_a_index] = timestamp + 1;
                                ram_word_address = a;
                                is_v_write_ram = true;
                            }
                        };
                    } else {
                        a_ram.push(0);
                        for ram_byte_index in [RAM_1_INDEX, RAM_2_INDEX, RAM_3_INDEX, RAM_4_INDEX] {
                            match step[ram_byte_index] {
                                MemoryOp::Read(a, v) => {
                                    assert_eq!(a, 0);
                                    assert_eq!(v, 0);
                                }
                                MemoryOp::Write(a, v) => {
                                    assert_eq!(a, 0);
                                    assert_eq!(v, 0);
                                }
                            }
                            v_read_ram[ram_byte_index].push(0);
                            t_read_ram[ram_byte_index].push(0);
                        }
                        for v in v_write_ram.iter_mut() {
                            v.push(0);
                        }
                        for t in t_write_ram.iter_mut() {
                            t.push(0);
                        }
                        continue;
                    }

                    // Only the LH/SH/LW/SW instructions access ≥2 byte of RAM
                    if lh_flag[i].is_one() || sh_flag[i].is_one() || sw_flag[i].is_one() {
                        match step[RAM_2_INDEX] {
                            MemoryOp::Read(a, v) => {
                                assert!(!is_v_write_ram);
                                assert_eq!(a, ram_word_address + 1);
                                let remapped_a = remap_address(a, &program_io.memory_layout);
                                let remapped_a_index = remap_address_index(remapped_a);
                                debug_assert_eq!(v, v_final_ram[remapped_a_index]);

                                #[cfg(test)]
                                {
                                    r_tuples_ram.lock().unwrap().insert((
                                        remapped_a,
                                        v,
                                        t_final_ram[remapped_a_index],
                                    ));
                                    w_tuples_ram
                                        .lock()
                                        .unwrap()
                                        .insert((remapped_a, v, timestamp));
                                }

                                v_read_ram[RAM_2_INDEX].push(v);
                                t_read_ram[RAM_2_INDEX].push(t_final_ram[remapped_a_index]);
                                v_write_ram[1].push(v);
                                t_write_ram[1].push(timestamp);
                                t_final_ram[remapped_a_index] = timestamp;
                            }
                            MemoryOp::Write(a, v_new) => {
                                assert!(is_v_write_ram);
                                assert_eq!(a, ram_word_address + 1);
                                let remapped_a = remap_address(a, &program_io.memory_layout);
                                let remapped_a_index = remap_address_index(remapped_a);
                                let v_old = v_final_ram[remapped_a_index];

                                #[cfg(test)]
                                {
                                    r_tuples_ram.lock().unwrap().insert((
                                        remapped_a,
                                        v_old,
                                        t_final_ram[remapped_a_index],
                                    ));
                                    w_tuples_ram.lock().unwrap().insert((
                                        remapped_a,
                                        v_new,
                                        timestamp + 1,
                                    ));
                                }

                                v_read_ram[RAM_2_INDEX].push(v_old);
                                t_read_ram[RAM_2_INDEX].push(t_final_ram[remapped_a_index]);
                                v_write_ram[1].push(v_new);
                                t_write_ram[1].push(timestamp + 1);
                                v_final_ram[remapped_a_index] = v_new;
                                t_final_ram[remapped_a_index] = timestamp + 1;
                            }
                        };
                    } else {
                        for ram_byte_index in [RAM_2_INDEX, RAM_3_INDEX, RAM_4_INDEX] {
                            match step[ram_byte_index] {
                                MemoryOp::Read(a, v) => {
                                    assert_eq!(a, 0);
                                    assert_eq!(v, 0);
                                }
                                MemoryOp::Write(a, v) => {
                                    assert_eq!(a, 0);
                                    assert_eq!(v, 0);
                                }
                            }
                            v_read_ram[ram_byte_index].push(0);
                            t_read_ram[ram_byte_index].push(0);
                        }
                        for v in v_write_ram[1..].iter_mut() {
                            v.push(0);
                        }
                        for t in t_write_ram[1..].iter_mut() {
                            t.push(0);
                        }
                        continue;
                    }

                    // Only the LW/SW instructions access ≥3 byte of RAM
                    // Both LW and SW are represented by `sw_flag` for the purpose of lookups
                    if sw_flag[i].is_one() {
                        match step[RAM_3_INDEX] {
                            MemoryOp::Read(a, v) => {
                                assert!(!is_v_write_ram);
                                assert_eq!(a, ram_word_address + 2);
                                let remapped_a = remap_address(a, &program_io.memory_layout);
                                let remapped_a_index = remap_address_index(remapped_a);
                                debug_assert_eq!(v, v_final_ram[remapped_a_index]);

                                #[cfg(test)]
                                {
                                    r_tuples_ram.lock().unwrap().insert((
                                        remapped_a,
                                        v,
                                        t_final_ram[remapped_a_index],
                                    ));
                                    w_tuples_ram
                                        .lock()
                                        .unwrap()
                                        .insert((remapped_a, v, timestamp));
                                }

                                v_read_ram[RAM_3_INDEX].push(v);
                                t_read_ram[RAM_3_INDEX].push(t_final_ram[remapped_a_index]);
                                v_write_ram[2].push(v);
                                t_write_ram[2].push(timestamp);
                                t_final_ram[remapped_a_index] = timestamp;
                            }
                            MemoryOp::Write(a, v_new) => {
                                assert!(is_v_write_ram);
                                assert_eq!(a, ram_word_address + 2);
                                let remapped_a = remap_address(a, &program_io.memory_layout);
                                let remapped_a_index = remap_address_index(remapped_a);
                                let v_old = v_final_ram[remapped_a_index];

                                #[cfg(test)]
                                {
                                    r_tuples_ram.lock().unwrap().insert((
                                        remapped_a,
                                        v_old,
                                        t_final_ram[remapped_a_index],
                                    ));
                                    w_tuples_ram.lock().unwrap().insert((
                                        remapped_a,
                                        v_new,
                                        timestamp + 1,
                                    ));
                                }

                                v_read_ram[RAM_3_INDEX].push(v_old);
                                t_read_ram[RAM_3_INDEX].push(t_final_ram[remapped_a_index]);
                                v_write_ram[2].push(v_new);
                                t_write_ram[2].push(timestamp + 1);
                                v_final_ram[remapped_a_index] = v_new;
                                t_final_ram[remapped_a_index] = timestamp + 1;
                            }
                        };
                        match step[RAM_4_INDEX] {
                            MemoryOp::Read(a, v) => {
                                assert!(!is_v_write_ram);
                                assert_eq!(a, ram_word_address + 3);
                                let remapped_a = remap_address(a, &program_io.memory_layout);
                                let remapped_a_index = remap_address_index(remapped_a);
                                debug_assert_eq!(v, v_final_ram[remapped_a_index]);

                                #[cfg(test)]
                                {
                                    r_tuples_ram.lock().unwrap().insert((
                                        remapped_a,
                                        v,
                                        t_final_ram[remapped_a_index],
                                    ));
                                    w_tuples_ram
                                        .lock()
                                        .unwrap()
                                        .insert((remapped_a, v, timestamp));
                                }

                                v_read_ram[RAM_4_INDEX].push(v);
                                t_read_ram[RAM_4_INDEX].push(t_final_ram[remapped_a_index]);
                                v_write_ram[3].push(v);
                                t_write_ram[3].push(timestamp);
                                t_final_ram[remapped_a_index] = timestamp;
                            }
                            MemoryOp::Write(a, v_new) => {
                                assert!(is_v_write_ram);
                                assert_eq!(a, ram_word_address + 3);
                                let remapped_a = remap_address(a, &program_io.memory_layout);
                                let remapped_a_index = remap_address_index(remapped_a);
                                let v_old = v_final_ram[remapped_a_index];

                                #[cfg(test)]
                                {
                                    r_tuples_ram.lock().unwrap().insert((
                                        remapped_a,
                                        v_old,
                                        t_final_ram[remapped_a_index],
                                    ));
                                    w_tuples_ram.lock().unwrap().insert((
                                        remapped_a,
                                        v_new,
                                        timestamp + 1,
                                    ));
                                }

                                v_read_ram[RAM_4_INDEX].push(v_old);
                                t_read_ram[RAM_4_INDEX].push(t_final_ram[remapped_a_index]);
                                v_write_ram[3].push(v_new);
                                t_write_ram[3].push(timestamp + 1);
                                v_final_ram[remapped_a_index] = v_new;
                                t_final_ram[remapped_a_index] = timestamp + 1;
                            }
                        };
                    } else {
                        for ram_byte_index in [RAM_3_INDEX, RAM_4_INDEX] {
                            match step[ram_byte_index] {
                                MemoryOp::Read(a, v) => {
                                    assert_eq!(a, 0);
                                    assert_eq!(v, 0);
                                }
                                MemoryOp::Write(a, v) => {
                                    assert_eq!(a, 0);
                                    assert_eq!(v, 0);
                                }
                            }
                            v_read_ram[ram_byte_index].push(0);
                            t_read_ram[ram_byte_index].push(0);
                        }
                        for v in v_write_ram[2..].iter_mut() {
                            v.push(0);
                        }
                        for t in t_write_ram[2..].iter_mut() {
                            t.push(0);
                        }
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
                        MemoryOp::Read(a, v) => {
                            assert!(a < REGISTER_COUNT);
                            debug_assert_eq!(v, v_final_reg[a as usize]);

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
                    };

                    match step[RS2] {
                        MemoryOp::Read(a, v) => {
                            assert!(a < REGISTER_COUNT);
                            debug_assert_eq!(v, v_final_reg[a as usize]);

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
                            panic!("Unexpected rs2 MemoryOp::Write({}, {})", a, v)
                        }
                    };

                    match step[RD] {
                        MemoryOp::Read(a, v) => {
                            panic!("Unexpected rd MemoryOp::Read({}, {})", a, v)
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

        let (mut v_final_reg, mut t_final_reg, v_read_reg, t_read_reg, v_write_rd) = result.1;
        let (v_final_ram, t_final_ram, v_read_ram, t_read_ram, v_write_ram, t_write_ram, a_ram) =
            result.0;

        let v_final = {
            v_final_reg.extend(v_final_ram);
            v_final_reg
        };
        let t_final = {
            t_final_reg.extend(t_final_ram);
            t_final_reg
        };
        let v_read: [Vec<u64>; MEMORY_OPS_PER_INSTRUCTION] =
            merge_vec_array(v_read_reg, v_read_ram, m);
        let t_read: [Vec<u64>; MEMORY_OPS_PER_INSTRUCTION] =
            merge_vec_array(t_read_reg, t_read_ram, m);

        #[cfg(test)]
        {
            let read_tuples = Arc::try_unwrap(read_tuples).unwrap().into_inner().unwrap();
            let write_tuples = Arc::try_unwrap(write_tuples).unwrap().into_inner().unwrap();

            let mut final_tuples: HashSet<(u64, u64, u64)> = HashSet::new();
            for (a, (v, t)) in v_final.iter().zip(t_final.iter()).enumerate() {
                final_tuples.insert((a as u64, *v, *t));
            }

            let init_write: HashSet<_> = init_tuples.union(&write_tuples).collect();
            let read_final: HashSet<_> = read_tuples.union(&final_tuples).collect();
            let set_difference: Vec<_> = init_write.symmetric_difference(&read_final).collect();
            assert_eq!(set_difference.len(), 0);
        }

        let (
            [a_ram, v_write_rd, v_init, v_final, t_final],
            v_read,
            v_write_ram,
            t_read_polys,
            t_write_ram,
        ): (
            [DensePolynomial<F>; 5],
            [DensePolynomial<F>; MEMORY_OPS_PER_INSTRUCTION],
            [DensePolynomial<F>; 4],
            [DensePolynomial<F>; MEMORY_OPS_PER_INSTRUCTION],
            [DensePolynomial<F>; 4],
        ) = common::par_join_5!(
            || map_to_polys(&[a_ram, v_write_rd, v_init, v_final, t_final]),
            || map_to_polys(&v_read),
            || map_to_polys(&v_write_ram),
            || map_to_polys(&t_read),
            || map_to_polys(&t_write_ram)
        );
        (
            Self {
                _group: PhantomData,
                memory_size,
                v_init,
                a_ram,
                v_read,
                v_write_rd,
                v_write_ram,
                v_final,
                t_read: t_read_polys,
                t_write_ram,
                t_final,
            },
            t_read,
        )
    }

    #[tracing::instrument(skip_all, name = "ReadWriteMemory::get_polys_r1cs")]
    pub fn get_polys_r1cs<'a>(&'a self) -> (&'a [F], Vec<&'a F>, Vec<&'a F>) {
        let (a_polys, (v_read_polys, v_write_polys)) = rayon::join(
            || self.a_ram.evals_ref(),
            || {
                rayon::join(
                    || {
                        self.v_read
                            .par_iter()
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
        let t_read_write_len = (max_trace_length * MEMORY_OPS_PER_INSTRUCTION).next_power_of_two();
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
    pub v_final_commitment: C::Commitment,
    pub t_final_commitment: C::Commitment,
}

impl<C: CommitmentScheme> AppendToTranscript for MemoryCommitment<C> {
    fn append_to_transcript(&self, label: &'static [u8], transcript: &mut ProofTranscript) {
        transcript.append_message(label, b"MemoryCommitment_begin");
        for commitment in &self.trace_commitments {
            commitment.append_to_transcript(b"trace_commit", transcript);
        }
        self.v_final_commitment
            .append_to_transcript(b"v_final_commit", transcript);
        self.t_final_commitment
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
    pub a_read_write_opening: [F; 4],
    /// Evaluation of the v_read polynomial at the opening point.
    pub v_read_opening: [F; MEMORY_OPS_PER_INSTRUCTION],
    /// Evaluation of the v_write polynomial at the opening point.
    pub v_write_opening: [F; 5],
    /// Evaluation of the t_read polynomial at the opening point.
    pub t_read_opening: [F; MEMORY_OPS_PER_INSTRUCTION],
    /// Evaluation of the t_write_ram polynomial at the opening point.
    pub t_write_ram_opening: [F; 4],
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
            &polynomials.bytecode.v_read_write[1],
            &polynomials.bytecode.v_read_write[2],
            &polynomials.bytecode.v_read_write[3],
            &polynomials.read_write_memory.a_ram,
        ]
        .into_par_iter()
        .chain(polynomials.read_write_memory.v_read.par_iter())
        .chain([&polynomials.read_write_memory.v_write_rd].into_par_iter())
        .chain(polynomials.read_write_memory.v_write_ram.par_iter())
        .chain(polynomials.read_write_memory.t_read.par_iter())
        .chain(polynomials.read_write_memory.t_write_ram.par_iter())
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

    #[tracing::instrument(skip_all, name = "MemoryReadWriteOpenings::prove_openings")]
    fn prove_openings(
        polynomials: &JoltPolynomials<F, C>,
        opening_point: &[F],
        openings: &Self,
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        let read_write_polys = [
            &polynomials.bytecode.v_read_write[1],
            &polynomials.bytecode.v_read_write[2],
            &polynomials.bytecode.v_read_write[3],
            &polynomials.read_write_memory.a_ram,
        ]
        .into_iter()
        .chain(polynomials.read_write_memory.v_read.iter())
        .chain([&polynomials.read_write_memory.v_write_rd].into_iter())
        .chain(polynomials.read_write_memory.v_write_ram.iter())
        .chain(polynomials.read_write_memory.t_read.iter())
        .chain(polynomials.read_write_memory.t_write_ram.iter())
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
            &commitment.bytecode.trace_commitments[3..6]
                .iter()
                .chain(commitment.read_write_memory.trace_commitments.iter())
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
    a_init_final: Option<F>,
    /// Evaluation of the v_init polynomial at the opening point. Computed by the verifier in `compute_verifier_openings`.
    v_init: Option<F>,
    /// Evaluation of the v_final polynomial at the opening point.
    v_final: F,
    /// Evaluation of the t_final polynomial at the opening point.
    t_final: F,
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
        let (v_final, t_final) = rayon::join(
            || polynomials.read_write_memory.v_final.evaluate_at_chi(&chis),
            || polynomials.read_write_memory.t_final.evaluate_at_chi(&chis),
        );

        Self {
            a_init_final: None,
            v_init: None,
            v_final,
            t_final,
        }
    }

    #[tracing::instrument(skip_all, name = "MemoryInitFinalOpenings::prove_openings")]
    fn prove_openings(
        polynomials: &JoltPolynomials<F, C>,
        opening_point: &[F],
        openings: &Self,
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        let v_t_opening_proof = C::batch_prove(
            &[
                &polynomials.read_write_memory.v_final,
                &polynomials.read_write_memory.t_final,
            ],
            opening_point,
            &[openings.v_final, openings.t_final],
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
        self.a_init_final =
            Some(IdentityPolynomial::new(opening_point.len()).evaluate(opening_point));

        let memory_layout = &preprocessing.program_io.as_ref().unwrap().memory_layout;

        // TODO(moodlezoup): Compute opening without instantiating v_init polynomial itself
        let memory_size = opening_point.len().pow2();
        let mut v_init: Vec<u64> = vec![0; memory_size];
        // Copy bytecode
        let mut v_init_index = memory_address_to_witness_index(
            preprocessing.min_bytecode_address,
            memory_layout.ram_witness_offset,
        );
        for byte in preprocessing.bytecode_bytes.iter() {
            v_init[v_init_index] = *byte as u64;
            v_init_index += 1;
        }
        // Copy input bytes
        v_init_index = memory_address_to_witness_index(
            memory_layout.input_start,
            memory_layout.ram_witness_offset,
        );
        for byte in preprocessing.program_io.as_ref().unwrap().inputs.iter() {
            v_init[v_init_index] = *byte as u64;
            v_init_index += 1;
        }

        self.v_init = Some(DensePolynomial::from_u64(&v_init).evaluate(opening_point));
    }

    fn verify_openings(
        &self,
        generators: &C::Setup,
        opening_proof: &Self::Proof,
        commitment: &JoltCommitments<C>,
        opening_point: &[F],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        C::batch_verify(
            &opening_proof.v_t_opening_proof,
            generators,
            opening_point,
            &[self.v_final, self.t_final],
            &[
                &commitment.read_write_memory.v_final_commitment,
                &commitment.read_write_memory.t_final_commitment,
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
        t * gamma.square() + v * *gamma + a - tau
    }

    #[tracing::instrument(skip_all, name = "ReadWriteMemory::compute_leaves")]
    fn compute_leaves(
        _: &Self::Preprocessing,
        polynomials: &JoltPolynomials<F, C>,
        gamma: &F,
        tau: &F,
    ) -> (Vec<Vec<F>>, Vec<Vec<F>>) {
        let gamma_squared = gamma.square();
        let num_ops = polynomials.read_write_memory.a_ram.len();

        let read_write_leaves = (0..MEMORY_OPS_PER_INSTRUCTION)
            .into_par_iter()
            .flat_map(|i| {
                let read_fingerprints = (0..num_ops)
                    .into_par_iter()
                    .map(|j| {
                        let a = match i {
                            RS1 => polynomials.bytecode.v_read_write[2][j],
                            RS2 => polynomials.bytecode.v_read_write[3][j],
                            RD => polynomials.bytecode.v_read_write[1][j],
                            _ => {
                                polynomials.read_write_memory.a_ram[j]
                                    + F::from_u64((i - RAM_1) as u64).unwrap()
                            }
                        };
                        polynomials.read_write_memory.t_read[i][j] * gamma_squared
                            + mul_0_optimized(&polynomials.read_write_memory.v_read[i][j], gamma)
                            + a
                            - *tau
                    })
                    .collect();
                let v_write = match i {
                    RS1 => &polynomials.read_write_memory.v_read[0], // rs1
                    RS2 => &polynomials.read_write_memory.v_read[1], // rs2
                    RD => &polynomials.read_write_memory.v_write_rd, // rd
                    _ => &polynomials.read_write_memory.v_write_ram[i - 3], // RAM
                };
                let write_fingerprints = (0..num_ops)
                    .into_par_iter()
                    .map(|j| match i {
                        RS1 => {
                            F::from_u64(j as u64).unwrap() * gamma_squared
                                + mul_0_optimized(&v_write[j], gamma)
                                + polynomials.bytecode.v_read_write[2][j]
                                - *tau
                        }
                        RS2 => {
                            F::from_u64(j as u64).unwrap() * gamma_squared
                                + mul_0_optimized(&v_write[j], gamma)
                                + polynomials.bytecode.v_read_write[3][j]
                                - *tau
                        }
                        RD => {
                            F::from_u64(j as u64 + 1).unwrap() * gamma_squared
                                + mul_0_optimized(&v_write[j], gamma)
                                + polynomials.bytecode.v_read_write[1][j]
                                - *tau
                        }
                        _ => {
                            polynomials.read_write_memory.t_write_ram[i - RAM_1][j] * gamma_squared
                                + mul_0_optimized(&v_write[j], gamma)
                                + polynomials.read_write_memory.a_ram[j]
                                + F::from_u64((i - RAM_1) as u64).unwrap()
                                - *tau
                        }
                    })
                    .collect();
                [read_fingerprints, write_fingerprints]
            })
            .collect();

        let init_fingerprints = (0..polynomials.read_write_memory.memory_size)
            .into_par_iter()
            .map(|i| /* 0 * gamma^2 + */ mul_0_optimized(&polynomials.read_write_memory.v_init[i], gamma) + F::from_u64(i as u64).unwrap() - *tau)
            .collect();
        let final_fingerprints = (0..polynomials.read_write_memory.memory_size)
            .into_par_iter()
            .map(|i| {
                mul_0_optimized(&polynomials.read_write_memory.t_final[i], &gamma_squared)
                    + mul_0_optimized(&polynomials.read_write_memory.v_final[i], gamma)
                    + F::from_u64(i as u64).unwrap()
                    - *tau
            })
            .collect();

        (
            read_write_leaves,
            vec![init_fingerprints, final_fingerprints],
        )
    }

    fn uninterleave_hashes(
        _preprocessing: &Self::Preprocessing,
        read_write_hashes: Vec<F>,
        init_final_hashes: Vec<F>,
    ) -> MultisetHashes<F> {
        assert_eq!(read_write_hashes.len(), 2 * MEMORY_OPS_PER_INSTRUCTION);
        let mut read_hashes = Vec::with_capacity(MEMORY_OPS_PER_INSTRUCTION);
        let mut write_hashes = Vec::with_capacity(MEMORY_OPS_PER_INSTRUCTION);
        for i in 0..MEMORY_OPS_PER_INSTRUCTION {
            read_hashes.push(read_write_hashes[2 * i]);
            write_hashes.push(read_write_hashes[2 * i + 1]);
        }

        assert_eq!(init_final_hashes.len(), 2);
        let init_hash = init_final_hashes[0];
        let final_hash = init_final_hashes[1];

        MultisetHashes {
            read_hashes,
            write_hashes,
            init_hashes: vec![init_hash],
            final_hashes: vec![final_hash],
        }
    }

    fn check_multiset_equality(
        _preprocessing: &Self::Preprocessing,
        multiset_hashes: &MultisetHashes<F>,
    ) {
        assert_eq!(
            multiset_hashes.read_hashes.len(),
            MEMORY_OPS_PER_INSTRUCTION
        );
        assert_eq!(
            multiset_hashes.write_hashes.len(),
            MEMORY_OPS_PER_INSTRUCTION
        );
        assert_eq!(multiset_hashes.init_hashes.len(), 1);
        assert_eq!(multiset_hashes.final_hashes.len(), 1);

        let read_hash: F = multiset_hashes.read_hashes.iter().product();
        let write_hash: F = multiset_hashes.write_hashes.iter().product();
        let init_hash = multiset_hashes.init_hashes[0];
        let final_hash = multiset_hashes.final_hashes[0];

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
        (0..MEMORY_OPS_PER_INSTRUCTION)
            .map(|i| {
                let a = match i {
                    RD => openings.a_read_write_opening[0],
                    RS1 => openings.a_read_write_opening[1],
                    RS2 => openings.a_read_write_opening[2],
                    _ => {
                        openings.a_read_write_opening[3] + F::from_u64((i - RAM_1) as u64).unwrap()
                    }
                };
                (a, openings.v_read_opening[i], openings.t_read_opening[i])
            })
            .collect()
    }
    fn write_tuples(
        &_: &Self::Preprocessing,
        openings: &Self::ReadWriteOpenings,
    ) -> Vec<Self::MemoryTuple> {
        (0..MEMORY_OPS_PER_INSTRUCTION)
            .map(|i| {
                let a = match i {
                    RD => openings.a_read_write_opening[0],
                    RS1 => openings.a_read_write_opening[1],
                    RS2 => openings.a_read_write_opening[2],
                    _ => {
                        openings.a_read_write_opening[3] + F::from_u64((i - RAM_1) as u64).unwrap()
                    }
                };
                let v = if i == RS1 || i == RS2 {
                    // For rs1 and rs2, v_write = v_read
                    openings.v_read_opening[i]
                } else {
                    openings.v_write_opening[i - 2]
                };
                let t = if i == RS1 || i == RS2 {
                    openings.identity_poly_opening.unwrap()
                } else if i == RD {
                    openings.identity_poly_opening.unwrap() + F::one()
                } else {
                    openings.t_write_ram_opening[i - RAM_1]
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
            openings.a_init_final.unwrap(),
            openings.v_init.unwrap(),
            F::zero(),
        )]
    }
    fn final_tuples(
        &_: &Self::Preprocessing,
        openings: &Self::InitFinalOpenings,
    ) -> Vec<Self::MemoryTuple> {
        vec![(
            openings.a_init_final.unwrap(),
            openings.v_final,
            openings.t_final,
        )]
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
    opening: F,
    /// Hyrax opening proof of the v_final opening
    opening_proof: C::Proof,
}

impl<F, C> OutputSumcheckProof<F, C>
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
{
    fn prove_outputs(
        polynomials: &ReadWriteMemory<F, C>,
        program_io: &JoltDevice,
        transcript: &mut ProofTranscript,
    ) -> Self {
        let num_rounds = polynomials.memory_size.log_2();
        let r_eq = transcript.challenge_vector(b"output_sumcheck", num_rounds);
        let eq: DensePolynomial<F> = DensePolynomial::new(EqPolynomial::evals(&r_eq));

        let io_witness_range: Vec<_> = (0..polynomials.memory_size as u64)
            .map(|i| {
                if i >= program_io.memory_layout.input_start
                    && i < program_io.memory_layout.ram_witness_offset
                {
                    F::one()
                } else {
                    F::zero()
                }
            })
            .collect();

        let mut v_io: Vec<u64> = vec![0; polynomials.memory_size];
        // Copy input bytes
        let mut input_index = memory_address_to_witness_index(
            program_io.memory_layout.input_start,
            program_io.memory_layout.ram_witness_offset,
        );
        for byte in program_io.inputs.iter() {
            v_io[input_index] = *byte as u64;
            input_index += 1;
        }
        // Copy output bytes
        let mut output_index = memory_address_to_witness_index(
            program_io.memory_layout.output_start,
            program_io.memory_layout.ram_witness_offset,
        );
        for byte in program_io.outputs.iter() {
            v_io[output_index] = *byte as u64;
            output_index += 1;
        }
        // Copy panic bit
        v_io[memory_address_to_witness_index(
            program_io.memory_layout.panic,
            program_io.memory_layout.ram_witness_offset,
        )] = program_io.panic as u64;

        let mut sumcheck_polys = vec![
            eq,
            DensePolynomial::new(io_witness_range),
            polynomials.v_final.clone(),
            DensePolynomial::from_u64(&v_io),
        ];

        // eq * io_witness_range * (v_final - v_io)
        let output_check_fn = |vals: &[F]| -> F { vals[0] * vals[1] * (vals[2] - vals[3]) };

        let (sumcheck_proof, r_sumcheck, sumcheck_openings) =
            SumcheckInstanceProof::<F>::prove_arbitrary::<_>(
                &F::zero(),
                num_rounds,
                &mut sumcheck_polys,
                output_check_fn,
                3,
                transcript,
            );

        let sumcheck_opening_proof = C::prove(&polynomials.v_final, &r_sumcheck, transcript);

        Self {
            num_rounds,
            sumcheck_proof,
            opening: sumcheck_openings[2], // only need v_final; verifier computes the rest on its own
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

        let (sumcheck_claim, r_sumcheck) =
            proof
                .sumcheck_proof
                .verify(F::zero(), proof.num_rounds, 3, transcript)?;

        let eq_eval = EqPolynomial::new(r_eq.to_vec()).evaluate(&r_sumcheck);

        let memory_layout = &preprocessing.program_io.as_ref().unwrap().memory_layout;

        let nonzero_memory_size = memory_layout.ram_witness_offset as usize;
        let log_nonzero_memory_size = nonzero_memory_size.log_2();
        assert!(
            nonzero_memory_size.is_power_of_two(),
            "Ram witness offset must be a power of two"
        );

        let io_witness_range: Vec<_> = (0..nonzero_memory_size as u64)
            .map(|i| {
                if i >= memory_layout.input_start {
                    F::one()
                } else {
                    F::zero()
                }
            })
            .collect();
        let mut io_witness_range_eval = DensePolynomial::new(io_witness_range)
            .evaluate(&r_sumcheck[0..log_nonzero_memory_size]);

        let r_prod: F = r_sumcheck[log_nonzero_memory_size..].iter().product();
        io_witness_range_eval *= r_prod;

        let mut v_io: Vec<u64> = vec![0; nonzero_memory_size];
        // Copy input bytes
        let mut input_index = memory_address_to_witness_index(
            memory_layout.input_start,
            memory_layout.ram_witness_offset,
        );
        for byte in preprocessing.program_io.as_ref().unwrap().inputs.iter() {
            v_io[input_index] = *byte as u64;
            input_index += 1;
        }
        // Copy output bytes
        let mut output_index = memory_address_to_witness_index(
            memory_layout.output_start,
            memory_layout.ram_witness_offset,
        );
        for byte in preprocessing.program_io.as_ref().unwrap().outputs.iter() {
            v_io[output_index] = *byte as u64;
            output_index += 1;
        }
        // Copy panic bit
        v_io[memory_address_to_witness_index(
            memory_layout.panic,
            memory_layout.ram_witness_offset,
        )] = preprocessing.program_io.as_ref().unwrap().panic as u64;
        let mut v_io_eval =
            DensePolynomial::from_u64(&v_io).evaluate(&r_sumcheck[..log_nonzero_memory_size]);
        v_io_eval *= r_prod;

        assert_eq!(
            eq_eval * io_witness_range_eval * (proof.opening - v_io_eval),
            sumcheck_claim,
            "Output sumcheck check failed."
        );

        C::verify(
            &proof.opening_proof,
            generators,
            transcript,
            &r_sumcheck,
            &proof.opening,
            &commitment.v_final_commitment,
        )
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
        preprocessing: &ReadWriteMemoryPreprocessing,
        polynomials: &JoltPolynomials<F, C>,
        program_io: &JoltDevice,
        transcript: &mut ProofTranscript,
    ) -> Self {
        let memory_checking_proof =
            ReadWriteMemoryProof::prove_memory_checking(preprocessing, polynomials, transcript);

        let output_proof = OutputSumcheckProof::prove_outputs(
            &polynomials.read_write_memory,
            program_io,
            transcript,
        );

        let timestamp_validity_proof = TimestampValidityProof::prove(
            &polynomials.timestamp_range_check,
            &polynomials.read_write_memory.t_read,
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
