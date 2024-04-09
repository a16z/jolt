use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use merlin::Transcript;
use rand::rngs::StdRng;
use rand_core::RngCore;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
#[cfg(test)]
use std::collections::HashSet;
use std::marker::PhantomData;

use crate::utils::transcript::AppendToTranscript;
use crate::{
    lasso::memory_checking::{
        MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier, MultisetHashes,
        NoPreprocessing,
    },
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        hyrax::{matrix_dimensions, BatchedHyraxOpeningProof, HyraxCommitment, HyraxOpeningProof},
        identity_poly::IdentityPolynomial,
        pedersen::PedersenGenerators,
        structured_poly::StructuredOpeningProof,
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{errors::ProofVerifyError, math::Math, mul_0_optimized, transcript::ProofTranscript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::{
    memory_address_to_witness_index, BYTES_PER_INSTRUCTION, INPUT_START_ADDRESS, MAX_INPUT_SIZE,
    MAX_OUTPUT_SIZE, MEMORY_OPS_PER_INSTRUCTION, NUM_R1CS_POLYS, OUTPUT_START_ADDRESS,
    PANIC_ADDRESS, RAM_START_ADDRESS, RAM_WITNESS_OFFSET, REGISTER_COUNT,
};
use common::rv_trace::{ELFInstruction, JoltDevice, MemoryOp, RV32IM};
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

pub fn random_memory_trace<F: PrimeField>(
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

fn remap_address(a: u64) -> u64 {
    if a >= INPUT_START_ADDRESS {
        memory_address_to_witness_index(a) as u64
    } else if a < REGISTER_COUNT {
        // If a < REGISTER_COUNT, it is one of the registers and doesn't
        // need to be remapped
        a
    } else {
        panic!("Unexpected address {}", a)
    }
}

const RS1: usize = 0;
const RS2: usize = 1;
const RD: usize = 2;
const RAM_1: usize = 3;
const RAM_2: usize = 4;
const RAM_3: usize = 5;
const RAM_4: usize = 6;

pub struct ReadWriteMemory<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    _group: PhantomData<G>,
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

fn map_to_polys<F: PrimeField, const N: usize>(vals: &[Vec<u64>; N]) -> [DensePolynomial<F>; N] {
    vals.par_iter()
        .map(|vals| DensePolynomial::from_u64(vals))
        .collect::<Vec<DensePolynomial<F>>>()
        .try_into()
        .unwrap()
}

impl<F: PrimeField, G: CurveGroup<ScalarField = F>> ReadWriteMemory<F, G> {
    #[tracing::instrument(skip_all, name = "ReadWriteMemory::new")]
    pub fn new(
        program_io: &JoltDevice,
        load_store_flags: &[DensePolynomial<F>],
        preprocessing: &ReadWriteMemoryPreprocessing,
        memory_trace: Vec<[MemoryOp; MEMORY_OPS_PER_INSTRUCTION]>,
    ) -> (Self, [Vec<u64>; MEMORY_OPS_PER_INSTRUCTION]) {
        let lb_flag = &load_store_flags[0];
        let lh_flag = &load_store_flags[1];
        let sb_flag = &load_store_flags[2];
        let sh_flag = &load_store_flags[3];
        let sw_flag = &load_store_flags[4];

        assert!(program_io.inputs.len() <= MAX_INPUT_SIZE as usize);
        assert!(program_io.outputs.len() <= MAX_OUTPUT_SIZE as usize);

        let m = memory_trace.len();
        assert!(m.is_power_of_two());

        let max_trace_address = memory_trace
            .iter()
            .flat_map(|step| {
                step.iter().map(|op| match op {
                    MemoryOp::Read(a, _) => remap_address(*a),
                    MemoryOp::Write(a, _) => remap_address(*a),
                })
            })
            .max()
            .unwrap_or(0);

        let memory_size = (RAM_WITNESS_OFFSET + max_trace_address).next_power_of_two() as usize;
        let mut v_init: Vec<u64> = vec![0; memory_size];
        // Copy bytecode
        let mut v_init_index = memory_address_to_witness_index(preprocessing.min_bytecode_address);
        for byte in preprocessing.bytecode_bytes.iter() {
            v_init[v_init_index] = *byte as u64;
            v_init_index += 1;
        }
        // Copy input bytes
        v_init_index = memory_address_to_witness_index(INPUT_START_ADDRESS);
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

        let mut a_ram: Vec<u64> = Vec::with_capacity(m);

        let mut v_read: [Vec<u64>; MEMORY_OPS_PER_INSTRUCTION] =
            std::array::from_fn(|_| Vec::with_capacity(m));
        let mut v_write_rd: Vec<u64> = Vec::with_capacity(m);
        let mut v_write_ram: [Vec<u64>; 4] = std::array::from_fn(|_| Vec::with_capacity(m));

        let mut t_read: [Vec<u64>; MEMORY_OPS_PER_INSTRUCTION] =
            std::array::from_fn(|_| Vec::with_capacity(m));
        let mut t_write_ram: [Vec<u64>; 4] = std::array::from_fn(|_| Vec::with_capacity(m));

        let mut v_final: Vec<u64> = v_init.clone();
        let mut t_final: Vec<u64> = vec![0; memory_size];

        #[cfg(test)]
        let mut read_tuples: HashSet<(u64, u64, u64)> = HashSet::new();
        #[cfg(test)]
        let mut write_tuples: HashSet<(u64, u64, u64)> = HashSet::new();

        let mut timestamp: u64 = 0;
        let span = tracing::span!(tracing::Level::DEBUG, "memory_trace_processing");
        let _enter = span.enter();
        for step in memory_trace {
            match step[RS1] {
                MemoryOp::Read(a, v) => {
                    assert!(a < REGISTER_COUNT);
                    debug_assert_eq!(v, v_final[a as usize]);

                    #[cfg(test)]
                    {
                        read_tuples.insert((a, v, t_final[a as usize]));
                        write_tuples.insert((a, v, timestamp));
                    }

                    v_read[RS1].push(v);
                    t_read[RS1].push(t_final[a as usize]);
                    t_final[a as usize] = timestamp;
                }
                MemoryOp::Write(a, v) => {
                    panic!("Unexpected rs1 MemoryOp::Write({}, {})", a, v);
                }
            };

            match step[RS2] {
                MemoryOp::Read(a, v) => {
                    assert!(a < REGISTER_COUNT);
                    debug_assert_eq!(v, v_final[a as usize]);

                    #[cfg(test)]
                    {
                        read_tuples.insert((a, v, t_final[a as usize]));
                        write_tuples.insert((a, v, timestamp));
                    }

                    v_read[RS2].push(v);
                    t_read[RS2].push(t_final[a as usize]);
                    t_final[a as usize] = timestamp;
                }
                MemoryOp::Write(a, v) => panic!("Unexpected rs2 MemoryOp::Write({}, {})", a, v),
            };

            match step[RD] {
                MemoryOp::Read(a, v) => panic!("Unexpected rd MemoryOp::Read({}, {})", a, v),
                MemoryOp::Write(a, v_new) => {
                    assert!(a < REGISTER_COUNT);
                    let v_old = v_final[a as usize];

                    #[cfg(test)]
                    {
                        read_tuples.insert((a, v_old, t_final[a as usize]));
                        write_tuples.insert((a, v_new, timestamp + 1));
                    }

                    v_read[RD].push(v_old);
                    t_read[RD].push(t_final[a as usize]);
                    v_write_rd.push(v_new);
                    v_final[a as usize] = v_new;
                    t_final[a as usize] = timestamp + 1;
                }
            };

            let step_index = timestamp as usize;
            let mut is_v_write_ram = false;
            #[allow(unused_assignments)]
            let mut ram_word_address = 0;
            // Only the LB/SB/LH/SH/LW/SW instructions access ≥1 byte of RAM
            if lb_flag[step_index].is_one()
                || lh_flag[step_index].is_one()
                || sb_flag[step_index].is_one()
                || sh_flag[step_index].is_one()
                || sw_flag[step_index].is_one()
            {
                match step[RAM_1] {
                    MemoryOp::Read(a, v) => {
                        assert!(a >= INPUT_START_ADDRESS);
                        let remapped_a = remap_address(a);
                        debug_assert_eq!(v, v_final[remapped_a as usize]);

                        #[cfg(test)]
                        {
                            read_tuples.insert((remapped_a, v, t_final[remapped_a as usize]));
                            write_tuples.insert((remapped_a, v, timestamp));
                        }

                        a_ram.push(remapped_a);
                        v_read[RAM_1].push(v);
                        t_read[RAM_1].push(t_final[remapped_a as usize]);
                        v_write_ram[0].push(v);
                        t_write_ram[0].push(timestamp);
                        t_final[remapped_a as usize] = timestamp;
                        ram_word_address = a;
                    }
                    MemoryOp::Write(a, v_new) => {
                        assert!(a >= INPUT_START_ADDRESS);
                        let remapped_a = remap_address(a);
                        let v_old = v_final[remapped_a as usize];

                        #[cfg(test)]
                        {
                            read_tuples.insert((remapped_a, v_old, t_final[remapped_a as usize]));
                            write_tuples.insert((remapped_a, v_new, timestamp + 1));
                        }

                        a_ram.push(remapped_a);
                        v_read[RAM_1].push(v_old);
                        t_read[RAM_1].push(t_final[remapped_a as usize]);
                        v_write_ram[0].push(v_new);
                        t_write_ram[0].push(timestamp + 1);
                        v_final[remapped_a as usize] = v_new;
                        t_final[remapped_a as usize] = timestamp + 1;
                        ram_word_address = a;
                        is_v_write_ram = true;
                    }
                };
            } else {
                a_ram.push(0);
                for ram_byte_index in [RAM_1, RAM_2, RAM_3, RAM_4] {
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
                    v_read[ram_byte_index].push(0);
                    t_read[ram_byte_index].push(0);
                }
                for v in v_write_ram.iter_mut() {
                    v.push(0);
                }
                for t in t_write_ram.iter_mut() {
                    t.push(0);
                }
                // Increment global timestamp
                timestamp += 1;
                continue;
            }

            // Only the LH/SH/LW/SW instructions access ≥2 byte of RAM
            if lh_flag[step_index].is_one()
                || sh_flag[step_index].is_one()
                || sw_flag[step_index].is_one()
            {
                match step[RAM_2] {
                    MemoryOp::Read(a, v) => {
                        assert!(!is_v_write_ram);
                        assert_eq!(a, ram_word_address + 1);
                        let remapped_a = remap_address(a);
                        debug_assert_eq!(v, v_final[remapped_a as usize]);

                        #[cfg(test)]
                        {
                            read_tuples.insert((remapped_a, v, t_final[remapped_a as usize]));
                            write_tuples.insert((remapped_a, v, timestamp));
                        }

                        v_read[RAM_2].push(v);
                        t_read[RAM_2].push(t_final[remapped_a as usize]);
                        v_write_ram[1].push(v);
                        t_write_ram[1].push(timestamp);
                        t_final[remapped_a as usize] = timestamp;
                    }
                    MemoryOp::Write(a, v_new) => {
                        assert!(is_v_write_ram);
                        assert_eq!(a, ram_word_address + 1);
                        let remapped_a = remap_address(a);
                        let v_old = v_final[remapped_a as usize];

                        #[cfg(test)]
                        {
                            read_tuples.insert((remapped_a, v_old, t_final[remapped_a as usize]));
                            write_tuples.insert((remapped_a, v_new, timestamp + 1));
                        }

                        v_read[RAM_2].push(v_old);
                        t_read[RAM_2].push(t_final[remapped_a as usize]);
                        v_write_ram[1].push(v_new);
                        t_write_ram[1].push(timestamp + 1);
                        v_final[remapped_a as usize] = v_new;
                        t_final[remapped_a as usize] = timestamp + 1;
                    }
                };
            } else {
                for ram_byte_index in [RAM_2, RAM_3, RAM_4] {
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
                    v_read[ram_byte_index].push(0);
                    t_read[ram_byte_index].push(0);
                }
                for v in v_write_ram[1..].iter_mut() {
                    v.push(0);
                }
                for t in t_write_ram[1..].iter_mut() {
                    t.push(0);
                }

                // Increment global timestamp
                timestamp += 1;
                continue;
            }

            // Only the LW/SW instructions access ≥3 byte of RAM
            // Both LW and SW are represented by `sw_flag` for the purpose of lookups
            if sw_flag[step_index].is_one() {
                match step[RAM_3] {
                    MemoryOp::Read(a, v) => {
                        assert!(!is_v_write_ram);
                        assert_eq!(a, ram_word_address + 2);
                        let remapped_a = remap_address(a);
                        debug_assert_eq!(v, v_final[remapped_a as usize]);

                        #[cfg(test)]
                        {
                            read_tuples.insert((remapped_a, v, t_final[remapped_a as usize]));
                            write_tuples.insert((remapped_a, v, timestamp));
                        }

                        v_read[RAM_3].push(v);
                        t_read[RAM_3].push(t_final[remapped_a as usize]);
                        v_write_ram[2].push(v);
                        t_write_ram[2].push(timestamp);
                        t_final[remapped_a as usize] = timestamp;
                    }
                    MemoryOp::Write(a, v_new) => {
                        assert!(is_v_write_ram);
                        assert_eq!(a, ram_word_address + 2);
                        let remapped_a = remap_address(a);
                        let v_old = v_final[remapped_a as usize];

                        #[cfg(test)]
                        {
                            read_tuples.insert((remapped_a, v_old, t_final[remapped_a as usize]));
                            write_tuples.insert((remapped_a, v_new, timestamp + 1));
                        }

                        v_read[RAM_3].push(v_old);
                        t_read[RAM_3].push(t_final[remapped_a as usize]);
                        v_write_ram[2].push(v_new);
                        t_write_ram[2].push(timestamp + 1);
                        v_final[remapped_a as usize] = v_new;
                        t_final[remapped_a as usize] = timestamp + 1;
                    }
                };
                match step[RAM_4] {
                    MemoryOp::Read(a, v) => {
                        assert!(!is_v_write_ram);
                        assert_eq!(a, ram_word_address + 3);
                        let remapped_a = remap_address(a);
                        debug_assert_eq!(v, v_final[remapped_a as usize]);

                        #[cfg(test)]
                        {
                            read_tuples.insert((remapped_a, v, t_final[remapped_a as usize]));
                            write_tuples.insert((remapped_a, v, timestamp));
                        }

                        v_read[RAM_4].push(v);
                        t_read[RAM_4].push(t_final[remapped_a as usize]);
                        v_write_ram[3].push(v);
                        t_write_ram[3].push(timestamp);
                        t_final[remapped_a as usize] = timestamp;
                    }
                    MemoryOp::Write(a, v_new) => {
                        assert!(is_v_write_ram);
                        assert_eq!(a, ram_word_address + 3);
                        let remapped_a = remap_address(a);
                        let v_old = v_final[remapped_a as usize];

                        #[cfg(test)]
                        {
                            read_tuples.insert((remapped_a, v_old, t_final[remapped_a as usize]));
                            write_tuples.insert((remapped_a, v_new, timestamp + 1));
                        }

                        v_read[RAM_4].push(v_old);
                        t_read[RAM_4].push(t_final[remapped_a as usize]);
                        v_write_ram[3].push(v_new);
                        t_write_ram[3].push(timestamp + 1);
                        v_final[remapped_a as usize] = v_new;
                        t_final[remapped_a as usize] = timestamp + 1;
                    }
                };
            } else {
                for ram_byte_index in [RAM_3, RAM_4] {
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
                    v_read[ram_byte_index].push(0);
                    t_read[ram_byte_index].push(0);
                }
                for v in v_write_ram[2..].iter_mut() {
                    v.push(0);
                }
                for t in t_write_ram[2..].iter_mut() {
                    t.push(0);
                }
                // Increment global timestamp
                timestamp += 1;
                continue;
            }

            // Increment global timestamp
            timestamp += 1;
        }
        drop(_enter);
        drop(span);

        #[cfg(test)]
        {
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
    pub fn get_polys_r1cs(&self) -> (Vec<F>, Vec<F>, Vec<F>) {
        let (a_polys, (v_read_polys, v_write_polys)) = rayon::join(
            || self.a_ram.evals(),
            || {
                rayon::join(
                    || {
                        self.v_read
                            .par_iter()
                            .flat_map(|poly| poly.evals())
                            .collect::<Vec<_>>()
                    },
                    || {
                        [&self.v_write_rd]
                            .into_par_iter()
                            .chain(self.v_write_ram.par_iter())
                            .flat_map(|poly| poly.evals())
                            .collect::<Vec<_>>()
                    },
                )
            },
        );

        (a_polys, v_read_polys, v_write_polys)
    }

    /// Computes the maximum number of group generators needed to commit to read-write
    /// memory polynomials using Hyrax, given the maximum memory address and maximum trace length.
    pub fn num_generators(max_memory_address: usize, max_trace_length: usize) -> usize {
        let max_memory_address = max_memory_address.next_power_of_two();
        let max_trace_length = max_trace_length.next_power_of_two();

        // { rs1, rs2, rd, ram_byte_1, ram_byte_2, ram_byte_3, ram_byte_4 }
        let t_read_write_num_vars = (max_trace_length * MEMORY_OPS_PER_INSTRUCTION)
            .next_power_of_two()
            .log_2();
        // v_final, t_final
        let init_final_num_vars = max_memory_address.next_power_of_two().log_2();
        let num_read_write_generators = std::cmp::max(
            matrix_dimensions(max_trace_length.log_2(), NUM_R1CS_POLYS).1,
            matrix_dimensions(t_read_write_num_vars, 1).1,
        );
        let num_init_final_generators = matrix_dimensions(init_final_num_vars, 1).1;

        std::cmp::max(num_read_write_generators, num_init_final_generators)
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct MemoryCommitment<G: CurveGroup> {
    pub trace_commitments: Vec<HyraxCommitment<NUM_R1CS_POLYS, G>>,
    pub v_final_commitment: HyraxCommitment<1, G>,
    pub t_final_commitment: HyraxCommitment<1, G>,
}

impl<G: CurveGroup> AppendToTranscript<G> for MemoryCommitment<G> {
    fn append_to_transcript<T: ProofTranscript<G>>(
        &self,
        label: &'static [u8],
        transcript: &mut T,
    ) {
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
pub struct MemoryReadWriteOpenings<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
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

impl<F, G> StructuredOpeningProof<F, G, JoltPolynomials<F, G>> for MemoryReadWriteOpenings<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type Proof = BatchedHyraxOpeningProof<NUM_R1CS_POLYS, G>;

    #[tracing::instrument(skip_all, name = "MemoryReadWriteOpenings::open")]
    fn open(polynomials: &JoltPolynomials<F, G>, opening_point: &Vec<F>) -> Self {
        let chis = EqPolynomial::new(opening_point.to_vec()).evals();
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
        polynomials: &JoltPolynomials<F, G>,
        opening_point: &Vec<F>,
        openings: &Self,
        transcript: &mut Transcript,
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
        BatchedHyraxOpeningProof::prove(
            &read_write_polys,
            opening_point,
            &read_write_openings,
            transcript,
        )
    }

    fn compute_verifier_openings(&mut self, _: &NoPreprocessing, opening_point: &Vec<F>) {
        self.identity_poly_opening =
            Some(IdentityPolynomial::new(opening_point.len()).evaluate(opening_point));
    }

    fn verify_openings(
        &self,
        generators: &PedersenGenerators<G>,
        opening_proof: &Self::Proof,
        commitment: &JoltCommitments<G>,
        opening_point: &Vec<F>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        let openings = self
            .a_read_write_opening
            .into_iter()
            .chain(self.v_read_opening.into_iter())
            .chain(self.v_write_opening.into_iter())
            .chain(self.t_read_opening.into_iter())
            .chain(self.t_write_ram_opening.into_iter())
            .collect::<Vec<_>>();
        opening_proof.verify(
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
    F: PrimeField,
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
pub struct MemoryInitFinalOpeningProof<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    v_t_opening_proof: BatchedHyraxOpeningProof<1, G>,
}

impl<F, G> StructuredOpeningProof<F, G, JoltPolynomials<F, G>> for MemoryInitFinalOpenings<F>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type Proof = MemoryInitFinalOpeningProof<F, G>;
    type Preprocessing = ReadWriteMemoryPreprocessing;

    #[tracing::instrument(skip_all, name = "MemoryInitFinalOpenings::open")]
    fn open(polynomials: &JoltPolynomials<F, G>, opening_point: &Vec<F>) -> Self {
        let chis = EqPolynomial::new(opening_point.to_vec()).evals();
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
        polynomials: &JoltPolynomials<F, G>,
        opening_point: &Vec<F>,
        openings: &Self,
        transcript: &mut Transcript,
    ) -> Self::Proof {
        let v_t_opening_proof = BatchedHyraxOpeningProof::prove(
            &[
                &polynomials.read_write_memory.v_final,
                &polynomials.read_write_memory.t_final,
            ],
            &opening_point,
            &[openings.v_final, openings.t_final],
            transcript,
        );

        Self::Proof { v_t_opening_proof }
    }

    fn compute_verifier_openings(
        &mut self,
        preprocessing: &Self::Preprocessing,
        opening_point: &Vec<F>,
    ) {
        self.a_init_final =
            Some(IdentityPolynomial::new(opening_point.len()).evaluate(opening_point));

        // TODO(moodlezoup): Compute opening without instantiating v_init polynomial itself
        let memory_size = opening_point.len().pow2();
        let mut v_init: Vec<u64> = vec![0; memory_size];
        // Copy bytecode
        let mut v_init_index = memory_address_to_witness_index(preprocessing.min_bytecode_address);
        for byte in preprocessing.bytecode_bytes.iter() {
            v_init[v_init_index] = *byte as u64;
            v_init_index += 1;
        }
        // Copy input bytes
        v_init_index = memory_address_to_witness_index(INPUT_START_ADDRESS);
        for byte in preprocessing.program_io.as_ref().unwrap().inputs.iter() {
            v_init[v_init_index] = *byte as u64;
            v_init_index += 1;
        }

        self.v_init = Some(DensePolynomial::from_u64(&v_init).evaluate(opening_point));
    }

    fn verify_openings(
        &self,
        generators: &PedersenGenerators<G>,
        opening_proof: &Self::Proof,
        commitment: &JoltCommitments<G>,
        opening_point: &Vec<F>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        opening_proof.v_t_opening_proof.verify(
            generators,
            opening_point,
            &vec![self.v_final, self.t_final],
            &[
                &commitment.read_write_memory.v_final_commitment,
                &commitment.read_write_memory.t_final_commitment,
            ],
            transcript,
        )?;

        Ok(())
    }
}

impl<F, G> MemoryCheckingProver<F, G, JoltPolynomials<F, G>> for ReadWriteMemoryProof<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type Preprocessing = ReadWriteMemoryPreprocessing;
    type ReadWriteOpenings = MemoryReadWriteOpenings<F, G>;
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
        polynomials: &JoltPolynomials<F, G>,
        gamma: &F,
        tau: &F,
    ) -> (Vec<DensePolynomial<F>>, Vec<DensePolynomial<F>>) {
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
                [
                    DensePolynomial::new(read_fingerprints),
                    DensePolynomial::new(write_fingerprints),
                ]
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
            vec![
                DensePolynomial::new(init_fingerprints),
                DensePolynomial::new(final_fingerprints),
            ],
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

impl<F, G> MemoryCheckingVerifier<F, G, JoltPolynomials<F, G>> for ReadWriteMemoryProof<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
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
pub struct OutputSumcheckProof<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    num_rounds: usize,
    /// Sumcheck proof that v_final is equal to the program outputs at the relevant indices.
    sumcheck_proof: SumcheckInstanceProof<F>,
    /// Opening of v_final at the random point chosen over the course of sumcheck
    opening: F,
    /// Hyrax opening proof of the v_final opening
    opening_proof: HyraxOpeningProof<1, G>,
}

impl<F, G> OutputSumcheckProof<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    fn prove_outputs(
        polynomials: &ReadWriteMemory<F, G>,
        program_io: &JoltDevice,
        transcript: &mut Transcript,
    ) -> Self {
        let num_rounds = polynomials.memory_size.log_2();
        let r_eq = <Transcript as ProofTranscript<G>>::challenge_vector(
            transcript,
            b"output_sumcheck",
            num_rounds,
        );
        let eq: DensePolynomial<F> = DensePolynomial::new(EqPolynomial::new(r_eq.to_vec()).evals());

        let io_witness_range: Vec<_> = (0..polynomials.memory_size as u64)
            .into_iter()
            .map(|i| {
                if i >= INPUT_START_ADDRESS && i < RAM_WITNESS_OFFSET {
                    F::one()
                } else {
                    F::zero()
                }
            })
            .collect();

        let mut v_io: Vec<u64> = vec![0; polynomials.memory_size];
        // Copy input bytes
        let mut input_index = memory_address_to_witness_index(INPUT_START_ADDRESS);
        for byte in program_io.inputs.iter() {
            v_io[input_index] = *byte as u64;
            input_index += 1;
        }
        // Copy output bytes
        let mut output_index = memory_address_to_witness_index(OUTPUT_START_ADDRESS);
        for byte in program_io.outputs.iter() {
            v_io[output_index] = *byte as u64;
            output_index += 1;
        }
        // Copy panic bit
        v_io[memory_address_to_witness_index(PANIC_ADDRESS)] = program_io.panic as u64;

        let mut sumcheck_polys = vec![
            eq,
            DensePolynomial::new(io_witness_range),
            polynomials.v_final.clone(),
            DensePolynomial::from_u64(&v_io),
        ];

        // eq * io_witness_range * (v_final - v_io)
        let output_check_fn = |vals: &[F]| -> F { vals[0] * vals[1] * (vals[2] - vals[3]) };

        let (sumcheck_proof, r_sumcheck, sumcheck_openings) =
            SumcheckInstanceProof::<F>::prove_arbitrary::<_, G, Transcript>(
                &F::zero(),
                num_rounds,
                &mut sumcheck_polys,
                output_check_fn,
                3,
                transcript,
            );

        let sumcheck_opening_proof =
            HyraxOpeningProof::prove(&polynomials.v_final, &r_sumcheck, transcript);

        Self {
            num_rounds,
            sumcheck_proof,
            opening: sumcheck_openings[2], // only need v_final; verifier computes the rest on its own
            opening_proof: sumcheck_opening_proof,
        }
    }

    fn verify(
        proof: &Self,
        preprocessing: &mut ReadWriteMemoryPreprocessing,
        generators: &PedersenGenerators<G>,
        commitment: &MemoryCommitment<G>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        let r_eq = <Transcript as ProofTranscript<G>>::challenge_vector(
            transcript,
            b"output_sumcheck",
            proof.num_rounds,
        );

        let (sumcheck_claim, r_sumcheck) = proof.sumcheck_proof.verify::<G, Transcript>(
            F::zero(),
            proof.num_rounds,
            3,
            transcript,
        )?;

        let eq_eval = EqPolynomial::new(r_eq.to_vec()).evaluate(&r_sumcheck);

        // TODO(moodlezoup): Compute openings without instantiating io_witness_range polynomial itself
        let memory_size = proof.num_rounds.pow2();
        let io_witness_range: Vec<_> = (0..memory_size as u64)
            .into_iter()
            .map(|i| {
                if i >= INPUT_START_ADDRESS && i < RAM_WITNESS_OFFSET {
                    F::one()
                } else {
                    F::zero()
                }
            })
            .collect();
        let io_witness_range_eval = DensePolynomial::new(io_witness_range).evaluate(&r_sumcheck);

        // TODO(moodlezoup): Compute openings without instantiating v_io polynomial itself
        let mut v_io: Vec<u64> = vec![0; memory_size];
        // Copy input bytes
        let mut input_index = memory_address_to_witness_index(INPUT_START_ADDRESS);
        for byte in preprocessing.program_io.as_ref().unwrap().inputs.iter() {
            v_io[input_index] = *byte as u64;
            input_index += 1;
        }
        // Copy output bytes
        let mut output_index = memory_address_to_witness_index(OUTPUT_START_ADDRESS);
        for byte in preprocessing.program_io.as_ref().unwrap().outputs.iter() {
            v_io[output_index] = *byte as u64;
            output_index += 1;
        }
        // Copy panic bit
        v_io[memory_address_to_witness_index(PANIC_ADDRESS)] =
            preprocessing.program_io.as_ref().unwrap().panic as u64;
        let v_io_eval = DensePolynomial::from_u64(&v_io).evaluate(&r_sumcheck);

        assert_eq!(
            eq_eval * io_witness_range_eval * (proof.opening - v_io_eval),
            sumcheck_claim,
            "Output sumcheck check failed."
        );

        proof.opening_proof.verify(
            generators,
            transcript,
            &r_sumcheck,
            &proof.opening,
            &commitment.v_final_commitment,
        )
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct ReadWriteMemoryProof<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    pub memory_checking_proof: MemoryCheckingProof<
        G,
        JoltPolynomials<F, G>,
        MemoryReadWriteOpenings<F, G>,
        MemoryInitFinalOpenings<F>,
    >,
    pub timestamp_validity_proof: TimestampValidityProof<F, G>,
    pub output_proof: OutputSumcheckProof<F, G>,
}

impl<F, G> ReadWriteMemoryProof<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    #[tracing::instrument(skip_all, name = "ReadWriteMemoryProof::prove")]
    pub fn prove(
        preprocessing: &ReadWriteMemoryPreprocessing,
        polynomials: &JoltPolynomials<F, G>,
        program_io: &JoltDevice,
        transcript: &mut Transcript,
    ) -> Self {
        let memory_checking_proof =
            ReadWriteMemoryProof::prove_memory_checking(preprocessing, &polynomials, transcript);

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
        generators: &PedersenGenerators<G>,
        preprocessing: &mut ReadWriteMemoryPreprocessing,
        commitment: &JoltCommitments<G>,
        transcript: &mut Transcript,
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
