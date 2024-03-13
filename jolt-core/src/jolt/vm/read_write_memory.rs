use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use merlin::Transcript;
use rand::rngs::StdRng;
use rand_core::RngCore;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::cmp::max;
#[cfg(test)]
use std::collections::HashSet;
use std::marker::PhantomData;

use crate::{
    lasso::memory_checking::{
        MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier, MultisetHashes,
        NoPreprocessing,
    },
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        hyrax::matrix_dimensions,
        identity_poly::IdentityPolynomial,
        pedersen::PedersenGenerators,
        structured_poly::{BatchablePolynomials, StructuredOpeningProof},
    },
    subprotocols::batched_commitment::{
        BatchedPolynomialCommitment, BatchedPolynomialOpeningProof,
    },
    utils::{errors::ProofVerifyError, math::Math, mul_0_optimized},
};
use common::constants::{
    BYTES_PER_INSTRUCTION, MEMORY_OPS_PER_INSTRUCTION, RAM_START_ADDRESS, REGISTER_COUNT,
};
use common::{to_ram_address, ELFInstruction};

use super::timestamp_range_check::TimestampValidityProof;

pub trait RandomInstruction {
    fn random(index: usize, rng: &mut StdRng) -> Self;
}

impl RandomInstruction for ELFInstruction {
    fn random(index: usize, rng: &mut StdRng) -> Self {
        Self {
            address: to_ram_address(index) as u64,
            raw: rng.next_u32(),
            // Only `address` and `raw` are used in ReadWriteMemory; the rest don't matter
            opcode: common::RV32IM::ADD,
            rs1: None,
            rs2: None,
            rd: None,
            imm: None,
        }
    }
}

pub fn random_memory_trace(
    bytecode: &Vec<ELFInstruction>,
    max_memory_address: usize,
    m: usize,
    rng: &mut StdRng,
) -> Vec<[MemoryOp; MEMORY_OPS_PER_INSTRUCTION]> {
    let mut memory: Vec<u64> = vec![0; max_memory_address];
    for instr in bytecode {
        let address = instr.address - RAM_START_ADDRESS + REGISTER_COUNT;
        let raw = instr.raw;
        for i in 0..(BYTES_PER_INSTRUCTION as u64) {
            // Write one byte of raw to memory
            memory[(address + i) as usize] = ((raw >> (i * 8)) & 0xff) as u64;
        }
    }

    let m = m.next_power_of_two();
    let mut memory_trace = Vec::with_capacity(m);
    for _ in 0..m {
        let mut ops: [MemoryOp; MEMORY_OPS_PER_INSTRUCTION] =
            std::array::from_fn(|_| MemoryOp::no_op());

        let rs1 = rng.next_u64() % REGISTER_COUNT;
        ops[0] = MemoryOp::Read(rs1, memory[rs1 as usize]);

        let rs2 = rng.next_u64() % REGISTER_COUNT;
        ops[1] = MemoryOp::Read(rs2, memory[rs2 as usize]);

        // Don't write to the zero register
        let rd = rng.next_u64() % (REGISTER_COUNT - 1) + 1;
        // Registers are 32 bits
        let register_value = rng.next_u32() as u64;
        ops[2] = MemoryOp::Write(rd, register_value);
        memory[rd as usize] = register_value;

        if rng.next_u32() % 2 == 0 {
            // LOAD
            let remapped_address =
                REGISTER_COUNT + rng.next_u64() % (max_memory_address as u64 - REGISTER_COUNT - 4);
            let ram_address = remapped_address - REGISTER_COUNT + RAM_START_ADDRESS;
            for i in 0..4 {
                ops[i + 3] = MemoryOp::Read(
                    ram_address + i as u64,
                    memory[i + remapped_address as usize],
                );
            }
        } else {
            // STORE
            let remapped_address =
                REGISTER_COUNT + rng.next_u64() % (max_memory_address as u64 - REGISTER_COUNT - 4);
            let ram_address = remapped_address - REGISTER_COUNT + RAM_START_ADDRESS;
            for i in 0..4 {
                // RAM is byte-addressable, so values are a single byte
                let ram_value = rng.next_u64() & 0xff;
                ops[i + 3] = MemoryOp::Write(ram_address + i as u64, ram_value);
                memory[i + remapped_address as usize] = ram_value;
            }
        }

        memory_trace.push(ops);
    }

    memory_trace
}

pub struct ReadWriteMemoryProof<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    pub memory_checking_proof: MemoryCheckingProof<
        G,
        ReadWriteMemory<F, G>,
        MemoryReadWriteOpenings<F, G>,
        MemoryInitFinalOpenings<F>,
    >,
    pub timestamp_validity_proof: TimestampValidityProof<F, G>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum MemoryOp {
    Read(u64, u64),  // (address, value)
    Write(u64, u64), // (address, new_value)
}

impl MemoryOp {
    pub fn no_op() -> Self {
        Self::Read(0, 0)
    }
}

pub struct ReadWriteMemory<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    _group: PhantomData<G>,
    /// Size of entire address space (i.e. RAM + registers for RISC-V)
    memory_size: usize,
    /// MLE of initial memory values. RAM is initialized to contain the program bytecode.
    pub v_init: DensePolynomial<F>,
    /// MLE of read/write addresses. For offline memory checking, each read is paired with a "virtual" write
    /// and vice versa, so the read addresses and write addresses are the same.
    pub a_read_write: [DensePolynomial<F>; MEMORY_OPS_PER_INSTRUCTION],
    /// MLE of the read values.
    pub v_read: [DensePolynomial<F>; MEMORY_OPS_PER_INSTRUCTION],
    /// MLE of the write values.
    pub v_write: [DensePolynomial<F>; MEMORY_OPS_PER_INSTRUCTION],
    /// MLE of the final memory state.
    pub v_final: DensePolynomial<F>,
    /// MLE of the read timestamps.
    pub t_read: [DensePolynomial<F>; MEMORY_OPS_PER_INSTRUCTION],
    /// MLE of the write timestamps.
    pub t_write: [DensePolynomial<F>; MEMORY_OPS_PER_INSTRUCTION],
    /// MLE of the final timestamps.
    pub t_final: DensePolynomial<F>,
}

impl<F: PrimeField, G: CurveGroup<ScalarField = F>> ReadWriteMemory<F, G> {
    #[tracing::instrument(skip_all, name = "ReadWriteMemory::new")]
    pub fn new(
        bytecode: Vec<ELFInstruction>,
        memory_trace: Vec<[MemoryOp; MEMORY_OPS_PER_INSTRUCTION]>,
        transcript: &mut Transcript,
    ) -> (Self, [Vec<u64>; MEMORY_OPS_PER_INSTRUCTION]) {
        let m = memory_trace.len();
        assert!(m.is_power_of_two());

        let remap_address = |a: u64| {
            assert!(a < REGISTER_COUNT || a >= RAM_START_ADDRESS);
            if a >= RAM_START_ADDRESS {
                a - RAM_START_ADDRESS + REGISTER_COUNT
            } else {
                // If a < REGISTER_COUNT, it is one of the registers and doesn't
                // need to be remapped
                a
            }
        };

        let max_memory_address = memory_trace
            .iter()
            .flat_map(|step| {
                step.iter().map(|op| match op {
                    MemoryOp::Read(a, _) => remap_address(*a),
                    MemoryOp::Write(a, _) => remap_address(*a),
                })
            })
            .max()
            .unwrap_or(0);
        let max_bytecode_address = bytecode
            .iter()
            .map(|instr| remap_address(instr.address))
            .max()
            .unwrap_or(0)
            + (BYTES_PER_INSTRUCTION as u64 - 1); // For RV32I, instructions occupy 4 bytes, so the max bytecode address is the max instruction address + 3
        let memory_size =
            max(max_memory_address, max_bytecode_address).next_power_of_two() as usize;

        let mut v_init: Vec<u64> = vec![0; memory_size];
        for instr in bytecode {
            let address = remap_address(instr.address);
            let raw = instr.raw;
            for i in 0..(BYTES_PER_INSTRUCTION as u64) {
                // Write one byte of raw to v_init
                v_init[(address + i) as usize] = ((raw >> (i * 8)) & 0xff) as u64;
            }
        }

        #[cfg(test)]
        let mut init_tuples: HashSet<(u64, u64, u64)> = HashSet::new();
        #[cfg(test)]
        {
            for (a, v) in v_init.iter().enumerate() {
                init_tuples.insert((a as u64, *v, 0u64));
            }
        }

        let mut a_read_write: [Vec<u64>; MEMORY_OPS_PER_INSTRUCTION] =
            std::array::from_fn(|_| Vec::with_capacity(m));
        let mut v_read: [Vec<u64>; MEMORY_OPS_PER_INSTRUCTION] =
            std::array::from_fn(|_| Vec::with_capacity(m));
        let mut v_write: [Vec<u64>; MEMORY_OPS_PER_INSTRUCTION] =
            std::array::from_fn(|_| Vec::with_capacity(m));
        let mut t_read: [Vec<u64>; MEMORY_OPS_PER_INSTRUCTION] =
            std::array::from_fn(|_| Vec::with_capacity(m));
        let mut t_write: [Vec<u64>; MEMORY_OPS_PER_INSTRUCTION] =
            std::array::from_fn(|_| Vec::with_capacity(m));
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
            // Read operations
            for (i, memory_access) in step.iter().enumerate() {
                match memory_access {
                    MemoryOp::Read(a, v) => {
                        let remapped_a = remap_address(*a);
                        debug_assert_eq!(*v, v_final[remapped_a as usize]);

                        #[cfg(test)]
                        {
                            read_tuples.insert((remapped_a, *v, t_final[remapped_a as usize]));
                            write_tuples.insert((remapped_a, *v, timestamp));
                        }

                        a_read_write[i].push(remapped_a);
                        v_read[i].push(*v);
                        t_read[i].push(t_final[remapped_a as usize]);
                        v_write[i].push(*v);
                        t_write[i].push(timestamp);
                        t_final[remapped_a as usize] = timestamp;
                    }
                    _ => {}
                }
            }

            // Write operations
            for (i, memory_access) in step.iter().enumerate() {
                match memory_access {
                    MemoryOp::Write(a, v_new) => {
                        let remapped_a = remap_address(*a);
                        let v_old = v_final[remapped_a as usize];

                        #[cfg(test)]
                        {
                            read_tuples.insert((remapped_a, v_old, t_final[remapped_a as usize]));
                            write_tuples.insert((remapped_a, *v_new, timestamp + 1));
                        }

                        a_read_write[i].push(remapped_a);
                        v_read[i].push(v_old);
                        t_read[i].push(t_final[remapped_a as usize]);
                        v_write[i].push(*v_new);
                        t_write[i].push(timestamp + 1);
                        v_final[remapped_a as usize] = *v_new;
                        t_final[remapped_a as usize] = timestamp + 1;
                    }
                    _ => {}
                }
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
        let map_to_polys = |vals: &[Vec<u64>; MEMORY_OPS_PER_INSTRUCTION]| {
            vals.par_iter()
                .map(|vals| DensePolynomial::from_u64(vals))
                .collect::<Vec<DensePolynomial<F>>>()
                .try_into()
                .unwrap()
        };

        let (v_init, v_final, t_final, a_read_write, v_read, v_write, t_read_polys, t_write): (
            DensePolynomial<F>,
            DensePolynomial<F>,
            DensePolynomial<F>,
            [DensePolynomial<F>; MEMORY_OPS_PER_INSTRUCTION],
            [DensePolynomial<F>; MEMORY_OPS_PER_INSTRUCTION],
            [DensePolynomial<F>; MEMORY_OPS_PER_INSTRUCTION],
            [DensePolynomial<F>; MEMORY_OPS_PER_INSTRUCTION],
            [DensePolynomial<F>; MEMORY_OPS_PER_INSTRUCTION]
        ) = common::par_join_8!(
            || DensePolynomial::from_u64(&v_init),
            || DensePolynomial::from_u64(&v_final),
            || DensePolynomial::from_u64(&t_final),
            || map_to_polys(&a_read_write),
            || map_to_polys(&v_read),
            || map_to_polys(&v_write),
            || map_to_polys(&t_read),
            || map_to_polys(&t_write)
        );


        (
            Self {
                _group: PhantomData,
                memory_size,
                v_init,
                a_read_write,
                v_read,
                v_write,
                v_final,
                t_read: t_read_polys,
                t_write,
                t_final,
            },
            t_read,
        )
    }

    // TODO(arasuarun): This is seriously slow and duplicated work from above.
    #[tracing::instrument(skip_all, name = "ReadWriteMemory::get_r1cs_polys")]
    pub fn get_r1cs_polys(
        bytecode: Vec<ELFInstruction>,
        memory_trace: Vec<MemoryOp>,
        transcript: &mut Transcript,
    ) -> [Vec<F>; 4] {
        let m = memory_trace.len();

        let remap_address = |a: u64| {
            assert!(a < REGISTER_COUNT || a >= RAM_START_ADDRESS);
            if a >= RAM_START_ADDRESS {
                a - RAM_START_ADDRESS + REGISTER_COUNT
                // TODO(arasuarun): for r1cs, do not substract RAM_START_ADDRESS
                // a
            } else {
                // If a < REGISTER_COUNT, it is one of the registers and doesn't
                // need to be remapped
                a
            }
        };

        let span = tracing::span!(tracing::Level::DEBUG, "memory_size_calculation");
        let _enter = span.enter();
        let max_memory_address = memory_trace
            .iter()
            .map(|op| match op {
                MemoryOp::Read(a, _) => remap_address(*a),
                MemoryOp::Write(a, _) => remap_address(*a),
            })
            .max()
            .unwrap_or(0);
        let max_bytecode_address = bytecode
            .iter()
            .map(|instr| remap_address(instr.address))
            .max()
            .unwrap_or(0)
            + (BYTES_PER_INSTRUCTION as u64 - 1); // For RV32I, instructions occupy 4 bytes, so the max bytecode address is the max instruction address + 3
        let memory_size =
            max(max_memory_address, max_bytecode_address).next_power_of_two() as usize;
        drop(_enter);
        drop(span);

        let span = tracing::span!(tracing::Level::DEBUG, "initialize_memory");
        let _enter = span.enter();
        let mut v_init: Vec<u64> = vec![0; memory_size];
        for instr in bytecode {
            let address = remap_address(instr.address);
            let raw = instr.raw;
            for i in 0..(BYTES_PER_INSTRUCTION as u64) {
                // Write one byte of raw to v_init
                v_init[(address + i) as usize] = ((raw >> (i * 8)) & 0xff) as u64;
            }
        }
        drop(_enter);
        drop(span);

        let span = tracing::span!(tracing::Level::DEBUG, "initialize_vectors");
        let _enter = span.enter();
        let mut a_read_write: Vec<u64> = Vec::with_capacity(m);
        let mut v_read: Vec<u64> = Vec::with_capacity(m);
        let mut v_write: Vec<u64> = Vec::with_capacity(m);

        let span_clone = tracing::span!(tracing::Level::DEBUG, "clone_avoidance");
        let _enter_clone = span_clone.enter();
        let mut v_final: Vec<u64> = v_init.clone(); // TODO(moodlezoup): avoid clone
        drop(_enter_clone);
        drop(span_clone);

        let mut t_read: Vec<u64> = Vec::with_capacity(m);
        let mut t_write: Vec<u64> = Vec::with_capacity(m);

        let span = tracing::span!(tracing::Level::DEBUG, "initialize_t_final");
        let _enter = span.enter();
        let mut t_final: Vec<u64> = vec![0; memory_size];
        drop(_enter);
        drop(span);

        let mut timestamp: u64 = 0;
        let span = tracing::span!(tracing::Level::DEBUG, "memory_access_processing");
        let _enter = span.enter();
        for memory_access in memory_trace {
            match memory_access {
                MemoryOp::Read(a, v) => {
                    let remapped_a = remap_address(a);
                    debug_assert_eq!(v, v_final[remapped_a as usize]);
                    a_read_write.push(remapped_a);
                    v_read.push(v);
                    v_write.push(v);
                    t_read.push(t_final[remapped_a as usize]);
                    t_write.push(timestamp + 1);
                    t_final[remapped_a as usize] = timestamp + 1;
                }
                MemoryOp::Write(a, v_new) => {
                    let remapped_a = remap_address(a);
                    let v_old = v_final[remapped_a as usize];
                    a_read_write.push(remapped_a);
                    v_read.push(v_old);
                    v_write.push(v_new);
                    v_final[remapped_a as usize] = v_new;
                    t_read.push(t_final[remapped_a as usize]);
                    t_write.push(timestamp + 1);
                    t_final[remapped_a as usize] = timestamp + 1;
                }
            }
            timestamp += 1;
        }
        drop(_enter);
        drop(span);

        // create a closure to convert u64 to F vector
        let to_f_vec = |v: &Vec<u64>| -> Vec<F> {
            v.par_iter()
                .map(|i| F::from_u64(*i).unwrap())
                .collect::<Vec<F>>()
        };

        let un_remap_address = |a: &Vec<u64>| {
            a.iter()
                .map(|addr| {
                    if *addr >= REGISTER_COUNT {
                        addr + RAM_START_ADDRESS - REGISTER_COUNT
                    } else {
                        *addr
                    }
                })
                .collect::<Vec<u64>>()
        };

        [
            to_f_vec(&un_remap_address(&a_read_write)),
            to_f_vec(&v_read),
            to_f_vec(&v_write),
            to_f_vec(&t_read),
        ]
    }

    /// Computes the maximum number of group generators needed to commit to read-write
    /// memory polynomials using Hyrax, given the maximum memory address and maximum trace length.
    pub fn num_generators(max_memory_address: usize, max_trace_length: usize) -> usize {
        // { rs1, rs2, rd, ram_byte_1, ram_byte_2, ram_byte_3, ram_byte_4 } x { a_read, a_write, v_read, v_write, t_read_write }
        let read_write_num_vars = (max_trace_length * MEMORY_OPS_PER_INSTRUCTION * 5).log_2();
        // v_init, v_final, t_final
        let init_final_num_vars = (max_memory_address * 3).log_2();
        let max_num_vars = std::cmp::max(read_write_num_vars, init_final_num_vars);
        matrix_dimensions(max_num_vars).1.pow2()
    }
}

pub struct BatchedMemoryPolynomials<F: PrimeField> {
    /// Contains:
    /// a_read_write, v_read, v_write, t_read, t_write
    batched_read_write: DensePolynomial<F>,
    /// Contains:
    /// v_init, v_final, t_final
    batched_init_final: DensePolynomial<F>,
}

pub struct MemoryCommitment<G: CurveGroup> {
    /// Commitments for:
    /// a_read_write, v_read, v_write, t_read, t_write
    pub read_write_commitments: BatchedPolynomialCommitment<G>,

    /// Commitments for:
    /// v_init, v_final, t_final
    pub init_final_commitments: BatchedPolynomialCommitment<G>,
}

impl<F, G> BatchablePolynomials<G> for ReadWriteMemory<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type BatchedPolynomials = BatchedMemoryPolynomials<F>;
    type Commitment = MemoryCommitment<G>;

    #[tracing::instrument(skip_all, name = "ReadWriteMemory::batch")]
    fn batch(&self) -> Self::BatchedPolynomials {
        let batched_read_write = DensePolynomial::merge(
            self.a_read_write
                .iter()
                .chain(self.v_read.iter())
                .chain(self.v_write.iter())
                .chain(self.t_read.iter())
                .chain(self.t_write.iter()),
        );
        let batched_init_final =
            DensePolynomial::merge(&vec![&self.v_init, &self.v_final, &self.t_final]);

        Self::BatchedPolynomials {
            batched_read_write,
            batched_init_final,
        }
    }

    #[tracing::instrument(skip_all, name = "ReadWriteMemory::commit")]
    fn commit(
        batched_polys: &Self::BatchedPolynomials,
        pedersen_generators: &PedersenGenerators<G>,
    ) -> Self::Commitment {
        let read_write_commitments = batched_polys
            .batched_read_write
            .combined_commit(pedersen_generators);
        let init_final_commitments = batched_polys
            .batched_init_final
            .combined_commit(pedersen_generators);

        Self::Commitment {
            read_write_commitments,
            init_final_commitments,
        }
    }
}

pub struct MemoryReadWriteOpenings<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    /// Evaluation of the a_read_write polynomial at the opening point.
    pub a_read_write_opening: [F; MEMORY_OPS_PER_INSTRUCTION],
    /// Evaluation of the v_read polynomial at the opening point.
    pub v_read_opening: [F; MEMORY_OPS_PER_INSTRUCTION],
    /// Evaluation of the v_write polynomial at the opening point.
    pub v_write_opening: [F; MEMORY_OPS_PER_INSTRUCTION],
    /// Evaluation of the t_read polynomial at the opening point.
    pub t_read_opening: [F; MEMORY_OPS_PER_INSTRUCTION],
    /// Evaluation of the t_write polynomial at the opening point.
    pub t_write_opening: [F; MEMORY_OPS_PER_INSTRUCTION],
}

impl<F, G> StructuredOpeningProof<F, G, ReadWriteMemory<F, G>> for MemoryReadWriteOpenings<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    #[tracing::instrument(skip_all, name = "MemoryReadWriteOpenings::open")]
    fn open(polynomials: &ReadWriteMemory<F, G>, opening_point: &Vec<F>) -> Self {
        let chis = EqPolynomial::new(opening_point.to_vec()).evals();
        let mut openings = polynomials
            .a_read_write
            .par_iter()
            .chain(polynomials.v_read.par_iter())
            .chain(polynomials.v_write.par_iter())
            .chain(polynomials.t_read.par_iter())
            .chain(polynomials.t_write.par_iter())
            .map(|poly| poly.evaluate_at_chi(&chis))
            .collect::<Vec<F>>()
            .into_iter();

        let a_read_write_opening: [F; MEMORY_OPS_PER_INSTRUCTION] = openings.next_chunk().unwrap();
        let v_read_opening = openings.next_chunk().unwrap();
        let v_write_opening = openings.next_chunk().unwrap();
        let t_read_opening = openings.next_chunk().unwrap();
        let t_write_opening = openings.next_chunk().unwrap();

        Self {
            a_read_write_opening,
            v_read_opening,
            v_write_opening,
            t_read_opening,
            t_write_opening,
        }
    }

    #[tracing::instrument(skip_all, name = "MemoryReadWriteOpenings::prove_openings")]
    fn prove_openings(
        polynomials: &BatchedMemoryPolynomials<F>,
        opening_point: &Vec<F>,
        openings: &Self,
        transcript: &mut Transcript,
    ) -> Self::Proof {
        let combined_openings: Vec<F> = openings
            .a_read_write_opening
            .into_iter()
            .chain(openings.v_read_opening.into_iter())
            .chain(openings.v_write_opening.into_iter())
            .chain(openings.t_read_opening.into_iter())
            .chain(openings.t_write_opening.into_iter())
            .collect();

        BatchedPolynomialOpeningProof::prove(
            &polynomials.batched_read_write,
            &opening_point,
            &combined_openings,
            transcript,
        )
    }

    fn verify_openings(
        &self,
        opening_proof: &Self::Proof,
        commitment: &MemoryCommitment<G>,
        opening_point: &Vec<F>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        let combined_openings: Vec<F> = self
            .a_read_write_opening
            .into_iter()
            .chain(self.v_read_opening.into_iter())
            .chain(self.v_write_opening.into_iter())
            .chain(self.t_read_opening.into_iter())
            .chain(self.t_write_opening.into_iter())
            .collect();

        opening_proof.verify(
            opening_point,
            &combined_openings,
            &commitment.read_write_commitments,
            transcript,
        )
    }
}

pub struct MemoryInitFinalOpenings<F>
where
    F: PrimeField,
{
    /// Evaluation of the a_init_final polynomial at the opening point. Computed by the verifier in `compute_verifier_openings`.
    a_init_final: Option<F>,
    /// Evaluation of the v_init polynomial at the opening point.
    v_init: F,
    /// Evaluation of the v_final polynomial at the opening point.
    v_final: F,
    /// Evaluation of the t_final polynomial at the opening point.
    t_final: F,
}

impl<F, G> StructuredOpeningProof<F, G, ReadWriteMemory<F, G>> for MemoryInitFinalOpenings<F>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    #[tracing::instrument(skip_all, name = "MemoryInitFinalOpenings::open")]
    fn open(polynomials: &ReadWriteMemory<F, G>, opening_point: &Vec<F>) -> Self {
        let chis = EqPolynomial::new(opening_point.to_vec()).evals();
        let [v_init, v_final, t_final] = [
            &polynomials.v_init,
            &polynomials.v_final,
            &polynomials.t_final,
        ]
        .par_iter()
        .map(|poly| poly.evaluate_at_chi(&chis))
        .collect::<Vec<F>>()
        .try_into()
        .unwrap();

        Self {
            a_init_final: None,
            v_init,
            v_final,
            t_final,
        }
    }

    #[tracing::instrument(skip_all, name = "MemoryInitFinalOpenings::prove_openings")]
    fn prove_openings(
        polynomials: &BatchedMemoryPolynomials<F>,
        opening_point: &Vec<F>,
        openings: &Self,
        transcript: &mut Transcript,
    ) -> Self::Proof {
        BatchedPolynomialOpeningProof::prove(
            &polynomials.batched_init_final,
            &opening_point,
            &vec![openings.v_init, openings.v_final, openings.t_final],
            transcript,
        )
    }

    fn compute_verifier_openings(&mut self, opening_point: &Vec<F>) {
        self.a_init_final =
            Some(IdentityPolynomial::new(opening_point.len()).evaluate(opening_point));
    }

    fn verify_openings(
        &self,
        opening_proof: &Self::Proof,
        commitment: &MemoryCommitment<G>,
        opening_point: &Vec<F>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        opening_proof.verify(
            opening_point,
            &vec![self.v_init, self.v_final, self.t_final],
            &commitment.init_final_commitments,
            transcript,
        )
    }
}

impl<F, G> MemoryCheckingProver<F, G, ReadWriteMemory<F, G>, NoPreprocessing>
    for ReadWriteMemoryProof<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
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
        _preprocessing: &NoPreprocessing,
        polynomials: &ReadWriteMemory<F, G>,
        gamma: &F,
        tau: &F,
    ) -> (Vec<DensePolynomial<F>>, Vec<DensePolynomial<F>>) {
        let gamma_squared = gamma.square();
        let num_ops = polynomials.a_read_write[0].len();

        let read_write_leaves = (0..MEMORY_OPS_PER_INSTRUCTION)
            .into_par_iter()
            .flat_map(|i| {
                let read_fingerprints = (0..num_ops)
                    .into_par_iter()
                    .map(|j| {
                        polynomials.t_read[i][j] * gamma_squared
                            + mul_0_optimized(&polynomials.v_read[i][j], gamma)
                            + polynomials.a_read_write[i][j]
                            - *tau
                    })
                    .collect();
                let write_fingerprints = (0..num_ops)
                    .into_par_iter()
                    .map(|j| {
                        polynomials.t_write[i][j] * gamma_squared
                            + mul_0_optimized(&polynomials.v_write[i][j], gamma)
                            + polynomials.a_read_write[i][j]
                            - *tau
                    })
                    .collect();
                [
                    DensePolynomial::new(read_fingerprints),
                    DensePolynomial::new(write_fingerprints),
                ]
            })
            .collect();

        let init_fingerprints = (0..polynomials.memory_size)
            .into_par_iter()
            .map(|i| /* 0 * gamma^2 + */ mul_0_optimized(&polynomials.v_init[i], gamma) + F::from_u64(i as u64).unwrap() - *tau)
            .collect();
        let final_fingerprints = (0..polynomials.memory_size)
            .into_par_iter()
            .map(|i| {
                mul_0_optimized(&polynomials.t_final[i], &gamma_squared)
                    + mul_0_optimized(&polynomials.v_final[i], gamma)
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
        _preprocessing: &NoPreprocessing,
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
        _preprocessing: &NoPreprocessing,
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

impl<F, G> MemoryCheckingVerifier<F, G, ReadWriteMemory<F, G>, NoPreprocessing>
    for ReadWriteMemoryProof<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    fn read_tuples(
        &_: &NoPreprocessing,
        openings: &Self::ReadWriteOpenings,
    ) -> Vec<Self::MemoryTuple> {
        (0..MEMORY_OPS_PER_INSTRUCTION)
            .map(|i| {
                (
                    openings.a_read_write_opening[i],
                    openings.v_read_opening[i],
                    openings.t_read_opening[i],
                )
            })
            .collect()
    }
    fn write_tuples(
        &_: &NoPreprocessing,
        openings: &Self::ReadWriteOpenings,
    ) -> Vec<Self::MemoryTuple> {
        (0..MEMORY_OPS_PER_INSTRUCTION)
            .map(|i| {
                (
                    openings.a_read_write_opening[i],
                    openings.v_write_opening[i],
                    openings.t_write_opening[i],
                )
            })
            .collect()
    }
    fn init_tuples(
        &_: &NoPreprocessing,
        openings: &Self::InitFinalOpenings,
    ) -> Vec<Self::MemoryTuple> {
        vec![(openings.a_init_final.unwrap(), openings.v_init, F::zero())]
    }
    fn final_tuples(
        &_: &NoPreprocessing,
        openings: &Self::InitFinalOpenings,
    ) -> Vec<Self::MemoryTuple> {
        vec![(
            openings.a_init_final.unwrap(),
            openings.v_final,
            openings.t_final,
        )]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_curve25519::{EdwardsProjective, Fr};
    use rand_core::SeedableRng;

    #[test]
    fn e2e_memchecking() {
        const MEMORY_SIZE: usize = 1 << 16;
        const NUM_OPS: usize = 1 << 8;
        const BYTECODE_SIZE: usize = 1 << 8;

        let mut rng = rand::rngs::StdRng::seed_from_u64(1234567890);
        let bytecode = (0..BYTECODE_SIZE)
            .map(|i| ELFInstruction::random(i, &mut rng))
            .collect();
        let memory_trace = random_memory_trace(&bytecode, MEMORY_SIZE, NUM_OPS, &mut rng);

        let mut transcript = Transcript::new(b"test_transcript");

        let (rw_memory, _): (ReadWriteMemory<Fr, EdwardsProjective>, _) =
            ReadWriteMemory::new(bytecode, memory_trace, &mut transcript);
        let batched_polys = rw_memory.batch();
        let generators = PedersenGenerators::new(1 << 10, b"test");
        let commitments = ReadWriteMemory::commit(&batched_polys, &generators);

        let proof = ReadWriteMemoryProof::prove_memory_checking(
            &NoPreprocessing,
            &rw_memory,
            &batched_polys,
            &mut transcript,
        );

        let mut transcript = Transcript::new(b"test_transcript");
        ReadWriteMemoryProof::verify_memory_checking(
            &NoPreprocessing,
            proof,
            &commitments,
            &mut transcript,
        )
        .expect("proof should verify");
    }
}
