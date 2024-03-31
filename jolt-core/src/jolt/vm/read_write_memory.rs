use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use merlin::Transcript;
use rand::rngs::StdRng;
use rand_core::RngCore;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
#[cfg(test)]
use std::collections::HashSet;
use std::marker::PhantomData;

use crate::{
    lasso::memory_checking::{
        MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier, MultisetHashes,
    },
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        hyrax::{
            matrix_dimensions, BatchedHyraxOpeningProof, HyraxCommitment, HyraxGenerators,
            HyraxOpeningProof,
        },
        identity_poly::IdentityPolynomial,
        pedersen::PedersenGenerators,
        structured_poly::{BatchablePolynomials, StructuredOpeningProof},
    },
    subprotocols::{
        concatenated_commitment::{
            ConcatenatedPolynomialCommitment, ConcatenatedPolynomialOpeningProof,
        },
        sumcheck::SumcheckInstanceProof,
    },
    utils::{errors::ProofVerifyError, math::Math, mul_0_optimized, thread, transcript::ProofTranscript},
};
use common::constants::{
    memory_address_to_witness_index, BYTES_PER_INSTRUCTION, INPUT_START_ADDRESS, MAX_INPUT_SIZE,
    MAX_OUTPUT_SIZE, MEMORY_OPS_PER_INSTRUCTION, NUM_R1CS_POLYS, OUTPUT_START_ADDRESS,
    PANIC_ADDRESS, RAM_START_ADDRESS, RAM_WITNESS_OFFSET, REGISTER_COUNT,
};
use common::rv_trace::{ELFInstruction, JoltDevice, MemoryOp, RV32IM};
use common::to_ram_address;

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
            opcode: RV32IM::ADD,
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
    pub fn preprocess(bytecode: &Vec<ELFInstruction>) -> Self {
        let min_bytecode_address = bytecode
            .iter()
            .map(|instr| instr.address)
            .min()
            .unwrap_or(0);

        let max_bytecode_address = bytecode
            .iter()
            .map(|instr| instr.address)
            .max()
            .unwrap_or(0)
            + (BYTES_PER_INSTRUCTION as u64 - 1); // For RV32I, instructions occupy 4 bytes, so the max bytecode address is the max instruction address + 3

        let mut bytecode_bytes =
            vec![0u8; (max_bytecode_address - min_bytecode_address + 1) as usize];
        for instr in bytecode.iter() {
            let mut byte_index = instr.address - min_bytecode_address;
            let raw = instr.raw;
            for i in 0..(BYTES_PER_INSTRUCTION as u64) {
                // Write one byte of raw to bytes
                bytecode_bytes[byte_index as usize] = ((raw >> (i * 8)) & 0xff) as u8;
                byte_index += 1;
            }
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
        program_io: &JoltDevice,
        preprocessing: &ReadWriteMemoryPreprocessing,
        memory_trace: Vec<[MemoryOp; MEMORY_OPS_PER_INSTRUCTION]>,
        transcript: &mut Transcript,
    ) -> (Self, [Vec<u64>; MEMORY_OPS_PER_INSTRUCTION]) {
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
            [DensePolynomial<F>; MEMORY_OPS_PER_INSTRUCTION],
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

    #[tracing::instrument(skip_all, name = "ReadWriteMemory::get_polys_r1cs")]
    pub fn get_polys_r1cs(&self) -> (Vec<F>, Vec<F>, Vec<F>) {
        let (a_polys, v_read_polys, v_write_polys) = thread::join_triple(|| DensePolynomial::flatten(&self.a_read_write), || DensePolynomial::flatten(&self.v_read), || DensePolynomial::flatten(&self.v_write));
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
        let init_final_num_vars = (max_memory_address * 2).next_power_of_two().log_2();
        let num_read_write_generators = std::cmp::max(
            matrix_dimensions(max_trace_length.log_2(), NUM_R1CS_POLYS).1,
            matrix_dimensions(t_read_write_num_vars, 1).1,
        );
        let num_init_final_generators = matrix_dimensions(init_final_num_vars, 1).1;

        std::cmp::max(num_read_write_generators, num_init_final_generators)
    }
}

pub struct BatchedMemoryPolynomials<F: PrimeField> {
    /// Contains t_read and t_write
    pub(crate) batched_t_read_write: DensePolynomial<F>,
}

pub struct MemoryCommitment<G: CurveGroup> {
    /// Generators for a_read_write, v_read, v_write
    pub read_write_generators: HyraxGenerators<NUM_R1CS_POLYS, G>,
    pub a_v_read_write_commitments: Vec<HyraxCommitment<NUM_R1CS_POLYS, G>>,

    /// Commitments for t_read, t_write
    pub t_read_write_commitments: ConcatenatedPolynomialCommitment<G>,

    /// Commitments for v_final, t_final
    pub final_generators: HyraxGenerators<1, G>,
    pub v_final_commitment: HyraxCommitment<1, G>,
    pub t_final_commitment: HyraxCommitment<1, G>,
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
        let num = self.t_read.len() + self.t_write.len();
        let len = self.t_write[0].len();
        let batched_t_read_write =
            DensePolynomial::merge(self.t_read.par_iter().chain(self.t_write.par_iter()), num, len);

        Self::BatchedPolynomials {
            batched_t_read_write,
        }
    }

    #[tracing::instrument(skip_all, name = "ReadWriteMemory::commit")]
    fn commit(
        &self,
        batched_polys: &Self::BatchedPolynomials,
        pedersen_generators: &PedersenGenerators<G>,
    ) -> Self::Commitment {
        let read_write_num_vars = self.a_read_write[0].get_num_vars();
        let read_write_generators = HyraxGenerators::new(read_write_num_vars, pedersen_generators);
        let a_v_read_write_polys: Vec<&DensePolynomial<F>> = self
            .a_read_write
            .iter()
            .chain(self.v_read.iter())
            .chain(self.v_write.iter())
            .collect();
        let a_v_read_write_commitments = HyraxCommitment::batch_commit_polys(
            a_v_read_write_polys,
            read_write_num_vars,
            &read_write_generators,
        );

        let t_read_write_commitments = batched_polys
            .batched_t_read_write
            .combined_commit(pedersen_generators);

        let final_generators =
            HyraxGenerators::new(self.t_final.get_num_vars(), pedersen_generators);
        let (v_final_commitment, t_final_commitment) = rayon::join(
            || HyraxCommitment::commit(&self.v_final, &final_generators),
            || HyraxCommitment::commit(&self.t_final, &final_generators),
        );

        Self::Commitment {
            read_write_generators,
            a_v_read_write_commitments,
            t_read_write_commitments,
            final_generators,
            v_final_commitment,
            t_final_commitment,
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

pub struct MemoryReadWriteOpeningProof<G: CurveGroup> {
    a_v_opening_proof: BatchedHyraxOpeningProof<NUM_R1CS_POLYS, G>,
    t_opening_proof: ConcatenatedPolynomialOpeningProof<G>,
}

impl<F, G> StructuredOpeningProof<F, G, ReadWriteMemory<F, G>> for MemoryReadWriteOpenings<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type Proof = MemoryReadWriteOpeningProof<G>;

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
        polynomials: &ReadWriteMemory<F, G>,
        batched_polynomials: &BatchedMemoryPolynomials<F>,
        opening_point: &Vec<F>,
        openings: &Self,
        transcript: &mut Transcript,
    ) -> Self::Proof {
        let a_v_polys = polynomials
            .a_read_write
            .iter()
            .chain(polynomials.v_read.iter())
            .chain(polynomials.v_write.iter())
            .collect::<Vec<_>>();
        let a_v_openings = openings
            .a_read_write_opening
            .into_iter()
            .chain(openings.v_read_opening.into_iter())
            .chain(openings.v_write_opening.into_iter())
            .collect::<Vec<_>>();
        let a_v_opening_proof =
            BatchedHyraxOpeningProof::prove(&a_v_polys, &opening_point, &a_v_openings, transcript);

        let t_opening_proof = ConcatenatedPolynomialOpeningProof::prove(
            &batched_polynomials.batched_t_read_write,
            &opening_point,
            &openings
                .t_read_opening
                .into_iter()
                .chain(openings.t_write_opening.into_iter())
                .collect::<Vec<_>>(),
            transcript,
        );

        MemoryReadWriteOpeningProof {
            a_v_opening_proof,
            t_opening_proof,
        }
    }

    fn verify_openings(
        &self,
        opening_proof: &Self::Proof,
        commitment: &MemoryCommitment<G>,
        opening_point: &Vec<F>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        let a_v_openings = self
            .a_read_write_opening
            .into_iter()
            .chain(self.v_read_opening.into_iter())
            .chain(self.v_write_opening.into_iter())
            .collect::<Vec<_>>();
        opening_proof.a_v_opening_proof.verify(
            &commitment.read_write_generators,
            opening_point,
            &a_v_openings,
            &commitment
                .a_v_read_write_commitments
                .iter()
                .collect::<Vec<_>>(),
            transcript,
        )?;

        opening_proof.t_opening_proof.verify(
            opening_point,
            &self
                .t_read_opening
                .into_iter()
                .chain(self.t_write_opening.into_iter())
                .collect::<Vec<_>>(),
            &commitment.t_read_write_commitments,
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
    /// Evaluation of the v_init polynomial at the opening point. Computed by the verifier in `compute_verifier_openings`.
    v_init: Option<F>,
    /// Evaluation of the v_final polynomial at the opening point.
    v_final: F,
    /// Evaluation of the t_final polynomial at the opening point.
    t_final: F,
}

pub struct MemoryInitFinalOpeningProof<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    v_t_opening_proof: BatchedHyraxOpeningProof<1, G>,
}

impl<F, G> StructuredOpeningProof<F, G, ReadWriteMemory<F, G>> for MemoryInitFinalOpenings<F>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type Proof = MemoryInitFinalOpeningProof<F, G>;
    type Preprocessing = ReadWriteMemoryPreprocessing;

    #[tracing::instrument(skip_all, name = "MemoryInitFinalOpenings::open")]
    fn open(polynomials: &ReadWriteMemory<F, G>, opening_point: &Vec<F>) -> Self {
        let chis = EqPolynomial::new(opening_point.to_vec()).evals();
        let (v_final, t_final) = rayon::join(
            || polynomials.v_final.evaluate_at_chi(&chis),
            || polynomials.t_final.evaluate_at_chi(&chis),
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
        polynomials: &ReadWriteMemory<F, G>,
        _: &BatchedMemoryPolynomials<F>,
        opening_point: &Vec<F>,
        openings: &Self,
        transcript: &mut Transcript,
    ) -> Self::Proof {
        let v_t_opening_proof = BatchedHyraxOpeningProof::prove(
            &[&polynomials.v_final, &polynomials.t_final],
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
        opening_proof: &Self::Proof,
        commitment: &MemoryCommitment<G>,
        opening_point: &Vec<F>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        opening_proof.v_t_opening_proof.verify(
            &commitment.final_generators,
            opening_point,
            &vec![self.v_final, self.t_final],
            &[
                &commitment.v_final_commitment,
                &commitment.t_final_commitment,
            ],
            transcript,
        )?;

        Ok(())
    }
}

impl<F, G> MemoryCheckingProver<F, G, ReadWriteMemory<F, G>> for ReadWriteMemoryProof<F, G>
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

impl<F, G> MemoryCheckingVerifier<F, G, ReadWriteMemory<F, G>> for ReadWriteMemoryProof<F, G>
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
                (
                    openings.a_read_write_opening[i],
                    openings.v_read_opening[i],
                    openings.t_read_opening[i],
                )
            })
            .collect()
    }
    fn write_tuples(
        &_: &Self::Preprocessing,
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
            &commitment.final_generators,
            transcript,
            &r_sumcheck,
            &proof.opening,
            &commitment.v_final_commitment,
        )
    }
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
        polynomials: &ReadWriteMemory<F, G>,
        batched_polynomials: &BatchedMemoryPolynomials<F>,
        read_timestamps: [Vec<u64>; MEMORY_OPS_PER_INSTRUCTION],
        program_io: &JoltDevice,
        generators: &PedersenGenerators<G>,
        transcript: &mut Transcript,
    ) -> Self {
        let memory_checking_proof = ReadWriteMemoryProof::prove_memory_checking(
            preprocessing,
            polynomials,
            batched_polynomials,
            transcript,
        );

        let output_proof = OutputSumcheckProof::prove_outputs(polynomials, program_io, transcript);

        let timestamp_validity_proof = TimestampValidityProof::prove(
            read_timestamps,
            polynomials,
            batched_polynomials,
            &generators,
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
        preprocessing: &mut ReadWriteMemoryPreprocessing,
        commitment: &MemoryCommitment<G>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        ReadWriteMemoryProof::verify_memory_checking(
            preprocessing,
            self.memory_checking_proof,
            commitment,
            transcript,
        )?;
        OutputSumcheckProof::verify(&self.output_proof, preprocessing, commitment, transcript)?;
        TimestampValidityProof::verify(&mut self.timestamp_validity_proof, commitment, transcript)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::{Fr, G1Projective};
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

        let mut preprocessing = ReadWriteMemoryPreprocessing::preprocess(&bytecode);
        let (rw_memory, _): (ReadWriteMemory<Fr, G1Projective>, _) = ReadWriteMemory::new(
            &JoltDevice::new(),
            &preprocessing,
            memory_trace,
            &mut transcript,
        );
        let batched_polys = rw_memory.batch();
        let generators = PedersenGenerators::new(1 << 10, b"test");
        let commitments = rw_memory.commit(&batched_polys, &generators);

        let proof = ReadWriteMemoryProof::prove_memory_checking(
            &preprocessing,
            &rw_memory,
            &batched_polys,
            &mut transcript,
        );

        let mut transcript = Transcript::new(b"test_transcript");
        preprocessing.program_io = Some(JoltDevice::new());
        ReadWriteMemoryProof::verify_memory_checking(
            &preprocessing,
            proof,
            &commitments,
            &mut transcript,
        )
        .expect("proof should verify");
    }
}
