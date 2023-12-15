use std::cmp::max;
use std::marker::PhantomData;

use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use merlin::Transcript;
use rand::rngs::StdRng;
use rand_core::RngCore;

use crate::{
    lasso::memory_checking::{MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier},
    lasso::surge::SurgeProof,
    poly::{
        dense_mlpoly::{DensePolynomial, PolyCommitmentGens},
        identity_poly::IdentityPolynomial,
        structured_poly::{BatchablePolynomials, StructuredOpeningProof},
    },
    subprotocols::combined_table_proof::{CombinedTableCommitment, CombinedTableEvalProof},
    utils::{errors::ProofVerifyError, random::RandomTape},
};
use common::constants::{BYTES_PER_INSTRUCTION, RAM_START_ADDRESS, REGISTER_COUNT};
use common::{to_ram_address, ELFInstruction};

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
    num_ops: usize,
    rng: &mut StdRng,
) -> Vec<MemoryOp> {
    let mut memory: Vec<u64> = vec![0; max_memory_address];
    for instr in bytecode {
        let address = instr.address - RAM_START_ADDRESS + REGISTER_COUNT;
        let raw = instr.raw;
        for i in 0..(BYTES_PER_INSTRUCTION as u64) {
            // Write one byte of raw to memory
            memory[(address + i) as usize] = ((raw >> (i * 8)) & 0xff) as u64;
        }
    }

    let mut memory_trace = Vec::with_capacity(num_ops);
    for _ in 0..num_ops {
        let mut address = if rng.next_u32() % 3 == 0 {
            rng.next_u64() % max_memory_address as u64
        } else {
            rng.next_u64() % REGISTER_COUNT
        };
        if rng.next_u32() % 2 == 0 {
            let value = memory[address as usize];
            if address >= REGISTER_COUNT {
                address = address + RAM_START_ADDRESS;
            }
            memory_trace.push(MemoryOp::Read(address, value));
        } else {
            let new_value = rng.next_u64();
            memory[address as usize] = new_value;
            if address >= REGISTER_COUNT {
                address = address + RAM_START_ADDRESS;
            }
            memory_trace.push(MemoryOp::Write(address, new_value));
        }
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
        MemoryInitFinalOpenings<F, G>,
    >,
    pub commitment: MemoryCommitment<G>,
    pub timestamp_validity_proof: SurgeProof<F, G>,
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
    v_init: DensePolynomial<F>,
    /// MLE of read/write addresses. For offline memory checking, each read is paired with a "virtual" write
    /// and vice versa, so the read addresses and write addresses are the same.
    a_read_write: DensePolynomial<F>,
    /// MLE of the read values.
    v_read: DensePolynomial<F>,
    /// MLE of the write values.
    v_write: DensePolynomial<F>,
    /// MLE of the final memory state.
    v_final: DensePolynomial<F>,
    /// MLE of the read timestamps.
    t_read: DensePolynomial<F>,
    /// MLE of the write timestamps.
    t_write: DensePolynomial<F>,
    /// MLE of the final timestamps.
    t_final: DensePolynomial<F>,
}

impl<F: PrimeField, G: CurveGroup<ScalarField = F>> ReadWriteMemory<F, G> {
    #[tracing::instrument(skip_all, name = "ReadWriteMemory::new")]
    pub fn new(
        bytecode: Vec<ELFInstruction>,
        memory_trace: Vec<MemoryOp>,
        transcript: &mut Transcript,
    ) -> (Self, Vec<u64>) {
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

        let mut v_init: Vec<u64> = vec![0; memory_size];
        for instr in bytecode {
            let address = remap_address(instr.address);
            let raw = instr.raw;
            for i in 0..(BYTES_PER_INSTRUCTION as u64) {
                // Write one byte of raw to v_init
                v_init[(address + i) as usize] = ((raw >> (i * 8)) & 0xff) as u64;
            }
        }

        let mut a_read_write: Vec<u64> = Vec::with_capacity(m);
        let mut v_read: Vec<u64> = Vec::with_capacity(m);
        let mut v_write: Vec<u64> = Vec::with_capacity(m);
        let mut v_final: Vec<u64> = v_init.clone(); // TODO(moodlezoup): avoid clone
        let mut t_read: Vec<u64> = Vec::with_capacity(m);
        let mut t_write: Vec<u64> = Vec::with_capacity(m);
        let mut t_final: Vec<u64> = vec![0; memory_size];

        let mut timestamp: u64 = 0;
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

        (
            Self {
                _group: PhantomData,
                memory_size,
                v_init: DensePolynomial::from_u64(&v_init),
                a_read_write: DensePolynomial::from_u64(&a_read_write),
                v_read: DensePolynomial::from_u64(&v_read),
                v_write: DensePolynomial::from_u64(&v_write),
                v_final: DensePolynomial::from_u64(&v_final),
                t_read: DensePolynomial::from_u64(&t_read),
                t_write: DensePolynomial::from_u64(&t_write),
                t_final: DensePolynomial::from_u64(&t_final),
            },
            t_read,
        )
    }

    #[tracing::instrument(skip_all, name = "ReadWriteMemory::new")]
    pub fn get_r1cs_polys(
        bytecode: Vec<ELFInstruction>,
        memory_trace: Vec<MemoryOp>,
        transcript: &mut Transcript,
    ) -> [Vec<F>; 4] {
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

        let mut v_init: Vec<u64> = vec![0; memory_size];
        for instr in bytecode {
            let address = remap_address(instr.address);
            let raw = instr.raw;
            for i in 0..(BYTES_PER_INSTRUCTION as u64) {
                // Write one byte of raw to v_init
                v_init[(address + i) as usize] = ((raw >> (i * 8)) & 0xff) as u64;
            }
        }

        let mut a_read_write: Vec<u64> = Vec::with_capacity(m);
        let mut v_read: Vec<u64> = Vec::with_capacity(m);
        let mut v_write: Vec<u64> = Vec::with_capacity(m);
        let mut v_final: Vec<u64> = v_init.clone(); // TODO(moodlezoup): avoid clone
        let mut t_read: Vec<u64> = Vec::with_capacity(m);
        let mut t_write: Vec<u64> = Vec::with_capacity(m);
        let mut t_final: Vec<u64> = vec![0; memory_size];

        let mut timestamp: u64 = 0;
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

        [
            DensePolynomial::from_u64(&a_read_write).evals(),
            DensePolynomial::from_u64(&v_read).evals(),
            DensePolynomial::from_u64(&v_write).evals(),
            DensePolynomial::from_u64(&t_read).evals(),
        ]

        // (
        //     Self {
        //         _group: PhantomData,
        //         memory_size,
        //         v_init: DensePolynomial::from_u64(&v_init),
        //         a_read_write: DensePolynomial::from_u64(&a_read_write),
        //         v_read: DensePolynomial::from_u64(&v_read),
        //         v_write: DensePolynomial::from_u64(&v_write),
        //         v_final: DensePolynomial::from_u64(&v_final),
        //         t_read: DensePolynomial::from_u64(&t_read),
        //         t_write: DensePolynomial::from_u64(&t_write),
        //         t_final: DensePolynomial::from_u64(&t_final),
        //     },
        //     t_read,
        // )
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
    generators: MemoryCommitmentGenerators<G>,
    /// Commitments for:
    /// a_read_write, v_read, v_write, t_read, t_write
    pub read_write_commitments: CombinedTableCommitment<G>,

    /// Commitments for:
    /// v_init, v_final, t_final
    pub init_final_commitments: CombinedTableCommitment<G>,
}

pub struct MemoryCommitmentGenerators<G: CurveGroup> {
    pub gens_read_write: PolyCommitmentGens<G>,
    pub gens_init_final: PolyCommitmentGens<G>,
}

impl<F, G> BatchablePolynomials for ReadWriteMemory<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type BatchedPolynomials = BatchedMemoryPolynomials<F>;
    type Commitment = MemoryCommitment<G>;

    #[tracing::instrument(skip_all, name = "ReadWriteMemory::batch")]
    fn batch(&self) -> Self::BatchedPolynomials {
        let batched_read_write = DensePolynomial::merge(&vec![
            &self.a_read_write,
            &self.v_read,
            &self.v_write,
            &self.t_read,
            &self.t_write,
        ]);
        let batched_init_final =
            DensePolynomial::merge(&vec![&self.v_init, &self.v_final, &self.t_final]);

        Self::BatchedPolynomials {
            batched_read_write,
            batched_init_final,
        }
    }

    #[tracing::instrument(skip_all, name = "ReadWriteMemory::commit")]
    fn commit(batched_polys: &Self::BatchedPolynomials) -> Self::Commitment {
        let (gens_read_write, read_write_commitments) = batched_polys
            .batched_read_write
            .combined_commit(b"BatchedMemoryPolynomials.batched_read_write");
        let (gens_init_final, init_final_commitments) = batched_polys
            .batched_init_final
            .combined_commit(b"BatchedMemoryPolynomials.batched_init_final");

        let generators = MemoryCommitmentGenerators {
            gens_read_write,
            gens_init_final,
        };

        Self::Commitment {
            read_write_commitments,
            init_final_commitments,
            generators,
        }
    }
}

pub struct MemoryReadWriteOpenings<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    /// Evaluation of the a_read_write polynomial at the opening point.
    a_read_write_opening: F,
    /// Evaluation of the v_read polynomial at the opening point.
    v_read_opening: F,
    /// Evaluation of the v_write polynomial at the opening point.
    v_write_opening: F,
    /// Evaluation of the t_read polynomial at the opening point.
    t_read_opening: F,
    /// Evaluation of the t_write polynomial at the opening point.
    t_write_opening: F,
    read_write_opening_proof: CombinedTableEvalProof<G>,
}

impl<F, G> StructuredOpeningProof<F, G, ReadWriteMemory<F, G>> for MemoryReadWriteOpenings<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type Openings = [F; 5];

    #[tracing::instrument(skip_all, name = "MemoryReadWriteOpenings::open")]
    fn open(polynomials: &ReadWriteMemory<F, G>, opening_point: &Vec<F>) -> Self::Openings {
        [
            polynomials.a_read_write.evaluate(&opening_point),
            polynomials.v_read.evaluate(&opening_point),
            polynomials.v_write.evaluate(&opening_point),
            polynomials.t_read.evaluate(&opening_point),
            polynomials.t_write.evaluate(&opening_point),
        ]
    }

    #[tracing::instrument(skip_all, name = "MemoryReadWriteOpenings::prove_openings")]
    fn prove_openings(
        polynomials: &BatchedMemoryPolynomials<F>,
        commitment: &MemoryCommitment<G>,
        opening_point: &Vec<F>,
        openings: [F; 5],
        transcript: &mut Transcript,
        random_tape: &mut RandomTape<G>,
    ) -> Self {
        let read_write_opening_proof = CombinedTableEvalProof::prove(
            &polynomials.batched_read_write,
            &openings.to_vec(),
            &opening_point,
            &commitment.generators.gens_read_write,
            transcript,
            random_tape,
        );

        Self {
            a_read_write_opening: openings[0],
            v_read_opening: openings[1],
            v_write_opening: openings[2],
            t_read_opening: openings[3],
            t_write_opening: openings[4],
            read_write_opening_proof,
        }
    }

    fn verify_openings(
        &self,
        commitment: &MemoryCommitment<G>,
        opening_point: &Vec<F>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        let combined_openings: Vec<F> = vec![
            self.a_read_write_opening.clone(),
            self.v_read_opening.clone(),
            self.v_write_opening.clone(),
            self.t_read_opening.clone(),
            self.t_write_opening.clone(),
        ];

        self.read_write_opening_proof.verify(
            opening_point,
            &combined_openings,
            &commitment.generators.gens_read_write,
            &commitment.read_write_commitments,
            transcript,
        )
    }
}

pub struct MemoryInitFinalOpenings<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    /// Evaluation of the a_init_final polynomial at the opening point. Computed by the verifier in `compute_verifier_openings`.
    a_init_final: Option<F>,
    /// Evaluation of the v_init polynomial at the opening point.
    v_init: F,
    /// Evaluation of the v_final polynomial at the opening point.
    v_final: F,
    /// Evaluation of the t_final polynomial at the opening point.
    t_final: F,

    init_final_opening_proof: CombinedTableEvalProof<G>,
}

impl<F, G> StructuredOpeningProof<F, G, ReadWriteMemory<F, G>> for MemoryInitFinalOpenings<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type Openings = [F; 3];

    #[tracing::instrument(skip_all, name = "MemoryInitFinalOpenings::open")]
    fn open(polynomials: &ReadWriteMemory<F, G>, opening_point: &Vec<F>) -> Self::Openings {
        [
            polynomials.v_init.evaluate(&opening_point),
            polynomials.v_final.evaluate(&opening_point),
            polynomials.t_final.evaluate(&opening_point),
        ]
    }

    #[tracing::instrument(skip_all, name = "MemoryInitFinalOpenings::prove_openings")]
    fn prove_openings(
        polynomials: &BatchedMemoryPolynomials<F>,
        commitment: &MemoryCommitment<G>,
        opening_point: &Vec<F>,
        openings: [F; 3],
        transcript: &mut Transcript,
        random_tape: &mut RandomTape<G>,
    ) -> Self {
        let init_final_opening_proof = CombinedTableEvalProof::prove(
            &polynomials.batched_init_final,
            &openings.to_vec(),
            &opening_point,
            &commitment.generators.gens_init_final,
            transcript,
            random_tape,
        );

        Self {
            a_init_final: None, // Computed by verifier
            v_init: openings[0],
            v_final: openings[1],
            t_final: openings[2],
            init_final_opening_proof,
        }
    }

    fn verify_openings(
        &self,
        commitment: &MemoryCommitment<G>,
        opening_point: &Vec<F>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        self.init_final_opening_proof.verify(
            opening_point,
            &vec![self.v_init, self.v_final, self.t_final],
            &commitment.generators.gens_init_final,
            &commitment.init_final_commitments,
            transcript,
        )
    }
}

impl<F, G> MemoryCheckingProver<F, G, Self> for ReadWriteMemory<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type ReadWriteOpenings = MemoryReadWriteOpenings<F, G>;
    type InitFinalOpenings = MemoryInitFinalOpenings<F, G>;

    // (a, v, t)
    type MemoryTuple = (F, F, F);

    fn fingerprint(inputs: &(F, F, F), gamma: &F, tau: &F) -> F {
        let (a, v, t) = *inputs;
        t * gamma.square() + v * *gamma + a - tau
    }

    #[tracing::instrument(skip_all, name = "ReadWriteMemory::read_leaves")]
    fn read_leaves(&self, polynomials: &Self, gamma: &F, tau: &F) -> Vec<DensePolynomial<F>> {
        let num_ops = polynomials.a_read_write.len();
        let read_fingerprints = (0..num_ops)
            .map(|i| {
                <Self as MemoryCheckingProver<F, G, Self>>::fingerprint(
                    &(
                        polynomials.a_read_write[i],
                        polynomials.v_read[i],
                        polynomials.t_read[i],
                    ),
                    gamma,
                    tau,
                )
            })
            .collect();
        vec![DensePolynomial::new(read_fingerprints)]
    }
    #[tracing::instrument(skip_all, name = "ReadWriteMemory::write_leaves")]
    fn write_leaves(&self, polynomials: &Self, gamma: &F, tau: &F) -> Vec<DensePolynomial<F>> {
        let num_ops = polynomials.a_read_write.len();
        let write_fingerprints = (0..num_ops)
            .map(|i| {
                <Self as MemoryCheckingProver<F, G, Self>>::fingerprint(
                    &(
                        polynomials.a_read_write[i],
                        polynomials.v_write[i],
                        polynomials.t_write[i],
                    ),
                    gamma,
                    tau,
                )
            })
            .collect();
        vec![DensePolynomial::new(write_fingerprints)]
    }
    #[tracing::instrument(skip_all, name = "ReadWriteMemory::init_leaves")]
    fn init_leaves(&self, polynomials: &Self, gamma: &F, tau: &F) -> Vec<DensePolynomial<F>> {
        let init_fingerprints = (0..self.memory_size)
            .map(|i| {
                <Self as MemoryCheckingProver<F, G, Self>>::fingerprint(
                    &(F::from(i as u64), polynomials.v_init[i], F::zero()),
                    gamma,
                    tau,
                )
            })
            .collect();
        vec![DensePolynomial::new(init_fingerprints)]
    }
    #[tracing::instrument(skip_all, name = "ReadWriteMemory::final_leaves")]
    fn final_leaves(&self, polynomials: &Self, gamma: &F, tau: &F) -> Vec<DensePolynomial<F>> {
        let final_fingerprints = (0..self.memory_size)
            .map(|i| {
                <Self as MemoryCheckingProver<F, G, Self>>::fingerprint(
                    &(
                        F::from(i as u64),
                        polynomials.v_final[i],
                        polynomials.t_final[i],
                    ),
                    gamma,
                    tau,
                )
            })
            .collect();
        vec![DensePolynomial::new(final_fingerprints)]
    }

    fn protocol_name() -> &'static [u8] {
        b"Registers/RAM memory checking"
    }
}

impl<F, G> MemoryCheckingVerifier<F, G, Self> for ReadWriteMemory<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    fn compute_verifier_openings(openings: &mut Self::InitFinalOpenings, opening_point: &Vec<F>) {
        openings.a_init_final =
            Some(IdentityPolynomial::new(opening_point.len()).evaluate(opening_point));
    }

    fn read_tuples(openings: &Self::ReadWriteOpenings) -> Vec<Self::MemoryTuple> {
        vec![(
            openings.a_read_write_opening,
            openings.v_read_opening,
            openings.t_read_opening,
        )]
    }
    fn write_tuples(openings: &Self::ReadWriteOpenings) -> Vec<Self::MemoryTuple> {
        vec![(
            openings.a_read_write_opening,
            openings.v_write_opening,
            openings.t_write_opening,
        )]
    }
    fn init_tuples(openings: &Self::InitFinalOpenings) -> Vec<Self::MemoryTuple> {
        vec![(openings.a_init_final.unwrap(), openings.v_init, F::zero())]
    }
    fn final_tuples(openings: &Self::InitFinalOpenings) -> Vec<Self::MemoryTuple> {
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
    use ark_std::{log2, test_rng, One, Zero};
    use rand_chacha::rand_core::RngCore;
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
        let mut random_tape = RandomTape::new(b"test_tape");

        let (rw_memory, _): (ReadWriteMemory<Fr, EdwardsProjective>, Vec<u64>) =
            ReadWriteMemory::new(bytecode, memory_trace, &mut transcript);
        let batched_polys = rw_memory.batch();
        let commitments = ReadWriteMemory::commit(&batched_polys);

        let proof = rw_memory.prove_memory_checking(
            &rw_memory,
            &batched_polys,
            &commitments,
            &mut transcript,
            &mut random_tape,
        );

        let mut transcript = Transcript::new(b"test_transcript");
        ReadWriteMemory::verify_memory_checking(proof, &commitments, &mut transcript)
            .expect("proof should verify");
    }
}
