use crate::field::JoltField;
use crate::jolt::instruction::JoltInstructionSet;
use crate::lasso::memory_checking::{
    ExogenousOpenings, Initializable, StructuredPolynomialData, VerifierComputedOpening,
};
use crate::poly::compact_polynomial::{CompactPolynomial, SmallScalar};
use crate::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation};
use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use crate::subprotocols::grand_product::BatchedDenseGrandProduct;
use crate::utils::thread::unsafe_allocate_zero_vec;
use rayon::prelude::*;
#[cfg(test)]
use std::collections::HashSet;
use std::marker::PhantomData;

use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::utils::transcript::Transcript;
use crate::{
    lasso::memory_checking::{
        MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier, MultisetHashes,
    },
    poly::{
        dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial, identity_poly::IdentityPolynomial,
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{errors::ProofVerifyError, math::Math},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::{
    BYTES_PER_INSTRUCTION, MEMORY_OPS_PER_INSTRUCTION, RAM_START_ADDRESS, REGISTER_COUNT,
};
use common::rv_trace::{JoltDevice, MemoryLayout, MemoryOp};

use super::{timestamp_range_check::TimestampValidityProof, JoltCommitments};
use super::{JoltPolynomials, JoltStuff, JoltTraceStep};

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct ReadWriteMemoryPreprocessing {
    min_bytecode_address: u64,
    bytecode_words: Vec<u32>,
    // HACK: The verifier will populate this field by copying inputs/outputs from the
    // `ReadWriteMemoryProof` and the memory layout from preprocessing.
    // Having `program_io` in this preprocessing struct allows the verifier to access it
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

        let num_words = max_bytecode_address.next_multiple_of(4) / 4 - min_bytecode_address / 4 + 1;
        let mut bytecode_words = vec![0u32; num_words as usize];
        // Convert bytes into words and populate `bytecode_words`
        for chunk in
            memory_init.chunk_by(|(address_a, _), (address_b, _)| address_a / 4 == address_b / 4)
        {
            let mut word = [0u8; 4];
            for (address, byte) in chunk {
                word[(address % 4) as usize] = *byte;
            }
            let word = u32::from_le_bytes(word);
            let remapped_index = (chunk[0].0 / 4 - min_bytecode_address / 4) as usize;
            bytecode_words[remapped_index] = word;
        }

        Self {
            min_bytecode_address,
            bytecode_words,
            program_io: None,
        }
    }
}

fn memory_address_to_witness_index(address: u64, memory_layout: &MemoryLayout) -> usize {
    (REGISTER_COUNT + (address - memory_layout.input_start) / 4) as usize
}

fn remap_address(a: u64, memory_layout: &MemoryLayout) -> u64 {
    if a >= memory_layout.input_start {
        memory_address_to_witness_index(a, memory_layout) as u64
    } else if a < REGISTER_COUNT {
        // If a < REGISTER_COUNT, it is one of the registers and doesn't
        // need to be remapped
        a
    } else {
        panic!("Unexpected address {a}")
    }
}

const RS1: usize = 0;
const RS2: usize = 1;
const RD: usize = 2;
const RAM: usize = 3;

#[derive(Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct ReadWriteMemoryStuff<T: CanonicalSerialize + CanonicalDeserialize> {
    /// Read/write addresses. For offline memory checking, each read is paired with a "virtual" write
    /// and vice versa, so the read addresses and write addresses are the same.
    pub a_ram: T,
    /// RD read_value
    pub v_read_rd: T,
    /// RS1 read_value
    pub v_read_rs1: T,
    /// RS2 read_value
    pub v_read_rs2: T,
    /// RAM read_value
    pub v_read_ram: T,
    /// RD write value
    pub v_write_rd: T,
    /// RAM write value
    pub v_write_ram: T,
    /// Final memory state.
    pub v_final: T,
    /// RD read timestamp
    pub t_read_rd: T,
    /// RS1 read timestamp
    pub t_read_rs1: T,
    /// RS2 read timestamp
    pub t_read_rs2: T,
    /// RAM read timestamp
    pub t_read_ram: T,
    /// Final timestamps.
    pub t_final: T,

    a_init_final: VerifierComputedOpening<T>,
    /// Initial memory values. RAM is initialized to contain the program bytecode and inputs.
    v_init: VerifierComputedOpening<T>,
    identity: VerifierComputedOpening<T>,
}

impl<T: CanonicalSerialize + CanonicalDeserialize> StructuredPolynomialData<T>
    for ReadWriteMemoryStuff<T>
{
    fn read_write_values(&self) -> Vec<&T> {
        vec![
            &self.a_ram,
            &self.v_read_rd,
            &self.v_read_rs1,
            &self.v_read_rs2,
            &self.v_read_ram,
            &self.v_write_rd,
            &self.v_write_ram,
            &self.t_read_rd,
            &self.t_read_rs1,
            &self.t_read_rs2,
            &self.t_read_ram,
        ]
    }

    fn read_write_values_mut(&mut self) -> Vec<&mut T> {
        vec![
            &mut self.a_ram,
            &mut self.v_read_rd,
            &mut self.v_read_rs1,
            &mut self.v_read_rs2,
            &mut self.v_read_ram,
            &mut self.v_write_rd,
            &mut self.v_write_ram,
            &mut self.t_read_rd,
            &mut self.t_read_rs1,
            &mut self.t_read_rs2,
            &mut self.t_read_ram,
        ]
    }

    fn init_final_values(&self) -> Vec<&T> {
        vec![&self.v_final, &self.t_final]
    }

    fn init_final_values_mut(&mut self) -> Vec<&mut T> {
        vec![&mut self.v_final, &mut self.t_final]
    }
}

/// Note –– F: JoltField bound is not enforced.
///
/// See issue #112792 <https://github.com/rust-lang/rust/issues/112792>.
/// Adding #![feature(lazy_type_alias)] to the crate attributes seem to break
/// `alloy_sol_types`.
pub type ReadWriteMemoryPolynomials<F: JoltField> = ReadWriteMemoryStuff<MultilinearPolynomial<F>>;
/// Note –– F: JoltField bound is not enforced.
///
/// See issue #112792 <https://github.com/rust-lang/rust/issues/112792>.
/// Adding #![feature(lazy_type_alias)] to the crate attributes seem to break
/// `alloy_sol_types`.
pub type ReadWriteMemoryOpenings<F: JoltField> = ReadWriteMemoryStuff<F>;
/// Note –– PCS: CommitmentScheme bound is not enforced.
///
/// See issue #112792 <https://github.com/rust-lang/rust/issues/112792>.
/// Adding #![feature(lazy_type_alias)] to the crate attributes seem to break
/// `alloy_sol_types`.
pub type ReadWriteMemoryCommitments<
    PCS: CommitmentScheme<ProofTranscript>,
    ProofTranscript: Transcript,
> = ReadWriteMemoryStuff<PCS::Commitment>;

impl<T: CanonicalSerialize + CanonicalDeserialize + Default>
    Initializable<T, ReadWriteMemoryPreprocessing> for ReadWriteMemoryStuff<T>
{
}

#[derive(Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct RegisterAddressOpenings<F: JoltField> {
    pub a_rd: F,
    pub a_rs1: F,
    pub a_rs2: F,
}

impl<F: JoltField> ExogenousOpenings<F> for RegisterAddressOpenings<F> {
    fn openings(&self) -> Vec<&F> {
        vec![&self.a_rd, &self.a_rs1, &self.a_rs2]
    }

    fn openings_mut(&mut self) -> Vec<&mut F> {
        vec![&mut self.a_rd, &mut self.a_rs1, &mut self.a_rs2]
    }

    fn exogenous_data<T: CanonicalSerialize + CanonicalDeserialize + Sync>(
        polys_or_commitments: &JoltStuff<T>,
    ) -> Vec<&T> {
        vec![
            &polys_or_commitments.bytecode.v_read_write[2],
            &polys_or_commitments.bytecode.v_read_write[3],
            &polys_or_commitments.bytecode.v_read_write[4],
        ]
    }
}

fn map_to_polys<F: JoltField, const N: usize>(
    vals: [Vec<u32>; N],
) -> [MultilinearPolynomial<F>; N] {
    vals.into_par_iter()
        .map(MultilinearPolynomial::from)
        .collect::<Vec<MultilinearPolynomial<F>>>()
        .try_into()
        .unwrap()
}

impl<F: JoltField> ReadWriteMemoryPolynomials<F> {
    #[tracing::instrument(skip_all, name = "ReadWriteMemoryPolynomials::generate_witness")]
    pub fn generate_witness<InstructionSet: JoltInstructionSet>(
        program_io: &JoltDevice,
        preprocessing: &ReadWriteMemoryPreprocessing,
        trace: &[JoltTraceStep<InstructionSet>],
    ) -> Self {
        assert!(program_io.inputs.len() <= program_io.memory_layout.max_input_size as usize);
        assert!(program_io.outputs.len() <= program_io.memory_layout.max_output_size as usize);

        let m = trace.len();
        assert!(m.is_power_of_two());

        let max_trace_address = trace
            .iter()
            .map(|step| match step.memory_ops[RAM] {
                MemoryOp::Read(a) => remap_address(a, &program_io.memory_layout),
                MemoryOp::Write(a, _) => remap_address(a, &program_io.memory_layout),
            })
            .max()
            .unwrap();

        let memory_size = max_trace_address.next_power_of_two() as usize;
        let mut v_init: Vec<u32> = vec![0; memory_size];
        // Copy bytecode
        let mut v_init_index = memory_address_to_witness_index(
            preprocessing.min_bytecode_address,
            &program_io.memory_layout,
        );
        for word in preprocessing.bytecode_words.iter() {
            v_init[v_init_index] = *word;
            v_init_index += 1;
        }
        // Copy input bytes
        v_init_index = memory_address_to_witness_index(
            program_io.memory_layout.input_start,
            &program_io.memory_layout,
        );
        // Convert input bytes into words and populate `v_init`
        for chunk in program_io.inputs.chunks(4) {
            let mut word = [0u8; 4];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u32::from_le_bytes(word);
            v_init[v_init_index] = word;
            v_init_index += 1;
        }

        #[cfg(test)]
        let mut init_tuples: HashSet<(usize, u32, u32)> = HashSet::new();
        #[cfg(test)]
        {
            for (a, v) in v_init.iter().enumerate() {
                init_tuples.insert((a, *v, 0));
            }
        }
        #[cfg(test)]
        let mut read_tuples: HashSet<(usize, u32, u32)> = HashSet::new();
        #[cfg(test)]
        let mut write_tuples: HashSet<(usize, u32, u32)> = HashSet::new();

        let mut a_ram: Vec<u32> = Vec::with_capacity(m);

        let mut v_read_rs1: Vec<u32> = Vec::with_capacity(m);
        let mut v_read_rs2: Vec<u32> = Vec::with_capacity(m);
        let mut v_read_rd: Vec<u32> = Vec::with_capacity(m);
        let mut v_read_ram: Vec<u32> = Vec::with_capacity(m);

        let mut t_read_rs1: Vec<u32> = Vec::with_capacity(m);
        let mut t_read_rs2: Vec<u32> = Vec::with_capacity(m);
        let mut t_read_rd: Vec<u32> = Vec::with_capacity(m);
        let mut t_read_ram: Vec<u32> = Vec::with_capacity(m);

        let mut v_write_rd: Vec<u32> = Vec::with_capacity(m);
        let mut v_write_ram: Vec<u32> = Vec::with_capacity(m);

        let mut t_final = vec![0; memory_size];
        let mut v_final = v_init.clone();

        let span = tracing::span!(tracing::Level::DEBUG, "memory_trace_processing");
        let _enter = span.enter();

        for (i, step) in trace.iter().enumerate() {
            let timestamp = i as u32;

            match step.memory_ops[RS1] {
                MemoryOp::Read(a) => {
                    assert!(a < REGISTER_COUNT);
                    let a = a as usize;
                    let v = v_final[a];

                    #[cfg(test)]
                    {
                        read_tuples.insert((a, v, t_final[a]));
                        write_tuples.insert((a, v, timestamp));
                    }

                    v_read_rs1.push(v);
                    t_read_rs1.push(t_final[a]);
                    t_final[a] = timestamp;
                }
                MemoryOp::Write(a, v) => {
                    panic!("Unexpected rs1 MemoryOp::Write({a}, {v})");
                }
            };

            match step.memory_ops[RS2] {
                MemoryOp::Read(a) => {
                    assert!(a < REGISTER_COUNT);
                    let a = a as usize;
                    let v = v_final[a];

                    #[cfg(test)]
                    {
                        read_tuples.insert((a, v, t_final[a]));
                        write_tuples.insert((a, v, timestamp));
                    }

                    v_read_rs2.push(v);
                    t_read_rs2.push(t_final[a]);
                    t_final[a] = timestamp;
                }
                MemoryOp::Write(a, v) => {
                    panic!("Unexpected rs2 MemoryOp::Write({a}, {v})")
                }
            };

            match step.memory_ops[RD] {
                MemoryOp::Read(a) => {
                    panic!("Unexpected rd MemoryOp::Read({a})")
                }
                MemoryOp::Write(a, v_new) => {
                    assert!(a < REGISTER_COUNT);
                    let a = a as usize;
                    let v_old = v_final[a];

                    #[cfg(test)]
                    {
                        read_tuples.insert((a, v_old, t_final[a]));
                        write_tuples.insert((a, v_new as u32, timestamp));
                    }

                    v_read_rd.push(v_old);
                    t_read_rd.push(t_final[a]);
                    v_write_rd.push(v_new as u32);
                    v_final[a] = v_new as u32;
                    t_final[a] = timestamp;
                }
            };

            match step.memory_ops[RAM] {
                MemoryOp::Read(a) => {
                    debug_assert!(a % 4 == 0);
                    let remapped_a = remap_address(a, &program_io.memory_layout) as usize;
                    let v = v_final[remapped_a];

                    #[cfg(test)]
                    {
                        read_tuples.insert((remapped_a, v, t_final[remapped_a]));
                        write_tuples.insert((remapped_a, v, timestamp));
                    }

                    a_ram.push(remapped_a as u32);
                    v_read_ram.push(v);
                    t_read_ram.push(t_final[remapped_a]);
                    v_write_ram.push(v);
                    t_final[remapped_a] = timestamp;
                }
                MemoryOp::Write(a, v_new) => {
                    debug_assert!(a % 4 == 0);
                    let remapped_a = remap_address(a, &program_io.memory_layout) as usize;
                    let v_old = v_final[remapped_a];

                    #[cfg(test)]
                    {
                        read_tuples.insert((remapped_a, v_old, t_final[remapped_a]));
                        write_tuples.insert((remapped_a, v_new as u32, timestamp));
                    }

                    a_ram.push(remapped_a as u32);
                    v_read_ram.push(v_old);
                    t_read_ram.push(t_final[remapped_a]);
                    v_write_ram.push(v_new as u32);
                    v_final[remapped_a] = v_new as u32;
                    t_final[remapped_a] = timestamp;
                }
            }
        }

        drop(_enter);
        drop(span);

        #[cfg(test)]
        {
            let mut final_tuples: HashSet<(usize, u32, u32)> = HashSet::new();
            for (a, (v, t)) in v_final.iter().zip(t_final.iter()).enumerate() {
                final_tuples.insert((a, *v, *t));
            }

            let init_write: HashSet<_> = init_tuples.union(&write_tuples).collect();
            let read_final: HashSet<_> = read_tuples.union(&final_tuples).collect();
            let set_difference: Vec<_> = init_write.symmetric_difference(&read_final).collect();
            assert_eq!(set_difference.len(), 0);
        }

        let [a_ram, v_read_rd, v_read_rs1, v_read_rs2, v_read_ram, v_write_rd, v_write_ram, v_final, t_read_rd_poly, t_read_rs1_poly, t_read_rs2_poly, t_read_ram_poly, t_final, v_init] =
            map_to_polys([
                a_ram,
                v_read_rd,
                v_read_rs1,
                v_read_rs2,
                v_read_ram,
                v_write_rd,
                v_write_ram,
                v_final,
                t_read_rd,
                t_read_rs1,
                t_read_rs2,
                t_read_ram,
                t_final,
                v_init,
            ]);

        ReadWriteMemoryPolynomials {
            a_ram,
            v_read_rd,
            v_read_rs1,
            v_read_rs2,
            v_read_ram,
            v_write_rd,
            v_write_ram,
            v_final,
            t_read_rd: t_read_rd_poly,
            t_read_rs1: t_read_rs1_poly,
            t_read_rs2: t_read_rs2_poly,
            t_read_ram: t_read_ram_poly,
            t_final,
            v_init: Some(v_init),
            a_init_final: None,
            identity: None,
        }
    }
}

impl<F, PCS, ProofTranscript> MemoryCheckingProver<F, PCS, ProofTranscript>
    for ReadWriteMemoryProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    type ReadWriteGrandProduct = BatchedDenseGrandProduct<F>;
    type InitFinalGrandProduct = BatchedDenseGrandProduct<F>;

    type Polynomials = ReadWriteMemoryPolynomials<F>;
    type Openings = ReadWriteMemoryOpenings<F>;
    type Commitments = ReadWriteMemoryCommitments<PCS, ProofTranscript>;
    type ExogenousOpenings = RegisterAddressOpenings<F>;

    type Preprocessing = ReadWriteMemoryPreprocessing;

    // (a, v, t)
    type MemoryTuple = (F, F, F);

    fn fingerprint(inputs: &(F, F, F), gamma: &F, tau: &F) -> F {
        let (a, v, t) = *inputs;
        t * gamma.square() + v * *gamma + a - *tau
    }

    #[tracing::instrument(skip_all, name = "ReadWriteMemory::compute_leaves")]
    fn compute_leaves<'a>(
        _: &Self::Preprocessing,
        polynomials: &Self::Polynomials,
        jolt_polynomials: &'a JoltPolynomials<F>,
        gamma: &F,
        tau: &F,
    ) -> ((Vec<F>, usize), (Vec<F>, usize)) {
        let gamma_squared = gamma.square();
        let gamma = *gamma;

        let num_ops = polynomials.a_ram.len();
        let memory_size = polynomials.v_final.len();

        let a_rd: &CompactPolynomial<u8, F> = (&jolt_polynomials.bytecode.v_read_write[2])
            .try_into()
            .unwrap();
        let a_rs1: &CompactPolynomial<u8, F> = (&jolt_polynomials.bytecode.v_read_write[3])
            .try_into()
            .unwrap();
        let a_rs2: &CompactPolynomial<u8, F> = (&jolt_polynomials.bytecode.v_read_write[4])
            .try_into()
            .unwrap();
        let a_ram: &CompactPolynomial<u32, F> = (&polynomials.a_ram).try_into().unwrap();
        let v_read_rs1: &CompactPolynomial<u32, F> = (&polynomials.v_read_rs1).try_into().unwrap();
        let v_read_rs2: &CompactPolynomial<u32, F> = (&polynomials.v_read_rs2).try_into().unwrap();
        let v_read_rd: &CompactPolynomial<u32, F> = (&polynomials.v_read_rd).try_into().unwrap();
        let v_read_ram: &CompactPolynomial<u32, F> = (&polynomials.v_read_ram).try_into().unwrap();
        let v_write_rd: &CompactPolynomial<u32, F> = (&polynomials.v_write_rd).try_into().unwrap();
        let v_write_ram: &CompactPolynomial<u32, F> =
            (&polynomials.v_write_ram).try_into().unwrap();
        let t_read_rs1: &CompactPolynomial<u32, F> = (&polynomials.t_read_rs1).try_into().unwrap();
        let t_read_rs2: &CompactPolynomial<u32, F> = (&polynomials.t_read_rs2).try_into().unwrap();
        let t_read_rd: &CompactPolynomial<u32, F> = (&polynomials.t_read_rd).try_into().unwrap();
        let t_read_ram: &CompactPolynomial<u32, F> = (&polynomials.t_read_ram).try_into().unwrap();

        let mut read_write_leaves: Vec<F> =
            unsafe_allocate_zero_vec(2 * MEMORY_OPS_PER_INSTRUCTION * num_ops);
        for (i, chunk) in read_write_leaves.chunks_mut(2 * num_ops).enumerate() {
            chunk[..num_ops]
                .par_iter_mut()
                .enumerate()
                .for_each(|(j, read_fingerprint)| {
                    match i {
                        RS1 => {
                            *read_fingerprint = t_read_rs1[j].field_mul(gamma_squared)
                                + v_read_rs1[j].field_mul(gamma)
                                + F::from_u8(a_rs1[j])
                                - *tau;
                        }
                        RS2 => {
                            *read_fingerprint = t_read_rs2[j].field_mul(gamma_squared)
                                + v_read_rs2[j].field_mul(gamma)
                                + F::from_u8(a_rs2[j])
                                - *tau;
                        }
                        RD => {
                            *read_fingerprint = t_read_rd[j].field_mul(gamma_squared)
                                + v_read_rd[j].field_mul(gamma)
                                + F::from_u8(a_rd[j])
                                - *tau;
                        }
                        RAM => {
                            *read_fingerprint = t_read_ram[j].field_mul(gamma_squared)
                                + v_read_ram[j].field_mul(gamma)
                                + F::from_u32(a_ram[j])
                                - *tau;
                        }
                        _ => unreachable!(),
                    };
                });

            chunk[num_ops..].par_iter_mut().enumerate().for_each(
                |(j, write_fingerprint)| match i {
                    RS1 => {
                        *write_fingerprint = (j as u64).field_mul(gamma_squared)
                            + v_read_rs1[j].field_mul(gamma)
                            + F::from_u8(a_rs1[j])
                            - *tau;
                    }
                    RS2 => {
                        *write_fingerprint = (j as u64).field_mul(gamma_squared)
                            + v_read_rs2[j].field_mul(gamma)
                            + F::from_u8(a_rs2[j])
                            - *tau;
                    }
                    RD => {
                        *write_fingerprint = (j as u64).field_mul(gamma_squared)
                            + v_write_rd[j].field_mul(gamma)
                            + F::from_u8(a_rd[j])
                            - *tau
                    }
                    RAM => {
                        *write_fingerprint = (j as u64).field_mul(gamma_squared)
                            + v_write_ram[j].field_mul(gamma)
                            + F::from_u32(a_ram[j])
                            - *tau;
                    }
                    _ => unreachable!(),
                },
            );
        }

        let v_init: &CompactPolynomial<u32, F> =
            polynomials.v_init.as_ref().unwrap().try_into().unwrap();
        let init_fingerprints: Vec<F> = (0..memory_size)
            .into_par_iter()
            .map(|i| /* 0 * gamma^2 + */ v_init[i].field_mul(gamma) + F::from_u32(i as u32) - *tau)
            .collect();

        let v_final: &CompactPolynomial<u32, F> = (&polynomials.v_final).try_into().unwrap();
        let t_final: &CompactPolynomial<u32, F> = (&polynomials.t_final).try_into().unwrap();
        let final_fingerprints = (0..memory_size)
            .into_par_iter()
            .map(|i| {
                t_final[i].field_mul(gamma_squared)
                    + v_final[i].field_mul(gamma)
                    + F::from_u32(i as u32)
                    - *tau
            })
            .collect();

        (
            (read_write_leaves, 2 * MEMORY_OPS_PER_INSTRUCTION),
            ([init_fingerprints, final_fingerprints].concat(), 2), // TODO(moodlezoup): Avoid concat
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

impl<F, PCS, ProofTranscript> MemoryCheckingVerifier<F, PCS, ProofTranscript>
    for ReadWriteMemoryProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    fn compute_verifier_openings(
        openings: &mut Self::Openings,
        preprocessing: &Self::Preprocessing,
        r_read_write: &[F],
        r_init_final: &[F],
    ) {
        openings.identity =
            Some(IdentityPolynomial::new(r_read_write.len()).evaluate(r_read_write));

        openings.a_init_final =
            Some(IdentityPolynomial::new(r_init_final.len()).evaluate(r_init_final));

        let memory_layout = &preprocessing.program_io.as_ref().unwrap().memory_layout;

        // TODO(moodlezoup): Compute opening without instantiating v_init polynomial itself
        let memory_size = r_init_final.len().pow2();
        let mut v_init: Vec<u64> = vec![0; memory_size];
        // Copy bytecode
        let mut v_init_index =
            memory_address_to_witness_index(preprocessing.min_bytecode_address, memory_layout);
        for word in preprocessing.bytecode_words.iter() {
            v_init[v_init_index] = *word as u64;
            v_init_index += 1;
        }
        v_init_index = memory_address_to_witness_index(memory_layout.input_start, memory_layout);
        // Convert input bytes into words and populate `v_init`
        for chunk in preprocessing.program_io.as_ref().unwrap().inputs.chunks(4) {
            let mut word = [0u8; 4];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u32::from_le_bytes(word);
            v_init[v_init_index] = word as u64;
            v_init_index += 1;
        }

        openings.v_init = Some(DensePolynomial::from_u64(&v_init).evaluate(r_init_final));
    }

    fn read_tuples(
        &_: &Self::Preprocessing,
        openings: &Self::Openings,
        register_address_openings: &RegisterAddressOpenings<F>,
    ) -> Vec<Self::MemoryTuple> {
        vec![
            (
                register_address_openings.a_rs1,
                openings.v_read_rs1,
                openings.t_read_rs1,
            ),
            (
                register_address_openings.a_rs2,
                openings.v_read_rs2,
                openings.t_read_rs2,
            ),
            (
                register_address_openings.a_rd,
                openings.v_read_rd,
                openings.t_read_rd,
            ),
            (openings.a_ram, openings.v_read_ram, openings.t_read_ram),
        ]
    }
    fn write_tuples(
        &_: &Self::Preprocessing,
        openings: &Self::Openings,
        register_address_openings: &RegisterAddressOpenings<F>,
    ) -> Vec<Self::MemoryTuple> {
        // Write timestamp is always equal to the global timestamp
        vec![
            (
                register_address_openings.a_rs1,
                openings.v_read_rs1, // For rs1 and rs2, v_write = v_read
                openings.identity.unwrap(),
            ),
            (
                register_address_openings.a_rs2,
                openings.v_read_rs2, // For rs1 and rs2, v_write = v_read
                openings.identity.unwrap(),
            ),
            (
                register_address_openings.a_rd,
                openings.v_write_rd,
                openings.identity.unwrap(),
            ),
            (
                openings.a_ram,
                openings.v_write_ram,
                openings.identity.unwrap(),
            ),
        ]
    }
    fn init_tuples(
        &_: &Self::Preprocessing,
        openings: &Self::Openings,
        _: &RegisterAddressOpenings<F>,
    ) -> Vec<Self::MemoryTuple> {
        vec![(
            openings.a_init_final.unwrap(),
            openings.v_init.unwrap(),
            F::zero(),
        )]
    }
    fn final_tuples(
        &_: &Self::Preprocessing,
        openings: &Self::Openings,
        _: &RegisterAddressOpenings<F>,
    ) -> Vec<Self::MemoryTuple> {
        vec![(
            openings.a_init_final.unwrap(),
            openings.v_final,
            openings.t_final,
        )]
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct OutputSumcheckProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    _pcs: PhantomData<(PCS, ProofTranscript)>,
    num_rounds: usize,
    /// Sumcheck proof that v_final is equal to the program outputs at the relevant indices.
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    /// Opening of v_final at the random point chosen over the course of sumcheck
    opening: F,
}

impl<F, PCS, ProofTranscript> OutputSumcheckProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    fn prove_outputs(
        polynomials: &ReadWriteMemoryPolynomials<F>,
        program_io: &JoltDevice,
        opening_accumulator: &mut ProverOpeningAccumulator<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Self {
        let memory_size = polynomials.v_final.len();
        let num_rounds = memory_size.log_2();
        let r_eq: Vec<F> = transcript.challenge_vector(num_rounds);
        let eq = MultilinearPolynomial::from(EqPolynomial::evals(&r_eq));

        let input_start_index = memory_address_to_witness_index(
            program_io.memory_layout.input_start,
            &program_io.memory_layout,
        ) as u64;
        let ram_start_index =
            memory_address_to_witness_index(RAM_START_ADDRESS, &program_io.memory_layout) as u64;

        let io_witness_range: Vec<u8> = (0..memory_size as u64)
            .map(|i| {
                if i >= input_start_index && i < ram_start_index {
                    1
                } else {
                    0
                }
            })
            .collect();

        let mut v_io: Vec<u32> = vec![0; memory_size];
        let mut input_index = memory_address_to_witness_index(
            program_io.memory_layout.input_start,
            &program_io.memory_layout,
        );
        // Convert input bytes into words and populate `v_io`
        for chunk in program_io.inputs.chunks(4) {
            let mut word = [0u8; 4];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u32::from_le_bytes(word);
            v_io[input_index] = word;
            input_index += 1;
        }
        let mut output_index = memory_address_to_witness_index(
            program_io.memory_layout.output_start,
            &program_io.memory_layout,
        );
        // Convert output bytes into words and populate `v_io`
        for chunk in program_io.outputs.chunks(4) {
            let mut word = [0u8; 4];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u32::from_le_bytes(word);
            v_io[output_index] = word;
            output_index += 1;
        }

        // Copy panic bit
        v_io[memory_address_to_witness_index(
            program_io.memory_layout.panic,
            &program_io.memory_layout,
        )] = program_io.panic as u32;
        if !program_io.panic {
            // Set termination bit
            v_io[memory_address_to_witness_index(
                program_io.memory_layout.termination,
                &program_io.memory_layout,
            )] = 1;
        }

        let mut sumcheck_polys = vec![
            eq,
            MultilinearPolynomial::from(io_witness_range),
            polynomials.v_final.clone(),
            MultilinearPolynomial::from(v_io),
        ];

        // eq * io_witness_range * (v_final - v_io)
        let output_check_fn = |vals: &[F]| -> F { vals[0] * vals[1] * (vals[2] - vals[3]) };

        let (sumcheck_proof, r_sumcheck, sumcheck_openings) =
            SumcheckInstanceProof::<F, ProofTranscript>::prove_arbitrary::<_>(
                &F::zero(),
                num_rounds,
                &mut sumcheck_polys,
                output_check_fn,
                3,
                transcript,
            );

        opening_accumulator.append(
            &[&polynomials.v_final],
            DensePolynomial::new(EqPolynomial::evals(&r_sumcheck)),
            r_sumcheck.to_vec(),
            &[sumcheck_openings[2]],
            transcript,
        );

        Self {
            num_rounds,
            sumcheck_proof,
            opening: sumcheck_openings[2], // only need v_final; verifier computes the rest on its own
            _pcs: PhantomData,
        }
    }

    fn verify(
        proof: &Self,
        preprocessing: &ReadWriteMemoryPreprocessing,
        commitment: &ReadWriteMemoryCommitments<PCS, ProofTranscript>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let r_eq = transcript.challenge_vector(proof.num_rounds);

        let (sumcheck_claim, r_sumcheck) =
            proof
                .sumcheck_proof
                .verify(F::zero(), proof.num_rounds, 3, transcript)?;

        let eq_eval = EqPolynomial::new(r_eq.to_vec()).evaluate(&r_sumcheck);

        let program_io = preprocessing.program_io.as_ref().unwrap();
        let memory_layout = &program_io.memory_layout;

        let input_start_index =
            memory_address_to_witness_index(memory_layout.input_start, memory_layout);
        let ram_start_index =
            memory_address_to_witness_index(RAM_START_ADDRESS, memory_layout) as u64;
        assert!(
            ram_start_index.is_power_of_two(),
            "ram_start_index must be a power of two"
        );

        let io_memory_size = ram_start_index as usize;
        let log_io_memory_size = io_memory_size.log_2();

        let io_witness_range: Vec<_> = (0..io_memory_size)
            .map(|i| {
                if i >= input_start_index {
                    F::one()
                } else {
                    F::zero()
                }
            })
            .collect();
        let mut io_witness_range_eval = DensePolynomial::new(io_witness_range)
            .evaluate(&r_sumcheck[(proof.num_rounds - log_io_memory_size)..]);

        let r_prod: F = r_sumcheck[..(proof.num_rounds - log_io_memory_size)]
            .iter()
            .map(|r| F::one() - r)
            .product();
        io_witness_range_eval *= r_prod;

        let mut v_io: Vec<u64> = vec![0; io_memory_size];
        let mut input_index =
            memory_address_to_witness_index(memory_layout.input_start, memory_layout);
        // Convert input bytes into words and populate `v_io`
        for chunk in program_io.inputs.chunks(4) {
            let mut word = [0u8; 4];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u32::from_le_bytes(word);
            v_io[input_index] = word as u64;
            input_index += 1;
        }
        let mut output_index =
            memory_address_to_witness_index(memory_layout.output_start, memory_layout);
        // Convert output bytes into words and populate `v_io`
        for chunk in program_io.outputs.chunks(4) {
            let mut word = [0u8; 4];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u32::from_le_bytes(word);
            v_io[output_index] = word as u64;
            output_index += 1;
        }
        // Copy panic bit
        v_io[memory_address_to_witness_index(memory_layout.panic, memory_layout)] =
            program_io.panic as u64;
        if !program_io.panic {
            // Set termination bit
            v_io[memory_address_to_witness_index(memory_layout.termination, memory_layout)] = 1;
        }

        let mut v_io_eval = DensePolynomial::from_u64(&v_io)
            .evaluate(&r_sumcheck[(proof.num_rounds - log_io_memory_size)..]);
        v_io_eval *= r_prod;

        assert_eq!(
            eq_eval * io_witness_range_eval * (proof.opening - v_io_eval),
            sumcheck_claim,
            "Output sumcheck check failed."
        );

        opening_accumulator.append(
            &[&commitment.v_final],
            r_sumcheck,
            &[&proof.opening],
            transcript,
        );

        Ok(())
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct ReadWriteMemoryProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub memory_checking_proof: MemoryCheckingProof<
        F,
        PCS,
        ReadWriteMemoryOpenings<F>,
        RegisterAddressOpenings<F>,
        ProofTranscript,
    >,
    pub timestamp_validity_proof: TimestampValidityProof<F, PCS, ProofTranscript>,
    pub output_proof: OutputSumcheckProof<F, PCS, ProofTranscript>,
}

impl<F, PCS, ProofTranscript> ReadWriteMemoryProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    #[tracing::instrument(skip_all, name = "ReadWriteMemoryProof::prove")]
    pub fn prove<'a>(
        generators: &PCS::Setup,
        preprocessing: &ReadWriteMemoryPreprocessing,
        polynomials: &'a JoltPolynomials<F>,
        program_io: &JoltDevice,
        opening_accumulator: &mut ProverOpeningAccumulator<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Self {
        let memory_checking_proof = ReadWriteMemoryProof::prove_memory_checking(
            generators,
            preprocessing,
            &polynomials.read_write_memory,
            polynomials,
            opening_accumulator,
            transcript,
        );

        let output_proof = OutputSumcheckProof::prove_outputs(
            &polynomials.read_write_memory,
            program_io,
            opening_accumulator,
            transcript,
        );

        let timestamp_validity_proof = TimestampValidityProof::prove(
            generators,
            &polynomials.timestamp_range_check,
            polynomials,
            opening_accumulator,
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
        generators: &PCS::Setup,
        preprocessing: &ReadWriteMemoryPreprocessing,
        commitments: &JoltCommitments<PCS, ProofTranscript>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        ReadWriteMemoryProof::verify_memory_checking(
            preprocessing,
            generators,
            self.memory_checking_proof,
            &commitments.read_write_memory,
            commitments,
            opening_accumulator,
            transcript,
        )?;
        OutputSumcheckProof::verify(
            &self.output_proof,
            preprocessing,
            &commitments.read_write_memory,
            opening_accumulator,
            transcript,
        )?;
        TimestampValidityProof::verify(
            &mut self.timestamp_validity_proof,
            generators,
            commitments,
            opening_accumulator,
            transcript,
        )
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;

    use super::*;

    #[test]
    fn read_write_memory_stuff_ordering() {
        let preprocessing = ReadWriteMemoryPreprocessing::preprocess(vec![]);
        ReadWriteMemoryOpenings::<Fr>::test_ordering_consistency(&preprocessing);
    }
}
