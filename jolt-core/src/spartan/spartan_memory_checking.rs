use crate::field::{JoltField, OptimizedMul};
use crate::jolt::vm::{JoltCommitments, JoltPolynomials, JoltStuff};
use crate::lasso::memory_checking::{
    ExogenousOpenings, Initializable, StructuredPolynomialData, VerifierComputedOpening,
};
use crate::poly::commitment::commitment_scheme::{BatchType, CommitShape, CommitmentScheme};
use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use crate::subprotocols::grand_product::{
    BatchedDenseGrandProduct, BatchedGrandProduct, BatchedGrandProductLayer,
    BatchedGrandProductProof,
};
use crate::subprotocols::sumcheck::SumcheckInstanceProof;
use crate::utils::math::Math;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::Transcript;
use crate::{
    lasso::memory_checking::{
        MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier, MultisetHashes,
        NoPreprocessing,
    },
    poly::{
        dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial, identity_poly::IdentityPolynomial,
    },
    utils::errors::ProofVerifyError,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::interleave;
use rayon::prelude::*;
#[cfg(test)]
use std::collections::HashSet;

use super::{Assignment, Instance};

#[derive(Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct SpartanStuff<T: CanonicalSerialize + CanonicalDeserialize + Sync> {
    read_cts_rows: Vec<T>,
    read_cts_cols: Vec<T>,
    final_cts_rows: Vec<T>,
    final_cts_cols: Vec<T>,
    // identity: VerifierComputedOpening<T>,
}

impl<T: CanonicalSerialize + CanonicalDeserialize + Sync> StructuredPolynomialData<T>
    for SpartanStuff<T>
{
    fn read_write_values(&self) -> Vec<&T> {
        self.read_cts_rows
            .iter()
            .chain(self.read_cts_cols.iter())
            .collect()
    }

    fn read_write_values_mut(&mut self) -> Vec<&mut T> {
        self.read_cts_rows
            .iter_mut()
            .chain(self.read_cts_cols.iter_mut())
            .collect()
    }
}

pub type SpartanPolynomials<F: JoltField> = SpartanStuff<DensePolynomial<F>>;

pub type SpartanOpenings<F: JoltField> = SpartanStuff<F>;

pub type SpartanCommitments<PCS: CommitmentScheme<ProofTranscript>, ProofTranscript: Transcript> =
    SpartanStuff<PCS::Commitment>;

impl<T: CanonicalSerialize + CanonicalDeserialize + Default> Initializable<T, NoPreprocessing>
    for SpartanStuff<T>
{
}

#[derive(Clone)]
pub struct SpartanPreprocessing<F: JoltField> {
    inst: Instance<F>,
    vars: Assignment<F>,
    inputs: Assignment<F>,
}

impl<F: JoltField> SpartanPreprocessing<F> {
    #[tracing::instrument(skip_all, name = "InstructionLookups::preprocess")]
    pub fn preprocess() -> Self {
        //TODO(Ashish):- Read circom file of json file and convert that file into matrices.
        todo!()
    }
}

impl<F: JoltField, T: CanonicalSerialize + CanonicalDeserialize + Default>
    Initializable<T, SpartanPreprocessing<F>> for SpartanStuff<T>
{
    // fn initialize(_preprocessing: &SpartanPreprocessing<F>) -> Self {
    //     Self {
    //         read_cts_rows: std::iter::repeat_with(|| T::default()).take(3).collect(),
    //         read_cts_cols: std::iter::repeat_with(|| T::default()).take(3).collect(),
    //         final_cts_rows: std::iter::repeat_with(|| T::default()).take(3).collect(),
    //         final_cts_cols: std::iter::repeat_with(|| T::default()).take(3).collect(),
    //     }
    // }
}

impl<F: JoltField> SpartanPolynomials<F> {
    #[tracing::instrument(skip_all, name = "ReadWriteMemory::new")]
    pub fn generate_witness(preprocessing: &SpartanPreprocessing<F>) {
        //     assert!(program_io.inputs.len() <= program_io.memory_layout.max_input_size as usize);
        //     assert!(program_io.outputs.len() <= program_io.memory_layout.max_output_size as usize);

        //     let m = trace.len();
        //     assert!(m.is_power_of_two());

        //     let max_trace_address = trace
        //         .iter()
        //         .map(|step| match step.memory_ops[RAM] {
        //             MemoryOp::Read(a) => remap_address(a, &program_io.memory_layout),
        //             MemoryOp::Write(a, _) => remap_address(a, &program_io.memory_layout),
        //         })
        //         .max()
        //         .unwrap();

        //     let memory_size = max_trace_address.next_power_of_two() as usize;
        //     let mut v_init: Vec<u64> = vec![0; memory_size];
        //     // Copy bytecode
        //     let mut v_init_index = memory_address_to_witness_index(
        //         preprocessing.min_bytecode_address,
        //         &program_io.memory_layout,
        //     );
        //     for word in preprocessing.bytecode_words.iter() {
        //         v_init[v_init_index] = *word as u64;
        //         v_init_index += 1;
        //     }
        //     // Copy input bytes
        //     v_init_index = memory_address_to_witness_index(
        //         program_io.memory_layout.input_start,
        //         &program_io.memory_layout,
        //     );
        //     // Convert input bytes into words and populate `v_init`
        //     for chunk in program_io.inputs.chunks(4) {
        //         let mut word = [0u8; 4];
        //         for (i, byte) in chunk.iter().enumerate() {
        //             word[i] = *byte;
        //         }
        //         let word = u32::from_le_bytes(word);
        //         v_init[v_init_index] = word as u64;
        //         v_init_index += 1;
        //     }

        //     #[cfg(test)]
        //     let mut init_tuples: HashSet<(usize, u64, u64)> = HashSet::new();
        //     #[cfg(test)]
        //     {
        //         for (a, v) in v_init.iter().enumerate() {
        //             init_tuples.insert((a, *v, 0u64));
        //         }
        //     }
        //     #[cfg(test)]
        //     let mut read_tuples: HashSet<(usize, u64, u64)> = HashSet::new();
        //     #[cfg(test)]
        //     let mut write_tuples: HashSet<(usize, u64, u64)> = HashSet::new();

        //     let mut a_ram: Vec<u64> = Vec::with_capacity(m);

        //     let mut v_read_rs1: Vec<u64> = Vec::with_capacity(m);
        //     let mut v_read_rs2: Vec<u64> = Vec::with_capacity(m);
        //     let mut v_read_rd: Vec<u64> = Vec::with_capacity(m);
        //     let mut v_read_ram: Vec<u64> = Vec::with_capacity(m);

        //     let mut t_read_rs1: Vec<u64> = Vec::with_capacity(m);
        //     let mut t_read_rs2: Vec<u64> = Vec::with_capacity(m);
        //     let mut t_read_rd: Vec<u64> = Vec::with_capacity(m);
        //     let mut t_read_ram: Vec<u64> = Vec::with_capacity(m);

        //     let mut v_write_rd: Vec<u64> = Vec::with_capacity(m);
        //     let mut v_write_ram: Vec<u64> = Vec::with_capacity(m);

        //     let mut t_final = vec![0; memory_size];
        //     let mut v_final = v_init.clone();

        //     let span = tracing::span!(tracing::Level::DEBUG, "memory_trace_processing");
        //     let _enter = span.enter();

        //     for (i, step) in trace.iter().enumerate() {
        //         let timestamp = i as u64;

        //         match step.memory_ops[RS1] {
        //             MemoryOp::Read(a) => {
        //                 assert!(a < REGISTER_COUNT);
        //                 let a = a as usize;
        //                 let v = v_final[a];

        //                 #[cfg(test)]
        //                 {
        //                     read_tuples.insert((a, v, t_final[a]));
        //                     write_tuples.insert((a, v, timestamp));
        //                 }

        //                 v_read_rs1.push(v);
        //                 t_read_rs1.push(t_final[a]);
        //                 t_final[a] = timestamp;
        //             }
        //             MemoryOp::Write(a, v) => {
        //                 panic!("Unexpected rs1 MemoryOp::Write({}, {})", a, v);
        //             }
        //         };

        //         match step.memory_ops[RS2] {
        //             MemoryOp::Read(a) => {
        //                 assert!(a < REGISTER_COUNT);
        //                 let a = a as usize;
        //                 let v = v_final[a];

        //                 #[cfg(test)]
        //                 {
        //                     read_tuples.insert((a, v, t_final[a]));
        //                     write_tuples.insert((a, v, timestamp));
        //                 }

        //                 v_read_rs2.push(v);
        //                 t_read_rs2.push(t_final[a]);
        //                 t_final[a] = timestamp;
        //             }
        //             MemoryOp::Write(a, v) => {
        //                 panic!("Unexpected rs2 MemoryOp::Write({}, {})", a, v)
        //             }
        //         };

        //         match step.memory_ops[RD] {
        //             MemoryOp::Read(a) => {
        //                 panic!("Unexpected rd MemoryOp::Read({})", a)
        //             }
        //             MemoryOp::Write(a, v_new) => {
        //                 assert!(a < REGISTER_COUNT);
        //                 let a = a as usize;
        //                 let v_old = v_final[a];

        //                 #[cfg(test)]
        //                 {
        //                     read_tuples.insert((a, v_old, t_final[a]));
        //                     write_tuples.insert((a, v_new, timestamp));
        //                 }

        //                 v_read_rd.push(v_old);
        //                 t_read_rd.push(t_final[a]);
        //                 v_write_rd.push(v_new);
        //                 v_final[a] = v_new;
        //                 t_final[a] = timestamp;
        //             }
        //         };

        //         match step.memory_ops[RAM] {
        //             MemoryOp::Read(a) => {
        //                 debug_assert!(a % 4 == 0);
        //                 let remapped_a = remap_address(a, &program_io.memory_layout) as usize;
        //                 let v = v_final[remapped_a];

        //                 #[cfg(test)]
        //                 {
        //                     read_tuples.insert((remapped_a, v, t_final[remapped_a]));
        //                     write_tuples.insert((remapped_a, v, timestamp));
        //                 }

        //                 a_ram.push(remapped_a as u64);
        //                 v_read_ram.push(v);
        //                 t_read_ram.push(t_final[remapped_a]);
        //                 v_write_ram.push(v);
        //                 t_final[remapped_a] = timestamp;
        //             }
        //             MemoryOp::Write(a, v_new) => {
        //                 debug_assert!(a % 4 == 0);
        //                 let remapped_a = remap_address(a, &program_io.memory_layout) as usize;
        //                 let v_old = v_final[remapped_a];

        //                 #[cfg(test)]
        //                 {
        //                     read_tuples.insert((remapped_a, v_old, t_final[remapped_a]));
        //                     write_tuples.insert((remapped_a, v_new, timestamp));
        //                 }

        //                 a_ram.push(remapped_a as u64);
        //                 v_read_ram.push(v_old);
        //                 t_read_ram.push(t_final[remapped_a]);
        //                 v_write_ram.push(v_new);
        //                 v_final[remapped_a] = v_new;
        //                 t_final[remapped_a] = timestamp;
        //             }
        //         }
        //     }

        //     drop(_enter);
        //     drop(span);

        //     #[cfg(test)]
        //     {
        //         let mut final_tuples: HashSet<(usize, u64, u64)> = HashSet::new();
        //         for (a, (v, t)) in v_final.iter().zip(t_final.iter()).enumerate() {
        //             final_tuples.insert((a, *v, *t));
        //         }

        //         let init_write: HashSet<_> = init_tuples.union(&write_tuples).collect();
        //         let read_final: HashSet<_> = read_tuples.union(&final_tuples).collect();
        //         let set_difference: Vec<_> = init_write.symmetric_difference(&read_final).collect();
        //         assert_eq!(set_difference.len(), 0);
        //     }

        //     let [a_ram, v_read_rd, v_read_rs1, v_read_rs2, v_read_ram, v_write_rd, v_write_ram, v_final, t_read_rd_poly, t_read_rs1_poly, t_read_rs2_poly, t_read_ram_poly, t_final, v_init] =
        //         map_to_polys([
        //             &a_ram,
        //             &v_read_rd,
        //             &v_read_rs1,
        //             &v_read_rs2,
        //             &v_read_ram,
        //             &v_write_rd,
        //             &v_write_ram,
        //             &v_final,
        //             &t_read_rd,
        //             &t_read_rs1,
        //             &t_read_rs2,
        //             &t_read_ram,
        //             &t_final,
        //             &v_init,
        //         ]);

        //     let polynomials = ReadWriteMemoryPolynomials {
        //         a_ram,
        //         v_read_rd,
        //         v_read_rs1,
        //         v_read_rs2,
        //         v_read_ram,
        //         v_write_rd,
        //         v_write_ram,
        //         v_final,
        //         t_read_rd: t_read_rd_poly,
        //         t_read_rs1: t_read_rs1_poly,
        //         t_read_rs2: t_read_rs2_poly,
        //         t_read_ram: t_read_ram_poly,
        //         t_final,
        //         v_init: Some(v_init),
        //         a_init_final: None,
        //         identity: None,
        //     };

        //     (polynomials, [t_read_rd, t_read_rs1, t_read_rs2, t_read_ram])
    }

    // /// Computes the shape of all commitments.
    // pub fn commitment_shapes(
    //     max_memory_address: usize,
    //     max_trace_length: usize,
    // ) -> Vec<CommitShape> {
    //     let max_memory_address = max_memory_address.next_power_of_two();
    //     let max_trace_length = max_trace_length.next_power_of_two();

    //     let read_write_shape = CommitShape::new(max_trace_length, BatchType::Big);
    //     let init_final_shape = CommitShape::new(max_memory_address, BatchType::Small);

    //     vec![read_write_shape, init_final_shape]
    // }
}

pub type ReadTimestampOpenings<F> = [F; 3];
impl<F: JoltField> ExogenousOpenings<F> for ReadTimestampOpenings<F> {
    fn openings(&self) -> Vec<&F> {
        self.iter().collect()
    }

    fn openings_mut(&mut self) -> Vec<&mut F> {
        self.iter_mut().collect()
    }

    fn exogenous_data<T: CanonicalSerialize + CanonicalDeserialize + Sync>(
        polys_or_commitments: &JoltStuff<T>,
    ) -> Vec<&T> {
        vec![
            &polys_or_commitments.read_write_memory.t_read_rd,
            &polys_or_commitments.read_write_memory.t_read_rs1,
            &polys_or_commitments.read_write_memory.t_read_rs2,
            &polys_or_commitments.read_write_memory.t_read_ram,
        ]
    }
}

impl<F, PCS, ProofTranscript> SpartanProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    #[tracing::instrument(skip_all, name = "TimestampRangeCheckWitness::new")]
    pub fn generate_witness(
        read_timestamps: &[Vec<u64>; 3],
    ) -> (
        SpartanPolynomials<F>,
        SpartanCommitments<PCS, ProofTranscript>,
    ) {
        let M = read_timestamps[0].len();

        #[cfg(test)]
        let mut init_tuples: HashSet<(u64, u64)> = HashSet::new();
        #[cfg(test)]
        {
            for i in 0..M {
                init_tuples.insert((i as u64, 0u64));
            }
        }

        let read_and_final_cts: Vec<[Vec<u64>; 4]> = (0..3)
            .into_par_iter()
            .map(|i| {
                let mut read_cts_read_timestamp: Vec<u64> = vec![0; M];
                let mut read_cts_global_minus_read: Vec<u64> = vec![0; M];
                let mut final_cts_read_timestamp: Vec<u64> = vec![0; M];
                let mut final_cts_global_minus_read: Vec<u64> = vec![0; M];

                for (j, read_timestamp) in read_timestamps[i].iter().enumerate() {
                    read_cts_read_timestamp[j] = final_cts_read_timestamp[*read_timestamp as usize];
                    final_cts_read_timestamp[*read_timestamp as usize] += 1;
                    let lookup_index = j - *read_timestamp as usize;
                    read_cts_global_minus_read[j] = final_cts_global_minus_read[lookup_index];
                    final_cts_global_minus_read[lookup_index] += 1;
                }

                #[cfg(test)]
                {
                    let global_minus_read_timestamps = &read_timestamps[i]
                        .iter()
                        .enumerate()
                        .map(|(j, timestamp)| j as u64 - *timestamp)
                        .collect();

                    for (lookup_indices, read_cts, final_cts) in [
                        (
                            &read_timestamps[i],
                            &read_cts_read_timestamp,
                            &final_cts_read_timestamp,
                        ),
                        (
                            global_minus_read_timestamps,
                            &read_cts_global_minus_read,
                            &final_cts_global_minus_read,
                        ),
                    ]
                    .iter()
                    {
                        let mut read_tuples: HashSet<(u64, u64)> = HashSet::new();
                        let mut write_tuples: HashSet<(u64, u64)> = HashSet::new();
                        for (v, t) in lookup_indices.iter().zip(read_cts.iter()) {
                            read_tuples.insert((*v, *t));
                            write_tuples.insert((*v, *t + 1));
                        }

                        let mut final_tuples: HashSet<(u64, u64)> = HashSet::new();
                        for (i, t) in final_cts.iter().enumerate() {
                            final_tuples.insert((i as u64, *t));
                        }

                        let init_write: HashSet<_> = init_tuples.union(&write_tuples).collect();
                        let read_final: HashSet<_> = read_tuples.union(&final_tuples).collect();
                        let set_difference: Vec<_> =
                            init_write.symmetric_difference(&read_final).collect();
                        assert_eq!(set_difference.len(), 0);
                    }
                }

                [
                    read_cts_read_timestamp,
                    read_cts_global_minus_read,
                    final_cts_read_timestamp,
                    final_cts_global_minus_read,
                ]
            })
            .collect();

        let read_cts_read_timestamp = read_and_final_cts
            .par_iter()
            .map(|cts| DensePolynomial::from_u64(&cts[0]))
            .collect::<Vec<DensePolynomial<F>>>()
            .try_into()
            .unwrap();
        let read_cts_global_minus_read = read_and_final_cts
            .par_iter()
            .map(|cts| DensePolynomial::from_u64(&cts[1]))
            .collect::<Vec<DensePolynomial<F>>>()
            .try_into()
            .unwrap();
        let final_cts_read_timestamp = read_and_final_cts
            .par_iter()
            .map(|cts| DensePolynomial::from_u64(&cts[2]))
            .collect::<Vec<DensePolynomial<F>>>()
            .try_into()
            .unwrap();
        let final_cts_global_minus_read = read_and_final_cts
            .par_iter()
            .map(|cts| DensePolynomial::from_u64(&cts[3]))
            .collect::<Vec<DensePolynomial<F>>>()
            .try_into()
            .unwrap();

        // SpartanPolynomials {
        //     read_cts_read_timestamp,
        //     read_cts_global_minus_read,
        //     final_cts_read_timestamp,
        //     final_cts_global_minus_read,
        //     identity: None,
        // };
        todo!()
    }
}

impl<F, PCS, ProofTranscript> MemoryCheckingProver<F, PCS, ProofTranscript>
    for SpartanProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    type Polynomials = SpartanPolynomials<F>;
    type Openings = SpartanOpenings<F>;
    type Commitments = SpartanCommitments<PCS, ProofTranscript>;
    type ExogenousOpenings = ReadTimestampOpenings<F>;
    type MemoryTuple = (F, F); // a = v for all range check tuples

    // Init/final grand products are batched together with read/write grand products
    type InitFinalGrandProduct = NoopGrandProduct;

    fn prove_memory_checking(
        _: &PCS::Setup,
        _: &NoPreprocessing,
        _: &Self::Polynomials,
        _: &JoltPolynomials<F>,
        _: &mut ProverOpeningAccumulator<F, ProofTranscript>,
        _: &mut ProofTranscript,
    ) -> MemoryCheckingProof<F, PCS, Self::Openings, Self::ExogenousOpenings, ProofTranscript> {
        unimplemented!("Use SpartanProof::prove instead");
    }

    fn fingerprint(inputs: &(F, F), gamma: &F, tau: &F) -> F {
        let (a, t) = *inputs;
        a * gamma + t - *tau
    }

    #[tracing::instrument(skip_all, name = "RangeCheckPolynomials::compute_leaves")]
    /// For these timestamp range check polynomials, the init/final polynomials are the
    /// the same length as the read/write polynomials. This is because the init/final polynomials
    /// are determined by the range (0..N) that we are checking for, which in this case is
    /// determined by the length of the execution trace.
    /// Because all the polynomials are of the same length, the init/final grand products can be
    /// batched together with the read/write grand products. So, we only return one `Vec<Vec<F>>`
    /// from this `compute_leaves` function.
    fn compute_leaves(
        _: &NoPreprocessing,
        polynomials: &Self::Polynomials,
        jolt_polynomials: &JoltPolynomials<F>,
        gamma: &F,
        tau: &F,
    ) -> ((Vec<F>, usize), ()) {
        let read_timestamps = [
            &jolt_polynomials.read_write_memory.t_read_rd,
            &jolt_polynomials.read_write_memory.t_read_rs1,
            &jolt_polynomials.read_write_memory.t_read_rs2,
            &jolt_polynomials.read_write_memory.t_read_ram,
        ];

        let M = read_timestamps[0].len();

        let read_write_leaves: Vec<Vec<F>> = (0..3)
            .into_par_iter()
            .flat_map(|i| {
                let read_fingerprints_0: Vec<F> = (0..M)
                    .into_par_iter()
                    .map(|j| {
                        let read_timestamp = read_timestamps[i][j];
                        gamma.mul_01_optimized(read_timestamp)
                            + polynomials.read_cts_read_timestamp[i][j]
                            - *tau
                    })
                    .collect();
                let write_fingeprints_0 = read_fingerprints_0
                    .par_iter()
                    .map(|read_fingerprint| *read_fingerprint + F::one())
                    .collect();

                let read_fingerprints_1: Vec<F> = (0..M)
                    .into_par_iter()
                    .map(|j| {
                        let global_minus_read =
                            F::from_u64(j as u64).unwrap() - read_timestamps[i][j];
                        global_minus_read * gamma + polynomials.read_cts_global_minus_read[i][j]
                            - *tau
                    })
                    .collect();
                let write_fingeprints_1 = read_fingerprints_1
                    .par_iter()
                    .map(|read_fingerprint| *read_fingerprint + F::one())
                    .collect();

                [
                    read_fingerprints_0,
                    write_fingeprints_0,
                    read_fingerprints_1,
                    write_fingeprints_1,
                ]
            })
            .collect();

        let mut leaves = read_write_leaves;

        let init_leaves: Vec<F> = (0..M)
            .into_par_iter()
            .map(|i| {
                let index = F::from_u64(i as u64).unwrap();
                // t = 0
                index * gamma - *tau
            })
            .collect();

        leaves.par_extend((0..3).into_par_iter().flat_map(|i| {
            let final_fingerprints_0 = (0..M)
                .into_par_iter()
                .map(|j| polynomials.final_cts_read_timestamp[i][j] + init_leaves[j])
                .collect();

            let final_fingerprints_1 = (0..M)
                .into_par_iter()
                .map(|j| polynomials.final_cts_global_minus_read[i][j] + init_leaves[j])
                .collect();

            [final_fingerprints_0, final_fingerprints_1]
        }));
        leaves.push(init_leaves);

        let batch_size = leaves.len();

        // TODO(moodlezoup): Avoid concat
        ((leaves.concat(), batch_size), ())
    }

    fn interleave<T: Copy + Clone>(
        _: &NoPreprocessing,
        read_values: &Vec<T>,
        write_values: &Vec<T>,
        init_values: &Vec<T>,
        final_values: &Vec<T>,
    ) -> (Vec<T>, Vec<T>) {
        let read_write_values = interleave(read_values.clone(), write_values.clone()).collect();
        let mut init_final_values = final_values.clone();
        init_final_values.extend(init_values.clone());

        (read_write_values, init_final_values)
    }

    fn uninterleave_hashes(
        _: &NoPreprocessing,
        read_write_hashes: Vec<F>,
        init_final_hashes: Vec<F>,
    ) -> MultisetHashes<F> {
        let num_memories = 2 * 3;

        assert_eq!(read_write_hashes.len(), 2 * num_memories);
        let mut read_hashes = Vec::with_capacity(num_memories);
        let mut write_hashes = Vec::with_capacity(num_memories);
        for i in 0..num_memories {
            read_hashes.push(read_write_hashes[2 * i]);
            write_hashes.push(read_write_hashes[2 * i + 1]);
        }

        assert_eq!(init_final_hashes.len(), num_memories + 1);
        let mut final_hashes = init_final_hashes;
        let init_hash = final_hashes.pop().unwrap();

        MultisetHashes {
            read_hashes,
            write_hashes,
            init_hashes: vec![init_hash],
            final_hashes,
        }
    }

    fn check_multiset_equality(_: &NoPreprocessing, multiset_hashes: &MultisetHashes<F>) {
        let num_memories = 2 * 3;
        assert_eq!(multiset_hashes.read_hashes.len(), num_memories);
        assert_eq!(multiset_hashes.write_hashes.len(), num_memories);
        assert_eq!(multiset_hashes.final_hashes.len(), num_memories);
        assert_eq!(multiset_hashes.init_hashes.len(), 1);

        (0..num_memories).into_par_iter().for_each(|i| {
            let read_hash = multiset_hashes.read_hashes[i];
            let write_hash = multiset_hashes.write_hashes[i];
            let init_hash = multiset_hashes.init_hashes[0];
            let final_hash = multiset_hashes.final_hashes[i];
            assert_eq!(
                init_hash * write_hash,
                final_hash * read_hash,
                "Multiset hashes don't match"
            );
        });
    }

    fn protocol_name() -> &'static [u8] {
        b"Timestamp Validity Proof"
    }
}

impl<F, PCS, ProofTranscript> MemoryCheckingVerifier<F, PCS, ProofTranscript>
    for SpartanProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    fn compute_verifier_openings(_: &mut Self::Openings, _: &NoPreprocessing, _: &[F], _: &[F]) {
        unimplemented!("")
    }

    fn verify_memory_checking(
        _: &NoPreprocessing,
        _: &PCS::Setup,
        mut _proof: MemoryCheckingProof<
            F,
            PCS,
            Self::Openings,
            Self::ExogenousOpenings,
            ProofTranscript,
        >,
        _commitments: &Self::Commitments,
        _: &JoltCommitments<PCS, ProofTranscript>,
        _opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
        _transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        unimplemented!("Use SpartanProof::verify instead");
    }

    fn read_tuples(
        _: &NoPreprocessing,
        openings: &Self::Openings,
        read_timestamp_openings: &[F; 3],
    ) -> Vec<Self::MemoryTuple> {
        (0..3)
            .flat_map(|i| {
                [
                    (
                        read_timestamp_openings[i],
                        openings.read_cts_read_timestamp[i],
                    ),
                    (
                        openings.identity.unwrap() - read_timestamp_openings[i],
                        openings.read_cts_global_minus_read[i],
                    ),
                ]
            })
            .collect()
    }

    fn write_tuples(
        _: &NoPreprocessing,
        openings: &Self::Openings,
        read_timestamp_openings: &[F; 3],
    ) -> Vec<Self::MemoryTuple> {
        (0..3)
            .flat_map(|i| {
                [
                    (
                        read_timestamp_openings[i],
                        openings.read_cts_read_timestamp[i] + F::one(),
                    ),
                    (
                        openings.identity.unwrap() - read_timestamp_openings[i],
                        openings.read_cts_global_minus_read[i] + F::one(),
                    ),
                ]
            })
            .collect()
    }

    fn init_tuples(
        _: &NoPreprocessing,
        openings: &Self::Openings,
        _: &[F; 3],
    ) -> Vec<Self::MemoryTuple> {
        vec![(openings.identity.unwrap(), F::zero())]
    }

    fn final_tuples(
        _: &NoPreprocessing,
        openings: &Self::Openings,
        _: &[F; 3],
    ) -> Vec<Self::MemoryTuple> {
        (0..3)
            .flat_map(|i| {
                [
                    (
                        openings.identity.unwrap(),
                        openings.final_cts_read_timestamp[i],
                    ),
                    (
                        openings.identity.unwrap(),
                        openings.final_cts_global_minus_read[i],
                    ),
                ]
            })
            .collect()
    }
}

pub struct NoopGrandProduct;
impl<F, PCS, ProofTranscript> BatchedGrandProduct<F, PCS, ProofTranscript> for NoopGrandProduct
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    type Leaves = ();
    type Config = ();

    fn construct(_leaves: Self::Leaves) -> Self {
        unimplemented!("init/final grand products are batched with read/write grand products");
    }
    fn construct_with_config(_leaves: Self::Leaves, _config: Self::Config) -> Self {
        unimplemented!("init/final grand products are batched with read/write grand products");
    }
    fn num_layers(&self) -> usize {
        unimplemented!("init/final grand products are batched with read/write grand products");
    }
    fn claimed_outputs(&self) -> Vec<F> {
        unimplemented!("init/final grand products are batched with read/write grand products");
    }

    fn layers(
        &'_ mut self,
    ) -> impl Iterator<Item = &'_ mut dyn BatchedGrandProductLayer<F, ProofTranscript>> {
        std::iter::empty() // Needed to compile
    }

    fn prove_grand_product(
        &mut self,
        _opening_accumulator: Option<&mut ProverOpeningAccumulator<F, ProofTranscript>>,
        _transcript: &mut ProofTranscript,
        _setup: Option<&PCS::Setup>,
    ) -> (BatchedGrandProductProof<PCS, ProofTranscript>, Vec<F>) {
        unimplemented!("init/final grand products are batched with read/write grand products")
    }
    fn verify_grand_product(
        _proof: &BatchedGrandProductProof<PCS, ProofTranscript>,
        _claims: &[F],
        _opening_accumulator: Option<&mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>>,
        _transcript: &mut ProofTranscript,
        _setup: Option<&PCS::Setup>,
    ) -> (F, Vec<F>) {
        unimplemented!("init/final grand products are batched with read/write grand products")
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct SpartanProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    outer_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    inner_sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,

    multiset_hashes: MultisetHashes<F>,
    openings: SpartanOpenings<F>,
    exogenous_openings: ReadTimestampOpenings<F>,
    batched_grand_product: BatchedGrandProductProof<PCS, ProofTranscript>,
}

impl<F, PCS, ProofTranscript> SpartanProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    #[tracing::instrument(skip_all, name = "SpartanProof::prove")]
    pub fn prove<'a>(
        generators: &PCS::Setup,
        polynomials: &'a SpartanPolynomials<F>,
        jolt_polynomials: &'a JoltPolynomials<F>,
        preprocessing: &SpartanPreprocessing<F>,
        opening_accumulator: &mut ProverOpeningAccumulator<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Self {
        let num_inputs = preprocessing.inputs.assignment.len();
        let num_vars = preprocessing.vars.assignment.len();

        // we currently require the number of |inputs| + 1 to be at most number of vars
        assert!(num_inputs < num_vars);

        // let (poly_vars, comm_vars, blinds_vars) = {
        //     // create a multilinear polynomial using the supplied assignment for variables
        //     let poly_vars = DensePolynomial::new(vars.clone());

        //     // produce a commitment to the satisfying assignment
        //     // let (comm_vars, blinds_vars) = poly_vars.commit(&gens.gens_pc, Some(random_tape));

        //     // add the commitment to the prover's transcript
        //     comm_vars.append_to_transcript(b"poly_commitment", transcript);
        //     (poly_vars, comm_vars, blinds_vars)
        // };

        // append input to variables to create a single vector z
        let z = {
            let mut z = preprocessing.vars.assignment.clone();
            z.extend(&vec![F::one()]); // add constant term in z
            z.extend(preprocessing.inputs.assignment.clone());
            z.extend(&vec![F::zero(); num_vars - num_inputs - 1]); // we will pad with zeros
            DensePolynomial::new(z)
        };

        // derive the verifier's challenge tau
        let (num_rounds_x, num_rounds_y) = (
            preprocessing.inst.inst.get_num_cons().log_2(),
            z.len().log_2(),
        );
        let tau = transcript.challenge_vector(num_rounds_x);

        // compute the initial evaluation table for R(\tau, x)

        let eq_tau = DensePolynomial::new(EqPolynomial::evals(&tau));

        let (az, bz, cz) = preprocessing.inst.inst.multiply_vec(
            preprocessing.inst.inst.get_num_cons(),
            z.len(),
            &z.Z,
        );
        let comb_func = |polys: &[F]| -> F { polys[0] * (polys[1] * polys[2] - polys[3]) };

        let (outer_sumcheck_proof, outer_sumcheck_r, outer_sumcheck_claims) =
            SumcheckInstanceProof::prove_arbitrary(
                &F::zero(), // claim is zero
                num_rounds_x,
                &mut [eq_tau.clone(), az, bz, cz].to_vec(),
                comb_func,
                3,
                transcript,
            );

        //TODO:- Do we need to reverse?
        // let outer_sumcheck_r: Vec<F> = outer_sumcheck_r.into_iter().rev().collect();

        ProofTranscript::append_scalars(transcript, &outer_sumcheck_claims);

        // claims from the end of sum-check
        // claim_Az is the (scalar) value v_A = \sum_y A(r_x, y) * z(r_x) where r_x is the sumcheck randomness
        let (claim_Az, claim_Bz, claim_Cz): (F, F, F) = (
            outer_sumcheck_claims[0],
            outer_sumcheck_claims[1],
            outer_sumcheck_claims[2],
        );

        let r_inner_sumcheck_RLC: F = transcript.challenge_scalar();
        let r_inner_sumcheck_RLC_square = r_inner_sumcheck_RLC * r_inner_sumcheck_RLC;
        let claim_inner_joint =
            claim_Az + r_inner_sumcheck_RLC * claim_Bz + r_inner_sumcheck_RLC_square * claim_Cz;

        let poly_ABC = {
            // compute the initial evaluation table for R(\tau, x)
            let (evals_A, evals_B, evals_C) = preprocessing.inst.inst.compute_eval_table_sparse(
                preprocessing.inst.inst.get_num_cons(),
                z.len(),
                eq_tau.evals_ref(),
            );

            assert_eq!(evals_A.len(), evals_B.len());
            assert_eq!(evals_A.len(), evals_C.len());
            DensePolynomial::new(
                (0..evals_A.len())
                    .into_par_iter()
                    .map(|i| {
                        evals_A[i]
                            + r_inner_sumcheck_RLC * evals_B[i]
                            + r_inner_sumcheck_RLC_square * evals_C[i]
                    })
                    .collect::<Vec<F>>(),
            )
        };

        let comb_func = |polys: &[F]| -> F { polys[0] * polys[1] };
        let (inner_sumcheck_proof, inner_sumcheck_r, _claims_inner) =
            SumcheckInstanceProof::prove_arbitrary(
                &claim_inner_joint,
                num_rounds_y,
                &mut [poly_ABC, z].to_vec(),
                comb_func,
                2,
                transcript,
            );

        //     //TODO:- Change.
        //     let (batched_grand_product, multiset_hashes, r_grand_product) =
        //         SpartanProof::prove_grand_products(
        //             polynomials,
        //             jolt_polynomials,
        //             opening_accumulator,
        //             transcript,
        //             generators,
        //         );

        //     let mut openings = SpartanOpenings::default();
        //     let mut timestamp_openings = ReadTimestampOpenings::<F>::default();

        //     let batch_size = multiset_hashes.read_hashes.len()
        //         + multiset_hashes.write_hashes.len()
        //         + multiset_hashes.init_hashes.len()
        //         + multiset_hashes.final_hashes.len();
        //     let (_, r_opening) = r_grand_product.split_at(batch_size.next_power_of_two().log_2());
        //     let chis = EqPolynomial::evals(r_opening);

        //     polynomials
        //         .read_write_values()
        //         .into_par_iter()
        //         .zip(openings.read_write_values_mut().into_par_iter())
        //         .chain(
        //             ReadTimestampOpenings::<F>::exogenous_data(jolt_polynomials)
        //                 .into_par_iter()
        //                 .zip(timestamp_openings.openings_mut().into_par_iter()),
        //         )
        //         .for_each(|(poly, opening)| {
        //             let claim = poly.evaluate_at_chi_low_optimized(&chis);
        //             *opening = claim;
        //         });

        //     opening_accumulator.append(
        //         &polynomials
        //             .read_write_values()
        //             .into_iter()
        //             .chain(ReadTimestampOpenings::<F>::exogenous_data(jolt_polynomials).into_iter())
        //             .collect::<Vec<_>>(),
        //         DensePolynomial::new(chis),
        //         r_opening.to_vec(),
        //         &openings
        //             .read_write_values()
        //             .into_iter()
        //             .chain(timestamp_openings.openings())
        //             .collect::<Vec<_>>(),
        //         transcript,
        //     );

        //     Self {
        //         multiset_hashes,
        //         openings,
        //         exogenous_openings: timestamp_openings,
        //         batched_grand_product,
        //     }
        // }

        // #[tracing::instrument(skip_all, name = "SpartanProof::prove_grand_products")]
        // fn prove_grand_products(
        //     polynomials: &SpartanPolynomials<F>,
        //     jolt_polynomials: &JoltPolynomials<F>,
        //     opening_accumulator: &mut ProverOpeningAccumulator<F, ProofTranscript>,
        //     transcript: &mut ProofTranscript,
        //     setup: &PCS::Setup,
        // ) -> (
        //     BatchedGrandProductProof<PCS, ProofTranscript>,
        //     MultisetHashes<F>,
        //     Vec<F>,
        // ) {
        //     // Fiat-Shamir randomness for multiset hashes
        //     let gamma: F = transcript.challenge_scalar();
        //     let tau: F = transcript.challenge_scalar();

        //     let protocol_name = Self::protocol_name();
        //     transcript.append_message(protocol_name);

        //     let (leaves, _) = SpartanProof::<F, PCS, ProofTranscript>::compute_leaves(
        //         &NoPreprocessing,
        //         polynomials,
        //         jolt_polynomials,
        //         &gamma,
        //         &tau,
        //     );

        //     let mut batched_circuit = <BatchedDenseGrandProduct<F> as BatchedGrandProduct<
        //         F,
        //         PCS,
        //         ProofTranscript,
        //     >>::construct(leaves);

        //     let hashes: Vec<F> = <BatchedDenseGrandProduct<F> as BatchedGrandProduct<
        //         F,
        //         PCS,
        //         ProofTranscript,
        //     >>::claimed_outputs(&batched_circuit);
        //     let (read_write_hashes, init_final_hashes) = hashes.split_at(4 * 3);
        //     let multiset_hashes =
        //         SpartanProof::<F, PCS, ProofTranscript>::uninterleave_hashes(
        //             &NoPreprocessing,
        //             read_write_hashes.to_vec(),
        //             init_final_hashes.to_vec(),
        //         );
        //     SpartanProof::<F, PCS, ProofTranscript>::check_multiset_equality(
        //         &NoPreprocessing,
        //         &multiset_hashes,
        //     );
        //     multiset_hashes.append_to_transcript(transcript);

        //     let (batched_grand_product, r_grand_product) =
        //         batched_circuit.prove_grand_product(Some(opening_accumulator), transcript, Some(setup));

        //     drop_in_background_thread(batched_circuit);

        //     (batched_grand_product, multiset_hashes, r_grand_product)
        todo!()
    }

    pub fn verify(
        &mut self,
        generators: &PCS::Setup,
        commitments: &JoltCommitments<PCS, ProofTranscript>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        // Fiat-Shamir randomness for multiset hashes
        let gamma: F = transcript.challenge_scalar();
        let tau: F = transcript.challenge_scalar();

        let protocol_name = Self::protocol_name();
        transcript.append_message(protocol_name);

        // Multiset equality checks
        SpartanProof::<F, PCS, ProofTranscript>::check_multiset_equality(
            &NoPreprocessing,
            &self.multiset_hashes,
        );
        self.multiset_hashes.append_to_transcript(transcript);

        let (read_write_hashes, init_final_hashes) =
            SpartanProof::<F, PCS, ProofTranscript>::interleave(
                &NoPreprocessing,
                &self.multiset_hashes.read_hashes,
                &self.multiset_hashes.write_hashes,
                &self.multiset_hashes.init_hashes,
                &self.multiset_hashes.final_hashes,
            );
        let concatenated_hashes = [read_write_hashes, init_final_hashes].concat();
        let batch_size = concatenated_hashes.len();
        let (grand_product_claim, r_grand_product) = BatchedDenseGrandProduct::verify_grand_product(
            &self.batched_grand_product,
            &concatenated_hashes,
            Some(opening_accumulator),
            transcript,
            Some(generators),
        );
        let (r_batch_index, r_opening) =
            r_grand_product.split_at(batch_size.next_power_of_two().log_2());

        opening_accumulator.append(
            &commitments
                .timestamp_range_check
                .read_write_values()
                .into_iter()
                .chain(ReadTimestampOpenings::<F>::exogenous_data(commitments))
                .collect::<Vec<_>>(),
            r_opening.to_vec(),
            &self
                .openings
                .read_write_values()
                .into_iter()
                .chain(self.exogenous_openings.iter())
                .collect::<Vec<_>>(),
            transcript,
        );

        self.openings.identity = Some(IdentityPolynomial::new(r_opening.len()).evaluate(r_opening));

        let read_hashes: Vec<_> = SpartanProof::<F, PCS, ProofTranscript>::read_tuples(
            &NoPreprocessing,
            &self.openings,
            &self.exogenous_openings,
        )
        .iter()
        .map(|tuple| SpartanProof::<F, PCS, ProofTranscript>::fingerprint(tuple, &gamma, &tau))
        .collect();
        let write_hashes: Vec<_> = SpartanProof::<F, PCS, ProofTranscript>::write_tuples(
            &NoPreprocessing,
            &self.openings,
            &self.exogenous_openings,
        )
        .iter()
        .map(|tuple| SpartanProof::<F, PCS, ProofTranscript>::fingerprint(tuple, &gamma, &tau))
        .collect();
        let init_hashes: Vec<_> = SpartanProof::<F, PCS, ProofTranscript>::init_tuples(
            &NoPreprocessing,
            &self.openings,
            &self.exogenous_openings,
        )
        .iter()
        .map(|tuple| SpartanProof::<F, PCS, ProofTranscript>::fingerprint(tuple, &gamma, &tau))
        .collect();
        let final_hashes: Vec<_> = SpartanProof::<F, PCS, ProofTranscript>::final_tuples(
            &NoPreprocessing,
            &self.openings,
            &self.exogenous_openings,
        )
        .iter()
        .map(|tuple| SpartanProof::<F, PCS, ProofTranscript>::fingerprint(tuple, &gamma, &tau))
        .collect();

        let (read_write_hashes, init_final_hashes) =
            SpartanProof::<F, PCS, ProofTranscript>::interleave(
                &NoPreprocessing,
                &read_hashes,
                &write_hashes,
                &init_hashes,
                &final_hashes,
            );

        let combined_hash: F = read_write_hashes
            .iter()
            .chain(init_final_hashes.iter())
            .zip(EqPolynomial::evals(r_batch_index).iter())
            .map(|(hash, eq_eval)| *hash * eq_eval)
            .sum();
        assert_eq!(combined_hash, grand_product_claim);

        Ok(())
    }

    /// Computes the shape of all commitments.
    pub fn commitment_shapes(max_trace_length: usize) -> Vec<CommitShape> {
        let max_trace_length = max_trace_length.next_power_of_two();

        vec![CommitShape::new(max_trace_length, BatchType::Big)]
    }

    fn protocol_name() -> &'static [u8] {
        b"Spartan Proof"
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;

    use super::*;

    #[test]
    fn timestamp_range_check_stuff_ordering() {
        SpartanOpenings::<Fr>::test_ordering_consistency(&NoPreprocessing);
    }
}
