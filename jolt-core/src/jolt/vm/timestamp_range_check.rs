use super::{JoltCommitments, JoltPolynomials, JoltStuff};
use crate::field::JoltField;
use crate::lasso::memory_checking::{
    ExogenousOpenings, Initializable, StructuredPolynomialData, VerifierComputedOpening,
};
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::compact_polynomial::{CompactPolynomial, SmallScalar};
use crate::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation};
use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use crate::subprotocols::grand_product::{
    BatchedDenseGrandProduct, BatchedGrandProduct, BatchedGrandProductLayer,
    BatchedGrandProductProof,
};
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

use super::read_write_memory::ReadWriteMemoryPolynomials;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::MEMORY_OPS_PER_INSTRUCTION;
use itertools::interleave;
use rayon::prelude::*;
#[cfg(test)]
use std::collections::HashSet;

#[derive(Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct TimestampRangeCheckStuff<T: CanonicalSerialize + CanonicalDeserialize + Sync> {
    read_cts_read_timestamp: [T; MEMORY_OPS_PER_INSTRUCTION],
    read_cts_global_minus_read: [T; MEMORY_OPS_PER_INSTRUCTION],
    final_cts_read_timestamp: [T; MEMORY_OPS_PER_INSTRUCTION],
    final_cts_global_minus_read: [T; MEMORY_OPS_PER_INSTRUCTION],

    identity: VerifierComputedOpening<T>,
}

impl<T: CanonicalSerialize + CanonicalDeserialize + Sync> StructuredPolynomialData<T>
    for TimestampRangeCheckStuff<T>
{
    fn read_write_values(&self) -> Vec<&T> {
        self.read_cts_read_timestamp
            .iter()
            .chain(self.read_cts_global_minus_read.iter())
            // These are technically init/final values, but all
            // the polynomials are the same size so they can all
            // be batched together
            .chain(self.final_cts_read_timestamp.iter())
            .chain(self.final_cts_global_minus_read.iter())
            .collect()
    }

    fn read_write_values_mut(&mut self) -> Vec<&mut T> {
        self.read_cts_read_timestamp
            .iter_mut()
            .chain(self.read_cts_global_minus_read.iter_mut())
            // These are technically init/final values, but all
            // the polynomials are the same size so they can all
            // be batched together
            .chain(self.final_cts_read_timestamp.iter_mut())
            .chain(self.final_cts_global_minus_read.iter_mut())
            .collect()
    }
}

/// Note –– F: JoltField bound is not enforced.
///
/// See issue #112792 <https://github.com/rust-lang/rust/issues/112792>.
/// Adding #![feature(lazy_type_alias)] to the crate attributes seem to break
/// `alloy_sol_types`.
pub type TimestampRangeCheckPolynomials<F: JoltField> =
    TimestampRangeCheckStuff<MultilinearPolynomial<F>>;
/// Note –– F: JoltField bound is not enforced.
///
/// See issue #112792 <https://github.com/rust-lang/rust/issues/112792>.
/// Adding #![feature(lazy_type_alias)] to the crate attributes seem to break
/// `alloy_sol_types`.
pub type TimestampRangeCheckOpenings<F: JoltField> = TimestampRangeCheckStuff<F>;
/// Note –– PCS: CommitmentScheme bound is not enforced.
///
/// See issue #112792 <https://github.com/rust-lang/rust/issues/112792>.
/// Adding #![feature(lazy_type_alias)] to the crate attributes seem to break
/// `alloy_sol_types`.
pub type TimestampRangeCheckCommitments<
    PCS: CommitmentScheme<ProofTranscript>,
    ProofTranscript: Transcript,
> = TimestampRangeCheckStuff<PCS::Commitment>;

impl<T: CanonicalSerialize + CanonicalDeserialize + Default> Initializable<T, NoPreprocessing>
    for TimestampRangeCheckStuff<T>
{
}

pub type ReadTimestampOpenings<F> = [F; MEMORY_OPS_PER_INSTRUCTION];
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

impl<F, PCS, ProofTranscript> TimestampValidityProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    #[tracing::instrument(skip_all, name = "TimestampRangeCheckWitness::new")]
    pub fn generate_witness(
        read_write_memory_polys: &ReadWriteMemoryPolynomials<F>,
    ) -> TimestampRangeCheckPolynomials<F> {
        let read_timestamps: [&CompactPolynomial<u32, F>; 4] = [
            (&read_write_memory_polys.t_read_rd).try_into().unwrap(),
            (&read_write_memory_polys.t_read_rs1).try_into().unwrap(),
            (&read_write_memory_polys.t_read_rs2).try_into().unwrap(),
            (&read_write_memory_polys.t_read_ram).try_into().unwrap(),
        ];
        let M = read_timestamps[0].len();

        #[cfg(test)]
        let mut init_tuples: HashSet<(u32, u32)> = HashSet::new();
        #[cfg(test)]
        {
            for i in 0..M {
                init_tuples.insert((i as u32, 0));
            }
        }

        let mut read_and_final_cts: Vec<[Vec<u32>; 4]> = (0..MEMORY_OPS_PER_INSTRUCTION)
            .into_par_iter()
            .map(|i| {
                let mut read_cts_read_timestamp: Vec<u32> = vec![0; M];
                let mut read_cts_global_minus_read: Vec<u32> = vec![0; M];
                let mut final_cts_read_timestamp: Vec<u32> = vec![0; M];
                let mut final_cts_global_minus_read: Vec<u32> = vec![0; M];

                for (j, read_timestamp) in read_timestamps[i].iter().enumerate() {
                    read_cts_read_timestamp[j] = final_cts_read_timestamp[*read_timestamp as usize];
                    final_cts_read_timestamp[*read_timestamp as usize] += 1;
                    let lookup_index = j - *read_timestamp as usize;
                    read_cts_global_minus_read[j] = final_cts_global_minus_read[lookup_index];
                    final_cts_global_minus_read[lookup_index] += 1;
                }

                #[cfg(test)]
                {
                    let global_minus_read_timestamps: Vec<_> = read_timestamps[i]
                        .iter()
                        .enumerate()
                        .map(|(j, timestamp)| j as u32 - *timestamp)
                        .collect();

                    for (lookup_indices, read_cts, final_cts) in [
                        (
                            &read_timestamps[i].coeffs,
                            &read_cts_read_timestamp,
                            &final_cts_read_timestamp,
                        ),
                        (
                            &global_minus_read_timestamps,
                            &read_cts_global_minus_read,
                            &final_cts_global_minus_read,
                        ),
                    ]
                    .iter()
                    {
                        let mut read_tuples: HashSet<(u32, u32)> = HashSet::new();
                        let mut write_tuples: HashSet<(u32, u32)> = HashSet::new();
                        for (v, t) in lookup_indices.iter().zip(read_cts.iter()) {
                            read_tuples.insert((*v, *t));
                            write_tuples.insert((*v, *t + 1));
                        }

                        let mut final_tuples: HashSet<(u32, u32)> = HashSet::new();
                        for (i, t) in final_cts.iter().enumerate() {
                            final_tuples.insert((i as u32, *t));
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
            .par_iter_mut()
            .map(|cts| {
                let cts = std::mem::take(&mut cts[0]);
                MultilinearPolynomial::from(cts)
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let read_cts_global_minus_read = read_and_final_cts
            .par_iter_mut()
            .map(|cts| {
                let cts = std::mem::take(&mut cts[1]);
                MultilinearPolynomial::from(cts)
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let final_cts_read_timestamp = read_and_final_cts
            .par_iter_mut()
            .map(|cts| {
                let cts = std::mem::take(&mut cts[2]);
                MultilinearPolynomial::from(cts)
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let final_cts_global_minus_read = read_and_final_cts
            .par_iter_mut()
            .map(|cts| {
                let cts = std::mem::take(&mut cts[3]);
                MultilinearPolynomial::from(cts)
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        TimestampRangeCheckPolynomials {
            read_cts_read_timestamp,
            read_cts_global_minus_read,
            final_cts_read_timestamp,
            final_cts_global_minus_read,
            identity: None,
        }
    }
}

impl<F, PCS, ProofTranscript> MemoryCheckingProver<F, PCS, ProofTranscript>
    for TimestampValidityProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    type ReadWriteGrandProduct = BatchedDenseGrandProduct<F>;
    // Init/final grand products are batched together with read/write grand products
    type InitFinalGrandProduct = NoopGrandProduct;

    type Polynomials = TimestampRangeCheckPolynomials<F>;
    type Openings = TimestampRangeCheckOpenings<F>;
    type Commitments = TimestampRangeCheckCommitments<PCS, ProofTranscript>;
    type ExogenousOpenings = ReadTimestampOpenings<F>;

    type Preprocessing = NoPreprocessing;

    type MemoryTuple = (F, F); // a = v for all range check tuples

    fn prove_memory_checking(
        _: &PCS::Setup,
        _: &NoPreprocessing,
        _: &Self::Polynomials,
        _: &JoltPolynomials<F>,
        _: &mut ProverOpeningAccumulator<F, ProofTranscript>,
        _: &mut ProofTranscript,
    ) -> MemoryCheckingProof<F, PCS, Self::Openings, Self::ExogenousOpenings, ProofTranscript> {
        unimplemented!("Use TimestampValidityProof::prove instead");
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
        let read_timestamps: [&CompactPolynomial<u32, F>; 4] = [
            (&jolt_polynomials.read_write_memory.t_read_rd)
                .try_into()
                .unwrap(),
            (&jolt_polynomials.read_write_memory.t_read_rs1)
                .try_into()
                .unwrap(),
            (&jolt_polynomials.read_write_memory.t_read_rs2)
                .try_into()
                .unwrap(),
            (&jolt_polynomials.read_write_memory.t_read_ram)
                .try_into()
                .unwrap(),
        ];
        let read_cts_read_timestamp: [&CompactPolynomial<u32, F>; MEMORY_OPS_PER_INSTRUCTION] =
            polynomials
                .read_cts_read_timestamp
                .iter()
                .map(|poly| poly.try_into().unwrap())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
        let read_cts_global_minus_read: [&CompactPolynomial<u32, F>; MEMORY_OPS_PER_INSTRUCTION] =
            polynomials
                .read_cts_global_minus_read
                .iter()
                .map(|poly| poly.try_into().unwrap())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();

        let M = read_timestamps[0].len();

        let read_write_leaves: Vec<Vec<F>> = (0..MEMORY_OPS_PER_INSTRUCTION)
            .into_par_iter()
            .flat_map(|i| {
                let read_fingerprints_0: Vec<F> = (0..M)
                    .into_par_iter()
                    .map(|j| {
                        read_timestamps[i][j].field_mul(*gamma)
                            + F::from_u32(read_cts_read_timestamp[i][j])
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
                        let global_minus_read = j as u32 - read_timestamps[i][j];
                        global_minus_read.field_mul(*gamma)
                            + F::from_u32(read_cts_global_minus_read[i][j])
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
                // t = 0
                (i as u64).field_mul(*gamma) - *tau
            })
            .collect();

        let final_cts_read_timestamp: [&CompactPolynomial<u32, F>; MEMORY_OPS_PER_INSTRUCTION] =
            polynomials
                .final_cts_read_timestamp
                .iter()
                .map(|poly| poly.try_into().unwrap())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
        let final_cts_global_minus_read: [&CompactPolynomial<u32, F>; MEMORY_OPS_PER_INSTRUCTION] =
            polynomials
                .final_cts_global_minus_read
                .iter()
                .map(|poly| poly.try_into().unwrap())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
        leaves.par_extend(
            (0..MEMORY_OPS_PER_INSTRUCTION)
                .into_par_iter()
                .flat_map(|i| {
                    let final_fingerprints_0 = (0..M)
                        .into_par_iter()
                        .map(|j| F::from_u32(final_cts_read_timestamp[i][j]) + init_leaves[j])
                        .collect();

                    let final_fingerprints_1 = (0..M)
                        .into_par_iter()
                        .map(|j| F::from_u32(final_cts_global_minus_read[i][j]) + init_leaves[j])
                        .collect();

                    [final_fingerprints_0, final_fingerprints_1]
                }),
        );
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
        let num_memories = 2 * MEMORY_OPS_PER_INSTRUCTION;

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
        let num_memories = 2 * MEMORY_OPS_PER_INSTRUCTION;
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
    for TimestampValidityProof<F, PCS, ProofTranscript>
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
        unimplemented!("Use TimestampValidityProof::verify instead");
    }

    fn read_tuples(
        _: &NoPreprocessing,
        openings: &Self::Openings,
        read_timestamp_openings: &[F; MEMORY_OPS_PER_INSTRUCTION],
    ) -> Vec<Self::MemoryTuple> {
        (0..MEMORY_OPS_PER_INSTRUCTION)
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
        read_timestamp_openings: &[F; MEMORY_OPS_PER_INSTRUCTION],
    ) -> Vec<Self::MemoryTuple> {
        (0..MEMORY_OPS_PER_INSTRUCTION)
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
        _: &[F; MEMORY_OPS_PER_INSTRUCTION],
    ) -> Vec<Self::MemoryTuple> {
        vec![(openings.identity.unwrap(), F::zero())]
    }

    fn final_tuples(
        _: &NoPreprocessing,
        openings: &Self::Openings,
        _: &[F; MEMORY_OPS_PER_INSTRUCTION],
    ) -> Vec<Self::MemoryTuple> {
        (0..MEMORY_OPS_PER_INSTRUCTION)
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
pub struct TimestampValidityProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    multiset_hashes: MultisetHashes<F>,
    openings: TimestampRangeCheckOpenings<F>,
    exogenous_openings: ReadTimestampOpenings<F>,
    batched_grand_product: BatchedGrandProductProof<PCS, ProofTranscript>,
}

impl<F, PCS, ProofTranscript> TimestampValidityProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    #[tracing::instrument(skip_all, name = "TimestampValidityProof::prove")]
    pub fn prove<'a>(
        generators: &PCS::Setup,
        polynomials: &'a TimestampRangeCheckPolynomials<F>,
        jolt_polynomials: &'a JoltPolynomials<F>,
        opening_accumulator: &mut ProverOpeningAccumulator<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Self {
        let (batched_grand_product, multiset_hashes, r_grand_product) =
            TimestampValidityProof::prove_grand_products(
                polynomials,
                jolt_polynomials,
                opening_accumulator,
                transcript,
                generators,
            );

        let mut openings = TimestampRangeCheckOpenings::default();
        let mut timestamp_openings = ReadTimestampOpenings::<F>::default();

        let batch_size = multiset_hashes.read_hashes.len()
            + multiset_hashes.write_hashes.len()
            + multiset_hashes.init_hashes.len()
            + multiset_hashes.final_hashes.len();
        let (_, r_opening) = r_grand_product.split_at(batch_size.next_power_of_two().log_2());

        let read_write_polys = [
            polynomials.read_write_values(),
            ReadTimestampOpenings::<F>::exogenous_data(jolt_polynomials),
        ]
        .concat();
        let read_write_openings: Vec<&mut F> = openings
            .read_write_values_mut()
            .into_iter()
            .chain(timestamp_openings.openings_mut().into_iter())
            .collect();
        let (read_write_evals, chis) =
            MultilinearPolynomial::batch_evaluate(&read_write_polys, r_opening);
        for (opening, eval) in read_write_openings.into_iter().zip(read_write_evals.iter()) {
            *opening = *eval;
        }

        opening_accumulator.append(
            &read_write_polys,
            DensePolynomial::new(chis),
            r_opening.to_vec(),
            &read_write_evals,
            transcript,
        );

        Self {
            multiset_hashes,
            openings,
            exogenous_openings: timestamp_openings,
            batched_grand_product,
        }
    }

    #[tracing::instrument(skip_all, name = "TimestampValidityProof::prove_grand_products")]
    fn prove_grand_products(
        polynomials: &TimestampRangeCheckPolynomials<F>,
        jolt_polynomials: &JoltPolynomials<F>,
        opening_accumulator: &mut ProverOpeningAccumulator<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
        setup: &PCS::Setup,
    ) -> (
        BatchedGrandProductProof<PCS, ProofTranscript>,
        MultisetHashes<F>,
        Vec<F>,
    ) {
        // Fiat-Shamir randomness for multiset hashes
        let gamma: F = transcript.challenge_scalar();
        let tau: F = transcript.challenge_scalar();

        let protocol_name = Self::protocol_name();
        transcript.append_message(protocol_name);

        let (leaves, _) = TimestampValidityProof::<F, PCS, ProofTranscript>::compute_leaves(
            &NoPreprocessing,
            polynomials,
            jolt_polynomials,
            &gamma,
            &tau,
        );

        let mut batched_circuit = <BatchedDenseGrandProduct<F> as BatchedGrandProduct<
            F,
            PCS,
            ProofTranscript,
        >>::construct(leaves);

        let hashes: Vec<F> = <BatchedDenseGrandProduct<F> as BatchedGrandProduct<
            F,
            PCS,
            ProofTranscript,
        >>::claimed_outputs(&batched_circuit);
        let (read_write_hashes, init_final_hashes) =
            hashes.split_at(4 * MEMORY_OPS_PER_INSTRUCTION);
        let multiset_hashes =
            TimestampValidityProof::<F, PCS, ProofTranscript>::uninterleave_hashes(
                &NoPreprocessing,
                read_write_hashes.to_vec(),
                init_final_hashes.to_vec(),
            );
        #[cfg(test)]
        TimestampValidityProof::<F, PCS, ProofTranscript>::check_multiset_equality(
            &NoPreprocessing,
            &multiset_hashes,
        );
        multiset_hashes.append_to_transcript(transcript);

        let (batched_grand_product, r_grand_product) =
            batched_circuit.prove_grand_product(Some(opening_accumulator), transcript, Some(setup));

        drop_in_background_thread(batched_circuit);

        (batched_grand_product, multiset_hashes, r_grand_product)
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
        TimestampValidityProof::<F, PCS, ProofTranscript>::check_multiset_equality(
            &NoPreprocessing,
            &self.multiset_hashes,
        );
        self.multiset_hashes.append_to_transcript(transcript);

        let (read_write_hashes, init_final_hashes) =
            TimestampValidityProof::<F, PCS, ProofTranscript>::interleave(
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

        let read_hashes: Vec<_> = TimestampValidityProof::<F, PCS, ProofTranscript>::read_tuples(
            &NoPreprocessing,
            &self.openings,
            &self.exogenous_openings,
        )
        .iter()
        .map(|tuple| {
            TimestampValidityProof::<F, PCS, ProofTranscript>::fingerprint(tuple, &gamma, &tau)
        })
        .collect();
        let write_hashes: Vec<_> = TimestampValidityProof::<F, PCS, ProofTranscript>::write_tuples(
            &NoPreprocessing,
            &self.openings,
            &self.exogenous_openings,
        )
        .iter()
        .map(|tuple| {
            TimestampValidityProof::<F, PCS, ProofTranscript>::fingerprint(tuple, &gamma, &tau)
        })
        .collect();
        let init_hashes: Vec<_> = TimestampValidityProof::<F, PCS, ProofTranscript>::init_tuples(
            &NoPreprocessing,
            &self.openings,
            &self.exogenous_openings,
        )
        .iter()
        .map(|tuple| {
            TimestampValidityProof::<F, PCS, ProofTranscript>::fingerprint(tuple, &gamma, &tau)
        })
        .collect();
        let final_hashes: Vec<_> = TimestampValidityProof::<F, PCS, ProofTranscript>::final_tuples(
            &NoPreprocessing,
            &self.openings,
            &self.exogenous_openings,
        )
        .iter()
        .map(|tuple| {
            TimestampValidityProof::<F, PCS, ProofTranscript>::fingerprint(tuple, &gamma, &tau)
        })
        .collect();

        let (read_write_hashes, init_final_hashes) =
            TimestampValidityProof::<F, PCS, ProofTranscript>::interleave(
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

    fn protocol_name() -> &'static [u8] {
        b"Timestamp Validity Proof"
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;

    use super::*;

    #[test]
    fn timestamp_range_check_stuff_ordering() {
        TimestampRangeCheckOpenings::<Fr>::test_ordering_consistency(&NoPreprocessing);
    }
}
