use crate::field::JoltField;
use crate::lasso::memory_checking::{
    ExogenousOpenings, Initializable, StructuredPolynomialData, VerifierComputedOpening,
};
use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use crate::subprotocols::grand_product::{
    BatchedDenseGrandProduct, BatchedGrandProduct, BatchedGrandProductLayer,
    BatchedGrandProductProof,
};
use crate::utils::thread::drop_in_background_thread;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::MEMORY_OPS_PER_INSTRUCTION;
use itertools::interleave;
use rayon::prelude::*;
#[cfg(test)]
use std::collections::HashSet;
use std::iter::zip;

use crate::poly::commitment::commitment_scheme::{BatchType, CommitShape, CommitmentScheme};
use crate::{
    lasso::memory_checking::{
        MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier, MultisetHashes,
        NoPreprocessing,
    },
    poly::{
        dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial, identity_poly::IdentityPolynomial,
    },
    utils::{errors::ProofVerifyError, mul_0_1_optimized, transcript::ProofTranscript},
};

use super::{JoltCommitments, JoltPolynomials, JoltStuff};

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
/// See issue #112792 <https://github.com/rust-lang/rust/issues/112792>.
/// Adding #![feature(lazy_type_alias)] to the crate attributes seem to break
/// `alloy_sol_types`.
pub type TimestampRangeCheckPolynomials<F: JoltField> =
    TimestampRangeCheckStuff<DensePolynomial<F>>;
/// Note –– F: JoltField bound is not enforced.
/// See issue #112792 <https://github.com/rust-lang/rust/issues/112792>.
/// Adding #![feature(lazy_type_alias)] to the crate attributes seem to break
/// `alloy_sol_types`.
pub type TimestampRangeCheckOpenings<F: JoltField> = TimestampRangeCheckStuff<F>;
/// Note –– PCS: CommitmentScheme bound is not enforced.
/// See issue #112792 <https://github.com/rust-lang/rust/issues/112792>.
/// Adding #![feature(lazy_type_alias)] to the crate attributes seem to break
/// `alloy_sol_types`.
pub type TimestampRangeCheckCommitments<PCS: CommitmentScheme> =
    TimestampRangeCheckStuff<PCS::Commitment>;

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
        polys_or_commitments
            .read_write_memory
            .t_read
            .iter()
            .collect()
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> TimestampValidityProof<F, PCS> {
    #[tracing::instrument(skip_all, name = "TimestampRangeCheckWitness::new")]
    pub fn generate_witness(
        read_timestamps: &[Vec<u64>; MEMORY_OPS_PER_INSTRUCTION],
    ) -> TimestampRangeCheckPolynomials<F> {
        let M = read_timestamps[0].len();

        #[cfg(test)]
        let mut init_tuples: HashSet<(u64, u64, u64)> = HashSet::new();
        #[cfg(test)]
        {
            for i in 0..M {
                init_tuples.insert((i as u64, i as u64, 0u64));
            }
        }

        let read_and_final_cts: Vec<[Vec<u64>; 4]> = (0..MEMORY_OPS_PER_INSTRUCTION)
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
                        let mut read_tuples: HashSet<(u64, u64, u64)> = HashSet::new();
                        let mut write_tuples: HashSet<(u64, u64, u64)> = HashSet::new();
                        for (v, t) in lookup_indices.iter().zip(read_cts.iter()) {
                            read_tuples.insert((*v, *v, *t));
                            write_tuples.insert((*v, *v, *t + 1));
                        }

                        let mut final_tuples: HashSet<(u64, u64, u64)> = HashSet::new();
                        for (i, t) in final_cts.iter().enumerate() {
                            final_tuples.insert((i as u64, i as u64, *t));
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

        TimestampRangeCheckPolynomials {
            read_cts_read_timestamp,
            read_cts_global_minus_read,
            final_cts_read_timestamp,
            final_cts_global_minus_read,
            identity: None,
        }
    }
}

impl<F, PCS> MemoryCheckingProver<F, PCS> for TimestampValidityProof<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    type Polynomials = TimestampRangeCheckPolynomials<F>;
    type Openings = TimestampRangeCheckOpenings<F>;
    type Commitments = TimestampRangeCheckCommitments<PCS>;
    type ExogenousOpenings = ReadTimestampOpenings<F>;

    // Init/final grand products are batched together with read/write grand products
    type InitFinalGrandProduct = NoopGrandProduct;

    fn prove_memory_checking(
        _: &PCS::Setup,
        _: &NoPreprocessing,
        _: &Self::Polynomials,
        _: &JoltPolynomials<F>,
        _: &mut ProverOpeningAccumulator<F>,
        _: &mut ProofTranscript,
    ) -> MemoryCheckingProof<F, PCS, Self::Openings, Self::ExogenousOpenings> {
        unimplemented!("Use TimestampValidityProof::prove instead");
    }

    fn fingerprint(inputs: &(F, F, F), gamma: &F, tau: &F) -> F {
        let (a, v, t) = *inputs;
        t * gamma.square() + v * *gamma + a - *tau
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
    ) -> (Vec<Vec<F>>, ()) {
        let read_timestamps = &jolt_polynomials.read_write_memory.t_read;

        let M = read_timestamps[0].len();
        let gamma_squared = gamma.square();

        let read_write_leaves: Vec<Vec<F>> = (0..MEMORY_OPS_PER_INSTRUCTION)
            .into_par_iter()
            .flat_map(|i| {
                let read_fingerprints_0: Vec<F> = (0..M)
                    .into_par_iter()
                    .map(|j| {
                        let read_timestamp = read_timestamps[i][j];
                        polynomials.read_cts_read_timestamp[i][j] * gamma_squared
                            + read_timestamp * *gamma
                            + read_timestamp
                            - *tau
                    })
                    .collect();
                let write_fingeprints_0 = read_fingerprints_0
                    .par_iter()
                    .map(|read_fingerprint| *read_fingerprint + gamma_squared)
                    .collect();

                let read_fingerprints_1: Vec<F> = (0..M)
                    .into_par_iter()
                    .map(|j| {
                        let global_minus_read =
                            F::from_u64(j as u64).unwrap() - read_timestamps[i][j];
                        polynomials.read_cts_global_minus_read[i][j] * gamma_squared
                            + global_minus_read * *gamma
                            + global_minus_read
                            - *tau
                    })
                    .collect();
                let write_fingeprints_1 = read_fingerprints_1
                    .par_iter()
                    .map(|read_fingerprint| *read_fingerprint + gamma_squared)
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
                // 0 * gamma^2 +
                index * *gamma + index - *tau
            })
            .collect();

        leaves.par_extend(
            (0..MEMORY_OPS_PER_INSTRUCTION)
                .into_par_iter()
                .flat_map(|i| {
                    let final_fingerprints_0 = (0..M)
                        .into_par_iter()
                        .map(|j| {
                            mul_0_1_optimized(
                                &polynomials.final_cts_read_timestamp[i][j],
                                &gamma_squared,
                            ) + init_leaves[j]
                        })
                        .collect();

                    let final_fingerprints_1 = (0..M)
                        .into_par_iter()
                        .map(|j| {
                            mul_0_1_optimized(
                                &polynomials.final_cts_global_minus_read[i][j],
                                &gamma_squared,
                            ) + init_leaves[j]
                        })
                        .collect();

                    [final_fingerprints_0, final_fingerprints_1]
                }),
        );
        leaves.push(init_leaves);

        (leaves, ())
    }

    fn interleave_hashes(
        _: &NoPreprocessing,
        multiset_hashes: &MultisetHashes<F>,
    ) -> (Vec<F>, Vec<F>) {
        let read_write_hashes = interleave(
            multiset_hashes.read_hashes.clone(),
            multiset_hashes.write_hashes.clone(),
        )
        .collect();
        let mut init_final_hashes = multiset_hashes.final_hashes.clone();
        init_final_hashes.extend(multiset_hashes.init_hashes.clone());

        (read_write_hashes, init_final_hashes)
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

impl<F, PCS> MemoryCheckingVerifier<F, PCS> for TimestampValidityProof<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn compute_verifier_openings(_: &mut Self::Openings, _: &NoPreprocessing, _: &[F], _: &[F]) {
        unimplemented!("")
    }

    fn verify_memory_checking(
        _: &NoPreprocessing,
        _: &PCS::Setup,
        mut _proof: MemoryCheckingProof<F, PCS, Self::Openings, Self::ExogenousOpenings>,
        _commitments: &Self::Commitments,
        _: &JoltCommitments<PCS>,
        _opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS>,
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
                        read_timestamp_openings[i],
                        openings.read_cts_read_timestamp[i],
                    ),
                    (
                        openings.identity.unwrap() - read_timestamp_openings[i],
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
                        read_timestamp_openings[i],
                        openings.read_cts_read_timestamp[i] + F::one(),
                    ),
                    (
                        openings.identity.unwrap() - read_timestamp_openings[i],
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
        vec![(
            openings.identity.unwrap(),
            openings.identity.unwrap(),
            F::zero(),
        )]
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
                        openings.identity.unwrap(),
                        openings.final_cts_read_timestamp[i],
                    ),
                    (
                        openings.identity.unwrap(),
                        openings.identity.unwrap(),
                        openings.final_cts_global_minus_read[i],
                    ),
                ]
            })
            .collect()
    }
}

pub struct NoopGrandProduct;
impl<F: JoltField, PCS: CommitmentScheme<Field = F>> BatchedGrandProduct<F, PCS>
    for NoopGrandProduct
{
    type Leaves = ();

    fn construct(_leaves: Self::Leaves) -> Self {
        unimplemented!("init/final grand products are batched with read/write grand products");
    }
    fn num_layers(&self) -> usize {
        unimplemented!("init/final grand products are batched with read/write grand products");
    }
    fn claims(&self) -> Vec<F> {
        unimplemented!("init/final grand products are batched with read/write grand products");
    }

    fn layers(&'_ mut self) -> impl Iterator<Item = &'_ mut dyn BatchedGrandProductLayer<F>> {
        std::iter::empty() // Needed to compile
    }

    fn prove_grand_product(
        &mut self,
        _opening_accumulator: Option<&mut ProverOpeningAccumulator<F>>,
        _transcript: &mut ProofTranscript,
        _setup: Option<&PCS::Setup>,
    ) -> (BatchedGrandProductProof<PCS>, Vec<F>) {
        unimplemented!("init/final grand products are batched with read/write grand products")
    }
    fn verify_grand_product(
        _proof: &BatchedGrandProductProof<PCS>,
        _claims: &Vec<F>,
        _opening_accumulator: Option<&mut VerifierOpeningAccumulator<F, PCS>>,
        _transcript: &mut ProofTranscript,
        _setup: Option<&PCS::Setup>,
    ) -> (Vec<F>, Vec<F>) {
        unimplemented!("init/final grand products are batched with read/write grand products")
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct TimestampValidityProof<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    multiset_hashes: MultisetHashes<F>,
    openings: TimestampRangeCheckOpenings<F>,
    exogenous_openings: ReadTimestampOpenings<F>,
    batched_grand_product: BatchedGrandProductProof<PCS>,
}

impl<F, PCS> TimestampValidityProof<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    #[tracing::instrument(skip_all, name = "TimestampValidityProof::prove")]
    pub fn prove<'a>(
        generators: &PCS::Setup,
        polynomials: &'a TimestampRangeCheckPolynomials<F>,
        jolt_polynomials: &'a JoltPolynomials<F>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
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

        let chis = EqPolynomial::evals(&r_grand_product);

        polynomials
            .read_write_values()
            .into_par_iter()
            .zip(openings.read_write_values_mut().into_par_iter())
            .chain(
                ReadTimestampOpenings::<F>::exogenous_data(jolt_polynomials)
                    .into_par_iter()
                    .zip(timestamp_openings.openings_mut().into_par_iter()),
            )
            .for_each(|(poly, opening)| {
                let claim = poly.evaluate_at_chi(&chis);
                *opening = claim;
            });

        opening_accumulator.append(
            &polynomials
                .read_write_values()
                .into_iter()
                .chain(ReadTimestampOpenings::<F>::exogenous_data(jolt_polynomials).into_iter())
                .collect::<Vec<_>>(),
            DensePolynomial::new(chis),
            r_grand_product.clone(),
            &openings
                .read_write_values()
                .into_iter()
                .chain(timestamp_openings.openings())
                .collect::<Vec<_>>(),
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
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
        setup: &PCS::Setup,
    ) -> (BatchedGrandProductProof<PCS>, MultisetHashes<F>, Vec<F>) {
        // Fiat-Shamir randomness for multiset hashes
        let gamma: F = transcript.challenge_scalar();
        let tau: F = transcript.challenge_scalar();

        transcript.append_protocol_name(Self::protocol_name());

        let (leaves, _) = TimestampValidityProof::<F, PCS>::compute_leaves(
            &NoPreprocessing,
            polynomials,
            jolt_polynomials,
            &gamma,
            &tau,
        );

        let mut batched_circuit =
            <BatchedDenseGrandProduct<F> as BatchedGrandProduct<F, PCS>>::construct(leaves);

        let hashes: Vec<F> =
            <BatchedDenseGrandProduct<F> as BatchedGrandProduct<F, PCS>>::claims(&batched_circuit);
        let (read_write_hashes, init_final_hashes) =
            hashes.split_at(4 * MEMORY_OPS_PER_INSTRUCTION);
        let multiset_hashes = TimestampValidityProof::<F, PCS>::uninterleave_hashes(
            &NoPreprocessing,
            read_write_hashes.to_vec(),
            init_final_hashes.to_vec(),
        );
        TimestampValidityProof::<F, PCS>::check_multiset_equality(
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
        commitments: &JoltCommitments<PCS>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        // Fiat-Shamir randomness for multiset hashes
        let gamma: F = transcript.challenge_scalar();
        let tau: F = transcript.challenge_scalar();

        transcript.append_protocol_name(Self::protocol_name());

        // Multiset equality checks
        TimestampValidityProof::<F, PCS>::check_multiset_equality(
            &NoPreprocessing,
            &self.multiset_hashes,
        );
        self.multiset_hashes.append_to_transcript(transcript);

        let (read_write_hashes, init_final_hashes) =
            TimestampValidityProof::<F, PCS>::interleave_hashes(
                &NoPreprocessing,
                &self.multiset_hashes,
            );
        let concatenated_hashes = [read_write_hashes, init_final_hashes].concat();
        let (grand_product_claims, r_grand_product) =
            BatchedDenseGrandProduct::verify_grand_product(
                &self.batched_grand_product,
                &concatenated_hashes,
                Some(opening_accumulator),
                transcript,
                Some(generators),
            );

        opening_accumulator.append(
            &commitments
                .timestamp_range_check
                .read_write_values()
                .into_iter()
                .chain(commitments.read_write_memory.t_read.iter())
                .collect::<Vec<_>>(),
            r_grand_product.clone(),
            &self
                .openings
                .read_write_values()
                .into_iter()
                .chain(self.exogenous_openings.iter())
                .collect::<Vec<_>>(),
            transcript,
        );

        self.openings.identity =
            Some(IdentityPolynomial::new(r_grand_product.len()).evaluate(&r_grand_product));

        let read_hashes: Vec<_> = TimestampValidityProof::<F, PCS>::read_tuples(
            &NoPreprocessing,
            &self.openings,
            &self.exogenous_openings,
        )
        .iter()
        .map(|tuple| TimestampValidityProof::<F, PCS>::fingerprint(tuple, &gamma, &tau))
        .collect();
        let write_hashes: Vec<_> = TimestampValidityProof::<F, PCS>::write_tuples(
            &NoPreprocessing,
            &self.openings,
            &self.exogenous_openings,
        )
        .iter()
        .map(|tuple| TimestampValidityProof::<F, PCS>::fingerprint(tuple, &gamma, &tau))
        .collect();
        let init_hashes: Vec<_> = TimestampValidityProof::<F, PCS>::init_tuples(
            &NoPreprocessing,
            &self.openings,
            &self.exogenous_openings,
        )
        .iter()
        .map(|tuple| TimestampValidityProof::<F, PCS>::fingerprint(tuple, &gamma, &tau))
        .collect();
        let final_hashes: Vec<_> = TimestampValidityProof::<F, PCS>::final_tuples(
            &NoPreprocessing,
            &self.openings,
            &self.exogenous_openings,
        )
        .iter()
        .map(|tuple| TimestampValidityProof::<F, PCS>::fingerprint(tuple, &gamma, &tau))
        .collect();

        assert_eq!(
            grand_product_claims.len(),
            6 * MEMORY_OPS_PER_INSTRUCTION + 1
        );
        let (read_write_claims, init_final_claims) =
            grand_product_claims.split_at(4 * MEMORY_OPS_PER_INSTRUCTION);

        let multiset_hashes = MultisetHashes {
            read_hashes,
            write_hashes,
            init_hashes,
            final_hashes,
        };
        let (read_write_hashes, init_final_hashes) =
            TimestampValidityProof::<F, PCS>::interleave_hashes(&NoPreprocessing, &multiset_hashes);

        for (claim, fingerprint) in zip(read_write_claims, read_write_hashes) {
            assert_eq!(*claim, fingerprint);
        }
        for (claim, fingerprint) in zip(init_final_claims, init_final_hashes) {
            assert_eq!(*claim, fingerprint);
        }

        Ok(())
    }

    /// Computes the shape of all commitments.
    pub fn commitment_shapes(max_trace_length: usize) -> Vec<CommitShape> {
        let max_trace_length = max_trace_length.next_power_of_two();

        vec![CommitShape::new(max_trace_length, BatchType::Big)]
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
