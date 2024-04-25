use crate::field::JoltField;
use crate::subprotocols::grand_product::{
    BatchedDenseGrandProduct, BatchedGrandProduct, BatchedGrandProductLayer,
    BatchedGrandProductProof,
};
use crate::utils::thread::drop_in_background_thread;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::MEMORY_OPS_PER_INSTRUCTION;
use itertools::interleave;
use rayon::iter::{
    IntoParallelIterator, IntoParallelRefIterator, ParallelExtend, ParallelIterator,
};
#[cfg(test)]
use std::collections::HashSet;
use std::{iter::zip, marker::PhantomData};

use crate::poly::commitment::commitment_scheme::{BatchType, CommitShape, CommitmentScheme};
use crate::utils::transcript::AppendToTranscript;
use crate::{
    lasso::memory_checking::{
        MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier, MultisetHashes,
        NoPreprocessing,
    },
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        structured_poly::{StructuredCommitment, StructuredOpeningProof},
    },
    utils::{errors::ProofVerifyError, mul_0_1_optimized, transcript::ProofTranscript},
};

use super::read_write_memory::MemoryCommitment;

pub struct RangeCheckPolynomials<F, C>
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
{
    _group: PhantomData<C>,
    pub read_timestamps: [Vec<u64>; MEMORY_OPS_PER_INSTRUCTION],
    pub read_cts_read_timestamp: [DensePolynomial<F>; MEMORY_OPS_PER_INSTRUCTION],
    pub read_cts_global_minus_read: [DensePolynomial<F>; MEMORY_OPS_PER_INSTRUCTION],
    pub final_cts_read_timestamp: [DensePolynomial<F>; MEMORY_OPS_PER_INSTRUCTION],
    pub final_cts_global_minus_read: [DensePolynomial<F>; MEMORY_OPS_PER_INSTRUCTION],
}

impl<F, C> RangeCheckPolynomials<F, C>
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
{
    #[tracing::instrument(skip_all, name = "RangeCheckPolynomials::new")]
    pub fn new(read_timestamps: [Vec<u64>; MEMORY_OPS_PER_INSTRUCTION]) -> Self {
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

        Self {
            _group: PhantomData,
            read_timestamps,
            read_cts_read_timestamp,
            read_cts_global_minus_read,
            final_cts_read_timestamp,
            final_cts_global_minus_read,
        }
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct RangeCheckCommitment<C: CommitmentScheme> {
    pub(super) commitments: Vec<C::Commitment>,
}

impl<C: CommitmentScheme> AppendToTranscript for RangeCheckCommitment<C> {
    fn append_to_transcript(&self, label: &'static [u8], transcript: &mut ProofTranscript) {
        transcript.append_message(label, b"RangeCheckCommitment_begin");
        for commitment in &self.commitments {
            commitment.append_to_transcript(b"range", transcript);
        }
        transcript.append_message(label, b"RangeCheckCommitment_end");
    }
}

impl<F, C> StructuredCommitment<C> for RangeCheckPolynomials<F, C>
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
{
    type Commitment = RangeCheckCommitment<C>;

    #[tracing::instrument(skip_all, name = "RangeCheckPolynomials::commit")]
    fn commit(&self, generators: &C::Setup) -> Self::Commitment {
        let polys: Vec<&DensePolynomial<F>> = self
            .read_cts_read_timestamp
            .iter()
            .chain(self.read_cts_global_minus_read.iter())
            .chain(self.final_cts_read_timestamp.iter())
            .chain(self.final_cts_global_minus_read.iter())
            .collect();
        let commitments = C::batch_commit_polys_ref(&polys, generators, BatchType::Big);

        Self::Commitment { commitments }
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct RangeCheckOpenings<F, C>
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
{
    read_cts_read_timestamp: [F; MEMORY_OPS_PER_INSTRUCTION],
    read_cts_global_minus_read: [F; MEMORY_OPS_PER_INSTRUCTION],
    final_cts_read_timestamp: [F; MEMORY_OPS_PER_INSTRUCTION],
    final_cts_global_minus_read: [F; MEMORY_OPS_PER_INSTRUCTION],
    memory_t_read: [F; MEMORY_OPS_PER_INSTRUCTION],
    identity_poly_opening: Option<F>,
}

impl<F, C> StructuredOpeningProof<F, C, RangeCheckPolynomials<F, C>> for RangeCheckOpenings<F, C>
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
{
    type Proof = C::BatchedProof;

    fn open(_polynomials: &RangeCheckPolynomials<F, C>, _opening_point: &[F]) -> Self {
        unimplemented!("Openings are computed in TimestampValidityProof::prove");
    }

    fn prove_openings(
        _generators: &C::Setup,
        _polynomials: &RangeCheckPolynomials<F, C>,
        _opening_point: &[F],
        _openings: &RangeCheckOpenings<F, C>,
        _transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        unimplemented!("Openings are proved in TimestampValidityProof::prove")
    }

    fn compute_verifier_openings(&mut self, _: &NoPreprocessing, opening_point: &[F]) {
        self.identity_poly_opening =
            Some(IdentityPolynomial::new(opening_point.len()).evaluate(opening_point));
    }

    fn verify_openings(
        &self,
        _generators: &C::Setup,
        _opening_proof: &Self::Proof,
        _commitment: &RangeCheckCommitment<C>,
        _opening_point: &[F],
        _transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        unimplemented!("Openings are verified in TimestampValidityProof::verify");
    }
}

impl<F, C> MemoryCheckingProver<F, C, RangeCheckPolynomials<F, C>> for TimestampValidityProof<F, C>
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
{
    // Init/final grand products are batched together with read/write grand products
    type InitFinalGrandProduct = NoopGrandProduct;

    type ReadWriteOpenings = RangeCheckOpenings<F, C>;
    type InitFinalOpenings = RangeCheckOpenings<F, C>;

    fn prove_memory_checking(
        _generators: &C::Setup,
        _: &NoPreprocessing,
        _polynomials: &RangeCheckPolynomials<F, C>,
        _transcript: &mut ProofTranscript,
    ) -> MemoryCheckingProof<
        F,
        C,
        RangeCheckPolynomials<F, C>,
        Self::ReadWriteOpenings,
        Self::InitFinalOpenings,
    > {
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
        polynomials: &RangeCheckPolynomials<F, C>,
        gamma: &F,
        tau: &F,
    ) -> (Vec<Vec<F>>, ()) {
        let M = polynomials.read_timestamps[0].len();
        let gamma_squared = gamma.square();

        let read_write_leaves: Vec<Vec<F>> = (0..MEMORY_OPS_PER_INSTRUCTION)
            .into_par_iter()
            .flat_map(|i| {
                let read_fingerprints_0: Vec<F> = (0..M)
                    .into_par_iter()
                    .map(|j| {
                        let read_timestamp =
                            F::from_u64(polynomials.read_timestamps[i][j]).unwrap();
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
                            F::from_u64(j as u64 - polynomials.read_timestamps[i][j]).unwrap();
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
        b"Timestamp validity proof memory checking"
    }
}

impl<F, C> MemoryCheckingVerifier<F, C, RangeCheckPolynomials<F, C>>
    for TimestampValidityProof<F, C>
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
{
    fn verify_memory_checking(
        _: &NoPreprocessing,
        _: &C::Setup,
        mut _proof: MemoryCheckingProof<
            F,
            C,
            RangeCheckPolynomials<F, C>,
            Self::ReadWriteOpenings,
            Self::InitFinalOpenings,
        >,
        _commitments: &RangeCheckCommitment<C>,
        _transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        unimplemented!("Use TimestampValidityProof::verify instead");
    }

    fn read_tuples(
        _: &NoPreprocessing,
        openings: &Self::ReadWriteOpenings,
    ) -> Vec<Self::MemoryTuple> {
        let t_read_openings = openings.memory_t_read;

        (0..MEMORY_OPS_PER_INSTRUCTION)
            .flat_map(|i| {
                [
                    (
                        t_read_openings[i],
                        t_read_openings[i],
                        openings.read_cts_read_timestamp[i],
                    ),
                    (
                        openings.identity_poly_opening.unwrap() - t_read_openings[i],
                        openings.identity_poly_opening.unwrap() - t_read_openings[i],
                        openings.read_cts_global_minus_read[i],
                    ),
                ]
            })
            .collect()
    }

    fn write_tuples(
        _: &NoPreprocessing,
        openings: &Self::ReadWriteOpenings,
    ) -> Vec<Self::MemoryTuple> {
        let t_read_openings = openings.memory_t_read;

        (0..MEMORY_OPS_PER_INSTRUCTION)
            .flat_map(|i| {
                [
                    (
                        t_read_openings[i],
                        t_read_openings[i],
                        openings.read_cts_read_timestamp[i] + F::one(),
                    ),
                    (
                        openings.identity_poly_opening.unwrap() - t_read_openings[i],
                        openings.identity_poly_opening.unwrap() - t_read_openings[i],
                        openings.read_cts_global_minus_read[i] + F::one(),
                    ),
                ]
            })
            .collect()
    }

    fn init_tuples(
        _: &NoPreprocessing,
        openings: &Self::InitFinalOpenings,
    ) -> Vec<Self::MemoryTuple> {
        vec![(
            openings.identity_poly_opening.unwrap(),
            openings.identity_poly_opening.unwrap(),
            F::zero(),
        )]
    }

    fn final_tuples(
        _: &NoPreprocessing,
        openings: &Self::InitFinalOpenings,
    ) -> Vec<Self::MemoryTuple> {
        (0..MEMORY_OPS_PER_INSTRUCTION)
            .flat_map(|i| {
                [
                    (
                        openings.identity_poly_opening.unwrap(),
                        openings.identity_poly_opening.unwrap(),
                        openings.final_cts_read_timestamp[i],
                    ),
                    (
                        openings.identity_poly_opening.unwrap(),
                        openings.identity_poly_opening.unwrap(),
                        openings.final_cts_global_minus_read[i],
                    ),
                ]
            })
            .collect()
    }
}

pub struct NoopGrandProduct;
impl<F: JoltField, C: CommitmentScheme<Field = F>> BatchedGrandProduct<F, C> for NoopGrandProduct {
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
        vec![].into_iter() // Needed to compile
    }

    fn prove_grand_product(
        &mut self,
        _transcript: &mut ProofTranscript,
        _setup: Option<&C::Setup>,
    ) -> (BatchedGrandProductProof<C>, Vec<F>) {
        unimplemented!("init/final grand products are batched with read/write grand products")
    }
    fn verify_grand_product(
        _proof: &BatchedGrandProductProof<C>,
        _claims: &Vec<F>,
        _transcript: &mut ProofTranscript,
        _setup: Option<&C::Setup>,
    ) -> (Vec<F>, Vec<F>) {
        unimplemented!("init/final grand products are batched with read/write grand products")
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct TimestampValidityProof<F, C>
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
{
    multiset_hashes: MultisetHashes<F>,
    openings: RangeCheckOpenings<F, C>,
    opening_proof: C::BatchedProof,
    batched_grand_product: BatchedGrandProductProof<C>,
}

impl<F, C> TimestampValidityProof<F, C>
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
{
    #[tracing::instrument(skip_all, name = "TimestampValidityProof::prove")]
    pub fn prove(
        generators: &C::Setup,
        range_check_polys: &RangeCheckPolynomials<F, C>,
        t_read_polynomials: &[DensePolynomial<F>; MEMORY_OPS_PER_INSTRUCTION],
        transcript: &mut ProofTranscript,
    ) -> Self {
        let (batched_grand_product, multiset_hashes, r_grand_product) =
            TimestampValidityProof::prove_grand_products(range_check_polys, transcript);

        let polys_iter = range_check_polys
            .read_cts_read_timestamp
            .par_iter()
            .chain(range_check_polys.read_cts_global_minus_read.par_iter())
            .chain(range_check_polys.final_cts_read_timestamp.par_iter())
            .chain(range_check_polys.final_cts_global_minus_read.par_iter())
            .chain(t_read_polynomials.par_iter());

        let polys: Vec<_> = polys_iter.clone().collect();

        let chis = EqPolynomial::evals(&r_grand_product);
        let openings = polys_iter
            .clone()
            .map(|poly| poly.evaluate_at_chi(&chis))
            .collect::<Vec<F>>();

        let opening_proof = C::batch_prove(
            generators,
            &polys,
            &r_grand_product,
            &openings,
            BatchType::Big,
            transcript,
        );

        let mut openings = openings.into_iter();
        let read_cts_read_timestamp: [F; MEMORY_OPS_PER_INSTRUCTION] =
            openings.next_chunk().unwrap();
        let read_cts_global_minus_read = openings.next_chunk().unwrap();
        let final_cts_read_timestamp = openings.next_chunk().unwrap();
        let final_cts_global_minus_read = openings.next_chunk().unwrap();
        let memory_t_read = openings.next_chunk().unwrap();

        let openings = RangeCheckOpenings {
            read_cts_read_timestamp,
            read_cts_global_minus_read,
            final_cts_read_timestamp,
            final_cts_global_minus_read,
            memory_t_read,
            identity_poly_opening: None,
        };

        Self {
            multiset_hashes,
            openings,
            opening_proof,
            batched_grand_product,
        }
    }

    #[tracing::instrument(skip_all, name = "TimestampValidityProof::prove_grand_products")]
    fn prove_grand_products(
        polynomials: &RangeCheckPolynomials<F, C>,
        transcript: &mut ProofTranscript,
    ) -> (BatchedGrandProductProof<C>, MultisetHashes<F>, Vec<F>) {
        // Fiat-Shamir randomness for multiset hashes
        let gamma: F = transcript.challenge_scalar(b"Memory checking gamma");
        let tau: F = transcript.challenge_scalar(b"Memory checking tau");

        transcript.append_protocol_name(Self::protocol_name());

        let (leaves, _) =
            TimestampValidityProof::compute_leaves(&NoPreprocessing, polynomials, &gamma, &tau);

        let mut batched_circuit =
            <BatchedDenseGrandProduct<F> as BatchedGrandProduct<F, C>>::construct(leaves);

        let hashes: Vec<F> =
            <BatchedDenseGrandProduct<F> as BatchedGrandProduct<F, C>>::claims(&batched_circuit);
        let (read_write_hashes, init_final_hashes) =
            hashes.split_at(4 * MEMORY_OPS_PER_INSTRUCTION);
        let multiset_hashes = TimestampValidityProof::<F, C>::uninterleave_hashes(
            &NoPreprocessing,
            read_write_hashes.to_vec(),
            init_final_hashes.to_vec(),
        );
        TimestampValidityProof::<F, C>::check_multiset_equality(&NoPreprocessing, &multiset_hashes);
        multiset_hashes.append_to_transcript(transcript);

        let (batched_grand_product, r_grand_product) =
            batched_circuit.prove_grand_product(transcript, None);

        drop_in_background_thread(batched_circuit);

        (batched_grand_product, multiset_hashes, r_grand_product)
    }

    pub fn verify(
        &mut self,
        generators: &C::Setup,
        range_check_commitment: &RangeCheckCommitment<C>,
        memory_commitment: &MemoryCommitment<C>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        // Fiat-Shamir randomness for multiset hashes
        let gamma: F = transcript.challenge_scalar(b"Memory checking gamma");
        let tau: F = transcript.challenge_scalar(b"Memory checking tau");

        transcript.append_protocol_name(Self::protocol_name());

        // Multiset equality checks
        TimestampValidityProof::<F, C>::check_multiset_equality(
            &NoPreprocessing,
            &self.multiset_hashes,
        );
        self.multiset_hashes.append_to_transcript(transcript);

        let (read_write_hashes, init_final_hashes) =
            TimestampValidityProof::<F, C>::interleave_hashes(
                &NoPreprocessing,
                &self.multiset_hashes,
            );
        let concatenated_hashes = [read_write_hashes, init_final_hashes].concat();
        let (grand_product_claims, r_grand_product) =
            BatchedDenseGrandProduct::verify_grand_product(
                &self.batched_grand_product,
                &concatenated_hashes,
                transcript,
                None,
            );

        let openings: Vec<_> = self
            .openings
            .read_cts_read_timestamp
            .into_iter()
            .chain(self.openings.read_cts_global_minus_read)
            .chain(self.openings.final_cts_read_timestamp)
            .chain(self.openings.final_cts_global_minus_read)
            .chain(self.openings.memory_t_read)
            .collect();

        // TODO(moodlezoup): Make indexing less disgusting
        let t_read_commitments = &memory_commitment.trace_commitments
            [1 + MEMORY_OPS_PER_INSTRUCTION + 5..1 + 2 * MEMORY_OPS_PER_INSTRUCTION + 5];
        let commitments: Vec<_> = range_check_commitment
            .commitments
            .iter()
            .chain(t_read_commitments.iter())
            .collect();

        C::batch_verify(
            &self.opening_proof,
            generators,
            &r_grand_product,
            &openings,
            &commitments,
            transcript,
        )?;

        self.openings
            .compute_verifier_openings(&NoPreprocessing, &r_grand_product);

        let read_hashes: Vec<_> =
            TimestampValidityProof::read_tuples(&NoPreprocessing, &self.openings)
                .iter()
                .map(|tuple| TimestampValidityProof::<F, C>::fingerprint(tuple, &gamma, &tau))
                .collect();
        let write_hashes: Vec<_> =
            TimestampValidityProof::write_tuples(&NoPreprocessing, &self.openings)
                .iter()
                .map(|tuple| TimestampValidityProof::<F, C>::fingerprint(tuple, &gamma, &tau))
                .collect();
        let init_hashes: Vec<_> =
            TimestampValidityProof::init_tuples(&NoPreprocessing, &self.openings)
                .iter()
                .map(|tuple| TimestampValidityProof::<F, C>::fingerprint(tuple, &gamma, &tau))
                .collect();
        let final_hashes: Vec<_> =
            TimestampValidityProof::final_tuples(&NoPreprocessing, &self.openings)
                .iter()
                .map(|tuple| TimestampValidityProof::<F, C>::fingerprint(tuple, &gamma, &tau))
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
            TimestampValidityProof::<F, C>::interleave_hashes(&NoPreprocessing, &multiset_hashes);

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
        b"Timestamp validity proof memory checking"
    }
}
