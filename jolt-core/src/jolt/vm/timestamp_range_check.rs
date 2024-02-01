use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use common::constants::MEMORY_OPS_PER_INSTRUCTION;
use merlin::Transcript;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::{iter::zip, marker::PhantomData};
use tracing::trace_span;

use crate::{
    lasso::memory_checking::{
        MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier, MultisetHashes,
    },
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        structured_poly::{BatchablePolynomials, StructuredOpeningProof},
    },
    subprotocols::{
        batched_commitment::{BatchedPolynomialCommitment, BatchedPolynomialOpeningProof},
        grand_product::{
            BatchedGrandProductArgument, BatchedGrandProductCircuit, GrandProductCircuit,
        },
    },
    utils::{
        errors::ProofVerifyError, mul_0_1_optimized, random::RandomTape,
        transcript::ProofTranscript,
    },
};

use super::read_write_memory::{
    BatchedMemoryPolynomials, MemoryCommitment, MemoryReadWriteOpenings, ReadWriteMemory,
};

pub struct RangeCheckPolynomials<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    _group: PhantomData<G>,
    pub read_timestamps: [Vec<u64>; MEMORY_OPS_PER_INSTRUCTION],
    pub read_cts_read_timestamp: [DensePolynomial<F>; MEMORY_OPS_PER_INSTRUCTION],
    pub read_cts_global_minus_read: [DensePolynomial<F>; MEMORY_OPS_PER_INSTRUCTION],
    pub final_cts_read_timestamp: [DensePolynomial<F>; MEMORY_OPS_PER_INSTRUCTION],
    pub final_cts_global_minus_read: [DensePolynomial<F>; MEMORY_OPS_PER_INSTRUCTION],
}

impl<F, G> RangeCheckPolynomials<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    #[tracing::instrument(skip_all, name = "RangeCheckPolynomials::new")]
    pub fn new(read_timestamps: [Vec<u64>; MEMORY_OPS_PER_INSTRUCTION]) -> Self {
        let M = read_timestamps[0].len();

        let read_and_final_cts: Vec<[Vec<u64>; 4]> = (0..MEMORY_OPS_PER_INSTRUCTION)
            .into_par_iter()
            .map(|i| {
                let mut read_cts_read_timestamp: Vec<u64> = vec![0; M];
                let mut final_cts_read_timestamp: Vec<u64> = vec![0; M];
                let mut read_cts_global_minus_read: Vec<u64> = vec![0; M];
                let mut final_cts_global_minus_read: Vec<u64> = vec![0; M];

                for (j, read_timestamp) in read_timestamps[i].iter().enumerate() {
                    read_cts_read_timestamp[j] = final_cts_read_timestamp[*read_timestamp as usize];
                    final_cts_read_timestamp[*read_timestamp as usize] += 1;
                    let lookup_index = j - *read_timestamp as usize;
                    read_cts_global_minus_read[j] = final_cts_global_minus_read[lookup_index];
                    final_cts_global_minus_read[lookup_index] += 1;
                }

                [
                    read_cts_read_timestamp,
                    final_cts_read_timestamp,
                    read_cts_global_minus_read,
                    final_cts_global_minus_read,
                ]
            })
            .collect();

        let read_cts_read_timestamp = read_and_final_cts
            .iter()
            .map(|cts| DensePolynomial::from_u64(&cts[0]))
            .collect::<Vec<DensePolynomial<F>>>()
            .try_into()
            .unwrap();
        let read_cts_global_minus_read = read_and_final_cts
            .iter()
            .map(|cts| DensePolynomial::from_u64(&cts[1]))
            .collect::<Vec<DensePolynomial<F>>>()
            .try_into()
            .unwrap();
        let final_cts_read_timestamp = read_and_final_cts
            .iter()
            .map(|cts| DensePolynomial::from_u64(&cts[2]))
            .collect::<Vec<DensePolynomial<F>>>()
            .try_into()
            .unwrap();
        let final_cts_global_minus_read = read_and_final_cts
            .iter()
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

pub type RangeCheckCommitment<G> = BatchedPolynomialCommitment<G>;
pub type BatchedRangeCheckPolynomials<F> = DensePolynomial<F>;

impl<F, G> BatchablePolynomials for RangeCheckPolynomials<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type BatchedPolynomials = BatchedRangeCheckPolynomials<F>;
    type Commitment = RangeCheckCommitment<G>;

    #[tracing::instrument(skip_all, name = "RangeCheckPolynomials::batch")]
    fn batch(&self) -> Self::BatchedPolynomials {
        DensePolynomial::merge(
            self.read_cts_read_timestamp
                .iter()
                .chain(self.read_cts_global_minus_read.iter())
                .chain(self.final_cts_read_timestamp.iter())
                .chain(self.final_cts_global_minus_read.iter()),
        )
    }

    #[tracing::instrument(skip_all, name = "RangeCheckPolynomials::commit")]
    fn commit(batched_polys: &Self::BatchedPolynomials) -> Self::Commitment {
        batched_polys.combined_commit(b"BatchedRangeCheckPolynomials")
    }
}

pub struct RangeCheckOpenings<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    read_cts_read_timestamp: [F; MEMORY_OPS_PER_INSTRUCTION],
    read_cts_global_minus_read: [F; MEMORY_OPS_PER_INSTRUCTION],
    final_cts_read_timestamp: [F; MEMORY_OPS_PER_INSTRUCTION],
    final_cts_global_minus_read: [F; MEMORY_OPS_PER_INSTRUCTION],
    memory_poly_openings: MemoryReadWriteOpenings<F, G>,
    identity_poly_opening: Option<F>,
}

pub struct RangeCheckOpeningProof<G>
where
    G: CurveGroup,
{
    range_check_opening_proof: BatchedPolynomialOpeningProof<G>,
    memory_poly_opening_proof: Option<BatchedPolynomialOpeningProof<G>>,
}

impl<F, G> StructuredOpeningProof<F, G, RangeCheckPolynomials<F, G>> for RangeCheckOpenings<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type Proof = RangeCheckOpeningProof<G>;

    #[tracing::instrument(skip_all, name = "RangeCheckReadWriteOpenings::open")]
    fn open(_polynomials: &RangeCheckPolynomials<F, G>, _opening_point: &Vec<F>) -> Self {
        unimplemented!("Openings are computed in TimestampValidityProof::prove");
    }

    #[tracing::instrument(skip_all, name = "RangeCheckReadWriteOpenings::prove_openings")]
    fn prove_openings(
        polynomials: &BatchedRangeCheckPolynomials<F>,
        commitment: &RangeCheckCommitment<G>,
        opening_point: &Vec<F>,
        openings: &RangeCheckOpenings<F, G>,
        transcript: &mut Transcript,
        random_tape: &mut RandomTape<G>,
    ) -> Self::Proof {
        let range_check_openings: Vec<F> = openings
            .read_cts_read_timestamp
            .into_iter()
            .chain(openings.read_cts_global_minus_read.into_iter())
            .chain(openings.final_cts_read_timestamp.into_iter())
            .chain(openings.final_cts_global_minus_read.into_iter())
            .collect();
        let range_check_opening_proof = BatchedPolynomialOpeningProof::prove(
            &polynomials,
            &range_check_openings,
            opening_point,
            &commitment,
            transcript,
            random_tape,
        );
        RangeCheckOpeningProof {
            range_check_opening_proof,
            memory_poly_opening_proof: None,
        }
    }

    fn compute_verifier_openings(&mut self, opening_point: &Vec<F>) {
        self.identity_poly_opening =
            Some(IdentityPolynomial::new(opening_point.len()).evaluate(opening_point));
    }

    fn verify_openings(
        &self,
        opening_proof: &Self::Proof,
        commitment: &RangeCheckCommitment<G>,
        opening_point: &Vec<F>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        let range_check_openings: Vec<F> = self
            .read_cts_read_timestamp
            .into_iter()
            .chain(self.read_cts_global_minus_read.into_iter())
            .chain(self.final_cts_read_timestamp.into_iter())
            .chain(self.final_cts_global_minus_read.into_iter())
            .collect();
        opening_proof.range_check_opening_proof.verify(
            opening_point,
            &range_check_openings,
            &commitment,
            transcript,
        )
    }
}

impl<F, G> MemoryCheckingProver<F, G, RangeCheckPolynomials<F, G>> for RangeCheckPolynomials<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type ReadWriteOpenings = RangeCheckOpenings<F, G>;
    type InitFinalOpenings = RangeCheckOpenings<F, G>;

    fn prove_memory_checking(
        &self,
        _polynomials: &RangeCheckPolynomials<F, G>,
        _batched_polys: &BatchedRangeCheckPolynomials<F>,
        _commitments: &RangeCheckCommitment<G>,
        _transcript: &mut Transcript,
        _random_tape: &mut RandomTape<G>,
    ) -> MemoryCheckingProof<
        G,
        RangeCheckPolynomials<F, G>,
        Self::ReadWriteOpenings,
        Self::InitFinalOpenings,
    > {
        unimplemented!("Use TimestampValidityProof::prove instead");
    }

    fn fingerprint(inputs: &(F, F, F), gamma: &F, tau: &F) -> F {
        let (a, v, t) = *inputs;
        t * gamma.square() + v * *gamma + a - tau
    }

    #[tracing::instrument(skip_all, name = "RangeCheckPolynomials::compute_leaves")]
    fn compute_leaves(
        &self,
        polynomials: &RangeCheckPolynomials<F, G>,
        gamma: &F,
        tau: &F,
    ) -> (Vec<DensePolynomial<F>>, Vec<DensePolynomial<F>>) {
        let M = polynomials.read_timestamps.len();
        let gamma_squared = gamma.square();

        let read_write_leaves = (0..MEMORY_OPS_PER_INSTRUCTION)
            .into_par_iter()
            .flat_map(|i| {
                let read_fingerprints_0: Vec<F> = (0..M)
                    .into_par_iter()
                    .map(|j| {
                        let read_timestamp = F::from(polynomials.read_timestamps[i][j]);
                        polynomials.read_cts_read_timestamp[i][j] * gamma_squared
                            + read_timestamp * gamma
                            + read_timestamp
                            - tau
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
                            F::from(j as u64 - polynomials.read_timestamps[i][j]);
                        polynomials.read_cts_global_minus_read[i][j] * gamma_squared
                            + global_minus_read * gamma
                            + global_minus_read
                            - tau
                    })
                    .collect();
                let write_fingeprints_1 = read_fingerprints_1
                    .par_iter()
                    .map(|read_fingerprint| *read_fingerprint + gamma_squared)
                    .collect();

                [
                    DensePolynomial::new(read_fingerprints_0),
                    DensePolynomial::new(write_fingeprints_0),
                    DensePolynomial::new(read_fingerprints_1),
                    DensePolynomial::new(write_fingeprints_1),
                ]
            })
            .collect();

        let init_fingerprints = (0..M)
            .into_par_iter()
            .map(|i| {
                let index = F::from(i as u64);
                // 0 * gamma^2 +
                index * gamma + index - tau
            })
            .collect();
        let init_leaves = DensePolynomial::new(init_fingerprints);

        let final_leaves: Vec<DensePolynomial<F>> = (0..MEMORY_OPS_PER_INSTRUCTION)
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

                [
                    DensePolynomial::new(final_fingerprints_0),
                    DensePolynomial::new(final_fingerprints_1),
                ]
            })
            .collect();

        let mut init_final_leaves = final_leaves;
        init_final_leaves.push(init_leaves);

        (read_write_leaves, init_final_leaves)
    }

    fn protocol_name() -> &'static [u8] {
        b"Timestamp validity proof memory checking"
    }
}

impl<F, G> MemoryCheckingVerifier<F, G, RangeCheckPolynomials<F, G>> for RangeCheckPolynomials<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    fn verify_memory_checking(
        mut _proof: MemoryCheckingProof<
            G,
            RangeCheckPolynomials<F, G>,
            Self::ReadWriteOpenings,
            Self::InitFinalOpenings,
        >,
        _commitments: &RangeCheckCommitment<G>,
        _transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        unimplemented!("Use TimestampValidityProof::verify instead");
    }

    fn read_tuples(openings: &Self::ReadWriteOpenings) -> Vec<Self::MemoryTuple> {
        let t_read_openings = openings.memory_poly_openings.t_read_opening;

        (0..MEMORY_OPS_PER_INSTRUCTION)
            .into_iter()
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

    fn write_tuples(openings: &Self::ReadWriteOpenings) -> Vec<Self::MemoryTuple> {
        let t_read_openings = openings.memory_poly_openings.t_read_opening;

        (0..MEMORY_OPS_PER_INSTRUCTION)
            .into_iter()
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

    fn init_tuples(openings: &Self::InitFinalOpenings) -> Vec<Self::MemoryTuple> {
        vec![(
            openings.identity_poly_opening.unwrap(),
            openings.identity_poly_opening.unwrap(),
            F::zero(),
        )]
    }

    fn final_tuples(openings: &Self::InitFinalOpenings) -> Vec<Self::MemoryTuple> {
        (0..MEMORY_OPS_PER_INSTRUCTION)
            .into_iter()
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

pub struct TimestampValidityProof<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    multiset_hashes: MultisetHashes<F>,
    openings: RangeCheckOpenings<F, G>,
    opening_proof: RangeCheckOpeningProof<G>,
    batched_grand_product: BatchedGrandProductArgument<F>,
    pub commitment: RangeCheckCommitment<G>,
}

impl<F, G> TimestampValidityProof<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    #[tracing::instrument(skip_all, name = "TimestampValidityProof::prove")]
    pub fn prove(
        read_timestamps: [Vec<u64>; MEMORY_OPS_PER_INSTRUCTION],
        memory_polynomials: &ReadWriteMemory<F, G>,
        batched_memory_polynomials: &BatchedMemoryPolynomials<F>,
        memory_commitment: &MemoryCommitment<G>,
        transcript: &mut Transcript,
        random_tape: &mut RandomTape<G>,
    ) -> Self {
        let range_check_polys: RangeCheckPolynomials<F, G> =
            RangeCheckPolynomials::new(read_timestamps);
        let batched_range_check_polys = range_check_polys.batch();
        let range_check_commitment = RangeCheckPolynomials::commit(&batched_range_check_polys);
        let (batched_grand_product, multiset_hashes, r_grand_product) =
            TimestampValidityProof::prove_grand_products(&range_check_polys, transcript);

        let chis = EqPolynomial::new(r_grand_product.to_vec()).evals();
        let mut openings = range_check_polys
            .read_cts_read_timestamp
            .par_iter()
            .chain(range_check_polys.read_cts_global_minus_read.par_iter())
            .chain(range_check_polys.final_cts_read_timestamp.par_iter())
            .chain(range_check_polys.final_cts_global_minus_read.par_iter())
            .chain(memory_polynomials.a_read_write.par_iter())
            .chain(memory_polynomials.v_read.par_iter())
            .chain(memory_polynomials.v_write.par_iter())
            .chain(memory_polynomials.t_read.par_iter())
            .chain(memory_polynomials.t_write.par_iter())
            .map(|poly| poly.evaluate_at_chi(&chis))
            .collect::<Vec<F>>()
            .into_iter();

        let read_cts_read_timestamp: [F; MEMORY_OPS_PER_INSTRUCTION] =
            openings.next_chunk().unwrap();
        let read_cts_global_minus_read = openings.next_chunk().unwrap();
        let final_cts_read_timestamp = openings.next_chunk().unwrap();
        let final_cts_global_minus_read = openings.next_chunk().unwrap();
        let memory_a_read_write = openings.next_chunk().unwrap();
        let memory_v_read = openings.next_chunk().unwrap();
        let memory_v_write = openings.next_chunk().unwrap();
        let memory_t_read = openings.next_chunk().unwrap();
        let memory_t_write = openings.next_chunk().unwrap();

        let memory_poly_openings = MemoryReadWriteOpenings {
            a_read_write_opening: memory_a_read_write,
            v_read_opening: memory_v_read,
            v_write_opening: memory_v_write,
            t_read_opening: memory_t_read,
            t_write_opening: memory_t_write,
        };
        let openings = RangeCheckOpenings {
            read_cts_read_timestamp,
            read_cts_global_minus_read,
            final_cts_read_timestamp,
            final_cts_global_minus_read,
            memory_poly_openings,
            identity_poly_opening: None,
        };

        let mut opening_proof = RangeCheckOpenings::prove_openings(
            &batched_range_check_polys,
            &range_check_commitment,
            &r_grand_product,
            &openings,
            transcript,
            random_tape,
        );
        opening_proof.memory_poly_opening_proof = Some(MemoryReadWriteOpenings::prove_openings(
            batched_memory_polynomials,
            memory_commitment,
            &r_grand_product,
            &openings.memory_poly_openings,
            transcript,
            random_tape,
        ));

        Self {
            multiset_hashes,
            openings,
            opening_proof,
            batched_grand_product,
            commitment: range_check_commitment,
        }
    }

    fn prove_grand_products(
        polynomials: &RangeCheckPolynomials<F, G>,
        transcript: &mut Transcript,
    ) -> (BatchedGrandProductArgument<F>, MultisetHashes<F>, Vec<F>) {
        // Fiat-Shamir randomness for multiset hashes
        let gamma: F = <Transcript as ProofTranscript<G>>::challenge_scalar(
            transcript,
            b"Memory checking gamma",
        );
        let tau: F = <Transcript as ProofTranscript<G>>::challenge_scalar(
            transcript,
            b"Memory checking tau",
        );

        <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

        // fka "ProductLayerProof"
        let (read_write_leaves, init_final_leaves) =
            polynomials.compute_leaves(polynomials, &gamma, &tau);
        let leaves = [
            &init_final_leaves[0], // init
            &read_write_leaves[0], // read
            &read_write_leaves[1], // read
            &read_write_leaves[2], // write
            &read_write_leaves[3], // write
            &init_final_leaves[1], // final
            &init_final_leaves[2], // final
        ];

        let _span = trace_span!("TimestampValidityProof: construct grand product circuits");
        let _enter = _span.enter();

        let circuits: Vec<GrandProductCircuit<F>> = leaves
            .into_par_iter()
            .map(|leaves_poly| GrandProductCircuit::new(leaves_poly))
            .collect();

        drop(_enter);
        drop(_span);

        let hashes: Vec<F> = circuits
            .par_iter()
            .map(|circuit| circuit.evaluate())
            .collect();
        let multiset_hashes = MultisetHashes {
            init_hashes: vec![hashes[0]],
            read_hashes: vec![hashes[1], hashes[2]],
            write_hashes: vec![hashes[3], hashes[4]],
            final_hashes: vec![hashes[5], hashes[6]],
        };
        RangeCheckPolynomials::<F, G>::check_multiset_equality(&multiset_hashes);
        multiset_hashes.append_to_transcript::<G>(transcript);

        let batched_circuit = BatchedGrandProductCircuit::new_batch(circuits);

        let _span = trace_span!("TimestampValidityProof: prove grand products");
        let _enter = _span.enter();
        let (batched_grand_product, r_grand_product) =
            BatchedGrandProductArgument::prove::<G>(batched_circuit, transcript);
        drop(_enter);
        drop(_span);

        (batched_grand_product, multiset_hashes, r_grand_product)
    }

    pub fn verify(
        &mut self,
        memory_commitment: &MemoryCommitment<G>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        // Fiat-Shamir randomness for multiset hashes
        let gamma: F = <Transcript as ProofTranscript<G>>::challenge_scalar(
            transcript,
            b"Memory checking gamma",
        );
        let tau: F = <Transcript as ProofTranscript<G>>::challenge_scalar(
            transcript,
            b"Memory checking tau",
        );

        <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

        // Multiset equality checks
        RangeCheckPolynomials::<F, G>::check_multiset_equality(&self.multiset_hashes);
        self.multiset_hashes.append_to_transcript::<G>(transcript);

        let concatenated_hashes = [
            self.multiset_hashes.init_hashes.clone(),
            self.multiset_hashes.read_hashes.clone(),
            self.multiset_hashes.write_hashes.clone(),
            self.multiset_hashes.final_hashes.clone(),
        ]
        .concat();

        let (grand_product_claims, r_grand_product) = self
            .batched_grand_product
            .verify::<G, Transcript>(&concatenated_hashes, transcript);

        self.openings.verify_openings(
            &self.opening_proof,
            &self.commitment,
            &r_grand_product,
            transcript,
        )?;
        self.openings.memory_poly_openings.verify_openings(
            &self
                .opening_proof
                .memory_poly_opening_proof
                .as_ref()
                .unwrap(),
            memory_commitment,
            &r_grand_product,
            transcript,
        )?;

        self.openings.compute_verifier_openings(&r_grand_product);

        debug_assert_eq!(grand_product_claims.len(), 7);
        let read_fingerprints: Vec<_> = RangeCheckPolynomials::read_tuples(&self.openings)
            .iter()
            .map(|tuple| RangeCheckPolynomials::<F, G>::fingerprint(tuple, &gamma, &tau))
            .collect();
        let write_fingerprints: Vec<_> = RangeCheckPolynomials::write_tuples(&self.openings)
            .iter()
            .map(|tuple| RangeCheckPolynomials::<F, G>::fingerprint(tuple, &gamma, &tau))
            .collect();
        let init_fingerprints: Vec<_> = RangeCheckPolynomials::init_tuples(&self.openings)
            .iter()
            .map(|tuple| RangeCheckPolynomials::<F, G>::fingerprint(tuple, &gamma, &tau))
            .collect();
        let final_fingerprints: Vec<_> = RangeCheckPolynomials::final_tuples(&self.openings)
            .iter()
            .map(|tuple| RangeCheckPolynomials::<F, G>::fingerprint(tuple, &gamma, &tau))
            .collect();
        assert_eq!(
            init_fingerprints.len()
                + read_fingerprints.len()
                + write_fingerprints.len()
                + final_fingerprints.len(),
            grand_product_claims.len()
        );
        for (claim, fingerprint) in zip(
            grand_product_claims,
            [
                init_fingerprints,
                read_fingerprints,
                write_fingerprints,
                final_fingerprints,
            ]
            .concat(),
        ) {
            assert_eq!(claim, fingerprint);
        }
        Ok(())
    }

    fn protocol_name() -> &'static [u8] {
        b"Timestamp validity proof memory checking"
    }
}
