use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use merlin::Transcript;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::marker::PhantomData;

use crate::{
    lasso::memory_checking::{
        MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier, MultisetHashes,
    },
    poly::{
        dense_mlpoly::{DensePolynomial, PolyCommitmentGens},
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        structured_poly::{BatchablePolynomials, StructuredOpeningProof},
    },
    subprotocols::{
        combined_table_proof::{CombinedTableCommitment, CombinedTableEvalProof},
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
    pub read_timestamps: Vec<u64>,
    pub read_cts_read_timestamp: DensePolynomial<F>,
    pub read_cts_global_minus_read: DensePolynomial<F>,
    pub final_cts_read_timestamp: DensePolynomial<F>,
    pub final_cts_global_minus_read: DensePolynomial<F>,
}

impl<F, G> RangeCheckPolynomials<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    #[tracing::instrument(skip_all, name = "RangeCheckPolynomials::new")]
    pub fn new(read_timestamps: Vec<u64>) -> Self {
        let M = read_timestamps.len();
        let mut read_cts_read_timestamp = vec![0usize; M];
        let mut final_cts_read_timestamp = vec![0usize; M];
        let mut read_cts_global_minus_read = vec![0usize; M];
        let mut final_cts_global_minus_read = vec![0usize; M];
        for (i, read_timestamp) in read_timestamps.iter().enumerate() {
            read_cts_read_timestamp[i] = final_cts_read_timestamp[*read_timestamp as usize];
            final_cts_read_timestamp[*read_timestamp as usize] += 1;
            let lookup_index = i - *read_timestamp as usize;
            read_cts_global_minus_read[i] = final_cts_global_minus_read[lookup_index];
            final_cts_global_minus_read[lookup_index] += 1;
        }

        Self {
            _group: PhantomData,
            read_timestamps,
            read_cts_read_timestamp: DensePolynomial::from_usize(&read_cts_read_timestamp),
            read_cts_global_minus_read: DensePolynomial::from_usize(&read_cts_global_minus_read),
            final_cts_read_timestamp: DensePolynomial::from_usize(&final_cts_read_timestamp),
            final_cts_global_minus_read: DensePolynomial::from_usize(&final_cts_global_minus_read),
        }
    }
}

pub struct RangeCheckCommitment<G: CurveGroup> {
    generators: RangeCheckCommitmentGenerators<G>,
    pub commitment: CombinedTableCommitment<G>,
}

/// Container for generators for polynomial commitments. These preallocate memory
/// and allow commitments to `DensePolynomials`.
pub type RangeCheckCommitmentGenerators<G> = PolyCommitmentGens<G>;
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
        DensePolynomial::merge(&vec![
            &self.read_cts_read_timestamp,
            &self.read_cts_global_minus_read,
            &self.final_cts_read_timestamp,
            &self.final_cts_global_minus_read,
        ])
    }

    #[tracing::instrument(skip_all, name = "RangeCheckPolynomials::commit")]
    fn commit(batched_polys: &Self::BatchedPolynomials) -> Self::Commitment {
        let (generators, commitment) =
            batched_polys.combined_commit(b"BatchedRangeCheckPolynomials");

        Self::Commitment {
            generators,
            commitment,
        }
    }
}

pub struct RangeCheckOpenings<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    openings: [F; 4],
    opening_proof: CombinedTableEvalProof<G>,
    memory_poly_openings: Option<MemoryReadWriteOpenings<F, G>>,
    identity_poly_opening: Option<F>,
}

impl<F, G> StructuredOpeningProof<F, G, RangeCheckPolynomials<F, G>> for RangeCheckOpenings<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type Openings = [F; 4];

    #[tracing::instrument(skip_all, name = "RangeCheckReadWriteOpenings::open")]
    fn open(polynomials: &RangeCheckPolynomials<F, G>, opening_point: &Vec<F>) -> Self::Openings {
        let chis = EqPolynomial::new(opening_point.to_vec()).evals();
        [
            &polynomials.read_cts_read_timestamp,
            &polynomials.read_cts_global_minus_read,
            &polynomials.final_cts_read_timestamp,
            &polynomials.final_cts_global_minus_read,
        ]
        .par_iter()
        .map(|poly| poly.evaluate_at_chi(&chis))
        .collect::<Vec<F>>()
        .try_into()
        .unwrap()
    }

    #[tracing::instrument(skip_all, name = "RangeCheckReadWriteOpenings::prove_openings")]
    fn prove_openings(
        polynomials: &BatchedRangeCheckPolynomials<F>,
        commitment: &RangeCheckCommitment<G>,
        opening_point: &Vec<F>,
        openings: [F; 4],
        transcript: &mut Transcript,
        random_tape: &mut RandomTape<G>,
    ) -> Self {
        let opening_proof = CombinedTableEvalProof::prove(
            &polynomials,
            &openings,
            opening_point,
            &commitment.generators,
            transcript,
            random_tape,
        );

        Self {
            openings,
            opening_proof,
            memory_poly_openings: None,
            identity_poly_opening: None,
        }
    }

    fn compute_verifier_openings(&mut self, opening_point: &Vec<F>) {
        self.identity_poly_opening =
            Some(IdentityPolynomial::new(opening_point.len()).evaluate(opening_point));
    }

    fn verify_openings(
        &self,
        commitment: &RangeCheckCommitment<G>,
        opening_point: &Vec<F>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        self.opening_proof.verify(
            opening_point,
            &self.openings,
            &commitment.generators,
            &commitment.commitment,
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

    #[tracing::instrument(skip_all, name = "RangeCheckPolynomials::read_leaves")]
    fn read_leaves(
        &self,
        _polynomials: &RangeCheckPolynomials<F, G>,
        _gamma: &F,
        _tau: &F,
    ) -> Vec<DensePolynomial<F>> {
        unimplemented!("Leaves are constructed in TimestampValidityProof::prove")
    }

    #[tracing::instrument(skip_all, name = "RangeCheckPolynomials::write_leaves")]
    fn write_leaves(
        &self,
        _polynomials: &RangeCheckPolynomials<F, G>,
        _gamma: &F,
        _tau: &F,
    ) -> Vec<DensePolynomial<F>> {
        unimplemented!("Leaves are constructed in TimestampValidityProof::prove")
    }

    #[tracing::instrument(skip_all, name = "RangeCheckPolynomials::init_leaves")]
    fn init_leaves(
        &self,
        _polynomials: &RangeCheckPolynomials<F, G>,
        _gamma: &F,
        _tau: &F,
    ) -> Vec<DensePolynomial<F>> {
        unimplemented!("Leaves are constructed in TimestampValidityProof::prove")
    }

    #[tracing::instrument(skip_all, name = "RangeCheckPolynomials::final_leaves")]
    fn final_leaves(
        &self,
        _polynomials: &RangeCheckPolynomials<F, G>,
        _gamma: &F,
        _tau: &F,
    ) -> Vec<DensePolynomial<F>> {
        unimplemented!("Leaves are constructed in TimestampValidityProof::prove")
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
        let t_read_opening = openings
            .memory_poly_openings
            .as_ref()
            .unwrap()
            .t_read_opening;
        vec![
            (
                t_read_opening,
                t_read_opening,
                openings.openings[0],
            ),
            (
                openings.identity_poly_opening.unwrap() - t_read_opening,
                openings.identity_poly_opening.unwrap() - t_read_opening,
                openings.openings[1],
            ),
        ]
    }

    fn write_tuples(openings: &Self::ReadWriteOpenings) -> Vec<Self::MemoryTuple> {
        let t_read_opening = openings
            .memory_poly_openings
            .as_ref()
            .unwrap()
            .t_read_opening;
        vec![
            (
                t_read_opening,
                t_read_opening,
                openings.openings[0] + F::one(),
            ),
            (
                openings.identity_poly_opening.unwrap() - t_read_opening,
                openings.identity_poly_opening.unwrap() - t_read_opening,
                openings.openings[1] + F::one(),
            ),
        ]
    }

    fn init_tuples(openings: &Self::InitFinalOpenings) -> Vec<Self::MemoryTuple> {
        vec![
            (
                openings.identity_poly_opening.unwrap(),
                openings.identity_poly_opening.unwrap(),
                F::zero(),
            ),
            (
                openings.identity_poly_opening.unwrap(),
                openings.identity_poly_opening.unwrap(),
                F::zero(),
            ),
        ]
    }

    fn final_tuples(openings: &Self::InitFinalOpenings) -> Vec<Self::MemoryTuple> {
        vec![
            (
                openings.identity_poly_opening.unwrap(),
                openings.identity_poly_opening.unwrap(),
                openings.openings[2],
            ),
            (
                openings.identity_poly_opening.unwrap(),
                openings.identity_poly_opening.unwrap(),
                openings.openings[3],
            ),
        ]
    }
}

pub struct TimestampValidityProof<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    read_timestamp_multiset_hashes: MultisetHashes<F>,
    global_minus_read_multiset_hashes: MultisetHashes<F>,
    opening_proof: RangeCheckOpenings<F, G>,
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
        read_timestamps: Vec<u64>,
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
        let (
            batched_grand_product,
            read_timestamp_multiset_hashes,
            global_minus_read_multiset_hashes,
            r_grand_product,
        ) = TimestampValidityProof::prove_grand_products(&range_check_polys, transcript);

        let chis = EqPolynomial::new(r_grand_product.to_vec()).evals();
        let openings: Vec<F> = [
            &range_check_polys.read_cts_read_timestamp,
            &range_check_polys.read_cts_global_minus_read,
            &range_check_polys.final_cts_read_timestamp,
            &range_check_polys.final_cts_global_minus_read,
            &memory_polynomials.a_read_write,
            &memory_polynomials.v_read,
            &memory_polynomials.v_write,
            &memory_polynomials.t_read,
            &memory_polynomials.t_write,
        ]
        .par_iter()
        .map(|poly| poly.evaluate_at_chi(&chis))
        .collect();

        let mut opening_proof = RangeCheckOpenings::prove_openings(
            &batched_range_check_polys,
            &range_check_commitment,
            &r_grand_product,
            [openings[0], openings[1], openings[2], openings[3]],
            transcript,
            random_tape,
        );
        opening_proof.memory_poly_openings = Some(MemoryReadWriteOpenings::prove_openings(
            batched_memory_polynomials,
            memory_commitment,
            &r_grand_product,
            [
                openings[4],
                openings[5],
                openings[6],
                openings[7],
                openings[8],
            ],
            transcript,
            random_tape,
        ));

        Self {
            read_timestamp_multiset_hashes,
            global_minus_read_multiset_hashes,
            opening_proof,
            batched_grand_product,
            commitment: range_check_commitment,
        }
    }

    fn prove_grand_products(
        polynomials: &RangeCheckPolynomials<F, G>,
        transcript: &mut Transcript,
    ) -> (
        BatchedGrandProductArgument<F>,
        MultisetHashes<F>,
        MultisetHashes<F>,
        Vec<F>,
    ) {
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
        let _span = tracing::span!(
            tracing::Level::TRACE,
            "TimestampValiditProof: compute leaves"
        );
        let _enter = _span.enter();

        let M = polynomials.read_timestamps.len();

        let gamma_squared = gamma.square();

        let (init_leaves, read_leaves) = rayon::join(
            || {
                let init_fingerprints: Vec<F> = (0..M)
                    .into_par_iter()
                    .map(|i| {
                        // 0 * gamma^2 +
                        F::from(i as u64) * gamma + F::from(i as u64) - tau
                    })
                    .collect();
                DensePolynomial::new(init_fingerprints)
            },
            || {
                let read_fingerprints = rayon::join(
                    || {
                        (0..M)
                            .into_par_iter()
                            .map(|i| {
                                polynomials.read_cts_read_timestamp[i] * gamma_squared
                                    + F::from(polynomials.read_timestamps[i]) * gamma
                                    + F::from(polynomials.read_timestamps[i])
                                    - tau
                            })
                            .collect()
                    },
                    || {
                        (0..M)
                            .into_par_iter()
                            .map(|i| {
                                polynomials.read_cts_global_minus_read[i] * gamma_squared
                                    + (F::from(i as u64 - polynomials.read_timestamps[i])) * gamma
                                    + (F::from(i as u64 - polynomials.read_timestamps[i]))
                                    - tau
                            })
                            .collect()
                    },
                );
                vec![
                    DensePolynomial::new(read_fingerprints.0),
                    DensePolynomial::new(read_fingerprints.1),
                ]
            },
        );
        let (final_leaves, write_leaves) = rayon::join(
            || {
                let final_fingerprints = rayon::join(
                    || {
                        (0..M)
                            .into_par_iter()
                            .map(|i| {
                                mul_0_1_optimized(
                                    &polynomials.final_cts_read_timestamp[i],
                                    &gamma_squared,
                                ) + init_leaves[i]
                            })
                            .collect()
                    },
                    || {
                        (0..M)
                            .into_par_iter()
                            .map(|i| {
                                mul_0_1_optimized(
                                    &polynomials.final_cts_global_minus_read[i],
                                    &gamma_squared,
                                ) + init_leaves[i]
                            })
                            .collect()
                    },
                );
                vec![
                    DensePolynomial::new(final_fingerprints.0),
                    DensePolynomial::new(final_fingerprints.1),
                ]
            },
            || {
                let write_fingerprints = rayon::join(
                    || {
                        (0..M)
                            .into_par_iter()
                            .map(|i| read_leaves[0].evals_ref()[i] + gamma_squared)
                            .collect()
                    },
                    || {
                        (0..M)
                            .into_par_iter()
                            .map(|i| read_leaves[1].evals_ref()[i] + gamma_squared)
                            .collect()
                    },
                );
                vec![
                    DensePolynomial::new(write_fingerprints.0),
                    DensePolynomial::new(write_fingerprints.1),
                ]
            },
        );
        drop(_enter);
        drop(_span);

        let leaves = [
            &init_leaves,
            &read_leaves[0],
            &write_leaves[0],
            &final_leaves[0],
            &read_leaves[1],
            &write_leaves[1],
            &final_leaves[1],
        ];

        let _span = tracing::span!(
            tracing::Level::TRACE,
            "TimestampValiditProof: construct grand product circuits"
        );
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

        let read_timestamp_hashes = MultisetHashes {
            hash_init: hashes[0],
            hash_read: hashes[1],
            hash_write: hashes[2],
            hash_final: hashes[3],
        };
        debug_assert_eq!(
            read_timestamp_hashes.hash_init * read_timestamp_hashes.hash_write,
            read_timestamp_hashes.hash_final * read_timestamp_hashes.hash_read,
            "Multiset hashes don't match"
        );
        read_timestamp_hashes.append_to_transcript::<G>(transcript);

        let global_minus_read_hashes = MultisetHashes {
            hash_init: hashes[0],
            hash_read: hashes[4],
            hash_write: hashes[5],
            hash_final: hashes[6],
        };
        debug_assert_eq!(
            global_minus_read_hashes.hash_init * global_minus_read_hashes.hash_write,
            global_minus_read_hashes.hash_final * global_minus_read_hashes.hash_read,
            "Multiset hashes don't match"
        );
        global_minus_read_hashes.append_to_transcript::<G>(transcript);

        let batched_circuit = BatchedGrandProductCircuit::new_batch(circuits);

        let _span = tracing::span!(
            tracing::Level::TRACE,
            "TimestampValiditProof: prove grand products"
        );
        let _enter = _span.enter();
        let (batched_grand_product, r_grand_product) =
            BatchedGrandProductArgument::prove::<G>(batched_circuit, transcript);
        drop(_enter);
        drop(_span);

        (
            batched_grand_product,
            read_timestamp_hashes,
            global_minus_read_hashes,
            r_grand_product,
        )
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
        assert_eq!(
            self.read_timestamp_multiset_hashes.hash_init
                * self.read_timestamp_multiset_hashes.hash_write,
            self.read_timestamp_multiset_hashes.hash_read
                * self.read_timestamp_multiset_hashes.hash_final
        );
        self.read_timestamp_multiset_hashes
            .append_to_transcript::<G>(transcript);

        assert_eq!(
            self.global_minus_read_multiset_hashes.hash_init
                * self.global_minus_read_multiset_hashes.hash_write,
            self.global_minus_read_multiset_hashes.hash_read
                * self.global_minus_read_multiset_hashes.hash_final
        );
        self.global_minus_read_multiset_hashes
            .append_to_transcript::<G>(transcript);

        assert_eq!(
            self.read_timestamp_multiset_hashes.hash_init,
            self.global_minus_read_multiset_hashes.hash_init
        );

        let interleaved_hashes = vec![
            self.read_timestamp_multiset_hashes.hash_init,
            self.read_timestamp_multiset_hashes.hash_read,
            self.read_timestamp_multiset_hashes.hash_write,
            self.read_timestamp_multiset_hashes.hash_final,
            self.global_minus_read_multiset_hashes.hash_read,
            self.global_minus_read_multiset_hashes.hash_write,
            self.global_minus_read_multiset_hashes.hash_final,
        ];

        let (grand_product_claims, r_grand_product) = self
            .batched_grand_product
            .verify::<G, Transcript>(&interleaved_hashes, transcript);

        self.opening_proof
            .verify_openings(&self.commitment, &r_grand_product, transcript)?;
        self.opening_proof
            .memory_poly_openings
            .as_ref()
            .unwrap()
            .verify_openings(memory_commitment, &r_grand_product, transcript)?;

        self.opening_proof
            .compute_verifier_openings(&r_grand_product);

        debug_assert_eq!(grand_product_claims.len(), 7);
        let grand_product_claims: Vec<MultisetHashes<F>> = vec![
            MultisetHashes {
                hash_init: grand_product_claims[0],
                hash_read: grand_product_claims[1],
                hash_write: grand_product_claims[2],
                hash_final: grand_product_claims[3],
            },
            MultisetHashes {
                hash_init: grand_product_claims[0],
                hash_read: grand_product_claims[4],
                hash_write: grand_product_claims[5],
                hash_final: grand_product_claims[6],
            },
        ];
        RangeCheckPolynomials::check_fingerprints(
            grand_product_claims,
            &self.opening_proof,
            &self.opening_proof,
            &gamma,
            &tau,
        );

        Ok(())
    }

    fn protocol_name() -> &'static [u8] {
        b"Timestamp validity proof memory checking"
    }
}
