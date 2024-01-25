use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use merlin::Transcript;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
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
    M: usize,
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
            M,
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
    pub read_cts_commitment: CombinedTableCommitment<G>,
    pub final_cts_commitment: CombinedTableCommitment<G>,
}

/// Container for generators for polynomial commitments. These preallocate memory
/// and allow commitments to `DensePolynomials`.
pub struct RangeCheckCommitmentGenerators<G: CurveGroup> {
    pub read_commitment_gens: PolyCommitmentGens<G>,
    pub final_commitment_gens: PolyCommitmentGens<G>,
}

pub struct BatchedRangeCheckPolynomials<F: PrimeField> {
    pub batched_read_cts: DensePolynomial<F>,
    pub batched_final_cts: DensePolynomial<F>,
}

impl<F, G> BatchablePolynomials for RangeCheckPolynomials<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type BatchedPolynomials = BatchedRangeCheckPolynomials<F>;
    type Commitment = RangeCheckCommitment<G>;

    #[tracing::instrument(skip_all, name = "RangeCheckPolynomials::batch")]
    fn batch(&self) -> Self::BatchedPolynomials {
        let (batched_read_cts, batched_final_cts) = rayon::join(
            || {
                DensePolynomial::merge(&vec![
                    &self.read_cts_read_timestamp,
                    &self.read_cts_global_minus_read,
                ])
            },
            || {
                DensePolynomial::merge(&vec![
                    &self.final_cts_read_timestamp,
                    &self.final_cts_global_minus_read,
                ])
            },
        );
        BatchedRangeCheckPolynomials {
            batched_read_cts,
            batched_final_cts,
        }
    }

    #[tracing::instrument(skip_all, name = "RangeCheckPolynomials::commit")]
    fn commit(batched_polys: &Self::BatchedPolynomials) -> Self::Commitment {
        let (read_commitment_gens, read_cts_commitment) = batched_polys
            .batched_read_cts
            .combined_commit(b"BatchedRangeCheckPolynomials.batched_read_cts");
        let (final_commitment_gens, final_cts_commitment) = batched_polys
            .batched_final_cts
            .combined_commit(b"BatchedRangeCheckPolynomials.batched_final_cts");

        let generators = RangeCheckCommitmentGenerators {
            read_commitment_gens,
            final_commitment_gens,
        };

        Self::Commitment {
            generators,
            read_cts_commitment,
            final_cts_commitment,
        }
    }
}

pub struct RangeCheckReadWriteOpenings<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    read_cts_openings: [F; 2],
    read_cts_opening_proof: CombinedTableEvalProof<G>,
    memory_poly_openings: Option<MemoryReadWriteOpenings<F, G>>,
    identity_poly_opening: Option<F>,
}

impl<F, G> StructuredOpeningProof<F, G, RangeCheckPolynomials<F, G>>
    for RangeCheckReadWriteOpenings<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type Openings = [F; 2];

    #[tracing::instrument(skip_all, name = "RangeCheckReadWriteOpenings::open")]
    fn open(polynomials: &RangeCheckPolynomials<F, G>, opening_point: &Vec<F>) -> Self::Openings {
        let chis = EqPolynomial::new(opening_point.to_vec()).evals();
        [
            &polynomials.read_cts_read_timestamp,
            &polynomials.read_cts_global_minus_read,
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
        openings: [F; 2],
        transcript: &mut Transcript,
        random_tape: &mut RandomTape<G>,
    ) -> Self {
        let read_cts_opening_proof = CombinedTableEvalProof::prove(
            &polynomials.batched_read_cts,
            &openings,
            opening_point,
            &commitment.generators.read_commitment_gens,
            transcript,
            random_tape,
        );

        Self {
            read_cts_openings: [openings[0], openings[1]],
            read_cts_opening_proof,
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
        self.read_cts_opening_proof.verify(
            opening_point,
            &self.read_cts_openings,
            &commitment.generators.read_commitment_gens,
            &commitment.read_cts_commitment,
            transcript,
        )
    }
}

pub struct RangeCheckFinalOpenings<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    identity_poly_opening: Option<F>,
    final_cts_openings: [F; 2],
    final_cts_opening_proof: CombinedTableEvalProof<G>,
}

impl<F, G> StructuredOpeningProof<F, G, RangeCheckPolynomials<F, G>>
    for RangeCheckFinalOpenings<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type Openings = [F; 2];

    #[tracing::instrument(skip_all, name = "RangeCheckFinalOpenings::open")]
    fn open(polynomials: &RangeCheckPolynomials<F, G>, opening_point: &Vec<F>) -> Self::Openings {
        let chis = EqPolynomial::new(opening_point.to_vec()).evals();
        [
            &polynomials.final_cts_read_timestamp,
            &polynomials.final_cts_global_minus_read,
        ]
        .par_iter()
        .map(|poly| poly.evaluate_at_chi(&chis))
        .collect::<Vec<F>>()
        .try_into()
        .unwrap()
    }

    #[tracing::instrument(skip_all, name = "RangeCheckFinalOpenings::prove_openings")]
    fn prove_openings(
        polynomials: &BatchedRangeCheckPolynomials<F>,
        commitment: &RangeCheckCommitment<G>,
        opening_point: &Vec<F>,
        final_cts_openings: [F; 2],
        transcript: &mut Transcript,
        random_tape: &mut RandomTape<G>,
    ) -> Self {
        let final_cts_opening_proof = CombinedTableEvalProof::prove(
            &polynomials.batched_final_cts,
            &final_cts_openings,
            opening_point,
            &commitment.generators.final_commitment_gens,
            transcript,
            random_tape,
        );

        Self {
            identity_poly_opening: None,
            final_cts_openings,
            final_cts_opening_proof,
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
        self.final_cts_opening_proof.verify(
            opening_point,
            &self.final_cts_openings,
            &commitment.generators.final_commitment_gens,
            &commitment.final_cts_commitment,
            transcript,
        )
    }
}

impl<F, G> MemoryCheckingProver<F, G, RangeCheckPolynomials<F, G>> for RangeCheckPolynomials<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type ReadWriteOpenings = RangeCheckReadWriteOpenings<F, G>;
    type InitFinalOpenings = RangeCheckFinalOpenings<F, G>;

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
        polynomials: &RangeCheckPolynomials<F, G>,
        gamma: &F,
        tau: &F,
    ) -> Vec<DensePolynomial<F>> {
        let gamma_squared = gamma.square();
        let leaf_fingerprints = (
            (0..self.M)
                .into_par_iter()
                .map(|i| {
                    polynomials.read_cts_read_timestamp[i] * gamma_squared
                        + F::from(self.read_timestamps[i]) * gamma
                        + F::from(self.read_timestamps[i])
                        - *tau
                })
                .collect(),
            (0..self.M)
                .into_par_iter()
                .map(|i| {
                    polynomials.read_cts_global_minus_read[i] * gamma_squared
                        + (F::from(i as u64 - self.read_timestamps[i])) * gamma
                        + (F::from(i as u64 - self.read_timestamps[i]))
                        - *tau
                })
                .collect(),
        );
        vec![
            DensePolynomial::new(leaf_fingerprints.0),
            DensePolynomial::new(leaf_fingerprints.1),
        ]
    }

    #[tracing::instrument(skip_all, name = "RangeCheckPolynomials::write_leaves")]
    fn write_leaves(
        &self,
        polynomials: &RangeCheckPolynomials<F, G>,
        gamma: &F,
        tau: &F,
    ) -> Vec<DensePolynomial<F>> {
        let gamma_squared = gamma.square();
        let leaf_fingerprints = (
            (0..self.M)
                .into_par_iter()
                .map(|i| {
                    (polynomials.read_cts_read_timestamp[i] + F::one()) * gamma_squared
                        + F::from(self.read_timestamps[i]) * gamma
                        + F::from(self.read_timestamps[i])
                        - *tau
                })
                .collect(),
            (0..self.M)
                .into_par_iter()
                .map(|i| {
                    (polynomials.read_cts_global_minus_read[i] + F::one()) * gamma_squared
                        + (F::from(i as u64 - self.read_timestamps[i])) * gamma
                        + (F::from(i as u64 - self.read_timestamps[i]))
                        - *tau
                })
                .collect(),
        );
        vec![
            DensePolynomial::new(leaf_fingerprints.0),
            DensePolynomial::new(leaf_fingerprints.1),
        ]
    }

    #[tracing::instrument(skip_all, name = "RangeCheckPolynomials::init_leaves")]
    fn init_leaves(
        &self,
        _polynomials: &RangeCheckPolynomials<F, G>,
        gamma: &F,
        tau: &F,
    ) -> Vec<DensePolynomial<F>> {
        let leaf_fingerprints: Vec<_> = (0..self.M)
            .into_par_iter()
            .map(|i| {
                // 0 * gamma^2 +
                F::from(i as u64) * gamma + F::from(i as u64) - *tau
            })
            .collect();
        vec![
            DensePolynomial::new(leaf_fingerprints.clone()),
            DensePolynomial::new(leaf_fingerprints),
        ]
    }

    #[tracing::instrument(skip_all, name = "RangeCheckPolynomials::final_leaves")]
    fn final_leaves(
        &self,
        polynomials: &RangeCheckPolynomials<F, G>,
        gamma: &F,
        tau: &F,
    ) -> Vec<DensePolynomial<F>> {
        let gamma_squared = gamma.square();
        let leaf_fingerprints = (
            (0..self.M)
                .into_par_iter()
                .map(|i| {
                    mul_0_1_optimized(&polynomials.final_cts_read_timestamp[i], &gamma_squared)
                        + F::from(i as u64) * gamma
                        + F::from(i as u64)
                        - *tau
                })
                .collect(),
            (0..self.M)
                .into_par_iter()
                .map(|i| {
                    mul_0_1_optimized(&polynomials.final_cts_global_minus_read[i], &gamma_squared)
                        + F::from(i as u64) * gamma
                        + F::from(i as u64)
                        - *tau
                })
                .collect(),
        );
        vec![
            DensePolynomial::new(leaf_fingerprints.0),
            DensePolynomial::new(leaf_fingerprints.1),
        ]
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
                openings.read_cts_openings[0],
            ),
            (
                openings.identity_poly_opening.unwrap() - t_read_opening,
                openings.identity_poly_opening.unwrap() - t_read_opening,
                openings.read_cts_openings[1],
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
                openings.read_cts_openings[0] + F::one(),
            ),
            (
                openings.identity_poly_opening.unwrap() - t_read_opening,
                openings.identity_poly_opening.unwrap() - t_read_opening,
                openings.read_cts_openings[1] + F::one(),
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
                openings.final_cts_openings[0],
            ),
            (
                openings.identity_poly_opening.unwrap(),
                openings.identity_poly_opening.unwrap(),
                openings.final_cts_openings[1],
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
    read_write_openings: RangeCheckReadWriteOpenings<F, G>,
    init_final_openings: RangeCheckFinalOpenings<F, G>,
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

        // TODO(moodlezoup): parallelize openings
        let mut read_write_openings = RangeCheckReadWriteOpenings::prove_openings(
            &batched_range_check_polys,
            &range_check_commitment,
            &r_grand_product,
            [openings[0], openings[1]],
            transcript,
            random_tape,
        );
        read_write_openings.memory_poly_openings = Some(MemoryReadWriteOpenings::prove_openings(
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
        let init_final_openings = RangeCheckFinalOpenings::prove_openings(
            &batched_range_check_polys,
            &range_check_commitment,
            &r_grand_product,
            [openings[2], openings[3]],
            transcript,
            random_tape,
        );

        Self {
            read_timestamp_multiset_hashes,
            global_minus_read_multiset_hashes,
            read_write_openings,
            init_final_openings,
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
        let ((read_leaves, write_leaves), (init_leaves, final_leaves)) = rayon::join(
            || {
                rayon::join(
                    || polynomials.read_leaves(polynomials, &gamma, &tau),
                    || polynomials.write_leaves(polynomials, &gamma, &tau),
                )
            },
            || {
                rayon::join(
                    || polynomials.init_leaves(polynomials, &gamma, &tau),
                    || polynomials.final_leaves(polynomials, &gamma, &tau),
                )
            },
        );

        let leaves = vec![
            &read_leaves[0],
            &write_leaves[0],
            &init_leaves[0],
            &final_leaves[0],
            &read_leaves[1],
            &write_leaves[1],
            &init_leaves[1],
            &final_leaves[1],
        ];

        let circuits: Vec<GrandProductCircuit<F>> = leaves
            .into_par_iter()
            .map(|leaves_poly| GrandProductCircuit::new(leaves_poly))
            .collect();

        let hashes: Vec<F> = circuits
            .par_iter()
            .map(|circuit| circuit.evaluate())
            .collect();

        let read_timestamp_hashes = MultisetHashes {
            hash_read: hashes[0],
            hash_write: hashes[1],
            hash_init: hashes[2],
            hash_final: hashes[3],
        };
        debug_assert_eq!(
            read_timestamp_hashes.hash_init * read_timestamp_hashes.hash_write,
            read_timestamp_hashes.hash_final * read_timestamp_hashes.hash_read,
            "Multiset hashes don't match"
        );
        read_timestamp_hashes.append_to_transcript::<G>(transcript);

        let global_minus_read_hashes = MultisetHashes {
            hash_read: hashes[4],
            hash_write: hashes[5],
            hash_init: hashes[6],
            hash_final: hashes[7],
        };
        debug_assert_eq!(
            global_minus_read_hashes.hash_init * global_minus_read_hashes.hash_write,
            global_minus_read_hashes.hash_final * global_minus_read_hashes.hash_read,
            "Multiset hashes don't match"
        );
        global_minus_read_hashes.append_to_transcript::<G>(transcript);
        let batched_circuit = BatchedGrandProductCircuit::new_batch(circuits);

        let (batched_grand_product, r_grand_product) =
            BatchedGrandProductArgument::prove::<G>(batched_circuit, transcript);
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

        let interleaved_hashes = vec![
            self.read_timestamp_multiset_hashes.hash_read,
            self.read_timestamp_multiset_hashes.hash_write,
            self.read_timestamp_multiset_hashes.hash_init,
            self.read_timestamp_multiset_hashes.hash_final,
            self.global_minus_read_multiset_hashes.hash_read,
            self.global_minus_read_multiset_hashes.hash_write,
            self.global_minus_read_multiset_hashes.hash_init,
            self.global_minus_read_multiset_hashes.hash_final,
        ];

        let (grand_product_claims, r_grand_product) = self
            .batched_grand_product
            .verify::<G, Transcript>(&interleaved_hashes, transcript);

        self.read_write_openings
            .verify_openings(&self.commitment, &r_grand_product, transcript)?;
        self.read_write_openings
            .memory_poly_openings
            .as_ref()
            .unwrap()
            .verify_openings(memory_commitment, &r_grand_product, transcript)?;
        self.init_final_openings
            .verify_openings(&self.commitment, &r_grand_product, transcript)?;

        self.read_write_openings
            .compute_verifier_openings(&r_grand_product);
        self.init_final_openings
            .compute_verifier_openings(&r_grand_product);

        debug_assert_eq!(grand_product_claims.len(), 8);
        let grand_product_claims: Vec<MultisetHashes<F>> = vec![
            MultisetHashes {
                hash_read: grand_product_claims[0],
                hash_write: grand_product_claims[1],
                hash_init: grand_product_claims[2],
                hash_final: grand_product_claims[3],
            },
            MultisetHashes {
                hash_read: grand_product_claims[4],
                hash_write: grand_product_claims[5],
                hash_init: grand_product_claims[6],
                hash_final: grand_product_claims[7],
            },
        ];
        RangeCheckPolynomials::check_fingerprints(
            grand_product_claims,
            &self.read_write_openings,
            &self.init_final_openings,
            &gamma,
            &tau,
        );

        Ok(())
    }

    fn protocol_name() -> &'static [u8] {
        b"Timestamp validity proof memory checking"
    }
}
