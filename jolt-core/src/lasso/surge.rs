use std::marker::{PhantomData, Sync};

use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use merlin::Transcript;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{
    jolt::instruction::JoltInstruction,
    lasso::memory_checking::{MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier},
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        structured_poly::{BatchablePolynomials, StructuredOpeningProof},
    },
    subprotocols::{
        batched_commitment::{BatchedPolynomialCommitment, BatchedPolynomialOpeningProof},
        sumcheck::SumcheckInstanceProof,
    },
    utils::{
        errors::ProofVerifyError, math::Math, mul_0_1_optimized, 
        transcript::ProofTranscript,
    },
};

pub struct SurgePolys<F: PrimeField, G: CurveGroup<ScalarField = F>> {
    _group: PhantomData<G>,
    pub dim: Vec<DensePolynomial<F>>,
    pub read_cts: Vec<DensePolynomial<F>>,
    pub final_cts: Vec<DensePolynomial<F>>,
    pub E_polys: Vec<DensePolynomial<F>>,
}

pub struct BatchedSurgePolynomials<F: PrimeField> {
    pub batched_dim_read: DensePolynomial<F>,
    pub batched_final: DensePolynomial<F>,
    pub batched_E: DensePolynomial<F>,
}

pub struct SurgeCommitment<G: CurveGroup> {
    pub dim_read_commitment: BatchedPolynomialCommitment<G>,
    pub final_commitment: BatchedPolynomialCommitment<G>,
    pub E_commitment: BatchedPolynomialCommitment<G>,
}

impl<F, G> BatchablePolynomials for SurgePolys<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type BatchedPolynomials = BatchedSurgePolynomials<F>;
    type Commitment = SurgeCommitment<G>;

    #[tracing::instrument(skip_all, name = "SurgePolys::batch")]
    fn batch(&self) -> Self::BatchedPolynomials {
        let (batched_dim_read, (batched_final, batched_E)) = rayon::join(
            || DensePolynomial::merge(self.dim.iter().chain(&self.read_cts)),
            || {
                rayon::join(
                    || DensePolynomial::merge(&self.final_cts),
                    || DensePolynomial::merge(&self.E_polys),
                )
            },
        );

        Self::BatchedPolynomials {
            batched_dim_read,
            batched_final,
            batched_E,
        }
    }

    #[tracing::instrument(skip_all, name = "SurgePolys::commit")]
    fn commit(batched_polys: &Self::BatchedPolynomials) -> Self::Commitment {
        let dim_read_commitment = batched_polys
            .batched_dim_read
            .combined_commit(b"BatchedSurgePolynomials.dim_read");
        let final_commitment = batched_polys
            .batched_final
            .combined_commit(b"BatchedSurgePolynomials.final_cts");
        let E_commitment = batched_polys
            .batched_E
            .combined_commit(b"BatchedSurgePolynomials.E_poly");

        Self::Commitment {
            dim_read_commitment,
            final_commitment,
            E_commitment,
        }
    }
}

type PrimarySumcheckOpenings<F> = Vec<F>;

impl<F: PrimeField, G: CurveGroup<ScalarField = F>> StructuredOpeningProof<F, G, SurgePolys<F, G>>
    for PrimarySumcheckOpenings<F>
{
    #[tracing::instrument(skip_all, name = "PrimarySumcheckOpenings::open")]
    fn open(polynomials: &SurgePolys<F, G>, opening_point: &Vec<F>) -> Self {
        let chis = EqPolynomial::new(opening_point.to_vec()).evals();
        polynomials
            .E_polys
            .par_iter()
            .map(|poly| poly.evaluate_at_chi(&chis))
            .collect()
    }

    #[tracing::instrument(skip_all, name = "PrimarySumcheckOpenings::prove_openings")]
    fn prove_openings(
        polynomials: &BatchedSurgePolynomials<F>,
        opening_point: &Vec<F>,
        E_poly_openings: &Vec<F>,
        transcript: &mut Transcript,
    ) -> Self::Proof {
        BatchedPolynomialOpeningProof::prove(
            &polynomials.batched_E,
            opening_point,
            E_poly_openings,
            transcript,
        )
    }

    fn verify_openings(
        &self,
        opening_proof: &Self::Proof,
        commitment: &SurgeCommitment<G>,
        opening_point: &Vec<F>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        opening_proof.verify(opening_point, &self, &commitment.E_commitment, transcript)
    }
}

pub struct SurgeReadWriteOpenings<F>
where
    F: PrimeField,
{
    dim_openings: Vec<F>,    // C-sized
    read_openings: Vec<F>,   // C-sized
    E_poly_openings: Vec<F>, // NUM_MEMORIES-sized
}

pub struct SurgeReadWriteOpeningProof<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    dim_read_opening_proof: BatchedPolynomialOpeningProof<G>,
    E_poly_opening_proof: BatchedPolynomialOpeningProof<G>,
}

impl<F, G> StructuredOpeningProof<F, G, SurgePolys<F, G>> for SurgeReadWriteOpenings<F>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    type Proof = SurgeReadWriteOpeningProof<F, G>;

    #[tracing::instrument(skip_all, name = "SurgeReadWriteOpenings::open")]
    fn open(polynomials: &SurgePolys<F, G>, opening_point: &Vec<F>) -> Self {
        let chis = EqPolynomial::new(opening_point.to_vec()).evals();
        let evaluate = |poly: &DensePolynomial<F>| -> F { poly.evaluate_at_chi(&chis) };
        Self {
            dim_openings: polynomials.dim.par_iter().map(evaluate).collect(),
            read_openings: polynomials.read_cts.par_iter().map(evaluate).collect(),
            E_poly_openings: polynomials.E_polys.par_iter().map(evaluate).collect(),
        }
    }

    #[tracing::instrument(skip_all, name = "SurgeReadWriteOpenings::prove_openings")]
    fn prove_openings(
        polynomials: &BatchedSurgePolynomials<F>,
        opening_point: &Vec<F>,
        openings: &Self,
        transcript: &mut Transcript,
    ) -> Self::Proof {
        let mut dim_read_openings = [
            openings.dim_openings.as_slice(),
            openings.read_openings.as_slice(),
        ]
        .concat()
        .to_vec();
        dim_read_openings.resize(dim_read_openings.len().next_power_of_two(), F::zero());

        let dim_read_opening_proof = BatchedPolynomialOpeningProof::prove(
            &polynomials.batched_dim_read,
            &opening_point,
            &dim_read_openings,
            transcript,
        );
        let E_poly_opening_proof = BatchedPolynomialOpeningProof::prove(
            &polynomials.batched_E,
            &opening_point,
            &openings.E_poly_openings,
            transcript,
        );

        SurgeReadWriteOpeningProof {
            dim_read_opening_proof,
            E_poly_opening_proof,
        }
    }

    fn verify_openings(
        &self,
        opening_proof: &Self::Proof,
        commitment: &SurgeCommitment<G>,
        opening_point: &Vec<F>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        let mut dim_read_openings = [self.dim_openings.as_slice(), self.read_openings.as_slice()]
            .concat()
            .to_vec();
        dim_read_openings.resize(dim_read_openings.len().next_power_of_two(), F::zero());

        opening_proof.dim_read_opening_proof.verify(
            opening_point,
            &dim_read_openings,
            &commitment.dim_read_commitment,
            transcript,
        )?;

        opening_proof.E_poly_opening_proof.verify(
            opening_point,
            &self.E_poly_openings,
            &commitment.E_commitment,
            transcript,
        )?;

        Ok(())
    }
}

pub struct SurgeFinalOpenings<F, Instruction, const C: usize>
where
    F: PrimeField,
    Instruction: JoltInstruction + Default,
{
    _instruction: PhantomData<Instruction>,
    final_openings: Vec<F>,       // C-sized
    a_init_final: Option<F>,      // Computed by verifier
    v_init_final: Option<Vec<F>>, // Computed by verifier
}

impl<F, G, Instruction, const C: usize> StructuredOpeningProof<F, G, SurgePolys<F, G>>
    for SurgeFinalOpenings<F, Instruction, C>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
    Instruction: JoltInstruction + Default,
{
    #[tracing::instrument(skip_all, name = "SurgeFinalOpenings::open")]
    fn open(polynomials: &SurgePolys<F, G>, opening_point: &Vec<F>) -> Self {
        let chis = EqPolynomial::new(opening_point.to_vec()).evals();
        let final_openings = polynomials
            .final_cts
            .par_iter()
            .map(|poly| poly.evaluate_at_chi(&chis))
            .collect();
        Self {
            _instruction: PhantomData,
            final_openings,
            a_init_final: None,
            v_init_final: None,
        }
    }

    #[tracing::instrument(skip_all, name = "SurgeFinalOpenings::prove_openings")]
    fn prove_openings(
        polynomials: &BatchedSurgePolynomials<F>,
        opening_point: &Vec<F>,
        openings: &Self,
        transcript: &mut Transcript,
    ) -> Self::Proof {
        BatchedPolynomialOpeningProof::prove(
            &polynomials.batched_final,
            &opening_point,
            &openings.final_openings,
            transcript,
        )
    }

    fn compute_verifier_openings(&mut self, opening_point: &Vec<F>) {
        self.a_init_final =
            Some(IdentityPolynomial::new(opening_point.len()).evaluate(opening_point));
        self.v_init_final = Some(
            Instruction::default()
                .subtables(C)
                .iter()
                .map(|subtable| subtable.evaluate_mle(opening_point))
                .collect(),
        );
    }

    fn verify_openings(
        &self,
        opening_proof: &Self::Proof,
        commitment: &SurgeCommitment<G>,
        opening_point: &Vec<F>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        opening_proof.verify(
            opening_point,
            &self.final_openings,
            &commitment.final_commitment,
            transcript,
        )
    }
}

impl<F, G, Instruction, const C: usize> MemoryCheckingProver<F, G, SurgePolys<F, G>>
    for Surge<F, G, Instruction, C>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
    Instruction: JoltInstruction + Default + Sync,
{
    type ReadWriteOpenings = SurgeReadWriteOpenings<F>;
    type InitFinalOpenings = SurgeFinalOpenings<F, Instruction, C>;

    fn fingerprint(inputs: &(F, F, F), gamma: &F, tau: &F) -> F {
        let (a, v, t) = *inputs;
        t * gamma.square() + v * *gamma + a - tau
    }

    #[tracing::instrument(skip_all, name = "Surge::compute_leaves")]
    fn compute_leaves(
        &self,
        polynomials: &SurgePolys<F, G>,
        gamma: &F,
        tau: &F,
    ) -> (Vec<DensePolynomial<F>>, Vec<DensePolynomial<F>>) {
        let gamma_squared = gamma.square();

        let read_write_leaves = (0..Self::num_memories())
            .into_par_iter()
            .flat_map_iter(|memory_index| {
                let dim_index = Self::memory_to_dimension_index(memory_index);
                let read_fingerprints: Vec<F> = (0..self.num_lookups)
                    .map(|i| {
                        mul_0_1_optimized(&polynomials.read_cts[dim_index][i], &gamma_squared)
                            + mul_0_1_optimized(&polynomials.E_polys[memory_index][i], gamma)
                            + polynomials.dim[dim_index][i]
                            - *tau
                    })
                    .collect();
                let write_fingerprints = read_fingerprints
                    .iter()
                    .map(|read_fingerprint| *read_fingerprint + gamma_squared)
                    .collect();

                vec![
                    DensePolynomial::new(read_fingerprints),
                    DensePolynomial::new(write_fingerprints),
                ]
            })
            .collect();

        let init_final_leaves = (0..Self::num_memories())
            .into_par_iter()
            .flat_map_iter(|memory_index| {
                let dim_index = Self::memory_to_dimension_index(memory_index);
                let subtable_index = Self::memory_to_subtable_index(memory_index);
                // TODO(moodlezoup): Only need one init polynomial per subtable
                let init_fingerprints: Vec<F> = (0..self.M)
                    .map(|i| {
                        // 0 * gamma^2 +
                        mul_0_1_optimized(&self.materialized_subtables[subtable_index][i], gamma)
                            + F::from(i as u64)
                            - *tau
                    })
                    .collect();
                let final_fingerprints = init_fingerprints
                    .iter()
                    .enumerate()
                    .map(|(i, init_fingerprint)| {
                        *init_fingerprint
                            + mul_0_1_optimized(
                                &polynomials.final_cts[dim_index][i],
                                &gamma_squared,
                            )
                    })
                    .collect();

                vec![
                    DensePolynomial::new(init_fingerprints),
                    DensePolynomial::new(final_fingerprints),
                ]
            })
            .collect();

        (read_write_leaves, init_final_leaves)
    }

    fn protocol_name() -> &'static [u8] {
        b"Surge memory checking"
    }
}

impl<F, G, Instruction, const C: usize> MemoryCheckingVerifier<F, G, SurgePolys<F, G>>
    for Surge<F, G, Instruction, C>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
    Instruction: JoltInstruction + Default + Sync,
{
    fn read_tuples(openings: &Self::ReadWriteOpenings) -> Vec<Self::MemoryTuple> {
        (0..Self::num_memories())
            .map(|memory_index| {
                let dim_index = Self::memory_to_dimension_index(memory_index);
                (
                    openings.dim_openings[dim_index],
                    openings.E_poly_openings[memory_index],
                    openings.read_openings[dim_index],
                )
            })
            .collect()
    }
    fn write_tuples(openings: &Self::ReadWriteOpenings) -> Vec<Self::MemoryTuple> {
        (0..Self::num_memories())
            .map(|memory_index| {
                let dim_index = Self::memory_to_dimension_index(memory_index);
                (
                    openings.dim_openings[dim_index],
                    openings.E_poly_openings[memory_index],
                    openings.read_openings[dim_index] + F::one(),
                )
            })
            .collect()
    }
    fn init_tuples(openings: &Self::InitFinalOpenings) -> Vec<Self::MemoryTuple> {
        let a_init = openings.a_init_final.unwrap();
        let v_init = openings.v_init_final.as_ref().unwrap();

        (0..Self::num_memories())
            .map(|memory_index| {
                (
                    a_init,
                    v_init[Self::memory_to_subtable_index(memory_index)],
                    F::zero(),
                )
            })
            .collect()
    }
    fn final_tuples(openings: &Self::InitFinalOpenings) -> Vec<Self::MemoryTuple> {
        let a_init = openings.a_init_final.unwrap();
        let v_init = openings.v_init_final.as_ref().unwrap();

        (0..Self::num_memories())
            .map(|memory_index| {
                let dim_index = Self::memory_to_dimension_index(memory_index);
                (
                    a_init,
                    v_init[Self::memory_to_subtable_index(memory_index)],
                    openings.final_openings[dim_index],
                )
            })
            .collect()
    }
}

pub struct SurgePrimarySumcheck<F: PrimeField, G: CurveGroup<ScalarField = F>> {
    sumcheck_proof: SumcheckInstanceProof<F>,
    num_rounds: usize,
    claimed_evaluation: F,
    openings: PrimarySumcheckOpenings<F>,
    opening_proof: BatchedPolynomialOpeningProof<G>,
}

pub struct Surge<F, G, Instruction, const C: usize>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
    Instruction: JoltInstruction + Default + Sync,
{
    _field: PhantomData<F>,
    _group: PhantomData<G>,
    _instruction: PhantomData<Instruction>,
    ops: Vec<Instruction>,
    materialized_subtables: Vec<Vec<F>>,
    num_lookups: usize,
    M: usize,
}

pub struct SurgeProof<F, G, Instruction, const C: usize>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
    Instruction: JoltInstruction + Default,
{
    /// Commitments to all polynomials
    commitment: SurgeCommitment<G>,

    /// Primary collation sumcheck proof
    primary_sumcheck: SurgePrimarySumcheck<F, G>,

    memory_checking: MemoryCheckingProof<
        G,
        SurgePolys<F, G>,
        SurgeReadWriteOpenings<F>,
        SurgeFinalOpenings<F, Instruction, C>,
    >,
}

impl<F, G, Instruction, const C: usize> Surge<F, G, Instruction, C>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
    Instruction: JoltInstruction + Default + Sync,
{
    fn num_memories() -> usize {
        C * Instruction::default().subtables::<F>(C).len()
    }

    #[tracing::instrument(skip_all, name = "Surge::new")]
    pub fn new(ops: Vec<Instruction>, M: usize) -> Self {
        let num_lookups = ops.len().next_power_of_two();
        let instruction = Instruction::default();

        let materialized_subtables = instruction
            .subtables(C)
            .par_iter()
            .map(|subtable| subtable.materialize(M))
            .collect();

        Self {
            _field: PhantomData,
            _group: PhantomData,
            _instruction: PhantomData,
            ops,
            materialized_subtables,
            num_lookups,
            M,
        }
    }

    /// Maps an index [0, NUM_MEMORIES) -> [0, NUM_SUBTABLES)
    fn memory_to_subtable_index(i: usize) -> usize {
        i / C
    }

    /// Maps an index [0, NUM_MEMORIES) -> [0, C)
    fn memory_to_dimension_index(i: usize) -> usize {
        i % C
    }

    fn protocol_name() -> &'static [u8] {
        b"Surge"
    }

    #[tracing::instrument(skip_all, name = "Surge::prove")]
    pub fn prove(&self, transcript: &mut Transcript) -> SurgeProof<F, G, Instruction, C> {
        <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

        let polynomials = self.construct_polys();
        let batched_polys = polynomials.batch();
        let commitment = SurgePolys::commit(&batched_polys);
        let num_rounds = self.num_lookups.log_2();
        let instruction = Instruction::default();

        // TODO(sragss): Commit some of this stuff to transcript?

        // Primary sumcheck
        let r_primary_sumcheck = <Transcript as ProofTranscript<G>>::challenge_vector(
            transcript,
            b"primary_sumcheck",
            num_rounds,
        );
        let eq = DensePolynomial::new(EqPolynomial::new(r_primary_sumcheck.to_vec()).evals());
        let sumcheck_claim: F = Self::compute_primary_sumcheck_claim(&polynomials, &eq, self.M);

        <Transcript as ProofTranscript<G>>::append_scalar(
            transcript,
            b"sumcheck_claim",
            &sumcheck_claim,
        );
        let mut combined_sumcheck_polys = polynomials.E_polys.clone();
        combined_sumcheck_polys.push(eq);

        let combine_lookups_eq = |vals: &[F]| -> F {
            let vals_no_eq: &[F] = &vals[0..(vals.len() - 1)];
            let eq = vals[vals.len() - 1];
            instruction.combine_lookups(vals_no_eq, C, self.M) * eq
        };

        let (primary_sumcheck_proof, r_z, _) =
            SumcheckInstanceProof::<F>::prove_arbitrary::<_, G, Transcript>(
                &sumcheck_claim,
                num_rounds,
                &mut combined_sumcheck_polys,
                combine_lookups_eq,
                instruction.g_poly_degree(C) + 1, // combined degree + eq term
                transcript,
            );

        let sumcheck_openings = PrimarySumcheckOpenings::open(&polynomials, &r_z); // TODO: use return value from prove_arbitrary?
                                                                                   // Create a single opening proof for the E polynomials
        let sumcheck_opening_proof = PrimarySumcheckOpenings::prove_openings(
            &batched_polys,
            &r_z,
            &sumcheck_openings,
            transcript,
        );

        let primary_sumcheck = SurgePrimarySumcheck {
            claimed_evaluation: sumcheck_claim,
            sumcheck_proof: primary_sumcheck_proof,
            num_rounds,
            openings: sumcheck_openings,
            opening_proof: sumcheck_opening_proof,
        };

        let memory_checking = self.prove_memory_checking(&polynomials, &batched_polys, transcript);

        SurgeProof {
            commitment,
            primary_sumcheck,
            memory_checking,
        }
    }

    pub fn verify(
        proof: SurgeProof<F, G, Instruction, C>,
        transcript: &mut Transcript,
        M: usize,
    ) -> Result<(), ProofVerifyError> {
        <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());
        let instruction = Instruction::default();

        let r_primary_sumcheck = <Transcript as ProofTranscript<G>>::challenge_vector(
            transcript,
            b"primary_sumcheck",
            proof.primary_sumcheck.num_rounds,
        );

        <Transcript as ProofTranscript<G>>::append_scalar(
            transcript,
            b"sumcheck_claim",
            &proof.primary_sumcheck.claimed_evaluation,
        );
        let primary_sumcheck_poly_degree = instruction.g_poly_degree(C) + 1;
        let (claim_last, r_z) = proof
            .primary_sumcheck
            .sumcheck_proof
            .verify::<G, Transcript>(
                proof.primary_sumcheck.claimed_evaluation,
                proof.primary_sumcheck.num_rounds,
                primary_sumcheck_poly_degree,
                transcript,
            )?;

        let eq_eval = EqPolynomial::new(r_primary_sumcheck.to_vec()).evaluate(&r_z);
        assert_eq!(
            eq_eval * instruction.combine_lookups(&proof.primary_sumcheck.openings, C, M),
            claim_last,
            "Primary sumcheck check failed."
        );

        proof.primary_sumcheck.openings.verify_openings(
            &proof.primary_sumcheck.opening_proof,
            &proof.commitment,
            &r_z,
            transcript,
        )?;

        Self::verify_memory_checking(proof.memory_checking, &proof.commitment, transcript)
    }

    #[tracing::instrument(skip_all, name = "Surge::construct_polys")]
    fn construct_polys(&self) -> SurgePolys<F, G> {
        let mut dim_usize: Vec<Vec<usize>> = vec![vec![0; self.num_lookups]; C];

        let mut read_cts = vec![vec![0usize; self.num_lookups]; C];
        let mut final_cts = vec![vec![0usize; self.M]; C];
        let log_M = ark_std::log2(self.M) as usize;

        for (op_index, op) in self.ops.iter().enumerate() {
            let access_sequence = op.to_indices(C, log_M);
            assert_eq!(access_sequence.len(), C);

            for dimension_index in 0..C {
                let memory_address = access_sequence[dimension_index];
                debug_assert!(memory_address < self.M);

                dim_usize[dimension_index][op_index] = memory_address;

                let ts = final_cts[dimension_index][memory_address];
                read_cts[dimension_index][op_index] = ts;
                let write_timestamp = ts + 1;
                final_cts[dimension_index][memory_address] = write_timestamp;
            }
        }

        // num_ops is padded to the nearest power of 2 for the usage of DensePolynomial. We cannot just fill
        // in zeros for read_cts and final_cts as this implicitly specifies a read at address 0. The prover
        // and verifier plumbing assume write_ts(r) = read_ts(r) + 1. This will not hold unless we update
        // the final_cts for these phantom reads.
        for fake_ops_index in self.ops.len()..self.num_lookups {
            for dimension_index in 0..C {
                let memory_address = 0;
                let ts = final_cts[dimension_index][memory_address];
                read_cts[dimension_index][fake_ops_index] = ts;
                let write_timestamp = ts + 1;
                final_cts[dimension_index][memory_address] = write_timestamp;
            }
        }

        let dim: Vec<DensePolynomial<F>> = dim_usize
            .iter()
            .map(|dim| DensePolynomial::from_usize(dim))
            .collect();
        let read_cts: Vec<DensePolynomial<F>> = read_cts
            .iter()
            .map(|read| DensePolynomial::from_usize(read))
            .collect();
        let final_cts: Vec<DensePolynomial<F>> = final_cts
            .iter()
            .map(|fin| DensePolynomial::from_usize(fin))
            .collect();

        // Construct E
        let mut E_i_evals = Vec::with_capacity(Self::num_memories());
        for E_index in 0..Self::num_memories() {
            let mut E_evals = Vec::with_capacity(self.num_lookups);
            for op_index in 0..self.num_lookups {
                let dimension_index = Self::memory_to_dimension_index(E_index);
                let subtable_index = Self::memory_to_subtable_index(E_index);

                let eval_index = dim_usize[dimension_index][op_index];
                let eval = self.materialized_subtables[subtable_index][eval_index];
                E_evals.push(eval);
            }
            E_i_evals.push(E_evals);
        }
        let E_poly: Vec<DensePolynomial<F>> = E_i_evals
            .iter()
            .map(|E| DensePolynomial::new(E.to_vec()))
            .collect();

        SurgePolys {
            _group: PhantomData,
            dim,
            read_cts,
            final_cts,
            E_polys: E_poly,
        }
    }

    #[tracing::instrument(skip_all, name = "Surge::compute_primary_sumcheck_claim")]
    fn compute_primary_sumcheck_claim(
        polys: &SurgePolys<F, G>,
        eq: &DensePolynomial<F>,
        M: usize,
    ) -> F {
        let g_operands = &polys.E_polys;
        let hypercube_size = g_operands[0].len();
        g_operands
            .iter()
            .for_each(|operand| assert_eq!(operand.len(), hypercube_size));

        let instruction = Instruction::default();

        (0..hypercube_size)
            .into_par_iter()
            .map(|eval_index| {
                let g_operands: Vec<F> = (0..Self::num_memories())
                    .map(|memory_index| g_operands[memory_index][eval_index])
                    .collect();
                eq[eval_index] * instruction.combine_lookups(&g_operands, C, M)
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use merlin::Transcript;

    use super::Surge;
    use crate::jolt::instruction::xor::XORInstruction;
    use ark_curve25519::{EdwardsProjective, Fr};

    #[test]
    fn e2e() {
        let ops = vec![
            XORInstruction(12, 12),
            XORInstruction(12, 82),
            XORInstruction(12, 12),
            XORInstruction(25, 12),
        ];
        const C: usize = 8;
        const M: usize = 1 << 8;

        let mut transcript = Transcript::new(b"test_transcript");
        let surge = <Surge<Fr, EdwardsProjective, XORInstruction, C>>::new(ops, M);
        let proof = surge.prove(&mut transcript);

        let mut transcript = Transcript::new(b"test_transcript");
        <Surge<Fr, EdwardsProjective, XORInstruction, C>>::verify(proof, &mut transcript, M)
            .expect("should work");
    }

    #[test]
    fn e2e_non_pow_2() {
        let ops = vec![
            XORInstruction(0, 1),
            XORInstruction(101, 101),
            XORInstruction(202, 1),
            XORInstruction(220, 1),
            XORInstruction(220, 1),
        ];
        const C: usize = 2;
        const M: usize = 1 << 8;

        let mut transcript = Transcript::new(b"test_transcript");
        let surge = <Surge<Fr, EdwardsProjective, XORInstruction, C>>::new(ops, M);
        let proof = surge.prove(&mut transcript);

        let mut transcript = Transcript::new(b"test_transcript");
        <Surge<Fr, EdwardsProjective, XORInstruction, C>>::verify(proof, &mut transcript, M)
            .expect("should work");
    }
}
