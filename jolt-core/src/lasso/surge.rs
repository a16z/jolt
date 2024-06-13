use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::BatchType;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::marker::{PhantomData, Sync};

use crate::{
    jolt::instruction::JoltInstruction,
    lasso::memory_checking::{MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier},
    poly::{
        commitment::{commitment_scheme::CommitmentScheme, hyrax::matrix_dimensions},
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        structured_poly::{StructuredCommitment, StructuredOpeningProof},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{errors::ProofVerifyError, math::Math, mul_0_1_optimized, transcript::ProofTranscript},
};

pub struct SurgePolys<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    _marker: PhantomData<PCS>,
    pub dim: Vec<DensePolynomial<F>>,
    pub read_cts: Vec<DensePolynomial<F>>,
    pub final_cts: Vec<DensePolynomial<F>>,
    pub E_polys: Vec<DensePolynomial<F>>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct SurgeCommitment<CS: CommitmentScheme> {
    /// Commitments to dim_i and read_cts_i polynomials.
    pub dim_read_commitment: Vec<CS::Commitment>,
    /// Commitment to final_cts_i polynomials.
    pub final_commitment: Vec<CS::Commitment>,
    /// Commitments to E_i polynomials.
    pub E_commitment: Vec<CS::Commitment>,
}

impl<F, PCS> StructuredCommitment<PCS> for SurgePolys<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    type Commitment = SurgeCommitment<PCS>;

    #[tracing::instrument(skip_all, name = "SurgePolys::commit")]
    fn commit(&self, generators: &PCS::Setup) -> Self::Commitment {
        let _read_write_num_vars = self.dim[0].get_num_vars();
        let dim_read_polys: Vec<&DensePolynomial<F>> =
            self.dim.iter().chain(self.read_cts.iter()).collect();
        let dim_read_commitment =
            PCS::batch_commit_polys_ref(&dim_read_polys, generators, BatchType::SurgeReadWrite);
        let E_commitment =
            PCS::batch_commit_polys(&self.E_polys, generators, BatchType::SurgeReadWrite);

        let _final_num_vars = self.final_cts[0].get_num_vars();
        let final_commitment =
            PCS::batch_commit_polys(&self.final_cts, generators, BatchType::SurgeInitFinal);

        Self::Commitment {
            dim_read_commitment,
            final_commitment,
            E_commitment,
        }
    }
}

type PrimarySumcheckOpenings<F> = Vec<F>;

impl<F, PCS> StructuredOpeningProof<F, PCS, SurgePolys<F, PCS>> for PrimarySumcheckOpenings<F>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    type Proof = PCS::BatchedProof;

    #[tracing::instrument(skip_all, name = "PrimarySumcheckOpenings::open")]
    fn open(polynomials: &SurgePolys<F, PCS>, opening_point: &[F]) -> Self {
        let chis = EqPolynomial::evals(opening_point);
        polynomials
            .E_polys
            .par_iter()
            .map(|poly| poly.evaluate_at_chi(&chis))
            .collect()
    }

    #[tracing::instrument(skip_all, name = "PrimarySumcheckOpenings::prove_openings")]
    fn prove_openings(
        generators: &PCS::Setup,
        polynomials: &SurgePolys<F, PCS>,
        opening_point: &[F],
        E_poly_openings: &Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        PCS::batch_prove(
            generators,
            &polynomials.E_polys.iter().collect::<Vec<_>>(),
            opening_point,
            E_poly_openings,
            BatchType::SurgeReadWrite,
            transcript,
        )
    }

    fn verify_openings(
        &self,
        generators: &PCS::Setup,
        opening_proof: &Self::Proof,
        commitment: &SurgeCommitment<PCS>,
        opening_point: &[F],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        PCS::batch_verify(
            opening_proof,
            generators,
            opening_point,
            self,
            &commitment.E_commitment.iter().collect::<Vec<_>>(),
            transcript,
        )
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct SurgeReadWriteOpenings<F>
where
    F: JoltField,
{
    dim_openings: Vec<F>,    // C-sized
    read_openings: Vec<F>,   // C-sized
    E_poly_openings: Vec<F>, // NUM_MEMORIES-sized
}

impl<F, PCS> StructuredOpeningProof<F, PCS, SurgePolys<F, PCS>> for SurgeReadWriteOpenings<F>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    type Proof = PCS::BatchedProof;

    #[tracing::instrument(skip_all, name = "SurgeReadWriteOpenings::open")]
    fn open(polynomials: &SurgePolys<F, PCS>, opening_point: &[F]) -> Self {
        let chis = EqPolynomial::evals(opening_point);
        let evaluate = |poly: &DensePolynomial<F>| -> F { poly.evaluate_at_chi(&chis) };
        Self {
            dim_openings: polynomials.dim.par_iter().map(evaluate).collect(),
            read_openings: polynomials.read_cts.par_iter().map(evaluate).collect(),
            E_poly_openings: polynomials.E_polys.par_iter().map(evaluate).collect(),
        }
    }

    #[tracing::instrument(skip_all, name = "SurgeReadWriteOpenings::prove_openings")]
    fn prove_openings(
        generators: &PCS::Setup,
        polynomials: &SurgePolys<F, PCS>,
        opening_point: &[F],
        openings: &Self,
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        let read_write_polys = polynomials
            .dim
            .iter()
            .chain(polynomials.read_cts.iter())
            .chain(polynomials.E_polys.iter())
            .collect::<Vec<_>>();
        let read_write_openings = [
            openings.dim_openings.as_slice(),
            openings.read_openings.as_slice(),
            openings.E_poly_openings.as_slice(),
        ]
        .concat();

        PCS::batch_prove(
            generators,
            &read_write_polys,
            opening_point,
            &read_write_openings,
            BatchType::SurgeReadWrite,
            transcript,
        )
    }

    fn verify_openings(
        &self,
        generators: &PCS::Setup,
        opening_proof: &Self::Proof,
        commitment: &SurgeCommitment<PCS>,
        opening_point: &[F],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let read_write_openings: Vec<F> = [
            self.dim_openings.as_slice(),
            self.read_openings.as_slice(),
            self.E_poly_openings.as_slice(),
        ]
        .concat();
        PCS::batch_verify(
            opening_proof,
            generators,
            opening_point,
            &read_write_openings,
            &commitment
                .dim_read_commitment
                .iter()
                .chain(commitment.E_commitment.iter())
                .collect::<Vec<_>>(),
            transcript,
        )
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct SurgeFinalOpenings<F, Instruction, const C: usize, const M: usize>
where
    F: JoltField,
    Instruction: JoltInstruction + Default,
{
    _instruction: PhantomData<Instruction>,
    final_openings: Vec<F>,       // C-sized
    a_init_final: Option<F>,      // Computed by verifier
    v_init_final: Option<Vec<F>>, // Computed by verifier
}

impl<F, PCS, Instruction, const C: usize, const M: usize>
    StructuredOpeningProof<F, PCS, SurgePolys<F, PCS>> for SurgeFinalOpenings<F, Instruction, C, M>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    Instruction: JoltInstruction + Default,
{
    type Proof = PCS::BatchedProof;
    type Preprocessing = SurgePreprocessing<F, Instruction, C, M>;

    #[tracing::instrument(skip_all, name = "SurgeFinalOpenings::open")]
    fn open(polynomials: &SurgePolys<F, PCS>, opening_point: &[F]) -> Self {
        let chis = EqPolynomial::evals(opening_point);
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
        generators: &PCS::Setup,
        polynomials: &SurgePolys<F, PCS>,
        opening_point: &[F],
        openings: &Self,
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        PCS::batch_prove(
            generators,
            &polynomials.final_cts.iter().collect::<Vec<_>>(),
            opening_point,
            &openings.final_openings,
            BatchType::SurgeInitFinal,
            transcript,
        )
    }

    fn compute_verifier_openings(&mut self, _: &Self::Preprocessing, opening_point: &[F]) {
        self.a_init_final =
            Some(IdentityPolynomial::new(opening_point.len()).evaluate(opening_point));
        self.v_init_final = Some(
            Instruction::default()
                .subtables(C, M)
                .iter()
                .map(|(subtable, _)| subtable.evaluate_mle(opening_point))
                .collect(),
        );
    }

    fn verify_openings(
        &self,
        generators: &PCS::Setup,
        opening_proof: &Self::Proof,
        commitment: &SurgeCommitment<PCS>,
        opening_point: &[F],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        PCS::batch_verify(
            opening_proof,
            generators,
            opening_point,
            &self.final_openings,
            &commitment.final_commitment.iter().collect::<Vec<_>>(),
            transcript,
        )
    }
}

impl<F, PCS, Instruction, const C: usize, const M: usize>
    MemoryCheckingProver<F, PCS, SurgePolys<F, PCS>> for SurgeProof<F, PCS, Instruction, C, M>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    Instruction: JoltInstruction + Default + Sync,
{
    type Preprocessing = SurgePreprocessing<F, Instruction, C, M>;
    type ReadWriteOpenings = SurgeReadWriteOpenings<F>;
    type InitFinalOpenings = SurgeFinalOpenings<F, Instruction, C, M>;

    fn fingerprint(inputs: &(F, F, F), gamma: &F, tau: &F) -> F {
        let (a, v, t) = *inputs;
        t * gamma.square() + v * *gamma + a - *tau
    }

    #[tracing::instrument(skip_all, name = "Surge::compute_leaves")]
    fn compute_leaves(
        preprocessing: &SurgePreprocessing<F, Instruction, C, M>,
        polynomials: &SurgePolys<F, PCS>,
        gamma: &F,
        tau: &F,
    ) -> (Vec<Vec<F>>, Vec<Vec<F>>) {
        let gamma_squared = gamma.square();
        let num_lookups = polynomials.dim[0].len();

        let read_write_leaves = (0..Self::num_memories())
            .into_par_iter()
            .flat_map_iter(|memory_index| {
                let dim_index = Self::memory_to_dimension_index(memory_index);
                let read_fingerprints: Vec<F> = (0..num_lookups)
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

                vec![read_fingerprints, write_fingerprints]
            })
            .collect();

        let init_final_leaves = (0..Self::num_memories())
            .into_par_iter()
            .flat_map_iter(|memory_index| {
                let dim_index = Self::memory_to_dimension_index(memory_index);
                let subtable_index = Self::memory_to_subtable_index(memory_index);
                // TODO(moodlezoup): Only need one init polynomial per subtable
                let init_fingerprints: Vec<F> = (0..M)
                    .map(|i| {
                        // 0 * gamma^2 +
                        mul_0_1_optimized(
                            &preprocessing.materialized_subtables[subtable_index][i],
                            gamma,
                        ) + F::from_u64(i as u64).unwrap()
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

                vec![init_fingerprints, final_fingerprints]
            })
            .collect();

        (read_write_leaves, init_final_leaves)
    }

    fn protocol_name() -> &'static [u8] {
        b"Surge memory checking"
    }
}

impl<F, CS, Instruction, const C: usize, const M: usize>
    MemoryCheckingVerifier<F, CS, SurgePolys<F, CS>> for SurgeProof<F, CS, Instruction, C, M>
where
    F: JoltField,
    CS: CommitmentScheme<Field = F>,
    Instruction: JoltInstruction + Default + Sync,
{
    fn read_tuples(
        _preprocessing: &SurgePreprocessing<F, Instruction, C, M>,
        openings: &Self::ReadWriteOpenings,
    ) -> Vec<Self::MemoryTuple> {
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
    fn write_tuples(
        _preprocessing: &SurgePreprocessing<F, Instruction, C, M>,
        openings: &Self::ReadWriteOpenings,
    ) -> Vec<Self::MemoryTuple> {
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
    fn init_tuples(
        _preprocessing: &SurgePreprocessing<F, Instruction, C, M>,
        openings: &Self::InitFinalOpenings,
    ) -> Vec<Self::MemoryTuple> {
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
    fn final_tuples(
        _preprocessing: &SurgePreprocessing<F, Instruction, C, M>,
        openings: &Self::InitFinalOpenings,
    ) -> Vec<Self::MemoryTuple> {
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

pub struct SurgePrimarySumcheck<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    sumcheck_proof: SumcheckInstanceProof<F>,
    num_rounds: usize,
    claimed_evaluation: F,
    openings: PrimarySumcheckOpenings<F>,
    opening_proof: PCS::BatchedProof,
}

pub struct SurgePreprocessing<F, Instruction, const C: usize, const M: usize>
where
    F: JoltField,
    Instruction: JoltInstruction + Default,
{
    _instruction: PhantomData<Instruction>,
    materialized_subtables: Vec<Vec<F>>,
}

#[allow(clippy::type_complexity)]
pub struct SurgeProof<F, PCS, Instruction, const C: usize, const M: usize>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    Instruction: JoltInstruction + Default,
{
    /// Commitments to all polynomials
    commitment: SurgeCommitment<PCS>,

    /// Primary collation sumcheck proof
    primary_sumcheck: SurgePrimarySumcheck<F, PCS>,

    memory_checking: MemoryCheckingProof<
        F,
        PCS,
        SurgePolys<F, PCS>,
        SurgeReadWriteOpenings<F>,
        SurgeFinalOpenings<F, Instruction, C, M>,
    >,
}

impl<F, Instruction, const C: usize, const M: usize> SurgePreprocessing<F, Instruction, C, M>
where
    F: JoltField,
    Instruction: JoltInstruction + Default + Sync,
{
    #[tracing::instrument(skip_all, name = "Surge::preprocess")]
    pub fn preprocess() -> Self {
        let instruction = Instruction::default();

        let materialized_subtables = instruction
            .subtables(C, M)
            .par_iter()
            .map(|(subtable, _)| subtable.materialize(M))
            .collect();

        Self {
            _instruction: PhantomData,
            materialized_subtables,
        }
    }
}

impl<F, PCS, Instruction, const C: usize, const M: usize> SurgeProof<F, PCS, Instruction, C, M>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    Instruction: JoltInstruction + Default + Sync,
{
    fn num_memories() -> usize {
        C * Instruction::default().subtables::<F>(C, M).len()
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

    /// Computes the maximum number of group generators needed to commit to Surge polynomials
    /// using Hyrax, given `M` and the maximum number of lookups.
    pub fn num_generators(max_num_lookups: usize) -> usize {
        let max_num_lookups = max_num_lookups.next_power_of_two();
        let num_read_write_generators = matrix_dimensions(max_num_lookups.log_2(), 16).1;
        let num_init_final_generators =
            matrix_dimensions((M * Self::num_memories()).next_power_of_two().log_2(), 4).1;
        std::cmp::max(num_read_write_generators, num_init_final_generators)
    }

    #[tracing::instrument(skip_all, name = "Surge::prove")]
    pub fn prove(
        preprocessing: &SurgePreprocessing<F, Instruction, C, M>,
        generators: &PCS::Setup,
        ops: Vec<Instruction>,
        transcript: &mut ProofTranscript,
    ) -> Self {
        transcript.append_protocol_name(Self::protocol_name());

        let num_lookups = ops.len().next_power_of_two();
        let polynomials = Self::construct_polys(preprocessing, &ops);
        let commitment = polynomials.commit(generators);

        let num_rounds = num_lookups.log_2();
        let instruction = Instruction::default();

        // TODO(sragss): Commit some of this stuff to transcript?

        // Primary sumcheck
        let r_primary_sumcheck = transcript.challenge_vector(b"primary_sumcheck", num_rounds);
        let eq: DensePolynomial<F> = DensePolynomial::new(EqPolynomial::evals(&r_primary_sumcheck));
        let sumcheck_claim: F = Self::compute_primary_sumcheck_claim(&polynomials, &eq);

        transcript.append_scalar(b"sumcheck_claim", &sumcheck_claim);
        let mut combined_sumcheck_polys = polynomials.E_polys.clone();
        combined_sumcheck_polys.push(eq);

        let combine_lookups_eq = |vals: &[F]| -> F {
            let vals_no_eq: &[F] = &vals[0..(vals.len() - 1)];
            let eq = vals[vals.len() - 1];
            instruction.combine_lookups(vals_no_eq, C, M) * eq
        };

        let (primary_sumcheck_proof, r_z, _) = SumcheckInstanceProof::<F>::prove_arbitrary::<_>(
            &sumcheck_claim,
            num_rounds,
            &mut combined_sumcheck_polys,
            combine_lookups_eq,
            instruction.g_poly_degree(C) + 1, // combined degree + eq term
            transcript,
        );

        let sumcheck_openings = PrimarySumcheckOpenings::open(&polynomials, &r_z); // TODO: use return value from prove_arbitrary?
        let sumcheck_opening_proof = PrimarySumcheckOpenings::prove_openings(
            generators,
            &polynomials,
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

        let memory_checking =
            SurgeProof::prove_memory_checking(generators, preprocessing, &polynomials, transcript);

        SurgeProof {
            commitment,
            primary_sumcheck,
            memory_checking,
        }
    }

    pub fn verify(
        preprocessing: &SurgePreprocessing<F, Instruction, C, M>,
        generators: &PCS::Setup,
        proof: SurgeProof<F, PCS, Instruction, C, M>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        transcript.append_protocol_name(Self::protocol_name());
        let instruction = Instruction::default();

        let r_primary_sumcheck =
            transcript.challenge_vector(b"primary_sumcheck", proof.primary_sumcheck.num_rounds);

        transcript.append_scalar(
            b"sumcheck_claim",
            &proof.primary_sumcheck.claimed_evaluation,
        );
        let primary_sumcheck_poly_degree = instruction.g_poly_degree(C) + 1;
        let (claim_last, r_z) = proof.primary_sumcheck.sumcheck_proof.verify(
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
            generators,
            &proof.primary_sumcheck.opening_proof,
            &proof.commitment,
            &r_z,
            transcript,
        )?;

        Self::verify_memory_checking(
            preprocessing,
            generators,
            proof.memory_checking,
            &proof.commitment,
            transcript,
        )
    }

    #[tracing::instrument(skip_all, name = "Surge::construct_polys")]
    fn construct_polys(
        preprocessing: &SurgePreprocessing<F, Instruction, C, M>,
        ops: &[Instruction],
    ) -> SurgePolys<F, PCS> {
        let num_lookups = ops.len().next_power_of_two();
        let mut dim_usize: Vec<Vec<usize>> = vec![vec![0; num_lookups]; C];

        let mut read_cts = vec![vec![0usize; num_lookups]; C];
        let mut final_cts = vec![vec![0usize; M]; C];
        let log_M = ark_std::log2(M) as usize;

        for (op_index, op) in ops.iter().enumerate() {
            let access_sequence = op.to_indices(C, log_M);
            assert_eq!(access_sequence.len(), C);

            for dimension_index in 0..C {
                let memory_address = access_sequence[dimension_index];
                debug_assert!(memory_address < M);

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
        for fake_ops_index in ops.len()..num_lookups {
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
            let mut E_evals = Vec::with_capacity(num_lookups);
            for op_index in 0..num_lookups {
                let dimension_index = Self::memory_to_dimension_index(E_index);
                let subtable_index = Self::memory_to_subtable_index(E_index);

                let eval_index = dim_usize[dimension_index][op_index];
                let eval = preprocessing.materialized_subtables[subtable_index][eval_index];
                E_evals.push(eval);
            }
            E_i_evals.push(E_evals);
        }
        let E_poly: Vec<DensePolynomial<F>> = E_i_evals
            .iter()
            .map(|E| DensePolynomial::new(E.to_vec()))
            .collect();

        SurgePolys {
            _marker: PhantomData,
            dim,
            read_cts,
            final_cts,
            E_polys: E_poly,
        }
    }

    #[tracing::instrument(skip_all, name = "Surge::compute_primary_sumcheck_claim")]
    fn compute_primary_sumcheck_claim(polys: &SurgePolys<F, PCS>, eq: &DensePolynomial<F>) -> F {
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
    use super::SurgePreprocessing;
    use crate::{
        jolt::instruction::xor::XORInstruction,
        lasso::surge::SurgeProof,
        poly::{commitment::hyrax::HyraxScheme, commitment::pedersen::PedersenGenerators},
        utils::transcript::ProofTranscript,
    };
    use ark_bn254::{Fr, G1Projective};

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

        let mut transcript = ProofTranscript::new(b"test_transcript");
        let preprocessing = SurgePreprocessing::preprocess();
        let generators = PedersenGenerators::new(
            SurgeProof::<Fr, HyraxScheme<G1Projective>, XORInstruction, C, M>::num_generators(128),
            b"LassoV1",
        );
        let proof = SurgeProof::<Fr, HyraxScheme<G1Projective>, XORInstruction, C, M>::prove(
            &preprocessing,
            &generators,
            ops,
            &mut transcript,
        );

        let mut transcript = ProofTranscript::new(b"test_transcript");
        SurgeProof::verify(&preprocessing, &generators, proof, &mut transcript)
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

        let mut transcript = ProofTranscript::new(b"test_transcript");
        let preprocessing = SurgePreprocessing::preprocess();
        let generators = PedersenGenerators::new(
            SurgeProof::<Fr, HyraxScheme<G1Projective>, XORInstruction, C, M>::num_generators(128),
            b"LassoV1",
        );
        let proof = SurgeProof::<Fr, HyraxScheme<G1Projective>, XORInstruction, C, M>::prove(
            &preprocessing,
            &generators,
            ops,
            &mut transcript,
        );

        let mut transcript = ProofTranscript::new(b"test_transcript");
        SurgeProof::verify(&preprocessing, &generators, proof, &mut transcript)
            .expect("should work");
    }
}
