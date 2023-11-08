use ark_ec::CurveGroup;
use ark_ff::{Field, One, PrimeField, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use merlin::Transcript;
use std::any::TypeId;
use std::marker::PhantomData;
use strum::{EnumCount, IntoEnumIterator};

use crate::{
  jolt::{
    instruction::{JoltInstruction, Opcode},
    subtable::LassoSubtable,
  },
  poly::{
    dense_mlpoly::{DensePolynomial, PolyCommitmentGens},
    eq_poly::EqPolynomial,
    identity_poly::IdentityPolynomial,
    structured_poly::{StructuredOpeningProof, StructuredPolynomials},
  },
  subprotocols::{
    combined_table_proof::{CombinedTableCommitment, CombinedTableEvalProof},
    grand_product::{BatchedGrandProductArgument, BatchedGrandProductCircuit, GrandProductCircuit},
    sumcheck::SumcheckInstanceProof,
  },
  utils::{
    errors::ProofVerifyError,
    math::Math,
    random::RandomTape,
    transcript::{AppendToTranscript, ProofTranscript},
  },
};

// TODO(65): Refactor to make more specific.
/// All vectors to be committed in polynomial form.
pub struct InstructionPolynomials<F, G>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
{
  _group: PhantomData<G>,
  /// `C` sized vector of `DensePolynomials` whose evaluations correspond to
  /// indices at which the memories will be evaluated. Each `DensePolynomial` has size
  /// `sparsity`.
  pub dim: Vec<DensePolynomial<F>>,

  /// `C` sized vector of `DensePolynomials` whose evaluations correspond to
  /// read access counts to the memory. Each `DensePolynomial` has size `sparsity`.
  pub read_cts: Vec<DensePolynomial<F>>,

  /// `C` sized vector of `DensePolynomials` whose evaluations correspond to
  /// final access counts to the memory. Each `DensePolynomial` has size M, AKA subtable size.
  pub final_cts: Vec<DensePolynomial<F>>,

  /// `NUM_MEMORIES` sized vector of `DensePolynomials` whose evaluations correspond to
  /// the evaluation of memory accessed at each step of the CPU. Each `DensePolynomial` has
  /// size `sparsity`.
  pub E_polys: Vec<DensePolynomial<F>>,

  /// Polynomial encodings for flag polynomials for each instruction.
  /// Polynomial encodings for flag polynomials for each instruction.
  /// If using a single instruction this will be empty.
  /// NUM_INSTRUCTIONS sized, each polynomial of length 'm' (sparsity).
  ///
  /// Stored independently for use in sumchecking, combined into single DensePolynomial for commitment.
  pub instruction_flag_polys: Vec<DensePolynomial<F>>,

  // TODO(sragss): Storing both the polys and the combined polys may get expensive from a memory
  // perspective. Consider making an additional datastructure to handle the concept of combined polys
  // with a single reference to underlying evaluations.
  pub num_memories: usize,
  pub C: usize,
  pub memory_size: usize,
  pub num_ops: usize,
  pub num_instructions: usize,

  pub materialized_subtables: Vec<Vec<F>>, // NUM_SUBTABLES sized

  /// NUM_SUBTABLES sized – uncommitted but used by the prover for GrandProducts sumchecking. Can be derived by verifier
  /// via summation of all instruction_flags used by a given subtable (/memory)
  pub subtable_flag_polys: Vec<DensePolynomial<F>>,

  /// NUM_MEMORIES sized. Maps memory_to_subtable_map[memory_index] => subtable_index
  /// where memory_index: (0, ... NUM_MEMORIES), subtable_index: (0, ... NUM_SUBTABLES).
  pub memory_to_subtable_map: Vec<usize>,

  /// NUM_MEMORIES sized. Maps memory_to_instructions_map[memory_index] => [instruction_index_0, ...]
  pub memory_to_instructions_map: Vec<Vec<usize>>,
}

pub struct BatchedInstructionPolynomials<F: PrimeField> {
  batched_dim_read: DensePolynomial<F>,
  batched_final: DensePolynomial<F>,
  batched_E: DensePolynomial<F>,
  batched_flag: DensePolynomial<F>,
}

pub struct InstructionCommitment<G: CurveGroup> {
  pub generators: InstructionCommitmentGenerators<G>,

  pub dim_read_commitment: CombinedTableCommitment<G>,
  pub final_commitment: CombinedTableCommitment<G>,
  pub E_commitment: CombinedTableCommitment<G>,
  pub instruction_flag_commitment: CombinedTableCommitment<G>,
}

/// Container for generators for polynomial commitments. These preallocate memory
/// and allow commitments to `DensePolynomials`.
pub struct InstructionCommitmentGenerators<G: CurveGroup> {
  pub dim_read_commitment_gens: PolyCommitmentGens<G>,
  pub final_commitment_gens: PolyCommitmentGens<G>,
  pub E_commitment_gens: PolyCommitmentGens<G>,
  pub flag_commitment_gens: PolyCommitmentGens<G>,
}

// TODO: macro?
impl<F, G> StructuredPolynomials for InstructionPolynomials<F, G>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
{
  type Commitment = InstructionCommitment<G>;
  type BatchedPolynomials = BatchedInstructionPolynomials<F>;

  fn batch(&self) -> Self::BatchedPolynomials {
    let dim_read_polys = [self.dim.as_slice(), self.read_cts.as_slice()].concat();

    Self::BatchedPolynomials {
      batched_dim_read: DensePolynomial::merge(&dim_read_polys),
      batched_final: DensePolynomial::merge(&self.final_cts),
      batched_E: DensePolynomial::merge(&self.E_polys),
      batched_flag: DensePolynomial::merge(&self.instruction_flag_polys),
    }
  }

  fn commit(batched_polys: &Self::BatchedPolynomials) -> Self::Commitment {
    let (dim_read_commitment_gens, dim_read_commitment) = batched_polys
      .batched_dim_read
      .combined_commit(b"BatchedInstructionPolynomials.dim_read");
    let (final_commitment_gens, final_commitment) = batched_polys
      .batched_final
      .combined_commit(b"BatchedInstructionPolynomials.final_cts");
    let (E_commitment_gens, E_commitment) = batched_polys
      .batched_E
      .combined_commit(b"BatchedInstructionPolynomials.E_poly");
    let (flag_commitment_gens, instruction_flag_commitment) = batched_polys
      .batched_flag
      .combined_commit(b"BatchedInstructionPolynomials.flag");

    let generators = InstructionCommitmentGenerators {
      dim_read_commitment_gens,
      final_commitment_gens,
      E_commitment_gens,
      flag_commitment_gens,
    };

    Self::Commitment {
      dim_read_commitment,
      final_commitment,
      E_commitment,
      instruction_flag_commitment,
      generators,
    }
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
struct PrimarySumcheckOpenings<F, G>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
{
  E_poly_openings: Vec<F>,
  flag_openings: Vec<F>,

  E_poly_opening_proof: CombinedTableEvalProof<G>,
  flag_opening_proof: CombinedTableEvalProof<G>,
}

impl<F: PrimeField, G: CurveGroup<ScalarField = F>>
  StructuredOpeningProof<F, G, InstructionPolynomials<F, G>> for PrimarySumcheckOpenings<F, G>
{
  fn prove_openings(
    polynomials: &BatchedInstructionPolynomials<F>,
    commitment: &InstructionCommitment<G>,
    opening_point: &Vec<F>,
    openings: Vec<Vec<F>>,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self {
    debug_assert!(openings.len() == 2);
    let E_poly_openings = &openings[0];
    let flag_openings = &openings[1];

    let E_poly_opening_proof = CombinedTableEvalProof::prove(
      &polynomials.batched_E,
      E_poly_openings,
      opening_point,
      &commitment.generators.E_commitment_gens,
      transcript,
      random_tape,
    );
    let flag_opening_proof = CombinedTableEvalProof::prove(
      &polynomials.batched_flag,
      flag_openings,
      opening_point,
      &commitment.generators.flag_commitment_gens,
      transcript,
      random_tape,
    );

    Self {
      E_poly_openings: E_poly_openings.to_vec(),
      E_poly_opening_proof,
      flag_openings: flag_openings.to_vec(),
      flag_opening_proof,
    }
  }

  fn verify_openings(
    &self,
    commitment: &InstructionCommitment<G>,
    opening_point: &Vec<F>,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    self.E_poly_opening_proof.verify(
      opening_point,
      &self.E_poly_openings,
      &commitment.generators.E_commitment_gens,
      &commitment.E_commitment,
      transcript,
    )?;
    self.flag_opening_proof.verify(
      opening_point,
      &self.flag_openings,
      &commitment.generators.flag_commitment_gens,
      &commitment.instruction_flag_commitment,
      transcript,
    )?;

    Ok(())
  }
}

pub struct InstructionReadWriteOpenings<F, G>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
{
  dim_openings: Vec<F>,    // C-sized
  read_openings: Vec<F>,   // NUM_MEMORIES-sized
  E_poly_openings: Vec<F>, // NUM_MEMORIES-sized
  flag_openings: Vec<F>,   // NUM_INSTRUCTIONS-sized

  dim_read_opening_proof: CombinedTableEvalProof<G>,
  E_poly_opening_proof: CombinedTableEvalProof<G>,
  flag_opening_proof: CombinedTableEvalProof<G>,
}

impl<F, G> StructuredOpeningProof<F, G, InstructionPolynomials<F, G>>
  for InstructionReadWriteOpenings<F, G>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
{
  fn prove_openings(
    polynomials: &BatchedInstructionPolynomials<F>,
    commitment: &InstructionCommitment<G>,
    opening_point: &Vec<F>,
    openings: Vec<Vec<F>>,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self {
    debug_assert!(openings.len() == 4);
    let dim_openings = &openings[0];
    let read_openings = &openings[1];
    let E_poly_openings = &openings[2];
    let flag_openings = &openings[3];

    let dim_read_openings = [dim_openings.as_slice(), read_openings.as_slice()]
      .concat()
      .to_vec();

    let dim_read_opening_proof = CombinedTableEvalProof::prove(
      &polynomials.batched_dim_read,
      &dim_read_openings,
      &opening_point,
      &commitment.generators.dim_read_commitment_gens,
      transcript,
      random_tape,
    );
    let E_poly_opening_proof = CombinedTableEvalProof::prove(
      &polynomials.batched_E,
      E_poly_openings,
      &opening_point,
      &commitment.generators.E_commitment_gens,
      transcript,
      random_tape,
    );
    let flag_opening_proof = CombinedTableEvalProof::prove(
      &polynomials.batched_flag,
      flag_openings,
      &opening_point,
      &commitment.generators.flag_commitment_gens,
      transcript,
      random_tape,
    );

    Self {
      dim_openings: dim_openings.to_vec(),
      read_openings: read_openings.to_vec(),
      E_poly_openings: E_poly_openings.to_vec(),
      flag_openings: flag_openings.to_vec(),
      dim_read_opening_proof,
      E_poly_opening_proof,
      flag_opening_proof,
    }
  }

  fn verify_openings(
    &self,
    commitment: &InstructionCommitment<G>,
    opening_point: &Vec<F>,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    let dim_read_openings = [self.dim_openings.as_slice(), self.read_openings.as_slice()]
      .concat()
      .to_vec();

    self.dim_read_opening_proof.verify(
      opening_point,
      &dim_read_openings,
      &commitment.generators.dim_read_commitment_gens,
      &commitment.dim_read_commitment,
      transcript,
    )?;

    self.E_poly_opening_proof.verify(
      opening_point,
      &self.E_poly_openings,
      &commitment.generators.E_commitment_gens,
      &commitment.E_commitment,
      transcript,
    )?;

    self.flag_opening_proof.verify(
      opening_point,
      &self.flag_openings,
      &commitment.generators.flag_commitment_gens,
      &commitment.instruction_flag_commitment,
      transcript,
    )?;
    Ok(())
  }
}

pub struct InstructionInitFinalOpenings<F, G>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
{
  final_openings: Vec<F>, // NUM_MEMORIES-sized
  final_opening_proof: CombinedTableEvalProof<G>,
}

impl<F, G> StructuredOpeningProof<F, G, InstructionPolynomials<F, G>>
  for InstructionInitFinalOpenings<F, G>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
{
  fn prove_openings(
    polynomials: &BatchedInstructionPolynomials<F>,
    commitment: &InstructionCommitment<G>,
    opening_point: &Vec<F>,
    openings: Vec<Vec<F>>,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self {
    debug_assert!(openings.len() == 1);
    let final_opening_proof = CombinedTableEvalProof::prove(
      &polynomials.batched_flag,
      &openings[0],
      &opening_point,
      &commitment.generators.flag_commitment_gens,
      transcript,
      random_tape,
    );

    Self {
      final_openings: openings[0].clone(),
      final_opening_proof,
    }
  }

  fn verify_openings(
    &self,
    commitment: &InstructionCommitment<G>,
    opening_point: &Vec<F>,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    self.final_opening_proof.verify(
      opening_point,
      &self.final_openings,
      &commitment.generators.final_commitment_gens,
      &commitment.final_commitment,
      transcript,
    )
  }
}

struct MultisetHashes<F: PrimeField> {
  hash_init: F,
  hash_final: F,
  hash_read: F,
  hash_write: F,
}

pub struct MemoryCheckingProof<G, Polynomials, ReadWriteOpenings, InitFinalOpenings>
where
  G: CurveGroup,
  Polynomials: StructuredPolynomials + ?Sized,
  ReadWriteOpenings: StructuredOpeningProof<G::ScalarField, G, Polynomials>,
  InitFinalOpenings: StructuredOpeningProof<G::ScalarField, G, Polynomials>,
{
  _polys: PhantomData<Polynomials>,
  multiset_hashes: Vec<MultisetHashes<G::ScalarField>>,
  read_write_grand_product: BatchedGrandProductArgument<F>,
  init_final_grand_product: BatchedGrandProductArgument<F>,
  read_write_openings: ReadWriteOpenings,
  init_final_openings: InitFinalOpenings,
}

pub trait MemoryChecking<F, G>: StructuredPolynomials
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
{
  type ReadWriteOpenings: StructuredOpeningProof<F, G, Self>;
  type InitFinalOpenings: StructuredOpeningProof<F, G, Self>;

  fn prove_memory_checking(
    polynomials: &Self::BatchedPolynomials,
    commitments: &Self::Commitment,
    r_fingerprint: (&F, &F),
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> MemoryCheckingProof<G, Self, Self::ReadWriteOpenings, Self::InitFinalOpenings> {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    // let (proof_prod_layer, rand_mem, rand_ops) =
    //   ProductLayerProof::prove::<G, S::Polynomials>(polynomials, r_fingerprint, transcript);

    // fka "HashLayerProof"
    let read_write_openings = Self::ReadWriteOpenings::prove_openings(
      polynomials,
      commitments,
      &vec![], // TODO: rand_ops
      vec![],  // TODO: openings_init_final, openings_read_write
      transcript,
      random_tape,
    );
    let init_final_openings = Self::InitFinalOpenings::prove_openings(
      polynomials,
      commitments,
      &vec![], // TODO: rand_mem
      vec![],  // TODO: openings_init_final, openings_read_write
      transcript,
      random_tape,
    );

    unimplemented!("todo");
  }

  fn verify_memory_checking(
    proof: MemoryCheckingProof<G, Self, Self::ReadWriteOpenings, Self::InitFinalOpenings>,
    commitments: &Self::Commitment,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    // Verify grand product arguments

    proof
      .read_write_openings
      .verify_openings(commitments, &vec![] /* TODO */, transcript)?;
    proof
      .init_final_openings
      .verify_openings(commitments, &vec![] /* TODO */, transcript)?;

    // Verify Reed-Solomon fingerprints

    Ok(())
  }

  fn grand_product_leaves_init(&self) -> Vec<Vec<F>> {
    unimplemented!("todo");
  }
  fn grand_product_leaves_final(&self) -> Vec<Vec<F>> {
    unimplemented!("todo");
  }
  fn grand_product_leaves_read(&self) -> Vec<Vec<F>> {
    unimplemented!("todo");
  }
  fn grand_product_leaves_write(&self) -> Vec<Vec<F>> {
    unimplemented!("todo");
  }

  fn batched_init_final_grand_product(&self) -> (BatchedGrandProductCircuit<F>, Vec<F>, Vec<F>) {
    unimplemented!("todo");
  }
  fn batched_read_write_grand_product(&self) -> (BatchedGrandProductCircuit<F>, Vec<F>, Vec<F>) {
    unimplemented!("todo");
  }

  fn protocol_name() -> &'static [u8] {
    unimplemented!("todo");
  }
}

impl<F, G> MemoryChecking<F, G> for InstructionPolynomials<F, G>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
{
  type ReadWriteOpenings = InstructionReadWriteOpenings<F, G>;
  type InitFinalOpenings = InstructionInitFinalOpenings<F, G>;
}

/// Proof of a single Jolt execution.
pub struct InstructionLookupsProof<F, G>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
{
  /// Commitments to all polynomials
  commitment: InstructionCommitment<G>,

  /// Primary collation sumcheck proof
  primary_sumcheck: PrimarySumcheck<F, G>,

  memory_checking: MemoryCheckingProof<
    G,
    InstructionPolynomials<F, G>,
    InstructionReadWriteOpenings<F, G>,
    InstructionInitFinalOpenings<F, G>,
  >,
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PrimarySumcheck<F: PrimeField, G: CurveGroup<ScalarField = F>> {
  sumcheck_proof: SumcheckInstanceProof<F>,
  num_rounds: usize,
  claimed_evaluation: F,
  openings: PrimarySumcheckOpenings<F, G>,
}

pub trait InstructionLookups<F, G>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
{
  type InstructionSet: JoltInstruction + Opcode + IntoEnumIterator + EnumCount;
  type Subtables: LassoSubtable<F> + IntoEnumIterator + EnumCount + From<TypeId> + Into<usize>;

  const C: usize;
  const M: usize;
  const NUM_SUBTABLES: usize = Self::Subtables::COUNT;
  const NUM_INSTRUCTIONS: usize = Self::InstructionSet::COUNT;
  const NUM_MEMORIES: usize = Self::C * Self::Subtables::COUNT;

  fn prove_lookups(
    ops: Vec<Self::InstructionSet>,
    r: Vec<F>,
    transcript: &mut Transcript,
  ) -> InstructionLookupsProof<F, G> {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    let materialized_subtables: Vec<Vec<F>> = Self::materialize_subtables();
    let subtable_lookup_indices: Vec<Vec<usize>> = Self::subtable_lookup_indices(&ops);

    let polynomials = Self::polynomialize(&ops, &subtable_lookup_indices, materialized_subtables);
    let batched_polys = polynomials.batch();
    let commitment = InstructionPolynomials::commit(&batched_polys);

    commitment
      .E_commitment
      .append_to_transcript(b"comm_poly_row_col_ops_val", transcript);

    let eq = EqPolynomial::new(r.to_vec());
    let sumcheck_claim = Self::compute_sumcheck_claim(&ops, &polynomials.E_polys, &eq);

    <Transcript as ProofTranscript<G>>::append_scalar(
      transcript,
      b"claim_eval_scalar_product",
      &sumcheck_claim,
    );

    let mut eq_poly = DensePolynomial::new(EqPolynomial::new(r).evals());
    let num_rounds = ops.len().log_2();

    let (primary_sumcheck_proof, r_primary_sumcheck, flag_evals, E_evals) =
      SumcheckInstanceProof::prove_jolt::<G, Self, Transcript>(
        &F::zero(),
        num_rounds,
        &mut eq_poly,
        &mut polynomials.E_polys.clone(),
        &mut polynomials.instruction_flag_polys.clone(),
        Self::sumcheck_poly_degree(),
        transcript,
      );

    let mut random_tape = RandomTape::new(b"proof");

    // TODO: combine this with prove_jolt?
    // Create a single opening proof for the flag_evals and memory_evals
    let sumcheck_openings = PrimarySumcheckOpenings::prove_openings(
      &batched_polys,
      &commitment,
      &r_primary_sumcheck,
      vec![E_evals, flag_evals],
      transcript,
      &mut random_tape,
    );

    let primary_sumcheck = PrimarySumcheck {
      sumcheck_proof: primary_sumcheck_proof,
      num_rounds,
      claimed_evaluation: sumcheck_claim,
      openings: sumcheck_openings,
    };

    let r_fingerprints: Vec<G::ScalarField> =
      <Transcript as ProofTranscript<G>>::challenge_vector(transcript, b"challenge_r_hash", 2);
    let r_fingerprint = (&r_fingerprints[0], &r_fingerprints[1]);

    let memory_checking = InstructionPolynomials::prove_memory_checking(
      &batched_polys,
      &commitment,
      r_fingerprint,
      transcript,
      &mut random_tape,
    );

    InstructionLookupsProof {
      commitment,
      primary_sumcheck,
      memory_checking,
    }
  }

  fn verify(
    proof: InstructionLookupsProof<F, G>,
    r_eq: &[G::ScalarField],
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    proof
      .commitment
      .E_commitment
      .append_to_transcript(b"comm_poly_row_col_ops_val", transcript);

    <Transcript as ProofTranscript<G>>::append_scalar(
      transcript,
      b"claim_eval_scalar_product",
      &proof.primary_sumcheck.claimed_evaluation,
    );

    let (claim_last, r_primary_sumcheck) = proof
      .primary_sumcheck
      .sumcheck_proof
      .verify::<G, Transcript>(
        proof.primary_sumcheck.claimed_evaluation,
        proof.primary_sumcheck.num_rounds,
        Self::sumcheck_poly_degree(),
        transcript,
      )?;

    // Verify that eq(r, r_z) * [f_1(r_z) * g(E_1(r_z)) + ... + f_F(r_z) * E_F(r_z))] = claim_last
    let eq_eval = EqPolynomial::new(r_eq.to_vec()).evaluate(&r_primary_sumcheck);
    assert_eq!(
      eq_eval
        * Self::combine_lookups_flags(
          &proof.primary_sumcheck.openings.E_poly_openings,
          &proof.primary_sumcheck.openings.flag_openings,
        ),
      claim_last,
      "Primary sumcheck check failed."
    );

    proof.primary_sumcheck.openings.verify_openings(
      &proof.commitment,
      &r_primary_sumcheck,
      transcript,
    );

    let r_mem_check =
      <Transcript as ProofTranscript<G>>::challenge_vector(transcript, b"challenge_r_hash", 2);

    // TODO
    // proof.memory_checking.verify(
    //   &proof.commitment,
    //   &proof.commitment.generators,
    //   Self::memory_to_dimension_index,
    //   Self::evaluate_memory_mle,
    //   (&r_mem_check[0], &r_mem_check[1]),
    //   transcript,
    // )?;

    Ok(())
  }

  fn polynomialize(
    ops: &Vec<Self::InstructionSet>,
    subtable_lookup_indices: &Vec<Vec<usize>>,
    materialized_subtables: Vec<Vec<F>>,
  ) -> InstructionPolynomials<F, G> {
    let m: usize = ops.len().next_power_of_two();

    let mut opcodes: Vec<u8> = ops.iter().map(|op| op.to_opcode()).collect();
    opcodes.resize(m, 0);

    let mut dim: Vec<DensePolynomial<_>> = Vec::with_capacity(Self::C);
    let mut read_cts: Vec<DensePolynomial<_>> = Vec::with_capacity(Self::NUM_MEMORIES);
    let mut final_cts: Vec<DensePolynomial<_>> = Vec::with_capacity(Self::NUM_MEMORIES);
    let mut E_polys: Vec<DensePolynomial<_>> = Vec::with_capacity(Self::NUM_MEMORIES);

    let subtable_map = Self::instruction_to_subtable_map();
    for memory_index in 0..Self::NUM_MEMORIES {
      let access_sequence: &Vec<usize> =
        &subtable_lookup_indices[Self::memory_to_dimension_index(memory_index)];

      let mut final_cts_i = vec![0usize; Self::M];
      let mut read_cts_i = vec![0usize; m];

      for op_index in 0..m {
        let memory_address = access_sequence[op_index];
        debug_assert!(memory_address < Self::M);

        // TODO(JOLT-11): Simplify using subtable map + instruction_map
        // Only increment if the flag is used at this step
        let subtables = &subtable_map[opcodes[op_index] as usize];
        if subtables.contains(&Self::memory_to_subtable_index(memory_index)) {
          let counter = final_cts_i[memory_address];
          read_cts_i[op_index] = counter;
          final_cts_i[memory_address] = counter + 1;
        }
      }

      read_cts.push(DensePolynomial::from_usize(&read_cts_i));
      final_cts.push(DensePolynomial::from_usize(&final_cts_i));
    }

    for i in 0..Self::C {
      let access_sequence: &Vec<usize> = &subtable_lookup_indices[i];

      dim.push(DensePolynomial::from_usize(access_sequence));
    }

    for subtable_index in 0..Self::NUM_SUBTABLES {
      for i in 0..Self::C {
        let subtable_lookups = subtable_lookup_indices[i]
          .iter()
          .map(|&lookup_index| materialized_subtables[subtable_index][lookup_index])
          .collect();
        E_polys.push(DensePolynomial::new(subtable_lookups))
      }
    }

    // TODO(JOLT-11)
    let mut instruction_flag_bitvectors: Vec<Vec<usize>> =
      vec![vec![0usize; m]; Self::NUM_INSTRUCTIONS];
    let mut subtable_flag_bitvectors: Vec<Vec<usize>> = vec![vec![0usize; m]; Self::NUM_SUBTABLES];
    for lookup_index in 0..m {
      let opcode_index = opcodes[lookup_index] as usize;
      instruction_flag_bitvectors[opcode_index][lookup_index] = 1;

      let subtable_indices = &subtable_map[opcode_index];
      for subtable_index in subtable_indices {
        subtable_flag_bitvectors[*subtable_index][lookup_index] = 1;
      }
    }
    let instruction_flag_polys: Vec<DensePolynomial<F>> = instruction_flag_bitvectors
      .iter()
      .map(|flag_bitvector| DensePolynomial::from_usize(&flag_bitvector))
      .collect();
    let subtable_flag_polys: Vec<DensePolynomial<F>> = subtable_flag_bitvectors
      .iter()
      .map(|flag_bitvector| DensePolynomial::from_usize(&flag_bitvector))
      .collect();

    let memory_to_subtable_map: Vec<usize> = (0..Self::NUM_MEMORIES)
      .map(|memory_index| Self::memory_to_subtable_index(memory_index))
      .collect();
    let subtable_to_instructions_map: Vec<Vec<usize>> = Self::subtable_to_instruction_indices();
    let memory_to_instructions_map: Vec<Vec<usize>> = memory_to_subtable_map
      .iter()
      .map(|subtable_index| subtable_to_instructions_map[*subtable_index].clone())
      .collect();

    InstructionPolynomials {
      _group: PhantomData,
      dim,
      read_cts,
      final_cts,
      instruction_flag_polys,
      E_polys,
      num_memories: Self::NUM_MEMORIES,
      C: Self::C,
      memory_size: Self::M,
      num_ops: m,
      num_instructions: Self::NUM_INSTRUCTIONS,
      materialized_subtables,
      subtable_flag_polys,
      memory_to_subtable_map,
      memory_to_instructions_map,
    }
  }

  fn compute_sumcheck_claim(
    ops: &Vec<Self::InstructionSet>,
    E_polys: &Vec<DensePolynomial<F>>,
    eq: &EqPolynomial<F>,
  ) -> F {
    let m = ops.len().next_power_of_two();
    E_polys.iter().for_each(|E_i| assert_eq!(E_i.len(), m));

    let eq_evals = eq.evals();

    let mut claim = F::zero();
    for (k, op) in ops.iter().enumerate() {
      let memory_indices = Self::instruction_to_memory_indices(&op);
      let mut filtered_operands: Vec<F> = Vec::with_capacity(memory_indices.len());

      for memory_index in memory_indices {
        filtered_operands.push(E_polys[memory_index][k]);
      }

      let collation_eval = op.combine_lookups(&filtered_operands, Self::C, Self::M);
      let combined_eval = eq_evals[k] * collation_eval;
      claim += combined_eval;
    }

    claim
  }

  fn instruction_to_memory_indices(op: &Self::InstructionSet) -> Vec<usize> {
    let instruction_subtables: Vec<Self::Subtables> = op
      .subtables::<F>(Self::C)
      .iter()
      .map(|subtable| Self::Subtables::from(subtable.subtable_id()))
      .collect();

    let mut memory_indices = Vec::with_capacity(Self::C * instruction_subtables.len());
    for subtable in instruction_subtables {
      let index: usize = subtable.into();
      memory_indices.extend((Self::C * index)..(Self::C * (index + 1)));
    }

    memory_indices
  }

  fn combine_lookups_flags(vals: &[F], flags: &[F]) -> F {
    assert_eq!(vals.len(), Self::NUM_MEMORIES);
    assert_eq!(flags.len(), Self::NUM_INSTRUCTIONS);

    let mut sum = F::zero();
    for instruction in Self::InstructionSet::iter() {
      let instruction_index = instruction.to_opcode() as usize;
      let memory_indices = Self::instruction_to_memory_indices(&instruction);
      let mut filtered_operands = Vec::with_capacity(memory_indices.len());
      for index in memory_indices {
        filtered_operands.push(vals[index]);
      }
      sum += flags[instruction_index]
        * instruction.combine_lookups(&filtered_operands, Self::C, Self::M);
    }

    sum
  }

  fn sumcheck_poly_degree() -> usize {
    Self::InstructionSet::iter()
      .map(|instruction| instruction.g_poly_degree(Self::C))
      .max()
      .unwrap()
      + 2 // eq and flag
  }

  fn materialize_subtables() -> Vec<Vec<F>> {
    let mut subtables: Vec<Vec<_>> = Vec::with_capacity(Self::Subtables::COUNT);
    for subtable in Self::Subtables::iter() {
      subtables.push(subtable.materialize(Self::M));
    }
    subtables
  }

  fn subtable_lookup_indices(ops: &Vec<Self::InstructionSet>) -> Vec<Vec<usize>> {
    let m = ops.len().next_power_of_two();
    let log_m = Self::M.log_2();
    let chunked_indices: Vec<Vec<usize>> =
      ops.iter().map(|op| op.to_indices(Self::C, log_m)).collect();

    let mut subtable_lookup_indices: Vec<Vec<usize>> = Vec::with_capacity(Self::C);
    for i in 0..Self::C {
      let mut access_sequence: Vec<usize> =
        chunked_indices.iter().map(|chunks| chunks[i]).collect();
      access_sequence.resize(m, 0);
      subtable_lookup_indices.push(access_sequence);
    }
    subtable_lookup_indices
  }

  /// Computes which subtables indices are active for a given instruction.
  /// vec[instruction_index] = [subtable_id_a, subtable_id_b, ...]
  fn instruction_to_subtable_map() -> Vec<Vec<usize>> {
    Self::InstructionSet::iter()
      .map(|instruction| {
        // TODO(sragss): Box<dyn SubtableTrait>.into() should work via additional functionality on the trait .
        let instruction_subtable_ids: Vec<usize> = instruction
          .subtables::<F>(Self::C)
          .iter()
          .map(|subtable| Self::Subtables::from(subtable.subtable_id()).into())
          .collect();

        instruction_subtable_ids
      })
      .collect()
  }

  fn evaluate_memory_mle(memory_index: usize, point: &[F]) -> F {
    let subtable = Self::Subtables::iter()
      .nth(Self::memory_to_subtable_index(memory_index))
      .expect("should exist");
    subtable.evaluate_mle(point)
  }

  fn subtable_to_instruction_indices() -> Vec<Vec<usize>> {
    let mut indices: Vec<Vec<usize>> =
      vec![Vec::with_capacity(Self::NUM_INSTRUCTIONS); Self::NUM_SUBTABLES];

    for instruction in Self::InstructionSet::iter() {
      let instruction_subtables: Vec<Self::Subtables> = instruction
        .subtables::<F>(Self::C)
        .iter()
        .map(|subtable| Self::Subtables::from(subtable.subtable_id()))
        .collect();
      for subtable in instruction_subtables {
        let subtable_index: usize = subtable.into();
        indices[subtable_index].push(instruction.to_opcode() as usize);
      }
    }

    indices
  }

  /// Maps an index [0, NUM_MEMORIES) -> [0, NUM_SUBTABLES)
  fn memory_to_subtable_index(i: usize) -> usize {
    i / Self::C
  }

  /// Maps an index [0, NUM_MEMORIES) -> [0, C)
  fn memory_to_dimension_index(i: usize) -> usize {
    i % Self::C
  }

  fn protocol_name() -> &'static [u8] {
    b"Jolt instruction lookups"
  }
}
