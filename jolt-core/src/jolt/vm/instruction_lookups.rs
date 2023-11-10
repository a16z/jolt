use ark_ec::CurveGroup;
use ark_ff::{One, PrimeField, Zero};
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
  lasso::memory_checking::{MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier},
  poly::{
    dense_mlpoly::{DensePolynomial, PolyCommitmentGens},
    eq_poly::EqPolynomial,
    identity_poly::IdentityPolynomial,
    structured_poly::{StructuredOpeningProof, StructuredPolynomials},
  },
  subprotocols::{
    combined_table_proof::{CombinedTableCommitment, CombinedTableEvalProof},
    grand_product::BatchedGrandProductCircuit,
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

  /// NUM_SUBTABLES sized – uncommitted but used by the prover for GrandProducts sumchecking. Can be derived by verifier
  /// via summation of all instruction_flags used by a given subtable (/memory)
  pub subtable_flag_polys: Vec<DensePolynomial<F>>,
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
  type Openings = (Vec<F>, Vec<F>);

  fn open(_polynomials: &InstructionPolynomials<F, G>, _opening_point: &Vec<F>) -> Self::Openings {
    unimplemented!("Openings are output by sumcheck protocol");
  }

  fn prove_openings(
    polynomials: &BatchedInstructionPolynomials<F>,
    commitment: &InstructionCommitment<G>,
    opening_point: &Vec<F>,
    openings: (Vec<F>, Vec<F>),
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self {
    let E_poly_openings = &openings.0;
    let flag_openings = &openings.1;

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
  type Openings = [Vec<F>; 4];

  fn open(polynomials: &InstructionPolynomials<F, G>, opening_point: &Vec<F>) -> Self::Openings {
    let evaluate = |poly: &DensePolynomial<F>| -> F { poly.evaluate(&opening_point) };
    [
      polynomials.dim.iter().map(evaluate).collect(),
      polynomials.read_cts.iter().map(evaluate).collect(),
      polynomials.E_polys.iter().map(evaluate).collect(),
      polynomials
        .instruction_flag_polys
        .iter()
        .map(evaluate)
        .collect(),
    ]
  }

  fn prove_openings(
    polynomials: &BatchedInstructionPolynomials<F>,
    commitment: &InstructionCommitment<G>,
    opening_point: &Vec<F>,
    openings: [Vec<F>; 4],
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self {
    let dim_openings = &openings[0];
    let read_openings = &openings[1];
    let E_poly_openings = &openings[2];
    let flag_openings = &openings[3];

    let mut dim_read_openings = [dim_openings.as_slice(), read_openings.as_slice()]
      .concat()
      .to_vec();
    dim_read_openings.resize(dim_read_openings.len().next_power_of_two(), F::zero());

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
    let mut dim_read_openings = [self.dim_openings.as_slice(), self.read_openings.as_slice()]
      .concat()
      .to_vec();
    dim_read_openings.resize(dim_read_openings.len().next_power_of_two(), F::zero());

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

pub struct InstructionFinalOpenings<F, G>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
{
  final_openings: Vec<F>, // NUM_MEMORIES-sized
  final_opening_proof: CombinedTableEvalProof<G>,
  a_init_final: Option<F>,      // Computed by verifier
  v_init_final: Option<Vec<F>>, // Computed by verifier
}

impl<F, G> StructuredOpeningProof<F, G, InstructionPolynomials<F, G>>
  for InstructionFinalOpenings<F, G>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
{
  type Openings = Vec<F>;

  fn open(polynomials: &InstructionPolynomials<F, G>, opening_point: &Vec<F>) -> Self::Openings {
    polynomials
      .final_cts
      .iter()
      .map(|final_cts_i| final_cts_i.evaluate(opening_point))
      .collect()
  }

  fn prove_openings(
    polynomials: &BatchedInstructionPolynomials<F>,
    commitment: &InstructionCommitment<G>,
    opening_point: &Vec<F>,
    openings: Vec<F>,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self {
    let final_opening_proof = CombinedTableEvalProof::prove(
      &polynomials.batched_final,
      &openings,
      &opening_point,
      &commitment.generators.final_commitment_gens,
      transcript,
      random_tape,
    );

    Self {
      final_openings: openings,
      final_opening_proof,
      a_init_final: None,
      v_init_final: None,
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

impl<F, G, InstructionSet, Subtables, const C: usize, const M: usize>
  MemoryCheckingProver<F, G, InstructionPolynomials<F, G>>
  for InstructionLookups<F, G, InstructionSet, Subtables, C, M>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
  InstructionSet: JoltInstruction + Opcode + IntoEnumIterator + EnumCount,
  Subtables: LassoSubtable<F> + IntoEnumIterator + EnumCount + From<TypeId> + Into<usize>,
{
  type ReadWriteOpenings = InstructionReadWriteOpenings<F, G>;
  type InitFinalOpenings = InstructionFinalOpenings<F, G>;

  type MemoryTuple = (F, F, F, Option<F>); // (a, v, t, flag)

  fn fingerprint(inputs: &(F, F, F, Option<F>), gamma: &F, tau: &F) -> F {
    let (a, v, t, flag) = *inputs;
    match flag {
      Some(val) => val * (t * gamma.square() + v * *gamma + a - tau) + F::one() - val,
      None => t * gamma.square() + v * *gamma + a - tau,
    }
  }

  fn read_leaves(
    &self,
    polynomials: &InstructionPolynomials<F, G>,
    gamma: &F,
    tau: &F,
  ) -> Vec<DensePolynomial<F>> {
    (0..Self::NUM_MEMORIES)
      .map(|memory_index| {
        let dim_index = Self::memory_to_dimension_index(memory_index);
        let subtable_index = Self::memory_to_subtable_index(memory_index);
        let leaf_fingerprints = (0..self.num_lookups)
          .map(|i| {
            (
              polynomials.dim[dim_index][i],
              polynomials.E_polys[memory_index][i],
              polynomials.read_cts[memory_index][i],
              Some(polynomials.subtable_flag_polys[subtable_index][i]),
            )
          })
          .map(|tuple| Self::fingerprint(&tuple, gamma, tau))
          .collect();
        DensePolynomial::new(leaf_fingerprints)
      })
      .collect()
  }
  fn write_leaves(
    &self,
    polynomials: &InstructionPolynomials<F, G>,
    gamma: &F,
    tau: &F,
  ) -> Vec<DensePolynomial<F>> {
    (0..Self::NUM_MEMORIES)
      .map(|memory_index| {
        let dim_index = Self::memory_to_dimension_index(memory_index);
        let subtable_index = Self::memory_to_subtable_index(memory_index);
        let leaf_fingerprints = (0..self.num_lookups)
          .map(|i| {
            (
              polynomials.dim[dim_index][i],
              polynomials.E_polys[memory_index][i],
              polynomials.read_cts[memory_index][i] + F::one(),
              Some(polynomials.subtable_flag_polys[subtable_index][i]),
            )
          })
          .map(|tuple| Self::fingerprint(&tuple, gamma, tau))
          .collect();
        DensePolynomial::new(leaf_fingerprints)
      })
      .collect()
  }
  fn init_leaves(
    &self,
    _polynomials: &InstructionPolynomials<F, G>,
    gamma: &F,
    tau: &F,
  ) -> Vec<DensePolynomial<F>> {
    (0..Self::NUM_MEMORIES)
      .map(|memory_index| {
        let subtable_index = Self::memory_to_subtable_index(memory_index);
        let leaf_fingerprints = (0..self.num_lookups)
          .map(|i| {
            (
              F::from(i as u64),
              self.materialized_subtables[subtable_index][i],
              F::zero(),
              None,
            )
          })
          .map(|tuple| Self::fingerprint(&tuple, gamma, tau))
          .collect();
        DensePolynomial::new(leaf_fingerprints)
      })
      .collect()
  }
  fn final_leaves(
    &self,
    polynomials: &InstructionPolynomials<F, G>,
    gamma: &F,
    tau: &F,
  ) -> Vec<DensePolynomial<F>> {
    (0..Self::NUM_MEMORIES)
      .map(|memory_index| {
        let subtable_index = Self::memory_to_subtable_index(memory_index);
        let leaf_fingerprints = (0..self.num_lookups)
          .map(|i| {
            (
              F::from(i as u64),
              self.materialized_subtables[subtable_index][i],
              polynomials.final_cts[memory_index][i],
              None,
            )
          })
          .map(|tuple| Self::fingerprint(&tuple, gamma, tau))
          .collect();
        DensePolynomial::new(leaf_fingerprints)
      })
      .collect()
  }

  fn read_write_grand_product(
    &self,
    polynomials: &InstructionPolynomials<F, G>,
    gamma: &F,
    tau: &F,
  ) -> (BatchedGrandProductCircuit<F>, Vec<F>, Vec<F>) {
    todo!();
  }

  fn protocol_name() -> &'static [u8] {
    b"Instruction lookups memory checking"
  }
}

impl<F, G, InstructionSet, Subtables, const C: usize, const M: usize>
  MemoryCheckingVerifier<F, G, InstructionPolynomials<F, G>>
  for InstructionLookups<F, G, InstructionSet, Subtables, C, M>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
  InstructionSet: JoltInstruction + Opcode + IntoEnumIterator + EnumCount,
  Subtables: LassoSubtable<F> + IntoEnumIterator + EnumCount + From<TypeId> + Into<usize>,
{
  fn compute_verifier_openings(openings: &mut Self::InitFinalOpenings, opening_point: &Vec<F>) {
    openings.a_init_final =
      Some(IdentityPolynomial::new(opening_point.len()).evaluate(opening_point));
    openings.v_init_final = Some(
      Subtables::iter()
        .map(|subtable| subtable.evaluate_mle(opening_point))
        .collect(),
    );
  }

  fn read_tuples(openings: &Self::ReadWriteOpenings) -> Vec<Self::MemoryTuple> {
    let subtable_flags = Self::subtable_flags(&openings.flag_openings);
    (0..Self::NUM_MEMORIES)
      .map(|memory_index| {
        let subtable_index = Self::memory_to_subtable_index(memory_index);
        (
          openings.dim_openings[memory_index],
          openings.E_poly_openings[memory_index],
          openings.read_openings[memory_index],
          Some(subtable_flags[subtable_index]),
        )
      })
      .collect()
  }
  fn write_tuples(openings: &Self::ReadWriteOpenings) -> Vec<Self::MemoryTuple> {
    Self::read_tuples(openings)
      .iter()
      .map(|(a, v, t, flag)| (*a, *v, *t + F::one(), *flag))
      .collect()
  }
  fn init_tuples(openings: &Self::InitFinalOpenings) -> Vec<Self::MemoryTuple> {
    let a_init = openings.a_init_final.unwrap();
    let v_init = openings.v_init_final.as_ref().unwrap();

    (0..Self::NUM_MEMORIES)
      .map(|memory_index| {
        (
          a_init,
          v_init[Self::memory_to_subtable_index(memory_index)],
          F::zero(),
          None,
        )
      })
      .collect()
  }
  fn final_tuples(openings: &Self::InitFinalOpenings) -> Vec<Self::MemoryTuple> {
    let a_init = openings.a_init_final.unwrap();
    let v_init = openings.v_init_final.as_ref().unwrap();

    (0..Self::NUM_MEMORIES)
      .map(|memory_index| {
        (
          a_init,
          v_init[Self::memory_to_subtable_index(memory_index)],
          openings.final_openings[memory_index],
          None,
        )
      })
      .collect()
  }
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
    InstructionFinalOpenings<F, G>,
  >,
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PrimarySumcheck<F: PrimeField, G: CurveGroup<ScalarField = F>> {
  sumcheck_proof: SumcheckInstanceProof<F>,
  num_rounds: usize,
  claimed_evaluation: F,
  openings: PrimarySumcheckOpenings<F, G>,
}

pub struct InstructionLookups<F, G, InstructionSet, Subtables, const C: usize, const M: usize>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
  InstructionSet: JoltInstruction + Opcode + IntoEnumIterator + EnumCount,
  Subtables: LassoSubtable<F> + IntoEnumIterator + EnumCount + From<TypeId> + Into<usize>,
{
  _field: PhantomData<F>,
  _group: PhantomData<G>,
  _instructions: PhantomData<InstructionSet>,
  _subtables: PhantomData<Subtables>,
  ops: Vec<InstructionSet>,
  materialized_subtables: Vec<Vec<F>>,
  num_lookups: usize,
}

impl<F, G, InstructionSet, Subtables, const C: usize, const M: usize>
  InstructionLookups<F, G, InstructionSet, Subtables, C, M>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
  InstructionSet: JoltInstruction + Opcode + IntoEnumIterator + EnumCount,
  Subtables: LassoSubtable<F> + IntoEnumIterator + EnumCount + From<TypeId> + Into<usize>,
{
  const NUM_SUBTABLES: usize = Subtables::COUNT;
  const NUM_INSTRUCTIONS: usize = InstructionSet::COUNT;
  const NUM_MEMORIES: usize = C * Subtables::COUNT;

  pub fn new(ops: Vec<InstructionSet>) -> Self {
    let materialized_subtables = Self::materialize_subtables();
    let num_lookups = ops.len().next_power_of_two();

    Self {
      _field: PhantomData,
      _group: PhantomData,
      _instructions: PhantomData,
      _subtables: PhantomData,
      ops,
      materialized_subtables,
      num_lookups,
    }
  }

  pub fn prove_lookups(
    &self,
    r: Vec<F>,
    transcript: &mut Transcript,
  ) -> InstructionLookupsProof<F, G> {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());
    let polynomials = self.polynomialize();
    let batched_polys = polynomials.batch();
    let commitment = InstructionPolynomials::commit(&batched_polys);

    commitment
      .E_commitment
      .append_to_transcript(b"comm_poly_row_col_ops_val", transcript);

    let eq = EqPolynomial::new(r.to_vec());
    let sumcheck_claim = Self::compute_sumcheck_claim(&self.ops, &polynomials.E_polys, &eq);

    <Transcript as ProofTranscript<G>>::append_scalar(
      transcript,
      b"claim_eval_scalar_product",
      &sumcheck_claim,
    );

    let mut eq_poly = DensePolynomial::new(EqPolynomial::new(r).evals());
    let num_rounds = self.ops.len().log_2();

    // TODO: compartmentalize all primary sumcheck logic
    let (primary_sumcheck_proof, r_primary_sumcheck, flag_evals, E_evals) =
      SumcheckInstanceProof::prove_jolt::<G, Transcript>(
        &F::zero(),
        num_rounds,
        &mut eq_poly,
        &mut polynomials.E_polys.clone(),
        &mut polynomials.instruction_flag_polys.clone(),
        Self::sumcheck_poly_degree(),
        transcript,
      );

    let mut random_tape = RandomTape::new(b"proof");

    // Create a single opening proof for the flag_evals and memory_evals
    let sumcheck_openings = PrimarySumcheckOpenings::prove_openings(
      &batched_polys,
      &commitment,
      &r_primary_sumcheck,
      (E_evals, flag_evals),
      transcript,
      &mut random_tape,
    );

    let primary_sumcheck = PrimarySumcheck {
      sumcheck_proof: primary_sumcheck_proof,
      num_rounds,
      claimed_evaluation: sumcheck_claim,
      openings: sumcheck_openings,
    };

    let memory_checking = self.prove_memory_checking(
      &polynomials,
      &batched_polys,
      &commitment,
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

    // TODO: compartmentalize all primary sumcheck logic
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
        * Self::combine_lookups(
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
    )?;

    Self::verify_memory_checking(proof.memory_checking, &proof.commitment, transcript)?;

    Ok(())
  }

  fn polynomialize(&self) -> InstructionPolynomials<F, G> {
    let m: usize = self.ops.len().next_power_of_two();

    let mut dim: Vec<DensePolynomial<_>> = Vec::with_capacity(C);
    let mut read_cts: Vec<DensePolynomial<_>> = Vec::with_capacity(Self::NUM_MEMORIES);
    let mut final_cts: Vec<DensePolynomial<_>> = Vec::with_capacity(Self::NUM_MEMORIES);
    let mut E_polys: Vec<DensePolynomial<_>> = Vec::with_capacity(Self::NUM_MEMORIES);

    let subtable_lookup_indices: Vec<Vec<usize>> = Self::subtable_lookup_indices(&self.ops);
    for memory_index in 0..Self::NUM_MEMORIES {
      let dim_index = Self::memory_to_dimension_index(memory_index);
      let subtable_index = Self::memory_to_subtable_index(memory_index);
      let access_sequence: &Vec<usize> = &subtable_lookup_indices[dim_index];

      let mut final_cts_i = vec![0usize; M];
      let mut read_cts_i = vec![0usize; m];
      let mut subtable_lookups = vec![F::zero(); m];

      for (j, op) in self.ops.iter().enumerate() {
        let memories_used = Self::instruction_to_memory_indices(&op);
        if memories_used.contains(&memory_index) {
          let memory_address = access_sequence[j];
          debug_assert!(memory_address < M);

          let counter = final_cts_i[memory_address];
          read_cts_i[j] = counter;
          final_cts_i[memory_address] = counter + 1;
          subtable_lookups[j] = self.materialized_subtables[subtable_index][memory_address];
        }
      }

      E_polys.push(DensePolynomial::new(subtable_lookups));
      read_cts.push(DensePolynomial::from_usize(&read_cts_i));
      final_cts.push(DensePolynomial::from_usize(&final_cts_i));
    }

    for i in 0..C {
      let access_sequence: &Vec<usize> = &subtable_lookup_indices[i];
      dim.push(DensePolynomial::from_usize(access_sequence));
    }

    let mut instruction_flag_bitvectors: Vec<Vec<usize>> =
      vec![vec![0usize; m]; Self::NUM_INSTRUCTIONS];
    for (j, op) in self.ops.iter().enumerate() {
      let opcode_index = op.to_opcode() as usize;
      instruction_flag_bitvectors[opcode_index][j] = 1;
    }
    let instruction_flag_polys: Vec<DensePolynomial<F>> = instruction_flag_bitvectors
      .iter()
      .map(|flag_bitvector| DensePolynomial::from_usize(&flag_bitvector))
      .collect();

    let subtable_flag_polys = Self::subtable_flag_polys(&instruction_flag_polys);

    InstructionPolynomials {
      _group: PhantomData,
      dim,
      read_cts,
      final_cts,
      instruction_flag_polys,
      subtable_flag_polys,
      E_polys,
    }
  }

  fn compute_sumcheck_claim(
    ops: &Vec<InstructionSet>,
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

      let collation_eval = op.combine_lookups(&filtered_operands, C, M);
      let combined_eval = eq_evals[k] * collation_eval;
      claim += combined_eval;
    }

    claim
  }

  fn combine_lookups(vals: &[F], flags: &[F]) -> F {
    assert_eq!(vals.len(), Self::NUM_MEMORIES);
    assert_eq!(flags.len(), Self::NUM_INSTRUCTIONS);

    let mut sum = F::zero();
    for instruction in InstructionSet::iter() {
      let instruction_index = instruction.to_opcode() as usize;
      let memory_indices = Self::instruction_to_memory_indices(&instruction);
      let mut filtered_operands = Vec::with_capacity(memory_indices.len());
      for index in memory_indices {
        filtered_operands.push(vals[index]);
      }
      sum += flags[instruction_index] * instruction.combine_lookups(&filtered_operands, C, M);
    }

    sum
  }

  fn subtable_flags(instruction_flags: &Vec<F>) -> Vec<F> {
    let mut subtable_flags = vec![F::zero(); Self::NUM_SUBTABLES];
    for (i, instruction) in InstructionSet::iter().enumerate() {
      let instruction_subtables: Vec<Subtables> = instruction
        .subtables::<F>(C)
        .iter()
        .map(|subtable| Subtables::from(subtable.subtable_id()))
        .collect();
      for subtable in instruction_subtables {
        let subtable_index: usize = subtable.into();
        subtable_flags[subtable_index] += &instruction_flags[i];
      }
    }
    subtable_flags
  }

  fn subtable_flag_polys(
    instruction_flag_polys: &Vec<DensePolynomial<F>>,
  ) -> Vec<DensePolynomial<F>> {
    let m = instruction_flag_polys[0].len();
    let mut subtable_flag_polys =
      vec![DensePolynomial::new(vec![F::zero(); m]); Self::NUM_SUBTABLES];
    for (i, instruction) in InstructionSet::iter().enumerate() {
      let instruction_subtables: Vec<Subtables> = instruction
        .subtables::<F>(C)
        .iter()
        .map(|subtable| Subtables::from(subtable.subtable_id()))
        .collect();
      for subtable in instruction_subtables {
        let subtable_index: usize = subtable.into();
        subtable_flag_polys[subtable_index] += &instruction_flag_polys[i];
      }
    }
    subtable_flag_polys
  }

  fn instruction_to_memory_indices(op: &InstructionSet) -> Vec<usize> {
    let instruction_subtables: Vec<Subtables> = op
      .subtables::<F>(C)
      .iter()
      .map(|subtable| Subtables::from(subtable.subtable_id()))
      .collect();

    let mut memory_indices = Vec::with_capacity(C * instruction_subtables.len());
    for subtable in instruction_subtables {
      let index: usize = subtable.into();
      memory_indices.extend((C * index)..(C * (index + 1)));
    }

    memory_indices
  }

  fn sumcheck_poly_degree() -> usize {
    InstructionSet::iter()
      .map(|instruction| instruction.g_poly_degree(C))
      .max()
      .unwrap()
      + 2 // eq and flag
  }

  fn materialize_subtables() -> Vec<Vec<F>> {
    let mut subtables: Vec<Vec<_>> = Vec::with_capacity(Subtables::COUNT);
    for subtable in Subtables::iter() {
      subtables.push(subtable.materialize(M));
    }
    subtables
  }

  fn subtable_lookup_indices(ops: &Vec<InstructionSet>) -> Vec<Vec<usize>> {
    let m = ops.len().next_power_of_two();
    let log_M = M.log_2();
    let chunked_indices: Vec<Vec<usize>> = ops.iter().map(|op| op.to_indices(C, log_M)).collect();

    let mut subtable_lookup_indices: Vec<Vec<usize>> = Vec::with_capacity(C);
    for i in 0..C {
      let mut access_sequence: Vec<usize> =
        chunked_indices.iter().map(|chunks| chunks[i]).collect();
      access_sequence.resize(m, 0);
      subtable_lookup_indices.push(access_sequence);
    }
    subtable_lookup_indices
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
    b"Jolt instruction lookups"
  }
}
