use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::interleave;
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
    unipoly::{CompressedUniPoly, UniPoly},
  },
  subprotocols::{
    combined_table_proof::{CombinedTableCommitment, CombinedTableEvalProof},
    grand_product::{BatchedGrandProductCircuit, GrandProductCircuit},
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
        let leaf_fingerprints = (0..self.num_lookups)
          .map(|i| {
            (
              polynomials.dim[dim_index][i],
              polynomials.E_polys[memory_index][i],
              polynomials.read_cts[memory_index][i],
              None,
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
        let leaf_fingerprints = (0..self.num_lookups)
          .map(|i| {
            (
              polynomials.dim[dim_index][i],
              polynomials.E_polys[memory_index][i],
              polynomials.read_cts[memory_index][i] + F::one(),
              None,
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
    let read_fingerprints: Vec<DensePolynomial<F>> = self.read_leaves(polynomials, gamma, tau);
    let write_fingerprints: Vec<DensePolynomial<F>> = self.write_leaves(polynomials, gamma, tau);
    debug_assert_eq!(read_fingerprints.len(), write_fingerprints.len());

    let mut circuits = Vec::with_capacity(2 * Self::NUM_MEMORIES);
    let mut read_hashes = Vec::with_capacity(Self::NUM_MEMORIES);
    let mut write_hashes = Vec::with_capacity(Self::NUM_MEMORIES);

    for i in 0..Self::NUM_MEMORIES {
      let mut toggled_read_fingerprints = read_fingerprints[i].evals();
      let mut toggled_write_fingerprints = write_fingerprints[i].evals();
      let subtable_index = Self::memory_to_subtable_index(i);
      for j in 0..self.num_lookups {
        let flag = polynomials.subtable_flag_polys[subtable_index][j];
        if flag == F::zero() {
          toggled_read_fingerprints[j] = F::one();
          toggled_write_fingerprints[j] = F::one();
        }
      }

      let read_circuit = GrandProductCircuit::new(&DensePolynomial::new(toggled_read_fingerprints));
      let write_circuit =
        GrandProductCircuit::new(&DensePolynomial::new(toggled_write_fingerprints));
      read_hashes.push(read_circuit.evaluate());
      write_hashes.push(write_circuit.evaluate());
      circuits.push(read_circuit);
      circuits.push(write_circuit);
    }

    // self.memory_to_subtable map has to be expanded because we've doubled the number of "grand products memorys": [read_0, write_0, ... read_NUM_MEMOREIS, write_NUM_MEMORIES]
    let expanded_flag_map: Vec<usize> = (0..2 * Self::NUM_MEMORIES)
      .map(|i| Self::memory_to_subtable_index(i / 2))
      .collect();

    // Prover has access to subtable_flag_polys, which are uncommitted, but verifier can derive from instruction_flag commitments.
    let batched_circuits = BatchedGrandProductCircuit::new_batch_flags(
      circuits,
      polynomials.subtable_flag_polys.clone(),
      expanded_flag_map,
      interleave(read_fingerprints, write_fingerprints).collect(),
    );

    (batched_circuits, read_hashes, write_hashes)
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
      Self::prove_primary_sumcheck(
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

  pub fn verify(
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

  /// Prove Jolt primary sumcheck including instruction collation.
  ///
  /// Computes \sum{ eq(r,x) * [ flags_0(x) * g_0(E(x)) + flags_1(x) * g_1(E(x)) + ... + flags_{NUM_INSTRUCTIONS}(E(x)) * g_{NUM_INSTRUCTIONS}(E(x)) ]}
  /// via the sumcheck protocol.
  /// Note: These E(x) terms differ from term to term depending on the memories used in the instruction.
  ///
  /// Returns: (SumcheckProof, Random evaluation point, claimed evaluations of polynomials)
  ///
  /// Params:
  /// - `claim`: Claimed sumcheck evaluation.
  /// - `num_rounds`: Number of rounds to run sumcheck. Corresponds to the number of free bits or free variables in the polynomials.
  /// - `memory_polys`: Each of the `E` polynomials or "dereferenced memory" polynomials.
  /// - `flag_polys`: Each of the flag selector polynomials describing which instruction is used at a given step of the CPU.
  /// - `degree`: Degree of the inner sumcheck polynomial. Corresponds to number of evaluation points per round.
  /// - `transcript`: Fiat-shamir transcript.
  fn prove_primary_sumcheck(
    _claim: &F,
    num_rounds: usize,
    eq_poly: &mut DensePolynomial<F>,
    memory_polys: &mut Vec<DensePolynomial<F>>,
    flag_polys: &mut Vec<DensePolynomial<F>>,
    degree: usize,
    transcript: &mut Transcript,
  ) -> (SumcheckInstanceProof<F>, Vec<F>, Vec<F>, Vec<F>) {
    // Check all polys are the same size
    let poly_len = eq_poly.len();
    for index in 0..Self::NUM_MEMORIES {
      debug_assert_eq!(memory_polys[index].len(), poly_len);
    }
    for index in 0..Self::NUM_INSTRUCTIONS {
      debug_assert_eq!(flag_polys[index].len(), poly_len);
    }

    let mut random_vars: Vec<F> = Vec::with_capacity(num_rounds);
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

    let num_eval_points = degree + 1;
    for _round in 0..num_rounds {
      let mle_len = eq_poly.len();
      let mle_half = mle_len / 2;

      // Store evaluations of each polynomial at all poly_size / 2 points
      let mut eq_evals: Vec<Vec<F>> = vec![Vec::with_capacity(num_eval_points); mle_half];
      let mut multi_flag_evals: Vec<Vec<Vec<F>>> =
        vec![vec![Vec::with_capacity(num_eval_points); mle_half]; Self::NUM_INSTRUCTIONS];
      let mut multi_memory_evals: Vec<Vec<Vec<F>>> =
        vec![vec![Vec::with_capacity(num_eval_points); mle_half]; Self::NUM_MEMORIES];

      let evaluate_mles_iterator = (0..mle_half).into_iter();

      // Loop over half MLE size (size of MLE next round)
      //   - Compute evaluations of eq, flags, E, at p {0, 1, ..., degree}:
      //       eq(p, _boolean_hypercube_), flags(p, _boolean_hypercube_), E(p, _boolean_hypercube_)
      // After: Sum over MLE elements (with combine)

      for mle_leaf_index in evaluate_mles_iterator {
        // 0
        eq_evals[mle_leaf_index].push(eq_poly[mle_leaf_index]);
        for flag_instruction_index in 0..multi_flag_evals.len() {
          multi_flag_evals[flag_instruction_index][mle_leaf_index]
            .push(flag_polys[flag_instruction_index][mle_leaf_index]);
        }
        for memory_index in 0..multi_memory_evals.len() {
          multi_memory_evals[memory_index][mle_leaf_index]
            .push(memory_polys[memory_index][mle_leaf_index]);
        }

        // 1
        eq_evals[mle_leaf_index].push(eq_poly[mle_half + mle_leaf_index]);
        for flag_instruction_index in 0..multi_flag_evals.len() {
          multi_flag_evals[flag_instruction_index][mle_leaf_index]
            .push(flag_polys[flag_instruction_index][mle_half + mle_leaf_index]);
        }
        for memory_index in 0..multi_memory_evals.len() {
          multi_memory_evals[memory_index][mle_leaf_index]
            .push(memory_polys[memory_index][mle_half + mle_leaf_index]);
        }

        // (2, ...)
        for eval_index in 2..num_eval_points {
          let eq_eval = eq_evals[mle_leaf_index][eval_index - 1]
            + eq_poly[mle_half + mle_leaf_index]
            - eq_poly[mle_leaf_index];
          eq_evals[mle_leaf_index].push(eq_eval);

          for flag_instruction_index in 0..multi_flag_evals.len() {
            let flag_eval = multi_flag_evals[flag_instruction_index][mle_leaf_index]
              [eval_index - 1]
              + flag_polys[flag_instruction_index][mle_half + mle_leaf_index]
              - flag_polys[flag_instruction_index][mle_leaf_index];
            multi_flag_evals[flag_instruction_index][mle_leaf_index].push(flag_eval);
          }
          for memory_index in 0..multi_memory_evals.len() {
            let memory_eval = multi_memory_evals[memory_index][mle_leaf_index][eval_index - 1]
              + memory_polys[memory_index][mle_half + mle_leaf_index]
              - memory_polys[memory_index][mle_leaf_index];
            multi_memory_evals[memory_index][mle_leaf_index].push(memory_eval);
          }
        }
      }

      // Accumulate inner terms.
      // S({0,1,... num_eval_points}) = eq * [ INNER TERMS ] = eq * [ flags_0 * g_0(E_0) + flags_1 * g_1(E_1)]
      let mut evaluations: Vec<F> = Vec::with_capacity(num_eval_points);
      for eval_index in 0..num_eval_points {
        evaluations.push(F::zero());
        for instruction in InstructionSet::iter() {
          let instruction_index = instruction.to_opcode() as usize;
          let memory_indices: Vec<usize> = Self::instruction_to_memory_indices(&instruction);

          for mle_leaf_index in 0..mle_half {
            let mut terms = Vec::with_capacity(memory_indices.len());
            for memory_index in &memory_indices {
              terms.push(multi_memory_evals[*memory_index][mle_leaf_index][eval_index]);
            }

            let instruction_collation_eval = instruction.combine_lookups(&terms, C, M);
            let flag_eval = multi_flag_evals[instruction_index][mle_leaf_index][eval_index];

            // TODO(sragss): May have an excessive group mul here.
            evaluations[eval_index] +=
              eq_evals[mle_leaf_index][eval_index] * flag_eval * instruction_collation_eval;
          }
        }
      } // End accumulation

      let round_uni_poly = UniPoly::from_evals(&evaluations);
      compressed_polys.push(round_uni_poly.compress());

      <UniPoly<F> as AppendToTranscript<G>>::append_to_transcript(
        &round_uni_poly,
        b"poly",
        transcript,
      );

      let r_j =
        <Transcript as ProofTranscript<G>>::challenge_scalar(transcript, b"challenge_nextround");
      random_vars.push(r_j);

      // Bind all polys
      eq_poly.bound_poly_var_top(&r_j);
      for flag_instruction_index in 0..flag_polys.len() {
        flag_polys[flag_instruction_index].bound_poly_var_top(&r_j);
      }
      for memory_index in 0..multi_memory_evals.len() {
        memory_polys[memory_index].bound_poly_var_top(&r_j);
      }
    } // End rounds

    // Pass evaluations at point r back in proof:
    // - flags(r) * NUM_INSTRUCTIONS
    // - E(r) * NUM_SUBTABLES

    // Polys are fully defined so we can just take the first (and only) evaluation
    let flag_evals = (0..flag_polys.len()).map(|i| flag_polys[i][0]).collect();
    let memory_evals = (0..memory_polys.len())
      .map(|i| memory_polys[i][0])
      .collect();

    (
      SumcheckInstanceProof::new(compressed_polys),
      random_vars,
      flag_evals,
      memory_evals,
    )
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
