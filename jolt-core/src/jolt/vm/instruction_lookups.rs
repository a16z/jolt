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
  lasso::{fingerprint_strategy::FingerprintStrategy, memory_checking::MemoryCheckingProof},
  poly::{
    dense_mlpoly::{DensePolynomial, PolyCommitmentGens},
    eq_poly::EqPolynomial,
    identity_poly::IdentityPolynomial,
  },
  subprotocols::{
    combined_table_proof::{CombinedTableCommitment, CombinedTableEvalProof},
    grand_product::{BGPCInterpretable, BatchedGrandProductCircuit, GPEvals, GrandProductCircuit},
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
pub struct InstructionPolynomials<F: PrimeField, G: CurveGroup<ScalarField = F>> {
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

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
struct PrimarySumcheckOpenings<F: PrimeField, G: CurveGroup<ScalarField = F>> {
  E_poly_openings: Vec<F>,
  flag_openings: Vec<F>,

  E_poly_opening_proof: CombinedTableEvalProof<G>,
  flag_opening_proof: CombinedTableEvalProof<G>,
}

struct MemoryCheckingOpeningProof<F: PrimeField, G: CurveGroup<ScalarField = F>> {
  dim_openings: Vec<F>,    // C-sized
  read_openings: Vec<F>,   // NUM_MEMORIES-sized
  E_poly_openings: Vec<F>, // NUM_MEMORIES-sized
  final_openings: Vec<F>,  // NUM_MEMORIES-sized
  flag_openings: Vec<F>,   // NUM_INSTRUCTIONS-sized

  dim_read_opening_proof: CombinedTableEvalProof<G>,
  E_poly_opening_proof: CombinedTableEvalProof<G>,
  final_opening_proof: CombinedTableEvalProof<G>,
  flag_opening_proof: CombinedTableEvalProof<G>,

  /// Maps memory_index to relevant instruction_flag indices.
  /// Used by verifier to construct subtable_flag from instruction_flags.
  memory_to_flag_indices: Vec<Vec<usize>>,
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

pub struct BatchedInstructionPolynomials<F: PrimeField> {
  batched_dim_read: DensePolynomial<F>,
  batched_final: DensePolynomial<F>,
  batched_E: DensePolynomial<F>,
  batched_flag: DensePolynomial<F>,
}

// alt: BatchablePolynomials
pub trait StructuredPolynomials {
  type Commitment;
  type BatchedPolynomials;

  fn batch(&self) -> Self::BatchedPolynomials;
  fn commit(batched_polys: Self::BatchedPolynomials) -> Self::Commitment;
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

  fn commit(batched_polys: Self::BatchedPolynomials) -> Self::Commitment {
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

pub trait StructuredOpeningProof<F: PrimeField, G: CurveGroup<ScalarField = F>> {
  type Polynomials: StructuredPolynomials;
  type BatchedPolynomials = <Self::Polynomials as StructuredPolynomials>::BatchedPolynomials;
  type Commitment = <Self::Polynomials as StructuredPolynomials>::Commitment;
  // TODO: associated type for openings?

  fn prove_openings(
    polynomials: Self::BatchedPolynomials,
    commitment: Self::Commitment,
    r: &Vec<F>,
    openings: Vec<Vec<F>>,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self;

  fn verify_openings(&self, commitment: Self::Commitment, r: &Vec<F>);
}

impl<F: PrimeField, G: CurveGroup<ScalarField = F>> StructuredOpeningProof<F, G>
  for PrimarySumcheckOpenings<F, G>
{
  type Polynomials = InstructionPolynomials<F, G>;

  fn prove_openings(
    polynomials: Self::BatchedPolynomials,
    commitment: Self::Commitment,
    r: &Vec<F>,
    openings: Vec<Vec<F>>,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self {
    let E_poly_openings = openings[0];
    let flag_openings = openings[1];

    let flag_proof = CombinedTableEvalProof::prove(
      &polynomials.batched_flag,
      &flag_openings,
      &r,
      &commitment.generators.flag_commitment_gens,
      transcript,
      &mut random_tape,
    );

    let memory_proof = CombinedTableEvalProof::prove(
      &polynomials.batched_E,
      &E_poly_openings,
      &r,
      &commitment.generators.E_commitment_gens,
      transcript,
      &mut random_tape,
    );

    unimplemented!("todo");
  }

  fn verify_openings(&self, commitment: Self::Commitment, r: &Vec<F>) {
    unimplemented!("todo");
  }
}

impl<F: PrimeField, G: CurveGroup<ScalarField = F>> StructuredOpeningProof<F, G>
  for MemoryCheckingOpeningProof<F, G>
{
  type Polynomials = InstructionPolynomials<F, G>;

  fn prove_openings(
    polynomials: Self::BatchedPolynomials,
    commitment: Self::Commitment,
    r: &Vec<F>,
    openings: Vec<Vec<F>>,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self {
    unimplemented!("todo");
  }

  fn verify_openings(&self, commitment: Self::Commitment, r: &Vec<F>) {
    unimplemented!("todo");
  }
}

/// Proof of a single Jolt execution.
pub struct InstructionLookupsProof<G: CurveGroup> {
  /// Commitments to all polynomials
  commitment: InstructionCommitment<G>,

  /// Primary collation sumcheck proof
  primary_sumcheck: PrimarySumcheck<G>,

  memory_checking: MemoryCheckingProof<G, ROFlagsFingerprintProof<G>>,
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PrimarySumcheck<G: CurveGroup> {
  sumcheck_proof: SumcheckInstanceProof<G::ScalarField>,
  num_rounds: usize,
  claimed_evaluation: G::ScalarField,
  openings: PrimarySumcheckOpenings<G::ScalarField, G>,
}

pub trait InstructionLookups<F: PrimeField, G: CurveGroup<ScalarField = F>> {
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
  ) -> InstructionLookupsProof<G> {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    let m = ops.len().next_power_of_two();

    let materialized_subtables: Vec<Vec<F>> = Self::materialize_subtables();
    let subtable_lookup_indices: Vec<Vec<usize>> = Self::subtable_lookup_indices(&ops);
    let polynomials = Self::polynomialize(&ops, &subtable_lookup_indices, materialized_subtables);
    let batched_polys = polynomials.batch();
    let commitment = InstructionPolynomials::commit(batched_polys);

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

    let num_rounds = ops.len().log_2();
    let mut eq_poly = DensePolynomial::new(EqPolynomial::new(r).evals());
    let (primary_sumcheck_proof, r_primary_sumcheck, (_eq_eval, flag_evals, E_evals)) =
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

    // Create a single opening proof for the flag_evals and memory_evals
    let openings = PrimarySumcheckOpenings::prove_openings(
      batched_polys,
      commitment,
      &r_primary_sumcheck,
      vec![E_evals, flag_evals],
      &mut transcript,
      &mut random_tape,
    );

    let primary_sumcheck = PrimarySumcheck {
      sumcheck_proof: primary_sumcheck_proof,
      num_rounds,
      claimed_evaluation: sumcheck_claim,
      openings,
    };

    let r_fingerprints: Vec<G::ScalarField> =
      <Transcript as ProofTranscript<G>>::challenge_vector(transcript, b"challenge_r_hash", 2);
    let r_fingerprint = (&r_fingerprints[0], &r_fingerprints[1]);

    let memory_checking = MemoryCheckingProof::prove(
      &polynomials,
      r_fingerprint,
      &commitment.generators,
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
    proof: InstructionLookupsProof<G>,
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

    proof
      .primary_sumcheck
      .openings
      .verify_openings(proof.commitment, &r_primary_sumcheck);

    let r_mem_check =
      <Transcript as ProofTranscript<G>>::challenge_vector(transcript, b"challenge_r_hash", 2);

    proof.memory_checking.verify(
      &proof.commitment,
      &proof.commitment.generators,
      Self::memory_to_dimension_index,
      Self::evaluate_memory_mle,
      (&r_mem_check[0], &r_mem_check[1]),
      transcript,
    )?;

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

  /// Maps an index [0, num_memories) -> [0, num_subtables)
  fn memory_to_subtable_index(i: usize) -> usize {
    i / Self::C
  }

  /// Maps an index [0, num_memories) -> [0, subtable_dimensionality]
  fn memory_to_dimension_index(i: usize) -> usize {
    i % Self::C
  }

  fn protocol_name() -> &'static [u8] {
    b"Jolt instruction lookups"
  }
}

impl<F: PrimeField, G: CurveGroup<ScalarField = F>> BGPCInterpretable<F>
  for InstructionPolynomials<F, G>
{
  fn compute_leaves(
    &self,
    memory_index: usize,
    r_hash: (&F, &F),
  ) -> (
    DensePolynomial<F>,
    DensePolynomial<F>,
    DensePolynomial<F>,
    DensePolynomial<F>,
  ) {
    let (gamma, tau) = r_hash;
    let fingerprint = |a: F, v: F, t: F| -> F { t * gamma.square() + v * gamma + a - tau };

    let dimension_index = memory_index % self.C;

    let init_leaves = (0..self.memory_size)
      .map(|i| {
        fingerprint(
          F::from(i as u64),
          self.materialized_subtables[self.memory_to_subtable_map[memory_index]][i],
          F::zero(),
        )
      })
      .collect();
    let final_leaves = (0..self.memory_size)
      .map(|i| {
        fingerprint(
          F::from(i as u64),
          self.materialized_subtables[self.memory_to_subtable_map[memory_index]][i],
          self.final_cts[memory_index][i],
        )
      })
      .collect();
    let read_leaves = (0..self.num_ops)
      .map(|i| {
        fingerprint(
          self.dim[dimension_index][i],
          self.E_polys[memory_index][i],
          self.read_cts[memory_index][i],
        )
      })
      .collect();
    let write_leaves = (0..self.num_ops)
      .map(|i| {
        fingerprint(
          self.dim[dimension_index][i],
          self.E_polys[memory_index][i],
          self.read_cts[memory_index][i] + F::one(),
        )
      })
      .collect();

    (
      DensePolynomial::new(init_leaves),
      DensePolynomial::new(read_leaves),
      DensePolynomial::new(write_leaves),
      DensePolynomial::new(final_leaves),
    )
  }

  // TODO(sragss): Some if this logic is sharable.
  fn construct_batches(
    &self,
    r_hash: (&F, &F),
  ) -> (
    BatchedGrandProductCircuit<F>,
    BatchedGrandProductCircuit<F>,
    Vec<GPEvals<F>>,
  ) {
    // compute leaves for all the batches                     (shared)
    // convert the rw leaves to flagged leaves                (custom)
    // create GPCs for each of the leaves (&leaves)           (custom)
    // evaluate the GPCs                                      (shared)
    // construct 1x batch with flags, 1x batch without flags  (custom)

    let mut rw_circuits = Vec::with_capacity(self.num_memories * 2);
    let mut if_circuits = Vec::with_capacity(self.num_memories * 2);
    let mut gp_evals = Vec::with_capacity(self.num_memories);

    // Stores the initial fingerprinted values for read and write memories. GPC stores the upper portion of the tree after the fingerprints at the leaves
    // experience flagging (toggling based on the flag value at that leaf).
    let mut rw_fingerprints: Vec<DensePolynomial<F>> = Vec::with_capacity(self.num_memories * 2);
    for memory_index in 0..self.num_memories {
      let (init_fingerprints, read_fingerprints, write_fingerprints, final_fingerprints) =
        self.compute_leaves(memory_index, r_hash);

      let (mut read_leaves, mut write_leaves) =
        (read_fingerprints.evals(), write_fingerprints.evals());
      rw_fingerprints.push(read_fingerprints);
      rw_fingerprints.push(write_fingerprints);
      for leaf_index in 0..self.num_ops {
        // TODO(sragss): Would be faster if flags were non-FF repr
        let flag = self.subtable_flag_polys[self.memory_to_subtable_map[memory_index]][leaf_index];
        if flag == F::zero() {
          read_leaves[leaf_index] = F::one();
          write_leaves[leaf_index] = F::one();
        }
      }

      let (init_gpc, final_gpc) = (
        GrandProductCircuit::new(&init_fingerprints),
        GrandProductCircuit::new(&final_fingerprints),
      );
      let (read_gpc, write_gpc) = (
        GrandProductCircuit::new(&DensePolynomial::new(read_leaves)),
        GrandProductCircuit::new(&DensePolynomial::new(write_leaves)),
      );

      gp_evals.push(GPEvals::new(
        init_gpc.evaluate(),
        read_gpc.evaluate(),
        write_gpc.evaluate(),
        final_gpc.evaluate(),
      ));

      rw_circuits.push(read_gpc);
      rw_circuits.push(write_gpc);
      if_circuits.push(init_gpc);
      if_circuits.push(final_gpc);
    }

    // self.memory_to_subtable map has to be expanded because we've doubled the number of "grand products memorys": [read_0, write_0, ... read_NUM_MEMOREIS, write_NUM_MEMORIES]
    let expanded_flag_map: Vec<usize> = self
      .memory_to_subtable_map
      .iter()
      .flat_map(|subtable_index| [*subtable_index, *subtable_index])
      .collect();

    // Prover has access to subtable_flag_polys, which are uncommitted, but verifier can derive from instruction_flag commitments.
    let rw_batch = BatchedGrandProductCircuit::new_batch_flags(
      rw_circuits,
      self.subtable_flag_polys.clone(),
      expanded_flag_map,
      rw_fingerprints,
    );

    let if_batch = BatchedGrandProductCircuit::new_batch(if_circuits);

    (rw_batch, if_batch, gp_evals)
  }
}

// /// Read Only flags fingerprint Proof.
// #[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
// pub struct ROFlagsFingerprintProof<G: CurveGroup> {
//   eval_dim: Vec<G::ScalarField>,    // C-sized
//   eval_read: Vec<G::ScalarField>,   // NUM_MEMORIES-sized
//   eval_final: Vec<G::ScalarField>,  // NUM_MEMORIES-sized
//   eval_derefs: Vec<G::ScalarField>, // NUM_MEMORIES-sized
//   eval_flags: Vec<G::ScalarField>,  // NUM_INSTRUCTIONS-sized

//   proof_ops: CombinedTableEvalProof<G>,
//   proof_mem: CombinedTableEvalProof<G>,
//   proof_derefs: CombinedTableEvalProof<G>,
//   proof_flags: CombinedTableEvalProof<G>,

//   /// Maps memory_index to relevant instruction_flag indices.
//   /// Used by verifier to construct subtable_flag from instruction_flags.
//   memory_to_flag_indices: Vec<Vec<usize>>,
// }

// impl<G: CurveGroup> FingerprintStrategy<G> for ROFlagsFingerprintProof<G> {
//   type Polynomials = InstructionPolynomials<G::ScalarField>;
//   type Generators = SurgeCommitmentGenerators<G>;
//   type Commitments = SurgeCommitment<G>;

//   fn prove(
//     rand: (&Vec<G::ScalarField>, &Vec<G::ScalarField>),
//     polynomials: &Self::Polynomials,
//     generators: &Self::Generators,
//     transcript: &mut Transcript,
//     random_tape: &mut RandomTape<G>,
//   ) -> Self {
//     <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

//     let (rand_mem, rand_ops) = rand;

//     // decommit derefs at rand_ops
//     let eval_derefs: Vec<G::ScalarField> = (0..polynomials.num_memories)
//       .map(|i| polynomials.E_polys[i].evaluate(rand_ops))
//       .collect();
//     let proof_derefs = CombinedTableEvalProof::prove(
//       &polynomials.combined_E_poly,
//       eval_derefs.as_ref(),
//       rand_ops,
//       &generators.E_commitment_gens,
//       transcript,
//       random_tape,
//     );

//     // form a single decommitment using comm_comb_ops
//     let mut evals_ops: Vec<G::ScalarField> = Vec::new(); // moodlezoup: changed order of evals_ops

//     let eval_dim: Vec<G::ScalarField> = (0..polynomials.C)
//       .map(|i| polynomials.dim[i].evaluate(rand_ops))
//       .collect();
//     let eval_read: Vec<G::ScalarField> = (0..polynomials.num_memories)
//       .map(|i| polynomials.read_cts[i].evaluate(rand_ops))
//       .collect();

//     evals_ops.extend(eval_dim.clone());
//     evals_ops.extend(eval_read.clone());
//     evals_ops.resize(evals_ops.len().next_power_of_two(), G::ScalarField::zero());
//     let proof_ops = CombinedTableEvalProof::prove(
//       &polynomials.combined_dim_read_poly,
//       &evals_ops,
//       &rand_ops,
//       &generators.dim_read_commitment_gens,
//       transcript,
//       random_tape,
//     );

//     let eval_final: Vec<G::ScalarField> = (0..polynomials.num_memories)
//       .map(|i| polynomials.final_cts[i].evaluate(rand_mem))
//       .collect();

//     let proof_mem = CombinedTableEvalProof::prove(
//       &polynomials.combined_final_poly,
//       &eval_final,
//       &rand_mem,
//       &generators.final_commitment_gens,
//       transcript,
//       random_tape,
//     );

//     // TODO(sragss): flags combined with proof_ops?
//     let eval_flags: Vec<G::ScalarField> = (0..polynomials.num_instructions)
//       .map(|i| polynomials.instruction_flag_polys[i].evaluate(rand_ops))
//       .collect();
//     let proof_flags = CombinedTableEvalProof::prove(
//       &polynomials.combined_instruction_flag_poly,
//       &eval_flags,
//       &rand_ops,
//       &generators.flag_commitment_gens.as_ref().unwrap(),
//       transcript,
//       random_tape,
//     );

//     Self {
//       eval_dim,
//       eval_read,
//       eval_final,
//       eval_flags,
//       proof_ops,
//       proof_mem,
//       eval_derefs,
//       proof_derefs,
//       proof_flags,

//       memory_to_flag_indices: polynomials.memory_to_instructions_map.clone(), // TODO(sragss): Would be better as static
//     }
//   }

//   fn verify<F1: Fn(usize) -> usize, F2: Fn(usize, &[G::ScalarField]) -> G::ScalarField>(
//     &self,
//     rand: (&Vec<G::ScalarField>, &Vec<G::ScalarField>),
//     grand_product_claims: &[GPEvals<G::ScalarField>], // NUM_MEMORIES-sized
//     memory_to_dimension_index: F1,
//     evaluate_memory_mle: F2,
//     commitments: &Self::Commitments,
//     generators: &Self::Generators,
//     r_hash: &G::ScalarField,
//     r_multiset_check: &G::ScalarField,
//     transcript: &mut Transcript,
//   ) -> Result<(), ProofVerifyError> {
//     <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

//     let (rand_mem, rand_ops) = rand;

//     // verify derefs at rand_ops
//     // E_i(r_i''') ?= v_{E_i}
//     self.proof_derefs.verify(
//       rand_ops,
//       &self.eval_derefs,
//       &generators.E_commitment_gens,
//       &commitments.E_commitment,
//       transcript,
//     )?;

//     let mut evals_ops: Vec<G::ScalarField> = Vec::new();
//     evals_ops.extend(self.eval_dim.clone());
//     evals_ops.extend(self.eval_read.clone());
//     evals_ops.resize(evals_ops.len().next_power_of_two(), G::ScalarField::zero());

//     // dim_i(r_i''') ?= v_i
//     // read_i(r_i''') ?= v_{read_i}
//     self.proof_ops.verify(
//       rand_ops,
//       &evals_ops,
//       &generators.dim_read_commitment_gens,
//       &commitments.dim_read_commitment,
//       transcript,
//     )?;

//     // final_i(r_i'') ?= v_{final_i}
//     self.proof_mem.verify(
//       rand_mem,
//       &self.eval_final,
//       &generators.final_commitment_gens,
//       &commitments.final_commitment,
//       transcript,
//     )?;

//     self.proof_flags.verify(
//       rand_ops,
//       &self.eval_flags,
//       &generators.flag_commitment_gens.as_ref().unwrap(),
//       &commitments.instruction_flag_commitment.as_ref().unwrap(),
//       transcript,
//     )?;

//     // verify the claims from the product layer
//     let init_addr = IdentityPolynomial::new(rand_mem.len()).evaluate(rand_mem);
//     for memory_index in 0..grand_product_claims.len() {
//       let dimension_index = memory_to_dimension_index(memory_index);

//       // Compute the flag eval from opening proofs.
//       // We need the subtable_flags evaluation, which can be derived from instruction_flags, by summing
//       // the relevant indices from memory_to_flag_indices.
//       let instruction_flag_eval = self.memory_to_flag_indices[memory_index]
//         .iter()
//         .map(|flag_index| self.eval_flags[*flag_index])
//         .sum();

//       // Check ALPHA memories / lookup polys / grand products
//       // Only need 'C' indices / dimensions / read_timestamps / final_timestamps
//       Self::check_reed_solomon_fingerprints(
//         &grand_product_claims[memory_index],
//         &self.eval_derefs[memory_index],
//         &self.eval_dim[dimension_index],
//         &self.eval_read[memory_index],
//         &self.eval_final[memory_index],
//         &instruction_flag_eval,
//         &init_addr,
//         &evaluate_memory_mle(memory_index, rand_mem),
//         r_hash,
//         r_multiset_check,
//       )?;
//     }
//     Ok(())
//   }
// }

// impl<G: CurveGroup> ROFlagsFingerprintProof<G> {
//   /// Checks that the Reed-Solomon fingerprints of init, read, write, and final multisets
//   /// are as claimed by the final sumchecks of their respective grand product arguments.
//   ///
//   /// Params
//   /// - `claims`: Fingerprint values of the init, read, write, and final multisets, as
//   /// as claimed by their respective grand product arguments.
//   /// - `eval_deref`: The evaluation E_i(r'''_i).
//   /// - `eval_dim`: The evaluation dim_i(r'''_i).
//   /// - `eval_read`: The evaluation read_i(r'''_i).
//   /// - `eval_final`: The evaluation final_i(r''_i).
//   /// - `init_addr`: The MLE of the memory addresses, evaluated at r''_i.
//   /// - `init_memory`: The MLE of the initial memory values, evaluated at r''_i.
//   /// - `r_i`: One chunk of the evaluation point at which the Lasso commitment is being opened.
//   /// - `gamma`: Random value used to compute the Reed-Solomon fingerprint.
//   /// - `tau`: Random value used to compute the Reed-Solomon fingerprint.
//   fn check_reed_solomon_fingerprints(
//     claims: &GPEvals<G::ScalarField>,
//     eval_deref: &G::ScalarField,
//     eval_dim: &G::ScalarField,
//     eval_read: &G::ScalarField,
//     eval_final: &G::ScalarField,
//     eval_flag: &G::ScalarField,
//     init_addr: &G::ScalarField,
//     init_memory: &G::ScalarField,
//     gamma: &G::ScalarField,
//     tau: &G::ScalarField,
//   ) -> Result<(), ProofVerifyError> {
//     // Computes the Reed-Solomon fingerprint of the tuple (a, v, t)
//     let hash_func = |a: G::ScalarField, v: G::ScalarField, t: G::ScalarField| -> G::ScalarField {
//       t * gamma.square() + v * *gamma + a - tau
//     };
//     let hash_func_flag = |a: G::ScalarField,
//                           v: G::ScalarField,
//                           t: G::ScalarField,
//                           flag: G::ScalarField|
//      -> G::ScalarField {
//       flag * (t * gamma.square() + v * *gamma + a - tau) + G::ScalarField::one() - flag
//     };

//     let claim_init = claims.hash_init;
//     let claim_read = claims.hash_read;
//     let claim_write = claims.hash_write;
//     let claim_final = claims.hash_final;

//     // init
//     let hash_init = hash_func(*init_addr, *init_memory, G::ScalarField::zero());
//     assert_eq!(hash_init, claim_init); // verify the last claim of the `init` grand product sumcheck

//     // read
//     let hash_read = hash_func_flag(*eval_dim, *eval_deref, *eval_read, *eval_flag);
//     assert_eq!(hash_read, claim_read); // verify the last claim of the `read` grand product sumcheck

//     // write: shares addr, val with read
//     let eval_write = *eval_read + G::ScalarField::one();
//     let hash_write = hash_func_flag(*eval_dim, *eval_deref, eval_write, *eval_flag);
//     assert_eq!(hash_write, claim_write); // verify the last claim of the `write` grand product sumcheck

//     // final: shares addr and val with init
//     let eval_final_addr = init_addr;
//     let eval_final_val = init_memory;
//     let hash_final = hash_func(*eval_final_addr, *eval_final_val, *eval_final);
//     assert_eq!(hash_final, claim_final); // verify the last claim of the `final` grand product sumcheck

//     Ok(())
//   }

//   fn protocol_name() -> &'static [u8] {
//     b"Lasso HashLayerProof"
//   }
// }
