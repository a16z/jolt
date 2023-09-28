use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use merlin::Transcript;
use std::any::TypeId;
use strum::{EnumCount, IntoEnumIterator};

use crate::{
  jolt::{
    instruction::{JoltInstruction, Opcode},
    subtable::LassoSubtable,
  },
  lasso::memory_checking::MemoryCheckingProof,
  poly::{
    dense_mlpoly::{DensePolynomial, PolyCommitment, PolyCommitmentGens},
    eq_poly::EqPolynomial,
  },
  subprotocols::sumcheck::SumcheckInstanceProof,
  subtables::{CombinedTableCommitment, CombinedTableEvalProof},
  utils::{
    errors::ProofVerifyError,
    math::Math,
    random::RandomTape,
    transcript::{AppendToTranscript, ProofTranscript},
  },
};

/// All vectors to be committed in polynomial form.
pub struct PolynomialRepresentation<F: PrimeField> {
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
  /// NUM_INSTRUCTIONS sized, each polynomial of length 'm' (sparsity).
  ///
  /// Stored independently for use in sumchecking, combined into single DensePolynomial for commitment.
  pub flag_polys: Vec<DensePolynomial<F>>,

  // TODO(sragss): Storing both the polys and the combined polys may get expensive from a memory
  // perspective. Consier making an additional datastructure to handle the concept of combined polys
  // with a single reference to underlying evaluations.

  // TODO(moodlezoup): Consider pulling out combined polys into separate struct
  pub combined_flag_poly: DensePolynomial<F>,
  pub combined_dim_read_poly: DensePolynomial<F>,
  pub combined_final_poly: DensePolynomial<F>,
  pub combined_E_poly: DensePolynomial<F>,
}

impl<F: PrimeField> PolynomialRepresentation<F> {
  fn commit<G: CurveGroup<ScalarField = F>>(
    &self,
    generators: &SurgeCommitmentGenerators<G>,
  ) -> SurgeCommitment<G> {
    let (dim_read_commitment, _) = self
      .combined_dim_read_poly
      .commit(&generators.dim_read_commitment_gens, None);
    let (final_commitment, _) = self
      .combined_final_poly
      .commit(&generators.final_commitment_gens, None);
    let (flag_commitment, _) = self
      .combined_flag_poly
      .commit(&generators.flag_commitment_gens, None);
    let (E_commitment, _) = self
      .combined_E_poly
      .commit(&generators.E_commitment_gens, None);

    SurgeCommitment {
      dim_read_commitment,
      final_commitment,
      flag_commitment,
      E_commitment,
    }
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct SurgeCommitment<G: CurveGroup> {
  pub dim_read_commitment: PolyCommitment<G>,
  pub final_commitment: PolyCommitment<G>,
  pub flag_commitment: PolyCommitment<G>,
  pub E_commitment: PolyCommitment<G>,
}

/// Container for generators for polynomial commitments. These preallocate memory
/// and allow commitments to `DensePolynomials`.
pub struct SurgeCommitmentGenerators<G: CurveGroup> {
  pub dim_read_commitment_gens: PolyCommitmentGens<G>,
  pub final_commitment_gens: PolyCommitmentGens<G>,
  pub flag_commitment_gens: PolyCommitmentGens<G>,
  pub E_commitment_gens: PolyCommitmentGens<G>,
}

/// Proof of a single Jolt execution.
pub struct JoltProof<G: CurveGroup> {
  /// Commitments to all polynomials
  commitments: SurgeCommitment<G>,

  /// Primary collation sumcheck proof
  primary_sumcheck_proof: PrimarySumcheck<G>,

  /// Sparsity: Total number of operations. AKA 'm'.
  s: usize,
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PrimarySumcheck<G: CurveGroup> {
  proof: SumcheckInstanceProof<G::ScalarField>,
  claimed_evaluation: G::ScalarField,
  eval_derefs: Vec<G::ScalarField>,
  proof_derefs: CombinedTableEvalProof<G>,

  /// Evaluations of each of the `NUM_INSTRUCTIONS` flags polynomials at the random point.
  eval_flags: Vec<G::ScalarField>, // TODO: flag proof
}

pub trait Jolt<F: PrimeField, G: CurveGroup<ScalarField = F>> {
  type InstructionSet: JoltInstruction + Opcode + IntoEnumIterator + EnumCount;
  type Subtables: LassoSubtable<F> + IntoEnumIterator + EnumCount + From<TypeId> + Into<usize>;

  const C: usize;
  const M: usize;
  const NUM_SUBTABLES: usize = Self::Subtables::COUNT;
  const NUM_INSTRUCTIONS: usize = Self::InstructionSet::COUNT;
  const NUM_MEMORIES: usize = Self::C * Self::Subtables::COUNT;

  fn prove(ops: Vec<Self::InstructionSet>, r: Vec<F>, transcript: &mut Transcript) -> JoltProof<G> {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    let m = ops.len().next_power_of_two();

    let materialized_subtables: Vec<Vec<F>> = Self::materialize_subtables();
    let subtable_lookup_indices: Vec<Vec<usize>> = Self::subtable_lookup_indices(&ops);
    let polynomials: PolynomialRepresentation<F> =
      Self::polynomialize(&ops, &subtable_lookup_indices, &materialized_subtables);

    let commitment_generators = Self::commitment_generators(m);
    let commitments = polynomials.commit(&commitment_generators);

    commitments
      .E_commitment
      .append_to_transcript(b"comm_poly_row_col_ops_val", transcript);

    let eq = EqPolynomial::new(r.to_vec());
    let sumcheck_claim = Self::compute_sumcheck_claim(&ops, &polynomials.E_polys, &eq);

    // TODO(sragss): rm
    println!("Jolt::vm::prove() compute_sumcheck_claim result: {sumcheck_claim:?}");

    <Transcript as ProofTranscript<G>>::append_scalar(
      transcript,
      b"claim_eval_scalar_product",
      &sumcheck_claim,
    );

    let num_rounds = ops.len().log_2();
    let mut eq_poly = DensePolynomial::new(EqPolynomial::new(r).evals());
    // TODO(sragss): Deal with last parameter
    let (primary_sumcheck_instance_proof, r_primary_sumcheck, _) =
      SumcheckInstanceProof::prove_jolt::<G, Self, Transcript>(
        &F::zero(),
        num_rounds,
        &mut eq_poly,
        &mut polynomials.E_polys.clone(),
        &mut polynomials.flag_polys.clone(),
        Self::sumcheck_poly_degree(),
        transcript,
      );

    let mut random_tape = RandomTape::new(b"proof");

    let eval_derefs: Vec<G::ScalarField> = (0..Self::NUM_MEMORIES)
      .map(|i| polynomials.E_polys[i].evaluate(&r_primary_sumcheck))
      .collect();
    let proof_E = CombinedTableEvalProof::prove(
      &polynomials.combined_E_poly,
      &eval_derefs.to_vec(),
      &r_primary_sumcheck,
      &commitment_generators.E_commitment_gens, // TODO: Shouldn't this really be a PolyCommitment ?
      transcript,
      &mut random_tape,
    );

    let eval_flags: Vec<F> = polynomials
      .flag_polys
      .iter()
      .map(|flag_poly| flag_poly.evaluate(&r_primary_sumcheck))
      .collect();

    let primary_sumcheck_proof = PrimarySumcheck {
      proof: primary_sumcheck_instance_proof,
      claimed_evaluation: sumcheck_claim,
      eval_derefs,
      proof_derefs: proof_E,
      eval_flags,
    };
    // TODO(sragss): Joint flags proof

    // TODO: Prove memory checking.

    // let gamma = F::zero();
    // let tau = F::zero();
    // let mut random_tape = RandomTape::new(b"proof");
    // let commitment_generators = Self::commitment_generators(m);
    // MemoryCheckingProof::prove(
    //   polynomials,
    //   gamma,
    //   tau,
    //   &materialized_subtables,
    //   &commitment_generators,
    //   &mut transcript,
    //   &mut random_tape,
    // );

    JoltProof {
      commitments,
      primary_sumcheck_proof,
      s: ops.len(),
    }
  }

  fn verify(
    proof: JoltProof<G>,
    r_eq: &[G::ScalarField],
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    // TODO(sragss): rm
    println!("\n\nVerify");
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    proof
      .commitments
      .E_commitment
      .append_to_transcript(b"comm_poly_row_col_ops_val", transcript);

    <Transcript as ProofTranscript<G>>::append_scalar(
      transcript,
      b"claim_eval_scalar_product",
      &proof.primary_sumcheck_proof.claimed_evaluation,
    );

    let (claim_last, r_primary_sumcheck) =
      proof.primary_sumcheck_proof.proof.verify::<G, Transcript>(
        proof.primary_sumcheck_proof.claimed_evaluation,
        proof.s.log_2(),
        Self::sumcheck_poly_degree(),
        transcript,
      )?;

    // TODO(sragss): rm
    println!("r_primary_sumcheck {:?}", r_primary_sumcheck);

    // Verify that eq(r, r_z) * g(E_1(r_z) * ... * E_c(r_z)) = claim_last
    // TODO: Add in the flags
    let eq_eval = EqPolynomial::new(r_eq.to_vec()).evaluate(&r_primary_sumcheck);
    assert_eq!(
      eq_eval
        * Self::combine_lookups_flags(
          &proof.primary_sumcheck_proof.eval_derefs,
          &proof.primary_sumcheck_proof.eval_flags
        ),
      claim_last,
      "Primary sumcheck check failed."
    );

    Ok(())
  }

  fn commitment_generators(m: usize) -> SurgeCommitmentGenerators<G> {
    // dim_1, ... dim_C, read_1, ..., read_C
    // log_2(C * m + C * m)
    let num_vars_dim_read = (2 * Self::C * m).next_power_of_two().log_2();
    // final_1, ..., final_C
    // log_2(C * M)
    let num_vars_final = (Self::C * Self::M).next_power_of_two().log_2();
    // E_1, ..., E_alpha
    // log_2(alpha * m)
    let num_vars_E = (Self::NUM_MEMORIES * m).next_power_of_two().log_2();
    let num_vars_flag =
      m.next_power_of_two().log_2() + Self::NUM_INSTRUCTIONS.next_power_of_two().log_2();

    let dim_read_commitment_gens = PolyCommitmentGens::new(num_vars_dim_read, b"asdf");
    let final_commitment_gens = PolyCommitmentGens::new(num_vars_final, b"asd");
    let E_commitment_gens = PolyCommitmentGens::new(num_vars_E, b"asdf");
    let flag_commitment_gens = PolyCommitmentGens::new(num_vars_flag, b"1234");

    SurgeCommitmentGenerators {
      dim_read_commitment_gens,
      final_commitment_gens,
      flag_commitment_gens,
      E_commitment_gens,
    }
  }

  fn polynomialize(
    ops: &Vec<Self::InstructionSet>,
    subtable_lookup_indices: &Vec<Vec<usize>>,
    materialized_subtables: &Vec<Vec<F>>,
  ) -> PolynomialRepresentation<F> {
    let m: usize = ops.len().next_power_of_two();

    let mut opcodes: Vec<u8> = ops.iter().map(|op| op.to_opcode()).collect();
    opcodes.resize(m, 0);

    let mut dim: Vec<DensePolynomial<_>> = Vec::with_capacity(Self::C);
    let mut read_cts: Vec<DensePolynomial<_>> = Vec::with_capacity(Self::C);
    let mut final_cts: Vec<DensePolynomial<_>> = Vec::with_capacity(Self::C);
    let mut E_polys: Vec<DensePolynomial<_>> = Vec::with_capacity(Self::NUM_MEMORIES);

    for i in 0..Self::C {
      let access_sequence: &Vec<usize> = &subtable_lookup_indices[i];

      let mut final_cts_i = vec![0usize; Self::M];
      let mut read_cts_i = vec![0usize; m];

      for j in 0..m {
        let memory_address = access_sequence[j];
        debug_assert!(memory_address < Self::M);
        let counter = final_cts_i[memory_address];
        read_cts_i[j] = counter;
        final_cts_i[memory_address] = counter + 1;
      }

      dim.push(DensePolynomial::from_usize(access_sequence));
      read_cts.push(DensePolynomial::from_usize(&read_cts_i));
      final_cts.push(DensePolynomial::from_usize(&final_cts_i));
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

    let mut flag_bitvectors: Vec<Vec<usize>> = vec![vec![0usize; m]; Self::NUM_INSTRUCTIONS];
    for lookup_index in 0..m {
      let opcode_index = opcodes[lookup_index] as usize;
      flag_bitvectors[opcode_index][lookup_index] = 1;
    }
    let flag_polys: Vec<DensePolynomial<F>> = flag_bitvectors
      .iter()
      .map(|flag_bitvector| DensePolynomial::from_usize(&flag_bitvector))
      .collect();

    let dim_read_polys = [dim.as_slice(), read_cts.as_slice()].concat();

    let combined_flag_poly = DensePolynomial::merge(&flag_polys);
    let combined_dim_read_poly = DensePolynomial::merge(&dim_read_polys);
    let combined_final_poly = DensePolynomial::merge(&final_cts);
    let combined_E_poly = DensePolynomial::merge(&E_polys);

    PolynomialRepresentation {
      dim,
      read_cts,
      final_cts,
      flag_polys,
      E_polys,
      combined_flag_poly,
      combined_dim_read_poly,
      combined_final_poly,
      combined_E_poly,
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
      .subtables::<F>()
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

  /// Similar to combine_lookups but includes spaces in vals for 2 additional terms: eq, flags
  fn combine_lookups_plus_terms(vals: &[F]) -> F {
    assert_eq!(vals.len(), Self::NUM_MEMORIES + 2);

    let mut sum = F::zero();
    for instruction in Self::InstructionSet::iter() {
      let memory_indices = Self::instruction_to_memory_indices(&instruction);
      let mut filtered_operands = Vec::with_capacity(memory_indices.len());
      for index in memory_indices {
        filtered_operands.push(vals[index]);
      }
      sum += instruction.combine_lookups(&filtered_operands, Self::C, Self::M);
    }
    // eq(...) * flag(...) * g(...)
    vals[vals.len() - 2] * vals[vals.len() - 1] * sum
  }

  fn combine_lookups(vals: &[F]) -> F {
    assert_eq!(vals.len(), Self::NUM_MEMORIES);

    let mut sum = F::zero();
    for instruction in Self::InstructionSet::iter() {
      let memory_indices = Self::instruction_to_memory_indices(&instruction);
      let mut filtered_operands = Vec::with_capacity(memory_indices.len());
      for index in memory_indices {
        filtered_operands.push(vals[index]);
      }
      sum += instruction.combine_lookups(&filtered_operands, Self::C, Self::M);
    }

    sum
  }

  // TODO(sragss): Rename
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

  fn protocol_name() -> &'static [u8] {
    b"JoltVM_SparsePolynomialEvaluationProof"
  }
}

pub mod test_vm;
