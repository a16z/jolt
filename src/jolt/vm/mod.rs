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
  lasso::memory_checking::{HashLayerProof, MemoryCheckingProof},
  poly::{
    dense_mlpoly::{DensePolynomial, PolyCommitmentGens},
    eq_poly::EqPolynomial,
  },
  subprotocols::{
    combined_table_proof::{CombinedTableCommitment, CombinedTableEvalProof},
    grand_product::GrandProducts,
    sumcheck::SumcheckInstanceProof,
  },
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
  /// Polynomial encodings for flag polynomials for each instruction.
  /// If using a single instruction this will be empty.
  /// NUM_INSTRUCTIONS sized, each polynomial of length 'm' (sparsity).
  ///
  /// Stored independently for use in sumchecking, combined into single DensePolynomial for commitment.
  pub flag_polys: Option<Vec<DensePolynomial<F>>>,

  // TODO(sragss): Storing both the polys and the combined polys may get expensive from a memory
  // perspective. Consider making an additional datastructure to handle the concept of combined polys
  // with a single reference to underlying evaluations.

  // TODO(moodlezoup): Consider pulling out combined polys into separate struct
  pub combined_dim_read_poly: DensePolynomial<F>,
  pub combined_final_poly: DensePolynomial<F>,
  pub combined_E_poly: DensePolynomial<F>,
  pub combined_flag_poly: Option<DensePolynomial<F>>,

  pub num_memories: usize,
  pub C: usize,
  pub memory_size: usize,
  pub num_ops: usize,
  pub num_instructions: usize,
}

impl<F: PrimeField> PolynomialRepresentation<F> {
  fn commit<G: CurveGroup<ScalarField = F>>(
    &self,
    generators: &SurgeCommitmentGenerators<G>,
  ) -> SurgeCommitment<G> {
    let (dim_read_commitment, _) = self
      .combined_dim_read_poly
      .commit(&generators.dim_read_commitment_gens, None);
    let dim_read_commitment = CombinedTableCommitment::new(dim_read_commitment);
    let (final_commitment, _) = self
      .combined_final_poly
      .commit(&generators.final_commitment_gens, None);
    let final_commitment = CombinedTableCommitment::new(final_commitment);
    let (E_commitment, _) = self
      .combined_E_poly
      .commit(&generators.E_commitment_gens, None);
    let E_commitment = CombinedTableCommitment::new(E_commitment);
    let (flag_commitment, _) = self
      .combined_flag_poly
      .as_ref()
      .unwrap()
      .commit(generators.flag_commitment_gens.as_ref().unwrap(), None);
    let flag_commitment = CombinedTableCommitment::new(flag_commitment);

    SurgeCommitment {
      dim_read_commitment,
      final_commitment,
      E_commitment,
      flag_commitment: Some(flag_commitment),
    }
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct SurgeCommitment<G: CurveGroup> {
  pub dim_read_commitment: CombinedTableCommitment<G>,
  pub final_commitment: CombinedTableCommitment<G>,
  pub E_commitment: CombinedTableCommitment<G>,
  pub flag_commitment: Option<CombinedTableCommitment<G>>,
}

/// Container for generators for polynomial commitments. These preallocate memory
/// and allow commitments to `DensePolynomials`.
pub struct SurgeCommitmentGenerators<G: CurveGroup> {
  pub dim_read_commitment_gens: PolyCommitmentGens<G>,
  pub final_commitment_gens: PolyCommitmentGens<G>,
  pub E_commitment_gens: PolyCommitmentGens<G>,
  pub flag_commitment_gens: Option<PolyCommitmentGens<G>>,
}

/// Proof of a single Jolt execution.
pub struct JoltProof<G: CurveGroup> {
  /// Commitments to all polynomials
  commitments: SurgeCommitment<G>,

  /// Generators for commitments to polynomials
  commitment_generators: SurgeCommitmentGenerators<G>,

  /// Primary collation sumcheck proof
  primary_sumcheck_proof: PrimarySumcheck<G>,

  memory_checking_proof: MemoryCheckingProof<G, HashLayerProof<G>>,

  /// Sparsity: Total number of operations. AKA 'm'.
  s: usize,
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PrimarySumcheck<G: CurveGroup> {
  proof: SumcheckInstanceProof<G::ScalarField>,
  claimed_evaluation: G::ScalarField,
  memory_evals: Vec<G::ScalarField>,
  memory_proof: CombinedTableEvalProof<G>,

  /// Evaluations of each of the `NUM_INSTRUCTIONS` flags polynomials at the random point.
  flag_evals: Vec<G::ScalarField>,

  /// Combined proof of prior evals.
  flag_proof: CombinedTableEvalProof<G>,
}

pub enum MemoryOp {
  Read(u64, u64),       // (address, value)
  Write(u64, u64, u64), // (address, old_value, new_value)
}

pub struct MemoryTuple<F: PrimeField>(F, F, F);

pub trait Jolt<F: PrimeField, G: CurveGroup<ScalarField = F>> {
  type InstructionSet: JoltInstruction + Opcode + IntoEnumIterator + EnumCount;
  type Subtables: LassoSubtable<F> + IntoEnumIterator + EnumCount + From<TypeId> + Into<usize>;

  const MEMORY_OPS_PER_STEP: usize;
  const C: usize;
  const M: usize;
  const NUM_SUBTABLES: usize = Self::Subtables::COUNT;
  const NUM_INSTRUCTIONS: usize = Self::InstructionSet::COUNT;
  const NUM_MEMORIES: usize = Self::C * Self::Subtables::COUNT;

  fn prove() {
    // prove_program_code
    // prove_memory
    // prove_lookups
    // prove_r1cs
    unimplemented!("todo");
  }

  fn prove_program_code(
    program_code: &[u64],
    access_sequence: &[usize],
    code_size: usize,
    contiguous_reads_per_access: usize,
    r_mem_check: &(F, F),
    transcript: &mut Transcript,
  ) {
    let (gamma, tau) = r_mem_check;
    let hash_func = |a: &F, v: &F, t: &F| -> F { *t * gamma.square() + *v * *gamma + *a - tau };

    let m: usize = (access_sequence.len() * contiguous_reads_per_access).next_power_of_two();
    // TODO(moodlezoup): resize access_sequence?

    let mut read_addrs: Vec<usize> = Vec::with_capacity(m);
    let mut final_cts: Vec<usize> = vec![0; code_size];
    let mut read_cts: Vec<usize> = Vec::with_capacity(m);
    let mut read_values: Vec<u64> = Vec::with_capacity(m);

    for (j, code_address) in access_sequence.iter().enumerate() {
      debug_assert!(code_address + contiguous_reads_per_access <= code_size);
      debug_assert!(code_address % contiguous_reads_per_access == 0);

      for offset in 0..contiguous_reads_per_access {
        let addr = code_address + offset;
        let counter = final_cts[addr];
        read_addrs.push(addr);
        read_values.push(program_code[addr]);
        read_cts.push(counter);
        final_cts[addr] = counter + 1;
      }
    }

    let E_poly: DensePolynomial<F> = DensePolynomial::from_u64(&read_values);
    let dim: DensePolynomial<F> = DensePolynomial::from_usize(access_sequence);
    let read_cts: DensePolynomial<F> = DensePolynomial::from_usize(&read_cts);
    let final_cts: DensePolynomial<F> = DensePolynomial::from_usize(&final_cts);
    let init_values: DensePolynomial<F> = DensePolynomial::from_u64(program_code);
    unimplemented!("commit to these polynomials");

    let init_poly = DensePolynomial::new(
      (0..code_size)
        .map(|i| {
          // addr is given by i, init value is given by program_code, and ts = 0
          hash_func(&F::from(i as u64), &F::from(program_code[i]), &F::zero())
        })
        .collect::<Vec<F>>(),
    );

    let read_poly = DensePolynomial::new(
      (0..m)
        .map(|i| hash_func(&F::from(read_addrs[i] as u64), &E_poly[i], &read_cts[i]))
        .collect::<Vec<F>>(),
    );

    let write_poly = DensePolynomial::new(
      (0..m)
        .map(|i| {
          hash_func(
            &F::from(read_addrs[i] as u64),
            &F::from(read_values[i]),
            &(read_cts[i] + F::one()),
          )
        })
        .collect::<Vec<F>>(),
    );

    let final_poly = DensePolynomial::new(
      (0..code_size)
        .map(|i| {
          // addr is given by i, init value is given by program_code, and ts = 0
          hash_func(&F::from(i as u64), &F::from(program_code[i]), &final_cts[i])
        })
        .collect::<Vec<F>>(),
    );

    unimplemented!("memory checking");
  }

  fn prove_memory(
    memory_trace: Vec<[MemoryOp; Self::MEMORY_OPS_PER_STEP]>,
    memory_size: usize,
    r_mem_check: &(F, F),
    transcript: &mut Transcript,
  ) {
    let (gamma, tau) = r_mem_check;
    let hash_func = |a: &F, v: &F, t: &F| -> F { *t * gamma.square() + *v * *gamma + *a - tau };

    let m: usize = memory_trace.len().next_power_of_two();
    // TODO(moodlezoup): resize memory_trace

    let mut timestamp: u64 = 0;

    let mut read_set: Vec<(F, F, F)> = Vec::with_capacity(Self::MEMORY_OPS_PER_STEP * m);
    let mut write_set: Vec<(F, F, F)> = Vec::with_capacity(Self::MEMORY_OPS_PER_STEP * m);
    let mut final_set: Vec<(F, F, F)> = (0..memory_size)
      .map(|i| (F::from(i as u64), F::zero(), F::zero()))
      .collect();

    for memory_access in memory_trace {
      for memory_op in memory_access {
        match memory_op {
          MemoryOp::Read(a, v) => {
            read_set.push((F::from(a), F::from(v), F::from(timestamp)));
            write_set.push((F::from(a), F::from(v), F::from(timestamp + 1)));
            final_set[a as usize] = (F::from(a), F::from(v), F::from(timestamp + 1));
          }
          MemoryOp::Write(a, v_old, v_new) => {
            read_set.push((F::from(a), F::from(v_old), F::from(timestamp)));
            write_set.push((F::from(a), F::from(v_new), F::from(timestamp + 1)));
            final_set[a as usize] = (F::from(a), F::from(v_new), F::from(timestamp + 1));
          }
        }
      }
      timestamp += 1;
    }

    let init_poly = DensePolynomial::new(
      (0..memory_size)
        .map(|i| {
          // addr is given by i, init value is 0, and ts = 0
          hash_func(&F::from(i as u64), &F::zero(), &F::zero())
        })
        .collect::<Vec<F>>(),
    );
    let read_poly = DensePolynomial::new(
      read_set
        .iter()
        .map(|(a, v, t)| hash_func(a, v, t))
        .collect::<Vec<F>>(),
    );
    let write_poly = DensePolynomial::new(
      write_set
        .iter()
        .map(|(a, v, t)| hash_func(a, v, t))
        .collect::<Vec<F>>(),
    );
    let final_poly = DensePolynomial::new(
      final_set
        .iter()
        .map(|(a, v, t)| hash_func(a, v, t))
        .collect::<Vec<F>>(),
    );

    // Memory checking
    // Lasso range cheeck on read timestamps to enforce each timestamp read at step i is less than i
    unimplemented!("todo");
  }

  fn prove_lookups(
    ops: Vec<Self::InstructionSet>,
    r: Vec<F>,
    transcript: &mut Transcript,
  ) -> JoltProof<G> {
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
    let (primary_sumcheck_instance_proof, r_primary_sumcheck, (_eq_eval, flag_evals, memory_evals)) =
      SumcheckInstanceProof::prove_jolt::<G, Self, Transcript>(
        &F::zero(),
        num_rounds,
        &mut eq_poly,
        &mut polynomials.E_polys.clone(),
        &mut polynomials.flag_polys.as_ref().unwrap().clone(),
        Self::sumcheck_poly_degree(),
        transcript,
      );

    let mut random_tape = RandomTape::new(b"proof");

    // Create a single opening proof for the flag_evals and memory_evals
    let flag_proof = CombinedTableEvalProof::prove(
      &polynomials.combined_flag_poly.as_ref().unwrap(),
      &flag_evals.to_vec(),
      &r_primary_sumcheck,
      &commitment_generators.flag_commitment_gens.as_ref().unwrap(),
      transcript,
      &mut random_tape,
    );
    let memory_proof = CombinedTableEvalProof::prove(
      &polynomials.combined_E_poly,
      &memory_evals.to_vec(),
      &r_primary_sumcheck,
      &commitment_generators.E_commitment_gens,
      transcript,
      &mut random_tape,
    );

    let primary_sumcheck_proof = PrimarySumcheck {
      proof: primary_sumcheck_instance_proof,
      claimed_evaluation: sumcheck_claim,
      memory_evals,
      memory_proof,
      flag_evals,
      flag_proof,
    };

    let r_mem_check =
      <Transcript as ProofTranscript<G>>::challenge_vector(transcript, b"challenge_r_hash", 2);
    let gamma = r_mem_check[0];
    let tau = r_mem_check[1];
    let commitment_generators = Self::commitment_generators(m);

    let mut flag_map = vec![vec![false; ops.len().next_power_of_two()]; Self::NUM_SUBTABLES];
    let subtable_map = Self::subtable_map();
    for (instruction_index, instruction) in ops.iter().enumerate() {
      let subtable_indices = &subtable_map[instruction.to_opcode() as usize];
      for subtable_index in subtable_indices {
        flag_map[*subtable_index][instruction_index] = true;
      }
    }

    let mut grand_products: Vec<GrandProducts<F>> = (0..Self::NUM_MEMORIES)
      .map(|memory_index| {
        GrandProducts::<F>::new_read_only_with_flags(
          &materialized_subtables[Self::memory_to_subtable_index(memory_index)],
          &polynomials.dim[Self::memory_to_dimension_index(memory_index)],
          &subtable_lookup_indices[Self::memory_to_dimension_index(memory_index)],
          &polynomials.read_cts[memory_index],
          &polynomials.final_cts[memory_index],
          &flag_map[Self::memory_to_subtable_index(memory_index)],
          &(gamma, tau),
        )
      })
      .collect();

    let memory_checking_proof = MemoryCheckingProof::prove(
      &polynomials,
      grand_products,
      &commitment_generators,
      transcript,
      &mut random_tape,
    );

    JoltProof {
      commitments,
      commitment_generators,
      primary_sumcheck_proof,
      memory_checking_proof,
      s: ops.len(),
    }
  }

  fn prove_r1cs() {
    unimplemented!("todo")
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

    // Verify that eq(r, r_z) * [f_1(r_z) * g(E_1(r_z)) + ... + f_F(r_z) * E_F(r_z))] = claim_last
    let eq_eval = EqPolynomial::new(r_eq.to_vec()).evaluate(&r_primary_sumcheck);
    assert_eq!(
      eq_eval
        * Self::combine_lookups_flags(
          &proof.primary_sumcheck_proof.memory_evals,
          &proof.primary_sumcheck_proof.flag_evals
        ),
      claim_last,
      "Primary sumcheck check failed."
    );

    // Verify joint opening proofs to flag polynomials
    proof.primary_sumcheck_proof.flag_proof.verify(
      &r_primary_sumcheck,
      &proof.primary_sumcheck_proof.flag_evals,
      &proof
        .commitment_generators
        .flag_commitment_gens
        .as_ref()
        .unwrap(),
      &proof.commitments.flag_commitment.as_ref().unwrap(),
      transcript,
    )?;

    // Verify joint opening proofs to E polynomials
    proof.primary_sumcheck_proof.memory_proof.verify(
      &r_primary_sumcheck,
      &proof.primary_sumcheck_proof.memory_evals,
      &proof.commitment_generators.E_commitment_gens,
      &proof.commitments.E_commitment,
      transcript,
    )?;

    let r_mem_check =
      <Transcript as ProofTranscript<G>>::challenge_vector(transcript, b"challenge_r_hash", 2);

    proof.memory_checking_proof.verify(
      &proof.commitments,
      &proof.commitment_generators,
      Self::memory_to_dimension_index,
      Self::evaluate_memory_mle,
      &(r_mem_check[0], r_mem_check[1]),
      transcript,
    )?;

    Ok(())
  }

  fn commitment_generators(m: usize) -> SurgeCommitmentGenerators<G> {
    // dim_1, ... dim_C, read_1, ..., read_{NUM_MEMORIES}
    // log_2(C * m + NUM_MEMORIES * m)
    let num_vars_dim_read = (Self::C * m + Self::NUM_MEMORIES * m).next_power_of_two().log_2();
    // final_1, ..., final_{NUM_MEMORIES}
    // log_2(NUM_MEMORIES * M)
    let num_vars_final = (Self::NUM_MEMORIES * Self::M).next_power_of_two().log_2();
    // E_1, ..., E_{NUM_MEMORIES}
    // log_2(NUM_MEMORIES * m)
    let num_vars_E = (Self::NUM_MEMORIES * m).next_power_of_two().log_2();
    let num_vars_flag =
      m.next_power_of_two().log_2() + Self::NUM_INSTRUCTIONS.next_power_of_two().log_2();

    let dim_read_commitment_gens =
      PolyCommitmentGens::new(num_vars_dim_read, b"dim_read_commitment");
    let final_commitment_gens = PolyCommitmentGens::new(num_vars_final, b"final_commitment");
    let E_commitment_gens = PolyCommitmentGens::new(num_vars_E, b"memory_evals_commitment");
    let flag_commitment_gens = PolyCommitmentGens::new(num_vars_flag, b"flag_evals_commitment");

    SurgeCommitmentGenerators {
      dim_read_commitment_gens,
      final_commitment_gens,
      E_commitment_gens,
      flag_commitment_gens: Some(flag_commitment_gens),
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
    let mut read_cts: Vec<DensePolynomial<_>> = Vec::with_capacity(Self::NUM_MEMORIES);
    let mut final_cts: Vec<DensePolynomial<_>> = Vec::with_capacity(Self::NUM_MEMORIES);
    let mut E_polys: Vec<DensePolynomial<_>> = Vec::with_capacity(Self::NUM_MEMORIES);

    let subtable_map = Self::subtable_map();
    for memory_index in 0..Self::NUM_MEMORIES {
      let access_sequence: &Vec<usize> =
        &subtable_lookup_indices[Self::memory_to_dimension_index(memory_index)];

      let mut final_cts_i = vec![0usize; Self::M];
      let mut read_cts_i = vec![0usize; m];

      for op_index in 0..m {
        let memory_address = access_sequence[op_index];
        debug_assert!(memory_address < Self::M);

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
      flag_polys: Some(flag_polys),
      E_polys,
      combined_dim_read_poly,
      combined_final_poly,
      combined_E_poly,
      combined_flag_poly: Some(combined_flag_poly),
      num_memories: Self::NUM_MEMORIES,
      C: Self::C,
      memory_size: Self::M,
      num_ops: m, // TODO(sragss): should this be real num_ops or padded?
      num_instructions: Self::NUM_INSTRUCTIONS,
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

  /// Computes which subtables indices are active for a given instruction.
  /// vec[instruction_index] = [subtable_id_a, subtable_id_b, ...]
  fn subtable_map() -> Vec<Vec<usize>> {
    return Self::InstructionSet::iter()
      .map(|instruction| {
        // TODO(sragss): Box<dyn SubtableTrait>.into() should work via additional functionality on the trait .
        let instruction_subtable_ids: Vec<usize> = instruction
          .subtables::<F>()
          .iter()
          .map(|subtable| Self::Subtables::from(subtable.subtable_id()).into())
          .collect();

        instruction_subtable_ids
      })
      .collect();
  }

  fn evaluate_memory_mle(memory_index: usize, point: &[F]) -> F {
    let subtable = Self::Subtables::iter()
      .nth(Self::memory_to_subtable_index(memory_index))
      .expect("should exist");
    subtable.evaluate_mle(point)
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
    b"JoltVM_SparsePolynomialEvaluationProof"
  }
}

pub mod test_vm;
