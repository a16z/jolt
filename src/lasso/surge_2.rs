use std::marker::PhantomData;

use ark_ec::CurveGroup;
use ark_ff::{PrimeField, Field};
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use ark_std::{Zero, One};
use merlin::Transcript;

use crate::{poly::{dense_mlpoly::{DensePolynomial, PolyCommitmentGens}, identity_poly::IdentityPolynomial, eq_poly::EqPolynomial}, subprotocols::{grand_product::BGPCInterpretable, combined_table_proof::{CombinedTableEvalProof, CombinedTableCommitment}, sumcheck::SumcheckInstanceProof}, lasso::fingerprint_strategy::MemBatchInfo, utils::{errors::ProofVerifyError, transcript::ProofTranscript, math::Math, random::RandomTape}, jolt::instruction::JoltInstruction};

use super::{gp_evals::GPEvals, fingerprint_strategy::FingerprintStrategy, memory_checking::MemoryCheckingProof};



pub struct SurgePolys<F: PrimeField> {
    pub dim_i_usize: Vec<Vec<usize>>,
    pub dim_i: Vec<DensePolynomial<F>>,
    pub read_i: Vec<DensePolynomial<F>>,
    pub final_i: Vec<DensePolynomial<F>>,
    pub E_poly_i: Vec<DensePolynomial<F>>,
  
    pub combined_dim_read_polys: DensePolynomial<F>,
    pub combined_final_polys: DensePolynomial<F>,
    pub combined_E_polys: DensePolynomial<F>,

    pub materialized_subtables: Vec<Vec<F>>,
  
    pub num_ops: usize, 
    pub m: usize,          // memory size
    pub log_m: usize,      // log memory size
    pub dimensions: usize, // C
    pub alpha: usize       // num_memories
}

impl<F: PrimeField> MemBatchInfo for SurgePolys<F> {
    fn ops_size(&self) -> usize {
      self.num_ops
    }

    fn mem_size(&self) -> usize {
      self.m
    }

    fn num_memories(&self) -> usize {
      self.alpha
    }
}

impl<F: PrimeField> SurgePolys<F> {
    fn commit<G: CurveGroup<ScalarField = F>>(
      &self,
      generators: &SurgeCommitmentGens<G>
    ) -> SurgeCommitment<G> {
      let (dim_read_commitment, _)  = self.combined_dim_read_polys.commit(&generators.dim_read_commitment_gens, None);
      let (final_commitment, _)  = self.combined_final_polys.commit(&generators.final_commitment_gens, None);
      let (E_commitment, _) = self.combined_E_polys.commit(&generators.E_commitment_gens, None);

      SurgeCommitment { 
        dim_read_commitment: CombinedTableCommitment::new(dim_read_commitment), 
        final_commitment: CombinedTableCommitment::new(final_commitment), 
        E_commitment: CombinedTableCommitment::new(E_commitment)
      }
    }
}

pub struct SurgeCommitment<G: CurveGroup> {
    pub dim_read_commitment: CombinedTableCommitment<G>,
    pub final_commitment: CombinedTableCommitment<G>,
    pub E_commitment: CombinedTableCommitment<G>,
}

pub struct SurgeCommitmentGens<G: CurveGroup> {
  pub dim_read_commitment_gens: PolyCommitmentGens<G>,
  pub final_commitment_gens: PolyCommitmentGens<G>,
  pub E_commitment_gens: PolyCommitmentGens<G>
}

impl<G: CurveGroup> SurgeCommitmentGens<G> {
  pub fn new(dimensions: usize, memory_size: usize, num_ops: usize, alpha: usize) -> Self {
    // dim_1, ... dim_C, read_1, ... read_C
    let num_vars_dim_read = (2 * num_ops * dimensions).next_power_of_two().log_2();

    // final_1, ... final_C
    let num_vars_final = (memory_size * dimensions).next_power_of_two().log_2();

    // E_1, ... E_alpha
    let num_vars_E = (alpha * num_ops).next_power_of_two().log_2();

    let dim_read_commitment_gens = PolyCommitmentGens::new(num_vars_dim_read, b"dim_read_commitment");
    let final_commitment_gens = PolyCommitmentGens::new(num_vars_final, b"final_commitment");
    let E_commitment_gens = PolyCommitmentGens::new(num_vars_E, b"memory_evals_commitment");

    SurgeCommitmentGens {
      dim_read_commitment_gens,
      final_commitment_gens,
      E_commitment_gens
    }
  }
}

impl<F: PrimeField> BGPCInterpretable<F> for SurgePolys<F> {
    fn a_ops(&self, memory_index: usize, leaf_index: usize) -> F {
      debug_assert!(memory_index < self.alpha);
      debug_assert!(leaf_index < self.num_ops);

      let dimension_index = memory_index % self.dimensions;
      self.dim_i[dimension_index][leaf_index]
    }

    fn v_mem(&self, memory_index: usize, leaf_index: usize) -> F {
      debug_assert!(memory_index < self.alpha);
      debug_assert!(leaf_index < self.mem_size());

      let subtable_index = memory_index / self.dimensions;
      self.materialized_subtables[subtable_index][leaf_index]
    }

    fn v_ops(&self, memory_index: usize, leaf_index: usize) -> F {
      debug_assert!(memory_index < self.alpha);
      debug_assert!(leaf_index < self.num_ops);

      let dimension_index = memory_index % self.dimensions;
      self.E_poly_i[dimension_index][leaf_index]
    }

    fn t_read(&self, memory_index: usize, leaf_index: usize) -> F {
      debug_assert!(memory_index < self.alpha);
      debug_assert!(leaf_index < self.num_ops);

      let dimension_index = memory_index % self.dimensions;
      self.read_i[dimension_index][leaf_index]
    }

    fn t_final(&self, memory_index: usize, leaf_index: usize) -> F {
      debug_assert!(memory_index < self.alpha);
      debug_assert!(leaf_index < self.mem_size());

      let dimension_index = memory_index % self.dimensions;
      self.final_i[dimension_index][leaf_index]
    }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct SurgeFingerprintProof<G: CurveGroup> {
  eval_dim: Vec<G::ScalarField>,    // C-sized
  eval_read: Vec<G::ScalarField>,   // C-sized
  eval_final: Vec<G::ScalarField>,  // C-sized
  eval_derefs: Vec<G::ScalarField>, // NUM_MEMORIES-sized

  proof_ops: CombinedTableEvalProof<G>,
  proof_mem: CombinedTableEvalProof<G>,
  proof_derefs: CombinedTableEvalProof<G>,
}

impl<G: CurveGroup> FingerprintStrategy<G> for SurgeFingerprintProof<G> {
  type Polynomials = SurgePolys<G::ScalarField>;
  type Generators = SurgeCommitmentGens<G>;
  type Commitments = SurgeCommitment<G>;

  fn prove(
    rand: (&Vec<<G>::ScalarField>, &Vec<<G>::ScalarField>),
    polynomials: &Self::Polynomials,
    generators: &Self::Generators,
    transcript: &mut merlin::Transcript,
    random_tape: &mut crate::utils::random::RandomTape<G>,
  ) -> Self {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    let (rand_mem, rand_ops) = rand;

    // decommit derefs at rand_ops
    let eval_derefs: Vec<G::ScalarField> = (0..polynomials.num_memories())
      .map(|i| polynomials.E_poly_i[i].evaluate(rand_ops))
      .collect();
    let proof_derefs = CombinedTableEvalProof::prove(
      &polynomials.combined_E_polys,
      eval_derefs.as_ref(),
      rand_ops,
      &generators.E_commitment_gens,
      transcript,
      random_tape,
    );

    // form a single decommitment using comm_comb_ops
    let mut evals_ops: Vec<G::ScalarField> = Vec::new();

    let eval_dim: Vec<G::ScalarField> = (0..polynomials.dimensions)
      .map(|i| polynomials.dim_i[i].evaluate(rand_ops))
      .collect();
    let eval_read: Vec<G::ScalarField> = (0..polynomials.dimensions)
      .map(|i| polynomials.read_i[i].evaluate(rand_ops))
      .collect();
    let eval_final: Vec<G::ScalarField> = (0..polynomials.dimensions)
      .map(|i| polynomials.final_i[i].evaluate(rand_mem))
      .collect();

    evals_ops.extend(eval_dim.clone());
    evals_ops.extend(eval_read.clone());
    evals_ops.resize(evals_ops.len().next_power_of_two(), G::ScalarField::zero());
    let proof_ops = CombinedTableEvalProof::prove(
      &polynomials.combined_dim_read_polys,
      &evals_ops,
      &rand_ops,
      &generators.dim_read_commitment_gens,
      transcript,
      random_tape,
    );

    let proof_mem = CombinedTableEvalProof::prove(
      &polynomials.combined_final_polys,
      &eval_final,
      &rand_mem,
      &generators.final_commitment_gens,
      transcript,
      random_tape,
    );

    Self {
      eval_dim,
      eval_read,
      eval_final,
      proof_ops,
      proof_mem,
      eval_derefs,
      proof_derefs,
    }
  }

  fn verify<F1: Fn(usize) -> usize, F2: Fn(usize, &[<G>::ScalarField]) -> <G>::ScalarField>(
    &self,
    rand: (&Vec<<G>::ScalarField>, &Vec<<G>::ScalarField>),
    grand_product_claims: &[super::gp_evals::GPEvals<<G>::ScalarField>], // NUM_MEMORIES-sized
    memory_to_dimension_index: F1,
    evaluate_memory_mle: F2,
    commitments: &Self::Commitments,
    generators: &Self::Generators,
    r_hash: &<G>::ScalarField,
    r_multiset_check: &<G>::ScalarField,
    transcript: &mut merlin::Transcript,
  ) -> Result<(), crate::utils::errors::ProofVerifyError> {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    let (rand_mem, rand_ops) = rand;

    // verify derefs at rand_ops
    // E_i(r_i''') ?= v_{E_i}
    self.proof_derefs.verify(
      rand_ops,
      &self.eval_derefs,
      &generators.E_commitment_gens,
      &commitments.E_commitment,
      transcript,
    )?;

    let mut evals_ops: Vec<G::ScalarField> = Vec::new();
    evals_ops.extend(self.eval_dim.clone());
    evals_ops.extend(self.eval_read.clone());
    evals_ops.resize(evals_ops.len().next_power_of_two(), G::ScalarField::zero());

    // dim_i(r_i''') ?= v_i
    // read_i(r_i''') ?= v_{read_i}
    self.proof_ops.verify(
      rand_ops,
      &evals_ops,
      &generators.dim_read_commitment_gens,
      &commitments.dim_read_commitment,
      transcript,
    )?;

    // final_i(r_i'') ?= v_{final_i}
    self.proof_mem.verify(
      rand_mem,
      &self.eval_final,
      &generators.final_commitment_gens,
      &commitments.final_commitment,
      transcript,
    )?;

    // verify the claims from the product layer
    let init_addr = IdentityPolynomial::new(rand_mem.len()).evaluate(rand_mem);
    for memory_index in 0..grand_product_claims.len() {
      let dimension_index = memory_to_dimension_index(memory_index);
      // Check ALPHA memories / lookup polys / grand products
      // Only need 'C' indices / dimensions / read_timestamps / final_timestamps
      Self::check_reed_solomon_fingerprints(
        &grand_product_claims[memory_index],
        &self.eval_derefs[memory_index],
        &self.eval_dim[dimension_index],
        &self.eval_read[dimension_index],
        &self.eval_final[dimension_index],
        &init_addr,
        &evaluate_memory_mle(memory_index, rand_mem),
        r_hash,
        r_multiset_check,
      )?;
    }
    Ok(())
  }
}

impl<G: CurveGroup> SurgeFingerprintProof<G> {
  fn check_reed_solomon_fingerprints(
    claims: &GPEvals<G::ScalarField>,
    eval_deref: &G::ScalarField,
    eval_dim: &G::ScalarField,
    eval_read: &G::ScalarField,
    eval_final: &G::ScalarField,
    init_addr: &G::ScalarField,
    init_memory: &G::ScalarField,
    gamma: &G::ScalarField,
    tau: &G::ScalarField,
  ) -> Result<(), ProofVerifyError> {
    // Computes the Reed-Solomon fingerprint of the tuple (a, v, t)
    // Note: this differs from the Lasso paper a little:
    // (t * gamma^2 + v * gamma + a) instead of (a * gamma^2 + v * gamma + t)
    let hash_func = |a: G::ScalarField, v: G::ScalarField, t: G::ScalarField| -> G::ScalarField {
      t * gamma.square() + v * *gamma + a - tau
    };

    // init
    let hash_init = hash_func(*init_addr, *init_memory, G::ScalarField::zero());
    assert_eq!(hash_init, claims.hash_init); // verify the last claim of the `init` grand product sumcheck

    // read
    let hash_read = hash_func(*eval_dim, *eval_deref, *eval_read);
    assert_eq!(hash_read, claims.hash_read); // verify the last claim of the `read` grand product sumcheck

    // write: shares addr, val with read
    let eval_write = *eval_read + G::ScalarField::one();
    let hash_write = hash_func(*eval_dim, *eval_deref, eval_write);
    assert_eq!(hash_write, claims.hash_write); // verify the last claim of the `write` grand product sumcheck

    // final: shares addr and val with init
    let eval_final_addr = init_addr;
    let eval_final_val = init_memory;
    let hash_final = hash_func(*eval_final_addr, *eval_final_val, *eval_final);
    assert_eq!(hash_final, claims.hash_final); // verify the last claim of the `final` grand product sumcheck

    Ok(())
  }

  fn protocol_name() -> &'static [u8] {
    b"Surge FingprintProof"
  }
}

pub struct SurgePrimarySumcheck<G: CurveGroup> {
  proof: SumcheckInstanceProof<G::ScalarField>,
  claimed_evaluation: G::ScalarField,
  eval_E: Vec<G::ScalarField>,
  proof_E: CombinedTableEvalProof<G>
}

// #[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct SurgeProof<G: CurveGroup, I: JoltInstruction + Default + std::marker::Sync> { // TODO(sragss): JoltInstruction trait add Default
  generators: SurgeCommitmentGens<G>,
  commitments: SurgeCommitment<G>,
  primary_sumcheck: SurgePrimarySumcheck<G>,
  memory_check: MemoryCheckingProof<G, SurgeFingerprintProof<G>>,

  num_ops: usize,
  C: usize,
  M: usize,

  _marker: PhantomData<I>
}

impl<G: CurveGroup, I: JoltInstruction + Default + std::marker::Sync> SurgeProof<G, I> {
  pub fn prove(
    ops: Vec<I>, 
    C: usize, // TODO(sragss): move to const generic?
    M: usize, // TODO(sragss): move to const generic or instruction?
    transcript: &mut Transcript) -> Self {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    let instruction = I::default();
    instruction.g_poly_degree(C);

    let num_ops: usize = ops.len();
    let log_num_ops: usize = num_ops.log_2();
    let num_memories: usize = instruction.subtables::<G::ScalarField>().len() * C; // alpha // TODO(sragss): Could move to JoltInstruction trait
    let memory_size: usize = M; // M

    let generators: SurgeCommitmentGens<G> = SurgeCommitmentGens::new(C, memory_size, num_ops, num_memories);
    let polynomials: SurgePolys<G::ScalarField> = Self::construct_polys(&ops, C, M);
    let commitments: SurgeCommitment<G> = polynomials.commit(&generators);
    let mut random_tape = RandomTape::new(b"proof");

    // TODO(sragss): Commit some of this stuff to transcript?

    // Primary sumcheck
    let r_primary_sumcheck = <Transcript as ProofTranscript<G>>::challenge_vector(transcript, b"primary_sumcheck", log_num_ops);
    let eq = DensePolynomial::new(EqPolynomial::new(r_primary_sumcheck.to_vec()).evals());
    let claimed_eval: G::ScalarField = Self::compute_primary_sumcheck_claim(&polynomials, &eq);

    <Transcript as ProofTranscript<G>>::append_scalar(transcript, b"claim_eval_scalar_product", &claimed_eval);
    let mut combined_sumcheck_polys = polynomials.E_poly_i.clone();
    combined_sumcheck_polys.push(eq);

    let combine_lookups_eq = | vals: &[G::ScalarField] | -> G::ScalarField {
      let vals_no_eq: &[G::ScalarField] = &vals[0..(vals.len() - 1)];
      let eq = vals[vals.len() - 1];
      instruction.combine_lookups(vals_no_eq, C, M) * eq
    };

    let (primary_sumcheck_proof, r_z, _) =
      SumcheckInstanceProof::<G::ScalarField>::prove_arbitrary::<_, G, Transcript>(
        &claimed_eval,
        log_num_ops,
        &mut combined_sumcheck_polys,
        combine_lookups_eq,
        instruction.g_poly_degree(C) + 1, // combined degree + eq term
        transcript,
      );

    let eval_E: Vec<G::ScalarField> = (0..num_memories)
      .map(|i| polynomials.E_poly_i[i].evaluate(&r_z))
      .collect();
    let proof_E = CombinedTableEvalProof::prove(
      &polynomials.combined_E_polys,
      &eval_E,
      &r_z,
      &generators.E_commitment_gens,
      transcript,
      &mut random_tape
    );
    

    let primary_sumcheck = SurgePrimarySumcheck {
      proof: primary_sumcheck_proof,
      claimed_evaluation: claimed_eval,
      eval_E,
      proof_E
    };

    let r_fingerprints: Vec<G::ScalarField> = <Transcript as ProofTranscript<G>>::challenge_vector(transcript, b"challenge_r_hash", 2);
    let r_fingerprint = (&r_fingerprints[0], &r_fingerprints[1]);

    let memory_check = MemoryCheckingProof::prove(
      &polynomials,
      r_fingerprint,
      &generators,
      transcript,
      &mut random_tape
    );

    SurgeProof { 
      generators,
      commitments, 
      primary_sumcheck,
      memory_check,

      num_ops,
      C,
      M,

      _marker: PhantomData
    }

  }

  pub fn verify(
    &self,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());
    let instruction = I::default();

    let log_num_ops = ark_std::log2(self.num_ops) as usize;
    let r_primary_sumcheck = <Transcript as ProofTranscript<G>>::challenge_vector(transcript, b"primary_sumcheck", log_num_ops);

    <Transcript as ProofTranscript<G>>::append_scalar(transcript, b"claim_eval_scalar_product", &self.primary_sumcheck.claimed_evaluation);
    let primary_sumcheck_poly_degree = instruction.g_poly_degree(self.C) + 1;
    let (claim_last, r_z) = self.primary_sumcheck.proof.verify::<G, Transcript>(
      self.primary_sumcheck.claimed_evaluation,
      log_num_ops,
      primary_sumcheck_poly_degree,
      transcript,
    )?;

    let eq_eval = EqPolynomial::new(r_primary_sumcheck.to_vec()).evaluate(&r_z);
    assert_eq!(
      eq_eval * instruction.combine_lookups(&self.primary_sumcheck.eval_E, self.C, self.M),
      claim_last,
      "Primary sumcheck check failed."
    );

    self.primary_sumcheck.proof_E.verify(
      &r_z,
      &self.primary_sumcheck.eval_E,
      &self.generators.E_commitment_gens,
      &self.commitments.E_commitment,
      transcript
    )?;

    // produce a random element from the transcript for hash function
    let r_mem_check =
      <Transcript as ProofTranscript<G>>::challenge_vector(transcript, b"challenge_r_hash", 2);
    let r_fingerprints = (&r_mem_check[0], &r_mem_check[1]);

    let memory_to_dimension_index = | memory_index: usize | { memory_index % self.C };
    let evaluate_memory_mle = | memory_index: usize, vals: &[G::ScalarField]| { 
        let subtable_index = memory_index / self.C;
        instruction.subtables()[subtable_index].evaluate_mle(vals)
      };
    
    self.memory_check.verify(
      &self.commitments, 
      &self.generators, 
      memory_to_dimension_index, 
      evaluate_memory_mle, 
      r_fingerprints, 
      transcript)?;


    Ok(())
  }

  fn construct_polys(ops: &Vec<I>, C: usize, M: usize) -> SurgePolys<G::ScalarField> {
    let num_ops = ops.len().next_power_of_two();
    let instruction = I::default();
    let num_unique_subtables = instruction.subtables::<G::ScalarField>().len();
    let alpha = C * num_unique_subtables;

    let mut dim_i_usize: Vec<Vec<usize>> = vec![vec![0; num_ops]; C];

    let mut read_cts= vec![vec![0usize; num_ops]; C];
    let mut final_cts= vec![vec![0usize; M]; C];
    let log_M = ark_std::log2(M) as usize;

    for (op_index, op) in ops.iter().enumerate() {
      let access_sequence = op.to_indices(C, log_M);
      assert_eq!(access_sequence.len(), C);

      for dimension_index in 0..C {
        let memory_address = access_sequence[dimension_index];
        debug_assert!(memory_address < M);

        dim_i_usize[dimension_index][op_index] = memory_address;

        let ts = final_cts[dimension_index][memory_address];
        read_cts[dimension_index][op_index] = ts;
        let write_timestamp = ts + 1;
        final_cts[dimension_index][memory_address] = write_timestamp;
      }
    }

    let dim_i: Vec<DensePolynomial<G::ScalarField>> = dim_i_usize.iter().map(|dim| DensePolynomial::from_usize(dim)).collect();
    let read_i: Vec<DensePolynomial<G::ScalarField>> = read_cts.iter().map(|read| DensePolynomial::from_usize(read)).collect();
    let final_i: Vec<DensePolynomial<G::ScalarField>> = final_cts.iter().map(|fin| DensePolynomial::from_usize(fin)).collect();

    // Construct E
    let mut E_i_evals = Vec::with_capacity(alpha);
    let materialized_subtables: Vec<Vec<G::ScalarField>> = instruction.subtables::<G::ScalarField>().iter().map(|subtable| subtable.materialize(M)).collect();
    for E_index in 0..alpha {
      let mut E_evals = Vec::with_capacity(num_ops);
      for op_index in 0..num_ops {
        let dimension_index = E_index % C;
        let subtable_index = E_index / C;

        let eval_index = dim_i_usize[dimension_index][op_index];
        let eval = materialized_subtables[subtable_index][eval_index];
        E_evals.push(eval);
      }
      E_i_evals.push(E_evals);
    }
    let E_poly_i: Vec<DensePolynomial<G::ScalarField>> = E_i_evals.iter().map(|E| DensePolynomial::new(E.to_vec())).collect();

    // Combine
    let dim_read_polys = [dim_i.as_slice(), read_i.as_slice()].concat();
    let combined_dim_read_polys = DensePolynomial::merge(&dim_read_polys);
    let combined_final_polys = DensePolynomial::merge(&final_i);
    let combined_E_polys = DensePolynomial::merge(&E_poly_i);

    SurgePolys { 
      dim_i_usize,
      dim_i, 
      read_i, 
      final_i, 
      E_poly_i, 

      combined_dim_read_polys, 
      combined_final_polys, 
      combined_E_polys,

      materialized_subtables,

      num_ops, 
      m: M, 
      log_m: log_M, 
      dimensions: C,
      alpha 
    }
  }

  fn compute_primary_sumcheck_claim(polys: &SurgePolys<G::ScalarField>, eq: &DensePolynomial<G::ScalarField>) -> G::ScalarField {
    let g_operands = &polys.E_poly_i;
    let hypercube_size = g_operands[0].len();
    g_operands.iter().for_each(|operand| assert_eq!(operand.len(), hypercube_size));

    let instruction = I::default();

    (0..hypercube_size)
      .map(|eval_index| {
        let g_operands: Vec<G::ScalarField> = (0..polys.num_memories()).map(|memory_index| g_operands[memory_index][eval_index]).collect();
        eq[eval_index] * instruction.combine_lookups(&g_operands, polys.dimensions, polys.mem_size())
      }).sum()
  }

  fn protocol_name() -> &'static [u8] {
    b"SurgeProof"
  }
}

#[cfg(test)]
mod tests {
    use merlin::Transcript;

    use crate::{jolt::instruction::eq::EQInstruction, lasso::surge_2::SurgeProof};
    use ark_curve25519::EdwardsProjective;

    #[test]
    fn prod_layer_proof() {
    }

    #[test]
    fn e2e() {
      let ops = vec![
        EQInstruction(12, 12),
        EQInstruction(12, 82),
        EQInstruction(12, 12),
        EQInstruction(25, 12),
        EQInstruction(25, 12),
      ];
      let C = 8;
      let M = 1 << 8;

      let mut transcript = Transcript::new(b"test_transcript");
      let proof: SurgeProof<EdwardsProjective, _> = SurgeProof::prove(ops, C, M, &mut transcript);

      let mut transcript = Transcript::new(b"test_transcript");
      proof.verify(&mut transcript).expect("should work");
    }
}