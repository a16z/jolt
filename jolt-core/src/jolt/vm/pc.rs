use std::marker::PhantomData;

use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{One, Zero};
use merlin::Transcript;

use crate::{
  lasso::fingerprint_strategy::FingerprintStrategy,
  poly::{
    dense_mlpoly::{DensePolynomial, PolyCommitmentGens},
    identity_poly::IdentityPolynomial,
  },
  subprotocols::{
    combined_table_proof::{CombinedTableCommitment, CombinedTableEvalProof},
    grand_product::{BGPCInterpretable, BatchedGrandProductCircuit, GPEvals, GrandProductCircuit},
  },
  utils::{
    self, errors::ProofVerifyError, is_power_of_two, math::Math, transcript::ProofTranscript,
  },
};

pub struct ELFRow {
  address: usize,
  opcode: u64,
  rd: u64,
  rs1: u64,
  rs2: u64,
  imm: u64,
}

impl ELFRow {
  #[cfg(test)]
  fn new(address: usize, opcode: u64, rd: u64, rs1: u64, rs2: u64, imm: u64) -> Self {
    Self {
      address,
      opcode,
      rd,
      rs1,
      rs2,
      imm,
    }
  }
}

pub struct FiveTuplePoly<F: PrimeField> {
  opcode: DensePolynomial<F>,
  rd: DensePolynomial<F>,
  rs1: DensePolynomial<F>,
  rs2: DensePolynomial<F>,
  imm: DensePolynomial<F>,
}

impl<F: PrimeField> FiveTuplePoly<F> {
  fn from_elf(elf: &Vec<ELFRow>) -> Self {
    let len = elf.len().next_power_of_two();
    let mut opcodes = Vec::with_capacity(len);
    let mut rds = Vec::with_capacity(len);
    let mut rs1s = Vec::with_capacity(len);
    let mut rs2s = Vec::with_capacity(len);
    let mut imms = Vec::with_capacity(len);

    for row in elf {
      opcodes.push(F::from(row.opcode));
      rds.push(F::from(row.rd));
      rs1s.push(F::from(row.rs1));
      rs2s.push(F::from(row.rs2));
      imms.push(F::from(row.imm));
    }
    // Padding
    for _ in elf.len()..len {
      opcodes.push(F::zero());
      rds.push(F::zero());
      rs1s.push(F::zero());
      rs2s.push(F::zero());
      imms.push(F::zero());
    }

    let opcode = DensePolynomial::new(opcodes);
    let rd = DensePolynomial::new(rds);
    let rs1 = DensePolynomial::new(rs1s);
    let rs2 = DensePolynomial::new(rs2s);
    let imm = DensePolynomial::new(imms);
    FiveTuplePoly {
      opcode,
      rd,
      rs1,
      rs2,
      imm,
    }
  }

  fn evaluate(&self, r: &[F]) -> Vec<F> {
    vec![
      self.opcode.evaluate(r),
      self.rd.evaluate(r),
      self.rs1.evaluate(r),
      self.rs2.evaluate(r),
      self.imm.evaluate(r),
    ]
  }

  fn fingerprint_term(&self, leaf_index: usize, gamma: &F) -> F {
    // v.opcode * gamma + v.rd * gamma^2 + v.rs1 * gamma^3 + v.rs2 * gamma^4 + v.imm * gamma^5
    let mut gamma_term = gamma.clone();
    let mut fingerprint = self.opcode[leaf_index] * gamma_term;
    gamma_term = gamma_term.square();
    fingerprint += self.rd[leaf_index] * gamma_term;
    gamma_term = gamma_term * gamma;
    fingerprint += self.rs1[leaf_index] * gamma_term;
    gamma_term = gamma_term * gamma;
    fingerprint += self.rs2[leaf_index] * gamma_term;
    gamma_term = gamma_term * gamma;
    fingerprint += self.imm[leaf_index] * gamma_term;
    fingerprint
  }
}

pub struct PCPolys<F: PrimeField> {
  a_read_write: DensePolynomial<F>,

  v_read_write: FiveTuplePoly<F>,
  v_init_final: FiveTuplePoly<F>,

  t_read: DensePolynomial<F>,
  t_final: DensePolynomial<F>,

  /// Contains:
  /// - a_read_write, t_read, v_read_write
  combined_read_write: DensePolynomial<F>,

  // Contains:
  // - t_final, v_init_final
  combined_init_final: DensePolynomial<F>,
}

impl<F: PrimeField> PCPolys<F> {
  // TODO(sragss): precommit PC strategy
  pub fn new_program(program: Vec<ELFRow>, trace: Vec<ELFRow>) -> Self {
    assert!(is_power_of_two(program.len()));
    assert!(is_power_of_two(trace.len()));

    let num_ops = trace.len().next_power_of_two();
    let code_size = program.len().next_power_of_two();

    println!("PCPolys::new_program num_ops {num_ops} code_size {code_size}");

    // Note: a_read_write and read_cts have been implicitly padded with 0s
    // to the nearest power of 2
    let mut a_read_write_usize: Vec<usize> = vec![0; num_ops];
    let mut read_cts: Vec<usize> = vec![0; num_ops];
    let mut final_cts: Vec<usize> = vec![0; code_size];

    // TODO(sragss): Current padding strategy doesn't work. As it adds phantom
    // reads, but no corresponding writes to final.
    for (trace_index, trace) in trace.iter().enumerate() {
      let address = trace.address;
      debug_assert!(address < code_size);
      a_read_write_usize[trace_index] = address;
      let counter = final_cts[address];
      read_cts[trace_index] = counter;
      final_cts[address] = counter + 1;
    }

    let v_read_write = FiveTuplePoly::from_elf(&trace);
    let v_init_final = FiveTuplePoly::from_elf(&program);

    let a_read_write = DensePolynomial::from_usize(&a_read_write_usize);
    let t_read = DensePolynomial::from_usize(&read_cts);
    let t_final = DensePolynomial::from_usize(&final_cts);
    println!("read_cts {read_cts:?}");
    println!("t_read {t_read:?}");

    let combined_read_write = DensePolynomial::merge(&vec![
      &a_read_write,
      &t_read,
      &v_read_write.opcode,
      &v_read_write.rd,
      &v_read_write.rs1,
      &v_read_write.rs2,
      &v_read_write.imm,
    ]);
    let combined_init_final = DensePolynomial::merge(&vec![
      &t_final,
      &v_init_final.opcode,
      &v_init_final.rd,
      &v_init_final.rs1,
      &v_init_final.rs2,
      &v_init_final.imm,
    ]);

    Self {
      a_read_write,
      v_read_write,
      v_init_final,
      t_read,
      t_final,
      combined_read_write,
      combined_init_final,
    }
  }

  pub fn commit<G: CurveGroup<ScalarField = F>>(
    &self,
  ) -> (ProgramCommitmentGens<G>, ProgramCommitment<G>) {
    let gens = ProgramCommitmentGens::new(self.a_read_write.len(), self.v_init_final.opcode.len());

    let (read_write_commitments, _) = self.combined_read_write.commit(&gens.gens_read_write, None);
    let read_write_commitments = CombinedTableCommitment::new(read_write_commitments);

    let (init_final_commitments, _) = self.combined_init_final.commit(&gens.gens_init_final, None);
    let init_final_commitments = CombinedTableCommitment::new(init_final_commitments);

    let commitments = ProgramCommitment {
      read_write_commitments,
      init_final_commitments,
    };

    (gens, commitments)
  }
}

pub struct PCProof<F: PrimeField> {
  _marker: PhantomData<F>,
}

pub struct ProgramCommitment<G: CurveGroup> {
  /// Contains:
  /// - a_read_write, t_read, v_read_write
  pub read_write_commitments: CombinedTableCommitment<G>,

  // Contains:
  // - t_final, v_init_final
  pub init_final_commitments: CombinedTableCommitment<G>,
}

pub struct ProgramCommitmentGens<G: CurveGroup> {
  pub gens_read_write: PolyCommitmentGens<G>,
  pub gens_init_final: PolyCommitmentGens<G>,
}

impl<G: CurveGroup> ProgramCommitmentGens<G> {
  pub fn new(ops_size: usize, mem_size: usize) -> Self {
    debug_assert!(utils::is_power_of_two(ops_size));
    debug_assert!(utils::is_power_of_two(mem_size));

    // a_read_write, t_read, v_read_write.opcode, v_read_write.rd, v_read_write.rs1, v_read_write.rs2, v_read_write.imm
    let num_vars_ops = (7 * ops_size).log_2();
    // t_final, v_init_final.opcode, v_read_write.rd, v_read_write.rs1, v_read_write.rs2, v_read_write.imm
    let num_vars_mem = (6 * mem_size).log_2();

    let gens_read_write = PolyCommitmentGens::new(num_vars_ops, b"read_write_commitment");
    let gens_init_final = PolyCommitmentGens::new(num_vars_mem, b"init_final_commitment");

    ProgramCommitmentGens {
      gens_read_write,
      gens_init_final,
    }
  }
}

impl<F: PrimeField> BGPCInterpretable<F> for PCPolys<F> {
  fn a_ops(&self, memory_index: usize, leaf_index: usize) -> F {
    debug_assert_eq!(memory_index, 0);
    self.a_read_write[leaf_index]
  }

  fn a_mem(&self, memory_index: usize, leaf_index: usize) -> F {
    debug_assert_eq!(memory_index, 0);
    F::from(leaf_index as u64)
  }

  fn v_mem(&self, _memory_index: usize, _leaf_index: usize) -> F {
    unimplemented!("should not be called by fingerprinting functions");
  }

  fn v_ops(&self, _memory_index: usize, _leaf_index: usize) -> F {
    unimplemented!("should not be called by fingerprinting functions");
  }

  fn t_init(&self, _memory_index: usize, _leaf_index: usize) -> F {
    F::zero()
  }

  fn t_final(&self, memory_index: usize, leaf_index: usize) -> F {
    debug_assert_eq!(memory_index, 0);
    self.t_final[leaf_index]
  }

  fn t_read(&self, memory_index: usize, leaf_index: usize) -> F {
    debug_assert_eq!(memory_index, 0);
    self.t_read[leaf_index]
  }

  fn t_write(&self, memory_index: usize, leaf_index: usize) -> F {
    self.t_read(memory_index, leaf_index) + F::one()
  }

  // Overrides
  fn fingerprint_init(&self, memory_index: usize, leaf_index: usize, gamma: &F, tau: &F) -> F {
    debug_assert_eq!(memory_index, 0);
    let a = self.a_mem(memory_index, leaf_index);
    let v = self.v_init_final.fingerprint_term(leaf_index, gamma);
    let t = self.t_init(memory_index, leaf_index);
    Self::fingerprint(a, v, t, gamma, tau)
  }

  fn fingerprint_final(&self, memory_index: usize, leaf_index: usize, gamma: &F, tau: &F) -> F {
    debug_assert_eq!(memory_index, 0);
    let a = self.a_mem(memory_index, leaf_index);
    let v = self.v_init_final.fingerprint_term(leaf_index, gamma);
    let t = self.t_final(memory_index, leaf_index);
    Self::fingerprint(a, v, t, gamma, tau)
  }

  fn fingerprint_read(&self, memory_index: usize, leaf_index: usize, gamma: &F, tau: &F) -> F {
    debug_assert_eq!(memory_index, 0);
    let a = self.a_ops(memory_index, leaf_index);
    let v = self.v_read_write.fingerprint_term(leaf_index, gamma);
    let t = self.t_read(memory_index, leaf_index);
    Self::fingerprint(a, v, t, gamma, tau)
  }

  fn fingerprint_write(&self, memory_index: usize, leaf_index: usize, gamma: &F, tau: &F) -> F {
    debug_assert_eq!(memory_index, 0);
    let a = self.a_ops(memory_index, leaf_index);
    let v = self.v_read_write.fingerprint_term(leaf_index, gamma);
    let t = self.t_write(memory_index, leaf_index);
    Self::fingerprint(a, v, t, gamma, tau)
  }

  fn fingerprint(a: F, v: F, t: F, gamma: &F, tau: &F) -> F {
    // Assumes the v passed in is v.opcode * gamma + v.rd * gamma^2 + ... + v.imm * gamma^5
    let t_gamma: F = *gamma * gamma * gamma * gamma * gamma * gamma;
    t * t_gamma + v + a - tau
  }

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
    let init_evals = (0..self.v_init_final.opcode.len())
      .map(|i| self.fingerprint_init(memory_index, i, r_hash.0, r_hash.1))
      .collect();
    let read_evals = (0..self.a_read_write.len())
      .map(|i| self.fingerprint_read(memory_index, i, r_hash.0, r_hash.1))
      .collect();
    let write_evals = (0..self.a_read_write.len())
      .map(|i| self.fingerprint_write(memory_index, i, r_hash.0, r_hash.1))
      .collect();
    let final_evals = (0..self.v_init_final.opcode.len())
      .map(|i| self.fingerprint_final(memory_index, i, r_hash.0, r_hash.1))
      .collect();
    (
      DensePolynomial::new(init_evals),
      DensePolynomial::new(read_evals),
      DensePolynomial::new(write_evals),
      DensePolynomial::new(final_evals),
    )
  }

  fn construct_batches(
    &self,
    r_hash: (&F, &F),
  ) -> (
    BatchedGrandProductCircuit<F>,
    BatchedGrandProductCircuit<F>,
    Vec<GPEvals<F>>,
  ) {
    let (init_leaves, read_leaves, write_leaves, final_leaves) = self.compute_leaves(0, r_hash);
    let (init_gpc, read_gpc, write_gpc, final_gpc) = (
      GrandProductCircuit::new(&init_leaves),
      GrandProductCircuit::new(&read_leaves),
      GrandProductCircuit::new(&write_leaves),
      GrandProductCircuit::new(&final_leaves),
    );

    let gp_eval = GPEvals::new(
      init_gpc.evaluate(),
      read_gpc.evaluate(),
      write_gpc.evaluate(),
      final_gpc.evaluate(),
    );
    (
      BatchedGrandProductCircuit::new_batch(vec![read_gpc, write_gpc]),
      BatchedGrandProductCircuit::new_batch(vec![init_gpc, final_gpc]),
      vec![gp_eval],
    )
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PCFingerprintProof<G: CurveGroup> {
  eval_a_read_write: G::ScalarField,

  eval_v_read_write: Vec<G::ScalarField>,
  eval_v_init_final: Vec<G::ScalarField>,

  eval_t_read: G::ScalarField,
  eval_t_final: G::ScalarField,

  proof_read_write: CombinedTableEvalProof<G>,
  proof_init_final: CombinedTableEvalProof<G>,
}

impl<G: CurveGroup> FingerprintStrategy<G> for PCFingerprintProof<G> {
  type Polynomials = PCPolys<G::ScalarField>;
  type Generators = ProgramCommitmentGens<G>;
  type Commitments = ProgramCommitment<G>;

  fn prove(
    rand: (&Vec<<G>::ScalarField>, &Vec<<G>::ScalarField>),
    polynomials: &Self::Polynomials,
    generators: &Self::Generators,
    transcript: &mut merlin::Transcript,
    random_tape: &mut utils::random::RandomTape<G>,
  ) -> Self {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    let (rand_mem, rand_ops) = rand;

    let eval_a_read_write = polynomials.a_read_write.evaluate(&rand_ops);
    let eval_v_read_write: Vec<G::ScalarField> = polynomials.v_read_write.evaluate(&rand_ops);
    let eval_v_init_final: Vec<G::ScalarField> = polynomials.v_init_final.evaluate(&rand_mem);
    let eval_t_read = polynomials.t_read.evaluate(&rand_ops);
    let eval_t_final = polynomials.t_final.evaluate(&rand_mem);

    let mut evals_read_write: Vec<G::ScalarField> =
      vec![eval_a_read_write.clone(), eval_t_read.clone()];
    evals_read_write.extend(eval_v_read_write.iter());

    let proof_read_write = CombinedTableEvalProof::prove(
      &polynomials.combined_read_write,
      &evals_read_write,
      &rand_ops,
      &generators.gens_read_write,
      transcript,
      random_tape,
    );

    let mut evals_init_final: Vec<G::ScalarField> = vec![eval_t_final.clone()];
    evals_init_final.extend(eval_v_init_final.iter());
    let proof_init_final = CombinedTableEvalProof::prove(
      &polynomials.combined_init_final,
      &evals_init_final,
      &rand_mem,
      &generators.gens_init_final,
      transcript,
      random_tape,
    );

    Self {
      eval_a_read_write,

      eval_v_read_write,
      eval_v_init_final,

      eval_t_read,
      eval_t_final,

      proof_read_write,
      proof_init_final,
    }
  }

  fn verify<F1: Fn(usize) -> usize, F2: Fn(usize, &[<G>::ScalarField]) -> <G>::ScalarField>(
    &self,
    rand: (&Vec<<G>::ScalarField>, &Vec<<G>::ScalarField>),
    grand_product_claims: &[GPEvals<<G>::ScalarField>],
    // TODO(JOLT-47): Refactor from interface
    _memory_to_dimension_index: F1,
    _evaluate_memory_mle: F2,
    commitments: &Self::Commitments,
    generators: &Self::Generators,
    r_hash: &<G>::ScalarField,
    r_multiset_check: &<G>::ScalarField,
    transcript: &mut merlin::Transcript,
  ) -> Result<(), ProofVerifyError> {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    let (rand_mem, rand_ops) = rand;

    let mut evals_read_write: Vec<G::ScalarField> = vec![self.eval_a_read_write, self.eval_t_read];
    evals_read_write.extend(self.eval_v_read_write.iter());
    self.proof_read_write.verify(
      rand_ops,
      &evals_read_write,
      &generators.gens_read_write,
      &commitments.read_write_commitments,
      transcript,
    )?;

    let mut evals_init_final: Vec<G::ScalarField> = vec![self.eval_t_final];
    evals_init_final.extend(self.eval_v_init_final.iter());
    self.proof_init_final.verify(
      rand_mem,
      &evals_init_final,
      &generators.gens_init_final,
      &commitments.init_final_commitments,
      transcript,
    )?;

    debug_assert_eq!(self.eval_v_read_write.len(), 5);
    debug_assert_eq!(self.eval_v_init_final.len(), 5);
    // compute v.opcode * gamma + v.rd * gamma^2 + v.rs1 * gamma^3 + v.rs2 * gamma^4 + v.imm * gamma^5
    let mut gamma_term = r_hash.clone();
    let mut eval_v_read_write = self.eval_v_read_write[0] * gamma_term;
    let mut eval_v_init_final = self.eval_v_init_final[0] * gamma_term;
    gamma_term *= r_hash;
    eval_v_read_write += self.eval_v_read_write[1] * gamma_term;
    eval_v_init_final += self.eval_v_init_final[1] * gamma_term;
    gamma_term *= r_hash;
    eval_v_read_write += self.eval_v_read_write[2] * gamma_term;
    eval_v_init_final += self.eval_v_init_final[2] * gamma_term;
    gamma_term *= r_hash;
    eval_v_read_write += self.eval_v_read_write[3] * gamma_term;
    eval_v_init_final += self.eval_v_init_final[3] * gamma_term;
    gamma_term *= r_hash;
    eval_v_read_write += self.eval_v_read_write[4] * gamma_term;
    eval_v_init_final += self.eval_v_init_final[4] * gamma_term;
    gamma_term *= r_hash;

    debug_assert_eq!(grand_product_claims.len(), 1);
    let claim = &grand_product_claims[0];
    let a_init_final = IdentityPolynomial::new(rand_mem.len()).evaluate(rand_mem);
    let hash_init = Self::Polynomials::fingerprint(
      a_init_final,
      eval_v_init_final,
      G::ScalarField::zero(),
      r_hash,
      r_multiset_check,
    );
    assert_eq!(claim.hash_init, hash_init);

    let hash_read = Self::Polynomials::fingerprint(
      self.eval_a_read_write,
      eval_v_read_write,
      self.eval_t_read,
      r_hash,
      r_multiset_check,
    );
    assert_eq!(claim.hash_read, hash_read);
    let hash_write = Self::Polynomials::fingerprint(
      self.eval_a_read_write,
      eval_v_read_write,
      self.eval_t_read + G::ScalarField::one(),
      r_hash,
      r_multiset_check,
    );
    assert_eq!(claim.hash_write, hash_write);

    let hash_final = Self::Polynomials::fingerprint(
      a_init_final,
      eval_v_init_final,
      self.eval_t_final,
      r_hash,
      r_multiset_check,
    );
    assert_eq!(claim.hash_final, hash_final);

    Ok(())
  }
}

impl<G: CurveGroup> PCFingerprintProof<G> {
  fn protocol_name() -> &'static [u8] {
    b"PCFingerprintProof"
  }
}

#[cfg(test)]
mod tests {
  use std::collections::HashSet;

  use crate::subprotocols::grand_product::BGPCInterpretable;
  use crate::{
    lasso::memory_checking::{MemoryCheckingProof, ProductLayerProof},
    poly::dense_mlpoly::DensePolynomial,
    utils::random::RandomTape,
  };
  use ark_curve25519::{EdwardsProjective, Fr};
  use merlin::Transcript;

  use super::{ELFRow, FiveTuplePoly, PCFingerprintProof, PCPolys};

  #[test]
  fn five_tuple_poly() {
    let program = vec![
      ELFRow::new(0, 2u64, 3u64, 4u64, 5u64, 6u64),
      ELFRow::new(1, 7u64, 8u64, 9u64, 10u64, 11u64),
      ELFRow::new(2, 12u64, 13u64, 14u64, 15u64, 16u64),
      ELFRow::new(3, 17u64, 18u64, 19u64, 20u64, 21u64),
    ];
    let tuple: FiveTuplePoly<Fr> = FiveTuplePoly::from_elf(&program);
    let expected_opcode: DensePolynomial<Fr> = DensePolynomial::from_usize(&vec![2, 7, 12, 17]);
    let expected_rd: DensePolynomial<Fr> = DensePolynomial::from_usize(&vec![3, 8, 13, 18]);
    let expected_rs1: DensePolynomial<Fr> = DensePolynomial::from_usize(&vec![4, 9, 14, 19]);
    let expected_rs2: DensePolynomial<Fr> = DensePolynomial::from_usize(&vec![5, 10, 15, 20]);
    let expected_imm: DensePolynomial<Fr> = DensePolynomial::from_usize(&vec![6, 11, 16, 21]);
    assert_eq!(tuple.opcode, expected_opcode);
    assert_eq!(tuple.rd, expected_rd);
    assert_eq!(tuple.rs1, expected_rs1);
    assert_eq!(tuple.rs2, expected_rs2);
    assert_eq!(tuple.imm, expected_imm);

    let gamma = Fr::from(100);
    let fingerprint = tuple.fingerprint_term(2, &gamma);
    let expected_fingerprint = gamma * Fr::from(12)
      + gamma * gamma * Fr::from(13)
      + gamma * gamma * gamma * Fr::from(14)
      + gamma * gamma * gamma * gamma * Fr::from(15)
      + gamma * gamma * gamma * gamma * gamma * Fr::from(16);
    assert_eq!(fingerprint, expected_fingerprint);
  }

  fn get_difference<T: Clone + Eq + std::hash::Hash>(vec1: &[T], vec2: &[T]) -> Vec<T> {
    let set1: HashSet<_> = vec1.iter().cloned().collect();
    let set2: HashSet<_> = vec2.iter().cloned().collect();
    set1.difference(&set2).cloned().collect()
  }

  // #[test]
  // fn pc_poly_leaf_construction() {
  //   let program = vec![
  //     ELFRow::new(0, 2u64, 2u64, 2u64, 2u64, 2u64),
  //     ELFRow::new(1, 4u64, 4u64, 4u64, 4u64, 4u64),
  //     ELFRow::new(2, 8u64, 8u64, 8u64, 8u64, 8u64),
  //     ELFRow::new(3, 16u64, 16u64, 16u64, 16u64, 16u64),
  //   ];
  //   let trace = vec![
  //     ELFRow::new(3, 16u64, 16u64, 16u64, 16u64, 16u64),
  //     ELFRow::new(2, 8u64, 8u64, 8u64, 8u64, 8u64),
  //   ];
  //   let polys: PCPolys<Fr> = PCPolys::new_program(program, trace);

  //   let r_fingerprints = (&Fr::from(100), &Fr::from(35));
  //   let (init_leaves, read_leaves, write_leaves, final_leaves) =
  //     polys.compute_leaves(0, r_fingerprints);
  //   let read_final_leaves = vec![read_leaves.evals(), final_leaves.evals()].concat();
  //   let init_write_leaves = vec![init_leaves.evals(), write_leaves.evals()].concat();
  //   let difference: Vec<Fr> = get_difference(&read_final_leaves, &init_write_leaves);
  //   assert_eq!(difference.len(), 0);
  // }

  #[test]
  fn product_layer_proof() {
    let program = vec![
      ELFRow::new(0, 2u64, 2u64, 2u64, 2u64, 2u64),
      ELFRow::new(1, 4u64, 4u64, 4u64, 4u64, 4u64),
      ELFRow::new(2, 8u64, 8u64, 8u64, 8u64, 8u64),
      ELFRow::new(3, 16u64, 16u64, 16u64, 16u64, 16u64),
    ];
    let trace = vec![
      ELFRow::new(3, 16u64, 16u64, 16u64, 16u64, 16u64),
      ELFRow::new(2, 8u64, 8u64, 8u64, 8u64, 8u64),
    ];
    let polys: PCPolys<Fr> = PCPolys::new_program(program, trace);
    let mut transcript = Transcript::new(b"test_transcript");
    let r_fingerprints = (&Fr::from(12), &Fr::from(35));
    let (proof, _, _) =
      ProductLayerProof::prove::<EdwardsProjective, _>(&polys, r_fingerprints, &mut transcript);

    let mut transcript = Transcript::new(b"test_transcript");
    proof
      .verify::<EdwardsProjective>(&mut transcript)
      .expect("proof should work");
  }

  #[test]
  #[should_panic]
  fn product_layer_proof_non_pow_two() {
    let program = vec![
      ELFRow::new(0, 2u64, 2u64, 2u64, 2u64, 2u64),
      ELFRow::new(1, 4u64, 4u64, 4u64, 4u64, 4u64),
      ELFRow::new(2, 8u64, 8u64, 8u64, 8u64, 8u64),
      ELFRow::new(3, 16u64, 16u64, 16u64, 16u64, 16u64),
      ELFRow::new(4, 32u64, 32u64, 32u64, 32u64, 32u64),
    ];
    let trace = vec![
      ELFRow::new(3, 16u64, 16u64, 16u64, 16u64, 16u64),
      ELFRow::new(2, 8u64, 8u64, 8u64, 8u64, 8u64),
      ELFRow::new(1, 4u64, 4u64, 4u64, 4u64, 4u64),
    ];
    let _polys: PCPolys<Fr> = PCPolys::new_program(program, trace);
    // let (gens, commitments) = polys.commit::<EdwardsProjective>();
    // let mut transcript = Transcript::new(b"test_transcript");
    // let r_fingerprints = (&Fr::from(12), &Fr::from(35));
    // let (proof, _, _) =
    //   ProductLayerProof::prove::<EdwardsProjective, _>(&polys, r_fingerprints, &mut transcript);

    // let mut transcript = Transcript::new(b"test_transcript");
    // proof
    //   .verify::<EdwardsProjective>(&mut transcript)
    //   .expect("proof should work");
  }

  #[test]
  fn e2e_mem_checking() {
    let program = vec![
      ELFRow::new(0, 2u64, 2u64, 2u64, 2u64, 2u64),
      ELFRow::new(1, 4u64, 4u64, 4u64, 4u64, 4u64),
      ELFRow::new(2, 8u64, 8u64, 8u64, 8u64, 8u64),
      ELFRow::new(3, 16u64, 16u64, 16u64, 16u64, 16u64),
    ];
    let trace = vec![
      ELFRow::new(3, 16u64, 16u64, 16u64, 16u64, 16u64),
      ELFRow::new(2, 8u64, 8u64, 8u64, 8u64, 8u64),
    ];
    let polys: PCPolys<Fr> = PCPolys::new_program(program, trace);
    let (gens, commitments) = polys.commit::<EdwardsProjective>();

    let mut transcript = Transcript::new(b"test_transcript");
    let mut random_tape = RandomTape::new(b"test_tape");
    let r_fingerprints = (&Fr::from(12), &Fr::from(35));
    let memory_checking_proof =
      MemoryCheckingProof::<EdwardsProjective, PCFingerprintProof<EdwardsProjective>>::prove(
        &polys,
        r_fingerprints,
        &gens,
        &mut transcript,
        &mut random_tape,
      );

    let memory_to_dimension_index = |memory_index: usize| {
      assert_eq!(memory_index, 0);
      0
    };
    let evaluate_memory_mle = |_: usize, _: &[Fr]| unimplemented!("shouldn't be called");
    let mut transcript = Transcript::new(b"test_transcript");
    memory_checking_proof
      .verify(
        &commitments,
        &gens,
        memory_to_dimension_index,
        evaluate_memory_mle,
        r_fingerprints,
        &mut transcript,
      )
      .expect("should verify");
  }
}
