use std::marker::PhantomData;

use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_std::Zero;
use merlin::Transcript;

use crate::{
  lasso::memory_checking::{MemoryCheckingProver, MemoryCheckingVerifier},
  poly::{
    dense_mlpoly::{DensePolynomial, PolyCommitmentGens},
    identity_poly::IdentityPolynomial,
    structured_poly::{StructuredOpeningProof, StructuredPolynomials},
  },
  subprotocols::combined_table_proof::{CombinedTableCommitment, CombinedTableEvalProof},
  utils::{errors::ProofVerifyError, is_power_of_two, random::RandomTape},
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
}

pub struct PCPolys<F: PrimeField, G: CurveGroup<ScalarField = F>> {
  _group: PhantomData<G>,
  a_read_write: DensePolynomial<F>,

  v_read_write: FiveTuplePoly<F>,
  v_init_final: FiveTuplePoly<F>,

  t_read: DensePolynomial<F>,
  t_final: DensePolynomial<F>,
}

pub struct BatchedPCPolys<F: PrimeField> {
  /// Contains:
  /// - a_read_write, t_read, v_read_write
  combined_read_write: DensePolynomial<F>,

  // Contains:
  // - t_final, v_init_final
  combined_init_final: DensePolynomial<F>,
}

pub struct ProgramCommitment<G: CurveGroup> {
  generators: PCCommitmentGenerators<G>,
  /// Contains:
  /// - a_read_write, t_read, v_read_write
  pub read_write_commitments: CombinedTableCommitment<G>,

  // Contains:
  // - t_final, v_init_final
  pub init_final_commitments: CombinedTableCommitment<G>,
}

pub struct PCCommitmentGenerators<G: CurveGroup> {
  pub gens_read_write: PolyCommitmentGens<G>,
  pub gens_init_final: PolyCommitmentGens<G>,
}

impl<F, G> StructuredPolynomials for PCPolys<F, G>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
{
  type Commitment = ProgramCommitment<G>;
  type BatchedPolynomials = BatchedPCPolys<F>;

  fn batch(&self) -> Self::BatchedPolynomials {
    let combined_read_write = DensePolynomial::merge(&vec![
      &self.a_read_write,
      &self.t_read,
      &self.v_read_write.opcode,
      &self.v_read_write.rd,
      &self.v_read_write.rs1,
      &self.v_read_write.rs2,
      &self.v_read_write.imm,
    ]);
    let combined_init_final = DensePolynomial::merge(&vec![
      &self.t_final,
      &self.v_init_final.opcode,
      &self.v_init_final.rd,
      &self.v_init_final.rs1,
      &self.v_init_final.rs2,
      &self.v_init_final.imm,
    ]);

    Self::BatchedPolynomials {
      combined_read_write,
      combined_init_final,
    }
  }

  fn commit(batched_polys: &Self::BatchedPolynomials) -> Self::Commitment {
    let (gens_read_write, read_write_commitments) = batched_polys
      .combined_read_write
      .combined_commit(b"BatchedPCPolys.read_write");
    let (gens_init_final, init_final_commitments) = batched_polys
      .combined_init_final
      .combined_commit(b"BatchedPCPolys.init_final");

    let generators = PCCommitmentGenerators {
      gens_read_write,
      gens_init_final,
    };

    Self::Commitment {
      read_write_commitments,
      init_final_commitments,
      generators,
    }
  }
}

impl<F: PrimeField, G: CurveGroup<ScalarField = F>> PCPolys<F, G> {
  // TODO(sragss): precommit PC strategy
  pub fn new_program(program: Vec<ELFRow>, trace: Vec<ELFRow>) -> Self {
    assert!(is_power_of_two(program.len()));
    assert!(is_power_of_two(trace.len()));

    let num_ops = trace.len().next_power_of_two();
    let code_size = program.len().next_power_of_two();

    // Note: a_read_write and read_cts have been implicitly padded with 0s
    // to the nearest power of 2
    let mut a_read_write_usize: Vec<usize> = vec![0; num_ops];
    let mut read_cts: Vec<usize> = vec![0; num_ops];
    let mut final_cts: Vec<usize> = vec![0; code_size];

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

    Self {
      _group: PhantomData,
      a_read_write,
      v_read_write,
      v_init_final,
      t_read,
      t_final,
    }
  }
}

pub struct PCProof<F: PrimeField>(PhantomData<F>);

impl<F, G> MemoryCheckingProver<F, G, PCPolys<F, G>> for PCProof<F>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
{
  type ReadWriteOpenings = PCReadWriteOpenings<F, G>;
  type InitFinalOpenings = PCInitFinalOpenings<F, G>;

  // [a, opcode, rd, rs1, rs2, imm, t]
  type MemoryTuple = [F; 7];

  fn fingerprint(inputs: &Self::MemoryTuple, gamma: &F, tau: &F) -> F {
    let mut result = F::zero();
    let mut gamma_term = F::one();
    for input in inputs {
      result += *input * gamma_term;
      gamma_term *= gamma;
    }
    result - tau
  }

  fn read_leaves(
    &self,
    polynomials: &PCPolys<F, G>,
    gamma: &F,
    tau: &F,
  ) -> Vec<DensePolynomial<F>> {
    let num_ops = polynomials.a_read_write.len();
    let read_fingerprints = (0..num_ops)
      .map(|i| {
        <Self as MemoryCheckingProver<F, G, PCPolys<F, G>>>::fingerprint(
          &[
            polynomials.a_read_write[i],
            polynomials.v_read_write.opcode[i],
            polynomials.v_read_write.rd[i],
            polynomials.v_read_write.rs1[i],
            polynomials.v_read_write.rs2[i],
            polynomials.v_read_write.imm[i],
            polynomials.t_read[i],
          ],
          gamma,
          tau,
        )
      })
      .collect();
    vec![DensePolynomial::new(read_fingerprints)]
  }
  fn write_leaves(
    &self,
    polynomials: &PCPolys<F, G>,
    gamma: &F,
    tau: &F,
  ) -> Vec<DensePolynomial<F>> {
    let num_ops = polynomials.a_read_write.len();
    let read_fingerprints = (0..num_ops)
      .map(|i| {
        <Self as MemoryCheckingProver<F, G, PCPolys<F, G>>>::fingerprint(
          &[
            polynomials.a_read_write[i],
            polynomials.v_read_write.opcode[i],
            polynomials.v_read_write.rd[i],
            polynomials.v_read_write.rs1[i],
            polynomials.v_read_write.rs2[i],
            polynomials.v_read_write.imm[i],
            polynomials.t_read[i] + F::one(),
          ],
          gamma,
          tau,
        )
      })
      .collect();
    vec![DensePolynomial::new(read_fingerprints)]
  }
  fn init_leaves(
    &self,
    polynomials: &PCPolys<F, G>,
    gamma: &F,
    tau: &F,
  ) -> Vec<DensePolynomial<F>> {
    let memory_size = polynomials.v_init_final.opcode.len();
    let init_fingerprints = (0..memory_size)
      .map(|i| {
        <Self as MemoryCheckingProver<F, G, PCPolys<F, G>>>::fingerprint(
          &[
            F::from(i as u64),
            polynomials.v_init_final.opcode[i],
            polynomials.v_init_final.rd[i],
            polynomials.v_init_final.rs1[i],
            polynomials.v_init_final.rs2[i],
            polynomials.v_init_final.imm[i],
            F::zero(),
          ],
          gamma,
          tau,
        )
      })
      .collect();
    vec![DensePolynomial::new(init_fingerprints)]
  }
  fn final_leaves(
    &self,
    polynomials: &PCPolys<F, G>,
    gamma: &F,
    tau: &F,
  ) -> Vec<DensePolynomial<F>> {
    let memory_size = polynomials.v_init_final.opcode.len();
    let init_fingerprints = (0..memory_size)
      .map(|i| {
        <Self as MemoryCheckingProver<F, G, PCPolys<F, G>>>::fingerprint(
          &[
            F::from(i as u64),
            polynomials.v_init_final.opcode[i],
            polynomials.v_init_final.rd[i],
            polynomials.v_init_final.rs1[i],
            polynomials.v_init_final.rs2[i],
            polynomials.v_init_final.imm[i],
            polynomials.t_final[i],
          ],
          gamma,
          tau,
        )
      })
      .collect();
    vec![DensePolynomial::new(init_fingerprints)]
  }

  fn protocol_name() -> &'static [u8] {
    b"Bytecode memory checking"
  }
}

impl<F, G> MemoryCheckingVerifier<F, G, PCPolys<F, G>> for PCProof<F>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
{
  fn compute_verifier_openings(openings: &mut Self::InitFinalOpenings, opening_point: &Vec<F>) {
    openings.a_init_final =
      Some(IdentityPolynomial::new(opening_point.len()).evaluate(opening_point));
  }

  fn read_tuples(openings: &Self::ReadWriteOpenings) -> Vec<Self::MemoryTuple> {
    vec![[
      openings.a_read_write_opening,
      openings.v_read_write_openings[0], // opcode
      openings.v_read_write_openings[1], // rd
      openings.v_read_write_openings[2], // rs1
      openings.v_read_write_openings[3], // rs2
      openings.v_read_write_openings[4], // imm
      openings.t_read_opening,
    ]]
  }
  fn write_tuples(openings: &Self::ReadWriteOpenings) -> Vec<Self::MemoryTuple> {
    vec![[
      openings.a_read_write_opening,
      openings.v_read_write_openings[0], // opcode
      openings.v_read_write_openings[1], // rd
      openings.v_read_write_openings[2], // rs1
      openings.v_read_write_openings[3], // rs2
      openings.v_read_write_openings[4], // imm
      openings.t_read_opening + F::one(),
    ]]
  }
  fn init_tuples(openings: &Self::InitFinalOpenings) -> Vec<Self::MemoryTuple> {
    vec![[
      openings.a_init_final.unwrap(),
      openings.v_init_final[0], // opcode
      openings.v_init_final[1], // rd
      openings.v_init_final[2], // rs1
      openings.v_init_final[3], // rs2
      openings.v_init_final[4], // imm
      F::zero(),
    ]]
  }
  fn final_tuples(openings: &Self::InitFinalOpenings) -> Vec<Self::MemoryTuple> {
    vec![[
      openings.a_init_final.unwrap(),
      openings.v_init_final[0], // opcode
      openings.v_init_final[1], // rd
      openings.v_init_final[2], // rs1
      openings.v_init_final[3], // rs2
      openings.v_init_final[4], // imm
      openings.t_final,
    ]]
  }
}

pub struct PCReadWriteOpenings<F, G>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
{
  a_read_write_opening: F,
  v_read_write_openings: Vec<F>,
  t_read_opening: F,

  read_write_opening_proof: CombinedTableEvalProof<G>,
}

impl<F, G> StructuredOpeningProof<F, G, PCPolys<F, G>> for PCReadWriteOpenings<F, G>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
{
  type Openings = (F, Vec<F>, F);

  fn open(polynomials: &PCPolys<F, G>, opening_point: &Vec<F>) -> Self::Openings {
    (
      polynomials.a_read_write.evaluate(&opening_point),
      polynomials.v_read_write.evaluate(&opening_point),
      polynomials.t_read.evaluate(&opening_point),
    )
  }

  fn prove_openings(
    polynomials: &BatchedPCPolys<F>,
    commitment: &ProgramCommitment<G>,
    opening_point: &Vec<F>,
    openings: (F, Vec<F>, F),
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self {
    let a_read_write_opening = openings.0;
    let v_read_write_openings = openings.1;
    let t_read_opening = openings.2;

    let mut combined_openings: Vec<F> = vec![a_read_write_opening.clone(), t_read_opening.clone()];
    combined_openings.extend(v_read_write_openings.iter());

    let read_write_opening_proof = CombinedTableEvalProof::prove(
      &polynomials.combined_read_write,
      &combined_openings,
      &opening_point,
      &commitment.generators.gens_read_write,
      transcript,
      random_tape,
    );

    Self {
      a_read_write_opening,
      v_read_write_openings,
      t_read_opening,
      read_write_opening_proof,
    }
  }

  fn verify_openings(
    &self,
    commitment: &ProgramCommitment<G>,
    opening_point: &Vec<F>,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    let mut combined_openings: Vec<F> = vec![
      self.a_read_write_opening.clone(),
      self.t_read_opening.clone(),
    ];
    combined_openings.extend(self.v_read_write_openings.iter());

    self.read_write_opening_proof.verify(
      opening_point,
      &combined_openings,
      &commitment.generators.gens_read_write,
      &commitment.read_write_commitments,
      transcript,
    )
  }
}

pub struct PCInitFinalOpenings<F, G>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
{
  a_init_final: Option<F>, // Computed by verifier
  v_init_final: Vec<F>,
  t_final: F,

  init_final_opening_proof: CombinedTableEvalProof<G>,
}

impl<F, G> StructuredOpeningProof<F, G, PCPolys<F, G>> for PCInitFinalOpenings<F, G>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
{
  type Openings = (Vec<F>, F);

  fn open(polynomials: &PCPolys<F, G>, opening_point: &Vec<F>) -> Self::Openings {
    (
      polynomials.v_init_final.evaluate(&opening_point),
      polynomials.t_final.evaluate(&opening_point),
    )
  }

  fn prove_openings(
    polynomials: &BatchedPCPolys<F>,
    commitment: &ProgramCommitment<G>,
    opening_point: &Vec<F>,
    openings: Self::Openings,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self {
    let v_init_final = openings.0;
    let t_final = openings.1;

    let mut combined_openings: Vec<F> = vec![t_final];
    combined_openings.extend(v_init_final.iter());
    let init_final_opening_proof = CombinedTableEvalProof::prove(
      &polynomials.combined_init_final,
      &combined_openings,
      &opening_point,
      &commitment.generators.gens_init_final,
      transcript,
      random_tape,
    );

    Self {
      a_init_final: None, // Computed by verifier
      v_init_final,
      t_final,
      init_final_opening_proof,
    }
  }

  fn verify_openings(
    &self,
    commitment: &ProgramCommitment<G>,
    opening_point: &Vec<F>,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    let mut combined_openings: Vec<F> = vec![self.t_final.clone()];
    combined_openings.extend(self.v_init_final.iter());

    self.init_final_opening_proof.verify(
      opening_point,
      &combined_openings,
      &commitment.generators.gens_init_final,
      &commitment.init_final_commitments,
      transcript,
    )
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use ark_curve25519::{EdwardsProjective, Fr};
  use std::collections::HashSet;

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
  }

  fn get_difference<T: Clone + Eq + std::hash::Hash>(vec1: &[T], vec2: &[T]) -> Vec<T> {
    let set1: HashSet<_> = vec1.iter().cloned().collect();
    let set2: HashSet<_> = vec2.iter().cloned().collect();
    set1.difference(&set2).cloned().collect()
  }

  #[test]
  fn pc_poly_leaf_construction() {
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
    let polys: PCPolys<Fr, EdwardsProjective> = PCPolys::new_program(program, trace);

    let (gamma, tau)= (&Fr::from(100), &Fr::from(35));
    let pc_prover: PCProof<Fr> = PCProof(PhantomData::<_>);
    let init_leaves: Vec<DensePolynomial<Fr>> = pc_prover.init_leaves(&polys, gamma, tau);
    let read_leaves: Vec<DensePolynomial<Fr>> = pc_prover.read_leaves(&polys, gamma, tau);
    let write_leaves: Vec<DensePolynomial<Fr>> = pc_prover.write_leaves(&polys, gamma, tau);
    let final_leaves: Vec<DensePolynomial<Fr>> = pc_prover.final_leaves(&polys, gamma, tau);

    let init_leaves = &init_leaves[0];
    let read_leaves = &read_leaves[0];
    let write_leaves = &write_leaves[0];
    let final_leaves = &final_leaves[0];

    let read_final_leaves = vec![read_leaves.evals(), final_leaves.evals()].concat();
    let init_write_leaves = vec![init_leaves.evals(), write_leaves.evals()].concat();
    let difference: Vec<Fr> = get_difference(&read_final_leaves, &init_write_leaves);
    assert_eq!(difference.len(), 0);
  }

  #[test]
  fn e2e_memchecking() {
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
    let polys: PCPolys<Fr, EdwardsProjective> = PCPolys::new_program(program, trace);
    let pc_prover: PCProof<Fr> = PCProof(PhantomData::<_>); // TODO(sragss): Why is this necessary? -- default?

    let mut transcript = Transcript::new(b"test_transcript");
    let mut random_tape = RandomTape::new(b"test_tape");

    let batched_polys = polys.batch();
    let commitments = PCPolys::commit(&batched_polys);
    let proof = pc_prover.prove_memory_checking(&polys, &batched_polys, &commitments, &mut transcript, &mut random_tape);

    let mut transcript = Transcript::new(b"test_transcript");
    PCProof::verify_memory_checking(proof, &commitments, &mut transcript).expect("proof should verify");
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
    let _polys: PCPolys<Fr, EdwardsProjective> = PCPolys::new_program(program, trace);
  }
}
