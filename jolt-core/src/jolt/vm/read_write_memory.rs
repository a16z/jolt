use std::marker::PhantomData;

use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use merlin::Transcript;

use crate::{
  lasso::memory_checking::{MemoryCheckingProver, MemoryCheckingVerifier},
  poly::{
    dense_mlpoly::{DensePolynomial, PolyCommitmentGens},
    identity_poly::IdentityPolynomial,
    structured_poly::{BatchablePolynomials, StructuredOpeningProof},
  },
  subprotocols::combined_table_proof::{CombinedTableCommitment, CombinedTableEvalProof},
  subprotocols::grand_product::BatchedGrandProductCircuit,
  utils::{errors::ProofVerifyError, random::RandomTape},
};
use common::constants::{RAM_START_ADDRESS, REGISTER_COUNT};

#[derive(Debug, PartialEq, Clone)]
pub enum MemoryOp {
  Read(u64, u64),       // (address, value)
  Write(u64, u64, u64), // (address, old_value, new_value)
}

impl MemoryOp {
  pub fn no_op() -> Self {
    Self::Read(0, 0)
  }
}

pub struct ReadWriteMemory<F, G>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
{
  _group: PhantomData<G>,
  memory_size: usize,

  a_read_write: DensePolynomial<F>,

  v_read: DensePolynomial<F>,
  v_write: DensePolynomial<F>,
  v_final: DensePolynomial<F>,

  t_read: DensePolynomial<F>,
  t_write: DensePolynomial<F>,
  t_final: DensePolynomial<F>,
}

impl<F: PrimeField, G: CurveGroup<ScalarField = F>> ReadWriteMemory<F, G> {
  pub fn new(
    memory_trace: Vec<MemoryOp>,
    memory_size: usize,
    transcript: &mut Transcript,
  ) -> (Self, Vec<u64>) {
    let m = memory_trace.len();
    assert!(m.is_power_of_two());

    let mut a_read_write: Vec<u64> = Vec::with_capacity(m);
    let mut v_read: Vec<u64> = Vec::with_capacity(m);
    let mut v_write: Vec<u64> = Vec::with_capacity(m);
    let mut v_final: Vec<u64> = vec![0; memory_size];
    let mut t_read: Vec<u64> = Vec::with_capacity(m);
    let mut t_write: Vec<u64> = Vec::with_capacity(m);
    let mut t_final: Vec<u64> = vec![0; memory_size];

    let mut timestamp: u64 = 0;
    for memory_access in memory_trace {
      match memory_access {
        MemoryOp::Read(a, v) => {
          assert!(a < REGISTER_COUNT || a >= RAM_START_ADDRESS);
          let remapped_a = if a >= RAM_START_ADDRESS {
            a - RAM_START_ADDRESS + REGISTER_COUNT
          } else {
            a
          };
          a_read_write.push(remapped_a);
          v_read.push(v);
          v_write.push(v);
          t_read.push(t_final[remapped_a as usize]);
          t_write.push(timestamp + 1);
          t_final[remapped_a as usize] = timestamp + 1;
        }
        MemoryOp::Write(a, v_old, v_new) => {
          assert!(a < REGISTER_COUNT || a >= RAM_START_ADDRESS);
          let remapped_a = if a >= RAM_START_ADDRESS {
            a - RAM_START_ADDRESS + REGISTER_COUNT
          } else {
            a
          };
          a_read_write.push(remapped_a);
          v_read.push(v_old);
          v_write.push(v_new);
          v_final[remapped_a as usize] = v_new;
          t_read.push(t_final[remapped_a as usize]);
          t_write.push(timestamp + 1);
          t_final[remapped_a as usize] = timestamp + 1;
        }
      }
      timestamp += 1;
    }

    (
      Self {
        _group: PhantomData,
        memory_size,
        a_read_write: DensePolynomial::from_u64(&a_read_write),
        v_read: DensePolynomial::from_u64(&v_read),
        v_write: DensePolynomial::from_u64(&v_write),
        v_final: DensePolynomial::from_u64(&v_final),
        t_read: DensePolynomial::from_u64(&t_read),
        t_write: DensePolynomial::from_u64(&t_write),
        t_final: DensePolynomial::from_u64(&t_final),
      },
      t_read,
    )
  }
}

pub struct BatchedMemoryPolynomials<F: PrimeField> {
  /// Contains:
  /// a_read_write, v_read, v_write, t_read, t_write
  batched_read_write: DensePolynomial<F>,
  /// Contains:
  /// v_final, t_final
  batched_init_final: DensePolynomial<F>,
}

pub struct MemoryCommitment<G: CurveGroup> {
  generators: MemoryCommitmentGenerators<G>,
  /// Contains:
  /// a_read_write, v_read, v_write, t_read, t_write
  pub read_write_commitments: CombinedTableCommitment<G>,

  /// Contains:
  /// v_final, t_final
  pub init_final_commitments: CombinedTableCommitment<G>,
}

pub struct MemoryCommitmentGenerators<G: CurveGroup> {
  pub gens_read_write: PolyCommitmentGens<G>,
  pub gens_init_final: PolyCommitmentGens<G>,
}

impl<F, G> BatchablePolynomials for ReadWriteMemory<F, G>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
{
  type Commitment = MemoryCommitment<G>;
  type BatchedPolynomials = BatchedMemoryPolynomials<F>;

  fn batch(&self) -> Self::BatchedPolynomials {
    let batched_read_write = DensePolynomial::merge(&vec![
      &self.a_read_write,
      &self.v_read,
      &self.v_write,
      &self.t_read,
      &self.t_write,
    ]);
    let batched_init_final = DensePolynomial::merge(&vec![&self.v_final, &self.t_final]);

    Self::BatchedPolynomials {
      batched_read_write,
      batched_init_final,
    }
  }

  fn commit(batched_polys: &Self::BatchedPolynomials) -> Self::Commitment {
    let (gens_read_write, read_write_commitments) = batched_polys
      .batched_read_write
      .combined_commit(b"BatchedMemoryPolynomials.batched_read_write");
    let (gens_init_final, init_final_commitments) = batched_polys
      .batched_init_final
      .combined_commit(b"BatchedMemoryPolynomials.batched_init_final");

    let generators = MemoryCommitmentGenerators {
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

pub struct MemoryReadWriteOpenings<F, G>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
{
  a_read_write_opening: F,
  v_read_opening: F,
  v_write_opening: F,
  t_read_opening: F,
  t_write_opening: F,

  read_write_opening_proof: CombinedTableEvalProof<G>,
}

impl<F, G> StructuredOpeningProof<F, G, ReadWriteMemory<F, G>> for MemoryReadWriteOpenings<F, G>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
{
  type Openings = [F; 5];

  fn open(polynomials: &ReadWriteMemory<F, G>, opening_point: &Vec<F>) -> Self::Openings {
    [
      polynomials.a_read_write.evaluate(&opening_point),
      polynomials.v_read.evaluate(&opening_point),
      polynomials.v_write.evaluate(&opening_point),
      polynomials.t_read.evaluate(&opening_point),
      polynomials.t_write.evaluate(&opening_point),
    ]
  }

  fn prove_openings(
    polynomials: &BatchedMemoryPolynomials<F>,
    commitment: &MemoryCommitment<G>,
    opening_point: &Vec<F>,
    openings: [F; 5],
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self {
    let read_write_opening_proof = CombinedTableEvalProof::prove(
      &polynomials.batched_read_write,
      &openings.to_vec(),
      &opening_point,
      &commitment.generators.gens_read_write,
      transcript,
      random_tape,
    );

    Self {
      a_read_write_opening: openings[0],
      v_read_opening: openings[1],
      v_write_opening: openings[2],
      t_read_opening: openings[3],
      t_write_opening: openings[4],
      read_write_opening_proof,
    }
  }

  fn verify_openings(
    &self,
    commitment: &MemoryCommitment<G>,
    opening_point: &Vec<F>,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    let combined_openings: Vec<F> = vec![
      self.a_read_write_opening.clone(),
      self.v_read_opening.clone(),
      self.v_write_opening.clone(),
      self.t_read_opening.clone(),
      self.t_write_opening.clone(),
    ];

    self.read_write_opening_proof.verify(
      opening_point,
      &combined_openings,
      &commitment.generators.gens_read_write,
      &commitment.read_write_commitments,
      transcript,
    )
  }
}

pub struct MemoryInitFinalOpenings<F, G>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
{
  a_init_final: Option<F>, // Computed by verifier
  v_final: F,
  t_final: F,

  init_final_opening_proof: CombinedTableEvalProof<G>,
}

impl<F, G> StructuredOpeningProof<F, G, ReadWriteMemory<F, G>> for MemoryInitFinalOpenings<F, G>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
{
  type Openings = [F; 2];

  fn open(polynomials: &ReadWriteMemory<F, G>, opening_point: &Vec<F>) -> Self::Openings {
    [
      polynomials.v_final.evaluate(&opening_point),
      polynomials.t_final.evaluate(&opening_point),
    ]
  }

  fn prove_openings(
    polynomials: &BatchedMemoryPolynomials<F>,
    commitment: &MemoryCommitment<G>,
    opening_point: &Vec<F>,
    openings: [F; 2],
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self {
    let init_final_opening_proof = CombinedTableEvalProof::prove(
      &polynomials.batched_init_final,
      &openings.to_vec(),
      &opening_point,
      &commitment.generators.gens_init_final,
      transcript,
      random_tape,
    );

    Self {
      a_init_final: None, // Computed by verifier
      v_final: openings[0],
      t_final: openings[1],
      init_final_opening_proof,
    }
  }

  fn verify_openings(
    &self,
    commitment: &MemoryCommitment<G>,
    opening_point: &Vec<F>,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    self.init_final_opening_proof.verify(
      opening_point,
      &vec![self.v_final, self.t_final],
      &commitment.generators.gens_init_final,
      &commitment.init_final_commitments,
      transcript,
    )
  }
}

impl<F, G> MemoryCheckingProver<F, G, Self> for ReadWriteMemory<F, G>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
{
  type ReadWriteOpenings = MemoryReadWriteOpenings<F, G>;
  type InitFinalOpenings = MemoryInitFinalOpenings<F, G>;

  // (a, v, t)
  type MemoryTuple = (F, F, F);

  fn fingerprint(inputs: &(F, F, F), gamma: &F, tau: &F) -> F {
    let (a, v, t) = *inputs;
    t * gamma.square() + v * *gamma + a - tau
  }

  fn read_leaves(&self, polynomials: &Self, gamma: &F, tau: &F) -> Vec<DensePolynomial<F>> {
    let num_ops = polynomials.a_read_write.len();
    let read_fingerprints = (0..num_ops)
      .map(|i| {
        <Self as MemoryCheckingProver<F, G, Self>>::fingerprint(
          &(
            polynomials.a_read_write[i],
            polynomials.v_read[i],
            polynomials.t_read[i],
          ),
          gamma,
          tau,
        )
      })
      .collect();
    vec![DensePolynomial::new(read_fingerprints)]
  }
  fn write_leaves(&self, polynomials: &Self, gamma: &F, tau: &F) -> Vec<DensePolynomial<F>> {
    let num_ops = polynomials.a_read_write.len();
    let write_fingerprints = (0..num_ops)
      .map(|i| {
        <Self as MemoryCheckingProver<F, G, Self>>::fingerprint(
          &(
            polynomials.a_read_write[i],
            polynomials.v_write[i],
            polynomials.t_write[i],
          ),
          gamma,
          tau,
        )
      })
      .collect();
    vec![DensePolynomial::new(write_fingerprints)]
  }
  fn init_leaves(&self, polynomials: &Self, gamma: &F, tau: &F) -> Vec<DensePolynomial<F>> {
    let init_fingerprints = (0..self.memory_size)
      .map(|i| {
        <Self as MemoryCheckingProver<F, G, Self>>::fingerprint(
          &(F::from(i as u64), F::zero(), F::zero()),
          gamma,
          tau,
        )
      })
      .collect();
    vec![DensePolynomial::new(init_fingerprints)]
  }
  fn final_leaves(&self, polynomials: &Self, gamma: &F, tau: &F) -> Vec<DensePolynomial<F>> {
    let final_fingerprints = (0..self.memory_size)
      .map(|i| {
        <Self as MemoryCheckingProver<F, G, Self>>::fingerprint(
          &(
            F::from(i as u64),
            polynomials.v_final[i],
            polynomials.t_final[i],
          ),
          gamma,
          tau,
        )
      })
      .collect();
    vec![DensePolynomial::new(final_fingerprints)]
  }

  fn protocol_name() -> &'static [u8] {
    b"Registers/RAM memory checking"
  }
}

impl<F, G> MemoryCheckingVerifier<F, G, Self> for ReadWriteMemory<F, G>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
{
  fn compute_verifier_openings(openings: &mut Self::InitFinalOpenings, opening_point: &Vec<F>) {
    openings.a_init_final =
      Some(IdentityPolynomial::new(opening_point.len()).evaluate(opening_point));
  }

  fn read_tuples(openings: &Self::ReadWriteOpenings) -> Vec<Self::MemoryTuple> {
    vec![(
      openings.a_read_write_opening,
      openings.v_read_opening,
      openings.t_read_opening,
    )]
  }
  fn write_tuples(openings: &Self::ReadWriteOpenings) -> Vec<Self::MemoryTuple> {
    vec![(
      openings.a_read_write_opening,
      openings.v_write_opening,
      openings.t_write_opening,
    )]
  }
  fn init_tuples(openings: &Self::InitFinalOpenings) -> Vec<Self::MemoryTuple> {
    vec![(openings.a_init_final.unwrap(), F::zero(), F::zero())]
  }
  fn final_tuples(openings: &Self::InitFinalOpenings) -> Vec<Self::MemoryTuple> {
    vec![(
      openings.a_init_final.unwrap(),
      openings.v_final,
      openings.t_final,
    )]
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use ark_curve25519::{EdwardsProjective, Fr};
  use ark_std::{log2, test_rng, One, Zero};
  use rand_chacha::rand_core::RngCore;

  fn generate_memory_trace(memory_size: usize, num_ops: usize) -> Vec<MemoryOp> {
    let mut rng = test_rng();
    let mut memory = vec![0u64; memory_size];
    let mut memory_trace = Vec::with_capacity(num_ops);

    for _ in 0..num_ops {
      if rng.next_u32() % 2 == 0 {
        let address: usize = rng.next_u32() as usize % memory_size;
        let value = memory[address];
        memory_trace.push(MemoryOp::Read(address as u64 + RAM_START_ADDRESS, value));
      } else {
        let address: usize = rng.next_u32() as usize % memory_size;
        let old_value = memory[address];
        let new_value = rng.next_u64();
        memory_trace.push(MemoryOp::Write(
          address as u64 + RAM_START_ADDRESS,
          old_value,
          new_value,
        ));
        memory[address] = new_value;
      }
    }
    memory_trace
  }

  #[test]
  fn e2e_memchecking() {
    const MEMORY_SIZE: usize = 1 << 16;
    const NUM_OPS: usize = 1 << 8;
    let memory_trace = generate_memory_trace(MEMORY_SIZE, NUM_OPS);

    let mut transcript = Transcript::new(b"test_transcript");
    let mut random_tape = RandomTape::new(b"test_tape");

    let (rw_memory, _): (ReadWriteMemory<Fr, EdwardsProjective>, Vec<u64>) =
      ReadWriteMemory::new(memory_trace, MEMORY_SIZE, &mut transcript);
    let batched_polys = rw_memory.batch();
    let commitments = ReadWriteMemory::commit(&batched_polys);

    let proof = rw_memory.prove_memory_checking(
      &rw_memory,
      &batched_polys,
      &commitments,
      &mut transcript,
      &mut random_tape,
    );

    let mut transcript = Transcript::new(b"test_transcript");
    ReadWriteMemory::verify_memory_checking(proof, &commitments, &mut transcript)
      .expect("proof should verify");
  }
}
