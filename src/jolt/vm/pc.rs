use std::marker::PhantomData;

use ark_ec::CurveGroup;
use ark_ff::{PrimeField, Field};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{Zero, One};
use merlin::Transcript;

use crate::{
  lasso::{fingerprint_strategy::{FingerprintStrategy, MemBatchInfo}, gp_evals::GPEvals},
  poly::{dense_mlpoly::{DensePolynomial, PolyCommitmentGens}, identity_poly::IdentityPolynomial},
  subprotocols::{
    combined_table_proof::{CombinedTableCommitment, CombinedTableEvalProof},
    grand_product::BGPCInterpretable,
  },
  utils::{self, errors::ProofVerifyError, math::Math, transcript::ProofTranscript},
};

pub struct PCPolys<F: PrimeField> {
  a_ops: DensePolynomial<F>,
  v_ops: DensePolynomial<F>,

  v_mem: DensePolynomial<F>,

  t_read: DensePolynomial<F>,
  t_final: DensePolynomial<F>,

  /// Starting address for a_ops
  address_offset: usize,

  /// Contains:
  /// - a_ops, v_ops (trace)
  /// - t_read
  combined_ops: DensePolynomial<F>,

  /// Contains:
  /// - v_mem (ELF)
  /// - t_final
  combined_mem: DensePolynomial<F>,
}

impl<F: PrimeField> PCPolys<F> {
  pub fn new(
    a_ops: DensePolynomial<F>,
    v_ops: DensePolynomial<F>,
    v_mem: DensePolynomial<F>,
    t_read: DensePolynomial<F>,
    t_final: DensePolynomial<F>,
    address_offset: usize,
  ) -> Self {
    // TODO(JOLT-48): DensePolynomial::merge should be parameterized with  Vec<&DensePolynomial> given it is read only. Avoids clones.
    let ops = [a_ops.clone(), v_ops.clone(), t_read.clone()];
    let combined_ops = DensePolynomial::merge(&ops);
    let mems = [v_mem.clone(), t_final.clone()];
    let combined_mem = DensePolynomial::merge(&mems);

    Self {
      a_ops,
      v_ops,
      v_mem,
      t_read,
      t_final,

      address_offset,

      combined_ops,
      combined_mem,
    }
  }

  pub fn commit<G: CurveGroup<ScalarField = F>>(
    &self,
  ) -> (ProgramCommitmentGens<G>, ProgramCommitment<G>) {
    let gens = ProgramCommitmentGens::new(self.ops_size(), self.mem_size());

    println!("committing ops");
    let (ops_sized_commitments, _) = self.combined_ops.commit(&gens.gens_ops, None);
    let ops_sized_commitments = CombinedTableCommitment::new(ops_sized_commitments);
    println!("Committed ops");
    let (mem_sized_commitments, _) = self.combined_mem.commit(&gens.gens_mem, None);
    let mem_sized_commitments = CombinedTableCommitment::new(mem_sized_commitments);

    let commitments = ProgramCommitment {
      ops_sized_commitments,
      mem_sized_commitments,
    };

    (gens, commitments)
  }
}

pub struct PCProof<F: PrimeField> {
  _marker: PhantomData<F>,
}

pub struct ProgramCommitment<G: CurveGroup> {
  // Contains:
  // - a_ops, v_ops (trace)
  // - t_read
  pub ops_sized_commitments: CombinedTableCommitment<G>,

  /// Contains:
  /// - v_mem (ELF)
  /// - t_final
  pub mem_sized_commitments: CombinedTableCommitment<G>,
}

pub struct ProgramCommitmentGens<G: CurveGroup> {
  pub gens_ops: PolyCommitmentGens<G>,
  pub gens_mem: PolyCommitmentGens<G>,
}

impl<G: CurveGroup> ProgramCommitmentGens<G> {
  pub fn new(ops_size: usize, mem_size: usize) -> Self {
    debug_assert!(utils::is_power_of_two(ops_size));
    debug_assert!(utils::is_power_of_two(mem_size));

    let num_vars_ops = (3 * ops_size).log_2();
    let num_vars_mem = (2 * mem_size).log_2();

    let gens_ops = PolyCommitmentGens::new(num_vars_ops, b"ops_commitment");
    let gens_mem = PolyCommitmentGens::new(num_vars_mem, b"mem_commitment");

    ProgramCommitmentGens { gens_ops, gens_mem }
  }
}

impl<F: PrimeField> MemBatchInfo for PCPolys<F> {
  fn ops_size(&self) -> usize {
    self.a_ops.len()
  }

  fn mem_size(&self) -> usize {
    self.v_mem.len()
  }

  fn num_memories(&self) -> usize {
    1
  }
}

impl<F: PrimeField> BGPCInterpretable<F> for PCPolys<F> {
  fn a_ops(&self, memory_index: usize, leaf_index: usize) -> F {
    debug_assert_eq!(memory_index, 0);
    self.a_ops[leaf_index]
  }

  fn a_mem(&self, memory_index: usize, leaf_index: usize) -> F {
    debug_assert_eq!(memory_index, 0);
    F::from((self.address_offset + leaf_index) as u64)
  }

  fn v_mem(&self, memory_index: usize, leaf_index: usize) -> F {
    debug_assert_eq!(memory_index, 0);
    self.v_mem[leaf_index]
  }

  fn v_ops(&self, memory_index: usize, leaf_index: usize) -> F {
    debug_assert_eq!(memory_index, 0);
    self.v_ops[leaf_index]
  }

  fn t_final(&self, memory_index: usize, leaf_index: usize) -> F {
    assert_eq!(memory_index, 0);
    self.t_final[leaf_index]
  }

  fn t_read(&self, memory_index: usize, leaf_index: usize) -> F {
    assert_eq!(memory_index, 0);
    self.t_read[leaf_index]
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PCFingerprintProof<G: CurveGroup> {
  eval_a_ops: G::ScalarField,
  eval_v_ops: G::ScalarField,
  eval_v_mem: G::ScalarField,
  eval_read: G::ScalarField,
  eval_final: G::ScalarField,

  proof_ops: CombinedTableEvalProof<G>,
  proof_mem: CombinedTableEvalProof<G>,
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

    let eval_a_ops = polynomials.a_ops.evaluate(&rand_ops);
    let eval_v_ops = polynomials.v_ops.evaluate(&rand_ops);
    let eval_v_mem = polynomials.v_mem.evaluate(&rand_mem);
    let eval_read = polynomials.t_read.evaluate(&rand_ops);
    let eval_final = polynomials.t_final.evaluate(&rand_mem);

    let evals_ops: Vec<G::ScalarField> =
      vec![eval_a_ops.clone(), eval_v_ops.clone(), eval_read.clone()];
    let evals_mem: Vec<G::ScalarField> = vec![eval_v_mem.clone(), eval_final.clone()];

    let proof_ops = CombinedTableEvalProof::prove(
      &polynomials.combined_ops,
      &evals_ops,
      &rand_ops,
      &generators.gens_ops,
      transcript,
      random_tape,
    );
    let proof_mem = CombinedTableEvalProof::prove(
      &polynomials.combined_mem,
      &evals_mem,
      &rand_mem,
      &generators.gens_mem,
      transcript,
      random_tape,
    );

    Self {
      eval_a_ops,
      eval_v_ops,
      eval_v_mem,
      eval_read,
      eval_final,
      proof_ops,
      proof_mem,
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

    let evals_ops = vec![self.eval_a_ops, self.eval_v_ops, self.eval_read];
    self.proof_ops.verify(rand_ops, &evals_ops, &generators.gens_ops, &commitments.ops_sized_commitments, transcript)?;

    let evals_mem = vec![self.eval_v_mem, self.eval_final];
    self.proof_mem.verify(rand_mem, &evals_mem, &generators.gens_mem, &commitments.mem_sized_commitments, transcript)?;

    let hash= |a: G::ScalarField, v: G::ScalarField, t: G::ScalarField| -> G::ScalarField {
        t * r_hash.square() + v * *r_hash + a - r_multiset_check 
    };

    debug_assert_eq!(grand_product_claims.len(), 1);
    let claim = &grand_product_claims[0];
    // TODO(JOLT-46): Doesn't work for offsets
    let a_mem = IdentityPolynomial::new(rand_mem.len()).evaluate(rand_mem);
    let hash_init = hash(a_mem, self.eval_v_mem, G::ScalarField::zero());
    assert_eq!(claim.hash_init, hash_init);
    let hash_read = hash(self.eval_a_ops, self.eval_v_ops, self.eval_read);
    assert_eq!(claim.hash_read, hash_read);
    let hash_write = hash(self.eval_a_ops, self.eval_v_ops, self.eval_read + G::ScalarField::one());
    assert_eq!(claim.hash_write, hash_write);
    let hash_final = hash(a_mem, self.eval_v_mem, self.eval_final);
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
  use crate::{
    lasso::memory_checking::{MemoryCheckingProof, ProductLayerProof},
    poly::dense_mlpoly::DensePolynomial,
    utils::random::RandomTape,
  };
  use ark_curve25519::{EdwardsProjective, Fr};
  use ark_std::{One, Zero};
  use merlin::Transcript;

  use super::{PCFingerprintProof, PCPolys};

  #[test]
  fn prod_layer_proof() {
    // Imagine a size-8 range-check table (addresses and values just ascending), with 4 lookups
    let v_mems = vec![
      Fr::from(0),
      Fr::from(1),
      Fr::from(2),
      Fr::from(3),
      Fr::from(4),
      Fr::from(5),
      Fr::from(6),
      Fr::from(7),
    ];

    // 2 lookups into the last 2 elements of memory each
    let a_ops = vec![Fr::from(6), Fr::from(7), Fr::from(6), Fr::from(7)];
    let v_ops = a_ops.clone();

    let t_reads = vec![Fr::zero(), Fr::zero(), Fr::one(), Fr::one()];
    let t_finals = vec![
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::from(2),
      Fr::from(2),
    ];

    let polys = PCPolys::new(
      DensePolynomial::new(a_ops),
      DensePolynomial::new(v_ops),
      DensePolynomial::new(v_mems),
      DensePolynomial::new(t_reads),
      DensePolynomial::new(t_finals),
      0,
    );
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
  fn prod_layer_proof_offset() {
    // Imagine a size-8 range-check table (values just ascending), with 4 lookups.
    // Addresses start at 1000, values start at 0
    let v_mems = vec![
      Fr::from(0),
      Fr::from(1),
      Fr::from(2),
      Fr::from(3),
      Fr::from(4),
      Fr::from(5),
      Fr::from(6),
      Fr::from(7),
    ];

    // 2 lookups into the last 2 elements of memory each. Addresses start from 1000
    let a_ops = vec![
      Fr::from(1006),
      Fr::from(1007),
      Fr::from(1006),
      Fr::from(1007),
    ];
    let v_ops = vec![Fr::from(6), Fr::from(7), Fr::from(6), Fr::from(7)];

    let t_reads = vec![Fr::zero(), Fr::zero(), Fr::one(), Fr::one()];
    let t_finals = vec![
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::from(2),
      Fr::from(2),
    ];

    let polys = PCPolys::new(
      DensePolynomial::new(a_ops),
      DensePolynomial::new(v_ops),
      DensePolynomial::new(v_mems),
      DensePolynomial::new(t_reads),
      DensePolynomial::new(t_finals),
      1000, // Big change here!
    );
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
  fn e2e_mem_checking() {
    // Imagine a size-8 range-check table (addresses and values just ascending), with 4 lookups
    let v_mems = vec![
      Fr::from(0),
      Fr::from(1),
      Fr::from(2),
      Fr::from(3),
      Fr::from(4),
      Fr::from(5),
      Fr::from(6),
      Fr::from(7),
    ];

    // 2 lookups into the last 2 elements of memory each
    let a_ops = vec![Fr::from(6), Fr::from(7), Fr::from(6), Fr::from(7)];
    let v_ops = a_ops.clone();

    let t_reads = vec![Fr::zero(), Fr::zero(), Fr::one(), Fr::one()];
    let t_finals = vec![
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::from(2),
      Fr::from(2),
    ];

    let polys = PCPolys::new(
      DensePolynomial::new(a_ops),
      DensePolynomial::new(v_ops),
      DensePolynomial::new(v_mems),
      DensePolynomial::new(t_reads),
      DensePolynomial::new(t_finals),
      0,
    );
    let (generators, commitments) = polys.commit();
    let mut transcript = Transcript::new(b"test_transcript");
    let mut random_tape = RandomTape::new(b"test_tape");
    let r_fingerprints = (&Fr::from(12), &Fr::from(35));
    let memory_checking_proof =
      MemoryCheckingProof::<EdwardsProjective, PCFingerprintProof<EdwardsProjective>>::prove(
        &polys,
        r_fingerprints,
        &generators,
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
        &generators,
        memory_to_dimension_index,
        evaluate_memory_mle,
        r_fingerprints,
        &mut transcript,
      )
      .expect("should verify");
  }

  #[test]
  fn e2e_mem_checking_offset() {
    // Imagine a size-8 range-check table (addresses and values just ascending), with 4 lookups
    let v_mems = vec![
      Fr::from(0),
      Fr::from(1),
      Fr::from(2),
      Fr::from(3),
      Fr::from(4),
      Fr::from(5),
      Fr::from(6),
      Fr::from(7),
    ];

    // 2 lookups into the last 2 elements of memory each. Addresses start from 1000
    let a_ops = vec![
        Fr::from(1006),
        Fr::from(1007),
        Fr::from(1006),
        Fr::from(1007),
    ];
    let v_ops = vec![Fr::from(6), Fr::from(7), Fr::from(6), Fr::from(7)];

    let t_reads = vec![Fr::zero(), Fr::zero(), Fr::one(), Fr::one()];
    let t_finals = vec![
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::from(2),
      Fr::from(2),
    ];

    let polys = PCPolys::new(
      DensePolynomial::new(a_ops),
      DensePolynomial::new(v_ops),
      DensePolynomial::new(v_mems),
      DensePolynomial::new(t_reads),
      DensePolynomial::new(t_finals),
      1000,
    );
    let (generators, commitments) = polys.commit();
    let mut transcript = Transcript::new(b"test_transcript");
    let mut random_tape = RandomTape::new(b"test_tape");
    let r_fingerprints = (&Fr::from(12), &Fr::from(35));
    let memory_checking_proof =
      MemoryCheckingProof::<EdwardsProjective, PCFingerprintProof<EdwardsProjective>>::prove(
        &polys,
        r_fingerprints,
        &generators,
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
        &generators,
        memory_to_dimension_index,
        evaluate_memory_mle,
        r_fingerprints,
        &mut transcript,
      )
      .expect("should verify");
  }
}
