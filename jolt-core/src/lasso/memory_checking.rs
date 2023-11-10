#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::poly::{
  dense_mlpoly::DensePolynomial,
  structured_poly::{StructuredOpeningProof, StructuredPolynomials},
};
use crate::subprotocols::grand_product::{
  BatchedGrandProductArgument, BatchedGrandProductCircuit, GrandProductCircuit,
};
use crate::utils::errors::ProofVerifyError;
use crate::utils::random::RandomTape;
use crate::utils::transcript::ProofTranscript;

use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use merlin::Transcript;
use std::marker::PhantomData;

use super::fingerprint_strategy::FingerprintStrategy;

struct MultisetHashes<F: PrimeField> {
  hash_init: F,
  hash_final: F,
  hash_read: F,
  hash_write: F,
}

pub struct MemoryCheckingProof<G, Polynomials, ReadWriteOpenings, InitFinalOpenings>
where
  G: CurveGroup,
  Polynomials: StructuredPolynomials + ?Sized,
  ReadWriteOpenings: StructuredOpeningProof<G::ScalarField, G, Polynomials>,
  InitFinalOpenings: StructuredOpeningProof<G::ScalarField, G, Polynomials>,
{
  _polys: PhantomData<Polynomials>,
  multiset_hashes: Vec<MultisetHashes<G::ScalarField>>,
  read_write_grand_product: BatchedGrandProductArgument<G::ScalarField>,
  init_final_grand_product: BatchedGrandProductArgument<G::ScalarField>,
  read_write_openings: ReadWriteOpenings,
  init_final_openings: InitFinalOpenings,
}

pub trait MemoryCheckingProver<F, G, Polynomials>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
  Polynomials: StructuredPolynomials,
{
  type ReadWriteOpenings: StructuredOpeningProof<F, G, Polynomials>;
  type InitFinalOpenings: StructuredOpeningProof<F, G, Polynomials>;
  type MemoryTuple = (F, F, F); // (a, v, t)

  fn prove_memory_checking(
    &self,
    polynomials: &Polynomials,
    batched_polys: &Polynomials::BatchedPolynomials,
    commitments: &Polynomials::Commitment,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> MemoryCheckingProof<G, Polynomials, Self::ReadWriteOpenings, Self::InitFinalOpenings> {
    // Fiat-Shamir randomness for multiset hashes
    let gamma: F =
      <Transcript as ProofTranscript<G>>::challenge_scalar(transcript, b"Memory checking gamma");
    let tau: F =
      <Transcript as ProofTranscript<G>>::challenge_scalar(transcript, b"Memory checking tau");

    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    // fka "ProductLayerProof"
    let (read_write_circuit, read_hashes, write_hashes) =
      self.read_write_grand_product(polynomials, &gamma, &tau);
    let (init_final_circuit, init_hashes, final_hashes) =
      self.init_final_grand_product(polynomials, &gamma, &tau);
    debug_assert_eq!(read_hashes.len(), init_hashes.len());
    let num_memories = read_hashes.len();

    let mut multiset_hashes = Vec::with_capacity(num_memories);
    for i in 0..num_memories {
      let hashes = MultisetHashes {
        hash_init: init_hashes[i],
        hash_final: final_hashes[i],
        hash_read: read_hashes[i],
        hash_write: write_hashes[i],
      };
      debug_assert_eq!(
        hashes.hash_init * hashes.hash_write,
        hashes.hash_final * hashes.hash_read,
        "Multiset hashes don't match"
      );
      multiset_hashes.push(hashes);
      // TODO: append to transcript
    }

    let (read_write_grand_product, r_read_write) =
      BatchedGrandProductArgument::prove::<G>(read_write_circuit, transcript);
    let (init_final_grand_product, r_init_final) =
      BatchedGrandProductArgument::prove::<G>(init_final_circuit, transcript);

    // fka "HashLayerProof"
    let read_write_openings = Self::ReadWriteOpenings::prove_openings(
      batched_polys,
      commitments,
      &r_read_write,
      Self::ReadWriteOpenings::open(polynomials, &r_read_write),
      transcript,
      random_tape,
    );
    let init_final_openings = Self::InitFinalOpenings::prove_openings(
      batched_polys,
      commitments,
      &r_init_final,
      Self::InitFinalOpenings::open(polynomials, &r_init_final),
      transcript,
      random_tape,
    );

    MemoryCheckingProof {
      _polys: PhantomData,
      multiset_hashes,
      read_write_grand_product,
      init_final_grand_product,
      read_write_openings,
      init_final_openings,
    }
  }

  fn read_write_grand_product(
    &self,
    polynomials: &Polynomials,
    gamma: &F,
    tau: &F,
  ) -> (BatchedGrandProductCircuit<F>, Vec<F>, Vec<F>) {
    let read_leaves: Vec<DensePolynomial<F>> = self.read_leaves(polynomials, gamma, tau);
    let write_leaves: Vec<DensePolynomial<F>> = self.write_leaves(polynomials, gamma, tau);
    debug_assert_eq!(read_leaves.len(), write_leaves.len());
    let num_memories = read_leaves.len();

    let mut circuits = Vec::with_capacity(2 * num_memories);
    let mut read_hashes = Vec::with_capacity(num_memories);
    let mut write_hashes = Vec::with_capacity(num_memories);
    for i in 0..num_memories {
      let read_circuit = GrandProductCircuit::new(&read_leaves[i]);
      let write_circuit = GrandProductCircuit::new(&write_leaves[i]);
      read_hashes.push(read_circuit.evaluate());
      write_hashes.push(write_circuit.evaluate());
      circuits.push(read_circuit);
      circuits.push(write_circuit);
    }

    (
      BatchedGrandProductCircuit::new_batch(circuits),
      read_hashes,
      write_hashes,
    )
  }

  fn init_final_grand_product(
    &self,
    polynomials: &Polynomials,
    gamma: &F,
    tau: &F,
  ) -> (BatchedGrandProductCircuit<F>, Vec<F>, Vec<F>) {
    let init_leaves: Vec<DensePolynomial<F>> = self.init_leaves(polynomials, gamma, tau);
    let final_leaves: Vec<DensePolynomial<F>> = self.final_leaves(polynomials, gamma, tau);
    debug_assert_eq!(init_leaves.len(), final_leaves.len());
    let num_memories = init_leaves.len();

    let mut circuits = Vec::with_capacity(2 * num_memories);
    let mut init_hashes = Vec::with_capacity(num_memories);
    let mut final_hashes = Vec::with_capacity(num_memories);
    for i in 0..num_memories {
      let init_circuit = GrandProductCircuit::new(&init_leaves[i]);
      let final_circuit = GrandProductCircuit::new(&final_leaves[i]);
      init_hashes.push(init_circuit.evaluate());
      final_hashes.push(final_circuit.evaluate());
      circuits.push(init_circuit);
      circuits.push(final_circuit);
    }

    (
      BatchedGrandProductCircuit::new_batch(circuits),
      init_hashes,
      final_hashes,
    )
  }

  fn read_leaves(&self, polynomials: &Polynomials, gamma: &F, tau: &F) -> Vec<DensePolynomial<F>>;
  fn write_leaves(&self, polynomials: &Polynomials, gamma: &F, tau: &F) -> Vec<DensePolynomial<F>>;
  fn init_leaves(&self, polynomials: &Polynomials, gamma: &F, tau: &F) -> Vec<DensePolynomial<F>>;
  fn final_leaves(&self, polynomials: &Polynomials, gamma: &F, tau: &F) -> Vec<DensePolynomial<F>>;
  fn fingerprint(tuple: &Self::MemoryTuple, gamma: &F, tau: &F) -> F;
  fn protocol_name() -> &'static [u8];
}

pub trait MemoryCheckingVerifier<F, G, Polynomials>:
  MemoryCheckingProver<F, G, Polynomials>
where
  F: PrimeField,
  G: CurveGroup<ScalarField = F>,
  Polynomials: StructuredPolynomials,
{
  fn verify_memory_checking(
    mut proof: MemoryCheckingProof<
      G,
      Polynomials,
      Self::ReadWriteOpenings,
      Self::InitFinalOpenings,
    >,
    commitments: &Polynomials::Commitment,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    // Fiat-Shamir randomness for multiset hashes
    let gamma: F =
      <Transcript as ProofTranscript<G>>::challenge_scalar(transcript, b"Memory checking gamma");
    let tau: F =
      <Transcript as ProofTranscript<G>>::challenge_scalar(transcript, b"Memory checking tau");

    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    for hash in &proof.multiset_hashes {
      // Multiset equality check
      assert_eq!(
        hash.hash_init * hash.hash_write,
        hash.hash_read * hash.hash_final
      );
      // TODO: append to transcript
    }

    let interleaved_read_write_hashes = proof
      .multiset_hashes
      .iter()
      .flat_map(|hash| [hash.hash_read, hash.hash_write])
      .collect();
    let interleaved_init_final_hashes = proof
      .multiset_hashes
      .iter()
      .flat_map(|hash| [hash.hash_init, hash.hash_final])
      .collect();

    let (claims_read_write, r_read_write) = proof
      .read_write_grand_product
      .verify::<G, Transcript>(&interleaved_read_write_hashes, transcript);
    let (claims_init_final, r_init_final) = proof
      .init_final_grand_product
      .verify::<G, Transcript>(&interleaved_init_final_hashes, transcript);

    proof
      .read_write_openings
      .verify_openings(commitments, &r_read_write, transcript)?;
    proof
      .init_final_openings
      .verify_openings(commitments, &r_init_final, transcript)?;

    Self::compute_verifier_openings(&mut proof.init_final_openings, &r_init_final);

    assert_eq!(claims_read_write.len(), claims_init_final.len());
    assert!(claims_read_write.len() % 2 == 0);
    let num_memories = claims_read_write.len() / 2;
    let grand_product_claims: Vec<MultisetHashes<F>> = (0..num_memories)
      .map(|i| MultisetHashes {
        hash_read: claims_read_write[2 * i],
        hash_write: claims_read_write[2 * i + 1],
        hash_init: claims_init_final[2 * i],
        hash_final: claims_init_final[2 * i + 1],
      })
      .collect();
    Self::check_fingerprints(
      grand_product_claims,
      &proof.read_write_openings,
      &proof.init_final_openings,
      &gamma,
      &tau,
    );

    Ok(())
  }

  fn compute_verifier_openings(openings: &mut Self::InitFinalOpenings, opening_point: &Vec<F>);
  fn read_tuples(openings: &Self::ReadWriteOpenings) -> Vec<Self::MemoryTuple>;
  fn write_tuples(openings: &Self::ReadWriteOpenings) -> Vec<Self::MemoryTuple>;
  fn init_tuples(openings: &Self::InitFinalOpenings) -> Vec<Self::MemoryTuple>;
  fn final_tuples(openings: &Self::InitFinalOpenings) -> Vec<Self::MemoryTuple>;

  fn check_fingerprints(
    claims: Vec<MultisetHashes<F>>,
    read_write_openings: &Self::ReadWriteOpenings,
    init_final_openings: &Self::InitFinalOpenings,
    gamma: &F,
    tau: &F,
  ) {
    let read_fingerprints: Vec<_> =
      <Self as MemoryCheckingVerifier<_, _, _>>::read_tuples(read_write_openings)
        .iter()
        .map(|tuple| Self::fingerprint(tuple, gamma, tau))
        .collect();
    let write_fingerprints: Vec<_> =
      <Self as MemoryCheckingVerifier<_, _, _>>::write_tuples(read_write_openings)
        .iter()
        .map(|tuple| Self::fingerprint(tuple, gamma, tau))
        .collect();
    let init_fingerprints: Vec<_> =
      <Self as MemoryCheckingVerifier<_, _, _>>::init_tuples(init_final_openings)
        .iter()
        .map(|tuple| Self::fingerprint(tuple, gamma, tau))
        .collect();
    let final_fingerprints: Vec<_> =
      <Self as MemoryCheckingVerifier<_, _, _>>::final_tuples(init_final_openings)
        .iter()
        .map(|tuple| Self::fingerprint(tuple, gamma, tau))
        .collect();
    for (i, claim) in claims.iter().enumerate() {
      assert_eq!(claim.hash_read, read_fingerprints[i]);
      assert_eq!(claim.hash_write, write_fingerprints[i]);
      assert_eq!(claim.hash_init, init_fingerprints[i]);
      assert_eq!(claim.hash_final, final_fingerprints[i]);
    }
  }
}

#[cfg(test)]
mod tests {
  use crate::{
    poly::dense_mlpoly::DensePolynomial,
    subprotocols::grand_product::{
      BGPCInterpretable, BatchedGrandProductCircuit, GPEvals, GrandProductCircuit,
    },
  };
  use ark_curve25519::{EdwardsProjective, Fr};
  use ark_std::{One, Zero};
  use merlin::Transcript;

  use super::ProductLayerProof;

  // #[test]
  // fn product_layer_proof_trivial() {
  //   // Define the most trivial GrandProduct memory checking layout
  //   struct NormalMems {
  //     a_ops: Vec<Fr>,

  //     v_ops: Vec<Fr>,
  //     v_mems: Vec<Fr>,

  //     t_reads: Vec<Fr>,
  //     t_finals: Vec<Fr>,
  //   }

  //   impl MemBatchInfo for NormalMems {
  //     fn mem_size(&self) -> usize {
  //       assert_eq!(self.v_mems.len(), self.t_finals.len());
  //       self.v_mems.len()
  //     }

  //     fn ops_size(&self) -> usize {
  //       assert_eq!(self.a_ops.len(), self.v_ops.len());
  //       assert_eq!(self.a_ops.len(), self.t_reads.len());
  //       self.a_ops.len()
  //     }

  //     fn num_memories(&self) -> usize {
  //       1
  //     }
  //   }

  //   impl BGPCInterpretable<Fr> for NormalMems {
  //     fn a_ops(&self, memory_index: usize, leaf_index: usize) -> Fr {
  //       assert_eq!(memory_index, 0);
  //       self.a_ops[leaf_index]
  //     }

  //     fn v_mem(&self, memory_index: usize, leaf_index: usize) -> Fr {
  //       assert_eq!(memory_index, 0);
  //       self.v_mems[leaf_index]
  //     }

  //     fn v_ops(&self, memory_index: usize, leaf_index: usize) -> Fr {
  //       assert_eq!(memory_index, 0);
  //       self.v_ops[leaf_index]
  //     }

  //     fn t_final(&self, memory_index: usize, leaf_index: usize) -> Fr {
  //       assert_eq!(memory_index, 0);
  //       self.t_finals[leaf_index]
  //     }

  //     fn t_read(&self, memory_index: usize, leaf_index: usize) -> Fr {
  //       assert_eq!(memory_index, 0);
  //       self.t_reads[leaf_index]
  //     }
  //   }

  //   // Imagine a size-8 range-check table (addresses and values just ascending), with 4 lookups
  //   let v_mems = vec![
  //     Fr::from(0),
  //     Fr::from(1),
  //     Fr::from(2),
  //     Fr::from(3),
  //     Fr::from(4),
  //     Fr::from(5),
  //     Fr::from(6),
  //     Fr::from(7),
  //   ];

  //   // 2 lookups into the last 2 elements of memory each
  //   let a_ops = vec![Fr::from(6), Fr::from(7), Fr::from(6), Fr::from(7)];
  //   let v_ops = a_ops.clone();

  //   let t_reads = vec![Fr::zero(), Fr::zero(), Fr::one(), Fr::one()];
  //   let t_finals = vec![
  //     Fr::zero(),
  //     Fr::zero(),
  //     Fr::zero(),
  //     Fr::zero(),
  //     Fr::zero(),
  //     Fr::zero(),
  //     Fr::from(2),
  //     Fr::from(2),
  //   ];

  //   let polys = NormalMems {
  //     a_ops,
  //     v_ops,
  //     v_mems,
  //     t_reads,
  //     t_finals,
  //   };

  //   let mut transcript = Transcript::new(b"test_transcript");
  //   let r_fingerprints = (&Fr::from(12), &Fr::from(35));
  //   let (proof, _, _) =
  //     ProductLayerProof::prove::<EdwardsProjective, _>(&polys, r_fingerprints, &mut transcript);

  //   let mut transcript = Transcript::new(b"test_transcript");
  //   proof
  //     .verify::<EdwardsProjective>(&mut transcript)
  //     .expect("proof should work");
  // }

  // #[test]
  // fn product_layer_proof_batched() {
  //   // Define a GrandProduct circuit that can be batched across 2 memories
  //   struct NormalMems {
  //     a_0_ops: Vec<Fr>,
  //     a_1_ops: Vec<Fr>,

  //     v_0_ops: Vec<Fr>,
  //     v_1_ops: Vec<Fr>,
  //     v_mems: Vec<Fr>,

  //     t_0_reads: Vec<Fr>,
  //     t_1_reads: Vec<Fr>,

  //     t_0_finals: Vec<Fr>,
  //     t_1_finals: Vec<Fr>,
  //   }

  //   impl MemBatchInfo for NormalMems {
  //     fn mem_size(&self) -> usize {
  //       assert_eq!(self.v_mems.len(), self.t_0_finals.len());
  //       assert_eq!(self.v_mems.len(), self.t_1_finals.len());
  //       self.v_mems.len()
  //     }

  //     fn ops_size(&self) -> usize {
  //       let ops_len = self.a_0_ops.len();
  //       assert_eq!(ops_len, self.a_1_ops.len());
  //       assert_eq!(ops_len, self.v_0_ops.len());
  //       assert_eq!(ops_len, self.v_1_ops.len());
  //       assert_eq!(ops_len, self.t_0_reads.len());
  //       assert_eq!(ops_len, self.t_1_reads.len());

  //       ops_len
  //     }

  //     fn num_memories(&self) -> usize {
  //       2
  //     }
  //   }

  //   impl BGPCInterpretable<Fr> for NormalMems {
  //     fn a_ops(&self, memory_index: usize, leaf_index: usize) -> Fr {
  //       assert!(memory_index < 2);
  //       match memory_index {
  //         0 => self.a_0_ops[leaf_index],
  //         1 => self.a_1_ops[leaf_index],
  //         _ => panic!("waaa"),
  //       }
  //     }

  //     fn v_mem(&self, memory_index: usize, leaf_index: usize) -> Fr {
  //       assert!(memory_index < 2);
  //       self.v_mems[leaf_index]
  //     }

  //     fn v_ops(&self, memory_index: usize, leaf_index: usize) -> Fr {
  //       assert!(memory_index < 2);
  //       match memory_index {
  //         0 => self.v_0_ops[leaf_index],
  //         1 => self.v_1_ops[leaf_index],
  //         _ => panic!("waaa"),
  //       }
  //     }

  //     fn t_final(&self, memory_index: usize, leaf_index: usize) -> Fr {
  //       assert!(memory_index < 2);
  //       match memory_index {
  //         0 => self.t_0_finals[leaf_index],
  //         1 => self.t_1_finals[leaf_index],
  //         _ => panic!("waaa"),
  //       }
  //     }

  //     fn t_read(&self, memory_index: usize, leaf_index: usize) -> Fr {
  //       assert!(memory_index < 2);
  //       match memory_index {
  //         0 => self.t_0_reads[leaf_index],
  //         1 => self.t_1_reads[leaf_index],
  //         _ => panic!("waaa"),
  //       }
  //     }
  //   }

  //   // Imagine a 2 memories. Size-8 range-check table (addresses and values just ascending), with 4 lookups into each
  //   let v_mems = vec![
  //     Fr::from(0),
  //     Fr::from(1),
  //     Fr::from(2),
  //     Fr::from(3),
  //     Fr::from(4),
  //     Fr::from(5),
  //     Fr::from(6),
  //     Fr::from(7),
  //   ];

  //   // 2 lookups into the last 2 elements of memory each
  //   let a_0_ops = vec![Fr::from(6), Fr::from(7), Fr::from(6), Fr::from(7)];
  //   let a_1_ops = vec![Fr::from(0), Fr::from(1), Fr::from(0), Fr::from(2)];
  //   let v_0_ops = a_0_ops.clone();
  //   let v_1_ops = a_1_ops.clone();

  //   let t_0_reads = vec![Fr::zero(), Fr::zero(), Fr::one(), Fr::one()];
  //   let t_1_reads = vec![Fr::zero(), Fr::zero(), Fr::one(), Fr::zero()];
  //   let t_0_finals = vec![
  //     Fr::zero(),
  //     Fr::zero(),
  //     Fr::zero(),
  //     Fr::zero(),
  //     Fr::zero(),
  //     Fr::zero(),
  //     Fr::from(2),
  //     Fr::from(2),
  //   ];
  //   let t_1_finals = vec![
  //     Fr::from(2),
  //     Fr::one(),
  //     Fr::one(),
  //     Fr::zero(),
  //     Fr::zero(),
  //     Fr::zero(),
  //     Fr::zero(),
  //     Fr::zero(),
  //   ];

  //   let polys = NormalMems {
  //     a_0_ops,
  //     a_1_ops,
  //     v_0_ops,
  //     v_1_ops,
  //     v_mems,
  //     t_0_reads,
  //     t_1_reads,
  //     t_0_finals,
  //     t_1_finals,
  //   };

  //   let mut transcript = Transcript::new(b"test_transcript");
  //   let r_fingerprints = (&Fr::from(12), &Fr::from(35));
  //   let (proof, _, _) =
  //     ProductLayerProof::prove::<EdwardsProjective, _>(&polys, r_fingerprints, &mut transcript);

  //   let mut transcript = Transcript::new(b"test_transcript");
  //   proof
  //     .verify::<EdwardsProjective>(&mut transcript)
  //     .expect("proof should work");
  // }

  // #[test]
  // fn product_layer_proof_flags_no_reuse() {
  //   // Define a GrandProduct circuit that can be batched across 2 memories
  //   struct FlagMems {
  //     a_0_ops: Vec<Fr>,
  //     a_1_ops: Vec<Fr>,

  //     v_0_ops: Vec<Fr>,
  //     v_1_ops: Vec<Fr>,
  //     v_mems: Vec<Fr>,

  //     t_0_reads: Vec<Fr>,
  //     t_1_reads: Vec<Fr>,

  //     t_0_finals: Vec<Fr>,
  //     t_1_finals: Vec<Fr>,

  //     flags_0: Vec<Fr>,
  //     flags_1: Vec<Fr>,
  //   }

  //   impl MemBatchInfo for FlagMems {
  //     fn mem_size(&self) -> usize {
  //       assert_eq!(self.v_mems.len(), self.t_0_finals.len());
  //       assert_eq!(self.v_mems.len(), self.t_1_finals.len());
  //       self.v_mems.len()
  //     }

  //     fn ops_size(&self) -> usize {
  //       let ops_len = self.a_0_ops.len();
  //       assert_eq!(ops_len, self.a_1_ops.len());
  //       assert_eq!(ops_len, self.v_0_ops.len());
  //       assert_eq!(ops_len, self.v_1_ops.len());
  //       assert_eq!(ops_len, self.t_0_reads.len());
  //       assert_eq!(ops_len, self.t_1_reads.len());
  //       assert_eq!(ops_len, self.flags_0.len());
  //       assert_eq!(ops_len, self.flags_1.len());

  //       ops_len
  //     }

  //     fn num_memories(&self) -> usize {
  //       2
  //     }
  //   }

  //   impl BGPCInterpretable<Fr> for FlagMems {
  //     fn a_ops(&self, memory_index: usize, leaf_index: usize) -> Fr {
  //       assert!(memory_index < 2);
  //       match memory_index {
  //         0 => self.a_0_ops[leaf_index],
  //         1 => self.a_1_ops[leaf_index],
  //         _ => panic!("waaa"),
  //       }
  //     }

  //     fn v_mem(&self, memory_index: usize, leaf_index: usize) -> Fr {
  //       assert!(memory_index < 2);
  //       self.v_mems[leaf_index]
  //     }

  //     fn v_ops(&self, memory_index: usize, leaf_index: usize) -> Fr {
  //       assert!(memory_index < 2);
  //       match memory_index {
  //         0 => self.v_0_ops[leaf_index],
  //         1 => self.v_1_ops[leaf_index],
  //         _ => panic!("waaa"),
  //       }
  //     }

  //     fn t_final(&self, memory_index: usize, leaf_index: usize) -> Fr {
  //       assert!(memory_index < 2);
  //       match memory_index {
  //         0 => self.t_0_finals[leaf_index],
  //         1 => self.t_1_finals[leaf_index],
  //         _ => panic!("waaa"),
  //       }
  //     }

  //     fn t_read(&self, memory_index: usize, leaf_index: usize) -> Fr {
  //       assert!(memory_index < 2);
  //       match memory_index {
  //         0 => self.t_0_reads[leaf_index],
  //         1 => self.t_1_reads[leaf_index],
  //         _ => panic!("waaa"),
  //       }
  //     }

  //     // FLAGS OVERRIDES

  //     fn construct_batches(
  //       &self,
  //       r_hash: (&Fr, &Fr),
  //     ) -> (
  //       BatchedGrandProductCircuit<Fr>,
  //       BatchedGrandProductCircuit<Fr>,
  //       Vec<GPEvals<Fr>>,
  //     ) {
  //       // compute leaves for all the batches                     (shared)
  //       // convert the rw leaves to flagged leaves                (custom)
  //       // create GPCs for each of the leaves (&leaves)           (custom)
  //       // evaluate the GPCs                                      (shared)
  //       // construct 1x batch with flags, 1x batch without flags  (custom)

  //       let mut rw_circuits = Vec::with_capacity(self.num_memories() * 2);
  //       let mut if_circuits = Vec::with_capacity(self.num_memories() * 2);
  //       let mut gp_evals = Vec::with_capacity(self.num_memories());

  //       // Stores the initial fingerprinted values for read and write memories. GPC stores the upper portion of the tree after the fingerprints at the leaves
  //       // experience flagging (toggling based on the flag value at that leaf).
  //       let mut rw_fingerprints: Vec<DensePolynomial<Fr>> =
  //         Vec::with_capacity(self.num_memories() * 2);
  //       for memory_index in 0..self.num_memories() {
  //         let (init_fingerprints, read_fingerprints, write_fingerprints, final_fingerprints) =
  //           self.compute_leaves(memory_index, r_hash);

  //         let (mut read_leaves, mut write_leaves) =
  //           (read_fingerprints.evals(), write_fingerprints.evals());
  //         rw_fingerprints.push(read_fingerprints);
  //         rw_fingerprints.push(write_fingerprints);
  //         for leaf_index in 0..self.ops_size() {
  //           let flag = match memory_index {
  //             0 => self.flags_0[leaf_index],
  //             1 => self.flags_1[leaf_index],
  //             _ => panic!("waa"),
  //           };
  //           // TODO(sragss): Would be faster if flags were non-FF repr
  //           if flag == Fr::zero() {
  //             read_leaves[leaf_index] = Fr::one();
  //             write_leaves[leaf_index] = Fr::one();
  //           }
  //         }

  //         let (init_gpc, final_gpc) = (
  //           GrandProductCircuit::new(&init_fingerprints),
  //           GrandProductCircuit::new(&final_fingerprints),
  //         );
  //         let (read_gpc, write_gpc) = (
  //           GrandProductCircuit::new(&DensePolynomial::new(read_leaves)),
  //           GrandProductCircuit::new(&DensePolynomial::new(write_leaves)),
  //         );

  //         gp_evals.push(GPEvals::new(
  //           init_gpc.evaluate(),
  //           read_gpc.evaluate(),
  //           write_gpc.evaluate(),
  //           final_gpc.evaluate(),
  //         ));

  //         rw_circuits.push(read_gpc);
  //         rw_circuits.push(write_gpc);
  //         if_circuits.push(init_gpc);
  //         if_circuits.push(final_gpc);
  //       }

  //       // self.memory_to_subtable map has to be expanded because we've doubled the number of "grand products memorys": [read_0, write_0, ... read_NUM_MEMORIES, write_NUM_MEMORIES]
  //       let expanded_flag_map = vec![0, 0, 1, 1];

  //       // Prover has access to subtable_flag_polys, which are uncommitted, but verifier can derive from instruction_flag commitments.
  //       let rw_batch = BatchedGrandProductCircuit::new_batch_flags(
  //         rw_circuits,
  //         vec![
  //           DensePolynomial::new(self.flags_0.clone()),
  //           DensePolynomial::new(self.flags_1.clone()),
  //         ],
  //         expanded_flag_map,
  //         rw_fingerprints,
  //       );

  //       let if_batch = BatchedGrandProductCircuit::new_batch(if_circuits);

  //       (rw_batch, if_batch, gp_evals)
  //     }
  //   }

  //   // Imagine a 2 memories. Size-8 range-check table (addresses and values just ascending), with 4 lookups into each
  //   let v_mems = vec![
  //     Fr::from(0),
  //     Fr::from(1),
  //     Fr::from(2),
  //     Fr::from(3),
  //     Fr::from(4),
  //     Fr::from(5),
  //     Fr::from(6),
  //     Fr::from(7),
  //   ];

  //   // 2 lookups into the last 2 elements of memory each
  //   let a_0_ops = vec![Fr::from(6), Fr::from(7), Fr::from(6), Fr::from(7)];
  //   let a_1_ops = vec![Fr::from(0), Fr::from(1), Fr::from(0), Fr::from(2)];
  //   let v_0_ops = a_0_ops.clone();
  //   let v_1_ops = a_1_ops.clone();

  //   let flags_0 = vec![Fr::one(), Fr::one(), Fr::one(), Fr::one()];
  //   let flags_1 = vec![
  //     Fr::one(),
  //     Fr::zero(), // Flagged off!
  //     Fr::one(),
  //     Fr::one(),
  //   ];

  //   let t_0_reads = vec![Fr::zero(), Fr::zero(), Fr::one(), Fr::one()];
  //   let t_1_reads = vec![Fr::zero(), Fr::zero(), Fr::one(), Fr::zero()];
  //   let t_0_finals = vec![
  //     Fr::zero(),
  //     Fr::zero(),
  //     Fr::zero(),
  //     Fr::zero(),
  //     Fr::zero(),
  //     Fr::zero(),
  //     Fr::from(2),
  //     Fr::from(2),
  //   ];
  //   let t_1_finals = vec![
  //     Fr::from(2),
  //     Fr::zero(), // Flagged off!
  //     Fr::one(),
  //     Fr::zero(),
  //     Fr::zero(),
  //     Fr::zero(),
  //     Fr::zero(),
  //     Fr::zero(),
  //   ];

  //   let polys = FlagMems {
  //     a_0_ops,
  //     a_1_ops,
  //     v_0_ops,
  //     v_1_ops,
  //     v_mems,
  //     t_0_reads,
  //     t_1_reads,
  //     t_0_finals,
  //     t_1_finals,
  //     flags_0,
  //     flags_1,
  //   };

  //   let mut transcript = Transcript::new(b"test_transcript");
  //   let r_fingerprints = (&Fr::from(12), &Fr::from(35));
  //   let (proof, _, _) =
  //     ProductLayerProof::prove::<EdwardsProjective, _>(&polys, r_fingerprints, &mut transcript);

  //   let mut transcript = Transcript::new(b"test_transcript");
  //   proof
  //     .verify::<EdwardsProjective>(&mut transcript)
  //     .expect("proof should work");
  // }
}
