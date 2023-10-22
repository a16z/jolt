#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::jolt::vm::{PolynomialRepresentation, SurgeCommitment, SurgeCommitmentGenerators};
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::identity_poly::IdentityPolynomial;
use crate::subprotocols::combined_table_proof::CombinedTableEvalProof;
use crate::subprotocols::grand_product::{
  BGPCInterpretable, BatchedGrandProductArgument, BatchedGrandProductCircuit, GrandProductCircuit,
  GrandProducts,
};
use crate::utils::errors::ProofVerifyError;
use crate::utils::random::RandomTape;
use crate::utils::transcript::ProofTranscript;

use ark_ec::CurveGroup;
use ark_ff::{Field, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{One, Zero};
use merlin::Transcript;

use super::fingerprint_strategy::{FingerprintStrategy, MemBatchInfo};
use super::gp_evals::GPEvals;

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct MemoryCheckingProof<G: CurveGroup, S: FingerprintStrategy<G>> {
  proof_prod_layer: ProductLayerProof<G::ScalarField>,
  proof_hash_layer: S,
  num_ops: usize,
  num_memories: usize,
  memory_size: usize,
}

impl<G: CurveGroup, S: FingerprintStrategy<G>> MemoryCheckingProof<G, S> {
  pub fn prove(
    polynomials: &S::Polynomials,
    r_fingerprint: (&G::ScalarField, &G::ScalarField),
    generators: &S::Generators,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    let (proof_prod_layer, rand_mem, rand_ops) =
      ProductLayerProof::prove::<G, S::Polynomials>(polynomials, r_fingerprint, transcript);

    let proof_hash_layer = S::prove(
      (&rand_mem, &rand_ops),
      &polynomials,
      &generators,
      transcript,
      random_tape,
    );

    MemoryCheckingProof {
      proof_prod_layer,
      proof_hash_layer,
      num_ops: polynomials.ops_size(),
      num_memories: polynomials.num_memories(),
      memory_size: polynomials.mem_size(),
    }
  }

  pub fn verify<F1: Fn(usize) -> usize, F2: Fn(usize, &[G::ScalarField]) -> G::ScalarField>(
    &self,
    commitments: &S::Commitments,
    generators: &S::Generators,
    // TODO(sragss): Consider hardcoding these params
    memory_to_dimension_index: F1,
    evaluate_memory_mle: F2,
    r_mem_check: &(G::ScalarField, G::ScalarField),
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    let (r_hash, r_multiset_check) = r_mem_check;

    let (claims_mem, rand_mem, claims_ops, rand_ops) =
      self.proof_prod_layer.verify::<G>(transcript)?;

    let claims: Vec<GPEvals<G::ScalarField>> = (0..self.num_memories)
      .map(|i| {
        GPEvals::new(
          claims_mem[2 * i],     // init
          claims_ops[2 * i],     // read
          claims_ops[2 * i + 1], // write
          claims_mem[2 * i + 1], // final
        )
      })
      .collect();

    // verify the proof of hash layer
    self.proof_hash_layer.verify(
      (&rand_mem, &rand_ops),
      &claims,
      memory_to_dimension_index,
      evaluate_memory_mle,
      commitments,
      generators,
      r_hash,
      r_multiset_check,
      transcript,
    )?;

    Ok(())
  }

  fn protocol_name() -> &'static [u8] {
    b"Lasso MemoryCheckingProof"
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
struct ProductLayerProof<F: PrimeField> {
  grand_product_evals: Vec<GPEvals<F>>,
  proof_mem: BatchedGrandProductArgument<F>,
  proof_ops: BatchedGrandProductArgument<F>,
  num_memories: usize,
}

impl<F: PrimeField> ProductLayerProof<F> {
  /// Performs grand product argument proofs required for memory-checking.
  /// Batches everything into two instances of BatchedGrandProductArgument.
  ///
  /// Params
  /// - `grand_products`: The grand product circuits whose evaluations are proven.
  /// - `transcript`: The proof transcript, used for Fiat-Shamir.
  #[tracing::instrument(skip_all, name = "ProductLayer.prove")]
  pub fn prove<G, P>(
    polys: &P,
    r_fingerprint: (&G::ScalarField, &G::ScalarField),
    transcript: &mut Transcript,
  ) -> (Self, Vec<F>, Vec<F>)
  where
    G: CurveGroup<ScalarField = F>,
    P: BGPCInterpretable<G::ScalarField>,
  {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    let num_memories = polys.num_memories();
    let (batched_rw, batched_if, grand_product_evals) =
      BatchedGrandProductCircuit::construct(polys, r_fingerprint);

    grand_product_evals.iter().for_each(|gp_eval| {
      assert_eq!(
        gp_eval.hash_init * gp_eval.hash_write,
        gp_eval.hash_final * gp_eval.hash_read
      );
      gp_eval.append_to_transcript::<G>(transcript);
    });

    let (proof_ops, rand_ops_sized_gps) =
      BatchedGrandProductArgument::prove::<G>(batched_rw, transcript);
    let (proof_mem, rand_mem_sized_gps) =
      BatchedGrandProductArgument::prove::<G>(batched_if, transcript);

    let product_layer_proof = ProductLayerProof {
      grand_product_evals,
      proof_mem,
      proof_ops,
      num_memories,
    };

    (product_layer_proof, rand_mem_sized_gps, rand_ops_sized_gps)
  }

  pub fn verify<G>(
    &self,
    transcript: &mut Transcript,
  ) -> Result<(Vec<F>, Vec<F>, Vec<F>, Vec<F>), ProofVerifyError>
  where
    G: CurveGroup<ScalarField = F>,
  {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    for eval in &self.grand_product_evals {
      // Multiset equality check
      debug_assert_eq!(
        eval.hash_init * eval.hash_write,
        eval.hash_read * eval.hash_final
      );

      eval.append_to_transcript::<G>(transcript);
    }

    let read_write_claims = GPEvals::flatten_read_write(&self.grand_product_evals);
    let (claims_ops, rand_ops) = self
      .proof_ops
      .verify::<G, Transcript>(&read_write_claims, transcript);

    let init_final_claims = GPEvals::flatten_init_final(&self.grand_product_evals);
    let (claims_mem, rand_mem) = self
      .proof_mem
      .verify::<G, Transcript>(&init_final_claims, transcript);

    Ok((claims_mem, rand_mem, claims_ops, rand_ops))
  }

  fn protocol_name() -> &'static [u8] {
    b"Lasso ProductLayerProof"
  }
}

#[cfg(test)]
mod tests {
  use crate::{subprotocols::grand_product::{BGPCInterpretable, GrandProductCircuit, BatchedGrandProductCircuit}, poly::dense_mlpoly::DensePolynomial, lasso::fingerprint_strategy::MemBatchInfo};
  use ark_curve25519::{EdwardsProjective, Fr};
  use ark_std::{One, Zero};
  use merlin::Transcript;

  use super::ProductLayerProof;

  #[test]
  fn product_layer_proof_trivial() {
    // Define the most trivial GrandProduct memory checking layout
    struct NormalMems {
      a_ops: Vec<Fr>,

      v_ops: Vec<Fr>,
      v_mems: Vec<Fr>,

      t_reads: Vec<Fr>,
      t_finals: Vec<Fr>,
    }

    impl MemBatchInfo for NormalMems {
      fn mem_size(&self) -> usize {
        assert_eq!(self.v_mems.len(), self.t_finals.len());
        self.v_mems.len()
      }

      fn ops_size(&self) -> usize {
        assert_eq!(self.a_ops.len(), self.v_ops.len());
        assert_eq!(self.a_ops.len(), self.t_reads.len());
        self.a_ops.len()
      }

      fn num_memories(&self) -> usize {
        1
      }
    }

    impl BGPCInterpretable<Fr> for NormalMems {
      fn a_ops(&self, memory_index: usize, leaf_index: usize) -> Fr {
        assert_eq!(memory_index, 0);
        self.a_ops[leaf_index]
      }

      fn v_mem(&self, memory_index: usize, leaf_index: usize) -> Fr {
        assert_eq!(memory_index, 0);
        self.v_mems[leaf_index]
      }

      fn v_ops(&self, memory_index: usize, leaf_index: usize) -> Fr {
        assert_eq!(memory_index, 0);
        self.v_ops[leaf_index]
      }

      fn t_final(&self, memory_index: usize, leaf_index: usize) -> Fr {
        assert_eq!(memory_index, 0);
        self.t_finals[leaf_index]
      }

      fn t_read(&self, memory_index: usize, leaf_index: usize) -> Fr {
        assert_eq!(memory_index, 0);
        self.t_reads[leaf_index]
      }
    }

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

    let polys = NormalMems {
      a_ops,
      v_ops,
      v_mems,
      t_reads,
      t_finals,
    };

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
  fn product_layer_proof_batched() {
    // Define a GrandProduct circuit that can be batched across 2 memories
    struct NormalMems {
      a_0_ops: Vec<Fr>,
      a_1_ops: Vec<Fr>,

      v_0_ops: Vec<Fr>,
      v_1_ops: Vec<Fr>,
      v_mems: Vec<Fr>,

      t_0_reads: Vec<Fr>,
      t_1_reads: Vec<Fr>,

      t_0_finals: Vec<Fr>,
      t_1_finals: Vec<Fr>,
    }

    impl MemBatchInfo for NormalMems {
      fn mem_size(&self) -> usize {
        assert_eq!(self.v_mems.len(), self.t_0_finals.len());
        assert_eq!(self.v_mems.len(), self.t_1_finals.len());
        self.v_mems.len()
      }

      fn ops_size(&self) -> usize {
        let ops_len = self.a_0_ops.len();
        assert_eq!(ops_len, self.a_1_ops.len());
        assert_eq!(ops_len, self.v_0_ops.len());
        assert_eq!(ops_len, self.v_1_ops.len());
        assert_eq!(ops_len, self.t_0_reads.len());
        assert_eq!(ops_len, self.t_1_reads.len());

        ops_len
      }

      fn num_memories(&self) -> usize {
        2
      }
    }

    impl BGPCInterpretable<Fr> for NormalMems {
      fn a_ops(&self, memory_index: usize, leaf_index: usize) -> Fr {
        assert!(memory_index < 2);
        match memory_index {
          0 => self.a_0_ops[leaf_index],
          1 => self.a_1_ops[leaf_index],
          _ => panic!("waaa"),
        }
      }

      fn v_mem(&self, memory_index: usize, leaf_index: usize) -> Fr {
        assert!(memory_index < 2);
        self.v_mems[leaf_index]
      }

      fn v_ops(&self, memory_index: usize, leaf_index: usize) -> Fr {
        assert!(memory_index < 2);
        match memory_index {
          0 => self.v_0_ops[leaf_index],
          1 => self.v_1_ops[leaf_index],
          _ => panic!("waaa"),
        }
      }

      fn t_final(&self, memory_index: usize, leaf_index: usize) -> Fr {
        assert!(memory_index < 2);
        match memory_index {
          0 => self.t_0_finals[leaf_index],
          1 => self.t_1_finals[leaf_index],
          _ => panic!("waaa"),
        }
      }

      fn t_read(&self, memory_index: usize, leaf_index: usize) -> Fr {
        assert!(memory_index < 2);
        match memory_index {
          0 => self.t_0_reads[leaf_index],
          1 => self.t_1_reads[leaf_index],
          _ => panic!("waaa"),
        }
      }

    }

    // Imagine a 2 memories. Size-8 range-check table (addresses and values just ascending), with 4 lookups into each
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
    let a_0_ops = vec![Fr::from(6), Fr::from(7), Fr::from(6), Fr::from(7)];
    let a_1_ops = vec![Fr::from(0), Fr::from(1), Fr::from(0), Fr::from(2)];
    let v_0_ops = a_0_ops.clone();
    let v_1_ops = a_1_ops.clone();

    let t_0_reads = vec![Fr::zero(), Fr::zero(), Fr::one(), Fr::one()];
    let t_1_reads = vec![Fr::zero(), Fr::zero(), Fr::one(), Fr::zero()];
    let t_0_finals = vec![
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::from(2),
      Fr::from(2),
    ];
    let t_1_finals = vec![
      Fr::from(2),
      Fr::one(),
      Fr::one(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
    ];

    let polys = NormalMems {
      a_0_ops,
      a_1_ops,
      v_0_ops,
      v_1_ops,
      v_mems,
      t_0_reads,
      t_1_reads,
      t_0_finals,
      t_1_finals,
    };

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
  fn product_layer_proof_flags_no_reuse() {
    // Define a GrandProduct circuit that can be batched across 2 memories
    struct FlagMems {
      a_0_ops: Vec<Fr>,
      a_1_ops: Vec<Fr>,

      v_0_ops: Vec<Fr>,
      v_1_ops: Vec<Fr>,
      v_mems: Vec<Fr>,

      t_0_reads: Vec<Fr>,
      t_1_reads: Vec<Fr>,

      t_0_finals: Vec<Fr>,
      t_1_finals: Vec<Fr>,

      flags_0: Vec<Fr>,
      flags_1: Vec<Fr>,
    }

    impl MemBatchInfo for FlagMems {
      fn mem_size(&self) -> usize {
        assert_eq!(self.v_mems.len(), self.t_0_finals.len());
        assert_eq!(self.v_mems.len(), self.t_1_finals.len());
        self.v_mems.len()
      }

      fn ops_size(&self) -> usize {
        let ops_len = self.a_0_ops.len();
        assert_eq!(ops_len, self.a_1_ops.len());
        assert_eq!(ops_len, self.v_0_ops.len());
        assert_eq!(ops_len, self.v_1_ops.len());
        assert_eq!(ops_len, self.t_0_reads.len());
        assert_eq!(ops_len, self.t_1_reads.len());
        assert_eq!(ops_len, self.flags_0.len());
        assert_eq!(ops_len, self.flags_1.len());

        ops_len
      }

      fn num_memories(&self) -> usize {
        2
      }
    }

    impl BGPCInterpretable<Fr> for FlagMems {
      fn a_ops(&self, memory_index: usize, leaf_index: usize) -> Fr {
        assert!(memory_index < 2);
        match memory_index {
          0 => self.a_0_ops[leaf_index],
          1 => self.a_1_ops[leaf_index],
          _ => panic!("waaa"),
        }
      }

      fn v_mem(&self, memory_index: usize, leaf_index: usize) -> Fr {
        assert!(memory_index < 2);
        self.v_mems[leaf_index]
      }

      fn v_ops(&self, memory_index: usize, leaf_index: usize) -> Fr {
        assert!(memory_index < 2);
        match memory_index {
          0 => self.v_0_ops[leaf_index],
          1 => self.v_1_ops[leaf_index],
          _ => panic!("waaa"),
        }
      }

      fn t_final(&self, memory_index: usize, leaf_index: usize) -> Fr {
        assert!(memory_index < 2);
        match memory_index {
          0 => self.t_0_finals[leaf_index],
          1 => self.t_1_finals[leaf_index],
          _ => panic!("waaa"),
        }
      }

      fn t_read(&self, memory_index: usize, leaf_index: usize) -> Fr {
        assert!(memory_index < 2);
        match memory_index {
          0 => self.t_0_reads[leaf_index],
          1 => self.t_1_reads[leaf_index],
          _ => panic!("waaa"),
        }
      }

      // FLAGS OVERRIDES

      fn fingerprint_read(&self, memory_index: usize, leaf_index: usize, gamma: &Fr, tau: &Fr) -> Fr {
        assert!(memory_index < 2);
        assert!(leaf_index < self.ops_size());

        let h = Self::fingerprint(
          self.a_ops(memory_index, leaf_index),
          self.v_ops(memory_index, leaf_index),
          self.t_read(memory_index, leaf_index),
          gamma,
          tau,
        );
        let flags = match memory_index {
          0 => self.flags_0[leaf_index],
          1 => self.flags_1[leaf_index],
          _ => panic!("waa")
        };
        flags * h + (Fr::one() - flags)
      }

      fn fingerprint_write(&self, memory_index: usize, leaf_index: usize, gamma: &Fr, tau: &Fr) -> Fr {
        assert!(memory_index < 2);
        assert!(leaf_index < self.ops_size());

        let h = Self::fingerprint(
          self.a_ops(memory_index, leaf_index),
          self.v_ops(memory_index, leaf_index),
          self.t_write(memory_index, leaf_index),
          gamma,
          tau,
        );
        let flags = match memory_index {
          0 => self.flags_0[leaf_index],
          1 => self.flags_1[leaf_index],
          _ => panic!("waa")
        };
        flags * h + (Fr::one() - flags)
      }

      fn construct_batched_read_write(
          &self, // TODO(sragss): Consume self?
          reads: Vec<GrandProductCircuit<Fr>>, 
          writes: Vec<GrandProductCircuit<Fr>>) -> BatchedGrandProductCircuit<Fr> {
        debug_assert_eq!(reads.len(), writes.len());
        let interleaves = reads.into_iter().zip(writes).flat_map(|(read, write)| [read, write]).collect();
        let flags = 
          vec![DensePolynomial::new(self.flags_0.clone()), DensePolynomial::new(self.flags_1.clone())];
        let flag_map = vec![0, 1];
        let expanded_flag_map = flag_map.iter().flat_map(|&i| vec![i; 2]).collect();
    
        BatchedGrandProductCircuit::new_batch_flags(interleaves, flags, expanded_flag_map)
      }
    }

    // Imagine a 2 memories. Size-8 range-check table (addresses and values just ascending), with 4 lookups into each
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
    let a_0_ops = vec![Fr::from(6), Fr::from(7), Fr::from(6), Fr::from(7)];
    let a_1_ops = vec![Fr::from(0), Fr::from(1), Fr::from(0), Fr::from(2)];
    let v_0_ops = a_0_ops.clone();
    let v_1_ops = a_1_ops.clone();

    let flags_0 = vec![
      Fr::one(), 
      Fr::one(), 
      Fr::one(), 
      Fr::one()
    ];
    let flags_1 = vec![
      Fr::one(), 
      Fr::zero(), // Flagged off!
      Fr::one(), 
      Fr::one()
      ];

    let t_0_reads = vec![Fr::zero(), Fr::zero(), Fr::one(), Fr::one()];
    let t_1_reads = vec![Fr::zero(), Fr::zero(), Fr::one(), Fr::zero()];
    let t_0_finals = vec![
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::from(2),
      Fr::from(2),
    ];
    let t_1_finals = vec![
      Fr::from(2),
      Fr::zero(), // Flagged off!
      Fr::one(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
      Fr::zero(),
    ];

    let polys = FlagMems {
      a_0_ops,
      a_1_ops,
      v_0_ops,
      v_1_ops,
      v_mems,
      t_0_reads,
      t_1_reads,
      t_0_finals,
      t_1_finals,
      flags_0,
      flags_1
    };

    let mut transcript = Transcript::new(b"test_transcript");
    let r_fingerprints = (&Fr::from(12), &Fr::from(35));
    let (proof, _, _) =
      ProductLayerProof::prove::<EdwardsProjective, _>(&polys, r_fingerprints, &mut transcript);

    let mut transcript = Transcript::new(b"test_transcript");
    proof
      .verify::<EdwardsProjective>(&mut transcript)
      .expect("proof should work");
  }
}
