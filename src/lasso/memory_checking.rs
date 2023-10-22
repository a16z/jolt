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

use super::fingerprint_strategy::FingerprintStrategy;
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

    // TODO(sragss): Wire up flags here
    panic!("");
    // let (proof_prod_layer, rand_mem, rand_ops) =
    //   ProductLayerProof::prove::<G, BGPCInterpretable<G::ScalarField>>(polynomials, r_fingerprint, transcript);

    // let proof_hash_layer = S::prove(
    //   (&rand_mem, &rand_ops),
    //   &polynomials,
    //   &generators,
    //   transcript,
    //   random_tape,
    // );

    // MemoryCheckingProof {
    //   proof_prod_layer,
    //   proof_hash_layer,
    //   num_ops: S::num_ops(&polynomials),
    //   num_memories: S::num_memories(&polynomials),
    //   memory_size: S::memory_size(&polynomials),
    // }
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
    polys: P,
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

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct HashLayerProof<G: CurveGroup> {
  eval_dim: Vec<G::ScalarField>,    // C-sized
  eval_read: Vec<G::ScalarField>,   // C-sized
  eval_final: Vec<G::ScalarField>,  // C-sized
  eval_derefs: Vec<G::ScalarField>, // NUM_MEMORIES-sized
  proof_ops: CombinedTableEvalProof<G>,
  proof_mem: CombinedTableEvalProof<G>,
  proof_derefs: CombinedTableEvalProof<G>,
}

impl<G: CurveGroup> FingerprintStrategy<G> for HashLayerProof<G> {
  type Polynomials = PolynomialRepresentation<G::ScalarField>;
  type Generators = SurgeCommitmentGenerators<G>;
  type Commitments = SurgeCommitment<G>;

  fn prove(
    rand: (&Vec<G::ScalarField>, &Vec<G::ScalarField>),
    polynomials: &Self::Polynomials,
    generators: &Self::Generators,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    let (rand_mem, rand_ops) = rand;

    // decommit derefs at rand_ops
    let eval_derefs: Vec<G::ScalarField> = (0..polynomials.num_memories)
      .map(|i| polynomials.E_polys[i].evaluate(rand_ops))
      .collect();
    let proof_derefs = CombinedTableEvalProof::prove(
      &polynomials.combined_E_poly,
      eval_derefs.as_ref(),
      rand_ops,
      &generators.E_commitment_gens,
      transcript,
      random_tape,
    );

    // form a single decommitment using comm_comb_ops
    let mut evals_ops: Vec<G::ScalarField> = Vec::new(); // moodlezoup: changed order of evals_ops

    let eval_dim: Vec<G::ScalarField> = (0..polynomials.C)
      .map(|i| polynomials.dim[i].evaluate(rand_ops))
      .collect();
    let eval_read: Vec<G::ScalarField> = (0..polynomials.num_memories)
      .map(|i| polynomials.read_cts[i].evaluate(rand_ops))
      .collect();
    let eval_final: Vec<G::ScalarField> = (0..polynomials.num_memories)
      .map(|i| polynomials.final_cts[i].evaluate(rand_mem))
      .collect();

    // TODO(sragss): flags?

    evals_ops.extend(eval_dim.clone());
    evals_ops.extend(eval_read.clone());
    evals_ops.resize(evals_ops.len().next_power_of_two(), G::ScalarField::zero());
    let proof_ops = CombinedTableEvalProof::prove(
      &polynomials.combined_dim_read_poly,
      &evals_ops,
      &rand_ops,
      &generators.dim_read_commitment_gens,
      transcript,
      random_tape,
    );

    let proof_mem = CombinedTableEvalProof::prove(
      &polynomials.combined_final_poly,
      &eval_final,
      &rand_mem,
      &generators.final_commitment_gens,
      transcript,
      random_tape,
    );

    HashLayerProof {
      eval_dim,
      eval_read,
      eval_final,
      proof_ops,
      proof_mem,
      eval_derefs,
      proof_derefs,
    }
  }

  fn verify<F1: Fn(usize) -> usize, F2: Fn(usize, &[G::ScalarField]) -> G::ScalarField>(
    &self,
    rand: (&Vec<G::ScalarField>, &Vec<G::ScalarField>),
    grand_product_claims: &[GPEvals<G::ScalarField>], // NUM_MEMORIES-sized
    memory_to_dimension_index: F1,
    evaluate_memory_mle: F2,
    commitments: &Self::Commitments,
    generators: &Self::Generators,
    r_hash: &G::ScalarField,
    r_multiset_check: &G::ScalarField,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
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
    for i in 0..grand_product_claims.len() {
      let j = memory_to_dimension_index(i);
      // Check ALPHA memories / lookup polys / grand products
      // Only need 'C' indices / dimensions / read_timestamps / final_timestamps
      Self::check_reed_solomon_fingerprints(
        &grand_product_claims[i],
        &self.eval_derefs[i],
        &self.eval_dim[j],
        &self.eval_read[j],
        &self.eval_final[j],
        &init_addr,
        &evaluate_memory_mle(i, rand_mem),
        r_hash,
        r_multiset_check,
      )?;
    }
    Ok(())
  }

  // TODO(sragss): Move these functions onto a trait all the PolynomialRepresentation types must implement
  fn num_ops(polys: &Self::Polynomials) -> usize {
    polys.num_ops
  }
  fn num_memories(polys: &Self::Polynomials) -> usize {
    polys.num_memories
  }
  fn memory_size(polys: &Self::Polynomials) -> usize {
    polys.memory_size
  }
}

impl<G: CurveGroup> HashLayerProof<G> {
  /// Checks that the Reed-Solomon fingerprints of init, read, write, and final multisets
  /// are as claimed by the final sumchecks of their respective grand product arguments.
  ///
  /// Params
  /// - `claims`: Fingerprint values of the init, read, write, and final multisets, as
  /// as claimed by their respective grand product arguments.
  /// - `eval_deref`: The evaluation E_i(r'''_i).
  /// - `eval_dim`: The evaluation dim_i(r'''_i).
  /// - `eval_read`: The evaluation read_i(r'''_i).
  /// - `eval_final`: The evaluation final_i(r''_i).
  /// - `init_addr`: The MLE of the memory addresses, evaluated at r''_i.
  /// - `init_memory`: The MLE of the initial memory values, evaluated at r''_i.
  /// - `r_i`: One chunk of the evaluation point at which the Lasso commitment is being opened.
  /// - `gamma`: Random value used to compute the Reed-Solomon fingerprint.
  /// - `tau`: Random value used to compute the Reed-Solomon fingerprint.
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
    let hash_func = |a: G::ScalarField, v: G::ScalarField, t: G::ScalarField| -> G::ScalarField {
      t * gamma.square() + v * *gamma + a - tau
    };
    // Note: this differs from the Lasso paper a little:
    // (t * gamma^2 + v * gamma + a) instead of (a * gamma^2 + v * gamma + t)

    let claim_init = claims.hash_init;
    let claim_read = claims.hash_read;
    let claim_write = claims.hash_write;
    let claim_final = claims.hash_final;

    // init
    let hash_init = hash_func(*init_addr, *init_memory, G::ScalarField::zero());
    assert_eq!(hash_init, claim_init); // verify the last claim of the `init` grand product sumcheck

    // read
    let hash_read = hash_func(*eval_dim, *eval_deref, *eval_read);
    assert_eq!(hash_read, claim_read); // verify the last claim of the `read` grand product sumcheck

    // write: shares addr, val with read
    let eval_write = *eval_read + G::ScalarField::one();
    let hash_write = hash_func(*eval_dim, *eval_deref, eval_write);
    assert_eq!(hash_write, claim_write); // verify the last claim of the `write` grand product sumcheck

    // final: shares addr and val with init
    let eval_final_addr = init_addr;
    let eval_final_val = init_memory;
    let hash_final = hash_func(*eval_final_addr, *eval_final_val, *eval_final);
    assert_eq!(hash_final, claim_final); // verify the last claim of the `final` grand product sumcheck

    Ok(())
  }

  fn protocol_name() -> &'static [u8] {
    b"Lasso HashLayerProof"
  }
}

#[cfg(test)]
mod tests {
  use crate::{subprotocols::grand_product::{BGPCInterpretable, GrandProductCircuit, BatchedGrandProductCircuit}, poly::dense_mlpoly::DensePolynomial};
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
      ProductLayerProof::prove::<EdwardsProjective, _>(polys, r_fingerprints, &mut transcript);

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
      ProductLayerProof::prove::<EdwardsProjective, _>(polys, r_fingerprints, &mut transcript);

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
      ProductLayerProof::prove::<EdwardsProjective, _>(polys, r_fingerprints, &mut transcript);

    let mut transcript = Transcript::new(b"test_transcript");
    proof
      .verify::<EdwardsProjective>(&mut transcript)
      .expect("proof should work");
  }
}
