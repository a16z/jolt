#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::jolt::vm::{PolynomialRepresentation, SurgeCommitment, SurgeCommitmentGenerators};
use crate::poly::identity_poly::IdentityPolynomial;
use crate::subprotocols::combined_table_proof::CombinedTableEvalProof;
use crate::subprotocols::grand_product::{
  BatchedGrandProductArgument, GrandProductCircuit, GrandProducts,
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

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct MemoryCheckingProof<G: CurveGroup, S: FingerprintStrategy<G>> {
  proof_prod_layer: ProductLayerProof<G::ScalarField>,
  proof_hash_layer: S,
  num_ops: usize,
  num_memories: usize,
  memory_size: usize,
}

/// Evaluations of a Grand Product Argument for the four required sets.
#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct GPEvals<F: PrimeField> {
  hash_init: F,
  hash_read: F,
  hash_write: F,
  hash_final: F,
}

impl<F: PrimeField> GPEvals<F> {
  fn new(hash_init: F, hash_read: F, hash_write: F, hash_final: F) -> Self {
    Self {
      hash_init,
      hash_read,
      hash_write,
      hash_final,
    }
  }

  /// Flattens a vector of GPEvals to a vector of field elements alternating between init evals and final evals.
  fn flatten_init_final(evals: &[Self]) -> Vec<F> {
    evals
      .iter()
      .flat_map(|eval| [eval.hash_init, eval.hash_final])
      .collect()
  }

  /// Flattens a vector of GPEvals to a vector of field elements alternating between read evals and write evals.
  fn flatten_read_write(evals: &[Self]) -> Vec<F> {
    evals
      .iter()
      .flat_map(|eval| [eval.hash_read, eval.hash_write])
      .collect()
  }
}

impl<G: CurveGroup, S: FingerprintStrategy<G>> MemoryCheckingProof<G, S> {
  pub fn prove(
    polynomials: &S::Polynomials,
    grand_products: &mut Vec<GrandProducts<G::ScalarField>>,
    generators: &S::Generators,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    let (proof_prod_layer, rand_mem, rand_ops) =
      ProductLayerProof::prove::<G>(grand_products, transcript);

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
      num_ops: S::num_ops(&polynomials),
      num_memories: S::num_memories(&polynomials),
      memory_size: S::memory_size(&polynomials),
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

    let num_ops = self.num_ops.next_power_of_two();

    let (claims_mem, rand_mem, claims_ops, rand_ops) =
      self
        .proof_prod_layer
        .verify::<G>(num_ops, self.memory_size, transcript)?;

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
  pub fn prove<G>(
    grand_products: &mut [GrandProducts<F>],
    transcript: &mut Transcript,
  ) -> (Self, Vec<F>, Vec<F>)
  where
    G: CurveGroup<ScalarField = F>,
  {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    let grand_product_evals: Vec<GPEvals<F>> = (0..grand_products.len())
      .map(|i| {
        let hash_init = grand_products[i].init.evaluate();
        let hash_read = grand_products[i].read.evaluate();
        let hash_write = grand_products[i].write.evaluate();
        let hash_final = grand_products[i].r#final.evaluate();

        assert_eq!(hash_init * hash_write, hash_read * hash_final);

        <Transcript as ProofTranscript<G>>::append_scalar(
          transcript,
          b"claim_hash_init",
          &hash_init,
        );
        <Transcript as ProofTranscript<G>>::append_scalar(
          transcript,
          b"claim_hash_read",
          &hash_read,
        );
        <Transcript as ProofTranscript<G>>::append_scalar(
          transcript,
          b"claim_hash_write",
          &hash_write,
        );
        <Transcript as ProofTranscript<G>>::append_scalar(
          transcript,
          b"claim_hash_final",
          &hash_final,
        );

        GPEvals::new(hash_init, hash_read, hash_write, hash_final)
      })
      .collect();

    let mut read_write_grand_products: Vec<&mut GrandProductCircuit<F>> = grand_products
      .iter_mut()
      .flat_map(|grand_product| [&mut grand_product.read, &mut grand_product.write])
      .collect();

    let (proof_ops, rand_ops_sized_gps) =
      BatchedGrandProductArgument::<F>::prove::<G>(&mut read_write_grand_products, transcript);

    let mut init_final_grand_products: Vec<&mut GrandProductCircuit<F>> = grand_products
      .iter_mut()
      .flat_map(|grand_product| [&mut grand_product.init, &mut grand_product.r#final])
      .collect();

    // produce a batched proof of memory-related product circuits
    let (proof_mem, rand_mem_sized_gps) =
      BatchedGrandProductArgument::<F>::prove::<G>(&mut init_final_grand_products, transcript);

    let product_layer_proof = ProductLayerProof {
      grand_product_evals,
      proof_mem,
      proof_ops,
      num_memories: grand_products.len(),
    };

    (product_layer_proof, rand_mem_sized_gps, rand_ops_sized_gps)
  }

  pub fn verify<G>(
    &self,
    num_ops: usize,
    num_cells: usize,
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

      <Transcript as ProofTranscript<G>>::append_scalar(
        transcript,
        b"claim_hash_init",
        &eval.hash_init,
      );
      <Transcript as ProofTranscript<G>>::append_scalar(
        transcript,
        b"claim_hash_read",
        &eval.hash_read,
      );
      <Transcript as ProofTranscript<G>>::append_scalar(
        transcript,
        b"claim_hash_write",
        &eval.hash_write,
      );
      <Transcript as ProofTranscript<G>>::append_scalar(
        transcript,
        b"claim_hash_final",
        &eval.hash_final,
      );
    }

    let read_write_claims = GPEvals::flatten_read_write(&self.grand_product_evals);
    let (claims_ops, rand_ops) =
      self
        .proof_ops
        .verify::<G, Transcript>(&read_write_claims, num_ops, transcript);

    let init_final_claims = GPEvals::flatten_init_final(&self.grand_product_evals);
    let (claims_mem, rand_mem) =
      self
        .proof_mem
        .verify::<G, Transcript>(&init_final_claims, num_cells, transcript);

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
    let eval_read: Vec<G::ScalarField> = (0..polynomials.C)
      .map(|i| polynomials.read_cts[i].evaluate(rand_ops))
      .collect();
    let eval_final: Vec<G::ScalarField> = (0..polynomials.C)
      .map(|i| polynomials.final_cts[i].evaluate(rand_mem))
      .collect();

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

// #[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
// struct HashLayerProofFlags<G: CurveGroup> {
//   eval_dim: Vec<G::ScalarField>,    // C-sized
//   eval_read: Vec<G::ScalarField>,   // C-sized
//   eval_final: Vec<G::ScalarField>,  // C-sized
//   eval_derefs: Vec<G::ScalarField>, // NUM_MEMORIES-sized

//   // NEW
//   eval_flags: Vec<G::ScalarField>, // NUM_MEMORIES-sized

//   proof_ops: CombinedTableEvalProof<G>,
//   proof_mem: CombinedTableEvalProof<G>,
//   proof_derefs: CombinedTableEvalProof<G>,

//   // NEW
//   proof_flags: CombinedTableEvalProof<G>,
// }

// impl<G: CurveGroup> FingerprintStrategy<G> for HashLayerProofFlags<G> {
//   type Polynomials = PolynomialRepresentation<G::ScalarField>;
//   type Generators = SurgeCommitmentGenerators<G>;
//   type Commitments = SurgeCommitment<G>;

//   fn prove(
//     rand: (&Vec<G::ScalarField>, &Vec<G::ScalarField>),
//     polynomials: &Self::Polynomials,
//     generators: &Self::Generators,
//     transcript: &mut Transcript,
//     random_tape: &mut RandomTape<G>,
//   ) -> Self {
//     todo!("unimpl") // TODO: Same as before + flags
//   }

//   fn verify<F1: Fn(usize) -> usize, F2: Fn(usize, &[G::ScalarField]) -> G::ScalarField>(
//     &self,
//     rand: (&Vec<G::ScalarField>, &Vec<G::ScalarField>),
//     grand_product_claims: &[GPEvals<G::ScalarField>], // NUM_MEMORIES-sized
//     memory_to_dimension_index: F1,
//     evaluate_memory_mle: F2,
//     commitments: &Self::Commitments,
//     generators: &Self::Generators,
//     r_hash: &G::ScalarField,
//     r_multiset_check: &G::ScalarField,
//     transcript: &mut Transcript,
//   ) -> Result<(), ProofVerifyError> {
//     todo!("unimpl")
//   }

//   // TODO(sragss): Move these functions onto a trait all the PolynomialRepresentation types must implement
//   fn num_ops(polys: &Self::Polynomials) -> usize {
//     polys.num_ops
//   }
//   fn num_memories(polys: &Self::Polynomials) -> usize {
//     polys.num_memories
//   }
//   fn memory_size(polys: &Self::Polynomials) -> usize {
//     polys.memory_size
//   }
// }
