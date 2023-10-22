use ark_ec::CurveGroup;
use ark_ff::{Field, PrimeField};
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use ark_std::{One, Zero};
use merlin::Transcript;

use crate::{utils::{random::RandomTape, errors::ProofVerifyError, transcript::ProofTranscript}, poly::{dense_mlpoly::DensePolynomial, identity_poly::IdentityPolynomial}, subprotocols::{combined_table_proof::CombinedTableEvalProof, grand_product::BGPCInterpretable}, jolt::vm::{PolynomialRepresentation, SurgeCommitmentGenerators, SurgeCommitment}};

use super::gp_evals::GPEvals;

pub trait MemBatchInfo {
  fn ops_size(&self) -> usize;
  fn mem_size(&self) -> usize;
  fn num_memories(&self) -> usize;
}

impl<F: PrimeField> MemBatchInfo for PolynomialRepresentation<F> {
    fn ops_size(&self) -> usize {
      self.num_ops
    }

    fn mem_size(&self) -> usize {
      self.memory_size
    }

    fn num_memories(&self) -> usize {
      self.num_memories
    }
}

/// Trait which defines a strategy for creating opening proofs for multi-set fingerprints and verifies.
pub trait FingerprintStrategy<G: CurveGroup>:
  std::marker::Sync + CanonicalSerialize + CanonicalDeserialize
{
  type Polynomials: BGPCInterpretable<G::ScalarField> + MemBatchInfo;
  type Generators;
  type Commitments;

  fn prove(
    rand: (&Vec<G::ScalarField>, &Vec<G::ScalarField>),
    polynomials: &Self::Polynomials,
    generators: &Self::Generators,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self;

  // TODO(sragss): simplify signature
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
  ) -> Result<(), ProofVerifyError>;
}

/// Read Only Fingerprint Proof.
#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ROFingerprintProof<G: CurveGroup> {
  eval_dim: Vec<G::ScalarField>,    // C-sized
  eval_read: Vec<G::ScalarField>,   // NUM_MEMORIES-sized
  eval_final: Vec<G::ScalarField>,  // NUM_MEMORIES-sized
  eval_derefs: Vec<G::ScalarField>, // NUM_MEMORIES-sized

  proof_ops: CombinedTableEvalProof<G>,
  proof_mem: CombinedTableEvalProof<G>,
  proof_derefs: CombinedTableEvalProof<G>,
}

impl<G: CurveGroup> FingerprintStrategy<G> for ROFingerprintProof<G> {
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

    ROFingerprintProof {
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
}

impl<G: CurveGroup> ROFingerprintProof<G> {
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