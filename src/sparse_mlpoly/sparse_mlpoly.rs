#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

use crate::dense_mlpoly::{PolyCommitment, PolyCommitmentGens};
use crate::errors::ProofVerifyError;
use crate::math::Math;
use crate::random::RandomTape;
use crate::sparse_mlpoly::densified::DensifiedRepresentation;
use crate::sparse_mlpoly::memory_checking::MemoryCheckingProof;
use crate::sparse_mlpoly::subtables::{
  CombinedTableCommitment, CombinedTableEvalProof, SubtableStrategy, Subtables,
};
use crate::sumcheck::SumcheckInstanceProof;
use crate::transcript::{AppendToTranscript, ProofTranscript};
use ark_ec::CurveGroup;

use ark_serialize::*;

use merlin::Transcript;

pub struct SparsePolyCommitmentGens<G> {
  pub gens_combined_l_variate: PolyCommitmentGens<G>,
  pub gens_combined_log_m_variate: PolyCommitmentGens<G>,
  pub gens_derefs: PolyCommitmentGens<G>,
}

impl<G: CurveGroup> SparsePolyCommitmentGens<G> {
  pub fn new(
    label: &'static [u8],
    c: usize,
    s: usize,
    num_memories: usize,
    log_m: usize,
  ) -> SparsePolyCommitmentGens<G> {
    // dim_1, ... dim_c, read_1, ..., read_c
    // log_2(cs + cs)
    let num_vars_combined_l_variate = (2 * c * s).next_power_of_two().log_2();
    // final
    // log_2(cm) = log_2(c) + log_2(m)
    let num_vars_combined_log_m_variate = c.next_power_of_two().log_2() + log_m;
    // E_1, ..., E_alpha
    // log_2(alpha * s)
    let num_vars_derefs = (num_memories * s).next_power_of_two().log_2();

    let gens_combined_l_variate = PolyCommitmentGens::new(num_vars_combined_l_variate, label);
    let gens_combined_log_m_variate =
      PolyCommitmentGens::new(num_vars_combined_log_m_variate, label);
    let gens_derefs = PolyCommitmentGens::new(num_vars_derefs, label);
    SparsePolyCommitmentGens {
      gens_combined_l_variate,
      gens_combined_log_m_variate,
      gens_derefs,
    }
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct SparsePolynomialCommitment<G: CurveGroup> {
  pub l_variate_polys_commitment: PolyCommitment<G>,
  pub log_m_variate_polys_commitment: PolyCommitment<G>,
  pub s: usize, // sparsity
  pub log_m: usize,
  pub m: usize,
}

impl<G: CurveGroup> AppendToTranscript<G> for SparsePolynomialCommitment<G> {
  fn append_to_transcript<T: ProofTranscript<G>>(&self, _label: &'static [u8], transcript: &mut T) {
    self
      .l_variate_polys_commitment
      .append_to_transcript(b"l_variate_polys_commitment", transcript);
    self
      .log_m_variate_polys_commitment
      .append_to_transcript(b"log_m_variate_polys_commitment", transcript);
    transcript.append_u64(b"s", self.s as u64);
    transcript.append_u64(b"log_m", self.log_m as u64);
    transcript.append_u64(b"m", self.m as u64);
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct SparseLookupMatrix<const C: usize> {
  pub nz: Vec<[usize; C]>, // non-zero indices nz_1(i), ..., nz_c(i)
  pub s: usize,            // sparsity
  pub log_m: usize,
  pub m: usize,
}

impl<const C: usize> SparseLookupMatrix<C> {
  pub fn new(nonzero_indices: Vec<[usize; C]>, log_m: usize) -> Self {
    let s = nonzero_indices.len().next_power_of_two();
    // TODO(moodlezoup): nonzero_indices.resize?

    SparseLookupMatrix {
      nz: nonzero_indices,
      s,
      log_m,
      m: log_m.pow2(),
    }
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
struct PrimarySumcheck<G: CurveGroup, const ALPHA: usize> {
  proof: SumcheckInstanceProof<G::ScalarField>,
  claimed_evaluation: G::ScalarField,
  eval_derefs: [G::ScalarField; ALPHA],
  proof_derefs: CombinedTableEvalProof<G, ALPHA>,
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct SparsePolynomialEvaluationProof<
  G: CurveGroup,
  const C: usize,
  const M: usize,
  S: SubtableStrategy<G::ScalarField, C, M>,
> where
  [(); S::NUM_MEMORIES]: Sized,
{
  comm_derefs: CombinedTableCommitment<G>,
  primary_sumcheck: PrimarySumcheck<G, { S::NUM_MEMORIES }>,
  memory_check: MemoryCheckingProof<G, C, M, S>,
}

impl<G: CurveGroup, const C: usize, const M: usize, S: SubtableStrategy<G::ScalarField, C, M>>
  SparsePolynomialEvaluationProof<G, C, M, S>
where
  [(); S::NUM_SUBTABLES]: Sized,
  [(); S::NUM_MEMORIES]: Sized,
{
  fn protocol_name() -> &'static [u8] {
    b"Surge SparsePolynomialEvaluationProof"
  }
  /// Prove an opening of the Sparse Matrix Polynomial
  /// - `dense`: DensifiedRepresentation
  /// - `r`: c log_m sized coordinates at which to prove the evaluation of the sparse polynomial
  /// - `eval`: evaluation of \widetilde{M}(r = (r_1, ..., r_logM))
  /// - `gens`: Commitment generator
  pub fn prove(
    dense: &mut DensifiedRepresentation<G::ScalarField, C>,
    r: &[Vec<G::ScalarField>; C], // 'log-m' sized point at which the polynomial is evaluated across 'c' dimensions
    gens: &SparsePolyCommitmentGens<G>,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self
  where
    [(); S::NUM_SUBTABLES]: Sized,
  {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    r.iter().for_each(|r_i| assert_eq!(r_i.len(), dense.log_m));

    let subtables = Subtables::<_, C, M, S>::new(&dense.dim_usize, r, dense.m, dense.s);

    // commit to non-deterministic choices of the prover
    let comm_derefs = {
      let comm = subtables.commit(&gens.gens_derefs);
      comm.append_to_transcript(b"comm_poly_row_col_ops_val", transcript);
      comm
    };

    let claimed_eval = subtables.compute_sumcheck_claim();

    <Transcript as ProofTranscript<G>>::append_scalar(
      transcript,
      b"claim_eval_scalar_product",
      &claimed_eval,
    );

    let (primary_sumcheck_proof, r_z, _) = SumcheckInstanceProof::<G::ScalarField>::prove_arbitrary::<
      _,
      G,
      Transcript,
      { S::NUM_MEMORIES },
    >(
      &claimed_eval,
      dense.s.log_2(),
      &mut subtables.lookup_polys.clone(),
      S::combine_lookups,
      S::sumcheck_poly_degree(),
      transcript,
    );

    // TODO(moodlezoup): Is it safe to reuse gens_derefs here?
    // Combined eval proof for E_i(r_z)
    let eval_derefs: [G::ScalarField; S::NUM_MEMORIES] =
      std::array::from_fn(|i| subtables.lookup_polys[i].evaluate(&r_z));
    let proof_derefs = CombinedTableEvalProof::prove(
      &subtables.combined_poly,
      &eval_derefs.to_vec(),
      &r_z,
      &gens.gens_derefs,
      transcript,
      random_tape,
    );

    let memory_check = {
      // produce a random element from the transcript for hash function
      let r_hash_params: Vec<G::ScalarField> =
        <Transcript as ProofTranscript<G>>::challenge_vector(transcript, b"challenge_r_hash", 2);

      MemoryCheckingProof::prove(
        dense,
        &(r_hash_params[0], r_hash_params[1]),
        &subtables,
        gens,
        transcript,
        random_tape,
      )
    };

    Self {
      comm_derefs,
      primary_sumcheck: PrimarySumcheck {
        proof: primary_sumcheck_proof,
        claimed_evaluation: claimed_eval,
        eval_derefs,
        proof_derefs,
      },
      memory_check,
    }
  }

  pub fn verify(
    &self,
    commitment: &SparsePolynomialCommitment<G>,
    r: &[Vec<G::ScalarField>; C], // point at which the polynomial is evaluated
    gens: &SparsePolyCommitmentGens<G>,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    r.iter()
      .for_each(|r_i| assert_eq!(r_i.len(), commitment.log_m));

    // add claims to transcript and obtain challenges for randomized mem-check circuit
    self
      .comm_derefs
      .append_to_transcript(b"comm_poly_row_col_ops_val", transcript);

    <Transcript as ProofTranscript<G>>::append_scalar(
      transcript,
      b"claim_eval_scalar_product",
      &self.primary_sumcheck.claimed_evaluation,
    );

    let (claim_last, r_z) = self.primary_sumcheck.proof.verify::<G, Transcript>(
      self.primary_sumcheck.claimed_evaluation,
      commitment.s.log_2(),
      S::sumcheck_poly_degree(),
      transcript,
    )?;

    // Verify that E_1(r_z) * ... * E_c(r_z) = claim_last
    assert_eq!(
      S::combine_lookups(&self.primary_sumcheck.eval_derefs),
      claim_last
    );

    self.primary_sumcheck.proof_derefs.verify(
      &r_z,
      &self.primary_sumcheck.eval_derefs,
      &gens.gens_derefs,
      &self.comm_derefs,
      transcript,
    )?;

    // produce a random element from the transcript for hash function
    let r_mem_check =
      <Transcript as ProofTranscript<G>>::challenge_vector(transcript, b"challenge_r_hash", 2);

    self.memory_check.verify(
      commitment,
      &self.comm_derefs,
      gens,
      r,
      &(r_mem_check[0], r_mem_check[1]),
      commitment.s,
      transcript,
    )
  }
}