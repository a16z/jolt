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
    alpha: usize,
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
    let num_vars_derefs = (alpha * s).next_power_of_two().log_2();

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

#[cfg(test)]
mod tests {
  use super::*;
  use ark_curve25519::{Fr, EdwardsProjective as G1Projective};
  use ark_std::rand::RngCore;
  use ark_std::{test_rng, UniformRand};

  use crate::sparse_mlpoly::subtables::and::AndSubtableStrategy;
  use crate::sparse_mlpoly::subtables::lt::LTSubtableStrategy;
  use crate::sparse_mlpoly::subtables::spark::SparkSubtableStrategy;

  #[test]
  fn prove_3d() {
    let mut prng = test_rng();

    // parameters
    const C: usize = 3;
    const M: usize = 256;
    let log_M: usize = M.log_2();
    let s: usize = 16;

    // generate sparse polynomial
    let mut nz: Vec<[usize; C]> = Vec::new();
    for _ in 0..s {
      let indices = [
        (prng.next_u64() as usize) % M,
        (prng.next_u64() as usize) % M,
        (prng.next_u64() as usize) % M,
      ];
      nz.push(indices);
    }

    let lookup_matrix = SparseLookupMatrix::new(nz, log_M);

    let mut dense: DensifiedRepresentation<Fr, C> = DensifiedRepresentation::from(&lookup_matrix);
    let gens = SparsePolyCommitmentGens::<G1Projective>::new(b"gens_sparse_poly", C, s, C, log_M);
    let commitment = dense.commit(&gens);

    let r: [Vec<Fr>; C] = std::array::from_fn(|_| {
      let mut r_i: Vec<Fr> = Vec::with_capacity(log_M);
      for _ in 0..log_M {
        r_i.push(Fr::rand(&mut prng));
      }
      r_i
    });

    // Prove
    let mut random_tape = RandomTape::new(b"proof");
    let mut prover_transcript = Transcript::new(b"example");
    let proof = SparsePolynomialEvaluationProof::<G1Projective, C, M, SparkSubtableStrategy>::prove(
      &mut dense,
      &r,
      &gens,
      &mut prover_transcript,
      &mut random_tape,
    );

    let mut verifier_transcript = Transcript::new(b"example");
    assert!(proof
      .verify(&commitment, &r, &gens, &mut verifier_transcript)
      .is_ok());
  }

  #[test]
  fn prove_4d() {
    let mut prng = test_rng();

    // parameters
    const C: usize = 4;
    const M: usize = 256;
    let log_M: usize = M.log_2();
    let s: usize = 16;

    // generate sparse polynomial
    let mut nz: Vec<[usize; C]> = Vec::new();
    for _ in 0..s {
      let indices = [
        (prng.next_u64() as usize) % M,
        (prng.next_u64() as usize) % M,
        (prng.next_u64() as usize) % M,
        (prng.next_u64() as usize) % M,
      ];
      nz.push(indices);
    }

    let lookup_matrix = SparseLookupMatrix::new(nz, log_M);

    let mut dense: DensifiedRepresentation<Fr, C> = DensifiedRepresentation::from(&lookup_matrix);
    let gens = SparsePolyCommitmentGens::<G1Projective>::new(b"gens_sparse_poly", C, s, C, log_M);
    let commitment = dense.commit::<G1Projective>(&gens);

    let r: [Vec<Fr>; C] = std::array::from_fn(|_| {
      let mut r_i: Vec<Fr> = Vec::with_capacity(log_M);
      for _ in 0..log_M {
        r_i.push(Fr::rand(&mut prng));
      }
      r_i
    });

    let mut random_tape = RandomTape::new(b"proof");
    let mut prover_transcript = Transcript::new(b"example");
    let proof = SparsePolynomialEvaluationProof::<G1Projective, C, M, SparkSubtableStrategy>::prove(
      &mut dense,
      &r,
      &gens,
      &mut prover_transcript,
      &mut random_tape,
    );

    let mut verifier_transcript = Transcript::new(b"example");
    assert!(proof
      .verify(&commitment, &r, &gens, &mut verifier_transcript)
      .is_ok());
  }

  #[test]
  fn prove_4d_and() {
    let mut prng = test_rng();

    // parameters
    const C: usize = 4;
    const M: usize = 16;
    let log_M: usize = M.log_2();
    let s: usize = 16;

    // generate sparse polynomial
    let mut nz: Vec<[usize; C]> = Vec::new();
    for _ in 0..s {
      let indices = [
        (prng.next_u64() as usize) % M,
        (prng.next_u64() as usize) % M,
        (prng.next_u64() as usize) % M,
        (prng.next_u64() as usize) % M,
      ];
      nz.push(indices);
    }

    let lookup_matrix = SparseLookupMatrix::new(nz, log_M);

    let mut dense: DensifiedRepresentation<Fr, C> = DensifiedRepresentation::from(&lookup_matrix);
    let gens = SparsePolyCommitmentGens::<G1Projective>::new(b"gens_sparse_poly", C, s, C, log_M);
    let commitment = dense.commit::<G1Projective>(&gens);

    let r: [Vec<Fr>; C] = std::array::from_fn(|_| {
      let mut r_i: Vec<Fr> = Vec::with_capacity(log_M);
      for _ in 0..log_M {
        r_i.push(Fr::rand(&mut prng));
      }
      r_i
    });

    let mut random_tape = RandomTape::new(b"proof");
    let mut prover_transcript = Transcript::new(b"example");
    let proof = SparsePolynomialEvaluationProof::<G1Projective, C, M, AndSubtableStrategy>::prove(
      &mut dense,
      &r,
      &gens,
      &mut prover_transcript,
      &mut random_tape,
    );

    let mut verifier_transcript = Transcript::new(b"example");
    assert!(proof
      .verify(&commitment, &r, &gens, &mut verifier_transcript)
      .is_ok());
  }

  #[test]
  fn prove_4d_lt() {
    let mut prng = test_rng();

    // parameters
    const C: usize = 4;
    const M: usize = 16;
    let log_M: usize = M.log_2();
    let s: usize = 16;

    // generate sparse polynomial
    let mut nz: Vec<[usize; C]> = Vec::new();
    for _ in 0..s {
      let indices = [
        (prng.next_u64() as usize) % M,
        (prng.next_u64() as usize) % M,
        (prng.next_u64() as usize) % M,
        (prng.next_u64() as usize) % M,
      ];
      nz.push(indices);
    }

    let lookup_matrix = SparseLookupMatrix::new(nz, log_M);

    let mut dense: DensifiedRepresentation<Fr, C> = DensifiedRepresentation::from(&lookup_matrix);
    let gens =
      SparsePolyCommitmentGens::<G1Projective>::new(b"gens_sparse_poly", C, s, C * 2, log_M);
    let commitment = dense.commit::<G1Projective>(&gens);

    let r: [Vec<Fr>; C] = std::array::from_fn(|_| {
      let mut r_i: Vec<Fr> = Vec::with_capacity(log_M);
      for _ in 0..log_M {
        r_i.push(Fr::rand(&mut prng));
      }
      r_i
    });

    let mut random_tape = RandomTape::new(b"proof");
    let mut prover_transcript = Transcript::new(b"example");
    let proof = SparsePolynomialEvaluationProof::<G1Projective, C, M, LTSubtableStrategy>::prove(
      &mut dense,
      &r,
      &gens,
      &mut prover_transcript,
      &mut random_tape,
    );

    let mut verifier_transcript = Transcript::new(b"example");
    assert!(proof
      .verify(&commitment, &r, &gens, &mut verifier_transcript)
      .is_ok());
  }

  /// Construct a 2d sparse integer matrix like the following:
  /// ```
  ///     let M: Vec<usize> = vec! [
  ///         0, 0, 0, 0,
  ///         1, 0, 1, 0,
  ///         0, 1, 0, 1,
  ///         0, 0, 0, 0
  ///    ];
  /// ```
  fn construct_2d_sparse_mat_polynomial_from_ints(
    entries: Vec<bool>,
    m: usize,
    log_m: usize,
  ) -> SparseLookupMatrix<2> {
    assert_eq!(m, log_m.pow2());
    let mut row_index = 0usize;
    let mut column_index = 0usize;
    let mut nz: Vec<[usize; 2]> = Vec::new();
    for entry in entries {
      if entry {
        println!("Non-zero: (row, col): ({row_index}, {column_index})",);
        nz.push([row_index, column_index]);
      }
      column_index += 1;
      if column_index >= m {
        column_index = 0;
        row_index += 1;
      }
    }

    SparseLookupMatrix::<2>::new(nz, log_m)
  }

  /// Returns a tuple of (c, s, m, log_m, SparsePoly)
  fn construct_2d_small<G: CurveGroup>() -> (usize, usize, usize, usize, SparseLookupMatrix<2>) {
    let c = 2usize;
    let s = 4usize;
    let m = 4usize;
    let log_m = 2usize;

    let M: Vec<bool> = vec![0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0]
      .iter()
      .map(|&binary_val| binary_val != 0)
      .collect();
    (
      c,
      s,
      m,
      log_m,
      construct_2d_sparse_mat_polynomial_from_ints(M, m, log_m),
    )
  }

  // #[test]
  // fn prove_2d() {
  //   let mut prng = test_rng();
  //   const C: usize = 2;

  //   let (_, s, _m, log_m, lookup_matrix) = construct_2d_small::<G1Projective>();

  //   // Commit
  //   let mut dense: DensifiedRepresentation<Fr, C> = DensifiedRepresentation::from(&lookup_matrix);
  //   let gens = SparsePolyCommitmentGens::<G1Projective>::new(b"gens_sparse_poly", C, s, C, log_m);
  //   let commitment = dense.commit(&gens);

  //   let r: [Vec<Fr>; C] = std::array::from_fn(|_| {
  //     let mut r_i: Vec<Fr> = Vec::with_capacity(log_m);
  //     for _ in 0..log_m {
  //       r_i.push(Fr::rand(&mut prng));
  //     }
  //     r_i
  //   });

  //   // Prove
  //   let mut random_tape = RandomTape::new(b"proof");
  //   let mut prover_transcript = Transcript::new(b"example");
  //   let proof = SparsePolynomialEvaluationProof::<G1Projective, C, M, SparkSubtableStrategy>::prove(
  //     &mut dense,
  //     &r,
  //     &gens,
  //     &mut prover_transcript,
  //     &mut random_tape,
  //   );

  //   let mut verifier_transcript = Transcript::new(b"example");
  //   assert!(proof
  //     .verify(&commitment, &r, &gens, &mut verifier_transcript)
  //     .is_ok());
  // }
}
