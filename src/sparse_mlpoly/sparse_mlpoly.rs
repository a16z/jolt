#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

use crate::dense_mlpoly::{EqPolynomial, PolyCommitment, PolyCommitmentGens};
use crate::errors::ProofVerifyError;
use crate::math::Math;
use crate::random::RandomTape;
use crate::sparse_mlpoly::densified::DensifiedRepresentation;
use crate::sparse_mlpoly::memory_checking::MemoryCheckingProof;
use crate::sparse_mlpoly::subtables::{
  CombinedTableCommitment, CombinedTableEvalProof, EqSubtableStrategy, SubtableStrategy, Subtables,
};
use crate::sumcheck::SumcheckInstanceProof;
use crate::transcript::{AppendToTranscript, ProofTranscript};
use ark_ec::CurveGroup;
use ark_ff::PrimeField;
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
      gens_combined_l_variate: gens_combined_l_variate,
      gens_combined_log_m_variate: gens_combined_log_m_variate,
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

  // TODO(moodlezoup): r: &[Vec<F>; C]
  pub fn evaluate_mle<F: PrimeField>(&self, r: &Vec<F>) -> F {
    assert_eq!(C * self.log_m, r.len());

    // TODO: Move to DenseRepresenation call

    // \tilde{M}(r) = \sum_k [val(k) * \prod_i E_i(k)]
    // where E_i(k) = \tilde{eq}(to-bits(dim_i(k)), r_i)
    let evals: Vec<Vec<F>> = r
      .chunks_exact(self.log_m)
      .map(|r_i| EqPolynomial::new(r_i.to_vec()).evals())
      .collect();

    self
      .nz
      .iter()
      .map(|indices| (0..C).map(|i| evals[i][indices[i]]).product::<F>())
      .sum()
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
struct PrimarySumcheck<G: CurveGroup, const ALPHA: usize> {
  proof: SumcheckInstanceProof<G::ScalarField>,
  eval_derefs: [G::ScalarField; ALPHA],
  proof_derefs: CombinedTableEvalProof<G, ALPHA>,
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct SparsePolynomialEvaluationProof<G: CurveGroup, const C: usize, const ALPHA: usize> {
  comm_derefs: CombinedTableCommitment<G>,
  primary_sumcheck: PrimarySumcheck<G, ALPHA>,
  memory_check: MemoryCheckingProof<G, C, ALPHA>,
}

impl<G: CurveGroup, const C: usize, const ALPHA: usize>
  SparsePolynomialEvaluationProof<G, C, ALPHA>
{
  fn protocol_name() -> &'static [u8] {
    b"Surge SparsePolynomialEvaluationProof"
  }
  /// Prove an opening of the Sparse Matrix Polynomial
  /// - `dense`: DensifiedRepresentation
  /// - `r`: c log_m sized coordinates at which to prove the evaluation of the sparse polynomial
  /// - `eval`: evaluation of \widetilde{M}(r = (r_1, ..., r_logM))
  /// - `gens`: Commitment generator
  pub fn prove<S: SubtableStrategy<G::ScalarField, C, ALPHA>>(
    dense: &mut DensifiedRepresentation<G::ScalarField, C>,
    r: &[Vec<G::ScalarField>; C], // 'log-m' sized point at which the polynomial is evaluated across 'c' dimensions
    eval: &G::ScalarField,        // a evaluation of \widetilde{M}(r = (r_1, ..., r_logM))
    gens: &SparsePolyCommitmentGens<G>,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    r.iter().for_each(|r_i| assert_eq!(r_i.len(), dense.log_m));

    let subtables = Subtables::<_, C, ALPHA, S>::new(&dense.dim_usize, r, dense.m, dense.s);

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

    assert_eq!(claimed_eval, *eval);

    let (primary_sumcheck_proof, r_z, _) =
      SumcheckInstanceProof::<G::ScalarField>::prove_arbitrary::<_, G, Transcript, ALPHA>(
        &claimed_eval,
        dense.s.log_2(),
        &mut subtables.lookup_polys.clone(),
        &S::combine_lookups,
        S::sumcheck_poly_degree(),
        transcript,
      );

    // TODO(moodlezoup): Is it safe to reuse gens_derefs here?
    // Combined eval proof for E_i(r_z)
    let eval_derefs: [G::ScalarField; ALPHA] =
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

      let memory_checking_proof = MemoryCheckingProof::prove(
        dense,
        &(r_hash_params[0], r_hash_params[1]),
        &subtables,
        gens,
        transcript,
        random_tape,
      );

      memory_checking_proof
    };

    Self {
      comm_derefs,
      primary_sumcheck: PrimarySumcheck {
        proof: primary_sumcheck_proof,
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
    evaluation: &G::ScalarField,  // evaluation of \widetilde{M}(r = (rx,ry))
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
      &evaluation,
    );

    let (claim_last, r_z) = self.primary_sumcheck.proof.verify::<G, Transcript>(
      *evaluation,
      commitment.s.log_2(),
      C,
      transcript,
    )?;

    self.primary_sumcheck.proof_derefs.verify(
      &r_z,
      &self.primary_sumcheck.eval_derefs,
      &gens.gens_derefs,
      &self.comm_derefs,
      transcript,
    )?;

    // Verify that E_1(r_z) * ... * E_c(r_z) = claim_last
    assert_eq!(
      self
        .primary_sumcheck
        .eval_derefs
        .iter()
        .product::<G::ScalarField>(),
      claim_last
    );

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
  use ark_bls12_381::{Fr, G1Projective};
  use ark_std::rand::RngCore;
  use ark_std::{test_rng, One, UniformRand};

  use crate::sparse_mlpoly::subtables::EqSubtableStrategy;
  use crate::utils::{ff_bitvector_dbg, index_to_field_bitvector};

  #[test]
  fn check_evaluation() {
    check_evaluation_helper::<G1Projective>()
  }
  fn check_evaluation_helper<G: CurveGroup>() {
    let mut prng = test_rng();

    // parameters
    let num_entries: usize = 64;
    let s: usize = 8;
    const c: usize = 3;
    let log_m: usize = num_entries.log_2() / c; // 2
    let m: usize = log_m.pow2(); // 2 ^ 2 = 4

    let mut nz: Vec<[usize; c]> = Vec::new();
    for _ in 0..s {
      let indices = [
        (prng.next_u64() as usize) % m,
        (prng.next_u64() as usize) % m,
        (prng.next_u64() as usize) % m,
      ];
      nz.push(indices);
    }

    let lookup_matrix = SparseLookupMatrix::new(nz, log_m);
    let gens = SparsePolyCommitmentGens::<G>::new(b"gens_sparse_poly", c, s, c, log_m);

    // evaluation
    let r: Vec<G::ScalarField> = (0..c * log_m)
      .map(|_| G::ScalarField::rand(&mut prng))
      .collect();
    let evaluation = lookup_matrix.evaluate_mle(&r);
    // println!("r: {:?}", r);
    // println!("eval: {}", eval);

    let dense: DensifiedRepresentation<G::ScalarField, c> =
      DensifiedRepresentation::from(&lookup_matrix);
    // for i in 0..c {
    //   println!("i: {:?}", i);
    //   println!("dim: {:?}", dense.dim[i]);
    //   println!("read: {:?}", dense.read[i]);
    //   println!("final: {:?}\n", dense.r#final[i]);
    // }
    // println!("val: {:?}", dense.val);

    // // dim + read + val => log2((2c + 1) * s)
    // println!(
    //   "combined l-variate multilinear polynomial has {} variables",
    //   dense.combined_l_variate_polys.get_num_vars()
    // );
    // // final => log2(c * m)
    // println!(
    //   "combined log(m)-variate multilinear polynomial has {} variables",
    //   dense.combined_log_m_variate_polys.get_num_vars()
    // );

    let gens = SparsePolyCommitmentGens::<G>::new(b"gens_sparse_poly", c, s, c, log_m);
    let commitment = dense.commit::<G>(&gens);

    let mut random_tape = RandomTape::<G>::new(b"proof");
    let mut prover_transcript = Transcript::new(b"example");
    // let proof = SparsePolynomialEvaluationProof::prove(
    //   &dense,
    //   &r,
    //   &evaluation,
    //   &gens,
    //   &mut prover_transcript,
    //   &mut random_tape,
    // );

    // let mut verifier_transcript = Transcript::new(b"example");
    // assert!(proof
    //   .verify(&commitment, &r, &evals, &gens, &mut verifier_transcript)
    //   .is_ok());
  }

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
    let flat_r: Vec<Fr> = r.clone().into_iter().flatten().collect();
    let eval = lookup_matrix.evaluate_mle(&flat_r);

    // Prove
    let mut random_tape = RandomTape::new(b"proof");
    let mut prover_transcript = Transcript::new(b"example");
    let proof = SparsePolynomialEvaluationProof::<G1Projective, C, C>::prove::<EqSubtableStrategy>(
      &mut dense,
      &r,
      &eval,
      &gens,
      &mut prover_transcript,
      &mut random_tape,
    );

    let mut verifier_transcript = Transcript::new(b"example");
    assert!(proof
      .verify(&commitment, &r, &eval, &gens, &mut verifier_transcript)
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
    let flat_r: Vec<Fr> = r.clone().into_iter().flatten().collect();
    let eval = lookup_matrix.evaluate_mle(&flat_r);

    let mut random_tape = RandomTape::new(b"proof");
    let mut prover_transcript = Transcript::new(b"example");
    let proof = SparsePolynomialEvaluationProof::<G1Projective, C, C>::prove::<EqSubtableStrategy>(
      &mut dense,
      &r,
      &eval,
      &gens,
      &mut prover_transcript,
      &mut random_tape,
    );

    let mut verifier_transcript = Transcript::new(b"example");
    assert!(proof
      .verify(&commitment, &r, &eval, &gens, &mut verifier_transcript)
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

  #[test]
  fn evaluate_over_known_indices() {
    // Create SparseMLE and then evaluate over known indices and confirm correct evaluations
    let (c, s, m, log_m, lookup_matrix) = construct_2d_small::<G1Projective>();

    // Evaluations
    // poly[row, col] = eval
    // poly[1, 0] = 2
    // poly[1, 2] = 4
    // poly[2, 1] = 8
    // poly[2, 3] = 9
    // Check each and a few others over the boolean hypercube to be 0

    // poly[1, 0] = 2
    let row: Vec<Fr> = index_to_field_bitvector(1, log_m);
    let col: Vec<Fr> = index_to_field_bitvector(0, log_m);
    let combined_index: Vec<Fr> = vec![row, col].concat();
    assert_eq!(lookup_matrix.evaluate_mle(&combined_index), Fr::from(1));

    // poly[1, 2] = 4
    let row: Vec<Fr> = index_to_field_bitvector(1, log_m);
    let col: Vec<Fr> = index_to_field_bitvector(2, log_m);
    let combined_index: Vec<Fr> = vec![row, col].concat();
    assert_eq!(lookup_matrix.evaluate_mle(&combined_index), Fr::from(1));

    // poly[2, 1] = 8
    let row: Vec<Fr> = index_to_field_bitvector(2, log_m);
    let col: Vec<Fr> = index_to_field_bitvector(1, log_m);
    let combined_index: Vec<Fr> = vec![row, col].concat();
    assert_eq!(lookup_matrix.evaluate_mle(&combined_index), Fr::from(1));

    // poly[2, 3] = 9
    let row: Vec<Fr> = index_to_field_bitvector(2, log_m);
    let col: Vec<Fr> = index_to_field_bitvector(3, log_m);
    let combined_index: Vec<Fr> = vec![row, col].concat();
    assert_eq!(lookup_matrix.evaluate_mle(&combined_index), Fr::from(1));
  }

  #[test]
  fn prove_2d() {
    let mut prng = test_rng();
    const c: usize = 2;

    let (_, s, m, log_m, lookup_matrix) = construct_2d_small::<G1Projective>();

    // Commit
    let mut dense: DensifiedRepresentation<Fr, c> = DensifiedRepresentation::from(&lookup_matrix);
    let gens = SparsePolyCommitmentGens::<G1Projective>::new(b"gens_sparse_poly", c, s, c, log_m);
    let commitment = dense.commit(&gens);

    let r: [Vec<Fr>; c] = std::array::from_fn(|_| {
      let mut r_i: Vec<Fr> = Vec::with_capacity(log_m);
      for _ in 0..log_m {
        r_i.push(Fr::rand(&mut prng));
      }
      r_i
    });
    let flat_r: Vec<Fr> = r.clone().into_iter().flatten().collect();
    let eval = lookup_matrix.evaluate_mle(&flat_r);

    // Prove
    let mut random_tape = RandomTape::new(b"proof");
    let mut prover_transcript = Transcript::new(b"example");
    let proof = SparsePolynomialEvaluationProof::<G1Projective, c, c>::prove::<EqSubtableStrategy>(
      &mut dense,
      &r,
      &eval,
      &gens,
      &mut prover_transcript,
      &mut random_tape,
    );

    let mut verifier_transcript = Transcript::new(b"example");
    assert!(proof
      .verify(&commitment, &r, &eval, &gens, &mut verifier_transcript)
      .is_ok());
  }
}
