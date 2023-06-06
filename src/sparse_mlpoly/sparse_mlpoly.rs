#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
use crate::dense_mlpoly::DensePolynomial;
use crate::dense_mlpoly::{EqPolynomial, PolyCommitment, PolyCommitmentGens};
use crate::errors::ProofVerifyError;
use crate::math::Math;
use crate::product_tree::GeneralizedScalarProduct;
use crate::random::RandomTape;
use crate::sparse_mlpoly::densified::DensifiedRepresentation;
use crate::sparse_mlpoly::subtable_evaluations::CombinedTableCommitment;
use crate::sparse_mlpoly::memory_checking::MemoryCheckingProof;
use crate::sumcheck::SumcheckInstanceProof;
use crate::transcript::{AppendToTranscript, ProofTranscript};
use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_serialize::*;

use merlin::Transcript;

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct SparseMatEntry<F: PrimeField, const C: usize> {
  pub indices: [usize; C],
  pub val: F, // TODO(moodlezoup) always 1 for Lasso; delete?
}

impl<F: PrimeField, const C: usize> SparseMatEntry<F, C> {
  pub fn new(indices: [usize; C], val: F) -> Self {
    SparseMatEntry { indices, val }
  }
}

pub struct SparseMatPolyCommitmentGens<G> {
  pub gens_combined_l_variate: PolyCommitmentGens<G>,
  pub gens_combined_log_m_variate: PolyCommitmentGens<G>,
  pub gens_derefs: PolyCommitmentGens<G>,
}

impl<G: CurveGroup> SparseMatPolyCommitmentGens<G> {
  pub fn new(
    label: &'static [u8],
    c: usize,
    s: usize,
    log_m: usize,
  ) -> SparseMatPolyCommitmentGens<G> {
    // dim + read + val
    // log_2(cs + cs + s) = log_2(2cs + s)
    let num_vars_combined_l_variate = (2 * c * s + s).next_power_of_two().log_2();
    // final
    // log_2(c * m) = log_2(c) + log_2(m)
    let num_vars_combined_log_m_variate = c.next_power_of_two().log_2() + log_m;
    // TODO(moodlezoup): idk if this is the right number
    let num_vars_derefs = s.next_power_of_two().log_2() as usize + 1;

    let gens_combined_l_variate = PolyCommitmentGens::new(num_vars_combined_l_variate, label);
    let gens_combined_log_m_variate =
      PolyCommitmentGens::new(num_vars_combined_log_m_variate, label);
    let gens_derefs = PolyCommitmentGens::new(num_vars_derefs, label);
    SparseMatPolyCommitmentGens {
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
  pub m: usize, // TODO: big integer
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
pub struct SparseMatPolynomial<F: PrimeField, const C: usize> {
  pub nonzero_entries: Vec<SparseMatEntry<F, C>>,
  pub s: usize, // sparsity
  pub log_m: usize,
  pub m: usize, // TODO: big integer
}

impl<F: PrimeField, const C: usize> SparseMatPolynomial<F, C> {
  pub fn new(nonzero_entries: Vec<SparseMatEntry<F, C>>, log_m: usize) -> Self {
    let s = nonzero_entries.len().next_power_of_two();
    // TODO(moodlezoup):
    // nonzero_entries.resize(s, F::zero());

    SparseMatPolynomial {
      nonzero_entries,
      s,
      log_m,
      m: log_m.pow2(),
    }
  }

  pub fn evaluate(&self, r: &Vec<F>) -> F {
    assert_eq!(C * self.log_m, r.len());

    // \tilde{M}(r) = \sum_k [val(k) * \prod_i E_i(k)]
    // where E_i(k) = \tilde{eq}(to-bits(dim_i(k)), r_i)
    self
      .nonzero_entries
      .iter()
      .map(|entry| {
        r.chunks_exact(self.log_m)
          .enumerate()
          .map(|(i, r_i)| {
            let E_i = EqPolynomial::new(r_i.to_vec()).evals();
            E_i[entry.indices[i]]
          })
          .product::<F>()
          .mul(entry.val)
      })
      .sum()
  }

  fn to_densified(&self) -> DensifiedRepresentation<F, C> {
    // TODO(moodlezoup) Initialize as arrays using std::array::from_fn ?
    let mut dim_usize: Vec<Vec<usize>> = Vec::with_capacity(C);
    let mut dim: Vec<DensePolynomial<F>> = Vec::with_capacity(C);
    let mut read: Vec<DensePolynomial<F>> = Vec::with_capacity(C);
    let mut r#final: Vec<DensePolynomial<F>> = Vec::with_capacity(C);

    for i in 0..C {
      let mut access_sequence = self
        .nonzero_entries
        .iter()
        .map(|entry| entry.indices[i])
        .collect::<Vec<usize>>();
      // TODO(moodlezoup) Is this resize necessary/in the right place?
      access_sequence.resize(self.s, 0usize);

      let mut final_timestamps = vec![0usize; self.m];
      let mut read_timestamps = vec![0usize; self.s];

      // since read timestamps are trustworthy, we can simply increment the r-ts to obtain a w-ts
      // this is sufficient to ensure that the write-set, consisting of (addr, val, ts) tuples, is a set
      for i in 0..self.s {
        let memory_address = access_sequence[i];
        assert!(memory_address < self.m);
        let ts = final_timestamps[memory_address];
        read_timestamps[i] = ts;
        let write_timestamp = ts + 1;
        final_timestamps[memory_address] = write_timestamp;
      }

      dim.push(DensePolynomial::from_usize(&access_sequence));
      read.push(DensePolynomial::from_usize(&read_timestamps));
      r#final.push(DensePolynomial::from_usize(&final_timestamps));
      dim_usize.push(access_sequence);
    }

    let mut values: Vec<F> = self.nonzero_entries.iter().map(|entry| entry.val).collect();
    // TODO(moodlezoup) Is this resize necessary/in the right place?
    values.resize(self.s, F::zero());

    let val = DensePolynomial::new(values);

    let mut l_variate_polys = [dim.as_slice(), read.as_slice()].concat();
    l_variate_polys.push(val.clone());

    let combined_l_variate_polys = DensePolynomial::merge(&l_variate_polys);
    let combined_log_m_variate_polys = DensePolynomial::merge(&r#final);

    DensifiedRepresentation {
      dim_usize,
      dim: dim.try_into().unwrap(),
      read: read.try_into().unwrap(),
      r#final: r#final.try_into().unwrap(),
      val,
      combined_l_variate_polys,
      combined_log_m_variate_polys,
      s: self.s,
      log_m: self.log_m,
      m: self.m,
      table_evals: vec![]
    }
  }
}

#[derive(Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct SparsePolynomialEvaluationProof<G: CurveGroup, const C: usize> {
  comm_derefs: CombinedTableCommitment<G>,
  primary_sumcheck_proof: SumcheckInstanceProof<G::ScalarField>,
  memory_check: MemoryCheckingProof<G, C>,
}

impl<G: CurveGroup, const C: usize> SparsePolynomialEvaluationProof<G, C> {
  fn protocol_name() -> &'static [u8] {
    b"Sparse polynomial evaluation proof"
  }
  /// Prove an opening of the Sparse Matrix Polynomial
  /// - `dense`: DensifiedRepresentation
  /// - `r`: c log_m sized coordinates at which to prove the evaluation of the sparse polynomial
  /// - `eval`: evaluation of \widetilde{M}(r = (r_1, ..., r_logM))
  /// - `gens`: Commitment generator
  pub fn prove(
    dense: &mut DensifiedRepresentation<G::ScalarField, C>,
    r: &[Vec<G::ScalarField>; C], // 'log-m' sized point at which the polynomial is evaluated across 'c' dimensions
    eval: &G::ScalarField,        // a evaluation of \widetilde{M}(r = (r_1, ..., r_logM))
    gens: &SparseMatPolyCommitmentGens<G>,
    transcript: &mut Transcript,
    random_tape: &mut RandomTape<G>,
  ) -> Self {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    r.iter().for_each(|r_i| assert_eq!(r_i.len(), dense.log_m));


    // eqs are the evaluations of eq(i_1, r_1) , ... , eq(i_c, r_c)
    // Where i_1, ... i_c are all \in {0, 1}^logM for the non-sparse indices (s)-sized
    // And r_1, ... r_c are all \in F^logM
    // Derefs converts each eqs into E_{r_i}
    dense.materialize_table(r);

    // Combine subtable evaluations to allow commitment
    let combined_subtable_evaluations = dense.combine_subtable_evaluations();

    // commit to non-deterministic choices of the prover
    let comm_derefs = {
      let comm = combined_subtable_evaluations.commit(&gens.gens_derefs);
      comm.append_to_transcript(b"comm_poly_row_col_ops_val", transcript);
      comm
    };

    // TODO(moodlezoup): Move scalar product stuff into separate prove/verify
    // prepare scalar product
    let mut scalar_product_operands = combined_subtable_evaluations.subtable_evals.clone().to_vec();
    scalar_product_operands.push(dense.val.clone());
    let scalar_product = GeneralizedScalarProduct::new(scalar_product_operands.clone());
    let eval_scalar_product = scalar_product.evaluate();

    <Transcript as ProofTranscript<G>>::append_scalar(
      transcript,
      b"claim_eval_scalar_product",
      &eval_scalar_product,
    );

    assert_eq!(eval_scalar_product, *eval);

    let (primary_sumcheck_proof, _, primary_sumcheck_claims) =
      SumcheckInstanceProof::<G::ScalarField>::prove_arbitrary::<_, G, Transcript>(
        &eval_scalar_product,
        scalar_product_operands[0].len().log_2(),
        &mut scalar_product_operands,
        |polys| -> G::ScalarField { polys.iter().product() },
        transcript,
      );

    <Transcript as ProofTranscript<G>>::append_scalars(
      transcript,
      b"primary_sumcheck_claims",
      &primary_sumcheck_claims,
    );

    let memory_check = {
      // produce a random element from the transcript for hash function
      let r_hash_params: Vec<G::ScalarField> =
        <Transcript as ProofTranscript<G>>::challenge_vector(transcript, b"challenge_r_hash", 2);

      let mut grand_products = dense.to_grand_products(&(r_hash_params[0], r_hash_params[1]));

      let poly_eval_network_proof = MemoryCheckingProof::prove(
        &mut grand_products,
        dense,
        &combined_subtable_evaluations,
        gens,
        transcript,
        random_tape,
      );

      poly_eval_network_proof
    };

    Self {
      comm_derefs,
      primary_sumcheck_proof,
      memory_check,
    }
  }

  pub fn verify(
    &self,
    commitment: &SparsePolynomialCommitment<G>,
    r: &[Vec<G::ScalarField>; C], // point at which the polynomial is evaluated
    evaluation: &G::ScalarField,  // evaluation of \widetilde{M}(r = (rx,ry))
    gens: &SparseMatPolyCommitmentGens<G>,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    <Transcript as ProofTranscript<G>>::append_protocol_name(transcript, Self::protocol_name());

    assert_eq!(r[0].len(), commitment.log_m);

    // add claims to transcript and obtain challenges for randomized mem-check circuit
    self
      .comm_derefs
      .append_to_transcript(b"comm_poly_row_col_ops_val", transcript);

    // // TODO(moodlezoup): verify primary sumcheck
    // self.primary_sumcheck_proof.verify(evaluation, num_rounds, degree_bound, transcript)

    // // TODO(moodlezoup): verify the decommitments used in primary sum-check
    // let eval_val_vec = &self.eval_val;
    // assert_eq!(claims_dotp.len(), 3 * eval_row_ops_val.len());
    // for i in 0..claims_dotp.len() / 3 {
    //   let claim_row_ops_val = claims_dotp[3 * i];
    //   let claim_col_ops_val = claims_dotp[3 * i + 1];
    //   let claim_val = claims_dotp[3 * i + 2];

    //   assert_eq!(claim_row_ops_val, eval_row_ops_val[i]);
    //   assert_eq!(claim_col_ops_val, eval_col_ops_val[i]);
    //   assert_eq!(claim_val, eval_val_vec[i]);
    // }

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
  use ark_std::test_rng;
  use ark_std::UniformRand;

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

    let mut nonzero_entries: Vec<SparseMatEntry<G::ScalarField, c>> = Vec::new();
    for _ in 0..s {
      let indices = [
        (prng.next_u64() as usize) % m,
        (prng.next_u64() as usize) % m,
        (prng.next_u64() as usize) % m,
      ];
      let entry = SparseMatEntry::new(indices, G::ScalarField::rand(&mut prng));
      println!("{:?}", entry);
      nonzero_entries.push(entry);
    }

    let sparse_poly = SparseMatPolynomial::new(nonzero_entries, log_m);
    let gens = SparseMatPolyCommitmentGens::<G>::new(b"gens_sparse_poly", c, s, log_m);

    // evaluation
    let r: Vec<G::ScalarField> = (0..c * log_m)
      .map(|_| G::ScalarField::rand(&mut prng))
      .collect();
    let evaluation = sparse_poly.evaluate(&r);
    // println!("r: {:?}", r);
    // println!("eval: {}", eval);

    let dense: DensifiedRepresentation<G::ScalarField, c> = sparse_poly.into();
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

    let (gens, commitment) = dense.commit::<G>();

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

  // #[test]
  fn check_sparse_polyeval_proof() {
    check_sparse_polyeval_proof_helper::<G1Projective>()
  }
  fn check_sparse_polyeval_proof_helper<G: CurveGroup>() {
    let mut prng = test_rng();

    // parameters
    let num_entries: usize = 256 * 256;
    let s: usize = 256;
    const c: usize = 4;
    let log_m: usize = num_entries.log_2() / c; // 4
    let m: usize = log_m.pow2(); // 2 ^ 4 = 16

    // generate sparse polynomial
    let mut nonzero_entries: Vec<SparseMatEntry<G::ScalarField, c>> = Vec::new();
    for _ in 0..s {
      let indices = [
        (prng.next_u64() as usize) % m,
        (prng.next_u64() as usize) % m,
        (prng.next_u64() as usize) % m,
        (prng.next_u64() as usize) % m,
      ];
      let entry = SparseMatEntry::new(indices, G::ScalarField::rand(&mut prng));
      nonzero_entries.push(entry);
    }

    let sparse_poly = SparseMatPolynomial::new(nonzero_entries, log_m);

    // evaluation
    let r: Vec<G::ScalarField> = (0..c * log_m)
      .map(|_| G::ScalarField::rand(&mut prng))
      .collect();
    let eval = sparse_poly.evaluate(&r);

    // commitment
    let dense: DensifiedRepresentation<G::ScalarField, c> = sparse_poly.into();
    let (gens, commitment) = dense.commit::<G>();

    // let mut random_tape = RandomTape::new(b"proof");
    // let mut prover_transcript = Transcript::new(b"example");
    // let proof = SparseMatPolyEvalProof::prove(
    //   &dense,
    //   &r,
    //   &evals,
    //   &gens,
    //   &mut prover_transcript,
    //   &mut random_tape,
    // );

    // let mut verifier_transcript = Transcript::new(b"example");
    // assert!(proof
    //   .verify(&commitment, &r, &evals, &gens, &mut verifier_transcript)
    //   .is_ok());
  }

  /// Construct a 2d sparse integer matrix like the following:
  /// ```
  ///     let M: Vec<usize> = vec! [
  ///         0, 0, 0, 0,
  ///         2, 0, 4, 0,
  ///         0, 8, 0, 9,
  ///         0, 0, 0, 0
  ///    ];
  /// ```
  fn construct_2d_sparse_mat_polynomial_from_ints<F: PrimeField>(
    ints: Vec<usize>,
    m: usize,
    log_m: usize,
    s: usize,
  ) -> SparseMatPolynomial<F, 2> {
    assert_eq!(m, log_m.pow2());
    let mut row_index = 0usize;
    let mut column_index = 0usize;
    let mut sparse_evals: Vec<SparseMatEntry<F, 2>> = Vec::new();
    for entry_index in 0..ints.len() {
      if ints[entry_index] != 0 {
        println!(
          "Non-sparse: (row, col, val): ({row_index}, {column_index}, {})",
          ints[entry_index]
        );
        sparse_evals.push(SparseMatEntry::new(
          [row_index, column_index],
          F::from(ints[entry_index] as u64),
        ));
      }

      column_index += 1;
      if column_index >= m {
        column_index = 0;
        row_index += 1;
      }
    }

    SparseMatPolynomial::<F, 2>::new(sparse_evals, log_m)
  }

  /// Returns a tuple of (c, s, m, log_m, SparsePoly)
  fn construct_2d_small<G: CurveGroup>() -> (
    usize,
    usize,
    usize,
    usize,
    SparseMatPolynomial<G::ScalarField, 2>,
  ) {
    let c = 2usize;
    let s = 4usize;
    let m = 4usize;
    let log_m = 2usize;

    let M: Vec<usize> = vec![0, 0, 0, 0, 2, 0, 4, 0, 0, 8, 0, 9, 0, 0, 0, 0];
    (
      c,
      s,
      m,
      log_m,
      construct_2d_sparse_mat_polynomial_from_ints(M, m, log_m, s),
    )
  }

  #[test]
  fn evaluate_over_known_indices() {
    // Create SparseMLE and then evaluate over known indices and confirm correct evaluations
    let (c, s, m, log_m, sparse_poly) = construct_2d_small::<G1Projective>();

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
    assert_eq!(sparse_poly.evaluate(&combined_index), Fr::from(2));

    // poly[1, 2] = 4
    let row: Vec<Fr> = index_to_field_bitvector(1, log_m);
    let col: Vec<Fr> = index_to_field_bitvector(2, log_m);
    let combined_index: Vec<Fr> = vec![row, col].concat();
    assert_eq!(sparse_poly.evaluate(&combined_index), Fr::from(4));

    // poly[2, 1] = 8
    let row: Vec<Fr> = index_to_field_bitvector(2, log_m);
    let col: Vec<Fr> = index_to_field_bitvector(1, log_m);
    let combined_index: Vec<Fr> = vec![row, col].concat();
    assert_eq!(sparse_poly.evaluate(&combined_index), Fr::from(8));

    // poly[2, 3] = 9
    let row: Vec<Fr> = index_to_field_bitvector(2, log_m);
    let col: Vec<Fr> = index_to_field_bitvector(3, log_m);
    let combined_index: Vec<Fr> = vec![row, col].concat();
    assert_eq!(sparse_poly.evaluate(&combined_index), Fr::from(9));
  }

  #[test]
  fn prove() {
    let mut prng = test_rng();
    const c: usize = 2;

    let (_, s, m, log_m, sparse_poly) = construct_2d_small::<G1Projective>();

    // Commit
    let mut dense: DensifiedRepresentation<Fr, c> = sparse_poly.to_densified();
    let (gens, commitment) = dense.commit();

    let r: [Vec<Fr>; c] = std::array::from_fn(|_| {
      let mut r_i: Vec<Fr> = Vec::with_capacity(log_m);
      for _ in 0..log_m {
        r_i.push(Fr::rand(&mut prng));
      }
      r_i
    });
    let flat_r: Vec<Fr> = r.clone().into_iter().flatten().collect();
    let eval = sparse_poly.evaluate(&flat_r);

    // Prove
    let mut random_tape = RandomTape::new(b"proof");
    let mut prover_transcript = Transcript::new(b"example");
    let proof = SparsePolynomialEvaluationProof::<G1Projective, c>::prove(
      &mut dense,
      &r,
      &eval,
      &gens,
      &mut prover_transcript,
      &mut random_tape,
    );
  }
}
