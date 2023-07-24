use std::time::{Duration, Instant};

use crate::lasso::surge::SparsePolyCommitmentGens;
use crate::subtables::and::AndSubtableStrategy;
use crate::{
  lasso::{
    densified::DensifiedRepresentation,
    surge::{SparseLookupMatrix, SparsePolynomialEvaluationProof},
  },
  utils::random::RandomTape,
};
use ark_curve25519::{EdwardsProjective, Fr};
use ark_ff::PrimeField;
use ark_std::{log2, test_rng};
use merlin::Transcript;
use num_integer::Roots;
use rand_chacha::rand_core::RngCore;

pub fn gen_indices<const C: usize>(sparsity: usize, memory_size: usize) -> Vec<[usize; C]> {
  let mut rng = test_rng();
  let mut all_indices: Vec<[usize; C]> = Vec::new();
  for _ in 0..sparsity {
    let indices = [rng.next_u64() as usize % memory_size; C];
    all_indices.push(indices);
  }
  all_indices
}

pub fn gen_random_points<F: PrimeField, const C: usize>(memory_bits: usize) -> [Vec<F>; C] {
  std::array::from_fn(|_| gen_random_point(memory_bits))
}

pub fn gen_random_point<F: PrimeField>(memory_bits: usize) -> Vec<F> {
  let mut rng = test_rng();
  let mut r_i: Vec<F> = Vec::with_capacity(memory_bits);
  for _ in 0..memory_bits {
    r_i.push(F::rand(&mut rng));
  }
  r_i
}

macro_rules! single_pass_lasso {
  ($span_name:expr, $field:ty, $group:ty, $subtable_strategy:ty, $N:expr, $C:expr, $M:expr, $sparsity:expr) => {
    (tracing::info_span!($span_name), move || {
      const N: usize = $N;
      const C: usize = $C;
      const M: usize = $M;
      const S: usize = $sparsity;
      type F = $field;
      type G = $group;
      type SubtableStrategy = $subtable_strategy;

      let m_computed = N.nth_root(C as u32);
      assert_eq!(m_computed, M);
      let log_m = log2(m_computed) as usize;
      let log_s: usize = log2($sparsity) as usize;

      let r: Vec<F> = gen_random_point::<F>(log_s);

      let nz = gen_indices::<C>(S, M);
      let lookup_matrix = SparseLookupMatrix::new(nz.clone(), log_m);

      // Prove
      let mut dense: DensifiedRepresentation<F, C> = DensifiedRepresentation::from(&lookup_matrix);
      let gens = SparsePolyCommitmentGens::<G>::new(b"gens_sparse_poly", C, S, C, log_m);
      let _commitment = dense.commit::<$group>(&gens);
      let mut random_tape = RandomTape::new(b"proof");
      let mut prover_transcript = Transcript::new(b"example");
      let _proof = SparsePolynomialEvaluationProof::<G, C, M, SubtableStrategy>::prove(
        &mut dense,
        &r,
        &gens,
        &mut prover_transcript,
        &mut random_tape,
      );
    })
  };
}

pub fn benchmarks() -> Vec<(tracing::Span, fn())> {
  vec![
    single_pass_lasso!(
      "And(2^10)",
      Fr,
      EdwardsProjective,
      AndSubtableStrategy,
      /* N= */ 1 << 16,
      /* C= */ 1,
      /* M= */ 1 << 16,
      /* S= */ 1 << 10
    ),
    single_pass_lasso!(
      "And(2^12)",
      Fr,
      EdwardsProjective,
      AndSubtableStrategy,
      /* N= */ 1 << 16,
      /* C= */ 1,
      /* M= */ 1 << 16,
      /* S= */ 1 << 12
    ),
    single_pass_lasso!(
      "And(2^14)",
      Fr,
      EdwardsProjective,
      AndSubtableStrategy,
      /* N= */ 1 << 16,
      /* C= */ 1,
      /* M= */ 1 << 16,
      /* S= */ 1 << 14
    ),
    single_pass_lasso!(
      "And(2^16)",
      Fr,
      EdwardsProjective,
      AndSubtableStrategy,
      /* N= */ 1 << 16,
      /* C= */ 1,
      /* M= */ 1 << 16,
      /* S= */ 1 << 16
    ),
    single_pass_lasso!(
      "And(2^18)",
      Fr,
      EdwardsProjective,
      AndSubtableStrategy,
      /* N= */ 1 << 16,
      /* C= */ 1,
      /* M= */ 1 << 16,
      /* S= */ 1 << 18
    ),
    single_pass_lasso!(
      "And(2^20)",
      Fr,
      EdwardsProjective,
      AndSubtableStrategy,
      /* N= */ 1 << 16,
      /* C= */ 1,
      /* M= */ 1 << 16,
      /* S= */ 1 << 20
    ),
    single_pass_lasso!(
      "And(2^22)",
      Fr,
      EdwardsProjective,
      AndSubtableStrategy,
      /* N= */ 1 << 16,
      /* C= */ 1,
      /* M= */ 1 << 16,
      /* S= */ 1 << 22
    ),
    single_pass_lasso!(
      "And(2^24)",
      Fr,
      EdwardsProjective,
      AndSubtableStrategy,
      /* N= */ 1 << 16,
      /* C= */ 1,
      /* M= */ 1 << 16,
      /* S= */ 1 << 24
    ),
  ]
}
