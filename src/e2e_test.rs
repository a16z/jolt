use ark_curve25519::{EdwardsProjective as G1Projective, Fr};
use ark_ff::{BigInteger, BigInteger256};
use ark_std::log2;
use itertools::Itertools;
use merlin::Transcript;

use crate::{
  lasso::{
    densified::DensifiedRepresentation,
    surge::{SparsePolyCommitmentGens, SparsePolynomialEvaluationProof},
  },
  subtables::{
    and::AndSubtableStrategy, lt::LTSubtableStrategy, range_check::RangeCheckSubtableStrategy,
    SubtableStrategy, or::OrSubtableStrategy,
  },
  utils::random::RandomTape,
  utils::{math::Math, test::gen_random_point},
};

macro_rules! e2e_test {
  ($test_name:ident, $Strategy:ty, $G:ty, $F:ty, $C:expr, $M:expr, $sparsity:expr) => {
    #[test]
    fn $test_name() {
      use crate::utils::test::{gen_indices, gen_random_point};
      use ark_std::log2;

      const C: usize = $C;
      const M: usize = $M;

      // parameters
      const NUM_MEMORIES: usize = <$Strategy as SubtableStrategy<$F, C, M>>::NUM_MEMORIES;
      let log_M: usize = M.log_2();
      let log_s: usize = log2($sparsity) as usize;

      // generate sparse polynomial
      let nz: Vec<[usize; C]> = gen_indices($sparsity, M);

      let mut dense: DensifiedRepresentation<$F, C> =
        DensifiedRepresentation::from_lookup_indices(&nz, log_M);
      let gens =
        SparsePolyCommitmentGens::<$G>::new(b"gens_sparse_poly", C, $sparsity, NUM_MEMORIES, log_M);
      let commitment = dense.commit::<$G>(&gens);

      let r: Vec<$F> = gen_random_point(log_s);

      let mut random_tape = RandomTape::new(b"proof");
      let mut prover_transcript = Transcript::new(b"example");
      let proof = SparsePolynomialEvaluationProof::<$G, C, $M, $Strategy>::prove(
        &mut dense,
        &r,
        &gens,
        &mut prover_transcript,
        &mut random_tape,
      );

      let mut verifier_transcript = Transcript::new(b"example");
      assert!(
        proof
          .verify(&commitment, &r, &gens, &mut verifier_transcript)
          .is_ok(),
        "Failed to verify proof."
      );
    }
  };
}

e2e_test!(
  prove_4d_lt,
  LTSubtableStrategy,
  G1Projective,
  Fr,
  /* C= */ 4,
  /* M= */ 16,
  /* sparsity= */ 16
);
e2e_test!(
  prove_4d_lt_big_s,
  LTSubtableStrategy,
  G1Projective,
  Fr,
  /* C= */ 4,
  /* M= */ 16,
  /* sparsity= */ 128
);
e2e_test!(
  prove_4d_and,
  AndSubtableStrategy,
  G1Projective,
  Fr,
  /* C= */ 4,
  /* M= */ 16,
  /* sparsity= */ 16
);
e2e_test!(
  prove_3d_range,
  RangeCheckSubtableStrategy::<40>,
  G1Projective,
  Fr,
  /* C= */ 3,
  /* M= */ 256,
  /* sparsity= */ 16
);


#[test]
fn and_e2e() {
  const C: usize = 2;
  const M: usize = 16;
  const S: usize = 2;
  type F = Fr;
  type G = G1Projective;
  type SubtableStrategy = AndSubtableStrategy;

  let log_m = log2(M) as usize;

  let r: Vec<F> = vec![F::from(0)];

  // Elements are of the form [ x2 || y2, x1 || y1 ]
  // So nz[0] is performing a lookup of AND(x, y) = AND(1110, 0011) = 0010
  // nz[1] is unused; we just need s ≥ 2
  let nz = vec![[0b1011, 0b1100], [0b0000, 0b0000]];

  // Prove
  let mut dense: DensifiedRepresentation<F, C> =
    DensifiedRepresentation::from_lookup_indices(&nz, log_m);
  let gens = SparsePolyCommitmentGens::<G>::new(b"gens_sparse_poly", C, S, C, log_m);
  let commitment = dense.commit::<G>(&gens);
  let mut random_tape = RandomTape::new(b"proof");
  let mut prover_transcript = Transcript::new(b"example");
  let proof = SparsePolynomialEvaluationProof::<G, C, M, SubtableStrategy>::prove(
    &mut dense,
    &r,
    &gens,
    &mut prover_transcript,
    &mut random_tape,
  );

  // Need to make relevant struct fields `pub` for this to work
  let eval: BigInteger256 = proof.primary_sumcheck.claimed_evaluation.into();

  // Should print 0010
  println!(
    "{}",
    eval.to_bits_be().iter().map(|&b| if b { "1" } else { "0" }).collect::<String>()
  );

  let mut verify_transcript = Transcript::new(b"example");
  proof
    .verify(&commitment, &r, &gens, &mut verify_transcript)
    .expect("should verify");
}
