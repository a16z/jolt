use ark_curve25519::{Fr, EdwardsProjective as G1Projective};
use merlin::Transcript;

use crate::{
    math::Math, 
    sparse_mlpoly::{
        sparse_mlpoly::{
            SparseLookupMatrix, SparsePolynomialEvaluationProof, SparsePolyCommitmentGens
        }, 
        densified::DensifiedRepresentation, 
        subtables::{
            lt::LTSubtableStrategy, SubtableStrategy, and::AndSubtableStrategy, range_check::RangeCheckSubtableStrategy, spark::SparkSubtableStrategy
        }
    }, 
    random::RandomTape, 
};

macro_rules! e2e_test {
    ($test_name:ident, $Strategy:ty, $G:ty, $F:ty, $C:expr, $M:expr, $sparsity:expr) => {
        #[test]
        fn $test_name() {
            use crate::utils::test::{gen_indices, gen_random_point};

            const C: usize = $C;
            const M: usize = $M;

            // parameters
            const NUM_MEMORIES: usize = <$Strategy as SubtableStrategy<$F, C, M>>::NUM_MEMORIES;
            let log_M: usize = M.log_2();

            // generate sparse polynomial
            let nz: Vec<[usize; C]> = gen_indices($sparsity, M);

            let lookup_matrix = SparseLookupMatrix::new(nz, log_M);

            let mut dense: DensifiedRepresentation<$F, C> = DensifiedRepresentation::from(&lookup_matrix);
            let gens =
                SparsePolyCommitmentGens::<$G>::new(b"gens_sparse_poly", C, $sparsity, NUM_MEMORIES, log_M);
            let commitment = dense.commit::<$G>(&gens);

            let r: [Vec<$F>; C] = gen_random_point(log_M);

            let mut random_tape = RandomTape::new(b"proof");
            let mut prover_transcript = Transcript::new(b"example");
            let proof = 
                SparsePolynomialEvaluationProof::<$G, C, $M, $Strategy>::prove(
                    &mut dense,
                    &r,
                    &gens,
                    &mut prover_transcript,
                    &mut random_tape,
                );

            let mut verifier_transcript = Transcript::new(b"example");
            assert!(proof
                .verify(&commitment, &r, &gens, &mut verifier_transcript)
                .is_ok(),
                "Failed to verify proof."
            );
        }
    };
}

e2e_test!(prove_4d_lt, LTSubtableStrategy,  G1Projective, Fr, /* C= */ 4, /* M= */ 16, /* sparsity= */ 16);
e2e_test!(prove_4d_and, AndSubtableStrategy, G1Projective, Fr, /* C= */ 4, /* M= */ 16, /* sparsity= */ 16);
e2e_test!(prove_3d_range, RangeCheckSubtableStrategy::<40>, G1Projective, Fr, /* C= */ 3, /* M= */ 16, /* sparsity= */ 16);
e2e_test!(prove_4d_spark, SparkSubtableStrategy, G1Projective, Fr, /* C= */ 4, /* M= */ { 1 << 8 }, /* sparsity= */ 1 << 8);