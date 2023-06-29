use std::time::{Instant, Duration};

use ark_curve25519::{EdwardsProjective, Fr};
use ark_ff::PrimeField;
use ark_std::{log2, test_rng};
use crate::sparse_mlpoly::sparse_mlpoly::SparsePolyCommitmentGens;
use crate::sparse_mlpoly::subtables::and::AndSubtableStrategy;
use crate::sparse_mlpoly::subtables::spark::SparkSubtableStrategy;
use crate::{
  random::RandomTape,
  sparse_mlpoly::{
    densified::DensifiedRepresentation,
    sparse_mlpoly::{SparseLookupMatrix, SparsePolynomialEvaluationProof},
  },
};
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
  std::array::from_fn(|_| {
    gen_random_point(memory_bits)
  })
}

pub fn gen_random_point<F: PrimeField>(memory_bits: usize) -> Vec<F> {
  let mut rng = test_rng();
  let mut r_i: Vec<F> = Vec::with_capacity(memory_bits);
  for _ in 0..memory_bits {
    r_i.push(F::rand(&mut rng));
  }
  r_i
}

macro_rules! single_pass_surge {
    ($field:ty, $group:ty, $subtable_strategy:ty, $N:expr, $C:expr, $M:expr, $sparsity:expr, $field_name:expr) => {
      {
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
  
        let short_strat_name = std::any::type_name::<SubtableStrategy>().split("::").last().unwrap();
        println!("Running {}", format!("Surge(strat={}, N={}, C={}, S={}, F={})", short_strat_name, N, C, S, $field_name));
  
        let spark_randomness = gen_random_points::<F, C>(log_m);
        let eq_randomness: Vec<F> = gen_random_point::<F>(log_s);
  
        let nz = gen_indices::<C>(S, M);
        let lookup_matrix = SparseLookupMatrix::new(nz.clone(), log_m);
  
        // Densified creation
        println!("DensifiedRepresentation::from()");
        let before_densification = Instant::now();
        let _dense: DensifiedRepresentation<F, C> =
            DensifiedRepresentation::from(&lookup_matrix);
        let after_densification = Instant::now();
  
        // Densified commitment
        let dense: DensifiedRepresentation<F, C> = DensifiedRepresentation::from(&lookup_matrix);
        println!("DensifiedRepresentation::commit()");
        let before_dense_commitment = Instant::now();
        let gens = SparsePolyCommitmentGens::<G>::new(
            b"gens_sparse_poly",
            C,
            S,
            C,
            log_m,
            );
        let _commitment = dense.commit::<G>(&gens);
        let after_dense_commitment = Instant::now();
  
        // Prove
        let mut dense: DensifiedRepresentation<F, C> = DensifiedRepresentation::from(&lookup_matrix);
        let gens = SparsePolyCommitmentGens::<G>::new(
          b"gens_sparse_poly",
          C,
          S,
          C,
          log_m,
        );
        let _commitment = dense.commit::<$group>(&gens);
        println!("SparsePolynomialEvaluationProof::prove()");
        let before_proving = Instant::now();
        let mut random_tape = RandomTape::new(b"proof");
        let mut prover_transcript = Transcript::new(b"example");
        let _proof =
        SparsePolynomialEvaluationProof::<G, C, M, SubtableStrategy>::prove(
            &mut dense,
            &spark_randomness,
            &eq_randomness,
            &gens,
            &mut prover_transcript,
            &mut random_tape,
        );
        let after_proving = Instant::now();
        println!("");

        (after_proving - before_proving, after_dense_commitment - before_dense_commitment, after_densification - before_densification)
      }
    };
}

pub fn run() {
    let mut timings: Vec<(usize, (Duration, Duration, Duration))> = Vec::new();
    // timings.push((1 << 10, single_pass_surge!(Fr, EdwardsProjective, SparkSubtableStrategy, /* N= */ 1 << 16, /* C= */ 1, /* S= */ 1 << 10, "25519")));
    // timings.push((1 << 12, single_pass_surge!(Fr, EdwardsProjective, SparkSubtableStrategy, /* N= */ 1 << 16, /* C= */ 1, /* S= */ 1 << 12, "25519")));
    // timings.push((1 << 14, single_pass_surge!(Fr, EdwardsProjective, SparkSubtableStrategy, /* N= */ 1 << 16, /* C= */ 1, /* S= */ 1 << 14, "25519")));
    // timings.push((1 << 16, single_pass_surge!(Fr, EdwardsProjective, SparkSubtableStrategy, /* N= */ 1 << 16, /* C= */ 1, /* S= */ 1 << 16, "25519")));
    // timings.push((1 << 18, single_pass_surge!(Fr, EdwardsProjective, SparkSubtableStrategy, /* N= */ 1 << 16, /* C= */ 1, /* S= */ 1 << 18, "25519")));
    // timings.push((1 << 20, single_pass_surge!(Fr, EdwardsProjective, SparkSubtableStrategy, /* N= */ 1 << 16, /* C= */ 1, /* S= */ 1 << 20, "25519")));
    // timings.push((1 << 22, single_pass_surge!(Fr, EdwardsProjective, SparkSubtableStrategy, /* N= */ 1 << 16, /* C= */ 1, /* S= */ 1 << 22, "25519")));

    timings.push((1 << 10, single_pass_surge!(Fr, EdwardsProjective, AndSubtableStrategy, /* N= */ 1 << 16, /* C= */ 1, /* M= */ 1 << 16, /* S= */ 1 << 10, "25519")));
    timings.push((1 << 12, single_pass_surge!(Fr, EdwardsProjective, AndSubtableStrategy, /* N= */ 1 << 16, /* C= */ 1, /* M= */ 1 << 16, /* S= */ 1 << 12, "25519")));
    timings.push((1 << 14, single_pass_surge!(Fr, EdwardsProjective, AndSubtableStrategy, /* N= */ 1 << 16, /* C= */ 1, /* M= */ 1 << 16, /* S= */ 1 << 14, "25519")));
    timings.push((1 << 16, single_pass_surge!(Fr, EdwardsProjective, AndSubtableStrategy, /* N= */ 1 << 16, /* C= */ 1, /* M= */ 1 << 16, /* S= */ 1 << 16, "25519")));
    timings.push((1 << 18, single_pass_surge!(Fr, EdwardsProjective, AndSubtableStrategy, /* N= */ 1 << 16, /* C= */ 1, /* M= */ 1 << 16, /* S= */ 1 << 18, "25519")));
    timings.push((1 << 20, single_pass_surge!(Fr, EdwardsProjective, AndSubtableStrategy, /* N= */ 1 << 16, /* C= */ 1, /* M= */ 1 << 16, /* S= */ 1 << 20, "25519")));
    timings.push((1 << 22, single_pass_surge!(Fr, EdwardsProjective, AndSubtableStrategy, /* N= */ 1 << 16, /* C= */ 1, /* M= */ 1 << 16, /* S= */ 1 << 22, "25519")));

    println!("");
    for row in timings {
        let (s, (proving, commit, densify)) = row;
        println!("S={s} (2^{}), AndSubtableStrategy, Curve25519", log2(s));
        println!("- Densify:     {}ms", densify.as_millis());
        println!("- Dense Commit: {}ms", commit.as_millis());
        println!("- Prove:       {}ms", proving.as_millis());
        println!("- Total:       {}ms", densify.as_millis() + commit.as_millis() + proving.as_millis());
        println!("");
        println!("");
    }
}