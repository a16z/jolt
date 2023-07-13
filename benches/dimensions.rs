use ark_curve25519::{EdwardsProjective, Fr};
use ark_ff::PrimeField;
use ark_std::UniformRand;
use ark_std::{log2, test_rng};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use libspartan::sparse_mlpoly::sparse_mlpoly::SparsePolyCommitmentGens;
use libspartan::sparse_mlpoly::subtables::lt::LTSubtableStrategy;
use libspartan::sparse_mlpoly::subtables::and::AndSubtableStrategy;
use libspartan::{
  random::RandomTape,
  sparse_mlpoly::{
    densified::DensifiedRepresentation,
    sparse_mlpoly::{SparseLookupMatrix, SparsePolynomialEvaluationProof},
  },
};
use merlin::Transcript;
use num_integer::Roots;
use rand_chacha::rand_core::RngCore;

macro_rules! bench_surge {
  ($field:ty, $group:ty, $subtable_strategy:ty, $N:expr, $C:expr, $M:expr, $sparsity:expr, $criterion:expr, $field_name:expr) => {
    {
      const N: usize = $N;
      const C: usize = $C;
      const S: usize = $sparsity;
      const M: usize = $M;
      type F = $field;
      type G = $group;
      type SubtableStrategy = $subtable_strategy;

      let m = N.nth_root(C as u32);
      let log_m = log2(m) as usize;
      let log_s = log2(S) as usize;

      let short_strat_name = std::any::type_name::<SubtableStrategy>().split("::").last().unwrap();
      let mut group = $criterion.benchmark_group(format!("Surge(strat={}, N={}, C={}, S={}, F={})", short_strat_name, N, C, S, $field_name));
      group.sample_size(10);

      let r: Vec<F> = gen_random_point::<F>(log_s);

      let nz = gen_indices::<C>(S, m);
      let lookup_matrix = SparseLookupMatrix::new(nz.clone(), log_m);

      // Densified creation
      group.bench_function(
        "DensifiedRepresentation::from()",
        |bencher| {
          bencher.iter(|| {
            let _dense: DensifiedRepresentation<F, C> =
              DensifiedRepresentation::from(&lookup_matrix);
          })
        },
      );

      // Densified commitment
      let dense: DensifiedRepresentation<F, C> = DensifiedRepresentation::from(&lookup_matrix);
      group.bench_function(
        "DensifiedRepresentation::commit()",
        |bencher| {
          bencher.iter(|| {
            let gens = SparsePolyCommitmentGens::<G>::new(
              b"gens_sparse_poly",
              C,
              S,
              C,
              log_m,
            );
            let _commitment = dense.commit::<G>(&gens);
          })
        },
      );

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
      group.bench_function(
        "SparsePolynomialEvaluationProof::prove()",
        |bencher| {
          bencher.iter(|| {
            let mut random_tape = RandomTape::new(b"proof");
            let mut prover_transcript = Transcript::new(b"example");
            let _proof =
              SparsePolynomialEvaluationProof::<G, C, M, SubtableStrategy>::prove(
                &mut dense,
                &r,
                &gens,
                &mut prover_transcript,
                &mut random_tape,
              );
          })
        },
      );
    }
  };
}

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

fn bench(criterion: &mut Criterion) {
  bench_surge!(Fr, EdwardsProjective, LTSubtableStrategy, 1 << 32, 2, 1 << 16, criterion, "255199");
  bench_surge!(Fr, EdwardsProjective, AndSubtableStrategy, 1 << 32, 2, 1 << 16, criterion, "25519");
}



criterion_group!(benches, bench);
criterion_main!(benches);
