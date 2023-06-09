use ark_curve25519::{Fr, EdwardsProjective};
use ark_std::UniformRand;
use ark_std::{log2, test_rng};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
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

const NS: [usize; 2] = [1 << 32, 1 << 48];

fn bench(c: &mut Criterion) {
  let mut group = c.benchmark_group("SparseLookupMatrix::prove()");
  group.sample_size(10);
  let mut rng = test_rng();

  for N in NS {
    seq_macro::seq!(C in 2..=4  { // Macros to unroll due to generic constant C
      let M = N.nth_root(C as u32);
      let log_M = log2(M) as usize;
      let s = 1 << 12; // TODO: Variable sparsity

      // generate sparse polynomial
      let mut nz: Vec<[usize; C]> = Vec::new();
      for _ in 0..s {
          let indices = [rng.next_u64() as usize % M; C];
          nz.push(indices);
      }

      group.bench_with_input(
      BenchmarkId::new("SparseLookupMatrix", format!("(N, C): ({}, {})", N, C)),
      &N,
      |bencher, n| {
          bencher.iter(|| {
              let lookup_matrix = SparseLookupMatrix::new(nz.clone(), log_M);

              let mut dense: DensifiedRepresentation<Fr, C> = lookup_matrix.to_densified();
              let (gens, commitment) = dense.commit::<EdwardsProjective>();

              let r: [Vec<Fr>; C] = std::array::from_fn(|_| {
              let mut r_i: Vec<Fr> = Vec::with_capacity(log_M);
              for _ in 0..log_M {
                  r_i.push(Fr::rand(&mut rng));
              }
              r_i
              });
              let flat_r: Vec<Fr> = r.clone().into_iter().flatten().collect();
              let eval = lookup_matrix.evaluate_mle(&flat_r);

              let mut random_tape = RandomTape::new(b"proof");
              let mut prover_transcript = Transcript::new(b"example");
              let proof = SparsePolynomialEvaluationProof::<EdwardsProjective, C>::prove(
              &mut dense,
              &r,
              &eval,
              &gens,
              &mut prover_transcript,
              &mut random_tape,
              );
          })
      },
      );

    });
  }

  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
