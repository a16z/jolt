use ark_curve25519::{EdwardsProjective, Fr};
use ark_std::UniformRand;
use ark_std::{log2, test_rng};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use libspartan::sparse_mlpoly::sparse_mlpoly::SparsePolyCommitmentGens;
use libspartan::sparse_mlpoly::subtables::eq::EqSubtableStrategy;
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

const NS: [usize; 2] = [1 << 16, 1 << 24];

fn bench(c: &mut Criterion) {
  let mut group = c.benchmark_group("SparseLookupMatrix::prove()");
  group.sample_size(10);
  let mut rng = test_rng();

  for N in NS {
    seq_macro::seq!(C in 2..=4  { // Macros to unroll due to generic constant C
      let M = N.nth_root(C as u32);
      let log_M = log2(M) as usize;
      let s = 1 << 10; // TODO: Variable sparsity

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

              let mut dense: DensifiedRepresentation<Fr, C> = DensifiedRepresentation::from(&lookup_matrix);
              let gens = SparsePolyCommitmentGens::<EdwardsProjective>::new(b"gens_sparse_poly", C, s, C, log_M);
              let commitment = dense.commit::<EdwardsProjective>(&gens);

              let r: [Vec<Fr>; C] = std::array::from_fn(|_| {
              let mut r_i: Vec<Fr> = Vec::with_capacity(log_M);
              for _ in 0..log_M {
                  r_i.push(Fr::rand(&mut rng));
              }
              r_i
              });

              let mut random_tape = RandomTape::new(b"proof");
              let mut prover_transcript = Transcript::new(b"example");
              let proof = SparsePolynomialEvaluationProof::<EdwardsProjective, C, C>::prove::<EqSubtableStrategy>(
              &mut dense,
              &r,
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
