use std::time::Instant;

use ark_std::{test_rng, log2, UniformRand};
use libspartan::{random::RandomTape, sparse_mlpoly::{sparse_mlpoly::{SparsePolynomialEvaluationProof, SparseLookupMatrix}, densified::DensifiedRepresentation}};
use merlin::Transcript;
use num_integer::Roots;
use rand_chacha::rand_core::RngCore;
use ark_bls12_381::{Fr, G1Projective};

fn main() {
    let mut rng = test_rng();

    const N: usize = 1 << 48;
    const C: usize = 4;
    let s = 1 << 18;
    let M = N.nth_root(C as u32);

    let log_M = log2(M) as usize;

    // generate sparse polynomial
    let mut nz: Vec<[usize; C]> = Vec::new();
    for _ in 0..s {
        let indices = [rng.next_u64() as usize % M; C];
        nz.push(indices);
    }

    println!("SparseLookupMatrix::new()");
    let lookup_matrix = SparseLookupMatrix::new(nz.clone(), log_M);

    println!("SparseLookupMatrix.to_densified()");
    let before_densification = Instant::now();
    let mut dense: DensifiedRepresentation<Fr, C> = lookup_matrix.to_densified();
    println!("Dense.commit()");
    let before_commitment = Instant::now();
    let (gens, commitment) = dense.commit::<G1Projective>();

    let before_randomness = Instant::now();
    let r: [Vec<Fr>; C] = std::array::from_fn(|_| {
        let mut r_i: Vec<Fr> = Vec::with_capacity(log_M);
        for _ in 0..log_M {
            r_i.push(Fr::rand(&mut rng));
        }
        r_i
    });
    let flat_r: Vec<Fr> = r.clone().into_iter().flatten().collect();

    println!("SparseLookupMatrix.evaluate_mle()");
    let before_mle_eval = Instant::now();
    let eval = lookup_matrix.evaluate_mle(&flat_r);

    let mut random_tape = RandomTape::new(b"proof");
    let mut prover_transcript = Transcript::new(b"example");
    println!("SparsePolynomialEvaluationProof.prove()");
    let before_prove = Instant::now();
    let proof = SparsePolynomialEvaluationProof::<G1Projective, C>::prove(
        &mut dense,
        &r,
        &eval,
        &gens,
        &mut prover_transcript,
        &mut random_tape);
    let after_prove = Instant::now();

    println!("Densification: {}ms", (before_commitment - before_densification).as_millis());
    println!("Commitment : {}ms", (before_randomness - before_commitment).as_millis());
    println!("Randomness : {}ms", (before_mle_eval - before_randomness).as_millis());
    println!("MLE Eval: {}ms", (before_prove - before_mle_eval).as_millis());
    println!("Prove: {}ms", (after_prove - before_prove).as_millis());
}
