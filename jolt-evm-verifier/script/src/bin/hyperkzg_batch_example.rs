use alloy_primitives::{hex, U256};
use alloy_sol_types::{sol, SolType};

use ark_bn254::Bn254;
use ark_ec::pairing::Pairing;
use ark_ff::BigInteger;
use ark_ff::PrimeField;
use ark_std::UniformRand;
use jolt_core::poly::commitment::commitment_scheme::{BatchType, CommitmentScheme};
use jolt_core::poly::commitment::hyperkzg::*;
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::utils::transcript::{KeccakTranscript, Transcript};
use rand_core::SeedableRng;

use jolt_core::utils::sol_types::{HyperKZGProofSol, VK};

fn main() {
    // Testing 2^12 ie 4096 elements
    // We replicate the behavior of the standard rust tests, but output
    // the proof and verification key to ensure it is verified in sol as well.

    let ell = 12;
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(ell as u64);

    let n = 1 << ell; // n = 2^ell

    let srs = HyperKZGSRS::setup(&mut rng, n);
    let (pk, vk): (HyperKZGProverKey<Bn254>, HyperKZGVerifierKey<Bn254>) = srs.trim(n);

    let point = (0..ell)
        .map(|_| <Bn254 as Pairing>::ScalarField::rand(&mut rng))
        .collect::<Vec<_>>();

    let mut polys = vec![];
    let mut evals = vec![];
    let mut commitments = vec![];
    let mut borrowed = vec![];
    for _ in 0..8 {
        let poly = DensePolynomial::new(
            (0..n)
                .map(|_| <Bn254 as Pairing>::ScalarField::rand(&mut rng))
                .collect::<Vec<_>>(),
        );
        let eval = poly.evaluate(&point);
        commitments.push(HyperKZG::<_, KeccakTranscript>::commit(&pk, &poly).unwrap());
        polys.push(poly);
        evals.push(eval);
    }

    for poly in polys.iter() {
        borrowed.push(poly);
    }

    // prove an evaluation
    let mut prover_transcript = KeccakTranscript::new(b"TestEval");
    let proof: HyperKZGProof<Bn254> = HyperKZG::batch_prove(
        &(pk, vk),
        borrowed.as_slice(),
        &point,
        &evals,
        BatchType::Big,
        &mut prover_transcript,
    );

    sol!(struct BatchedExample {
        VK vk;
        HyperKZGProofSol proof;
        uint256[] commitments;
        uint256[] point;
        uint256[] claims;
    });

    let vk_sol = (&vk).into();
    let proof_sol = (&proof).into();

    let mut encoded_commitments = vec![];
    for point in commitments.iter() {
        let x = U256::from_be_slice(&point.0.x.into_bigint().to_bytes_be());
        let y = U256::from_be_slice(&point.0.y.into_bigint().to_bytes_be());
        encoded_commitments.push(x);
        encoded_commitments.push(y);
    }

    let point_encoded = point
        .iter()
        .map(|i| U256::from_be_slice(i.into_bigint().to_bytes_be().as_slice()))
        .collect();
    let mut evals_encoded = vec![];
    for eval in evals.iter() {
        evals_encoded.push(U256::from_be_slice(&eval.into_bigint().to_bytes_be()));
    }

    let example = BatchedExample {
        proof: proof_sol,
        vk: vk_sol,
        commitments: encoded_commitments,
        point: point_encoded,
        claims: evals_encoded,
    };

    print!("{}", hex::encode(BatchedExample::abi_encode(&example)));
}
