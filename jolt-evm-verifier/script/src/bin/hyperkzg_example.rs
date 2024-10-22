use alloy_primitives::{hex, U256};
use alloy_sol_types::{sol, SolType};

use ark_bn254::Bn254;
use ark_ec::pairing::Pairing;
use ark_ff::BigInteger;
use ark_ff::PrimeField;
use ark_std::UniformRand;
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

    let poly = DensePolynomial::new(
        (0..n)
            .map(|_| <Bn254 as Pairing>::ScalarField::rand(&mut rng))
            .collect::<Vec<_>>(),
    );
    let point = (0..ell)
        .map(|_| <Bn254 as Pairing>::ScalarField::rand(&mut rng))
        .collect::<Vec<_>>();
    let eval = poly.evaluate(&point);

    // make a commitment
    let c = HyperKZG::<_, KeccakTranscript>::commit(&pk, &poly).unwrap();

    // prove an evaluation
    let mut prover_transcript = KeccakTranscript::new(b"TestEval");
    let proof: HyperKZGProof<Bn254> =
        HyperKZG::open(&pk, &poly, &point, &eval, &mut prover_transcript).unwrap();

    let mut verifier_tr = KeccakTranscript::new(b"TestEval");
    assert!(HyperKZG::verify(&vk, &c, &point, &eval, &proof, &mut verifier_tr).is_ok());

    sol!(struct Example {
        VK vk;
        HyperKZGProofSol proof;
        uint256 commitment_x;
        uint256 commitment_y;
        uint256[] point;
        uint256 claim;
    });

    let vk_sol = (&vk).into();
    let proof_sol = (&proof).into();

    let x = U256::from_be_slice(c.0.x.into_bigint().to_bytes_be().as_ref());
    let y = U256::from_be_slice(c.0.y.into_bigint().to_bytes_be().as_ref());

    let point_encoded = point
        .iter()
        .map(|i| U256::from_be_slice(i.into_bigint().to_bytes_be().as_slice()))
        .collect();
    let eval_encoded = U256::from_be_slice(eval.into_bigint().to_bytes_be().as_slice());

    let example = Example {
        proof: proof_sol,
        vk: vk_sol,
        commitment_x: x,
        commitment_y: y,
        point: point_encoded,
        claim: eval_encoded,
    };

    print!("{}", hex::encode(Example::abi_encode(&example)));
}
