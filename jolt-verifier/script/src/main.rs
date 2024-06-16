use jolt_core::{
    field::JoltField,
    subprotocols::grand_product::{BatchedDenseGrandProduct, BatchedGrandProduct},
    utils::transcript::ProofTranscript,
};
use std::env;

use ark_ff::{BigInteger, PrimeField};

use alloy_primitives::{hex, U256};
use alloy_sol_types::{sol, SolType};
use ark_bn254::Fr;
use ark_std::test_rng;

fn get_proof_data<F: JoltField + PrimeField>(batched_circuit: &mut BatchedDenseGrandProduct<F>) {
    let mut transcript: ProofTranscript = ProofTranscript::new(b"test_transcript");

    let (proof, _r_prover) = batched_circuit.prove_grand_product(&mut transcript);

    //encoding the proof into abi

    sol!(struct SolBatchedGrandProductLayerProof {
        uint256[][] compressed_polys_coeffs_except_linear_term;
        uint256[] left_claims;
        uint256[] right_claims;
    });

    sol!(struct SolBatchedGrandProductProof {
        SolBatchedGrandProductLayerProof[] layers;
    });

    let layers: Vec<SolBatchedGrandProductLayerProof> = proof
        .layers
        .into_iter()
        .map(|l| SolBatchedGrandProductLayerProof {
            compressed_polys_coeffs_except_linear_term: l
                .proof
                .compressed_polys
                .iter()
                .map(|p| {
                    p.coeffs_except_linear_term
                        .clone()
                        .into_iter()
                        .map(|c| U256::from_le_slice(c.into_bigint().to_bytes_le().as_slice()))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),

            left_claims: l
                .left_claims
                .iter()
                .map(|c| U256::from_le_slice(c.into_bigint().to_bytes_le().as_slice()))
                .collect::<Vec<_>>(),
            right_claims: l
                .right_claims
                .iter()
                .map(|c| U256::from_le_slice(c.into_bigint().to_bytes_le().as_slice()))
                .collect::<Vec<_>>(),
        })
        .collect::<Vec<_>>();

    let encoded_proof = SolBatchedGrandProductProof::from(SolBatchedGrandProductProof { layers });

    print!(
        "{}",
        hex::encode(SolBatchedGrandProductProof::abi_encode(&encoded_proof))
    );
}

fn get_claims_data<F: JoltField + PrimeField>(batched_circuit: &BatchedDenseGrandProduct<F>) {
    let claims = batched_circuit
        .claims()
        .iter()
        .map(|c| U256::from_le_slice(c.into_bigint().to_bytes_le().as_slice()))
        .collect::<Vec<_>>();
    type U256Array = sol! { uint256[] };
    let encoded_claims = U256Array::abi_encode(&claims);
    print!("{}", hex::encode(encoded_claims));
}

fn main() {
    let args: Vec<_> = env::args().collect();

    //initial test taken from https://github.com/a16z/jolt/blob/main/jolt-core/src/subprotocols/grand_product.rs#L1522-L1545
    const LAYER_SIZE: usize = 1 << 8;
    const BATCH_SIZE: usize = 4;
    let mut rng = test_rng();
    let leaves: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
        std::iter::repeat_with(|| Fr::random(&mut rng))
            .take(LAYER_SIZE)
            .collect()
    })
    .take(BATCH_SIZE)
    .collect();

    let mut batched_circuit = BatchedDenseGrandProduct::construct(leaves);

    match args[1].as_str() {
        "proofs" => get_proof_data(&mut batched_circuit),
        "claims" => get_claims_data(&batched_circuit),
        _ => println!("invalid arguement"),
    };
}
