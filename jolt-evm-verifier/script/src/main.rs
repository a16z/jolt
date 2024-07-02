use jolt_core::{
    field::JoltField,
    poly::commitment::zeromorph::Zeromorph,
    subprotocols::grand_product::{BatchedDenseGrandProduct, BatchedGrandProduct},
    utils::transcript::ProofTranscript,
};
use std::env;

use alloy_primitives::{hex, U256};
use alloy_sol_types::{sol, SolType};
use ark_serialize::CanonicalSerialize;
use ark_bn254::{Bn254, Fr};
use ark_std::test_rng;

fn get_proof_data(batched_circuit: &mut BatchedDenseGrandProduct<Fr>) {
    let mut transcript: ProofTranscript = ProofTranscript::new(b"test_transcript");

    let (proof, r_prover) = <BatchedDenseGrandProduct<Fr> as BatchedGrandProduct<
        Fr,
        Zeromorph<Bn254>,
    >>::prove_grand_product(batched_circuit, &mut transcript, None);
    let claims = <BatchedDenseGrandProduct<Fr> as BatchedGrandProduct<
    Fr, Zeromorph<Bn254>>>::claims(batched_circuit);

    //encoding the proof into abi

    sol!(struct SolBatchedGrandProductLayerProof {
        uint256[][] compressed_polys_coeffs_except_linear_term;
        uint256[] left_claims;
        uint256[] right_claims;
    });

    sol!(struct SolBatchedGrandProductProof {
        SolBatchedGrandProductLayerProof[] layers;
    });

    sol!(struct SolProductProofAndClaims{
        SolBatchedGrandProductProof encoded_proof;
        uint256[] claims;
        uint256[] r_prover;
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
                        .map(|c| fr_to_uint256(&c))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),

            left_claims: l
                .left_claims
                .iter()
                .map(|c| fr_to_uint256(c))
                .collect::<Vec<_>>(),
            right_claims: l
                .right_claims
                .iter()
                .map(|c| fr_to_uint256(c))
                .collect::<Vec<_>>(),
        })
        .collect::<Vec<_>>();

    let encoded_proof = SolBatchedGrandProductProof::from(SolBatchedGrandProductProof { layers });

    let r_prover = r_prover
        .iter()
        .map(|c| fr_to_uint256(c))
        .collect::<Vec<_>>();

    let claims = claims
        .iter()
        .map(|c| fr_to_uint256(c))
        .collect::<Vec<_>>();

    let proof_plus_results = SolProductProofAndClaims{
        encoded_proof,
        claims,
        r_prover
    };

    print!(
        "{}",
        hex::encode(SolProductProofAndClaims::abi_encode(&proof_plus_results))
    );
}

fn fr_to_uint256(c: &Fr) -> U256 {
    let mut serialize = vec![];
    let _ = c.serialize_uncompressed(&mut serialize);
    U256::from_le_slice(&serialize)
} 

fn main() {
    let _: Vec<_> = env::args().collect();

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

    let mut batched_circuit = <BatchedDenseGrandProduct<Fr> as BatchedGrandProduct<
        Fr,
        Zeromorph<Bn254>,
    >>::construct(leaves);

    get_proof_data(&mut batched_circuit);
}
