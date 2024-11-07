use jolt_core::{
    field::JoltField,
    poly::commitment::hyperkzg::HyperKZG,
    subprotocols::grand_product::{BatchedDenseGrandProduct, BatchedGrandProduct},
    utils::sol_types::GrandProductProof,
};
use std::env;

use alloy_primitives::{hex, U256};
use alloy_sol_types::{sol, SolType};
use ark_bn254::{Bn254, Fr};
use ark_serialize::CanonicalSerialize;
use ark_std::test_rng;
use jolt_core::utils::transcript::{KeccakTranscript, Transcript};

fn get_proof_data(batched_circuit: &mut BatchedDenseGrandProduct<Fr>) {
    let mut transcript: KeccakTranscript = KeccakTranscript::new(b"test_transcript");

    let (proof, r_prover) =
        <BatchedDenseGrandProduct<Fr> as BatchedGrandProduct<
            Fr,
            HyperKZG<Bn254, KeccakTranscript>,
            KeccakTranscript,
        >>::prove_grand_product(batched_circuit, None, &mut transcript, None);
    let claims = <BatchedDenseGrandProduct<Fr> as BatchedGrandProduct<
        Fr,
        HyperKZG<Bn254, KeccakTranscript>,
        KeccakTranscript,
    >>::claimed_outputs(batched_circuit);

    // encoding the proof into abi

    sol!(struct SolProductProofAndClaims{
        GrandProductProof encoded_proof;
        uint256[] claims;
        uint256[] r_prover;
    });

    let r_prover = r_prover.iter().map(fr_to_uint256).collect::<Vec<_>>();

    let claims = claims.iter().map(fr_to_uint256).collect::<Vec<_>>();

    let proof_plus_results = SolProductProofAndClaims {
        encoded_proof: proof.into(),
        claims,
        r_prover,
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

    //initial test taken from https://github.com/a16z/jolt/blob/d5147f8d27bb4961f3d648b872b45ff99af860c0/jolt-core/src/subprotocols/grand_product.rs
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
        HyperKZG<Bn254, KeccakTranscript>,
        KeccakTranscript,
    >>::construct((leaves.concat(), BATCH_SIZE));

    get_proof_data(&mut batched_circuit);
}
