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
    // Initialize a transcript for the proof generation process
    let mut transcript: KeccakTranscript = KeccakTranscript::new(b"test_transcript");

    // Generate the proof and prover randomness
    let (proof, r_prover) =
        <BatchedDenseGrandProduct<Fr> as BatchedGrandProduct<
            Fr,
            HyperKZG<Bn254, KeccakTranscript>,
            KeccakTranscript,
        >>::prove_grand_product(batched_circuit, None, &mut transcript, None);

    // Retrieve the claimed outputs from the circuit
    let claims = <BatchedDenseGrandProduct<Fr> as BatchedGrandProduct<
        Fr,
        HyperKZG<Bn254, KeccakTranscript>,
        KeccakTranscript,
    >>::claimed_outputs(batched_circuit);

    // Serialize the proof into bytes
    let mut proof_bytes = vec![];
    proof
        .serialize_uncompressed(&mut proof_bytes)
        .expect("Failed to serialize proof");

    // Define a Solidity-compatible struct for ABI encoding
    sol!(struct SolProductProofAndClaims {
        bytes encoded_proof;  // Use `bytes` for the serialized proof
        uint256[] claims;     // Array of claims as uint256
        uint256[] r_prover;   // Array of prover randomness as uint256
    });

    // Convert the prover randomness and claims to U256
    let r_prover = r_prover.iter().map(fr_to_uint256).collect::<Vec<_>>();
    let claims = claims.iter().map(fr_to_uint256).collect::<Vec<_>>();

    // Create the struct instance for ABI encoding
    let proof_plus_results = SolProductProofAndClaims {
        encoded_proof: proof_bytes,  // Use the serialized proof bytes
        claims,
        r_prover,
    };

    // Encode the struct into ABI and print it as a hex string
    print!(
        "{}",
        hex::encode(SolProductProofAndClaims::abi_encode(&proof_plus_results))
    );
}

// Helper function to convert a field element (`Fr`) to `U256`
fn fr_to_uint256(c: &Fr) -> U256 {
    let mut serialize = vec![];
    c.serialize_uncompressed(&mut serialize)
        .expect("Serialization failed");
    U256::from_le_slice(&serialize)
}

fn main() {
    // Collect command-line arguments (unused in this example)
    let _: Vec<_> = env::args().collect();

    // Define constants for the circuit
    const LAYER_SIZE: usize = 1 << 8;  // Layer size = 256
    const BATCH_SIZE: usize = 4;       // Batch size = 4

    // Initialize a random number generator
    let mut rng = test_rng();

    // Generate random leaves for the circuit
    let leaves: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
        std::iter::repeat_with(|| Fr::random(&mut rng))
            .take(LAYER_SIZE)
            .collect()
    })
    .take(BATCH_SIZE)
    .collect();

    // Construct the batched grand product circuit
    let mut batched_circuit = <BatchedDenseGrandProduct<Fr> as BatchedGrandProduct<
        Fr,
        HyperKZG<Bn254, KeccakTranscript>,
        KeccakTranscript,
    >>::construct((leaves.concat(), BATCH_SIZE));

    // Generate and print the proof data
    get_proof_data(&mut batched_circuit);
}
