#![allow(dead_code, unused_variables, unused_imports)]
//! Debug commitment hashing
//!
//! Compare what the real transcript hashes vs what we generate

use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use jolt_core::zkvm::RV64IMACProof;
use num_bigint::BigUint;

fn fr_to_string(f: &Fr) -> String {
    let mut bytes = vec![];
    f.serialize_uncompressed(&mut bytes).unwrap();
    BigUint::from_bytes_le(&bytes).to_string()
}

fn load_proof() -> Result<RV64IMACProof, Box<dyn std::error::Error>> {
    use ark_serialize::CanonicalDeserialize;
    let bytes = std::fs::read("/tmp/fib_proof.bin")?;
    Ok(CanonicalDeserialize::deserialize_compressed(&bytes[..])?)
}

fn main() {
    let proof = load_proof().expect("Failed to load proof");

    println!("=== Debug Commitment Hashing ===\n");

    // Get first commitment
    let commitment = &proof.commitments[0];

    // Serialize uncompressed (this is what append_serializable does)
    let mut bytes = Vec::new();
    commitment.serialize_uncompressed(&mut bytes).unwrap();
    println!("Original bytes (LE): {} bytes", bytes.len());
    println!("First 32 bytes: {:?}", &bytes[0..32]);

    // Reverse (for EVM compat)
    let mut reversed = bytes.clone();
    reversed.reverse();
    println!("\nReversed bytes (BE):");
    println!("First 32 bytes (chunk 0): {:?}", &reversed[0..32]);

    // Compute Fr values for each chunk (what transcript hashes)
    println!("\n=== Fr values for each chunk (what gets hashed) ===");
    for i in 0..12 {
        let start = i * 32;
        let end = start + 32;
        let chunk = &reversed[start..end];
        let fr = Fr::from_le_bytes_mod_order(chunk);
        println!("Chunk {}: {}", i, fr_to_string(&fr));
    }

    // Now compare with what generate_witness produces
    println!("\n=== What generate_witness should produce ===");
    // Read from JSON
    let json_path = "gnark-transpiler/data/fib_stage1_data.json";
    let json_content = std::fs::read_to_string(json_path).expect("Failed to read JSON");
    let extracted: serde_json::Value = serde_json::from_str(&json_content).unwrap();

    let commitment_bytes: Vec<u8> = extracted["commitments"][0]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u8)
        .collect();

    println!("JSON commitment[0] bytes: {} bytes", commitment_bytes.len());
    println!("First 32 bytes: {:?}", &commitment_bytes[0..32]);

    // Compute chunks from JSON data (this is what generate_witness does)
    let modulus = BigUint::parse_bytes(
        b"21888242871839275222246405745257275088548364400416034343698204186575808495617",
        10,
    ).unwrap();

    println!("\n=== Fr values from JSON bytes ===");
    for i in 0..12 {
        let start = i * 32;
        let end = start + 32;
        let chunk = &commitment_bytes[start..end];
        let value = BigUint::from_bytes_le(chunk) % &modulus;
        println!("Chunk {}: {}", i, value);
    }
}
