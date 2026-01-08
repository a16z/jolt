#![allow(dead_code, unused_variables, unused_imports)]
//! Generate Go witness values for Stage1Circuit
//!
//! Usage: cargo run --bin generate_witness

use num_bigint::BigUint;
use serde::Deserialize;

#[derive(Deserialize)]
struct ExtractedPreamble {
    max_input_size: u64,
    max_output_size: u64,
    memory_size: u64,
    inputs: Vec<u8>,
    outputs: Vec<u8>,
    panic: bool,
    ram_k: u64,
    trace_length: u64,
}

#[derive(Deserialize)]
struct ExtractedStage1Data {
    preamble: ExtractedPreamble,
    uni_skip_poly_coeffs: Vec<String>,
    sumcheck_round_polys: Vec<Vec<String>>,
    commitments: Vec<Vec<u8>>,
    #[serde(default)]
    r1cs_input_evals: Vec<String>,
}

const CHUNKS_PER_COMMITMENT: usize = 12;

// BN254 scalar field modulus
fn bn254_modulus() -> BigUint {
    BigUint::parse_bytes(
        b"21888242871839275222246405745257275088548364400416034343698204186575808495617",
        10,
    ).unwrap()
}

fn bytes_chunk_le(bytes: &[u8], chunk_idx: usize) -> String {
    let start = chunk_idx * 32;
    let end = std::cmp::min(start + 32, bytes.len());
    if start >= bytes.len() {
        return "0".to_string();
    }

    let chunk = &bytes[start..end];
    let result = BigUint::from_bytes_le(chunk);
    // Reduce modulo BN254 scalar field, matching Fr::from_le_bytes_mod_order
    let reduced = result % bn254_modulus();
    reduced.to_string()
}

fn main() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let data_path = format!("{}/data/fib_stage1_data.json", manifest_dir);

    let json_content = std::fs::read_to_string(&data_path).expect("Failed to read data file");
    let extracted: ExtractedStage1Data =
        serde_json::from_str(&json_content).expect("Failed to parse JSON");

    println!("// Stage1Circuit witness values");
    println!("// Generated from fib_stage1_data.json");
    println!("// Note: Preamble values are now constants embedded in the circuit, not inputs");
    println!();
    println!("assignment := &Stage1Circuit{{}}");
    println!();

    // Commitments (41 × 12 chunks)
    for (c_idx, commitment) in extracted.commitments.iter().enumerate() {
        for chunk_idx in 0..CHUNKS_PER_COMMITMENT {
            let value = bytes_chunk_le(commitment, chunk_idx);
            println!("assignment.Commitment{}Chunk{}, _ = new(big.Int).SetString(\"{}\", 10)", c_idx, chunk_idx, value);
        }
    }

    // Uni-skip coefficients
    for (i, coeff) in extracted.uni_skip_poly_coeffs.iter().enumerate() {
        println!("assignment.UniSkipCoeff{}, _ = new(big.Int).SetString(\"{}\", 10)", i, coeff);
    }

    // Sumcheck round polys
    for (round, polys) in extracted.sumcheck_round_polys.iter().enumerate() {
        for (coeff, value) in polys.iter().enumerate() {
            println!("assignment.SumcheckR{}C{}, _ = new(big.Int).SetString(\"{}\", 10)", round, coeff, value);
        }
    }

    // R1CS input evaluations (36 elements)
    for (i, eval) in extracted.r1cs_input_evals.iter().enumerate() {
        println!("assignment.R1csInput{}, _ = new(big.Int).SetString(\"{}\", 10)", i, eval);
    }
}
