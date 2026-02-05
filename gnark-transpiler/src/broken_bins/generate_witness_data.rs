#![allow(dead_code, unused_variables, unused_imports)]
//! Generate witness data JSON for Go tests
//!
//! Outputs all input values (preamble, commitments, uni_skip coeffs, sumcheck polys)
//! that the Go circuit needs as witness.

use serde::{Deserialize, Serialize};

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
    r1cs_input_evals: Vec<String>,
}

#[derive(Serialize)]
struct WitnessData {
    preamble: PreambleData,
    commitments: Vec<Vec<String>>,  // 41 commitments, each with 12 chunks as strings
    uni_skip_coeffs: Vec<String>,
    sumcheck_polys: Vec<Vec<String>>,  // 11 rounds, each with 3 coefficients
    r1cs_input_evals: Vec<String>,     // 36 R1CS input evaluations
}

#[derive(Serialize)]
struct PreambleData {
    max_input_size: u64,
    max_output_size: u64,
    memory_size: u64,
    input_chunk0: String,
    output_chunk0: String,
    panic: u64,
    ram_k: u64,
    trace_length: u64,
}

use ark_bn254::Fr;
use ark_ff::PrimeField;

fn bytes_to_fr_chunks(bytes: &[u8]) -> Vec<Fr> {
    let num_chunks = (bytes.len() + 31) / 32;
    (0..num_chunks)
        .map(|i| {
            let start = i * 32;
            let end = std::cmp::min(start + 32, bytes.len());
            let chunk = &bytes[start..end];
            Fr::from_le_bytes_mod_order(chunk)
        })
        .collect()
}

fn main() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let data_path = format!("{}/data/fib_stage1_data.json", manifest_dir);

    let json_content = std::fs::read_to_string(&data_path).expect("Failed to read data file");
    let extracted: ExtractedStage1Data =
        serde_json::from_str(&json_content).expect("Failed to parse JSON");

    // Convert preamble inputs/outputs to field elements
    let input_chunks = bytes_to_fr_chunks(&extracted.preamble.inputs);
    let output_chunks = bytes_to_fr_chunks(&extracted.preamble.outputs);

    let preamble = PreambleData {
        max_input_size: extracted.preamble.max_input_size,
        max_output_size: extracted.preamble.max_output_size,
        memory_size: extracted.preamble.memory_size,
        input_chunk0: input_chunks.get(0).map(|f| f.to_string()).unwrap_or("0".to_string()),
        output_chunk0: output_chunks.get(0).map(|f| f.to_string()).unwrap_or("0".to_string()),
        panic: if extracted.preamble.panic { 1 } else { 0 },
        ram_k: extracted.preamble.ram_k,
        trace_length: extracted.preamble.trace_length,
    };

    // Convert commitments to chunks
    let commitments: Vec<Vec<String>> = extracted
        .commitments
        .iter()
        .map(|commitment| {
            let chunks = bytes_to_fr_chunks(commitment);
            chunks.iter().map(|f| f.to_string()).collect()
        })
        .collect();

    let witness = WitnessData {
        preamble,
        commitments,
        uni_skip_coeffs: extracted.uni_skip_poly_coeffs.clone(),
        sumcheck_polys: extracted.sumcheck_round_polys.clone(),
        r1cs_input_evals: extracted.r1cs_input_evals.clone(),
    };

    // Output JSON
    let json = serde_json::to_string_pretty(&witness).expect("Failed to serialize");
    println!("{}", json);

    // Also write to file
    let output_path = format!("{}/data/witness_data.json", manifest_dir);
    std::fs::write(&output_path, &json).expect("Failed to write output file");
    eprintln!("Written to: {}", output_path);
}
