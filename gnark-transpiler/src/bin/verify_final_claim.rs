#![allow(dead_code, unused_variables, unused_imports)]
//! Verify the expected final_claim using the REAL PoseidonTranscript
use ark_bn254::Fr;
use ark_ff::PrimeField;
use jolt_core::transcripts::{FrParams, PoseidonTranscript, Transcript};
use jolt_core::zkvm::stage1_only_verifier::{
    verify_stage1_with_transcript, Stage1PreambleData, Stage1TranscriptVerificationData,
};
use serde::Deserialize;
use std::str::FromStr;

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
}

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

    // Build concrete Fr values
    let preamble = Stage1PreambleData {
        max_input_size: Fr::from(extracted.preamble.max_input_size),
        max_output_size: Fr::from(extracted.preamble.max_output_size),
        memory_size: Fr::from(extracted.preamble.memory_size),
        inputs: bytes_to_fr_chunks(&extracted.preamble.inputs),
        outputs: bytes_to_fr_chunks(&extracted.preamble.outputs),
        panic: Fr::from(if extracted.preamble.panic { 1u64 } else { 0u64 }),
        ram_k: Fr::from(extracted.preamble.ram_k),
        trace_length: Fr::from(extracted.preamble.trace_length),
    };

    let commitments: Vec<Vec<Fr>> = extracted
        .commitments
        .iter()
        .map(|c| bytes_to_fr_chunks(c))
        .collect();

    let uni_skip_poly_coeffs: Vec<Fr> = extracted
        .uni_skip_poly_coeffs
        .iter()
        .map(|s| Fr::from_str(s).expect("parse coeff"))
        .collect();

    let sumcheck_round_polys: Vec<Vec<Fr>> = extracted
        .sumcheck_round_polys
        .iter()
        .map(|round| round.iter().map(|s| Fr::from_str(s).expect("parse")).collect())
        .collect();

    let data = Stage1TranscriptVerificationData {
        preamble: Some(preamble),
        commitments,
        uni_skip_poly_coeffs,
        sumcheck_round_polys,
        num_rounds: extracted.sumcheck_round_polys.len(),
    };

    // Run with REAL PoseidonTranscript (not MleTranscript)
    let mut transcript: PoseidonTranscript<Fr, FrParams> = Transcript::new(b"Jolt");
    let result = verify_stage1_with_transcript(data, &mut transcript);

    println!("=== Derived Challenges ===");
    println!("r0 = {}", result.derived_r0);
    for (i, tau) in result.derived_tau.iter().enumerate() {
        println!("tau[{}] = {}", i, tau);
    }
    for (i, r) in result.derived_sumcheck_challenges.iter().enumerate() {
        println!("sumcheck_challenge[{}] = {}", i, r);
    }

    println!();
    println!("=== Verification with REAL PoseidonTranscript ===");
    println!("power_sum_check = {}", result.power_sum_check);
    println!("final_claim = {}", result.final_claim);

    for (i, check) in result.sumcheck_consistency_checks.iter().enumerate() {
        println!("sumcheck_consistency_{} = {}", i, check);
    }
}
