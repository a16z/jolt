#![allow(dead_code, unused_variables, unused_imports)]
//! Generate expected intermediate values for Go tests
//!
//! Outputs JSON with all intermediate values from the Rust computation.

use ark_bn254::Fr;
use ark_ff::PrimeField;
use jolt_core::transcripts::{FrParams, PoseidonTranscript, Transcript};
use serde::{Deserialize, Serialize};
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

#[derive(Serialize)]
struct ExpectedValues {
    r0: String,
    claim_after_uni_skip: String,
    challenges: Vec<String>,
    claims_after_round: Vec<String>,
    final_claim: String,
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

fn evaluate_polynomial(coeffs: &[Fr], x: &Fr) -> Fr {
    if coeffs.is_empty() {
        return Fr::from(0u64);
    }
    let mut result = coeffs[0];
    let mut x_power = *x;
    for coeff in coeffs.iter().skip(1) {
        result += *coeff * x_power;
        x_power *= *x;
    }
    result
}

fn main() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let data_path = format!("{}/data/fib_stage1_data.json", manifest_dir);

    let json_content = std::fs::read_to_string(&data_path).expect("Failed to read data file");
    let extracted: ExtractedStage1Data =
        serde_json::from_str(&json_content).expect("Failed to parse JSON");

    // Parse inputs
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

    // === TRANSCRIPT OPERATIONS ===
    let mut transcript: PoseidonTranscript<Fr, FrParams> = Transcript::new(b"Jolt");

    // Append preamble
    let preamble_scalars = vec![
        Fr::from(extracted.preamble.max_input_size),
        Fr::from(extracted.preamble.max_output_size),
        Fr::from(extracted.preamble.memory_size),
    ];
    for s in &preamble_scalars {
        transcript.append_scalar(s);
    }
    for chunk in bytes_to_fr_chunks(&extracted.preamble.inputs) {
        transcript.append_scalar(&chunk);
    }
    for chunk in bytes_to_fr_chunks(&extracted.preamble.outputs) {
        transcript.append_scalar(&chunk);
    }
    transcript.append_scalar(&Fr::from(if extracted.preamble.panic { 1u64 } else { 0u64 }));
    transcript.append_scalar(&Fr::from(extracted.preamble.ram_k));
    transcript.append_scalar(&Fr::from(extracted.preamble.trace_length));

    // Append commitments
    for commitment in &extracted.commitments {
        let chunks = bytes_to_fr_chunks(commitment);
        for chunk in chunks {
            transcript.append_scalar(&chunk);
        }
    }

    // Derive tau (11 challenges)
    let num_rounds = sumcheck_round_polys.len();
    for _ in 0..num_rounds {
        let _tau: Fr = transcript.challenge_scalar();
    }

    // Append uni_skip poly coefficients
    for coeff in &uni_skip_poly_coeffs {
        transcript.append_scalar(coeff);
    }

    // Derive r0
    let r0: Fr = transcript.challenge_scalar();

    // Evaluate uni_skip polynomial at r0
    let claim_after_uni_skip = evaluate_polynomial(&uni_skip_poly_coeffs, &r0);

    // Process sumcheck rounds
    let mut claim = claim_after_uni_skip;
    let mut challenges = Vec::new();
    let mut claims_after_round = Vec::new();

    for round in 0..num_rounds {
        // Append round poly coefficients to transcript
        for coeff in &sumcheck_round_polys[round] {
            transcript.append_scalar(coeff);
        }

        // Derive challenge for this round
        let challenge: Fr = transcript.challenge_scalar();
        challenges.push(challenge.to_string());

        // Get compressed coefficients [c0, c2, c3]
        let compressed = &sumcheck_round_polys[round];
        let c0 = compressed[0];

        // Decompress: c1 = claim - 2*c0 - c2 - c3
        let mut c1 = claim - c0 - c0;
        for coeff in compressed.iter().skip(1) {
            c1 -= *coeff;
        }

        // Build full polynomial [c0, c1, c2, c3]
        let mut full_coeffs = vec![c0, c1];
        full_coeffs.extend(compressed.iter().skip(1).cloned());

        // Evaluate at challenge
        let next_claim = evaluate_polynomial(&full_coeffs, &challenge);
        claims_after_round.push(next_claim.to_string());

        claim = next_claim;
    }

    // Build output
    let expected = ExpectedValues {
        r0: r0.to_string(),
        claim_after_uni_skip: claim_after_uni_skip.to_string(),
        challenges,
        claims_after_round,
        final_claim: claim.to_string(),
    };

    // Output JSON
    let json = serde_json::to_string_pretty(&expected).expect("Failed to serialize");
    println!("{}", json);

    // Also write to file
    let output_path = format!("{}/data/expected_values.json", manifest_dir);
    std::fs::write(&output_path, &json).expect("Failed to write output file");
    eprintln!("Written to: {}", output_path);
}
