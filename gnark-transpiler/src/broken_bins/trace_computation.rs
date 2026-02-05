#![allow(dead_code, unused_variables, unused_imports)]
//! Trace the exact computation of final_claim step by step
//!
//! This shows every intermediate value so we can verify the Go circuit does the same.

use ark_bn254::Fr;
use ark_ff::PrimeField;
use jolt_core::transcripts::{FrParams, PoseidonTranscript, Transcript};
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

    println!("=== STEP-BY-STEP COMPUTATION TRACE ===\n");

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

    println!("Number of uni_skip coefficients: {}", uni_skip_poly_coeffs.len());
    println!("Number of sumcheck rounds: {}", sumcheck_round_polys.len());
    println!();

    // === TRANSCRIPT OPERATIONS ===
    println!("=== TRANSCRIPT (Fiat-Shamir) ===\n");

    let mut transcript: PoseidonTranscript<Fr, FrParams> = Transcript::new(b"Jolt");
    println!("1. Initialized transcript with label 'Jolt'");

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
    println!("2. Appended preamble (8 scalars + input/output chunks)");

    // Append commitments
    let num_commitments = extracted.commitments.len();
    let mut total_commitment_chunks = 0;
    for commitment in &extracted.commitments {
        let chunks = bytes_to_fr_chunks(commitment);
        total_commitment_chunks += chunks.len();
        for chunk in chunks {
            transcript.append_scalar(&chunk);
        }
    }
    println!("3. Appended {} commitments ({} total chunks)", num_commitments, total_commitment_chunks);

    // Derive tau (11 challenges, but we don't use them in final_claim)
    let num_rounds = sumcheck_round_polys.len();
    for _ in 0..num_rounds {
        let _tau: Fr = transcript.challenge_scalar();
    }
    println!("4. Derived {} tau challenges (not used in final_claim computation)", num_rounds);

    // Append uni_skip poly coefficients
    for coeff in &uni_skip_poly_coeffs {
        transcript.append_scalar(coeff);
    }
    println!("5. Appended {} uni_skip polynomial coefficients", uni_skip_poly_coeffs.len());

    // Derive r0
    let r0: Fr = transcript.challenge_scalar();
    println!("6. Derived r0 = {}", r0);
    println!();

    // === CLAIM COMPUTATION ===
    println!("=== CLAIM COMPUTATION ===\n");

    // Step 1: Evaluate uni_skip polynomial at r0
    let claim_after_uni_skip = evaluate_polynomial(&uni_skip_poly_coeffs, &r0);
    println!("Step 1: claim_after_uni_skip = uni_skip_poly(r0)");
    println!("        = sum(coeff[i] * r0^i for i in 0..28)");
    println!("        = {}", claim_after_uni_skip);
    println!();

    // Step 2-12: For each sumcheck round
    let mut claim = claim_after_uni_skip;

    for round in 0..num_rounds {
        // Append round poly coefficients to transcript
        for coeff in &sumcheck_round_polys[round] {
            transcript.append_scalar(coeff);
        }

        // Derive challenge for this round
        let challenge: Fr = transcript.challenge_scalar();

        // Get compressed coefficients [c0, c2, c3]
        let compressed = &sumcheck_round_polys[round];
        let c0 = compressed[0];

        // Decompress: c1 = claim - 2*c0 - c2 - c3
        let mut c1 = claim - c0 - c0;  // claim - 2*c0
        for coeff in compressed.iter().skip(1) {
            c1 -= *coeff;
        }

        // Build full polynomial [c0, c1, c2, c3]
        let mut full_coeffs = vec![c0, c1];
        full_coeffs.extend(compressed.iter().skip(1).cloned());

        // Evaluate at challenge
        let next_claim = evaluate_polynomial(&full_coeffs, &challenge);

        println!("Round {}: challenge = {}", round, challenge);
        println!("         compressed = [{}, {}, {}]", compressed[0], compressed[1], compressed[2]);
        println!("         c1 (derived) = {}", c1);
        println!("         claim_before = {}", claim);
        println!("         claim_after = poly(challenge) = {}", next_claim);
        println!();

        claim = next_claim;
    }

    println!("=== FINAL RESULT ===");
    println!("final_claim = {}", claim);
    println!();

    // Verify this matches what we expect
    let expected = Fr::from_str("18842501486210351966301476350246988864959268947339603959449950557432835978270").unwrap();
    if claim == expected {
        println!("✓ MATCHES expected value!");
    } else {
        println!("✗ DOES NOT MATCH expected value!");
        println!("  Expected: {}", expected);
    }
}
