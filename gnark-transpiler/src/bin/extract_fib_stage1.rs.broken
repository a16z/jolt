//! Extract Stage 1 inputs from Fibonacci proof
//!
//! This binary runs the Fibonacci prover to generate a real proof,
//! then extracts the Stage 1 verification inputs needed for transpilation.
//! It also computes the expected final_claim by running real verification.

use ark_bn254::Fr;
use ark_serialize::CanonicalSerialize;
use common::jolt_device::JoltDevice;
use jolt_core::poly::opening_proof::{OpeningId, SumcheckId};
use jolt_core::transcripts::{FrParams, PoseidonTranscript, Transcript};
use jolt_core::zkvm::r1cs::inputs::ALL_R1CS_INPUTS;
use jolt_core::zkvm::r1cs::inputs::NUM_R1CS_INPUTS;
use jolt_core::zkvm::stepwise_verifier::{
    verify_stage1_full, PreambleData as StepwisePreambleData, Stage1FullVerificationData,
};
use jolt_core::zkvm::witness::VirtualPolynomial;
use jolt_core::zkvm::RV64IMACProof;
use serde::{Deserialize, Serialize};

/// Preamble data (from JoltDevice) for Fiat-Shamir initialization
#[derive(Debug, Serialize, Deserialize)]
pub struct PreambleData {
    pub max_input_size: u64,
    pub max_output_size: u64,
    pub memory_size: u64,
    pub inputs: Vec<u8>,
    pub outputs: Vec<u8>,
    pub panic: bool,
    pub ram_k: usize,
    pub trace_length: usize,
}

/// Extracted Stage 1 data for transpilation
#[derive(Debug, Serialize, Deserialize)]
pub struct Stage1ExtractedData {
    /// Preamble data for Fiat-Shamir initialization
    pub preamble: Option<PreambleData>,
    /// Coefficients of the univariate-skip first round polynomial
    pub uni_skip_poly_coeffs: Vec<String>,
    /// Sumcheck round polynomial coefficients (one vec per round)
    pub sumcheck_round_polys: Vec<Vec<String>>,
    /// Number of rounds
    pub num_rounds: usize,
    /// Trace length
    pub trace_length: usize,
    /// Serialized commitments (for transcript preamble)
    pub commitments: Vec<Vec<u8>>,
    /// Number of commitments
    pub num_commitments: usize,
    /// Expected final claim (computed by running real verifier)
    pub expected_final_claim: String,
    /// R1CS input evaluations at the final sumcheck point (36 elements)
    /// These are needed for computing the R1CS constraint check.
    pub r1cs_input_evals: Vec<String>,
}

fn main() {
    println!("=== Extracting Stage 1 Inputs from Fibonacci Proof ===\n");

    let proof_path = "/tmp/fib_proof.bin";
    let io_device_path = "/tmp/fib_io_device.bin";

    if !std::path::Path::new(proof_path).exists() {
        eprintln!("No proof found at {}", proof_path);
        eprintln!("\nTo generate a proof, run the fibonacci example with --save:");
        eprintln!("  cd examples/fibonacci && cargo run --release -- --save");
        return;
    }

    println!("Found existing proof at {}", proof_path);

    // Load proof
    let full_proof = match load_proof_from_file(proof_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to load proof: {}", e);
            return;
        }
    };
    println!("Successfully loaded proof!");
    println!("Trace length: {}", full_proof.trace_length);

    // Load io_device (optional - for preamble data)
    let io_device = if std::path::Path::new(io_device_path).exists() {
        match load_io_device_from_file(io_device_path) {
            Ok(device) => {
                println!("Successfully loaded io_device!");
                Some(device)
            }
            Err(e) => {
                eprintln!("Warning: Failed to load io_device: {}", e);
                None
            }
        }
    } else {
        println!("Note: io_device not found at {}, preamble will be empty", io_device_path);
        None
    };

    // Extract Stage 1 data
    let extracted = extract_stage1_data(&full_proof, io_device.as_ref());

    // Print the extracted data
    println!("\n=== Extracted Stage 1 Data ===\n");

    if let Some(preamble) = &extracted.preamble {
        println!("Preamble:");
        println!("  max_input_size: {}", preamble.max_input_size);
        println!("  max_output_size: {}", preamble.max_output_size);
        println!("  memory_size: {}", preamble.memory_size);
        println!("  inputs: {} bytes", preamble.inputs.len());
        println!("  outputs: {} bytes", preamble.outputs.len());
        println!("  panic: {}", preamble.panic);
        println!("  ram_k: {}", preamble.ram_k);
        println!("  trace_length: {}", preamble.trace_length);
    }

    println!("\nTrace length: {}", extracted.trace_length);
    println!("Num rounds: {}", extracted.num_rounds);
    println!(
        "\nUni-skip polynomial coefficients ({} total):",
        extracted.uni_skip_poly_coeffs.len()
    );
    for (i, coeff) in extracted.uni_skip_poly_coeffs.iter().enumerate() {
        println!("  coeff[{}]: {}", i, coeff);
    }

    println!(
        "\nSumcheck round polynomials ({} rounds):",
        extracted.sumcheck_round_polys.len()
    );
    for (round, coeffs) in extracted.sumcheck_round_polys.iter().enumerate() {
        println!("  Round {} ({} coeffs):", round, coeffs.len());
        for (i, coeff) in coeffs.iter().enumerate() {
            println!("    coeff[{}]: {}", i, coeff);
        }
    }

    println!(
        "\nCommitments ({} total):",
        extracted.num_commitments
    );
    for (i, commitment_bytes) in extracted.commitments.iter().enumerate() {
        println!("  commitment[{}]: {} bytes", i, commitment_bytes.len());
    }

    // Save to JSON
    let json_path = "gnark-transpiler/data/fib_stage1_data.json";
    std::fs::create_dir_all("gnark-transpiler/data")
        .expect("Failed to create data dir");
    let json = serde_json::to_string_pretty(&extracted).expect("Failed to serialize");
    std::fs::write(json_path, &json).expect("Failed to write JSON");
    println!("\n✓ Saved to {}", json_path);
}

fn load_proof_from_file(path: &str) -> Result<RV64IMACProof, Box<dyn std::error::Error>> {
    use ark_serialize::CanonicalDeserialize;

    let bytes = std::fs::read(path)?;
    // Note: serialize_and_print_size uses serialize_compressed
    let full_proof: RV64IMACProof = CanonicalDeserialize::deserialize_compressed(&bytes[..])?;

    Ok(full_proof)
}

fn load_io_device_from_file(path: &str) -> Result<JoltDevice, Box<dyn std::error::Error>> {
    use ark_serialize::CanonicalDeserialize;

    let bytes = std::fs::read(path)?;
    let io_device: JoltDevice = CanonicalDeserialize::deserialize_compressed(&bytes[..])?;

    Ok(io_device)
}

fn extract_stage1_data(proof: &RV64IMACProof, io_device: Option<&JoltDevice>) -> Stage1ExtractedData {
    // Extract preamble from io_device if available
    let preamble = io_device.map(|device| PreambleData {
        max_input_size: device.memory_layout.max_input_size,
        max_output_size: device.memory_layout.max_output_size,
        memory_size: device.memory_layout.memory_size,
        inputs: device.inputs.clone(),
        outputs: device.outputs.clone(),
        panic: device.panic,
        ram_k: proof.ram_K,
        trace_length: proof.trace_length,
    });

    // Extract uni-skip polynomial coefficients
    let uni_skip_coeffs: Vec<String> = proof
        .stage1_uni_skip_first_round_proof
        .uni_poly
        .coeffs
        .iter()
        .map(|f| format!("{:?}", f))
        .collect();

    // Extract sumcheck round polynomial coefficients
    // Note: These are CompressedUniPoly, so we need to handle the compression
    let sumcheck_round_polys: Vec<Vec<String>> = proof
        .stage1_sumcheck_proof
        .compressed_polys
        .iter()
        .map(|compressed| {
            // For now, just extract the coefficients_except_linear_term
            // The linear term can be derived from hint during verification
            compressed
                .coeffs_except_linear_term
                .iter()
                .map(|f| format!("{:?}", f))
                .collect()
        })
        .collect();

    // Calculate num_rounds from trace length
    let num_rounds = proof.trace_length.trailing_zeros() as usize;

    // Extract serialized commitments
    // NOTE: PoseidonTranscript::append_serializable does:
    //   1. serialize_uncompressed -> bytes (LE)
    //   2. reverse bytes (for EVM compat)
    //   3. append_bytes (chunks of 32 bytes)
    // So we store the REVERSED bytes here, matching what the transcript hashes.
    let commitments: Vec<Vec<u8>> = proof
        .commitments
        .iter()
        .map(|commitment| {
            let mut bytes = Vec::new();
            commitment
                .serialize_uncompressed(&mut bytes)
                .expect("Failed to serialize commitment");
            // Reverse to match PoseidonTranscript::append_serializable
            bytes.reverse();
            bytes
        })
        .collect();

    let num_commitments = commitments.len();

    // =========================================================================
    // Compute expected final_claim by running real Stage 1 verification
    // Using verify_stage1_full from stepwise_verifier (matches transpilation!)
    // =========================================================================
    println!("\n=== Computing Expected Final Claim (using stepwise_verifier) ===");

    // First extract R1CS input evaluations from proof.opening_claims
    let r1cs_input_evals_fr: [Fr; NUM_R1CS_INPUTS] = {
        let evals: Vec<Fr> = ALL_R1CS_INPUTS
            .iter()
            .map(|input| {
                let virtual_poly = VirtualPolynomial::from(input);
                let opening_id = OpeningId::Virtual(virtual_poly, SumcheckId::SpartanOuter);
                let (_point, claim) = proof
                    .opening_claims
                    .0
                    .get(&opening_id)
                    .unwrap_or_else(|| panic!("Missing opening for R1CS input {:?}", input));
                *claim
            })
            .collect();
        evals.try_into().expect("Wrong number of R1CS inputs")
    };

    // Build preamble for stepwise_verifier (now uses concrete types, not Fr!)
    let stepwise_preamble = io_device.map(|device| {
        StepwisePreambleData {
            max_input_size: device.memory_layout.max_input_size,
            max_output_size: device.memory_layout.max_output_size,
            memory_size: device.memory_layout.memory_size,
            inputs: device.inputs.clone(),
            outputs: device.outputs.clone(),
            panic: device.panic,
            ram_k: proof.ram_K,
            trace_length: proof.trace_length,
        }
    }).expect("io_device required for stepwise verification");

    // Use original commitments directly (G1Affine implements CanonicalSerialize)
    // The generic Stage1FullVerificationData<F, C> now accepts any C: CanonicalSerialize
    // This matches the real verifier exactly: transcript.append_serializable(commitment)
    let commitments_for_verification = proof.commitments.clone();

    // Debug: check for advice commitments
    println!("\n=== Advice Commitments Debug ===");
    println!("  untrusted_advice_commitment: {:?}", proof.untrusted_advice_commitment.is_some());
    // Note: trusted_advice_commitment comes from preprocessing, not proof

    // Extract uni-skip coefficients as Fr
    let uni_skip_coeffs_fr: Vec<Fr> = proof
        .stage1_uni_skip_first_round_proof
        .uni_poly
        .coeffs
        .clone();

    // Extract sumcheck round polys as Fr
    let sumcheck_round_polys_fr: Vec<Vec<Fr>> = proof
        .stage1_sumcheck_proof
        .compressed_polys
        .iter()
        .map(|compressed| compressed.coeffs_except_linear_term.clone())
        .collect();

    // Build verification data for stepwise_verifier
    // num_cycle_vars = log2(trace_length) for trace_length 1024 = 10
    let num_cycle_vars = proof.trace_length.trailing_zeros() as usize;
    let verification_data = Stage1FullVerificationData {
        preamble: stepwise_preamble,
        commitments: commitments_for_verification,
        uni_skip_poly_coeffs: uni_skip_coeffs_fr,
        sumcheck_round_polys: sumcheck_round_polys_fr,
        r1cs_input_evals: r1cs_input_evals_fr,
        num_cycle_vars,
    };

    // Run verification with Poseidon transcript (must match Gnark circuit!)
    let mut transcript: PoseidonTranscript<Fr, FrParams> = Transcript::new(b"Jolt");
    let result = verify_stage1_full(verification_data, &mut transcript);

    let expected_final_claim = format!("{:?}", result.final_claim);
    println!("  final_claim: {}", expected_final_claim);
    println!("  power_sum_check: {:?}", result.power_sum_check);
    println!("  final_check: {:?} (should be 0!)", result.final_check);
    println!("  claims_after_round len: {}", result.claims_after_round.len());

    // Debug: print derived challenges
    println!("\n=== Stepwise Derived Challenges ===");
    println!("  tau[0]: {:?}", result.tau[0]);
    println!("  tau[len-1] (tau_high): {:?}", result.tau[result.tau.len() - 1]);
    println!("  r0: {:?}", result.r0);
    println!("  batching_coeff: {:?}", result.batching_coeff);
    println!("  claim_after_uni_skip: {:?}", result.claim_after_uni_skip);
    println!("  sumcheck_challenges[0]: {:?}", result.sumcheck_challenges[0]);
    if result.sumcheck_challenges.len() > 1 {
        println!("  sumcheck_challenges[1]: {:?}", result.sumcheck_challenges[1]);
    }
    println!("  tau_high_bound_r0: {:?}", result.tau_high_bound_r0);
    println!("  tau_bound_r_tail: {:?}", result.tau_bound_r_tail);
    println!("  inner_sum_prod: {:?}", result.inner_sum_prod);
    println!("  expected_output_claim: {:?}", result.expected_output_claim);

    // R1CS input evals already extracted above - convert to strings for JSON
    println!("\n=== R1CS Input Evaluations ===");
    let r1cs_input_evals: Vec<String> = r1cs_input_evals_fr
        .iter()
        .enumerate()
        .map(|(i, eval)| {
            let eval_str = format!("{:?}", eval);
            println!("  r1cs_input[{}]: {}...", i, &eval_str[..20.min(eval_str.len())]);
            eval_str
        })
        .collect();
    println!("  Total: {} R1CS input evaluations", r1cs_input_evals.len());

    Stage1ExtractedData {
        preamble,
        uni_skip_poly_coeffs: uni_skip_coeffs,
        sumcheck_round_polys,
        num_rounds,
        trace_length: proof.trace_length,
        commitments,
        num_commitments,
        expected_final_claim,
        r1cs_input_evals,
    }
}
