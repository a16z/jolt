//! Run full Stage 1 verification and print all intermediate values
//!
//! This runs verify_stage1_full with the actual proof data to compare
//! against the Gnark circuit computation.

use ark_bn254::Fr;
use ark_ff::{BigInt, PrimeField};
use ark_serialize::CanonicalDeserialize;
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
use jolt_core::transcripts::{FrParams, PoseidonTranscript, Transcript};
use jolt_core::zkvm::stepwise_verifier::{
    verify_stage1_full, PreambleData, Stage1FullVerificationData,
};
use jolt_core::zkvm::transpilable_verifier::JoltVerifierPreprocessing;
use jolt_core::zkvm::RV64IMACProof;
use common::jolt_device::JoltDevice;

fn fr_to_decimal(f: &Fr) -> String {
    let bigint: BigInt<4> = (*f).into();
    let mut bytes = [0u8; 32];
    for (i, limb) in bigint.0.iter().enumerate() {
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    num_bigint::BigUint::from_bytes_le(&bytes).to_string()
}

fn main() {
    println!("=== Running Full Stage 1 Verification ===\n");

    // Load proof
    let proof_path = "/tmp/fib_proof.bin";
    println!("Loading proof from: {}", proof_path);
    let proof_bytes = std::fs::read(proof_path).expect("Failed to read proof file");
    let proof: RV64IMACProof =
        CanonicalDeserialize::deserialize_compressed(&proof_bytes[..])
            .expect("Failed to deserialize proof");
    println!("  trace_length: {}", proof.trace_length);

    // Load io_device
    let io_device_path = "/tmp/fib_io_device.bin";
    println!("Loading io_device from: {}", io_device_path);
    let io_device_bytes = std::fs::read(io_device_path).expect("Failed to read io_device file");
    let io_device: JoltDevice = CanonicalDeserialize::deserialize_compressed(&io_device_bytes[..])
        .expect("Failed to deserialize io_device");

    // Load preprocessing
    let preprocessing_path = "/tmp/jolt_verifier_preprocessing.dat";
    println!("Loading preprocessing from: {}", preprocessing_path);
    let preprocessing_bytes =
        std::fs::read(preprocessing_path).expect("Failed to read preprocessing file");
    let preprocessing: JoltVerifierPreprocessing<Fr, DoryCommitmentScheme> =
        CanonicalDeserialize::deserialize_compressed(&preprocessing_bytes[..])
            .expect("Failed to deserialize preprocessing");

    // Build preamble
    let preamble = PreambleData {
        max_input_size: preprocessing.memory_layout.max_input_size as u64,
        max_output_size: preprocessing.memory_layout.max_output_size as u64,
        memory_size: io_device.memory_layout.memory_size as u64,
        inputs: io_device.inputs.clone(),
        outputs: io_device.outputs.clone(),
        panic: io_device.panic,
        ram_k: proof.ram_K,
        trace_length: proof.trace_length,
    };

    // Extract Stage 1 data from proof - new field names
    let uni_skip_poly_coeffs: Vec<Fr> = proof.stage1_uni_skip_first_round_proof.uni_poly.coeffs.clone();

    let sumcheck_round_polys: Vec<Vec<Fr>> = proof.stage1_sumcheck_proof
        .compressed_polys
        .iter()
        .map(|p| p.coeffs_except_linear_term.clone())
        .collect();

    let num_cycle_vars = proof.trace_length.ilog2() as usize;
    println!("  num_cycle_vars: {}", num_cycle_vars);
    println!("  num_sumcheck_rounds: {}", sumcheck_round_polys.len());
    println!("  uni_skip_coeffs: {}", uni_skip_poly_coeffs.len());

    // Extract R1CS input evaluations from the opening claims
    // These are the virtual polynomial claims at r_cycle
    use jolt_core::zkvm::r1cs::inputs::{JoltR1CSInputs, NUM_R1CS_INPUTS};
    use jolt_core::poly::opening_proof::SumcheckId;
    use jolt_core::zkvm::witness::VirtualPolynomial;
    use strum::IntoEnumIterator;

    let mut r1cs_input_evals = [Fr::from(0u64); NUM_R1CS_INPUTS];
    for input in JoltR1CSInputs::iter() {
        let poly = VirtualPolynomial::from(&input);
        let key = (poly, SumcheckId::SpartanOuter);
        if let Some((_, claim)) = proof.opening_claims.0.get(&key.into()) {
            r1cs_input_evals[input as usize] = *claim;
        } else {
            println!("Warning: Missing claim for {:?}", input);
        }
    }

    // Build verification data
    let data = Stage1FullVerificationData {
        preamble,
        commitments: proof.commitments.clone(),
        uni_skip_poly_coeffs,
        sumcheck_round_polys,
        r1cs_input_evals,
        num_cycle_vars,
    };

    // Run verification with real PoseidonTranscript
    println!("\n=== Running verify_stage1_full ===\n");
    let mut transcript: PoseidonTranscript<Fr, FrParams> = Transcript::new(b"Jolt");
    let result = verify_stage1_full(data, &mut transcript);

    // Print derived challenges
    println!("=== DERIVED CHALLENGES ===");
    println!("tau_len = {}", result.tau.len());
    for (i, tau) in result.tau.iter().enumerate() {
        let tau_fr: Fr = (*tau).into();
        println!("tau[{:2}] = {}", i, fr_to_decimal(&tau_fr));
    }
    println!();

    let r0_fr: Fr = result.r0.into();
    println!("r0 = {}", fr_to_decimal(&r0_fr));
    println!("batching_coeff = {}", fr_to_decimal(&result.batching_coeff));
    println!();

    println!("sumcheck_challenges_len = {}", result.sumcheck_challenges.len());
    for (i, r) in result.sumcheck_challenges.iter().enumerate() {
        let r_fr: Fr = (*r).into();
        println!("sumcheck_challenge[{:2}] = {}", i, fr_to_decimal(&r_fr));
    }

    // Print intermediate claims
    println!("\n=== INTERMEDIATE CLAIMS ===");
    println!("claim_after_uni_skip = {}", fr_to_decimal(&result.claim_after_uni_skip));
    for (i, claim) in result.claims_after_round.iter().enumerate() {
        println!("claim_after_round[{:2}] = {}", i, fr_to_decimal(claim));
    }
    println!("final_claim = {}", fr_to_decimal(&result.final_claim));

    // Print expected output claim components
    println!("\n=== EXPECTED OUTPUT CLAIM COMPONENTS ===");
    println!("tau_high_bound_r0 = {}", fr_to_decimal(&result.tau_high_bound_r0));
    println!("tau_bound_r_tail = {}", fr_to_decimal(&result.tau_bound_r_tail));
    println!("inner_sum_prod = {}", fr_to_decimal(&result.inner_sum_prod));
    println!("expected_output_claim = {}", fr_to_decimal(&result.expected_output_claim));

    // Print constraint checks
    println!("\n=== CONSTRAINT CHECKS ===");
    println!("power_sum_check = {} (should be 0)", fr_to_decimal(&result.power_sum_check));
    println!("final_check = {} (should be 0)", fr_to_decimal(&result.final_check));

    // Verify they're zero
    println!("\n=== VERIFICATION STATUS ===");
    if result.power_sum_check == Fr::from(0u64) {
        println!("PASS: power_sum_check == 0");
    } else {
        println!("FAIL: power_sum_check != 0");
    }
    if result.final_check == Fr::from(0u64) {
        println!("PASS: final_check == 0");
    } else {
        println!("FAIL: final_check != 0");
        println!("  final_claim          = {}", fr_to_decimal(&result.final_claim));
        println!("  expected_output_claim = {}", fr_to_decimal(&result.expected_output_claim));
    }
}
