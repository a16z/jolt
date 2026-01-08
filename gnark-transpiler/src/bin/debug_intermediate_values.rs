// //! Debug intermediate values from verify_stage1_full
// //!
// //! This script runs the REAL verifier with concrete Fr values and prints
// //! all intermediate values. Use this to compare against the Gnark circuit.
// //!
// //! IMPORTANT: This uses the same data flow as the Gnark circuit:
// //! - Preamble is appended via append_u64/append_bytes (concrete values)
// //! - Commitments are serialized G1Affine points (384 bytes each)
// //! - The transcript operations must match exactly
//
// use ark_bn254::Fr;
// use ark_bls12_381::G1Affine;
// use ark_ff::BigInt;
// use ark_serialize::CanonicalDeserialize;
// use jolt_core::transcripts::{FrParams, PoseidonTranscript, Transcript};
// use jolt_core::zkvm::r1cs::inputs::NUM_R1CS_INPUTS;
// use jolt_core::zkvm::stepwise_verifier::{
//     verify_stage1_full, PreambleData, Stage1FullVerificationData,
// };
// use serde::Deserialize;
// use std::str::FromStr;
//
// #[derive(Deserialize)]
// struct ExtractedPreamble {
//     max_input_size: u64,
//     max_output_size: u64,
//     memory_size: u64,
//     inputs: Vec<u8>,
//     outputs: Vec<u8>,
//     panic: bool,
//     ram_k: u64,
//     trace_length: u64,
// }
//
// #[derive(Deserialize)]
// struct ExtractedStage1Data {
//     preamble: ExtractedPreamble,
//     uni_skip_poly_coeffs: Vec<String>,
//     sumcheck_round_polys: Vec<Vec<String>>,
//     commitments: Vec<Vec<u8>>,
//     #[serde(default)]
//     r1cs_input_evals: Vec<String>,
// }
//
// fn fr_to_decimal(f: &Fr) -> String {
//     let bigint: BigInt<4> = (*f).into();
//     let mut bytes = [0u8; 32];
//     for (i, limb) in bigint.0.iter().enumerate() {
//         bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
//     }
//     // Convert to decimal string
//     let mut result = num_bigint::BigUint::from_bytes_le(&bytes);
//     result.to_string()
// }
//
// fn main() {
//     let manifest_dir = env!("CARGO_MANIFEST_DIR");
//     let data_path = format!("{}/data/fib_stage1_data.json", manifest_dir);
//
//     println!("Loading data from: {}", data_path);
//     let json_content = std::fs::read_to_string(&data_path).expect("Failed to read data file");
//     let extracted: ExtractedStage1Data =
//         serde_json::from_str(&json_content).expect("Failed to parse JSON");
//
//     let num_cycle_vars = extracted.sumcheck_round_polys.len() - 1;
//     println!("num_cycle_vars: {}", num_cycle_vars);
//     println!("num_sumcheck_rounds: {}", extracted.sumcheck_round_polys.len());
//
//     // Build preamble (concrete values)
//     let preamble = PreambleData {
//         max_input_size: extracted.preamble.max_input_size,
//         max_output_size: extracted.preamble.max_output_size,
//         memory_size: extracted.preamble.memory_size,
//         inputs: extracted.preamble.inputs.clone(),
//         outputs: extracted.preamble.outputs.clone(),
//         panic: extracted.preamble.panic,
//         ram_k: extracted.preamble.ram_k as usize,
//         trace_length: extracted.preamble.trace_length as usize,
//     };
//
//     // Build commitments - deserialize G1Affine from bytes
//     // The extracted data contains serialized G1Affine points (384 bytes each, uncompressed)
//     let commitments: Vec<G1Affine> = extracted
//         .commitments
//         .iter()
//         .map(|bytes| {
//             G1Affine::deserialize_uncompressed(&bytes[..]).expect("deserialize G1Affine")
//         })
//         .collect();
//
//     println!("Loaded {} G1Affine commitments", commitments.len());
//
//     // Parse uni_skip coefficients
//     let uni_skip_poly_coeffs: Vec<Fr> = extracted
//         .uni_skip_poly_coeffs
//         .iter()
//         .map(|s| Fr::from_str(s).expect("parse uni_skip coeff"))
//         .collect();
//
//     // Parse sumcheck polynomials
//     let sumcheck_round_polys: Vec<Vec<Fr>> = extracted
//         .sumcheck_round_polys
//         .iter()
//         .map(|round| {
//             round
//                 .iter()
//                 .map(|s| Fr::from_str(s).expect("parse sumcheck coeff"))
//                 .collect()
//         })
//         .collect();
//
//     // Parse R1CS input evals
//     let r1cs_input_evals: [Fr; NUM_R1CS_INPUTS] = {
//         let vec: Vec<Fr> = extracted
//             .r1cs_input_evals
//             .iter()
//             .map(|s| Fr::from_str(s).expect("parse r1cs eval"))
//             .collect();
//         vec.try_into().expect("wrong number of r1cs inputs")
//     };
//
//     // Build verification data
//     let data = Stage1FullVerificationData {
//         preamble,
//         commitments,
//         uni_skip_poly_coeffs,
//         sumcheck_round_polys,
//         r1cs_input_evals,
//         num_cycle_vars,
//     };
//
//     // Run with REAL PoseidonTranscript
//     println!("\n=== Running verify_stage1_full with PoseidonTranscript<Fr> ===\n");
//     let mut transcript: PoseidonTranscript<Fr, FrParams> = Transcript::new(b"Jolt");
//     let result = verify_stage1_full(data, &mut transcript);
//
//     // Print all intermediate values
//     println!("=== DERIVED CHALLENGES ===");
//     for (i, tau) in result.tau.iter().enumerate() {
//         let tau_fr: Fr = (*tau).into();
//         println!("tau[{:2}] = {}", i, fr_to_decimal(&tau_fr));
//     }
//     println!();
//
//     let r0_fr: Fr = result.r0.into();
//     println!("r0 = {}", fr_to_decimal(&r0_fr));
//     println!("batching_coeff = {}", fr_to_decimal(&result.batching_coeff));
//     println!();
//
//     for (i, r) in result.sumcheck_challenges.iter().enumerate() {
//         let r_fr: Fr = (*r).into();
//         println!("sumcheck_challenge[{:2}] = {}", i, fr_to_decimal(&r_fr));
//     }
//
//     println!("\n=== INTERMEDIATE CLAIMS ===");
//     println!("claim_after_uni_skip = {}", fr_to_decimal(&result.claim_after_uni_skip));
//     for (i, claim) in result.claims_after_round.iter().enumerate() {
//         println!("claim_after_round[{:2}] = {}", i, fr_to_decimal(claim));
//     }
//     println!("final_claim = {}", fr_to_decimal(&result.final_claim));
//
//     println!("\n=== EXPECTED OUTPUT CLAIM COMPONENTS ===");
//     println!("tau_high_bound_r0 = {}", fr_to_decimal(&result.tau_high_bound_r0));
//     println!("tau_bound_r_tail = {}", fr_to_decimal(&result.tau_bound_r_tail));
//     println!("inner_sum_prod = {}", fr_to_decimal(&result.inner_sum_prod));
//     println!("expected_output_claim = {}", fr_to_decimal(&result.expected_output_claim));
//
//     println!("\n=== CONSTRAINT CHECKS ===");
//     println!("power_sum_check = {}", fr_to_decimal(&result.power_sum_check));
//     println!("final_check = {}", fr_to_decimal(&result.final_check));
//
//     // Verify they're zero
//     println!("\n=== VERIFICATION ===");
//     if result.power_sum_check == Fr::from(0u64) {
//         println!("✓ power_sum_check == 0");
//     } else {
//         println!("✗ power_sum_check != 0 (FAIL)");
//     }
//     if result.final_check == Fr::from(0u64) {
//         println!("✓ final_check == 0");
//     } else {
//         println!("✗ final_check != 0 (FAIL)");
//     }
//
//     // Decompose final_check
//     println!("\n=== DECOMPOSITION OF final_check ===");
//     println!("final_check = final_claim - expected_output_claim");
//     println!("  final_claim          = {}", fr_to_decimal(&result.final_claim));
//     println!("  expected_output_claim = {}", fr_to_decimal(&result.expected_output_claim));
//     println!();
//     println!("expected_output_claim = tau_high_bound_r0 * tau_bound_r_tail * inner_sum_prod * batching_coeff");
//     println!("  tau_high_bound_r0 = {}", fr_to_decimal(&result.tau_high_bound_r0));
//     println!("  tau_bound_r_tail  = {}", fr_to_decimal(&result.tau_bound_r_tail));
//     println!("  inner_sum_prod    = {}", fr_to_decimal(&result.inner_sum_prod));
//     println!("  batching_coeff    = {}", fr_to_decimal(&result.batching_coeff));
// }

fn main() {
    println!("This binary is disabled - missing ark_bls12_381 dependency");
}
