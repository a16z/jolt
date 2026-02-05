#![allow(dead_code, unused_variables, unused_imports)]
//! Debug transcript step by step with real proof data
//!
//! Prints internal state after each operation to compare with Go.
//!
//! Usage: cargo run -p gnark-transpiler --bin debug_transcript_full --release

use ark_bn254::Fr;
use ark_ff::{Field, PrimeField};
use ark_serialize::CanonicalSerialize;
use common::jolt_device::JoltDevice;
use jolt_core::transcripts::{FrParams, PoseidonTranscript, Transcript};
use jolt_core::zkvm::RV64IMACProof;
use num_bigint::BigUint;

fn fr_to_string(f: &Fr) -> String {
    let mut bytes = vec![];
    f.serialize_uncompressed(&mut bytes).unwrap();
    BigUint::from_bytes_le(&bytes).to_string()
}

/// Convert u128 to Fr as MontU128Challenge does:
/// 1. Apply 125-bit mask: value & (u128::MAX >> 3)
/// 2. Multiply by 2^128 (store as [0, 0, low, high] in Montgomery form)
fn u128_to_mont_challenge_fr(value: u128) -> Fr {
    // Apply 125-bit mask
    let masked = value & (u128::MAX >> 3);

    // Create Fr from [0, 0, low, high] which represents masked * 2^128
    let low = masked as u64;
    let high = (masked >> 64) as u64;

    // MontU128Challenge stores [0, 0, low, high] directly as Montgomery form
    // This is equivalent to: Fr::from(masked) * Fr::from(2^128)
    let two_pow_128 = Fr::from(2u64).pow([128]);
    Fr::from(masked) * two_pow_128
}

fn state_to_fr(state: &[u8; 32]) -> Fr {
    Fr::from_le_bytes_mod_order(state)
}

fn load_proof() -> Result<RV64IMACProof, Box<dyn std::error::Error>> {
    use ark_serialize::CanonicalDeserialize;
    let bytes = std::fs::read("/tmp/fib_proof.bin")?;
    Ok(CanonicalDeserialize::deserialize_compressed(&bytes[..])?)
}

fn load_io_device() -> Result<JoltDevice, Box<dyn std::error::Error>> {
    use ark_serialize::CanonicalDeserialize;
    let bytes = std::fs::read("/tmp/fib_io_device.bin")?;
    Ok(CanonicalDeserialize::deserialize_compressed(&bytes[..])?)
}

fn main() {
    println!("=== Rust Transcript Debug (Full) ===\n");

    let proof = load_proof().expect("Failed to load proof");
    let io_device = load_io_device().expect("Failed to load io_device");

    // Create transcript
    let mut transcript: PoseidonTranscript<Fr, FrParams> = Transcript::new(b"Jolt");

    println!("After init (Transcript::new(b\"Jolt\")):");
    println!("  state = {}", fr_to_string(&state_to_fr(&transcript.state)));
    println!();

    // === PREAMBLE ===
    println!("=== PREAMBLE ===");

    transcript.append_u64(io_device.memory_layout.max_input_size);
    println!("After append_u64(max_input_size={}):", io_device.memory_layout.max_input_size);
    println!("  state = {}", fr_to_string(&state_to_fr(&transcript.state)));

    transcript.append_u64(io_device.memory_layout.max_output_size);
    println!("After append_u64(max_output_size={}):", io_device.memory_layout.max_output_size);
    println!("  state = {}", fr_to_string(&state_to_fr(&transcript.state)));

    transcript.append_u64(io_device.memory_layout.memory_size);
    println!("After append_u64(memory_size={}):", io_device.memory_layout.memory_size);
    println!("  state = {}", fr_to_string(&state_to_fr(&transcript.state)));

    transcript.append_bytes(&io_device.inputs);
    println!("After append_bytes(inputs, {} bytes):", io_device.inputs.len());
    println!("  state = {}", fr_to_string(&state_to_fr(&transcript.state)));

    transcript.append_bytes(&io_device.outputs);
    println!("After append_bytes(outputs, {} bytes):", io_device.outputs.len());
    println!("  state = {}", fr_to_string(&state_to_fr(&transcript.state)));

    transcript.append_u64(io_device.panic as u64);
    println!("After append_u64(panic={}):", io_device.panic as u64);
    println!("  state = {}", fr_to_string(&state_to_fr(&transcript.state)));

    transcript.append_u64(proof.ram_K as u64);
    println!("After append_u64(ram_K={}):", proof.ram_K);
    println!("  state = {}", fr_to_string(&state_to_fr(&transcript.state)));

    transcript.append_u64(proof.trace_length as u64);
    println!("After append_u64(trace_length={}):", proof.trace_length);
    println!("  state = {}", fr_to_string(&state_to_fr(&transcript.state)));
    println!();

    // Store state after preamble
    let state_after_preamble = fr_to_string(&state_to_fr(&transcript.state));
    println!("*** STATE AFTER PREAMBLE: {} ***", state_after_preamble);
    println!();

    // === COMMITMENTS ===
    println!("=== COMMITMENTS ({} total) ===", proof.commitments.len());

    for (i, commitment) in proof.commitments.iter().enumerate() {
        transcript.append_serializable(commitment);

        if i < 5 || i == proof.commitments.len() - 1 {
            println!("After commitment[{}]:", i);
            println!("  state = {}", fr_to_string(&state_to_fr(&transcript.state)));
        }
    }

    let state_after_commitments = fr_to_string(&state_to_fr(&transcript.state));
    println!();
    println!("*** STATE AFTER ALL COMMITMENTS: {} ***", state_after_commitments);
    println!();

    // === FIRST CHALLENGE (tau[0]) ===
    // El verifier real llama challenge_vector_optimized que usa challenge_u128 internamente
    // challenge_u128 hace: hash(state, n_rounds, 0) -> toma 16 bytes -> reverse -> u128
    // Luego MontU128Challenge aplica: mask 125 bits + shift por 2^128
    println!("=== FIRST CHALLENGE ===");
    println!("State before challenge: {}", fr_to_string(&state_to_fr(&transcript.state)));

    let tau0_u128: u128 = transcript.challenge_u128();
    let tau0_fr = u128_to_mont_challenge_fr(tau0_u128);
    println!("tau0 u128 raw: {}", tau0_u128);
    println!("tau0 as Fr (MontU128Challenge): {}", fr_to_string(&tau0_fr));
    println!("State after challenge: {}", fr_to_string(&state_to_fr(&transcript.state)));
    println!();

    // === REMAINING TAU CHALLENGES ===
    // We already consumed tau[0], now get tau[1..12]
    let num_rows_bits = (proof.trace_length.trailing_zeros() as usize) + 2; // 10 + 2 = 12
    println!("=== REMAINING TAU CHALLENGES (11 more, {} total) ===", num_rows_bits);
    for i in 1..num_rows_bits {
        println!("Before tau[{}]: state = {}", i, fr_to_string(&state_to_fr(&transcript.state)));
        let tau_u128: u128 = transcript.challenge_u128();
        let tau_fr = u128_to_mont_challenge_fr(tau_u128);
        println!("tau[{}] u128 raw: {}", i, tau_u128);
        println!("tau[{}] as Fr (MontU128Challenge): {}", i, fr_to_string(&tau_fr));
    }
    println!("state after tau = {}", fr_to_string(&state_to_fr(&transcript.state)));
    println!();

    // === UNIVARIATE SKIP ===
    println!("=== UNIVARIATE SKIP ===");
    transcript.append_message(b"UncompressedUniPoly_begin");
    for coeff in &proof.stage1_uni_skip_first_round_proof.uni_poly.coeffs {
        transcript.append_scalar(coeff);
    }
    transcript.append_message(b"UncompressedUniPoly_end");

    let r0: Fr = transcript.challenge_scalar();
    println!("r0 (after uni-skip) = {}", fr_to_string(&r0));
    println!("state after r0 = {}", fr_to_string(&state_to_fr(&transcript.state)));
    println!();

    // === BATCHING COEFFICIENT ===
    println!("=== BATCHING COEFFICIENT ===");
    let batching: Fr = transcript.challenge_scalar();
    println!("batching_coeff = {}", fr_to_string(&batching));
    println!();

    // === SUMCHECK ROUNDS ===
    println!("=== SUMCHECK ROUNDS ({} total) ===", proof.stage1_sumcheck_proof.compressed_polys.len());

    for (round, compressed_poly) in proof.stage1_sumcheck_proof.compressed_polys.iter().enumerate() {
        transcript.append_message(b"UniPoly_begin");
        for coeff in &compressed_poly.coeffs_except_linear_term {
            transcript.append_scalar(coeff);
        }
        transcript.append_message(b"UniPoly_end");

        let r: Fr = transcript.challenge_scalar();
        if round < 3 || round == proof.stage1_sumcheck_proof.compressed_polys.len() - 1 {
            println!("sumcheck_r[{}] = {}", round, fr_to_string(&r));
        }
    }

    println!();
    println!("*** FINAL STATE: {} ***", fr_to_string(&state_to_fr(&transcript.state)));
}
