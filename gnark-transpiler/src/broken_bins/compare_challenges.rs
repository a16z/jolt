//! Compare challenges between real verifier and stepwise verifier
//!
//! This identifies exactly where the transcript diverges.
//!
//! Usage: cargo run -p gnark-transpiler --bin compare_challenges --release

use ark_bn254::Fr;
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
    println!("=== Comparing Transcript Challenges ===\n");

    let proof = load_proof().expect("Failed to load proof");
    let io_device = load_io_device().expect("Failed to load io_device");

    // Create two identical transcripts
    let mut real_transcript: PoseidonTranscript<Fr, FrParams> = Transcript::new(b"Jolt");
    let mut stepwise_transcript: PoseidonTranscript<Fr, FrParams> = Transcript::new(b"Jolt");

    println!("=== Step 1: Preamble ===");

    // Real verifier preamble (from verifier.rs)
    real_transcript.append_u64(io_device.memory_layout.max_input_size);
    real_transcript.append_u64(io_device.memory_layout.max_output_size);
    real_transcript.append_u64(io_device.memory_layout.memory_size);
    real_transcript.append_bytes(&io_device.inputs);
    real_transcript.append_bytes(&io_device.outputs);
    real_transcript.append_u64(io_device.panic as u64);
    real_transcript.append_u64(proof.ram_K as u64);
    real_transcript.append_u64(proof.trace_length as u64);

    // Stepwise preamble (should be identical)
    stepwise_transcript.append_u64(io_device.memory_layout.max_input_size);
    stepwise_transcript.append_u64(io_device.memory_layout.max_output_size);
    stepwise_transcript.append_u64(io_device.memory_layout.memory_size);
    stepwise_transcript.append_bytes(&io_device.inputs);
    stepwise_transcript.append_bytes(&io_device.outputs);
    stepwise_transcript.append_u64(io_device.panic as u64);
    stepwise_transcript.append_u64(proof.ram_K as u64);
    stepwise_transcript.append_u64(proof.trace_length as u64);

    // Compare state after preamble
    let real_state_1: Fr = real_transcript.challenge_scalar();
    let stepwise_state_1: Fr = stepwise_transcript.challenge_scalar();
    println!("After preamble challenge:");
    println!("  Real:     {}", fr_to_string(&real_state_1));
    println!("  Stepwise: {}", fr_to_string(&stepwise_state_1));
    println!("  Match: {}", real_state_1 == stepwise_state_1);

    if real_state_1 != stepwise_state_1 {
        println!("\n*** DIVERGENCE at preamble! ***");
        return;
    }

    println!("\n=== Step 2: Commitments ===");

    // Real verifier: append_serializable for each commitment
    for (i, commitment) in proof.commitments.iter().enumerate() {
        real_transcript.append_serializable(commitment);
        stepwise_transcript.append_serializable(commitment);

        if i < 3 || i == proof.commitments.len() - 1 {
            let real_c: Fr = real_transcript.challenge_scalar();
            let stepwise_c: Fr = stepwise_transcript.challenge_scalar();
            let real_str = fr_to_string(&real_c);
            let stepwise_str = fr_to_string(&stepwise_c);
            println!("After commitment[{}]:", i);
            println!("  Real:     {}...", &real_str[..real_str.len().min(40)]);
            println!("  Stepwise: {}...", &stepwise_str[..stepwise_str.len().min(40)]);
            println!("  Match: {}", real_c == stepwise_c);

            if real_c != stepwise_c {
                println!("\n*** DIVERGENCE at commitment {}! ***", i);
                return;
            }
        }
    }

    println!("\n=== Step 3: Tau challenges ===");

    let num_cycle_vars = proof.trace_length.trailing_zeros() as usize;
    let real_tau: Vec<Fr> = real_transcript.challenge_vector(num_cycle_vars);
    let stepwise_tau: Vec<Fr> = stepwise_transcript.challenge_vector(num_cycle_vars);

    for i in 0..num_cycle_vars {
        println!("tau[{}]: Real={} Stepwise={} Match={}",
            i,
            &fr_to_string(&real_tau[i])[..20],
            &fr_to_string(&stepwise_tau[i])[..20],
            real_tau[i] == stepwise_tau[i]
        );
        if real_tau[i] != stepwise_tau[i] {
            println!("\n*** DIVERGENCE at tau[{}]! ***", i);
            return;
        }
    }

    println!("\n=== Step 4: Univariate-skip round ===");

    // Append uni-skip poly
    real_transcript.append_message(b"UncompressedUniPoly_begin");
    stepwise_transcript.append_message(b"UncompressedUniPoly_begin");

    for coeff in &proof.stage1_uni_skip_first_round_proof.uni_poly.coeffs {
        real_transcript.append_scalar(coeff);
        stepwise_transcript.append_scalar(coeff);
    }

    real_transcript.append_message(b"UncompressedUniPoly_end");
    stepwise_transcript.append_message(b"UncompressedUniPoly_end");

    let real_r0: Fr = real_transcript.challenge_scalar();
    let stepwise_r0: Fr = stepwise_transcript.challenge_scalar();
    println!("r0 challenge:");
    println!("  Real:     {}", fr_to_string(&real_r0));
    println!("  Stepwise: {}", fr_to_string(&stepwise_r0));
    println!("  Match: {}", real_r0 == stepwise_r0);

    if real_r0 != stepwise_r0 {
        println!("\n*** DIVERGENCE at r0! ***");
        return;
    }

    println!("\n=== Step 5: Batching coefficient ===");

    let real_batching: Fr = real_transcript.challenge_scalar();
    let stepwise_batching: Fr = stepwise_transcript.challenge_scalar();
    println!("batching_coeff:");
    println!("  Real:     {}", fr_to_string(&real_batching));
    println!("  Stepwise: {}", fr_to_string(&stepwise_batching));
    println!("  Match: {}", real_batching == stepwise_batching);

    if real_batching != stepwise_batching {
        println!("\n*** DIVERGENCE at batching_coeff! ***");
        return;
    }

    println!("\n=== Step 6: Sumcheck rounds ===");

    for (round, compressed_poly) in proof.stage1_sumcheck_proof.compressed_polys.iter().enumerate() {
        // Append compressed poly
        real_transcript.append_message(b"UniPoly_begin");
        stepwise_transcript.append_message(b"UniPoly_begin");

        for coeff in &compressed_poly.coeffs_except_linear_term {
            real_transcript.append_scalar(coeff);
            stepwise_transcript.append_scalar(coeff);
        }

        real_transcript.append_message(b"UniPoly_end");
        stepwise_transcript.append_message(b"UniPoly_end");

        let real_r: Fr = real_transcript.challenge_scalar();
        let stepwise_r: Fr = stepwise_transcript.challenge_scalar();

        let matches = real_r == stepwise_r;
        println!("Round {} challenge: Match={}", round, matches);

        if !matches {
            println!("  Real:     {}", fr_to_string(&real_r));
            println!("  Stepwise: {}", fr_to_string(&stepwise_r));
            println!("\n*** DIVERGENCE at sumcheck round {}! ***", round);
            return;
        }
    }

    println!("\n=== ALL CHALLENGES MATCH! ===");
    println!("\nIf FinalCheck still fails, the problem is in:");
    println!("1. PoseidonAstTranscript (symbolic transcript)");
    println!("2. The arithmetic computation in stepwise_verifier");
    println!("3. The Gnark circuit code generation");
}
