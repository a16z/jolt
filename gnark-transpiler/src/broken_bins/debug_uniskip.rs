#![allow(dead_code, unused_variables, unused_imports)]
//! Debug univariate-skip transcript operations step by step
//!
//! Usage: cargo run -p gnark-transpiler --bin debug_uniskip --release

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
    println!("=== Rust Transcript Debug (Univariate Skip Detail) ===\n");

    let proof = load_proof().expect("Failed to load proof");
    let io_device = load_io_device().expect("Failed to load io_device");

    // Create transcript
    let mut transcript: PoseidonTranscript<Fr, FrParams> = Transcript::new(b"Jolt");

    // === PREAMBLE ===
    transcript.append_u64(io_device.memory_layout.max_input_size);
    transcript.append_u64(io_device.memory_layout.max_output_size);
    transcript.append_u64(io_device.memory_layout.memory_size);
    transcript.append_bytes(&io_device.inputs);
    transcript.append_bytes(&io_device.outputs);
    transcript.append_u64(io_device.panic as u64);
    transcript.append_u64(proof.ram_K as u64);
    transcript.append_u64(proof.trace_length as u64);

    // === COMMITMENTS ===
    for commitment in proof.commitments.iter() {
        transcript.append_serializable(commitment);
    }

    // === TAU CHALLENGES ===
    let num_rows_bits = (proof.trace_length.trailing_zeros() as usize) + 2;
    for _ in 0..num_rows_bits {
        let _: u128 = transcript.challenge_u128();
    }

    println!("State after all tau: {}", fr_to_string(&state_to_fr(&transcript.state)));
    println!();

    // === UNIVARIATE SKIP - DETAILED ===
    println!("=== UNIVARIATE SKIP DETAILED ===");

    // append_message("UncompressedUniPoly_begin")
    transcript.append_message(b"UncompressedUniPoly_begin");
    println!("After append_message(\"UncompressedUniPoly_begin\"):");
    println!("  state = {}", fr_to_string(&state_to_fr(&transcript.state)));

    // Print the coefficients and their byte-reversed values
    println!("\nCoefficients (first 5):");
    for (i, coeff) in proof.stage1_uni_skip_first_round_proof.uni_poly.coeffs.iter().take(5).enumerate() {
        println!("  coeff[{}] = {}", i, fr_to_string(coeff));

        // Compute byte-reversed value (what gets hashed)
        let mut buf = vec![];
        coeff.serialize_uncompressed(&mut buf).unwrap();
        println!("    serialized (LE): {:?}", &buf[..8]);
        buf.reverse();
        println!("    reversed: {:?}", &buf[..8]);
        let reversed_fr = Fr::from_le_bytes_mod_order(&buf);
        println!("    from_le_bytes_mod_order: {}", fr_to_string(&reversed_fr));
    }

    // Append each coefficient
    for (i, coeff) in proof.stage1_uni_skip_first_round_proof.uni_poly.coeffs.iter().enumerate() {
        transcript.append_scalar(coeff);
        if i < 3 {
            println!("\nAfter append_scalar(coeff[{}]):", i);
            println!("  state = {}", fr_to_string(&state_to_fr(&transcript.state)));
        }
    }

    println!("\n...");
    println!("\nTotal coefficients appended: {}", proof.stage1_uni_skip_first_round_proof.uni_poly.coeffs.len());
    println!("State after all coefficients (before end message): {}", fr_to_string(&state_to_fr(&transcript.state)));

    // append_message("UncompressedUniPoly_end")
    transcript.append_message(b"UncompressedUniPoly_end");
    println!("\nAfter append_message(\"UncompressedUniPoly_end\"):");
    println!("  state = {}", fr_to_string(&state_to_fr(&transcript.state)));

    // Before challenge_scalar, we need to see what the raw hash output is
    // challenge_scalar does: challenge_bytes32 (which hashes state, n_rounds, 0) -> take 16 bytes -> reverse -> from_bytes
    // The state before challenge is what we just printed: 1870895976728698847681700511596816793827144742923278305181923343218103827913
    println!("\nState before challenge_scalar: {}", fr_to_string(&state_to_fr(&transcript.state)));

    // challenge_scalar
    let r0: Fr = transcript.challenge_scalar();
    println!("r0 = challenge_scalar() = {}", fr_to_string(&r0));
    println!("State after r0: {}", fr_to_string(&state_to_fr(&transcript.state)));

    // batching_coeff
    let batching: Fr = transcript.challenge_scalar();
    println!("\nbatching_coeff = {}", fr_to_string(&batching));
    println!("State after batching_coeff: {}", fr_to_string(&state_to_fr(&transcript.state)));

    // === SUMCHECK ROUND 0 ===
    println!("\n=== SUMCHECK ROUND 0 ===");

    // First we need to get the sumcheck polys
    let sumcheck_polys = &proof.stage1_sumcheck_proof.compressed_polys;
    println!("Number of sumcheck rounds: {}", sumcheck_polys.len());

    // Append round 0 polynomial
    // In Rust, CompressedUniPoly::append_to_transcript does append_message(begin/end) + append_vector
    let poly0 = &sumcheck_polys[0];
    println!("Round 0 coeffs:");
    for (i, coeff) in poly0.coeffs_except_linear_term.iter().enumerate() {
        println!("  coeff[{}] = {}", i, fr_to_string(coeff));
    }

    // Append the polynomial to transcript (matches real verifier)
    // CompressedUniPoly appends: begin, coeffs (no vector wrapper for compressed), end
    transcript.append_message(b"UniPoly_begin");
    for coeff in &poly0.coeffs_except_linear_term {
        transcript.append_scalar(coeff);
    }
    transcript.append_message(b"UniPoly_end");

    println!("State after round 0 poly: {}", fr_to_string(&state_to_fr(&transcript.state)));
    println!("n_rounds after round 0 poly: {}", transcript.n_rounds);

    // Count n_rounds manually:
    // init: 0
    // preamble: max_input(1), max_output(2), memory(3), inputs(4), outputs(5), panic(6), ram_k(7), trace_len(8)
    // commitments: 41 × 1 = 41 (9..49)
    // tau: 12 (50..61)
    // r_spartan: 1 (62)
    // uni_skip_begin: 1 (63)
    // uni_skip_coeffs: 28 (64..91)
    // uni_skip_end: 1 (92)
    // r0: 1 (93)
    // batching: 1 (94)
    // sumcheck_begin: 1 (95)
    // sumcheck_coeffs: 3 (96..98)
    // sumcheck_end: 1 (99)
    // → n_rounds antes del challenge = 98 (real value from transcript)
    let n_rounds_actual = transcript.n_rounds as u64;
    let n_rounds_estimate = n_rounds_actual;
    println!("n_rounds before sumcheck challenge (estimated): {}", n_rounds_estimate);

    // Compute what the hash should be
    use light_poseidon::{Poseidon, PoseidonHasher};
    let state_fr = state_to_fr(&transcript.state);
    let n_rounds_fr = Fr::from(n_rounds_estimate);
    let zero_fr = Fr::from(0u64);

    let mut poseidon_hasher = Poseidon::<Fr>::new_circom(3).unwrap();
    let hash_result = poseidon_hasher.hash(&[state_fr, n_rounds_fr, zero_fr]).unwrap();
    println!("Hash result = poseidon(state, {}, 0) = {}", n_rounds_estimate, fr_to_string(&hash_result));

    // Now serialize and see what bytes we get
    let mut hash_bytes = [0u8; 32];
    hash_result.serialize_uncompressed(&mut hash_bytes[..]).unwrap();
    println!("Hash bytes (LE first 16): {:?}", &hash_bytes[..16]);

    // First sumcheck challenge (uses challenge_scalar_optimized -> challenge_u128 -> MontU128Challenge)
    let r_sumcheck_0: <Fr as jolt_core::field::JoltField>::Challenge = transcript.challenge_scalar_optimized::<Fr>();
    // MontU128Challenge needs to be converted to Fr for display
    let r_sumcheck_0_fr: Fr = r_sumcheck_0.into();
    println!("r_sumcheck[0] = {}", fr_to_string(&r_sumcheck_0_fr));
    println!("State after r_sumcheck[0]: {}", fr_to_string(&state_to_fr(&transcript.state)));

    // Now let's manually replicate what challenge_scalar_optimized does
    println!("\n=== Manual replication of challenge_scalar_optimized ===");

    // Use the hash bytes we computed above
    let mut buf16: [u8; 16] = hash_bytes[..16].try_into().unwrap();
    println!("First 16 bytes of hash (LE): {:?}", buf16);

    // challenge_u128 does: buf.reverse() then from_be_bytes
    buf16.reverse();
    println!("After reverse: {:?}", buf16);

    let u128_val = u128::from_be_bytes(buf16);
    println!("u128 value: {}", u128_val);

    // MontU128Challenge::new applies 125-bit mask
    let masked = u128_val & (u128::MAX >> 3);
    println!("After 125-bit mask: {}", masked);

    // Create MontU128Challenge and convert to Fr
    use jolt_core::field::challenge::mont_ark_u128::MontU128Challenge;
    let challenge = MontU128Challenge::<Fr>::new(u128_val);
    let challenge_fr: Fr = challenge.into();
    println!("MontU128Challenge({}).into() = {}", u128_val, fr_to_string(&challenge_fr));

    println!("\nActual r_sumcheck[0] from transcript: {}", fr_to_string(&r_sumcheck_0_fr));
    println!("Match? {}", challenge_fr == r_sumcheck_0_fr);
}
