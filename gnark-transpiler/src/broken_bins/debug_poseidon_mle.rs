#![allow(dead_code, unused_variables, unused_imports)]
//! Debug PoseidonAstTranscript vs PoseidonTranscript
//!
//! Compares n_rounds progression between real and symbolic transcripts

use ark_bn254::Fr;
use ark_serialize::CanonicalSerialize;
use jolt_core::transcripts::{FrParams, PoseidonTranscript, Transcript};
use gnark_transpiler::poseidon::PoseidonAstTranscript;
use num_bigint::BigUint;

fn fr_to_string(f: &Fr) -> String {
    let mut bytes = vec![];
    f.serialize_uncompressed(&mut bytes).unwrap();
    BigUint::from_bytes_le(&bytes).to_string()
}

fn main() {
    println!("=== Debug PoseidonAstTranscript vs PoseidonTranscript ===\n");

    // Create both transcripts with same label
    let mut real: PoseidonTranscript<Fr, FrParams> = Transcript::new(b"Jolt");
    let mut symbolic: PoseidonAstTranscript = Transcript::new(b"Jolt");

    println!("After new(b\"Jolt\"):");
    println!("  Real n_rounds: (internal, not accessible)");
    println!("  Symbolic n_rounds: {}", symbolic.n_rounds());

    // Step 1: append_u64
    real.append_u64(4096);
    symbolic.append_u64(4096);
    println!("\nAfter append_u64(4096):");
    println!("  Symbolic n_rounds: {}", symbolic.n_rounds());

    // Step 2: append_u64
    real.append_u64(4096);
    symbolic.append_u64(4096);
    println!("\nAfter append_u64(4096) x2:");
    println!("  Symbolic n_rounds: {}", symbolic.n_rounds());

    // Step 3: append_u64(memory_size)
    real.append_u64(32768);
    symbolic.append_u64(32768);
    println!("\nAfter append_u64(32768):");
    println!("  Symbolic n_rounds: {}", symbolic.n_rounds());

    // Step 4: append_bytes(inputs)
    let inputs = vec![50u8];
    real.append_bytes(&inputs);
    symbolic.append_bytes(&inputs);
    println!("\nAfter append_bytes([50]):");
    println!("  Symbolic n_rounds: {}", symbolic.n_rounds());

    // Step 5: append_bytes(outputs)
    let outputs = vec![225u8, 242, 204, 241, 46];
    real.append_bytes(&outputs);
    symbolic.append_bytes(&outputs);
    println!("\nAfter append_bytes(outputs):");
    println!("  Symbolic n_rounds: {}", symbolic.n_rounds());

    // Step 6-8: more append_u64
    real.append_u64(0); // panic
    symbolic.append_u64(0);
    real.append_u64(8192); // ram_k
    symbolic.append_u64(8192);
    real.append_u64(1024); // trace_length
    symbolic.append_u64(1024);
    println!("\nAfter preamble complete:");
    println!("  Symbolic n_rounds: {}", symbolic.n_rounds());

    // Now get a challenge from real
    let real_challenge: Fr = real.challenge_scalar();
    println!("\nReal challenge after preamble: {}", fr_to_string(&real_challenge));

    // The symbolic transcript creates a Poseidon AST node
    // We need to check what n_rounds value it uses
    let symbolic_challenge = symbolic.challenge_scalar::<Fr>();
    println!("Symbolic n_rounds after challenge: {}", symbolic.n_rounds());

    println!("\n=== Key Question ===");
    println!("Does the symbolic transcript use the same n_rounds as the real one?");
    println!("Real transcript internal state is not directly accessible,");
    println!("but we can infer it from the challenge values in Gnark.");
}
