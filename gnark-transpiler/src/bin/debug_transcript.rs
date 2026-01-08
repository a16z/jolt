#![allow(dead_code, unused_variables, unused_imports)]
//! Debug Poseidon transcript step by step
//!
//! Usage: cargo run --bin debug_transcript --release

use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use light_poseidon::{Poseidon, PoseidonHasher};
use num_bigint::BigUint;

/// Compute Poseidon hash of 3 field elements and return as decimal string
fn poseidon_hash(a: &Fr, b: &Fr, c: &Fr) -> Fr {
    let mut hasher = Poseidon::<Fr>::new_circom(3).unwrap();
    hasher.hash(&[*a, *b, *c]).unwrap()
}

fn fr_to_decimal(f: &Fr) -> String {
    let mut bytes = vec![];
    f.serialize_uncompressed(&mut bytes).unwrap();
    BigUint::from_bytes_le(&bytes).to_string()
}

fn fr_from_bytes_le(bytes: &[u8]) -> Fr {
    let mut padded = [0u8; 32];
    let len = bytes.len().min(32);
    padded[..len].copy_from_slice(&bytes[..len]);
    Fr::from_le_bytes_mod_order(&padded)
}

/// Compute append_u64 transformation
fn append_u64_transform(x: u64) -> Fr {
    let mut packed = [0u8; 32];
    let be_bytes = x.to_be_bytes();
    packed[24..32].copy_from_slice(&be_bytes);
    Fr::from_le_bytes_mod_order(&packed)
}

fn main() {
    println!("=== Poseidon Transcript Debug ===");
    println!();

    // Initial state: hash(label, 0, 0) where label = "test"
    let label = b"test";
    let label_f = fr_from_bytes_le(label);
    println!("Label 'test' as Fr: {}", fr_to_decimal(&label_f));

    let state = poseidon_hash(&label_f, &Fr::from(0u64), &Fr::from(0u64));
    println!("Initial state = hash(label, 0, 0): {}", fr_to_decimal(&state));
    println!();

    // Track state through preamble operations
    let mut current_state = state;
    let mut n_rounds: u64 = 0;

    // Preamble values
    let max_input_size: u64 = 4096;
    let max_output_size: u64 = 4096;
    let memory_size: u64 = 32768;
    let inputs: Vec<u8> = vec![50];
    let outputs: Vec<u8> = vec![225, 242, 204, 241, 46];
    let panic: u64 = 0;
    let ram_k: u64 = 8192;
    let trace_length: u64 = 1024;

    // 1. append_u64(max_input_size)
    let transformed = append_u64_transform(max_input_size);
    println!("--- append_u64(max_input_size={}) ---", max_input_size);
    println!("  transformed value: {}", fr_to_decimal(&transformed));
    let new_state = poseidon_hash(&current_state, &Fr::from(n_rounds), &transformed);
    println!("  hash(state, {}, transformed): {}", n_rounds, fr_to_decimal(&new_state));
    current_state = new_state;
    n_rounds += 1;
    println!();

    // 2. append_u64(max_output_size)
    let transformed = append_u64_transform(max_output_size);
    println!("--- append_u64(max_output_size={}) ---", max_output_size);
    println!("  transformed value: {}", fr_to_decimal(&transformed));
    let new_state = poseidon_hash(&current_state, &Fr::from(n_rounds), &transformed);
    println!("  hash(state, {}, transformed): {}", n_rounds, fr_to_decimal(&new_state));
    current_state = new_state;
    n_rounds += 1;
    println!();

    // 3. append_u64(memory_size)
    let transformed = append_u64_transform(memory_size);
    println!("--- append_u64(memory_size={}) ---", memory_size);
    println!("  transformed value: {}", fr_to_decimal(&transformed));
    let new_state = poseidon_hash(&current_state, &Fr::from(n_rounds), &transformed);
    println!("  hash(state, {}, transformed): {}", n_rounds, fr_to_decimal(&new_state));
    current_state = new_state;
    n_rounds += 1;
    println!();

    // 4. append_bytes(inputs)
    println!("--- append_bytes(inputs={:?}) ---", inputs);
    let input_f = fr_from_bytes_le(&inputs);
    println!("  inputs as Fr: {}", fr_to_decimal(&input_f));
    let new_state = poseidon_hash(&current_state, &Fr::from(n_rounds), &input_f);
    println!("  hash(state, {}, input_f): {}", n_rounds, fr_to_decimal(&new_state));
    current_state = new_state;
    n_rounds += 1;
    println!();

    // 5. append_bytes(outputs)
    println!("--- append_bytes(outputs={:?}) ---", outputs);
    let output_f = fr_from_bytes_le(&outputs);
    println!("  outputs as Fr: {}", fr_to_decimal(&output_f));
    let new_state = poseidon_hash(&current_state, &Fr::from(n_rounds), &output_f);
    println!("  hash(state, {}, output_f): {}", n_rounds, fr_to_decimal(&new_state));
    current_state = new_state;
    n_rounds += 1;
    println!();

    // 6. append_u64(panic)
    let transformed = append_u64_transform(panic);
    println!("--- append_u64(panic={}) ---", panic);
    println!("  transformed value: {}", fr_to_decimal(&transformed));
    let new_state = poseidon_hash(&current_state, &Fr::from(n_rounds), &transformed);
    println!("  hash(state, {}, transformed): {}", n_rounds, fr_to_decimal(&new_state));
    current_state = new_state;
    n_rounds += 1;
    println!();

    // 7. append_u64(ram_k)
    let transformed = append_u64_transform(ram_k);
    println!("--- append_u64(ram_k={}) ---", ram_k);
    println!("  transformed value: {}", fr_to_decimal(&transformed));
    let new_state = poseidon_hash(&current_state, &Fr::from(n_rounds), &transformed);
    println!("  hash(state, {}, transformed): {}", n_rounds, fr_to_decimal(&new_state));
    current_state = new_state;
    n_rounds += 1;
    println!();

    // 8. append_u64(trace_length)
    let transformed = append_u64_transform(trace_length);
    println!("--- append_u64(trace_length={}) ---", trace_length);
    println!("  transformed value: {}", fr_to_decimal(&transformed));
    let new_state = poseidon_hash(&current_state, &Fr::from(n_rounds), &transformed);
    println!("  hash(state, {}, transformed): {}", n_rounds, fr_to_decimal(&new_state));
    current_state = new_state;
    n_rounds += 1;
    println!();

    println!("=== After preamble ===");
    println!("Final state: {}", fr_to_decimal(&current_state));
    println!("n_rounds: {}", n_rounds);
}
