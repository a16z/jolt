#![allow(dead_code, unused_variables, unused_imports)]
//! Debug the raw hash output for r0
//!
//! Usage: cargo run -p gnark-transpiler --bin debug_r0_hash --release

use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use light_poseidon::{Poseidon, PoseidonHasher};
use num_bigint::BigUint;

fn fr_to_string(f: &Fr) -> String {
    let mut bytes = vec![];
    f.serialize_uncompressed(&mut bytes).unwrap();
    BigUint::from_bytes_le(&bytes).to_string()
}

fn main() {
    println!("=== Debug r0 Hash ===\n");

    // State after end message from Rust - as a decimal string
    let state_decimal = BigUint::parse_bytes(b"1870895976728698847681700511596816793827144742923278305181923343218103827913", 10).unwrap();

    // The state in the transcript is stored as 32-byte LE representation of the Fr
    // When we serialize Fr, we get LE bytes. So state_bytes = serialize(state_fr)
    // When we want state_f for hashing, we do Fr::from_le_bytes_mod_order(state_bytes)
    // which should give us back the same Fr value (assuming it was < modulus)

    // Let's construct state_before_r0 properly:
    // The decimal represents the Fr value, so we serialize it to get the bytes
    let state_before_r0 = Fr::from(state_decimal.clone());
    println!("State as Fr (from decimal): {}", fr_to_string(&state_before_r0));

    // Verify the round-trip
    let mut state_bytes = [0u8; 32];
    state_before_r0.serialize_uncompressed(&mut state_bytes[..]).unwrap();
    println!("Serialized bytes (first 8): {:?}", &state_bytes[..8]);

    let state_roundtrip = Fr::from_le_bytes_mod_order(&state_bytes);
    println!("After from_le_bytes_mod_order: {}", fr_to_string(&state_roundtrip));

    assert_eq!(state_before_r0, state_roundtrip, "Round-trip should be identity");

    let n_rounds = Fr::from(91u64);
    println!("n_rounds: {}", fr_to_string(&n_rounds));

    let zero = Fr::from(0u64);

    // Compute the hash
    let mut poseidon = Poseidon::<Fr>::new_circom(3).unwrap();
    let r0_hash = poseidon.hash(&[state_before_r0, n_rounds, zero]).unwrap();
    println!("r0_hash = poseidon(state, 91, 0) = {}", fr_to_string(&r0_hash));

    // Serialize to bytes (this is what challenge_bytes32 does)
    let mut r0_bytes = [0u8; 32];
    r0_hash.serialize_uncompressed(&mut r0_bytes[..]).unwrap();
    println!("r0_hash serialized (LE): {:?}", &r0_bytes[..8]);

    // challenge_scalar_128_bits takes first 16 bytes, reverses, then from_le_bytes_mod_order
    let mut buf16 = [0u8; 16];
    buf16.copy_from_slice(&r0_bytes[..16]);
    println!("First 16 bytes (LE): {:?}", buf16);

    buf16.reverse();
    println!("After reverse: {:?}", buf16);

    // JF::from_bytes(&buf) = Fr::from_le_bytes_mod_order(&buf)
    // This interprets the reversed bytes as LITTLE ENDIAN
    let r0_fr = Fr::from_le_bytes_mod_order(&buf16);
    println!("\nr0 as Fr (from_le_bytes_mod_order): {}", fr_to_string(&r0_fr));

    // Also compute what from_be_bytes would give (for comparison)
    let r0_u128_be = u128::from_be_bytes(buf16);
    println!("As u128 (from_be_bytes, WRONG): {}", r0_u128_be);

    let r0_u128_le = u128::from_le_bytes(buf16);
    println!("As u128 (from_le_bytes): {}", r0_u128_le);

    // Compare with expected
    println!("\nExpected r0: 339287670347666308529122047999684515317");
}
