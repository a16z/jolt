#![allow(dead_code, unused_variables, unused_imports)]
//! Compute exact message values for Go tests
//!
//! Usage: cargo run -p gnark-transpiler --bin compute_message_values --release

use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use num_bigint::BigUint;

fn fr_to_string(f: &Fr) -> String {
    let mut bytes = vec![];
    f.serialize_uncompressed(&mut bytes).unwrap();
    BigUint::from_bytes_le(&bytes).to_string()
}

fn bytes_to_le_fr(bytes: &[u8]) -> Fr {
    let mut padded = [0u8; 32];
    padded[..bytes.len().min(32)].copy_from_slice(&bytes[..bytes.len().min(32)]);
    Fr::from_le_bytes_mod_order(&padded)
}

fn main() {
    println!("=== Message Values for Go Tests ===\n");

    // UncompressedUniPoly_begin
    let msg_begin = b"UncompressedUniPoly_begin";
    println!("\"UncompressedUniPoly_begin\" ({} bytes)", msg_begin.len());
    println!("  bytes: {:?}", msg_begin);
    let fr_begin = bytes_to_le_fr(msg_begin);
    println!("  as Fr (LE): {}\n", fr_to_string(&fr_begin));

    // UncompressedUniPoly_end
    let msg_end = b"UncompressedUniPoly_end";
    println!("\"UncompressedUniPoly_end\" ({} bytes)", msg_end.len());
    println!("  bytes: {:?}", msg_end);
    let fr_end = bytes_to_le_fr(msg_end);
    println!("  as Fr (LE): {}\n", fr_to_string(&fr_end));

    // UniPoly_begin
    let msg_uni_begin = b"UniPoly_begin";
    println!("\"UniPoly_begin\" ({} bytes)", msg_uni_begin.len());
    println!("  bytes: {:?}", msg_uni_begin);
    let fr_uni_begin = bytes_to_le_fr(msg_uni_begin);
    println!("  as Fr (LE): {}\n", fr_to_string(&fr_uni_begin));

    // UniPoly_end
    let msg_uni_end = b"UniPoly_end";
    println!("\"UniPoly_end\" ({} bytes)", msg_uni_end.len());
    println!("  bytes: {:?}", msg_uni_end);
    let fr_uni_end = bytes_to_le_fr(msg_uni_end);
    println!("  as Fr (LE): {}\n", fr_to_string(&fr_uni_end));

    // Also compute what ByteReverse does to a sample scalar
    println!("=== ByteReverse Example ===\n");
    let sample = Fr::from(12345u64);
    let mut sample_bytes = vec![];
    sample.serialize_uncompressed(&mut sample_bytes).unwrap();
    println!("Fr::from(12345) serialized (LE): {:?}", sample_bytes);

    // Reverse the bytes
    sample_bytes.reverse();
    println!("After reverse (BE-ish): {:?}", sample_bytes);

    // Interpret as LE
    let reversed_fr = Fr::from_le_bytes_mod_order(&sample_bytes);
    println!("from_le_bytes_mod_order of reversed: {}", fr_to_string(&reversed_fr));
}
