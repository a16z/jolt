#![allow(dead_code, unused_variables, unused_imports)]
//! Test ByteReverse and Truncate128Reverse hints against Rust behavior
//!
//! Usage: cargo run --bin test_hints --release

use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use num_bigint::BigUint;

fn fr_to_decimal(f: &Fr) -> String {
    let mut bytes = vec![];
    f.serialize_uncompressed(&mut bytes).unwrap();
    BigUint::from_bytes_le(&bytes).to_string()
}

fn fr_from_decimal(s: &str) -> Fr {
    let bytes = BigUint::parse_bytes(s.as_bytes(), 10).unwrap().to_bytes_le();
    let mut padded = [0u8; 32];
    let len = bytes.len().min(32);
    padded[..len].copy_from_slice(&bytes[..len]);
    Fr::from_le_bytes_mod_order(&padded)
}

/// This mirrors PoseidonTranscript::append_scalar behavior:
/// serialize_uncompressed(LE) -> reverse -> from_le_bytes_mod_order
fn byte_reverse(x: &Fr) -> Fr {
    // Serialize to LE bytes
    let mut le_bytes = vec![];
    x.serialize_uncompressed(&mut le_bytes).unwrap();

    // Reverse
    le_bytes.reverse();

    // Interpret as LE (from_le_bytes_mod_order)
    Fr::from_le_bytes_mod_order(&le_bytes)
}

/// This mirrors challenge_scalar_optimized behavior:
/// hash -> serialize(LE) -> take 16 bytes -> reverse -> from_be_bytes -> mask to 125 bits -> multiply by 2^128
fn truncate_128_reverse(x: &Fr) -> Fr {
    // Serialize to LE bytes
    let mut le_bytes = vec![];
    x.serialize_uncompressed(&mut le_bytes).unwrap();

    // Take first 16 bytes (low 128 bits in LE)
    let le16: Vec<u8> = le_bytes[..16].to_vec();

    // Interpret as u128 (LE) and mask to 125 bits
    let mut value: u128 = 0;
    for (i, &b) in le16.iter().enumerate() {
        value |= (b as u128) << (8 * i);
    }
    let mask_125: u128 = (1u128 << 125) - 1;
    let value_masked = value & mask_125;

    // Multiply by 2^128 (shift left by 128 bits)
    // This is the MontU128Challenge representation
    let two_pow_128 = BigUint::from(1u128) << 128;
    let result: BigUint = BigUint::from(value_masked) * two_pow_128;

    let result_bytes = result.to_bytes_le();
    let mut padded = [0u8; 32];
    let len = result_bytes.len().min(32);
    padded[..len].copy_from_slice(&result_bytes[..len]);
    Fr::from_le_bytes_mod_order(&padded)
}

fn main() {
    println!("=== Testing ByteReverse ===");

    // Test with a simple value
    let test_val = fr_from_decimal("12345678901234567890");
    println!("Input: {}", fr_to_decimal(&test_val));
    let reversed = byte_reverse(&test_val);
    println!("ByteReverse: {}", fr_to_decimal(&reversed));

    // Test with one of the uni-skip coefficients from fib
    let coeff = fr_from_decimal("4174971355090394500367463648627069291783169100582575189370224510836072005451");
    println!("\nUni-skip coeff: {}", fr_to_decimal(&coeff));
    let coeff_reversed = byte_reverse(&coeff);
    println!("ByteReverse: {}", fr_to_decimal(&coeff_reversed));

    println!("\n=== Testing Truncate128Reverse ===");

    // Test with the result after a challenge hash
    let hash_result = fr_from_decimal("15347626584322578373199561629687799511864240952792169390776318568321796642585");
    println!("Hash result: {}", fr_to_decimal(&hash_result));
    let truncated = truncate_128_reverse(&hash_result);
    println!("Truncate128Reverse: {}", fr_to_decimal(&truncated));

    // Also print the intermediate values for debugging
    let mut le_bytes = vec![];
    hash_result.serialize_uncompressed(&mut le_bytes).unwrap();
    println!("  LE bytes (first 16): {:?}", &le_bytes[..16]);

    let mut value: u128 = 0;
    for (i, &b) in le_bytes[..16].iter().enumerate() {
        value |= (b as u128) << (8 * i);
    }
    println!("  As u128: {}", value);
    let mask_125: u128 = (1u128 << 125) - 1;
    let value_masked = value & mask_125;
    println!("  Masked to 125 bits: {}", value_masked);
    println!("  * 2^128: {}", BigUint::from(value_masked) * (BigUint::from(1u128) << 128));
}
