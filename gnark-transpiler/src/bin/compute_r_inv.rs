#![allow(dead_code, unused_variables, unused_imports)]
//! Compute R and R^-1 for BN254 Fr Montgomery arithmetic
//!
//! In ark-ff, Fr elements are stored in Montgomery form internally.
//! R = 2^256 mod p for BN254
//!
//! Usage: cargo run -p gnark-transpiler --bin compute_r_inv --release

use ark_bn254::Fr;
use ark_ff::{BigInt, PrimeField};
use ark_serialize::CanonicalSerialize;
use num_bigint::BigUint;

fn fr_to_string(f: &Fr) -> String {
    let mut bytes = vec![];
    f.serialize_uncompressed(&mut bytes).unwrap();
    BigUint::from_bytes_le(&bytes).to_string()
}

fn main() {
    println!("=== BN254 Fr Montgomery Parameters ===\n");

    // BN254 Fr modulus
    let modulus = Fr::MODULUS;
    println!("Modulus (p): {:?}", modulus.0);

    // Convert modulus to BigUint for display
    let mut modulus_bytes = vec![];
    // BigInt stores in LE limbs, each limb is 64 bits
    for limb in modulus.0.iter() {
        modulus_bytes.extend_from_slice(&limb.to_le_bytes());
    }
    let modulus_biguint = BigUint::from_bytes_le(&modulus_bytes);
    println!("Modulus (decimal): {}", modulus_biguint);

    // R = 2^256 mod p
    // In ark-ff, R is stored as MODULUS_MINUS_ONE_DIV_TWO related... let me compute it differently

    // The Montgomery R is 2^(64*4) = 2^256 for 4-limb representation
    let two_256 = BigUint::from(1u8) << 256;
    let r_value: BigUint = &two_256 % &modulus_biguint;
    println!("\nR = 2^256 mod p = {}", r_value);

    // Verify: Fr::from(1).into_bigint() should be stored as 1 in ark's representation
    // Wait, I saw earlier that Fr(1).into_bigint() = [1, 0, 0, 0], which is 1, not R
    // This means ark-ff's into_bigint() returns the STANDARD form, not Montgomery!

    // Let me re-check what from_bigint_unchecked does
    println!("\n=== Testing from_bigint_unchecked ===");

    // Test: what does from_bigint_unchecked([1, 0, 0, 0]) give?
    let bigint_one = BigInt::new([1u64, 0, 0, 0]);
    let fr_unchecked = Fr::from_bigint_unchecked(bigint_one).unwrap();
    println!("from_bigint_unchecked([1, 0, 0, 0]) = {}", fr_to_string(&fr_unchecked));

    // If it's NOT 1, then from_bigint_unchecked does some conversion
    // Let me check what value it gives and reverse-engineer the relationship

    // Compute 1 * R^-1 mod p to see if that's what we get
    // R^-1 = inverse of R mod p
    let r_inv = r_value.modinv(&modulus_biguint).unwrap();
    println!("\nR^-1 mod p = {}", r_inv);

    // So if from_bigint_unchecked(x) = x * R^-1 mod p, then:
    // from_bigint_unchecked([1,0,0,0]) should equal R^-1
    println!("\nIf from_bigint_unchecked([1,0,0,0]) == R^-1, then from_bigint_unchecked multiplies by R^-1");

    // Now let's verify the MontU128Challenge conversion
    println!("\n=== Verifying MontU128Challenge conversion ===");

    // From the debug: masked value was 26521407449146688864712768543667632405
    // Result was 6546053814691896674008619906195315247069748678507320726103401577775255757342

    let masked: u128 = 26521407449146688864712768543667632405;
    let low = masked as u64;
    let high = (masked >> 64) as u64;
    println!("masked = {}", masked);
    println!("low = {}, high = {}", low, high);

    // BigInt value is low * 2^128 + high * 2^192
    let bigint_value = BigUint::from(low) * (BigUint::from(1u8) << 128)
        + BigUint::from(high) * (BigUint::from(1u8) << 192);
    println!("BigInt([0,0,low,high]) as integer = {}", bigint_value);

    // Expected result after from_bigint_unchecked
    let expected_result = BigUint::parse_bytes(b"6546053814691896674008619906195315247069748678507320726103401577775255757342", 10).unwrap();

    // Check: bigint_value * R^-1 mod p should equal expected_result
    let computed = (&bigint_value * &r_inv) % &modulus_biguint;
    println!("\nbigint_value * R^-1 mod p = {}", computed);
    println!("Expected result = {}", expected_result);
    println!("Match? {}", computed == expected_result);
}
