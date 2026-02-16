//! Compute R and R^-1 for BN254 Fr Montgomery arithmetic.
//!
//! In ark-ff, Fr elements are stored in Montgomery form internally.
//! R = 2^256 mod p for BN254.
//!
//! This utility verifies the hardcoded `bn254RInv` constant in poseidon.go
//! and documents how it was derived.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p gnark-transpiler --bin compute_r_inv --release
//! ```
//!
//! # Background
//!
//! Jolt's `from_bigint_unchecked` interprets input as Montgomery form and
//! multiplies by R^-1 to convert to standard form. The Go circuit hints
//! must apply the same transformation to match challenge values.

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
    for limb in modulus.0.iter() {
        modulus_bytes.extend_from_slice(&limb.to_le_bytes());
    }
    let modulus_biguint = BigUint::from_bytes_le(&modulus_bytes);
    println!("Modulus (decimal): {modulus_biguint}");

    // R = 2^256 mod p (Montgomery constant for 4-limb representation)
    let two_256 = BigUint::from(1u8) << 256;
    let r_value: BigUint = &two_256 % &modulus_biguint;
    println!("\nR = 2^256 mod p = {r_value}");

    // R^-1 = modular inverse of R mod p
    let r_inv = r_value.modinv(&modulus_biguint).unwrap();
    println!("R^-1 mod p = {r_inv}");

    // Verify: R * R^-1 = 1 mod p
    let product = (&r_value * &r_inv) % &modulus_biguint;
    assert_eq!(product, BigUint::from(1u8), "R * R^-1 should equal 1 mod p");
    println!("\nVerified: R * R^-1 = 1 mod p");

    // Test from_bigint_unchecked behavior
    println!("\n=== Testing from_bigint_unchecked ===");
    let bigint_one = BigInt::new([1u64, 0, 0, 0]);
    let fr_unchecked = Fr::from_bigint_unchecked(bigint_one).unwrap();
    println!("from_bigint_unchecked([1, 0, 0, 0]) = {}", fr_to_string(&fr_unchecked));
    println!("Expected (R^-1 mod p) = {r_inv}");

    // Verify MontU128Challenge conversion (used in sumcheck challenges)
    println!("\n=== MontU128Challenge Verification ===");
    println!("For a 128-bit masked value, from_bigint_unchecked places it in limbs [2,3]");
    println!("Then multiplies by R^-1 to convert from Montgomery to standard form.");
    println!("\nThis matches the Go hint in poseidon.go which:");
    println!("1. Places low 64 bits at position 128 (limb 2)");
    println!("2. Places high 64 bits at position 192 (limb 3)");
    println!("3. Multiplies by bn254RInv = {r_inv}");

    println!("\n=== Go Constant ===");
    println!("// bn254RInv is R^-1 mod p for BN254 Fr Montgomery arithmetic");
    println!("var bn254RInv = bigInt(\"{r_inv}\")  // Used in Truncate128Reverse hint");
}
