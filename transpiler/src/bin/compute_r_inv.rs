//! Compute R and R^-1 for BN254 Fr Montgomery arithmetic.
//!
//! # Purpose
//!
//! One-time utility to derive the `bn254RInv` constant used in `poseidon.go`.
//! The constant is now hardcoded; this binary documents how it was computed.
//!
//! # What is R?
//!
//! R is the Montgomery constant: R = 2^256 mod p (for 4×64-bit limb representation).
//! In ark-ff, field elements are stored as `a * R mod p` internally.
//! R^-1 is needed to convert from Montgomery form back to standard form.
//!
//! # Why we need R^-1
//!
//! Jolt's `from_bigint_unchecked` interprets input as already in Montgomery form
//! and multiplies by R^-1. The Go circuit's `Truncate128Reverse` must apply the
//! same R^-1 multiplication to match Rust's challenge derivation.
//!
//! # Scope
//!
//! This utility is **BN254 Fr specific**:
//! - Hardcodes R = 2^256 (4 limbs × 64 bits)
//! - Uses `ark_bn254::Fr` directly
//!
//! For other fields, you'd need to adjust the limb count and import the
//! appropriate field type. Each field has its own R value.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p transpiler --bin compute_r_inv --release
//! ```

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
