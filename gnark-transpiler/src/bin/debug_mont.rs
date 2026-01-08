#![allow(dead_code, unused_variables, unused_imports)]
//! Debug MontU128Challenge to Fr conversion
//!
//! Usage: cargo run -p gnark-transpiler --bin debug_mont --release

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
    println!("=== Debug MontU128Challenge to Fr conversion ===\n");

    // From debug_uniskip output:
    // Hash bytes (LE first 16): [172, 49, 16, 71, 251, 151, 70, 154, 99, 144, 28, 123, 66, 251, 188, 53]
    // r_sumcheck[0] = 6546053814691896674008619906195315247069748678507320726103401577775255757342

    let le_bytes: [u8; 16] = [172, 49, 16, 71, 251, 151, 70, 154, 99, 144, 28, 123, 66, 251, 188, 53];
    println!("LE bytes from hash: {:?}", le_bytes);

    // Rust does: buf.reverse() -> from_be_bytes() to get u128
    // Then masks and stores in MontU128Challenge
    let mut reversed = le_bytes;
    reversed.reverse();
    println!("After reverse: {:?}", reversed);

    let value_u128 = u128::from_be_bytes(reversed);
    println!("u128 from_be_bytes(reversed): {}", value_u128);

    let val_masked = value_u128 & (u128::MAX >> 3);
    println!("After 125-bit mask: {}", val_masked);

    let low = val_masked as u64;
    let high = (val_masked >> 64) as u64;
    println!("low (bits 0-63): {}", low);
    println!("high (bits 64-127): {}", high);

    // Create MontU128Challenge representation
    let limbs: [u64; 4] = [0, 0, low, high];
    println!("\nLimbs [0, 0, low, high]: {:?}", limbs);

    // Convert to Fr using from_bigint_unchecked (what MontU128Challenge does)
    let bigint = BigInt::new(limbs);
    let fr_from_unchecked = Fr::from_bigint_unchecked(bigint).unwrap();
    println!("\nFr::from_bigint_unchecked([0, 0, low, high]):");
    println!("  = {}", fr_to_string(&fr_from_unchecked));

    // What if we interpret [0, 0, low, high] as a regular integer and multiply by 2^128?
    let masked_big = BigUint::from(val_masked);
    let two_pow_128 = BigUint::from(1u128) << 128;
    let shifted: BigUint = &masked_big * &two_pow_128;
    println!("\nval_masked * 2^128 (standard interpretation):");
    println!("  = {}", shifted);

    // Create Fr from the shifted value (proper way)
    let shifted_bytes = shifted.to_bytes_le();
    let fr_shifted = Fr::from_le_bytes_mod_order(&shifted_bytes);
    println!("\nFr::from_le_bytes_mod_order(shifted):");
    println!("  = {}", fr_to_string(&fr_shifted));

    // Check if they're equal
    println!("\nAre they equal? {}", fr_from_unchecked == fr_shifted);

    // Let's also see what the BigInt value is as a regular integer
    // BigInt limbs are little-endian, so [0, 0, low, high] = low * 2^128 + high * 2^192
    let limb_value = BigUint::from(low) * (BigUint::from(1u128) << 128)
        + BigUint::from(high) * (BigUint::from(1u128) << 192);
    println!("\nBigInt([0,0,low,high]) as regular integer:");
    println!("  = {}", limb_value);

    // The key insight: from_bigint_unchecked treats the input as already in Montgomery form
    // To convert FROM Montgomery to standard, we need to multiply by R^-1
    // But ark_ff's implementation might do something different...

    // Let's check what the internal representation of a simple Fr looks like
    let fr_one = Fr::from(1u64);
    println!("\n=== Sanity check: Fr::from(1) ===");
    println!("Fr(1) = {}", fr_to_string(&fr_one));

    // Get its BigInt representation (this is the Montgomery form)
    let one_bigint = fr_one.into_bigint();
    println!("BigInt of Fr(1) (Montgomery form): {:?}", one_bigint.0);

    // So Fr stores values in Montgomery form internally
    // from_bigint_unchecked takes a Montgomery form and creates Fr without conversion
    // This means [0, 0, low, high] IS the Montgomery representation

    // To convert [0, 0, low, high] Montgomery to standard, multiply by R^-1
    // Or equivalently, from_bigint does the conversion

    // Let's test: what if we use from_bigint (with check) instead?
    let fr_with_check = Fr::from_bigint(bigint);
    println!("\n=== Fr::from_bigint([0, 0, low, high]) ===");
    if let Some(fr) = fr_with_check {
        println!("Result: {}", fr_to_string(&fr));
    } else {
        println!("Returns None (value >= modulus)");
    }

    // The expected Rust value is: 6546053814691896674008619906195315247069748678507320726103401577775255757342
    let expected = BigUint::parse_bytes(b"6546053814691896674008619906195315247069748678507320726103401577775255757342", 10).unwrap();
    println!("\n=== Expected Rust value ===");
    println!("Expected: {}", expected);

    // Check which matches
    let from_unchecked_str = fr_to_string(&fr_from_unchecked);
    let from_shifted_str = fr_to_string(&fr_shifted);
    println!("\nMatches from_bigint_unchecked? {}", from_unchecked_str == expected.to_string());
    println!("Matches shifted * 2^128? {}", from_shifted_str == expected.to_string());

    // === More investigation ===
    println!("\n=== More investigation ===");

    // Test: what is R for bn254?
    let fr_2 = Fr::from(2u64);
    let bigint_2 = fr_2.into_bigint();
    println!("Fr(2).into_bigint(): {:?}", bigint_2.0);

    // Test: from_bigint vs from_bigint_unchecked
    let test_limbs: [u64; 4] = [100, 0, 0, 0];
    let fr_checked = Fr::from_bigint(BigInt::new(test_limbs)).unwrap();
    let fr_unchecked = Fr::from_bigint_unchecked(BigInt::new(test_limbs)).unwrap();
    println!("\nTest limbs [100, 0, 0, 0]:");
    println!("  from_bigint: {}", fr_to_string(&fr_checked));
    println!("  from_bigint_unchecked: {}", fr_to_string(&fr_unchecked));
    println!("  Are equal? {}", fr_checked == fr_unchecked);

    // So from_bigint and from_bigint_unchecked give different results!
    // from_bigint_unchecked treats input as Montgomery form
    // from_bigint treats input as standard form

    // What about the actual MontU128Challenge behavior?
    // It does: Fr::from_bigint_unchecked(BigInt::new([0, 0, low, high]))
    // So it treats [0, 0, low, high] as Montgomery representation

    // Let me compute what the standard value would be if [0, 0, low, high] is Montgomery
    // Montgomery form: a_mont = a * R mod p
    // To get a from a_mont: a = a_mont * R^-1 mod p

    // For BN254, R = 2^256 mod p
    // But wait, Fr(1).into_bigint() = [1, 0, 0, 0], not R!
    // This means ark_ff stores in STANDARD form, not Montgomery?

    // Let me check more carefully
    let fr_big = Fr::from(12345678901234567890u64);
    let bigint_big = fr_big.into_bigint();
    println!("\nFr(12345678901234567890).into_bigint(): {:?}", bigint_big.0);
    println!("Expected if standard: [12345678901234567890, 0, 0, 0]");

    // === Test the actual MontU128Challenge behavior ===
    println!("\n=== Test actual MontU128Challenge ===");

    // Replicate MontU128Challenge::new(value_u128)
    // Then .into() Fr
    use jolt_core::field::challenge::mont_ark_u128::MontU128Challenge;

    let challenge = MontU128Challenge::<Fr>::new(value_u128);
    let challenge_fr: Fr = challenge.into();
    println!("MontU128Challenge::new({}).into() = {}", value_u128, fr_to_string(&challenge_fr));
    println!("Expected: {}", expected);
    println!("Match? {}", fr_to_string(&challenge_fr) == expected.to_string());
}
