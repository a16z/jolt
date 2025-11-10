#![cfg_attr(feature = "guest", no_std)]

extern crate alloc;
use num_bigint::BigUint;
use num_traits::Zero;

// Configurable bit lengths for base, exponent, and modulus
// This can be changed at compile time to support different bit lengths:
// - 32 bytes = 256 bits (default, similar to EVM MODEXP)
// - 64 bytes = 512 bits
// - 128 bytes = 1024 bits
const BITLEN_BYTES: usize = 32;

#[jolt::provable(memory_size = 10240, max_trace_length = 4194304)]
fn modexp_chain(
    base: [u8; BITLEN_BYTES],
    exponent: [u8; BITLEN_BYTES],
    modulus: [u8; BITLEN_BYTES],
    num_iters: u32, // Configurable number of iterations
) -> [u8; BITLEN_BYTES] {
    let mut result = BigUint::from_bytes_be(&base);
    let exp = BigUint::from_bytes_be(&exponent);
    let modulus_uint = BigUint::from_bytes_be(&modulus);

    // Validate modulus is not zero to prevent division by zero
    assert!(!modulus_uint.is_zero(), "Modulus cannot be zero");

    // Perform modexp num_iters times, chaining the result
    for _ in 0..num_iters {
        result = result.modpow(&exp, &modulus_uint);
    }

    // Convert result back to fixed-size array, padding with zeros on the left
    let result_bytes = result.to_bytes_be();
    let mut output = [0u8; BITLEN_BYTES];
    let len = result_bytes.len().min(BITLEN_BYTES);
    let start_idx = BITLEN_BYTES - len;
    output[start_idx..].copy_from_slice(&result_bytes[(result_bytes.len() - len)..]);
    output
}
