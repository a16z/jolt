#![cfg_attr(feature = "guest", no_std)]

extern crate alloc;
use num_bigint::BigUint;

// Configurable bit lengths (256 bits = 32 bytes by default)
const BITLEN_BYTES: usize = 32;

#[jolt::provable(memory_size = 10240, max_trace_length = 4194304)]
fn modexp_chain(
    base: [u8; BITLEN_BYTES],
    exponent: [u8; BITLEN_BYTES],
    modulus: [u8; BITLEN_BYTES],
    num_iters: u32,
) -> [u8; BITLEN_BYTES] {
    let mut result = BigUint::from_bytes_be(&base);
    let exp = BigUint::from_bytes_be(&exponent);
    let modulus_uint = BigUint::from_bytes_be(&modulus);

    // Perform modexp num_iters times
    for _ in 0..num_iters {
        result = result.modpow(&exp, &modulus_uint);
    }

    // Convert result back to fixed-size array
    let result_bytes = result.to_bytes_be();
    let mut output = [0u8; BITLEN_BYTES];
    let len = result_bytes.len().min(BITLEN_BYTES);
    output[BITLEN_BYTES - len..].copy_from_slice(&result_bytes[result_bytes.len() - len..]);
    output
}
