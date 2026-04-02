#![cfg_attr(feature = "guest", no_std)]

#[jolt::provable(heap_size = 65536, max_trace_length = 67108864)]
fn modexp(base: [u8; 32], exp: [u8; 32], modulus: [u8; 32], num_iters: u32) -> [u8; 32] {
    use num_bigint::BigUint;

    let exp = BigUint::from_bytes_be(&exp);
    let modulus = BigUint::from_bytes_be(&modulus);
    let mut result = BigUint::from_bytes_be(&base);

    for _ in 0..num_iters {
        result = result.modpow(&exp, &modulus);
    }

    let bytes = result.to_bytes_be();
    let mut out = [0u8; 32];
    let start = 32 - bytes.len();
    out[start..].copy_from_slice(&bytes);
    out
}
