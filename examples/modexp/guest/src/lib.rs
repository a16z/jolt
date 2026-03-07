#[jolt::provable(heap_size = 65536, max_trace_length = 4194304)]
fn modexp(base: Vec<u8>, exp: Vec<u8>, modulus: Vec<u8>, num_iters: u32) -> Vec<u8> {
    use num_bigint::BigUint;

    let exp = BigUint::from_bytes_be(&exp);
    let modulus = BigUint::from_bytes_be(&modulus);
    let mut result = BigUint::from_bytes_be(&base);

    for _ in 0..num_iters {
        result = result.modpow(&exp, &modulus);
    }

    result.to_bytes_be()
}
