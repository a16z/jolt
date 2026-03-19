#![cfg_attr(feature = "guest", no_std)]

#[jolt::provable(max_trace_length = 4194304)]
fn sha2_chain(
    input: jolt::UntrustedAdvice<[u8; 32]>,
    num_iters: jolt::UntrustedAdvice<u32>,
) -> [u8; 32] {
    let mut hash = *input;
    for _ in 0..*num_iters {
        hash = jolt_inlines_sha2::Sha256::digest(&hash);
    }
    hash
}
