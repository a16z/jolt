#![cfg_attr(feature = "guest", no_std)]

#[jolt::provable(heap_size = 32768, max_trace_length = 65536)]
fn advice_consumer(
    public_sum: u64,
    trusted: jolt::TrustedAdvice<u64>,
    untrusted: jolt::UntrustedAdvice<u64>,
) -> u64 {
    let trusted = *trusted;
    let untrusted = *untrusted;
    assert_eq!(trusted + untrusted, public_sum);
    trusted.wrapping_mul(3).wrapping_add(untrusted)
}
