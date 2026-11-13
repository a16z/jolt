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

/// Configured for 12 bytes of untrusted advice, which the layout rounds up to
/// 16. Returns the advice's high doubleword, so bytes 12..16 reach the output.
#[jolt::provable(
    heap_size = 32768,
    max_trace_length = 65536,
    max_untrusted_advice_size = 12
)]
fn rounded_advice_consumer(untrusted: jolt::UntrustedAdvice<[u8; 16]>) -> u64 {
    let bytes = *untrusted;
    let mut high = [0u8; 8];
    high.copy_from_slice(&bytes[8..]);
    u64::from_le_bytes(high)
}
