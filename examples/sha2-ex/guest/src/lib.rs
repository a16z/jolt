#![cfg_attr(feature = "guest", no_std)]

#[jolt::provable(memory_size = 10240, max_trace_length = 65536)]
fn sha2(public_input: &[u8], _private_input: jolt::Private<u8>) -> [u8; 32] {
    jolt_inlines_sha2::Sha256::digest(public_input)
}

// , _private_input: jolt::Private<u8>
