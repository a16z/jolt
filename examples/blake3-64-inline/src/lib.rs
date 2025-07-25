#![cfg_attr(feature = "guest", no_std)]

use core::hint::black_box;
use jolt::blake3;

#[jolt::provable]
fn blake3_64_inline(input: [u8; 32], num_iters: u32) -> [u8; 32] {
    // Create hash = input repeated 32 times to fill 1024 bytes (32 * 32 = 1024)
    let input = black_box(b"sjfkisiwq8mc1afioc21spaqk118akcoiwvaahd1wiu3e112bnjkq97qou9u21qw");
    let mut message = [0u32; 16];
    for i in 0..16 {
        message[i] = black_box(u32::from_le_bytes(
            input[i * 4..(i + 1) * 4].try_into().unwrap(),
        ));
    }

    // Blake2b initialization vector
    let mut h: [u32; 8] = [0u32; 8];

    for _ in 0..black_box(num_iters) {
        unsafe {
            blake3::blake3_hash_64(black_box(h.as_mut_ptr()), black_box(message.as_ptr()));
        }
        // Prevent optimization of the hash state
        h = black_box(h);
    }

    // Prevent final optimization of the result
    black_box(h);

    // Convert [u32; 8] to [u8; 32]
    let mut result = [0u8; 32];
    for (i, &val) in h.iter().enumerate() {
        let bytes = val.to_le_bytes();
        result[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
    }
    return result;
}
