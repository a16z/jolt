#![cfg_attr(feature = "guest", no_std)]

use core::hint::black_box;
use jolt::blake3;

#[jolt::provable]
fn blake3_192_inline(input: [u8; 32], num_iters: u32) -> [u32; 8] {
    // Create hash = input repeated 32 times to fill 1024 bytes (32 * 32 = 1024)
    let input = black_box(b"sjfkisiwq8mc1afioc21spaqk118sjfkisiwq8mc1afioc21spaqk118akcoiwvaahd1wiu3e112bnjkq97qou9u21qwsjfkisiwq8mc1afioc21spaqk118akcoiwvaahd1wiu3e112bnjkq97qou9u21qwakcoiwvaahd1wiu3e112bnjkq97qou9u21qw");
    let mut message = [0u32; 48];
    for i in 0..48 {
        message[i] = black_box(u32::from_le_bytes(
            input[i * 4..(i + 1) * 4].try_into().unwrap(),
        ));
    }

    // Blake2b initialization vector
    let mut h: [u32; 8] = [0u32; 8];

    for _ in 0..black_box(num_iters) {
        unsafe {
            blake3::blake3_hash_192(black_box(h.as_mut_ptr()), black_box(message.as_ptr()));
        }
        // Prevent optimization of the hash state
        h = black_box(h);
    }

    // Prevent final optimization of the result
    black_box(h);
    return h;
}
