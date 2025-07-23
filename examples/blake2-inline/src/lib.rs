#![cfg_attr(feature = "guest", no_std)]

use core::hint::black_box;
use jolt::blake2;

#[jolt::provable]
fn blake2_inline(input: [u8; 32], num_iters: u32) -> [u64; 8] {
    // Create hash = input repeated 32 times to fill 1024 bytes (32 * 32 = 1024)
    let input = black_box(b"abcabcabcabccabkshfswisjsjfkisiwwwqqq88wmm88scsc11azfiocssqkk118csscsakchnlhoihwowhd1wiu120u3e12312bnjkbnkaqqqqqou9u092312111qww");
    let mut message = [0u64; 16];
    for i in 0..16 {
        message[i] = black_box(u64::from_le_bytes(
            input[i * 8..(i + 1) * 8].try_into().unwrap(),
        ));
    }

    // Blake2b initialization vector
    let mut h: [u64; 8] = black_box([
        0x6a09e667f3bcc908,
        0xbb67ae8584caa73b,
        0x3c6ef372fe94f82b,
        0xa54ff53a5f1d36f1,
        0x510e527fade682d1,
        0x9b05688c2b3e6c1f,
        0x1f83d9abfb41bd6b,
        0x5be0cd19137e2179,
    ]);

    // XOR h[0] with parameter block: 0x01010000 ^ (kk << 8) ^ nn
    // where kk=0 (unkeyed) and nn=output_len
    h[0] ^= black_box(0x01010000 ^ (64 as u64));
    for _ in 0..black_box(num_iters) {
        unsafe {
            blake2::blake2b_compress(
                black_box(h.as_mut_ptr()),
                black_box(message.as_ptr()),
                black_box(128),
                black_box(1),
            );
        }
        // Prevent optimization of the hash state
        h = black_box(h);
        // blake2::Blake2b::digest(input);
    }

    // Prevent final optimization of the result
    black_box(h);
    return h;
}