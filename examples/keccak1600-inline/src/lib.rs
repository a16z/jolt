#![cfg_attr(feature = "guest", no_std)]

use core::hint::black_box;
use jolt::keccak256;

#[jolt::provable]
fn keccak1600_inline(input: [u8; 32], num_iters: u32) -> [u8; 32] {
    // Create 136-byte input for full Keccak-256 rate absorption
    let mut input = black_box([0u8; 136]);
    let base_data = b"jsjfkisiwwwqqq88wmm88sci21ad032ndi321mxiouowquesc11azfiocsqskk118csscsakchnlhoihwowhd1wiu120u3e12312bnjkbnkaqqqqqou9u092312111qwwsadqqee";

    // Fill 136 bytes by repeating base data
    for i in 0..136 {
        input[i] = base_data[i % base_data.len()];
    }

    let mut hash_result = black_box([0u8; 32]);

    // Reset state for next iteration
    let mut state = black_box([0u64; 25]);

    for _ in 0..black_box(num_iters) {
        // Absorb full 136 bytes into state (Keccak-256 rate)
        for i in 0..17 {
            // 136 bytes = 17 u64 words
            let word = black_box(u64::from_le_bytes(
                input[i * 8..(i + 1) * 8].try_into().unwrap(),
            ));
            state[i] ^= black_box(word);
        }

        // Add padding (after 136 bytes of data)
        state[17] ^= black_box(0x01); // 0x01 in first byte after data
        state[24] ^= black_box(0x8000000000000000u64); // 0x80 in last byte of capacity

        // Apply Keccak-f[1600] permutation
        unsafe {
            keccak256::keccak_f(black_box(state.as_mut_ptr()));
        }
        state = black_box(state);
    }

    // Extract hash result
    for i in 0..4 {
        let word_bytes = black_box(state[i].to_le_bytes());
        hash_result[i * 8..(i + 1) * 8].copy_from_slice(&word_bytes);
    }
    hash_result = black_box(hash_result);
    // Prevent final optimization of the result
    black_box(input);
    return hash_result;
}
