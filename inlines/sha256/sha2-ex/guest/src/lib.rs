#![cfg_attr(feature = "guest", no_std)]


use inline::Sha256;

#[jolt::provable(memory_size = 10240, max_trace_length = 65536)]
fn sha2(input: &[u8]) -> [u8; 32] {
    // Use Jolt's optimized SHA256 implementation
    Sha256::digest(input)
}



// #[jolt::provable]
// fn blake2_chain_inline(input: [u8; 32], num_iters: u32) -> [u8; 32] {
//     // Create a 1024 * 16 input (16,384 bytes)
//     let mut large_input = [0u8; 1024 * 2];
    
//     // Fill the large input by repeating the input pattern
//     for i in 0..(1024 * 2) {
//         large_input[i] = input[i % 32];
//     }
    
//     // Call digest on the large input
//     let hash_result = inline_hash::blake2::Blake2b::digest(black_box(&large_input));
    
//     // // Return the first half of the result (first 32 bytes)
//     large_input[0..32].try_into().unwrap()
// }






// #[jolt::provable]
// fn blake2_chain_inline(input: [u8; 32], num_iters: u32) -> u64 {
//     // Create hash = input repeated 32 times to fill 1024 bytes (32 * 32 = 1024)
//     let input = b"abcabcabcabccabkshfswisjsjfkisiwwwqqq88wmm88scsc11azfiocssqkk118csscsakchnlhoihwowhd1wiu120u3e12312bnjkbnkaqqqqqou9u092312111qww";
//     let mut message = [0u64; 16];
//         for i in 0..16 {
//             message[i] = u64::from_le_bytes(
//                 input[i * 8..(i + 1) * 8].try_into().unwrap()
//             );
//         }
    
//     // Blake2b initialization vector
//     let mut h: [u64; 8] = [
//         0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
//         0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
//     ];

//     let mut counter = 0;
    
//     // XOR h[0] with parameter block: 0x01010000 ^ (kk << 8) ^ nn
//     // where kk=0 (unkeyed) and nn=output_len
//     h[0] ^= 0x01010000 ^ (64 as u64);
//     for _ in 0..num_iters {
//         counter += 1;
//         // unsafe{
//         //     inline_hash::blake2::blake2b_compress(h.as_mut_ptr(), message.as_ptr(), 128, 1);
//         // }
//         inline_hash::blake2::Blake2b::digest(input);
//     }

//     // unsafe{
//     //     inline_hash::blake2::blake2b_compress(h.as_mut_ptr(), message.as_ptr(), 128, 1);
//     // }
//     // let mut hash_64 = inline_hash::blake2::Blake2b::digest(input);
//     // for _ in 0..num_iters {
//     //     hash_64 = inline_hash::blake2::Blake2b::digest(input);
//     // }
//     // for _ in 0..num_iters {
//     //     // Process 1024*1024 bytes in 1024 chunks of 1024 bytes each
//     //     let mut hasher = inline_hash::blake2::Blake2b::new(64);
        
//     //     for _ in 0..1024 { // Process 1024 chunks
//     //         let mut chunk = [0u8; 1024]; // 1KB chunk - fits on stack
//     //         for i in 0..16 { // Fill chunk: 1024 / 64 = 16 repetitions
//     //             chunk[i*64..(i+1)*64].copy_from_slice(&hash_64);
//     //         }
//     //         hasher.update(&chunk);
//     //     }
        
//     //     hash_64 = hasher.finalize();
//     // }
    
//     // hash_64[0..32].try_into().unwrap()
//     return counter;
// }

// #[jolt::provable]
// fn blake2_chain_inline(input: [u8; 32], num_iters: u32) -> [u8; 32] {
//     // Create hash = input repeated 32 times to fill 1024 bytes (32 * 32 = 1024)
//     let mut data_1024 = [0u8; 1024];
//     for i in 0..32 {
//         data_1024[i*32..(i+1)*32].copy_from_slice(&input);
//     }
    
//     let mut hash_64 = inline_hash::blake2::Blake2b::digest(&data_1024);
    
//     for _ in 0..num_iters {
//         // Process 1024*1024 bytes in 1024 chunks of 1024 bytes each
//         let mut hasher = inline_hash::blake2::Blake2b::new(64);
        
//         for _ in 0..1024 { // Process 1024 chunks
//             let mut chunk = [0u8; 1024]; // 1KB chunk - fits on stack
//             for i in 0..16 { // Fill chunk: 1024 / 64 = 16 repetitions
//                 chunk[i*64..(i+1)*64].copy_from_slice(&hash_64);
//             }
//             hasher.update(&chunk);
//         }
        
//         hash_64 = hasher.finalize();
//     }
    
//     hash_64[0..32].try_into().unwrap()
// }