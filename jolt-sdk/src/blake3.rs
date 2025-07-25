/// Blake3 initialization vector (IV)
#[rustfmt::skip]
const BLAKE3_IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

/// Calls the Blake3 compression custom instruction.
///
/// # Arguments
/// * `chaining_value` - Pointer to the 8-word (32-byte) Blake3 chaining value
/// * `message` - Pointer to the 16-word (64-byte) message block  
/// * `counter` - Block counter value
/// * `block_len` - Length of the block in bytes (typically 64)
/// * `flags` - Blake3 flags (chunk start/end, parent, root, etc.)
///
/// # Safety
/// - `chaining_value` must be a valid pointer to 32 bytes of readable and writable memory.
/// - `message` must be a valid pointer to 64 bytes of readable memory.
/// - Both pointers must be properly aligned for u32 access (4-byte alignment).
#[cfg(not(feature = "host"))]
pub unsafe fn blake3_compress(chaining_value: *mut u32, message: *const u32) {
    // Memory layout for Blake3 instruction:
    // rs1: points to chaining value (32 bytes)
    // rs2: points to message block (64 bytes) + counter (8 bytes) + block_len (4 bytes) + flags (4 bytes)

    // Call Blake3 instruction using funct7=0x03 to distinguish from Keccak (0x01), SHA-256 (0x00), and Blake2 (0x02)
    core::arch::asm!(
        ".insn r 0x0B, 0x0, 0x03, x0, {}, {}",
        in(reg) chaining_value,
        in(reg) message,
        options(nostack)
    );

    // // Return the modified chaining value as the result
    // let result_slice = core::slice::from_raw_parts(chaining_value, 16);
    // let mut result = [0u32; 16];
    // result.copy_from_slice(result_slice);
    // result
}

#[cfg(feature = "host")]
pub unsafe fn blake3_compress(
    chaining_value: *mut u32,
    message: *const u32,
    counter: u64,
    block_len: u32,
    flags: u32,
) {
    // On the host, we call our reference implementation from the tracer crate.
    tracer::instruction::inline_blake3::execute_blake3_compression(
        &mut *(chaining_value as *mut [u32; 16]),
        &*(message as *const [u32; 16]),
        &[counter as u32, (counter >> 32) as u32],
        block_len,
        flags,
    );
}

/// Macro to generate Blake3 hash mode functions for different input lengths
macro_rules! blake3_hash_mode {
    ($func_name:ident, $num_blocks:expr, $funct3:expr, $input_bytes:expr) => {
        #[cfg(not(feature = "host"))]
        pub unsafe fn $func_name(output: *mut u32, input: *const u32) {
            // Use custom instruction with the appropriate funct3 value for this hash mode
            core::arch::asm!(
                concat!(".insn r 0x0B, ", stringify!($funct3), ", 0x03, x0, {}, {}"),
                in(reg) output,
                in(reg) input,
                options(nostack)
            );
        }

        #[cfg(feature = "host")]
        pub unsafe fn $func_name(output: *mut u32, input: *const u32) {
            // Initialize chaining value with BLAKE3_IV
            let mut chaining_value = [0u32; 16];
            chaining_value[0..8].copy_from_slice(&BLAKE3_IV);

            // Perform chained compressions
            for i in 0..$num_blocks {
                // Get message block for this iteration (16 u32 words = 64 bytes)
                let message_offset = i * 16;
                let message_slice = core::slice::from_raw_parts(input.add(message_offset), 16);
                let mut message_block = [0u32; 16];
                message_block.copy_from_slice(message_slice);

                // Set flags: chunk_start (bit 0), chunk_end (bit 1), root (bit 3)
                let chunk_start = i == 0;
                let chunk_end = i == $num_blocks - 1;
                let flags = (chunk_start as u32) | ((chunk_end as u32) << 1) | ((chunk_end as u32) << 3);

                // Execute Blake3 compression
                tracer::instruction::inline_blake3::execute_blake3_compression(
                    &mut chaining_value,
                    &message_block,
                    &[0, 0], // counter (not used for hashing mode)
                    64,      // block length
                    flags,
                );
            }

            // Copy result to output (64 bytes = 16 u32 words)
            core::ptr::copy_nonoverlapping(chaining_value.as_ptr(), output, 16);
        }
    };
}

// Generate the 4 Blake3 hash mode functions
blake3_hash_mode!(blake3_hash_64, 1, 0x2, 64);
blake3_hash_mode!(blake3_hash_128, 2, 0x3, 128);
blake3_hash_mode!(blake3_hash_192, 3, 0x4, 192);
blake3_hash_mode!(blake3_hash_256, 4, 0x5, 256);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blake3_compress_basic() {
        // Test vector from tracer implementation
        let mut chaining_value = [0u32; 16]; // 16 words total
        chaining_value[0..8].copy_from_slice(&BLAKE3_IV); // Fill first 8 with IV
                                                          // Remaining 8 are already 0

        // Message block: sequential u32 values 0..15
        let message: [u32; 16] = [
            0u32, 1u32, 2u32, 3u32, 4u32, 5u32, 6u32, 7u32, 8u32, 9u32, 10u32, 11u32, 12u32, 13u32,
            14u32, 15u32,
        ];

        let counter = 0u64;
        let block_len = 64u32;
        let flags = 0u32;

        // Expected output from tracer test vector
        let expected: [u32; 16] = [
            0x5f98b37e, 0x26b0af2a, 0xdc58b278, 0x85d56ff6, 0x96f5d384, 0x42c9e776, 0xbeedd1e4,
            0xa03faf22, 0x8a4b2d59, 0x1a1c224d, 0x303f2ae7, 0xd36ee60c, 0xfba05dbb, 0xef024714,
            0xf597a6be, 0xd849c813,
        ];

        unsafe {
            blake3_compress(
                chaining_value.as_mut_ptr(),
                message.as_ptr(),
                counter,
                block_len,
                flags,
            )
        };

        assert_eq!(chaining_value, expected, "Blake3 compression test failed");
    }

    #[test]
    fn test_direct_tracer_call() {
        // Direct call to tracer function
        let mut chaining_value = [0u32; 16]; // 16 words total
        chaining_value[0..8].copy_from_slice(&BLAKE3_IV); // Fill first 8 with IV

        let message: [u32; 16] = [
            0u32, 1u32, 2u32, 3u32, 4u32, 5u32, 6u32, 7u32, 8u32, 9u32, 10u32, 11u32, 12u32, 13u32,
            14u32, 15u32,
        ];

        let counter = [0u32, 0u32];
        let block_len = 64u32;
        let flags = 0u32;

        // Expected output from tracer test vector
        let expected: [u32; 16] = [
            0x5f98b37e, 0x26b0af2a, 0xdc58b278, 0x85d56ff6, 0x96f5d384, 0x42c9e776, 0xbeedd1e4,
            0xa03faf22, 0x8a4b2d59, 0x1a1c224d, 0x303f2ae7, 0xd36ee60c, 0xfba05dbb, 0xef024714,
            0xf597a6be, 0xd849c813,
        ];

        // Call tracer function directly
        tracer::instruction::inline_blake3::execute_blake3_compression(
            &mut chaining_value,
            &message,
            &counter,
            block_len,
            flags,
        );

        println!("Direct tracer result: {:x?}", chaining_value);
        assert_eq!(chaining_value, expected, "Direct tracer call failed");
    }

    // Test the Blake3 hash mode functions
    #[test]
    fn test_blake3_hash_modes() {
        // Test data from tracer implementation - 4 blocks of 64 bytes each
        #[rustfmt::skip]
        const BLOCK_WORDS: [[u32; 16]; 4] = [
            // Block 0: Sequential byte pattern 0x00010203...0x3c3d3e3f
            [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c, 0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c,
             0x23222120, 0x27262524, 0x2b2a2928, 0x2f2e2d2c, 0x33323130, 0x37363534, 0x3b3a3938, 0x3f3e3d3c],
            // Block 1: Sequential byte pattern 0x40414243...0x7c7d7e7f
            [0x43424140, 0x47464544, 0x4b4a4948, 0x4f4e4d4c, 0x53525150, 0x57565554, 0x5b5a5958, 0x5f5e5d5c,
             0x63626160, 0x67666564, 0x6b6a6968, 0x6f6e6d6c, 0x73727170, 0x77767574, 0x7b7a7978, 0x7f7e7d7c],
            // Block 2: Sequential byte pattern 0x80818283...0xbcbdbebf
            [0x83828180, 0x87868584, 0x8b8a8988, 0x8f8e8d8c, 0x93929190, 0x97969594, 0x9b9a9998, 0x9f9e9d9c,
             0xa3a2a1a0, 0xa7a6a5a4, 0xabaaa9a8, 0xafaeadac, 0xb3b2b1b0, 0xb7b6b5b4, 0xbbbab9b8, 0xbfbebdbc],
            // Block 3: Sequential byte pattern 0xc0c1c2c3...0xfcfdfeff
            [0xc3c2c1c0, 0xc7c6c5c4, 0xcbcac9c8, 0xcfcecdcc, 0xd3d2d1d0, 0xd7d6d5d4, 0xdbdad9d8, 0xdfdedddc,
             0xe3e2e1e0, 0xe7e6e5e4, 0xebeae9e8, 0xefeeedec, 0xf3f2f1f0, 0xf7f6f5f4, 0xfbfaf9f8, 0xfffefdfc],
        ];

        // Expected results for each Blake3 mode (64, 128, 192, 256)
        #[rustfmt::skip]
        const EXPECTED_RESULTS: [[u32; 16]; 4] = [
            // BLAKE3_64 (1 block)
            [0x4171ed4e, 0xd45c4aea, 0x6b6088b7, 0xe2463fd2, 0xac9caf12, 0x7ddcaceb, 0xc76d4c1f,
             0x981b51f2, 0x6cc59cfc, 0xe3ff31b8, 0xe1e7a83e, 0xb209dfd1, 0x6727fd6e, 0xaa660067,
             0xb123d082, 0x1babe8df],
            // BLAKE3_128 (2 blocks)
            [0x05577ef1, 0x7865b264, 0xf4b73bc3, 0x39f54346, 0xdf054b62, 0x1fc8761a, 0x48d5ac30,
             0xef454bc4, 0xa0ab9fa6, 0x9c7f4291, 0x87aa4c5c, 0x2878a03a, 0xc5191f65, 0xc485ad5b,
             0xb168137d, 0x9ed96f1c],
            // BLAKE3_192 (3 blocks)
            [0x235dbc4a, 0xcc4afb28, 0x4bff9a54, 0x0bf07d87, 0x97462da5, 0x6b9d7457, 0x58d4338c,
             0xfffcb870, 0xdd896cf8, 0xa5f40732, 0xe842d97b, 0x4caa3db1, 0x0420a25b, 0x2cff4eb8,
             0xfd58405c, 0x9dcac30b],
            // BLAKE3_256 (4 blocks)
            [0xa45b494a, 0x8e746124, 0xd6da8fca, 0xaa76f918, 0x90c26c72, 0xb4fce93d, 0x86a73507,
             0x6b191cac, 0x94fc88e3, 0xb022df83, 0xe7d57d4a, 0xcb558b67, 0x70fc9994, 0x7961680b,
             0x84fb2342, 0x8d4015ab],
        ];

        // Flatten the blocks into contiguous input arrays for each test
        let input_64: [u32; 16] = BLOCK_WORDS[0];

        let mut input_128 = [0u32; 32];
        input_128[0..16].copy_from_slice(&BLOCK_WORDS[0]);
        input_128[16..32].copy_from_slice(&BLOCK_WORDS[1]);

        let mut input_192 = [0u32; 48];
        input_192[0..16].copy_from_slice(&BLOCK_WORDS[0]);
        input_192[16..32].copy_from_slice(&BLOCK_WORDS[1]);
        input_192[32..48].copy_from_slice(&BLOCK_WORDS[2]);

        let mut input_256 = [0u32; 64];
        input_256[0..16].copy_from_slice(&BLOCK_WORDS[0]);
        input_256[16..32].copy_from_slice(&BLOCK_WORDS[1]);
        input_256[32..48].copy_from_slice(&BLOCK_WORDS[2]);
        input_256[48..64].copy_from_slice(&BLOCK_WORDS[3]);

        // Test blake3_hash_64
        let mut output_64 = [0u32; 16];
        unsafe {
            blake3_hash_64(output_64.as_mut_ptr(), input_64.as_ptr());
        }
        assert_eq!(output_64, EXPECTED_RESULTS[0], "Blake3 hash 64 test failed");

        // Test blake3_hash_128
        let mut output_128 = [0u32; 16];
        unsafe {
            blake3_hash_128(output_128.as_mut_ptr(), input_128.as_ptr());
        }
        assert_eq!(
            output_128, EXPECTED_RESULTS[1],
            "Blake3 hash 128 test failed"
        );

        // Test blake3_hash_192
        let mut output_192 = [0u32; 16];
        unsafe {
            blake3_hash_192(output_192.as_mut_ptr(), input_192.as_ptr());
        }
        assert_eq!(
            output_192, EXPECTED_RESULTS[2],
            "Blake3 hash 192 test failed"
        );

        // Test blake3_hash_256
        let mut output_256 = [0u32; 16];
        unsafe {
            blake3_hash_256(output_256.as_mut_ptr(), input_256.as_ptr());
        }
        assert_eq!(
            output_256, EXPECTED_RESULTS[3],
            "Blake3 hash 256 test failed"
        );
    }
}
