/// Calls the Blake3 compression custom instruction.
/// # Safety
/// - `chaining_value` must be a valid pointer to 32 bytes of readable and writable memory.
/// - `message` must be a valid pointer to 64 bytes of readable memory.
/// - Both pointers must be properly aligned for u32 access (4-byte alignment).
#[cfg(not(feature = "host"))]
pub unsafe fn blake3_compress(chaining_value: *mut u32, message: *const u32) {
    // Memory layout for Blake3 instruction:
    // rs1: points to chaining value (32 bytes)
    // rs2: points to message block (64 bytes) + counter (8 bytes) + block_len (4 bytes) + flags (4 bytes)

    core::arch::asm!(
        ".insn r 0x0B, 0x0, 0x03, x0, {}, {}",
        in(reg) chaining_value,
        in(reg) message,
        options(nostack)
    );
}

#[cfg(feature = "host")]
pub unsafe fn blake3_compress(chaining_value: *mut u32, message: *const u32) {
    // Memory layout for Blake3 instruction:
    // message points to: message block (64 bytes / 16 u32s) + counter (8 bytes / 2 u32s) + block_len (4 bytes / 1 u32) + flags (4 bytes / 1 u32)

    // Extract the message block (first 16 u32s)
    let message_block = &*(message as *const [u32; 16]);

    // Extract counter (next 2 u32s at offset 16)
    let counter_ptr = message.add(16);
    let counter_low = *counter_ptr;
    let counter_high = *counter_ptr.add(1);
    let counter_array = [counter_low, counter_high];

    // Extract block_len (next u32 at offset 18)
    let block_len = *message.add(18);

    // Extract flags (next u32 at offset 19)
    let flags = *message.add(19);

    // On the host, we call our reference implementation from the exec module.
    crate::exec::execute_blake3_compression(
        &mut *(chaining_value as *mut [u32; 16]),
        message_block,
        &counter_array,
        block_len,
        flags,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IV;

    #[test]
    fn test_blake3_compress_basic() {
        // Test vector from tracer implementation
        let mut chaining_value = [0u32; 16]; // 16 words total
        chaining_value[0..8].copy_from_slice(&IV); // Fill first 8 with IV
                                                   // Remaining 8 are already 0

        // Prepare message buffer with the full layout:
        // message block (16 u32s) + counter (2 u32s) + block_len (1 u32) + flags (1 u32)
        let mut full_message = [0u32; 20];

        // Message block (first 16 u32s): sequential u32 values 0..15
        full_message[0..16].copy_from_slice(&[
            0u32, 1u32, 2u32, 3u32, 4u32, 5u32, 6u32, 7u32, 8u32, 9u32, 10u32, 11u32, 12u32, 13u32,
            14u32, 15u32,
        ]);

        // Counter (next 2 u32s at indices 16-17)
        let counter = 0u64;
        full_message[16] = counter as u32; // counter low
        full_message[17] = (counter >> 32) as u32; // counter high

        // Block length (index 18)
        full_message[18] = 64u32;

        // Flags (index 19)
        full_message[19] = 0u32;

        // Expected output from tracer test vector
        let expected: [u32; 16] = [
            0x5f98b37e, 0x26b0af2a, 0xdc58b278, 0x85d56ff6, 0x96f5d384, 0x42c9e776, 0xbeedd1e4,
            0xa03faf22, 0x8a4b2d59, 0x1a1c224d, 0x303f2ae7, 0xd36ee60c, 0xfba05dbb, 0xef024714,
            0xf597a6be, 0xd849c813,
        ];

        unsafe { blake3_compress(chaining_value.as_mut_ptr(), full_message.as_ptr()) };

        assert_eq!(chaining_value, expected, "Blake3 compression test failed");
    }

    #[cfg(feature = "host")]
    #[test]
    fn test_direct_tracer_call() {
        // Direct call to tracer function
        let mut chaining_value = [0u32; 16]; // 16 words total
        chaining_value[0..8].copy_from_slice(&IV); // Fill first 8 with IV

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

        // Call exec function directly
        crate::exec::execute_blake3_compression(
            &mut chaining_value,
            &message,
            &counter,
            block_len,
            flags,
        );

        assert_eq!(chaining_value, expected, "Direct tracer call failed");
    }
}
