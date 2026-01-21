use crate::{INLINE_OPCODE, SHA256_FUNCT3, SHA256_FUNCT7, SHA256_INIT_FUNCT3, SHA256_INIT_FUNCT7};
use tracer::emulator::cpu::Xlen;
use tracer::utils::inline_test_harness::{InlineMemoryLayout, InlineTestHarness};

pub fn create_sha256_harness(xlen: Xlen) -> InlineTestHarness {
    // SHA256: rs1=state/output, rs2=input (same as Blake/Keccak)
    let layout = InlineMemoryLayout::single_input(64, 32); // 64-byte block, 32-byte state
    InlineTestHarness::new(layout, xlen)
}

pub fn instruction_sha256() -> tracer::instruction::inline::INLINE {
    InlineTestHarness::create_default_instruction(INLINE_OPCODE, SHA256_FUNCT3, SHA256_FUNCT7)
}

pub fn instruction_sha256init() -> tracer::instruction::inline::INLINE {
    InlineTestHarness::create_default_instruction(
        INLINE_OPCODE,
        SHA256_INIT_FUNCT3,
        SHA256_INIT_FUNCT7,
    )
}

mod exec_functions {
    use crate::exec::{execute_sha256_compression, execute_sha256_compression_initial};
    use crate::sequence_builder::BLOCK;

    #[test]
    fn test_exec_sha256_compression_function() {
        // Test with standard test vectors
        let input = [
            0x61626380, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000018,
        ];

        let initial_state = BLOCK.map(|x| x as u32);

        let result = execute_sha256_compression(initial_state, input);

        // Expected result for SHA-256("abc")
        let expected = [
            0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223, 0xb00361a3, 0x96177a9c, 0xb410ff61,
            0xf20015ad,
        ];

        assert_eq!(
            result, expected,
            "SHA256 compression function failed for 'abc' test vector"
        );
    }

    #[test]
    fn test_exec_sha256_compression_initial_function() {
        // Test the initial compression function
        let input = [
            0x61626380, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000018,
        ];

        let result = execute_sha256_compression_initial(input);

        // Expected result for SHA-256("abc")
        let expected = [
            0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223, 0xb00361a3, 0x96177a9c, 0xb410ff61,
            0xf20015ad,
        ];

        assert_eq!(
            result, expected,
            "SHA256 initial compression function failed for 'abc' test vector"
        );
    }

    #[test]
    fn test_exec_sha256_multi_block() {
        // Test with a two-block message
        // First block
        let input1 = [
            0x61626364, 0x62636465, 0x63646566, 0x64656667, 0x65666768, 0x66676869, 0x6768696a,
            0x68696a6b, 0x696a6b6c, 0x6a6b6c6d, 0x6b6c6d6e, 0x6c6d6e6f, 0x6d6e6f70, 0x6e6f7071,
            0x80000000, 0x00000000,
        ];

        let state1 = execute_sha256_compression_initial(input1);

        // Second block with padding and length
        let input2 = [
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x000001c0,
        ];

        let result = execute_sha256_compression(state1, input2);

        // Expected result for SHA-256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")
        let expected = [
            0x248d6a61, 0xd20638b8, 0xe5c02693, 0x0c3e6039, 0xa33ce459, 0x64ff2167, 0xf6ecedd4,
            0x19db06c1,
        ];

        assert_eq!(result, expected, "SHA256 multi-block compression failed");
    }
}

mod sequence_tests {
    use super::*;
    use crate::test_constants::TestVectors;
    use tracer::emulator::cpu::Xlen;

    #[test]
    fn test_sha256_direct_execution() {
        for (desc, block, initial_state, expected) in TestVectors::get_standard_test_vectors() {
            for xlen in [Xlen::Bit32, Xlen::Bit64] {
                let mut harness = create_sha256_harness(xlen);
                harness.setup_registers();
                harness.load_input32(&block);
                harness.load_state32(&initial_state);
                harness.execute_inline(instruction_sha256());

                let result: [u32; 8] = harness.read_output32(8).try_into().unwrap();

                assert_eq!(
                    &expected, &result,
                    "SHA256 direct execution for {xlen:?}: {desc}, expected: {expected:08x?}, actual: {result:08x?}",
                );
            }
        }
    }

    #[test]
    fn test_sha256init_direct_execution() {
        for (desc, block, _initial_state, expected) in TestVectors::get_standard_test_vectors() {
            for xlen in [Xlen::Bit32, Xlen::Bit64] {
                let mut harness = create_sha256_harness(xlen);
                harness.setup_registers();
                harness.load_input32(&block);
                harness.execute_inline(instruction_sha256init());

                let result: [u32; 8] = harness.read_output32(8).try_into().unwrap();

                assert_eq!(
                    &expected,
                    &result,
                    "SHA256INIT direct execution for {xlen:?}: {desc}, expected: {expected:08x?}, actual: {result:08x?}",
                );
            }
        }
    }
}

mod sdk_tests {
    use crate::sdk::Sha256;
    #[test]
    fn test_sha256_sdk_digest() {
        // Test vector for "abc"
        let input = b"abc";
        let result = Sha256::digest(input);

        let expected = [
            0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
            0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
            0xf2, 0x00, 0x15, 0xad,
        ];

        assert_eq!(result, expected, "SHA256 SDK digest failed for 'abc'");
    }

    #[test]
    fn test_sha256_sdk_update_finalize() {
        // Test incremental hashing
        let mut hasher = Sha256::new();
        hasher.update(b"ab");
        hasher.update(b"c");
        let result = hasher.finalize();

        let expected = [
            0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
            0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
            0xf2, 0x00, 0x15, 0xad,
        ];

        assert_eq!(
            result, expected,
            "SHA256 SDK update/finalize failed for 'abc'"
        );
    }

    #[test]
    fn test_sha256_sdk_empty() {
        // Test empty input
        let result = Sha256::digest(b"");

        let expected = [
            0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f,
            0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b,
            0x78, 0x52, 0xb8, 0x55,
        ];

        assert_eq!(result, expected, "SHA256 SDK failed for empty input");
    }

    #[test]
    fn test_sha256_sdk_long_message() {
        // Test with a longer message that spans multiple blocks
        let input = b"The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.";
        let result = Sha256::digest(input);

        // This is the expected SHA-256 for this specific message
        let expected = [
            0x63, 0x52, 0x41, 0xac, 0x82, 0x3e, 0xe4, 0xa8, 0x1f, 0xbb, 0x41, 0x0c, 0x92, 0xbe,
            0x61, 0x6b, 0x0a, 0x89, 0x19, 0x10, 0x83, 0xd8, 0xd7, 0xb5, 0xd2, 0x32, 0xc8, 0x23,
            0xdc, 0x8d, 0xf4, 0xf5,
        ];

        assert_eq!(result, expected, "SHA256 SDK failed for long message");
    }

    #[test]
    fn test_sha256_aligned_vs_unaligned() {
        use sha2::{Digest, Sha256 as RefSha256};

        // Test various sizes including block boundary (64 bytes)
        let test_sizes = [
            0, 1, 3, 4, 7, 8, 31, 32, 55, 56, 63, 64, 65, 100, 128, 256, 512, 1024, 2048,
        ];

        for &size in &test_sizes {
            // Create aligned buffer
            let aligned: Vec<u8> = (0..size).map(|i| (i * 37 + 11) as u8).collect();

            // Create unaligned buffer by adding 1-byte offset
            let mut unaligned_buf = vec![0u8; size + 1];
            unaligned_buf[1..].copy_from_slice(&aligned);
            let unaligned = &unaligned_buf[1..];

            // Verify alignment difference
            if size > 0 {
                assert_ne!(
                    aligned.as_ptr() as usize % 4,
                    unaligned.as_ptr() as usize % 4,
                    "Test setup error: pointers should have different alignment"
                );
            }

            // Both should produce identical results
            let aligned_result = Sha256::digest(&aligned);
            let unaligned_result = Sha256::digest(unaligned);

            assert_eq!(
                aligned_result, unaligned_result,
                "SHA256: aligned vs unaligned mismatch at size {size}"
            );

            // Also verify against reference implementation
            let expected: [u8; 32] = RefSha256::digest(&aligned).into();
            assert_eq!(
                aligned_result, expected,
                "SHA256: result doesn't match reference at size {size}"
            );
        }
    }
}
