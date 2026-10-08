mod exec_functions {
    use crate::exec::{execute_sha256_compression, execute_sha256_compression_initial};

    #[test]
    fn test_exec_sha256_multi_block() {
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
    use std::collections::BTreeMap;

    use crate::sequence_builder::{Sha256Compression, Sha256CompressionInitial};
    use jolt_inlines_sdk::{
        assert_edge_cases_match_reference, assert_random_cases_match_reference,
    };
    use tracer::{
        instruction::Instruction,
        utils::{
            inline_test_harness::InlineTestHarness, virtual_registers::VirtualRegisterAllocator,
        },
    };

    fn inline_rows(funct3: u32) -> (usize, BTreeMap<&'static str, usize>) {
        let instruction = InlineTestHarness::create_default_instruction(
            crate::INLINE_OPCODE,
            funct3,
            crate::SHA256_FUNCT7,
        );
        let sequence = instruction.inline_sequence(&VirtualRegisterAllocator::default());
        let mut histogram = BTreeMap::new();
        for instruction in &sequence {
            let mnemonic: &'static str = <&Instruction>::into(instruction);
            *histogram.entry(mnemonic).or_default() += 1;
        }
        (sequence.len(), histogram)
    }

    #[test]
    fn test_sha256_inline_rows_per_block() {
        // Σ₀/Σ₁ are 3 rows each, σ₀/σ₁ are 4; fixed-IV folds both round-0 Σ values.
        let (custom_iv_rows, custom_iv_histogram) = inline_rows(crate::SHA256_FUNCT3);
        let (fixed_iv_rows, fixed_iv_histogram) = inline_rows(crate::SHA256_INIT_FUNCT3);
        assert_eq!(custom_iv_rows, 1900, "{custom_iv_histogram:?}");
        assert_eq!(fixed_iv_rows, 1864, "{fixed_iv_histogram:?}");
    }

    #[test]
    fn test_sha256_direct_execution() {
        assert_edge_cases_match_reference::<Sha256Compression>();
    }

    #[test]
    fn test_sha256init_direct_execution() {
        assert_edge_cases_match_reference::<Sha256CompressionInitial>();
    }

    #[test]
    fn test_sha256_random_direct_execution() {
        assert_random_cases_match_reference::<Sha256Compression>(0x5A256, 100);
    }

    #[test]
    fn test_sha256init_random_direct_execution() {
        assert_random_cases_match_reference::<Sha256CompressionInitial>(0x1256, 100);
    }
}

mod sdk_tests {
    use crate::sdk::Sha256;
    use sha2::{Digest, Sha256 as RefSha256};

    #[test]
    fn test_sha256_sdk_digest() {
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
    fn test_sha256_aligned_vs_unaligned() {
        // Test various sizes including block boundary (64 bytes)
        let test_sizes = [
            0, 1, 3, 4, 7, 8, 31, 32, 55, 56, 63, 64, 65, 100, 128, 256, 512, 1024, 2048,
        ];

        for &size in &test_sizes {
            let aligned: Vec<u8> = (0..size).map(|i| (i * 37 + 11) as u8).collect();

            let mut unaligned_buf = vec![0u8; size + 1];
            unaligned_buf[1..].copy_from_slice(&aligned);
            let unaligned = &unaligned_buf[1..];

            if size > 0 {
                assert_ne!(
                    aligned.as_ptr() as usize % 4,
                    unaligned.as_ptr() as usize % 4,
                    "Test setup error: pointers should have different alignment"
                );
            }

            let aligned_result = Sha256::digest(&aligned);
            let unaligned_result = Sha256::digest(unaligned);

            assert_eq!(
                aligned_result, unaligned_result,
                "SHA256: aligned vs unaligned mismatch at size {size}"
            );

            let expected: [u8; 32] = RefSha256::digest(&aligned).into();
            assert_eq!(
                aligned_result, expected,
                "SHA256: result doesn't match reference at size {size}"
            );
        }
    }
}
