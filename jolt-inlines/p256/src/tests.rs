mod p256_tests {
    use crate::sdk::P256PointExt;
    use crate::{
        INLINE_OPCODE, P256_DIVQ_FUNCT3, P256_DIVR_FUNCT3, P256_FUNCT7, P256_MULQ_FUNCT3,
        P256_MULR_FUNCT3, P256_SQUAREQ_FUNCT3, P256_SQUARER_FUNCT3,
    };
    use crate::{P256_CURVE_B, P256_GENERATOR_X, P256_GENERATOR_Y, P256_MODULUS, P256_ORDER};
    use num_bigint::BigUint;
    use tracer::emulator::mmu::DRAM_BASE;
    use tracer::utils::inline_test_harness::{InlineMemoryLayout, InlineTestHarness};

    fn limbs_to_biguint(limbs: &[u64; 4]) -> BigUint {
        let mut bytes = [0u8; 32];
        for (i, &limb) in limbs.iter().enumerate() {
            bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
        }
        BigUint::from_bytes_le(&bytes)
    }

    fn biguint_to_limbs(v: &BigUint) -> [u64; 4] {
        let bytes = v.to_bytes_le();
        let mut padded = [0u8; 32];
        let len = bytes.len().min(32);
        padded[..len].copy_from_slice(&bytes[..len]);
        let mut limbs = [0u64; 4];
        for i in 0..4 {
            limbs[i] = u64::from_le_bytes(padded[i * 8..(i + 1) * 8].try_into().unwrap());
        }
        limbs
    }

    fn bigint_mulmod(a: &[u64; 4], b: &[u64; 4], modulus: &[u64; 4]) -> [u64; 4] {
        let a_big = limbs_to_biguint(a);
        let b_big = limbs_to_biguint(b);
        let m_big = limbs_to_biguint(modulus);
        let result = (a_big * b_big) % m_big;
        biguint_to_limbs(&result)
    }

    fn bigint_divmod(a: &[u64; 4], b: &[u64; 4], modulus: &[u64; 4]) -> [u64; 4] {
        let a_big = limbs_to_biguint(a);
        let b_big = limbs_to_biguint(b);
        let m_big = limbs_to_biguint(modulus);
        // b^{-1} = b^{m-2} mod m  (Fermat's little theorem)
        let exp = &m_big - BigUint::from(2u64);
        let b_inv = b_big.modpow(&exp, &m_big);
        let result = (a_big * b_inv) % m_big;
        biguint_to_limbs(&result)
    }

    fn assert_mulq_trace_equiv(a: &[u64; 4], b: &[u64; 4]) {
        let expected = bigint_mulmod(a, b, &P256_MODULUS);
        let layout = InlineMemoryLayout::two_inputs(32, 32, 32);
        let mut harness = InlineTestHarness::new(layout);
        harness.setup_registers();
        harness.load_input64(a);
        harness.load_input2_64(b);
        harness.execute_inline(InlineTestHarness::create_default_instruction(
            INLINE_OPCODE,
            P256_MULQ_FUNCT3,
            P256_FUNCT7,
        ));
        let result_vec = harness.read_output64(4);
        let mut result = [0u64; 4];
        result.copy_from_slice(&result_vec);
        assert_eq!(result, expected, "p256_mulq result mismatch");
    }

    fn assert_squareq_trace_equiv(a: &[u64; 4]) {
        let expected = bigint_mulmod(a, a, &P256_MODULUS);
        let layout = InlineMemoryLayout::two_inputs(32, 32, 32);
        let mut harness = InlineTestHarness::new(layout);
        harness.setup_registers();
        harness.load_input64(a);
        harness.execute_inline(InlineTestHarness::create_default_instruction(
            INLINE_OPCODE,
            P256_SQUAREQ_FUNCT3,
            P256_FUNCT7,
        ));
        let result_vec = harness.read_output64(4);
        let mut result = [0u64; 4];
        result.copy_from_slice(&result_vec);
        assert_eq!(result, expected, "p256_squareq result mismatch");
    }

    fn assert_divq_trace_equiv(a: &[u64; 4], b: &[u64; 4]) {
        let expected = bigint_divmod(a, b, &P256_MODULUS);
        let layout = InlineMemoryLayout::two_inputs(32, 32, 32);
        let mut harness = InlineTestHarness::new(layout);
        harness.setup_registers();
        harness.load_input64(a);
        harness.load_input2_64(b);
        harness.execute_inline(InlineTestHarness::create_default_instruction(
            INLINE_OPCODE,
            P256_DIVQ_FUNCT3,
            P256_FUNCT7,
        ));
        let result_vec = harness.read_output64(4);
        let mut result = [0u64; 4];
        result.copy_from_slice(&result_vec);
        assert_eq!(result, expected, "p256_divq result mismatch");
    }

    /// Division with the result buffer aliasing the dividend (`rs3 == rs1`).
    /// The advised quotient must be stored only after every `VirtualAssertEQ`
    /// against the dividend has run, otherwise the checks compare the stored
    /// result against itself and any value would be accepted.
    fn assert_div_trace_equiv_aliased(funct3: u32, a: &[u64; 4], b: &[u64; 4], modulus: &[u64; 4]) {
        let expected = bigint_divmod(a, b, modulus);
        // rs1 == rs3: dividend and result share one 32-byte region
        let layout = InlineMemoryLayout {
            output_base: DRAM_BASE,
            ..InlineMemoryLayout::two_inputs(32, 32, 32)
        };
        let mut harness = InlineTestHarness::new(layout);
        harness.setup_registers();
        harness.load_input64(a);
        harness.load_input2_64(b);
        harness.execute_inline(InlineTestHarness::create_default_instruction(
            INLINE_OPCODE,
            funct3,
            P256_FUNCT7,
        ));
        let result_vec = harness.read_output64(4);
        let mut result = [0u64; 4];
        result.copy_from_slice(&result_vec);
        assert_eq!(result, expected, "aliased div result mismatch");
    }

    #[test]
    fn test_p256_div_aliased_dividend_and_result() {
        let a = [
            0x123456789ABCDEF0,
            0x0FEDCBA987654321,
            0x1111111111111111,
            0x2222222222222222,
        ];
        let b = [
            0x0FEDCBA987654321,
            0x123456789ABCDEF0,
            0x3333333333333333,
            0x4444444444444444,
        ];
        assert_div_trace_equiv_aliased(P256_DIVQ_FUNCT3, &a, &b, &P256_MODULUS);
        assert_div_trace_equiv_aliased(P256_DIVR_FUNCT3, &a, &b, &P256_ORDER);
    }

    fn assert_mulr_trace_equiv(a: &[u64; 4], b: &[u64; 4]) {
        let expected = bigint_mulmod(a, b, &P256_ORDER);
        let layout = InlineMemoryLayout::two_inputs(32, 32, 32);
        let mut harness = InlineTestHarness::new(layout);
        harness.setup_registers();
        harness.load_input64(a);
        harness.load_input2_64(b);
        harness.execute_inline(InlineTestHarness::create_default_instruction(
            INLINE_OPCODE,
            P256_MULR_FUNCT3,
            P256_FUNCT7,
        ));
        let result_vec = harness.read_output64(4);
        let mut result = [0u64; 4];
        result.copy_from_slice(&result_vec);
        assert_eq!(result, expected, "p256_mulr result mismatch");
    }

    fn assert_squarer_trace_equiv(a: &[u64; 4]) {
        let expected = bigint_mulmod(a, a, &P256_ORDER);
        let layout = InlineMemoryLayout::two_inputs(32, 32, 32);
        let mut harness = InlineTestHarness::new(layout);
        harness.setup_registers();
        harness.load_input64(a);
        harness.execute_inline(InlineTestHarness::create_default_instruction(
            INLINE_OPCODE,
            P256_SQUARER_FUNCT3,
            P256_FUNCT7,
        ));
        let result_vec = harness.read_output64(4);
        let mut result = [0u64; 4];
        result.copy_from_slice(&result_vec);
        assert_eq!(result, expected, "p256_squarer result mismatch");
    }

    fn assert_divr_trace_equiv(a: &[u64; 4], b: &[u64; 4]) {
        let expected = bigint_divmod(a, b, &P256_ORDER);
        let layout = InlineMemoryLayout::two_inputs(32, 32, 32);
        let mut harness = InlineTestHarness::new(layout);
        harness.setup_registers();
        harness.load_input64(a);
        harness.load_input2_64(b);
        harness.execute_inline(InlineTestHarness::create_default_instruction(
            INLINE_OPCODE,
            P256_DIVR_FUNCT3,
            P256_FUNCT7,
        ));
        let result_vec = harness.read_output64(4);
        let mut result = [0u64; 4];
        result.copy_from_slice(&result_vec);
        assert_eq!(result, expected, "p256_divr result mismatch");
    }

    #[test]
    fn test_p256_mulq() {
        let seven = [7u64, 0, 0, 0];
        assert_mulq_trace_equiv(&seven, &seven);

        assert_mulq_trace_equiv(&P256_GENERATOR_X, &P256_GENERATOR_X);
        {
            let expected_gx2: [u64; 4] = [
                0x002ae56c426b3f8c,
                0x33b699495d694dd1,
                0x81819a5e0e3690d8,
                0x98f6b84d29bef2b2,
            ];
            let computed = bigint_mulmod(&P256_GENERATOR_X, &P256_GENERATOR_X, &P256_MODULUS);
            assert_eq!(
                computed, expected_gx2,
                "GX^2 reference mismatch against Python value"
            );
        }

        let pm1: [u64; 4] = [
            P256_MODULUS[0].wrapping_sub(1),
            P256_MODULUS[1],
            P256_MODULUS[2],
            P256_MODULUS[3],
        ];
        assert_mulq_trace_equiv(&pm1, &pm1);

        let two = [2u64, 0, 0, 0];
        assert_mulq_trace_equiv(&pm1, &two);

        let a = [
            0x123456789ABCDEF0,
            0x0FEDCBA987654321,
            0x1111111111111111,
            0x2222222222222222,
        ];
        let b = [
            0x0FEDCBA987654321,
            0x123456789ABCDEF0,
            0x3333333333333333,
            0x4444444444444444,
        ];
        assert_mulq_trace_equiv(&a, &b);

        let a = [1u64, 2, 3, 4];
        let b = [5u64, 6, 7, 8];
        assert_mulq_trace_equiv(&a, &b);

        let one = [1u64, 0, 0, 0];
        assert_mulq_trace_equiv(&P256_GENERATOR_X, &one);
    }

    #[test]
    fn test_p256_squareq() {
        let seven = [7u64, 0, 0, 0];
        assert_squareq_trace_equiv(&seven);

        assert_squareq_trace_equiv(&P256_GENERATOR_X);

        assert_squareq_trace_equiv(&P256_GENERATOR_Y);

        let pm1: [u64; 4] = [
            P256_MODULUS[0].wrapping_sub(1),
            P256_MODULUS[1],
            P256_MODULUS[2],
            P256_MODULUS[3],
        ];
        assert_squareq_trace_equiv(&pm1);

        let a = [
            0x123456789ABCDEF0,
            0x0FEDCBA987654321,
            0x1111111111111111,
            0x2222222222222222,
        ];
        assert_squareq_trace_equiv(&a);

        let a = [1u64, 2, 3, 4];
        assert_squareq_trace_equiv(&a);

        let a = [1u64, 1, 1, 1];
        assert_squareq_trace_equiv(&a);
    }

    #[test]
    fn test_p256_divq() {
        let one = [1u64, 0, 0, 0];
        assert_divq_trace_equiv(&P256_GENERATOR_X, &one);

        {
            let a = [
                0x123456789ABCDEF0,
                0x0FEDCBA987654321,
                0x1111111111111111,
                0x2222222222222222,
            ];
            let b = [
                0x0FEDCBA987654321,
                0x123456789ABCDEF0,
                0x3333333333333333,
                0x4444444444444444,
            ];
            let ab = bigint_mulmod(&a, &b, &P256_MODULUS);
            let recovered = bigint_divmod(&ab, &b, &P256_MODULUS);
            assert_eq!(recovered, a, "a*b / b should equal a");
            assert_divq_trace_equiv(&ab, &b);
        }

        let a = [
            0x123456789ABCDEF0,
            0x0FEDCBA987654321,
            0x1111111111111111,
            0x2222222222222222,
        ];
        let b = [
            0x0FEDCBA987654321,
            0x123456789ABCDEF0,
            0x3333333333333333,
            0x4444444444444444,
        ];
        assert_divq_trace_equiv(&a, &b);

        let a = [1u64, 2, 3, 4];
        let b = [5u64, 6, 7, 8];
        assert_divq_trace_equiv(&a, &b);

        let a = [1u64, 1, 1, 1];
        let b = [1u64, 1, 1, 1];
        assert_divq_trace_equiv(&a, &b);
    }

    #[test]
    fn test_p256_mulr() {
        let a = [0u64, 0, 0, 1];
        let b = [0u64, 1, 0, 0];
        assert_mulr_trace_equiv(&a, &b);

        let a = [
            0x123456789ABCDEF0,
            0x0FEDCBA987654321,
            0x1111111111111111,
            0x2222222222222222,
        ];
        let b = [
            0x0FEDCBA987654321,
            0x123456789ABCDEF0,
            0x3333333333333333,
            0x4444444444444444,
        ];
        assert_mulr_trace_equiv(&a, &b);

        let a = [1u64, 2, 3, 4];
        let b = [5u64, 6, 7, 8];
        assert_mulr_trace_equiv(&a, &b);

        let a = [1u64, 1, 1, 1];
        let b = [1u64, 1, 1, 1];
        assert_mulr_trace_equiv(&a, &b);

        let nm1: [u64; 4] = [
            P256_ORDER[0].wrapping_sub(1),
            P256_ORDER[1],
            P256_ORDER[2],
            P256_ORDER[3],
        ];
        assert_mulr_trace_equiv(&nm1, &nm1);

        let one = [1u64, 0, 0, 0];
        let a = [
            0xAAAAAAAAAAAAAAAA,
            0xBBBBBBBBBBBBBBBB,
            0xCCCCCCCCCCCCCCCC,
            0x1111111111111111,
        ];
        assert_mulr_trace_equiv(&a, &one);
    }

    #[test]
    fn test_p256_squarer() {
        let a = [0u64, 0, 0, 1];
        assert_squarer_trace_equiv(&a);

        let a = [
            0x123456789ABCDEF0,
            0x0FEDCBA987654321,
            0x1111111111111111,
            0x2222222222222222,
        ];
        assert_squarer_trace_equiv(&a);

        let a = [1u64, 2, 3, 4];
        assert_squarer_trace_equiv(&a);

        let a = [1u64, 1, 1, 1];
        assert_squarer_trace_equiv(&a);

        let nm1: [u64; 4] = [
            P256_ORDER[0].wrapping_sub(1),
            P256_ORDER[1],
            P256_ORDER[2],
            P256_ORDER[3],
        ];
        assert_squarer_trace_equiv(&nm1);
    }

    #[test]
    fn test_p256_divr() {
        let one = [1u64, 0, 0, 0];
        let a = [
            0x123456789ABCDEF0,
            0x0FEDCBA987654321,
            0x1111111111111111,
            0x2222222222222222,
        ];
        assert_divr_trace_equiv(&a, &one);

        {
            let a = [
                0x123456789ABCDEF0,
                0x0FEDCBA987654321,
                0x1111111111111111,
                0x2222222222222222,
            ];
            let b = [
                0x0FEDCBA987654321,
                0x123456789ABCDEF0,
                0x3333333333333333,
                0x4444444444444444,
            ];
            let ab = bigint_mulmod(&a, &b, &P256_ORDER);
            let recovered = bigint_divmod(&ab, &b, &P256_ORDER);
            assert_eq!(recovered, a, "scalar: a*b / b should equal a");
            assert_divr_trace_equiv(&ab, &b);
        }

        let a = [
            0x123456789ABCDEF0,
            0x0FEDCBA987654321,
            0x1111111111111111,
            0x2222222222222222,
        ];
        let b = [
            0x0FEDCBA987654321,
            0x123456789ABCDEF0,
            0x3333333333333333,
            0x4444444444444444,
        ];
        assert_divr_trace_equiv(&a, &b);

        let a = [1u64, 2, 3, 4];
        let b = [5u64, 6, 7, 8];
        assert_divr_trace_equiv(&a, &b);

        let a = [1u64, 1, 1, 1];
        let b = [1u64, 1, 1, 1];
        assert_divr_trace_equiv(&a, &b);
    }

    #[test]
    fn test_p256_point_on_curve() {
        let p = limbs_to_biguint(&P256_MODULUS);
        let gx = limbs_to_biguint(&P256_GENERATOR_X);
        let gy = limbs_to_biguint(&P256_GENERATOR_Y);
        let b = limbs_to_biguint(&P256_CURVE_B);

        let lhs = gy.modpow(&BigUint::from(2u64), &p);

        let x3 = gx.modpow(&BigUint::from(3u64), &p);
        let three_x = (&gx * BigUint::from(3u64)) % &p;
        let rhs = (x3 + &p - &three_x + &b) % &p;

        assert_eq!(lhs, rhs, "P-256 generator is not on the curve");

        let gx2 = gx.modpow(&BigUint::from(2u64), &p);
        let expected_gx2_limbs: [u64; 4] = [
            0x002ae56c426b3f8c,
            0x33b699495d694dd1,
            0x81819a5e0e3690d8,
            0x98f6b84d29bef2b2,
        ];
        assert_eq!(
            biguint_to_limbs(&gx2),
            expected_gx2_limbs,
            "GX^2 intermediate does not match Python-computed value"
        );
    }

    /// Test double_and_add when 2P + Q = O (the infinity edge case fix).
    #[test]
    fn test_double_and_add_infinity() {
        use crate::sdk::P256Point;
        let g = P256Point::generator();
        let two_g = g.double();
        let neg_two_g = two_g.neg();

        let result = g.double_and_add(&neg_two_g);
        assert!(result.is_infinity(), "2G + (-2G) should be infinity");

        let naive = g.double().add(&neg_two_g);
        assert!(naive.is_infinity(), "naive 2G + (-2G) should be infinity");
    }

    #[test]
    fn test_double_and_add_edge_cases() {
        use crate::sdk::P256Point;
        let g = P256Point::generator();
        let inf = P256Point::infinity();

        let r = inf.double_and_add(&g);
        assert_eq!(r.x().e(), g.x().e());

        let r = g.double_and_add(&inf);
        let expected = g.double();
        assert_eq!(r.x().e(), expected.x().e());

        let r = g.double_and_add(&g);
        let expected = g.double().add(&g);
        assert_eq!(r.x().e(), expected.x().e());

        let neg_g = g.neg();
        let r = g.double_and_add(&neg_g);
        assert_eq!(r.x().e(), g.x().e());
        assert_eq!(r.y().e(), g.y().e());
    }

    #[test]
    fn test_ecdsa_verify_rejects_invalid() {
        use crate::sdk::{ecdsa_verify, P256Error, P256Fq, P256Fr, P256Point};

        let g = P256Point::generator();
        let z = P256Fr::from_u64_arr(&[1, 0, 0, 0]).unwrap();
        let r = P256Fr::from_u64_arr(&[1, 0, 0, 0]).unwrap();
        let s = P256Fr::from_u64_arr(&[1, 0, 0, 0]).unwrap();

        let result = ecdsa_verify(z.clone(), r.clone(), s.clone(), P256Point::infinity());
        assert!(matches!(result, Err(P256Error::QAtInfinity)));

        let zero = P256Fr::from_u64_arr(&[0, 0, 0, 0]).unwrap();
        let result = ecdsa_verify(z.clone(), zero.clone(), s.clone(), g.clone());
        assert!(matches!(result, Err(P256Error::ROrSZero)));

        let result = ecdsa_verify(z.clone(), r.clone(), zero, g.clone());
        assert!(matches!(result, Err(P256Error::ROrSZero)));

        let bad_z = P256Fr::from_u64_arr_unchecked(&[u64::MAX; 4]);
        let result = ecdsa_verify(bad_z, r.clone(), s.clone(), g.clone());
        assert!(matches!(result, Err(P256Error::InvalidFrElement)));

        let bad_r = P256Fr::from_u64_arr_unchecked(&[u64::MAX; 4]);
        let result = ecdsa_verify(z.clone(), bad_r, s.clone(), g.clone());
        assert!(matches!(result, Err(P256Error::InvalidFrElement)));

        let bad_s = P256Fr::from_u64_arr_unchecked(&[u64::MAX; 4]);
        let result = ecdsa_verify(z.clone(), r.clone(), bad_s, g.clone());
        assert!(matches!(result, Err(P256Error::InvalidFrElement)));

        let bad_x = P256Fq::from_u64_arr_unchecked(&[u64::MAX; 4]);
        let bad_q = P256Point::new_unchecked(bad_x, g.y());
        let result = ecdsa_verify(z.clone(), r.clone(), s.clone(), bad_q);
        assert!(matches!(result, Err(P256Error::InvalidFqElement)));

        let bad_y = P256Fq::from_u64_arr_unchecked(&[u64::MAX; 4]);
        let bad_q = P256Point::new_unchecked(g.x(), bad_y);
        let result = ecdsa_verify(z.clone(), r.clone(), s.clone(), bad_q);
        assert!(matches!(result, Err(P256Error::InvalidFqElement)));

        let off_curve_y = P256Fq::from_u64_arr(&[1, 0, 0, 0]).unwrap();
        let bad_q = P256Point::new_unchecked(g.x(), off_curve_y);
        let result = ecdsa_verify(z.clone(), r.clone(), s.clone(), bad_q);
        assert!(matches!(result, Err(P256Error::NotOnCurve)));
    }

    #[test]
    fn test_interop_multiple_messages() {
        use crate::sdk::{ecdsa_verify, P256Fr, P256Point};
        use p256::ecdsa::{signature::Signer, Signature, SigningKey};
        use sha2::{Digest, Sha256};

        let signing_key = SigningKey::random(&mut rand::thread_rng());
        let verifying_key = p256::ecdsa::VerifyingKey::from(&signing_key);
        let pubkey_point = verifying_key.to_encoded_point(false);

        let be_to_limbs = |bytes: &[u8]| -> [u64; 4] {
            let mut padded = [0u8; 32];
            let start = 32 - bytes.len().min(32);
            padded[start..].copy_from_slice(&bytes[..bytes.len().min(32)]);
            [
                u64::from_be_bytes(padded[24..32].try_into().unwrap()),
                u64::from_be_bytes(padded[16..24].try_into().unwrap()),
                u64::from_be_bytes(padded[8..16].try_into().unwrap()),
                u64::from_be_bytes(padded[0..8].try_into().unwrap()),
            ]
        };

        let qx_limbs = be_to_limbs(pubkey_point.x().unwrap().as_slice());
        let qy_limbs = be_to_limbs(pubkey_point.y().unwrap().as_slice());
        let mut q_arr = [0u64; 8];
        q_arr[..4].copy_from_slice(&qx_limbs);
        q_arr[4..].copy_from_slice(&qy_limbs);
        let q = P256Point::from_u64_arr(&q_arr).unwrap();

        let messages: &[&[u8]] = &[
            b"hello world",
            b"",
            b"a]",
            &[0u8; 1000],
            b"\xff\xff\xff\xff",
        ];

        for msg in messages {
            let signature: Signature = signing_key.sign(msg);
            let z_bytes: [u8; 32] = Sha256::digest(*msg).into();
            let z = P256Fr::from_u64_arr(&be_to_limbs(&z_bytes)).unwrap();
            let r =
                P256Fr::from_u64_arr(&be_to_limbs(signature.r().to_bytes().as_slice())).unwrap();
            let s =
                P256Fr::from_u64_arr(&be_to_limbs(signature.s().to_bytes().as_slice())).unwrap();

            assert!(
                ecdsa_verify(z.clone(), r, s, q.clone()).is_ok(),
                "Failed to verify p256-crate signature for message of len {}",
                msg.len()
            );
        }
    }

    #[test]
    fn test_interop_corrupted_signature_rejected() {
        use crate::sdk::{ecdsa_verify, P256Fr, P256Point};
        use p256::ecdsa::{signature::Signer, Signature, SigningKey};
        use sha2::{Digest, Sha256};

        let signing_key = SigningKey::from_bytes(
            &[
                0xC9, 0xAF, 0xA9, 0xD8, 0x45, 0xBA, 0x75, 0x16, 0x6B, 0x5C, 0x21, 0x57, 0x67, 0xB1,
                0xD6, 0x93, 0x4E, 0x50, 0xC3, 0xDB, 0x36, 0xE8, 0x9B, 0x12, 0x7B, 0x8A, 0x62, 0x2B,
                0x12, 0x0F, 0x67, 0x21,
            ]
            .into(),
        )
        .unwrap();

        let message = b"test corruption";
        let signature: Signature = signing_key.sign(message);
        let z_bytes: [u8; 32] = Sha256::digest(message).into();

        let verifying_key = p256::ecdsa::VerifyingKey::from(&signing_key);
        let pubkey_point = verifying_key.to_encoded_point(false);

        let be_to_limbs = |bytes: &[u8]| -> [u64; 4] {
            let mut padded = [0u8; 32];
            let start = 32 - bytes.len().min(32);
            padded[start..].copy_from_slice(&bytes[..bytes.len().min(32)]);
            [
                u64::from_be_bytes(padded[24..32].try_into().unwrap()),
                u64::from_be_bytes(padded[16..24].try_into().unwrap()),
                u64::from_be_bytes(padded[8..16].try_into().unwrap()),
                u64::from_be_bytes(padded[0..8].try_into().unwrap()),
            ]
        };

        let z = P256Fr::from_u64_arr(&be_to_limbs(&z_bytes)).unwrap();
        let r = P256Fr::from_u64_arr(&be_to_limbs(signature.r().to_bytes().as_slice())).unwrap();
        let s = P256Fr::from_u64_arr(&be_to_limbs(signature.s().to_bytes().as_slice())).unwrap();

        let qx_limbs = be_to_limbs(pubkey_point.x().unwrap().as_slice());
        let qy_limbs = be_to_limbs(pubkey_point.y().unwrap().as_slice());
        let mut q_arr = [0u64; 8];
        q_arr[..4].copy_from_slice(&qx_limbs);
        q_arr[4..].copy_from_slice(&qy_limbs);
        let q = P256Point::from_u64_arr(&q_arr).unwrap();

        assert!(ecdsa_verify(z.clone(), r.clone(), s.clone(), q.clone()).is_ok());

        let wrong_z_bytes: [u8; 32] = Sha256::digest(b"wrong message").into();
        let wrong_z = P256Fr::from_u64_arr(&be_to_limbs(&wrong_z_bytes)).unwrap();
        assert!(ecdsa_verify(wrong_z, r.clone(), s.clone(), q.clone()).is_err());

        let mut r_limbs = r.e();
        r_limbs[0] ^= 1;
        if let Ok(bad_r) = P256Fr::from_u64_arr(&r_limbs) {
            assert!(ecdsa_verify(z.clone(), bad_r, s.clone(), q.clone()).is_err());
        }
    }

    /// Regression test: ecdsa_verify must reject z=0 (message hash of zero).
    ///
    /// Before the fix, a zero message hash was accepted, which allowed a
    /// trivial forgery: with z=0 the verification equation degenerates to
    /// u1*G = (0/s)*G = O, so the attacker only needs u2*Q to have the
    /// right x-coordinate, which is easy to arrange.
    #[test]
    fn test_ecdsa_verify_rejects_zero_hash() {
        use crate::sdk::{ecdsa_verify, P256Error, P256Fr, P256Point};
        let g = P256Point::generator();
        let zero = P256Fr::from_u64_arr(&[0, 0, 0, 0]).unwrap();
        let r = P256Fr::from_u64_arr(&[1, 0, 0, 0]).unwrap();
        let s = P256Fr::from_u64_arr(&[1, 0, 0, 0]).unwrap();
        let result = ecdsa_verify(zero, r, s, g);
        assert!(matches!(result, Err(P256Error::ZeroMessageHash)));
    }

    /// Verify that a cross-cancellation attack with correlated forged advice is
    /// rejected by the independent per-point Shamir checks.
    ///
    /// The attack supplies R1=2G, R2=3G with correlated decompositions that
    /// satisfy a combined 4-scalar equation but NOT the independent per-point
    /// equations. With the old combined check, the weighted sum cancels:
    ///   1*G + (-2)*G - 1*(2G) + 1*(3G) = O
    /// But each independent check fails:
    ///   a1*G - b1*R1 = 1*G - 1*(2G) = -G ≠ O
    #[test]
    #[should_panic(expected = "proof spoiled")]
    fn test_cross_cancellation_attack_rejected() {
        use crate::sdk::{verify_ecdsa_inner, P256Fr, P256Point};

        let g = P256Point::generator();
        let two = P256Fr::from_u64_arr(&[2, 0, 0, 0]).unwrap();
        let one = P256Fr::from_u64_arr(&[1, 0, 0, 0]).unwrap();

        let two_g = g.double();
        let three_g = g.double_and_add(&g);
        let five_g = two_g.double_and_add(&g);

        let n = limbs_to_biguint(&P256_ORDER);
        let mut r_big = limbs_to_biguint(&five_g.x().e());
        if r_big >= n {
            r_big -= &n;
        }
        let r_fr = P256Fr::from_u64_arr(&biguint_to_limbs(&r_big)).unwrap();

        let _ = verify_ecdsa_inner(
            &one, &two, &r_fr, &g, two_g, 1, false, 1, false, // R1=2G, a1=1, b1=1
            three_g, 2, true, 1, true, // R2=3G, a2=-2, b2=-1
        );
    }

    /// Verify that a zero GLV decomposition (a=0, b=0) is rejected.
    ///
    /// A malicious prover could supply a=0, b=0 as the Fake GLV decomposition,
    /// which trivially satisfies `b*u = a (mod n)` for any u and collapses the
    /// Shamir MSM to the identity, leaving the prover-supplied points R1/R2
    /// unconstrained. This test exercises the `b_val == 0` guard with an attack
    /// vector that would otherwise pass all other checks.
    #[test]
    #[should_panic(expected = "proof spoiled")]
    fn test_zero_glv_decomposition_rejected() {
        use crate::sdk::{verify_ecdsa_inner, P256Fr, P256Point};

        let g = P256Point::generator();
        let r_fr = P256Fr::from_u64_arr(&[1, 0, 0, 0]).unwrap();
        let s_fr = P256Fr::from_u64_arr(&[1, 0, 0, 0]).unwrap();
        let z_fr = P256Fr::from_u64_arr(&[1, 0, 0, 0]).unwrap();

        let u1 = z_fr.div_assume_nonzero(&s_fr);
        let u2 = r_fr.div_assume_nonzero(&s_fr);

        // Attack vector: all-zero decomposition with R1 = G (on-curve, x mod n = r)
        // and R2 = infinity. With b=0, the Shamir MSM collapses to O, the
        // decomposition check 0*u = 0 passes, and (R1 + R2).x mod n = G.x mod n.
        let _ = verify_ecdsa_inner(
            &u1,
            &u2,
            &r_fr,
            &g,
            g.clone(),
            0,
            false,
            0,
            false, // r1=G, a1=0, b1=0
            P256Point::infinity(),
            0,
            false,
            0,
            false, // r2=O, a2=0, b2=0
        );
    }
}
