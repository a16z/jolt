mod p256_tests {
    use crate::sdk::P256PointExt;
    use crate::{
        INLINE_OPCODE, P256_DIVQ_FUNCT3, P256_DIVR_FUNCT3, P256_FUNCT7, P256_MULQ_FUNCT3,
        P256_MULR_FUNCT3, P256_SQUAREQ_FUNCT3, P256_SQUARER_FUNCT3,
    };
    use crate::{P256_CURVE_B, P256_GENERATOR_X, P256_GENERATOR_Y, P256_MODULUS, P256_ORDER};
    use num_bigint::BigUint;
    use tracer::emulator::cpu::Xlen;
    use tracer::utils::inline_test_harness::{InlineMemoryLayout, InlineTestHarness};

    // Helper: convert [u64; 4] little-endian limbs to BigUint
    fn limbs_to_biguint(limbs: &[u64; 4]) -> BigUint {
        let mut bytes = [0u8; 32];
        for (i, &limb) in limbs.iter().enumerate() {
            bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
        }
        BigUint::from_bytes_le(&bytes)
    }

    // Helper: convert BigUint back to [u64; 4] little-endian limbs
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

    // Reference modular arithmetic via BigUint
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

    // Inline harness wrappers
    fn assert_mulq_trace_equiv(a: &[u64; 4], b: &[u64; 4]) {
        let expected = bigint_mulmod(a, b, &P256_MODULUS);
        let layout = InlineMemoryLayout::two_inputs(32, 32, 32);
        let mut harness = InlineTestHarness::new(layout, Xlen::Bit64);
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
        let mut harness = InlineTestHarness::new(layout, Xlen::Bit64);
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
        let mut harness = InlineTestHarness::new(layout, Xlen::Bit64);
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

    fn assert_mulr_trace_equiv(a: &[u64; 4], b: &[u64; 4]) {
        let expected = bigint_mulmod(a, b, &P256_ORDER);
        let layout = InlineMemoryLayout::two_inputs(32, 32, 32);
        let mut harness = InlineTestHarness::new(layout, Xlen::Bit64);
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
        let mut harness = InlineTestHarness::new(layout, Xlen::Bit64);
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
        let mut harness = InlineTestHarness::new(layout, Xlen::Bit64);
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

    // 1. test_p256_mulq -- base field multiplication
    #[test]
    fn test_p256_mulq() {
        // 7 * 7 = 49  (small values, no reduction)
        let seven = [7u64, 0, 0, 0];
        assert_mulq_trace_equiv(&seven, &seven);

        // Generator.x * Generator.x -- compare against Python-computed value
        // GX^2 mod p = 0x98f6b84d29bef2b281819a5e0e3690d833b699495d694dd1002ae56c426b3f8c
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

        // Near-modulus values: (p-1) * (p-1) and (p-1) * 2
        let pm1: [u64; 4] = [
            P256_MODULUS[0].wrapping_sub(1),
            P256_MODULUS[1],
            P256_MODULUS[2],
            P256_MODULUS[3],
        ];
        assert_mulq_trace_equiv(&pm1, &pm1);

        let two = [2u64, 0, 0, 0];
        assert_mulq_trace_equiv(&pm1, &two);

        // Arbitrary large values
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

        // Small values
        let a = [1u64, 2, 3, 4];
        let b = [5u64, 6, 7, 8];
        assert_mulq_trace_equiv(&a, &b);

        // Identity: a * 1 = a
        let one = [1u64, 0, 0, 0];
        assert_mulq_trace_equiv(&P256_GENERATOR_X, &one);
    }

    // 2. test_p256_squareq -- base field squaring
    #[test]
    fn test_p256_squareq() {
        // Small value
        let seven = [7u64, 0, 0, 0];
        assert_squareq_trace_equiv(&seven);

        // Generator x-coordinate
        assert_squareq_trace_equiv(&P256_GENERATOR_X);

        // Generator y-coordinate
        assert_squareq_trace_equiv(&P256_GENERATOR_Y);

        // Near-modulus: (p-1)^2
        let pm1: [u64; 4] = [
            P256_MODULUS[0].wrapping_sub(1),
            P256_MODULUS[1],
            P256_MODULUS[2],
            P256_MODULUS[3],
        ];
        assert_squareq_trace_equiv(&pm1);

        // Arbitrary values
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

    // 3. test_p256_divq -- base field division
    #[test]
    fn test_p256_divq() {
        // a / 1 = a
        let one = [1u64, 0, 0, 0];
        assert_divq_trace_equiv(&P256_GENERATOR_X, &one);

        // a * b then result / b = a
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

        // Arbitrary test vectors
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

    // 4. test_p256_mulr -- scalar field multiplication
    #[test]
    fn test_p256_mulr() {
        // Small values
        let a = [0u64, 0, 0, 1];
        let b = [0u64, 1, 0, 0];
        assert_mulr_trace_equiv(&a, &b);

        // Arbitrary large values
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

        // Near-order: (n-1) * (n-1)
        let nm1: [u64; 4] = [
            P256_ORDER[0].wrapping_sub(1),
            P256_ORDER[1],
            P256_ORDER[2],
            P256_ORDER[3],
        ];
        assert_mulr_trace_equiv(&nm1, &nm1);

        // Identity: a * 1 = a
        let one = [1u64, 0, 0, 0];
        let a = [
            0xAAAAAAAAAAAAAAAA,
            0xBBBBBBBBBBBBBBBB,
            0xCCCCCCCCCCCCCCCC,
            0x1111111111111111,
        ];
        assert_mulr_trace_equiv(&a, &one);
    }

    // 5. test_p256_squarer -- scalar field squaring
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

        // Near-order: (n-1)^2
        let nm1: [u64; 4] = [
            P256_ORDER[0].wrapping_sub(1),
            P256_ORDER[1],
            P256_ORDER[2],
            P256_ORDER[3],
        ];
        assert_squarer_trace_equiv(&nm1);
    }

    // 6. test_p256_divr -- scalar field division
    #[test]
    fn test_p256_divr() {
        // a / 1 = a
        let one = [1u64, 0, 0, 0];
        let a = [
            0x123456789ABCDEF0,
            0x0FEDCBA987654321,
            0x1111111111111111,
            0x2222222222222222,
        ];
        assert_divr_trace_equiv(&a, &one);

        // a * b then result / b = a
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

        // Arbitrary test vectors
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

    // 7. test_p256_point_on_curve -- verify generator is on y^2 = x^3 - 3x + b
    #[test]
    fn test_p256_point_on_curve() {
        let p = limbs_to_biguint(&P256_MODULUS);
        let gx = limbs_to_biguint(&P256_GENERATOR_X);
        let gy = limbs_to_biguint(&P256_GENERATOR_Y);
        let b = limbs_to_biguint(&P256_CURVE_B);

        // LHS: y^2 mod p
        let lhs = gy.modpow(&BigUint::from(2u64), &p);

        // RHS: x^3 - 3x + b mod p
        let x3 = gx.modpow(&BigUint::from(3u64), &p);
        let three_x = (&gx * BigUint::from(3u64)) % &p;
        // x^3 - 3x + b mod p  =  (x^3 + p - 3x + b) mod p
        let rhs = (x3 + &p - &three_x + &b) % &p;

        assert_eq!(lhs, rhs, "P-256 generator is not on the curve");

        // Also verify via the Python-computed GX^2 intermediate
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

    // 8. test_p256_point_add_double -- point arithmetic via BigUint reference
    #[test]
    fn test_p256_point_add_double() {
        let p = limbs_to_biguint(&P256_MODULUS);
        let gx = limbs_to_biguint(&P256_GENERATOR_X);
        let gy = limbs_to_biguint(&P256_GENERATOR_Y);

        // a = p - 3 (P-256 uses a = -3)
        let a_coeff = &p - BigUint::from(3u64);

        // ---- Point doubling: compute 2G ----
        // s = (3 * x^2 + a) / (2 * y) mod p
        let x2 = gx.modpow(&BigUint::from(2u64), &p);
        let numerator = (BigUint::from(3u64) * &x2 + &a_coeff) % &p;
        let denominator = (BigUint::from(2u64) * &gy) % &p;
        let denom_inv = denominator.modpow(&(&p - BigUint::from(2u64)), &p);
        let s = (&numerator * &denom_inv) % &p;

        // x3 = s^2 - 2*x mod p
        let s2 = s.modpow(&BigUint::from(2u64), &p);
        let two_gx = (BigUint::from(2u64) * &gx) % &p;
        let x3 = (&s2 + &p - &two_gx) % &p;

        // y3 = s * (x - x3) - y mod p
        let diff = if gx >= x3 {
            (&gx - &x3) % &p
        } else {
            (&gx + &p - &x3) % &p
        };
        let y3 = (&s * &diff % &p + &p - &gy) % &p;

        // ---- Verify 2G is on the curve ----
        let lhs_2g = y3.modpow(&BigUint::from(2u64), &p);
        let x3_cubed = x3.modpow(&BigUint::from(3u64), &p);
        let b = limbs_to_biguint(&crate::P256_CURVE_B);
        let rhs_2g = (&x3_cubed + &p - (BigUint::from(3u64) * &x3 % &p) + &b) % &p;
        assert_eq!(lhs_2g, rhs_2g, "2G is not on the P-256 curve");

        // ---- Point addition: G + 2G = 3G ----
        // Use the standard addition formula with different x-coordinates
        let dx = if x3 >= gx {
            (&x3 - &gx) % &p
        } else {
            (&x3 + &p - &gx) % &p
        };
        let dy = if y3 >= gy {
            (&y3 - &gy) % &p
        } else {
            (&y3 + &p - &gy) % &p
        };
        let dx_inv = dx.modpow(&(&p - BigUint::from(2u64)), &p);
        let s_add = (&dy * &dx_inv) % &p;

        let s_add_sq = s_add.modpow(&BigUint::from(2u64), &p);
        let x_3g = (&s_add_sq + &p + &p - &gx - &x3) % &p;
        let diff_3g = if gx >= x_3g {
            (&gx - &x_3g) % &p
        } else {
            (&gx + &p - &x_3g) % &p
        };
        let y_3g = (&s_add * &diff_3g % &p + &p - &gy) % &p;

        // Verify 3G is on the curve
        let lhs_3g = y_3g.modpow(&BigUint::from(2u64), &p);
        let x_3g_cubed = x_3g.modpow(&BigUint::from(3u64), &p);
        let rhs_3g = (&x_3g_cubed + &p - (BigUint::from(3u64) * &x_3g % &p) + &b) % &p;
        assert_eq!(lhs_3g, rhs_3g, "3G is not on the P-256 curve");

        // Verify 2G and 3G have different coordinates from G
        assert_ne!(
            biguint_to_limbs(&x3),
            P256_GENERATOR_X,
            "2G.x should differ from G.x"
        );
        assert_ne!(
            biguint_to_limbs(&x_3g),
            P256_GENERATOR_X,
            "3G.x should differ from G.x"
        );
    }

    // 9. test_p256_ecdsa_verify -- full ECDSA verification with known vector
    //
    // Test vector derived from RFC 6979 private key (NIST P-256):
    //   Private key d  = 0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721
    //   Message hash z = SHA-256("sample") with RFC 6979 deterministic nonce
    //
    // We use a fixed known-good (r, s, Qx, Qy, z) tuple and verify
    // that u1*G + u2*Q has x-coordinate equal to r (mod n).
    #[test]
    fn test_p256_ecdsa_verify() {
        let p = limbs_to_biguint(&P256_MODULUS);
        let n = limbs_to_biguint(&P256_ORDER);
        let gx = limbs_to_biguint(&P256_GENERATOR_X);
        let gy = limbs_to_biguint(&P256_GENERATOR_Y);
        let b = limbs_to_biguint(&P256_CURVE_B);
        let a_coeff = &p - BigUint::from(3u64);

        // Helper closures for point operations on the curve
        let point_double = |px: &BigUint, py: &BigUint| -> (BigUint, BigUint) {
            if py.bits() == 0 {
                return (BigUint::from(0u64), BigUint::from(0u64));
            }
            let x2 = px.modpow(&BigUint::from(2u64), &p);
            let num = (BigUint::from(3u64) * &x2 + &a_coeff) % &p;
            let den = (BigUint::from(2u64) * py) % &p;
            let den_inv = den.modpow(&(&p - BigUint::from(2u64)), &p);
            let s = (&num * &den_inv) % &p;
            let s2 = s.modpow(&BigUint::from(2u64), &p);
            let x3 = (&s2 + &p + &p - px - px) % &p;
            let diff = (px + &p - &x3) % &p;
            let y3 = (&s * &diff % &p + &p - py) % &p;
            (x3, y3)
        };

        let point_add =
            |p1x: &BigUint, p1y: &BigUint, p2x: &BigUint, p2y: &BigUint| -> (BigUint, BigUint) {
                let zero = BigUint::from(0u64);
                if p1x == &zero && p1y == &zero {
                    return (p2x.clone(), p2y.clone());
                }
                if p2x == &zero && p2y == &zero {
                    return (p1x.clone(), p1y.clone());
                }
                if p1x == p2x && p1y == p2y {
                    return point_double(p1x, p1y);
                }
                if p1x == p2x {
                    return (zero.clone(), zero);
                }
                let dx = (p2x + &p - p1x) % &p;
                let dy = (p2y + &p - p1y) % &p;
                let dx_inv = dx.modpow(&(&p - BigUint::from(2u64)), &p);
                let s = (&dy * &dx_inv) % &p;
                let s2 = s.modpow(&BigUint::from(2u64), &p);
                let x3 = (&s2 + &p + &p - p1x - p2x) % &p;
                let diff = (p1x + &p - &x3) % &p;
                let y3 = (&s * &diff % &p + &p - p1y) % &p;
                (x3, y3)
            };

        // Scalar multiplication: k * P using double-and-add
        let scalar_mul = |k: &BigUint, px: &BigUint, py: &BigUint| -> (BigUint, BigUint) {
            let zero = BigUint::from(0u64);
            let mut rx = zero.clone();
            let mut ry = zero;
            let bits = k.bits();
            for i in (0..bits).rev() {
                let (drx, dry) = point_double(&rx, &ry);
                rx = drx;
                ry = dry;
                if k.bit(i) {
                    let (arx, ary) = point_add(&rx, &ry, px, py);
                    rx = arx;
                    ry = ary;
                }
            }
            (rx, ry)
        };

        // Known test vector (derived from RFC 6979, P-256):
        // Private key d:
        let d_bytes: [u8; 32] = [
            0xC9, 0xAF, 0xA9, 0xD8, 0x45, 0xBA, 0x75, 0x16, 0x6B, 0x5C, 0x21, 0x57, 0x67, 0xB1,
            0xD6, 0x93, 0x4E, 0x50, 0xC3, 0xDB, 0x36, 0xE8, 0x9B, 0x12, 0x7B, 0x8A, 0x62, 0x2B,
            0x12, 0x0F, 0x67, 0x21,
        ];
        let d = BigUint::from_bytes_be(&d_bytes);

        // Q = d * G (public key)
        let (qx, qy) = scalar_mul(&d, &gx, &gy);

        // Verify Q is on the curve
        let lhs_q = qy.modpow(&BigUint::from(2u64), &p);
        let qx3 = qx.modpow(&BigUint::from(3u64), &p);
        let rhs_q = (&qx3 + &p - (BigUint::from(3u64) * &qx % &p) + &b) % &p;
        assert_eq!(lhs_q, rhs_q, "Public key Q is not on the curve");

        // Use a known message hash z:
        let z_bytes: [u8; 32] = [
            0x48, 0x47, 0xBE, 0x4A, 0xC2, 0x1F, 0xE6, 0x8A, 0x06, 0xD6, 0x36, 0x4B, 0xD7, 0x84,
            0x67, 0xC1, 0x83, 0xF3, 0x0A, 0x85, 0x7A, 0xD8, 0xF6, 0x56, 0x21, 0x9F, 0x7C, 0x40,
            0x30, 0x7C, 0x8E, 0xDF,
        ];
        let z = BigUint::from_bytes_be(&z_bytes);

        // Sign with known nonce k (RFC 6979 deterministic for this key + message):
        // We compute r, s directly for a self-contained test.
        let k_nonce_bytes: [u8; 32] = [
            0xA6, 0xE3, 0xC5, 0x7D, 0xD0, 0x1A, 0xBE, 0x90, 0x08, 0x65, 0x38, 0x39, 0x83, 0x55,
            0xDD, 0x4C, 0x3B, 0x17, 0xAA, 0x87, 0x33, 0x82, 0xB0, 0xF2, 0x4D, 0x61, 0x29, 0x49,
            0x3D, 0x8A, 0xAD, 0x60,
        ];
        let k_nonce = BigUint::from_bytes_be(&k_nonce_bytes);

        // R = k * G
        let (rx, _ry) = scalar_mul(&k_nonce, &gx, &gy);
        let r = &rx % &n;
        assert_ne!(r, BigUint::from(0u64), "r must not be zero");

        // s = k^{-1} * (z + r*d) mod n
        let k_inv = k_nonce.modpow(&(&n - BigUint::from(2u64)), &n);
        let s = (&k_inv * ((&z + &r * &d) % &n)) % &n;
        assert_ne!(s, BigUint::from(0u64), "s must not be zero");

        // ---- ECDSA verification ----
        // u1 = z * s^{-1} mod n
        let s_inv = s.modpow(&(&n - BigUint::from(2u64)), &n);
        let u1 = (&z * &s_inv) % &n;
        let u2 = (&r * &s_inv) % &n;

        // R' = u1*G + u2*Q
        let (r1x, r1y) = scalar_mul(&u1, &gx, &gy);
        let (r2x, r2y) = scalar_mul(&u2, &qx, &qy);
        let (rpx, _rpy) = point_add(&r1x, &r1y, &r2x, &r2y);

        // Verify: R'.x mod n == r
        let rx_mod_n = &rpx % &n;
        assert_eq!(rx_mod_n, r, "ECDSA verification failed: R'.x mod n != r");
    }

    /// Test the actual `ecdsa_verify()` function from sdk.rs (with Fake GLV).
    /// Uses the same test vector derived from the RFC 6979 private key as above.
    #[test]
    fn test_p256_ecdsa_verify_sdk() {
        use crate::sdk::{ecdsa_verify, P256Fr, P256Point};

        let p = limbs_to_biguint(&P256_MODULUS);
        let n = limbs_to_biguint(&P256_ORDER);
        let gx = limbs_to_biguint(&P256_GENERATOR_X);
        let gy = limbs_to_biguint(&P256_GENERATOR_Y);

        // BigUint point operations for key/sig generation
        let point_double = |px: &BigUint, py: &BigUint| -> (BigUint, BigUint) {
            if py.bits() == 0 {
                return (BigUint::from(0u64), BigUint::from(0u64));
            }
            let a_coeff = &p - BigUint::from(3u64);
            let x2 = px.modpow(&BigUint::from(2u64), &p);
            let num = (BigUint::from(3u64) * &x2 + &a_coeff) % &p;
            let den = (BigUint::from(2u64) * py) % &p;
            let den_inv = den.modpow(&(&p - BigUint::from(2u64)), &p);
            let s = (&num * &den_inv) % &p;
            let s2 = s.modpow(&BigUint::from(2u64), &p);
            let x3 = (&s2 + &p + &p - px - px) % &p;
            let diff = (px + &p - &x3) % &p;
            let y3 = (&s * &diff % &p + &p - py) % &p;
            (x3, y3)
        };
        let point_add =
            |p1x: &BigUint, p1y: &BigUint, p2x: &BigUint, p2y: &BigUint| -> (BigUint, BigUint) {
                let zero = BigUint::from(0u64);
                if p1x == &zero && p1y == &zero {
                    return (p2x.clone(), p2y.clone());
                }
                if p2x == &zero && p2y == &zero {
                    return (p1x.clone(), p1y.clone());
                }
                if p1x == p2x && p1y == p2y {
                    return point_double(p1x, p1y);
                }
                if p1x == p2x {
                    return (zero.clone(), zero);
                }
                let dx = (p2x + &p - p1x) % &p;
                let dy = (p2y + &p - p1y) % &p;
                let dx_inv = dx.modpow(&(&p - BigUint::from(2u64)), &p);
                let s = (&dy * &dx_inv) % &p;
                let s2 = s.modpow(&BigUint::from(2u64), &p);
                let x3 = (&s2 + &p + &p - p1x - p2x) % &p;
                let diff = (p1x + &p - &x3) % &p;
                let y3 = (&s * &diff % &p + &p - p1y) % &p;
                (x3, y3)
            };
        let scalar_mul = |k: &BigUint, px: &BigUint, py: &BigUint| -> (BigUint, BigUint) {
            let zero = BigUint::from(0u64);
            let (mut rx, mut ry) = (zero.clone(), zero);
            for i in (0..k.bits()).rev() {
                let (drx, dry) = point_double(&rx, &ry);
                rx = drx;
                ry = dry;
                if k.bit(i) {
                    let (arx, ary) = point_add(&rx, &ry, px, py);
                    rx = arx;
                    ry = ary;
                }
            }
            (rx, ry)
        };

        // Generate key and signature
        let d_bytes: [u8; 32] = [
            0xC9, 0xAF, 0xA9, 0xD8, 0x45, 0xBA, 0x75, 0x16, 0x6B, 0x5C, 0x21, 0x57, 0x67, 0xB1,
            0xD6, 0x93, 0x4E, 0x50, 0xC3, 0xDB, 0x36, 0xE8, 0x9B, 0x12, 0x7B, 0x8A, 0x62, 0x2B,
            0x12, 0x0F, 0x67, 0x21,
        ];
        let d = BigUint::from_bytes_be(&d_bytes);
        let (qx, qy) = scalar_mul(&d, &gx, &gy);
        let z_bytes: [u8; 32] = [
            0x48, 0x47, 0xBE, 0x4A, 0xC2, 0x1F, 0xE6, 0x8A, 0x06, 0xD6, 0x36, 0x4B, 0xD7, 0x84,
            0x67, 0xC1, 0x83, 0xF3, 0x0A, 0x85, 0x7A, 0xD8, 0xF6, 0x56, 0x21, 0x9F, 0x7C, 0x40,
            0x30, 0x7C, 0x8E, 0xDF,
        ];
        let z = BigUint::from_bytes_be(&z_bytes);
        let k_nonce_bytes: [u8; 32] = [
            0xA6, 0xE3, 0xC5, 0x7D, 0xD0, 0x1A, 0xBE, 0x90, 0x08, 0x65, 0x38, 0x39, 0x83, 0x55,
            0xDD, 0x4C, 0x3B, 0x17, 0xAA, 0x87, 0x33, 0x82, 0xB0, 0xF2, 0x4D, 0x61, 0x29, 0x49,
            0x3D, 0x8A, 0xAD, 0x60,
        ];
        let k_nonce = BigUint::from_bytes_be(&k_nonce_bytes);
        let (rx_big, _) = scalar_mul(&k_nonce, &gx, &gy);
        let r_big = &rx_big % &n;
        let k_inv = k_nonce.modpow(&(&n - BigUint::from(2u64)), &n);
        let s_big = (&k_inv * ((&z + &r_big * &d) % &n)) % &n;

        // Convert to SDK types and call ecdsa_verify
        let q_point = P256Point::from_u64_arr(&{
            let qx_l = biguint_to_limbs(&qx);
            let qy_l = biguint_to_limbs(&qy);
            [
                qx_l[0], qx_l[1], qx_l[2], qx_l[3], qy_l[0], qy_l[1], qy_l[2], qy_l[3],
            ]
        })
        .unwrap();

        let result = ecdsa_verify(
            P256Fr::from_u64_arr(&biguint_to_limbs(&z)).unwrap(),
            P256Fr::from_u64_arr(&biguint_to_limbs(&r_big)).unwrap(),
            P256Fr::from_u64_arr(&biguint_to_limbs(&s_big)).unwrap(),
            q_point,
        );
        assert!(result.is_ok(), "ecdsa_verify failed: {:?}", result.err());
    }

    /// Test double_and_add when 2P + Q = O (the infinity edge case fix).
    #[test]
    fn test_double_and_add_infinity() {
        use crate::sdk::P256Point;
        let g = P256Point::generator();
        let two_g = g.double();
        let neg_two_g = two_g.neg();

        // 2*G + (-2G) = O
        let result = g.double_and_add(&neg_two_g);
        assert!(result.is_infinity(), "2G + (-2G) should be infinity");

        // Verify matches naive: double().add()
        let naive = g.double().add(&neg_two_g);
        assert!(naive.is_infinity(), "naive 2G + (-2G) should be infinity");
    }

    /// Test double_and_add edge cases: infinity inputs, P == Q, P == -Q.
    #[test]
    fn test_double_and_add_edge_cases() {
        use crate::sdk::P256Point;
        let g = P256Point::generator();
        let inf = P256Point::infinity();

        // infinity.double_and_add(Q) = Q
        let r = inf.double_and_add(&g);
        assert_eq!(r.x().e(), g.x().e());

        // P.double_and_add(infinity) = 2P
        let r = g.double_and_add(&inf);
        let expected = g.double();
        assert_eq!(r.x().e(), expected.x().e());

        // P.double_and_add(P) = 3P
        let r = g.double_and_add(&g);
        let expected = g.double().add(&g);
        assert_eq!(r.x().e(), expected.x().e());

        // P.double_and_add(-P) = P (since 2P + (-P) = P)
        let neg_g = g.neg();
        let r = g.double_and_add(&neg_g);
        assert_eq!(r.x().e(), g.x().e());
        assert_eq!(r.y().e(), g.y().e());
    }

    /// Negative ECDSA tests: invalid inputs should be rejected.
    #[test]
    fn test_ecdsa_verify_rejects_invalid() {
        use crate::sdk::{ecdsa_verify, P256Error, P256Fr, P256Point};

        let g = P256Point::generator();
        let z = P256Fr::from_u64_arr(&[1, 0, 0, 0]).unwrap();
        let r = P256Fr::from_u64_arr(&[1, 0, 0, 0]).unwrap();
        let s = P256Fr::from_u64_arr(&[1, 0, 0, 0]).unwrap();

        // Q = infinity → QAtInfinity
        let result = ecdsa_verify(z.clone(), r.clone(), s.clone(), P256Point::infinity());
        assert!(matches!(result, Err(P256Error::QAtInfinity)));

        // r = 0 → ROrSZero
        let zero = P256Fr::from_u64_arr(&[0, 0, 0, 0]).unwrap();
        let result = ecdsa_verify(z.clone(), zero.clone(), s.clone(), g.clone());
        assert!(matches!(result, Err(P256Error::ROrSZero)));

        // s = 0 → ROrSZero
        let result = ecdsa_verify(z.clone(), r.clone(), zero, g.clone());
        assert!(matches!(result, Err(P256Error::ROrSZero)));
    }
}
