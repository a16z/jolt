mod sequence_tests {
    use crate::sdk::Secp256k1Point;
    use crate::{
        Secp256k1Fq, Secp256k1Fr, INLINE_OPCODE, SECP256K1_DIVQ_ADV_FUNCT3, SECP256K1_DIVQ_FUNCT3,
        SECP256K1_FUNCT7, SECP256K1_MULQ_FUNCT3, SECP256K1_SQUAREQ_FUNCT3,
    };
    use ark_ff::{BigInt, Field, PrimeField};
    use ark_secp256k1::Fq;
    use num_bigint::BigUint as NBigUint;
    use num_integer::Integer;
    use tracer::emulator::cpu::Xlen;
    use tracer::utils::inline_test_harness::{InlineMemoryLayout, InlineTestHarness};

    fn assert_divq_trace_equiv(a: &[u64; 4], b: &[u64; 4]) {
        // get expected value
        let arr_to_fq = |arr: &[u64; 4]| Fq::new(BigInt(*arr));
        let expected = (arr_to_fq(b)
            .inverse()
            .expect("Attempted to invert zero in secp256k1 field")
            * arr_to_fq(a))
        .into_bigint()
        .0;
        // rs1=input1 (32 bytes), rs2=input2 (32 bytes), rs3=output (32 bytes)
        let layout = InlineMemoryLayout::two_inputs(32, 32, 32);
        let mut harness = InlineTestHarness::new(layout, Xlen::Bit64);
        harness.setup_registers();
        harness.load_input64(a);
        harness.load_input2_64(b);
        harness.execute_inline(InlineTestHarness::create_default_instruction(
            INLINE_OPCODE,
            SECP256K1_DIVQ_FUNCT3,
            SECP256K1_FUNCT7,
        ));
        let result_vec = harness.read_output64(4);
        let mut result = [0u64; 4];
        result.copy_from_slice(&result_vec);
        assert_eq!(result, expected, "secp256k1_divq result mismatch");
    }

    #[test]
    fn test_secp256k1_divq_direct_execution() {
        // arbitrary test vectors for direct execution
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
        let a = [1u64, 2u64, 3u64, 4u64];
        let b = [5u64, 6u64, 7u64, 8u64];
        assert_divq_trace_equiv(&a, &b);
        let a = [1u64, 1u64, 1u64, 1u64];
        let b = [1u64, 1u64, 1u64, 1u64];
        assert_divq_trace_equiv(&a, &b);
    }

    fn assert_mulq_trace_equiv(a: &[u64; 4], b: &[u64; 4]) {
        // get expected value
        let arr_to_fq = |arr: &[u64; 4]| Fq::new(BigInt(*arr));
        let expected = (arr_to_fq(a) * arr_to_fq(b)).into_bigint().0;
        // rs1=input1 (32 bytes), rs2=input2 (32 bytes), rs3=output (32 bytes)
        let layout = InlineMemoryLayout::two_inputs(32, 32, 32);
        let mut harness = InlineTestHarness::new(layout, Xlen::Bit64);
        harness.setup_registers();
        harness.load_input64(a);
        harness.load_input2_64(b);
        harness.execute_inline(InlineTestHarness::create_default_instruction(
            INLINE_OPCODE,
            SECP256K1_MULQ_FUNCT3,
            SECP256K1_FUNCT7,
        ));
        let result_vec = harness.read_output64(4);
        let mut result = [0u64; 4];
        result.copy_from_slice(&result_vec);
        assert_eq!(result, expected, "secp256k1_mulq result mismatch");
    }

    #[test]
    fn test_secp256k1_mulq_direct_execution() {
        // arbitrary test vectors for direct execution
        let a = [0u64, 0u64, 0u64, 1u64];
        let b = [0u64, 1u64, 0u64, 0u64];
        assert_mulq_trace_equiv(&a, &b);
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
        let a = [1u64, 2u64, 3u64, 4u64];
        let b = [5u64, 6u64, 7u64, 8u64];
        assert_mulq_trace_equiv(&a, &b);
        let a = [1u64, 1u64, 1u64, 1u64];
        let b = [1u64, 1u64, 1u64, 1u64];
        assert_mulq_trace_equiv(&a, &b);
    }

    fn assert_squareq_trace_equiv(a: &[u64; 4]) {
        // get expected value
        let arr_to_fq = |arr: &[u64; 4]| Fq::new(BigInt(*arr));
        let expected = (arr_to_fq(a) * arr_to_fq(a)).into_bigint().0;
        // rs1=input1 (32 bytes), rs2=input2 (32 bytes), rs3=output (32 bytes)
        let layout = InlineMemoryLayout::two_inputs(32, 32, 32);
        let mut harness = InlineTestHarness::new(layout, Xlen::Bit64);
        harness.setup_registers();
        harness.load_input64(a);
        harness.execute_inline(InlineTestHarness::create_default_instruction(
            INLINE_OPCODE,
            SECP256K1_SQUAREQ_FUNCT3,
            SECP256K1_FUNCT7,
        ));
        let result_vec = harness.read_output64(4);
        let mut result = [0u64; 4];
        result.copy_from_slice(&result_vec);
        assert_eq!(result, expected, "secp256k1_squareq result mismatch");
    }

    #[test]
    fn test_secp256k1_squareq_direct_execution() {
        // arbitrary test vectors for direct execution
        let a = [0u64, 0u64, 0u64, 1u64];
        assert_squareq_trace_equiv(&a);
        let a = [
            0x123456789ABCDEF0,
            0x0FEDCBA987654321,
            0x1111111111111111,
            0x2222222222222222,
        ];
        assert_squareq_trace_equiv(&a);
        let a = [1u64, 2u64, 3u64, 4u64];
        assert_squareq_trace_equiv(&a);
        let a = [1u64, 1u64, 1u64, 1u64];
        assert_squareq_trace_equiv(&a);
    }

    fn u128_point_mul(scalar: u128, point: &Secp256k1Point) -> Secp256k1Point {
        let mut res = Secp256k1Point::infinity();
        for i in (0..128).rev() {
            if (scalar >> i) & 1 == 1 {
                res = res.double_and_add(point);
            } else {
                res = res.double();
            }
        }
        res
    }

    fn fr_point_mul(scalar: &Secp256k1Fr, point: &Secp256k1Point) -> Secp256k1Point {
        let mut res = Secp256k1Point::infinity();
        let k = scalar.fr().into_bigint().0;
        for i in (0..256).rev() {
            if (k[i / 64] >> (i % 64)) & 1 == 1 {
                res = res.double_and_add(point);
            } else {
                res = res.double();
            }
        }
        res
    }

    #[test]
    fn test_endomorphism_consistency() {
        // get q and its endomorphism
        let q = Secp256k1Point::from_u64_arr(&[
            0x0012563f32ed0216,
            0xee00716af6a73670,
            0x91fc70e34e00e6c8,
            0xeeb6be8b9e68868b,
            0x4780de3d5fda972d,
            0xcb1b42d72491e47f,
            0xdc7f31262e4ba2b7,
            0xdc7b004d3bb2800d,
        ])
        .unwrap();
        let endo_q = q.endomorphism();
        // check that endo_q.y = q.y
        assert_eq!(endo_q.y(), q.y());
        // check that endo_q.x = beta * q.x
        let beta = Secp256k1Fq::from_u64_arr(&[
            0xc1396c28719501ee,
            0x9cf0497512f58995,
            0x6e64479eac3434e9,
            0x7ae96a2b657c0710,
        ])
        .unwrap();
        let expected_x = beta.mul(&q.x());
        assert_eq!(endo_q.x(), expected_x);
        // check that lambda * q == endo_q
        let lambda = Secp256k1Fr::from_u64_arr(&[
            0xdf02967c1b23bd72,
            0x122e22ea20816678,
            0xa5261c028812645a,
            0x5363ad4cc05c30e0,
        ])
        .unwrap();
        let endo_q_expected = fr_point_mul(&lambda, &q);
        assert_eq!(endo_q.x(), endo_q_expected.x());
        assert_eq!(endo_q.y(), endo_q_expected.y());
    }

    #[test]
    fn test_glv_decomposition() {
        // get q and its endomorphism
        let q = Secp256k1Point::from_u64_arr(&[
            0x0012563f32ed0216,
            0xee00716af6a73670,
            0x91fc70e34e00e6c8,
            0xeeb6be8b9e68868b,
            0x4780de3d5fda972d,
            0xcb1b42d72491e47f,
            0xdc7f31262e4ba2b7,
            0xdc7b004d3bb2800d,
        ])
        .unwrap();
        let endo_q = q.endomorphism();
        // and an arbitrary scalar k
        let k = Secp256k1Fr::from_u64_arr(&[
            0x1234567890ABCDEF,
            0x0FEDCBA987654321,
            0x1111111111111111,
            0x2222222222222222,
        ])
        .unwrap();
        // check that k * q == k1 * q + k2 * endo_q
        let expected = fr_point_mul(&k, &q);
        let decomp = Secp256k1Point::decompose_scalar(&k);
        let sq = if decomp[0].0 { q.neg() } else { q.clone() };
        let sq_endo = if decomp[1].0 {
            endo_q.neg()
        } else {
            endo_q.clone()
        };
        let p1 = u128_point_mul(decomp[0].1, &sq);
        let p2 = u128_point_mul(decomp[1].1, &sq_endo);
        let combined = p1.add(&p2);
        assert_eq!(combined.x().e(), expected.x().e());
        assert_eq!(combined.y().e(), expected.y().e());
    }

    #[test]
    fn test_ecdsa_verify() {
        let z = Secp256k1Fr::from_u64_arr(&[
            0x9088f7ace2efcde9,
            0xc484efe37a5380ee,
            0xa52e52d7da7dabfa,
            0xb94d27b9934d3e08,
        ])
        .unwrap();
        let r = Secp256k1Fr::from_u64_arr(&[
            0xb8fc413b4b967ed8,
            0x248d4b0b2829ab00,
            0x587f69296af3cd88,
            0x3a5d6a386e6cf7c0,
        ])
        .unwrap();
        let s = Secp256k1Fr::from_u64_arr(&[
            0x66a82f274e3dcafc,
            0x299a02486be40321,
            0x6212d714118f617e,
            0x9d452f63cf91018d,
        ])
        .unwrap();
        let q = Secp256k1Point::from_u64_arr(&[
            0x0012563f32ed0216,
            0xee00716af6a73670,
            0x91fc70e34e00e6c8,
            0xeeb6be8b9e68868b,
            0x4780de3d5fda972d,
            0xcb1b42d72491e47f,
            0xdc7f31262e4ba2b7,
            0xdc7b004d3bb2800d,
        ])
        .unwrap();
        assert!(crate::sdk::ecdsa_verify(z, r, s, q).is_ok());
    }

    /// Verify assumptions about Fq and Fr modulus limb structure used in
    /// `is_fq_non_canonical` and `is_fr_non_canonical`.
    #[test]
    fn test_modulus_limb_assumptions() {
        use ark_secp256k1::Fr;

        // Fq modulus p = 2^256 - 2^32 - 977
        // Expected: limbs[1], limbs[2], limbs[3] are all u64::MAX
        assert_eq!(
            Fq::MODULUS.0[3],
            u64::MAX,
            "Fq::MODULUS.0[3] should be u64::MAX"
        );
        assert_eq!(
            Fq::MODULUS.0[2],
            u64::MAX,
            "Fq::MODULUS.0[2] should be u64::MAX"
        );
        assert_eq!(
            Fq::MODULUS.0[1],
            u64::MAX,
            "Fq::MODULUS.0[1] should be u64::MAX"
        );

        // Fr modulus n (scalar field order)
        // Expected: only limbs[3] is u64::MAX
        assert_eq!(
            Fr::MODULUS.0[3],
            u64::MAX,
            "Fr::MODULUS.0[3] should be u64::MAX"
        );
        // limbs[2] should NOT be u64::MAX (it's 0xFFFFFFFFFFFFFFFE)
        assert_ne!(
            Fr::MODULUS.0[2],
            u64::MAX,
            "Fr::MODULUS.0[2] should NOT be u64::MAX"
        );
    }

    // helper function to convert from vector of u64 limbs to NBigUint
    fn limbs_to_nbiguint(limbs: &[u64]) -> NBigUint {
        let mut bytes = Vec::with_capacity(limbs.len() * 8);
        for &limb in limbs {
            for i in 0..8 {
                bytes.push(((limb >> (i * 8)) & 0xFF) as u8);
            }
        }
        NBigUint::from_bytes_le(&bytes)
    }

    // helper function to convert from NBigUint to vector of u64 limbs
    fn nbiguint_to_limbs(n: &NBigUint) -> Vec<u64> {
        let bytes = n.to_bytes_le();
        let mut limbs = vec![0u64; (bytes.len() + 7) / 8];
        for (i, byte) in bytes.iter().enumerate() {
            limbs[i / 8] |= (*byte as u64) << ((i % 8) * 8);
        }
        limbs
    }

    fn test_mul_helper_old(a: &[u64; 4], b: &[u64; 4]) {
        let expected = (Fq::new(BigInt(*a)) * Fq::new(BigInt(*b))).into_bigint().0;
        // s = a * b
        let mut s = [0u64; 8];
        let mut carry: u64;
        for i in 0..4 {
            carry = 0;
            for j in 0..4 {
                // (carry, s[i + j]) += self.e[i] * other.e[j] + carry
                let tmp = (s[i + j] as u128) + (a[i] as u128 * b[j] as u128) + (carry as u128);
                carry = (tmp >> 64) as u64;
                s[i + j] = tmp as u64;
            }
            s[i + 4] = carry;
        }
        // get w from inline
        let a_big: NBigUint = limbs_to_nbiguint(a);
        let b_big: NBigUint = limbs_to_nbiguint(b);
        let q_big: NBigUint = Fq::MODULUS.into();
        // compute floor(a * b / q)
        let quotient = (a_big * b_big).div_floor(&q_big);
        // convert back to limbs
        let mut w = nbiguint_to_limbs(&quotient);
        while w.len() < 4 {
            w.push(0u64);
        }
        // t = w * [2^256 - q]
        let mut t = [0u64; 5];
        let p = (1u64 << 32) + 977; // p = 2^256 - q
        carry = 0;
        for i in 0..4 {
            // (carry, t[i + j]) += w.e[i] * p + carry
            let tmp = (t[i] as u128) + (w[i] as u128 * p as u128) + (carry as u128);
            carry = (tmp >> 64) as u64;
            t[i] = tmp as u64;
        }
        t[4] = carry;
        // s += t
        carry = 0;
        for i in 0..5 {
            let tmp = (s[i] as u128) + (t[i] as u128) + (carry as u128);
            carry = (tmp >> 64) as u64;
            s[i] = tmp as u64;
        }
        for i in 5..8 {
            let tmp = (s[i] as u128) + (carry as u128);
            carry = (tmp >> 64) as u64;
            s[i] = tmp as u64;
        }
        // no additional carry allowed
        if carry != 0 {
            panic!("secp256k1_fq::mul: final carry nonzero");
        }
        // check that top 4 limbs match w
        if s[4] != w[0] || s[5] != w[1] || s[6] != w[2] || s[7] != w[3] {
            println!("a: {:?}", a);
            println!("b: {:?}", b);
            println!("s: {:?}", &s[4..8]);
            println!("w: {:?}", w);
            println!("actual  : {:?}", &s[0..4]);
            println!("expected: {:?}", expected);
            panic!("secp256k1_fq::mul: reduction check failed");
        }
        // get c from bottom 4 limbs
        let c = Secp256k1Fq::from_u64_arr(&s[0..4].try_into().unwrap());
        // ensure that c < q
        if c.is_err() {
            panic!("secp256k1_fq::mul: result non-canonical");
        }
        // check that output matches expected
        //println!("expected: {:?}", expected);
        //println!("computed: {:?}", c.clone().unwrap().e());
        assert!(
            expected == c.unwrap().e(),
            "secp256k1_fq::mul: result mismatch"
        );
    }

    // get low 64 bits of a*b
    #[inline(always)]
    fn mul_low(a: u64, b: u64) -> u64 {
        let ab = (a as u128) * (b as u128);
        ab as u64
    }

    // get high 64 bits of a*b
    #[inline(always)]
    fn mul_high(a: u64, b: u64) -> u64 {
        let ab = (a as u128) * (b as u128);
        (ab >> 64) as u64
    }

    // split u128 into low and high u64s
    #[inline(always)]
    fn split_u128(x: u128) -> (u64, u64) {
        let low = x as u64;
        let high = (x >> 64) as u64;
        (low, high)
    }

    fn test_mul_helper(a: &[u64; 4], b: &[u64; 4]) {
        let expected = (Fq::new(BigInt(*a)) * Fq::new(BigInt(*b))).into_bigint().0;
        // get w from inline
        let a_big: NBigUint = limbs_to_nbiguint(a);
        let b_big: NBigUint = limbs_to_nbiguint(b);
        let q_big: NBigUint = Fq::MODULUS.into();
        // compute floor(a * b / q)
        let quotient = (a_big * b_big).div_floor(&q_big);
        // convert back to limbs
        let mut w = nbiguint_to_limbs(&quotient);
        while w.len() < 4 {
            w.push(0u64);
        }
        // get constant p = 2^256 - q
        let p = (1u64 << 32) + 977;
        // s = a*b + w*p
        let mut s = [0u64; 8];
        // (limb 0, carry)
        (s[0], s[1]) = split_u128(mul_low(a[0], b[0]) as u128 + mul_low(w[0], p) as u128);
        // (limb 1, carry)
        (s[1], s[2]) = split_u128(
            s[1] as u128
                + mul_high(a[0], b[0]) as u128
                + mul_high(w[0], p) as u128
                + mul_low(a[0], b[1]) as u128
                + mul_low(a[1], b[0]) as u128
                + mul_low(w[1], p) as u128,
        );
        // (limb 2, carry)
        (s[2], s[3]) = split_u128(
            s[2] as u128
                + mul_high(a[0], b[1]) as u128
                + mul_high(a[1], b[0]) as u128
                + mul_high(w[1], p) as u128
                + mul_low(a[0], b[2]) as u128
                + mul_low(a[1], b[1]) as u128
                + mul_low(a[2], b[0]) as u128
                + mul_low(w[2], p) as u128,
        );
        // (limb 3, carry)
        (s[3], s[4]) = split_u128(
            s[3] as u128
                + mul_high(a[0], b[2]) as u128
                + mul_high(a[1], b[1]) as u128
                + mul_high(a[2], b[0]) as u128
                + mul_high(w[2], p) as u128
                + mul_low(a[0], b[3]) as u128
                + mul_low(a[1], b[2]) as u128
                + mul_low(a[2], b[1]) as u128
                + mul_low(a[3], b[0]) as u128
                + mul_low(w[3], p) as u128,
        );
        // (limb 4, carry)
        (s[4], s[5]) = split_u128(
            s[4] as u128
                + mul_high(a[0], b[3]) as u128
                + mul_high(a[1], b[2]) as u128
                + mul_high(a[2], b[1]) as u128
                + mul_high(a[3], b[0]) as u128
                + mul_high(w[3], p) as u128
                + mul_low(a[1], b[3]) as u128
                + mul_low(a[2], b[2]) as u128
                + mul_low(a[3], b[1]) as u128,
        );
        // (limb 5, carry)
        (s[5], s[6]) = split_u128(
            s[5] as u128
                + mul_high(a[1], b[3]) as u128
                + mul_high(a[2], b[2]) as u128
                + mul_high(a[3], b[1]) as u128
                + mul_low(a[2], b[3]) as u128
                + mul_low(a[3], b[2]) as u128,
        );
        // (limb 6, carry)
        (s[6], s[7]) = split_u128(
            s[6] as u128
                + mul_high(a[2], b[3]) as u128
                + mul_high(a[3], b[2]) as u128
                + mul_low(a[3], b[3]) as u128,
        );
        // (limb 7, carry)
        let carry: u64;
        (s[7], carry) = split_u128(s[7] as u128 + mul_high(a[3], b[3]) as u128);
        // no additional carry allowed
        if carry != 0 {
            panic!("secp256k1_fq::mul: final carry nonzero");
        }
        // check that top 4 limbs match w
        if s[4] != w[0] || s[5] != w[1] || s[6] != w[2] || s[7] != w[3] {
            println!("a: {:?}", a);
            println!("b: {:?}", b);
            println!("s: {:?}", &s[4..8]);
            println!("w: {:?}", w);
            println!("actual  : {:?}", &s[0..4]);
            println!("expected: {:?}", expected);
            panic!("secp256k1_fq::mul: reduction check failed");
        }
        // get c from bottom 4 limbs
        let c = Secp256k1Fq::from_u64_arr(&s[0..4].try_into().unwrap());
        // ensure that c < q
        if c.is_err() {
            panic!("secp256k1_fq::mul: result non-canonical");
        }
        // check that output matches expected
        //println!("expected: {:?}", expected);
        //println!("computed: {:?}", c.clone().unwrap().e());
        assert!(
            expected == c.unwrap().e(),
            "secp256k1_fq::mul: result mismatch"
        );
    }

    fn square_helper(a: u64, b: u64) -> (u128, u128) {
        let ab = (a as u128) * (b as u128);
        (((ab as u64) as u128) << 1, ((ab >> 64) << 1))
    }
    fn test_square_helper(a: &[u64; 4]) {
        let expected = (Fq::new(BigInt(*a)).square()).into_bigint().0;
        // get w from inline
        let a_big: NBigUint = limbs_to_nbiguint(a);
        let q_big: NBigUint = Fq::MODULUS.into();
        // compute floor(a * b / q)
        let quotient = (a_big.clone() * a_big).div_floor(&q_big);
        // convert back to limbs
        let mut w = nbiguint_to_limbs(&quotient);
        while w.len() < 4 {
            w.push(0u64);
        }
        // get constant p = 2^256 - q
        let p = (1u64 << 32) + 977;
        // s = a*b + w*p
        let mut s = [0u64; 8];
        // get offdiagonal products
        let (l01, h01) = square_helper(a[0], a[1]);
        let (l02, h02) = square_helper(a[0], a[2]);
        let (l03, h03) = square_helper(a[0], a[3]);
        let (l12, h12) = square_helper(a[1], a[2]);
        let (l13, h13) = square_helper(a[1], a[3]);
        let (l23, h23) = square_helper(a[2], a[3]);
        // (limb 0, carry)
        (s[0], s[1]) = split_u128(mul_low(a[0], a[0]) as u128 + mul_low(w[0], p) as u128);
        // (limb 1, carry)
        (s[1], s[2]) = split_u128(
            s[1] as u128
                + mul_high(a[0], a[0]) as u128
                + mul_high(w[0], p) as u128
                + l01
                + mul_low(w[1], p) as u128,
        );
        // (limb 2, carry)
        (s[2], s[3]) = split_u128(
            s[2] as u128
                + h01
                + mul_high(w[1], p) as u128
                + l02
                + mul_low(a[1], a[1]) as u128
                + mul_low(w[2], p) as u128,
        );
        // (limb 3, carry)
        (s[3], s[4]) = split_u128(
            s[3] as u128
                + h02
                + mul_high(a[1], a[1]) as u128
                + mul_high(w[2], p) as u128
                + l03
                + l12
                + mul_low(w[3], p) as u128,
        );
        // (limb 4, carry)
        (s[4], s[5]) = split_u128(
            s[4] as u128
                + h03
                + h12
                + mul_high(w[3], p) as u128
                + l13
                + mul_low(a[2], a[2]) as u128,
        );
        // (limb 5, carry)
        (s[5], s[6]) = split_u128(s[5] as u128 + h13 + mul_high(a[2], a[2]) as u128 + l23);
        // (limb 6, carry)
        (s[6], s[7]) = split_u128(s[6] as u128 + h23 as u128 + mul_low(a[3], a[3]) as u128);
        // (limb 7, carry)
        let carry: u64;
        (s[7], carry) = split_u128(s[7] as u128 + mul_high(a[3], a[3]) as u128);
        // no additional carry allowed
        if carry != 0 {
            panic!("secp256k1_fq::square: final carry nonzero");
        }
        // check that top 4 limbs match w
        if s[4] != w[0] || s[5] != w[1] || s[6] != w[2] || s[7] != w[3] {
            println!("a: {:?}", a);
            println!("s: {:?}", &s[4..8]);
            println!("w: {:?}", w);
            println!("actual  : {:?}", &s[0..4]);
            println!("expected: {:?}", expected);
            panic!("secp256k1_fq::square: reduction check failed");
        }
        // get c from bottom 4 limbs
        let c = Secp256k1Fq::from_u64_arr(&s[0..4].try_into().unwrap());
        // ensure that c < q
        if c.is_err() {
            panic!("secp256k1_fq::square: result non-canonical");
        }
        // check that output matches expected
        //println!("expected: {:?}", expected);
        //println!("computed: {:?}", c.clone().unwrap().e());
        assert!(
            expected == c.unwrap().e(),
            "secp256k1_fq::square: result mismatch"
        );
    }
    #[test]
    fn test_mul() {
        let a = [
            5152362328894379821u64,
            14635364905869501567u64,
            15888472050441626295u64,
            15887292442193592333u64,
        ];
        let b = [
            5152362328894379821u64,
            14635364905869501567u64,
            15888472050441626295u64,
            15887292442193592333u64,
        ];
        test_mul_helper(&a, &b);
        // arbitrary test vectors for direct execution
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
        test_mul_helper(&a, &b);
        let a = [1u64, 2u64, 3u64, 4u64];
        let b = [5u64, 6u64, 7u64, 8u64];
        test_mul_helper(&a, &b);
        let a = [1u64, 1u64, 1u64, 1u64];
        let b = [1u64, 1u64, 1u64, 1u64];
        test_mul_helper(&a, &b);
    }
    #[test]
    fn test_square() {
        let a = [
            5152362328894379821u64,
            14635364905869501567u64,
            15888472050441626295u64,
            15887292442193592333u64,
        ];
        test_square_helper(&a);
        // arbitrary test vectors for direct execution
        let a = [
            0x0FEDCBA987654321,
            0x123456789ABCDEF0,
            0x3333333333333333,
            0x4444444444444444,
        ];
        test_square_helper(&a);
        let a = [1u64, 2u64, 3u64, 4u64];
        test_square_helper(&a);
        let a = [1u64, 1u64, 1u64, 1u64];
        test_square_helper(&a);
    }
}
