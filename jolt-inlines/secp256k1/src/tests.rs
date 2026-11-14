mod sequence_tests {
    use crate::sdk::{decode_glv_sign_word, Secp256k1Point, Secp256k1PointExt};
    use crate::{
        Secp256k1Error, Secp256k1Fq, Secp256k1Fr, INLINE_OPCODE, SECP256K1_DIVQ_FUNCT3,
        SECP256K1_DIVR_FUNCT3, SECP256K1_FUNCT7, SECP256K1_MULQ_FUNCT3, SECP256K1_MULR_FUNCT3,
        SECP256K1_SQUAREQ_FUNCT3, SECP256K1_SQUARER_FUNCT3,
    };
    use ark_ff::{BigInt, Field, PrimeField};
    use ark_secp256k1::{Fq, Fr};
    use tracer::emulator::mmu::DRAM_BASE;
    use tracer::utils::inline_test_harness::{InlineMemoryLayout, InlineTestHarness};

    fn assert_divq_trace_equiv(a: &[u64; 4], b: &[u64; 4]) {
        let arr_to_fq = |arr: &[u64; 4]| Fq::new(BigInt(*arr));
        let expected = (arr_to_fq(b)
            .inverse()
            .expect("Attempted to invert zero in secp256k1 field")
            * arr_to_fq(a))
        .into_bigint()
        .0;
        let layout = InlineMemoryLayout::two_inputs(32, 32, 32);

        let mut harness = InlineTestHarness::new(layout);
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
        let arr_to_fq = |arr: &[u64; 4]| Fq::new(BigInt(*arr));
        let expected = (arr_to_fq(a) * arr_to_fq(b)).into_bigint().0;
        let layout = InlineMemoryLayout::two_inputs(32, 32, 32);

        let mut harness = InlineTestHarness::new(layout);
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
        let arr_to_fq = |arr: &[u64; 4]| Fq::new(BigInt(*arr));
        let expected = (arr_to_fq(a) * arr_to_fq(a)).into_bigint().0;
        let layout = InlineMemoryLayout::two_inputs(32, 32, 32);

        let mut harness = InlineTestHarness::new(layout);
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

    fn assert_divr_trace_equiv(a: &[u64; 4], b: &[u64; 4]) {
        let arr_to_fr = |arr: &[u64; 4]| Fr::new(BigInt(*arr));
        let expected = (arr_to_fr(b)
            .inverse()
            .expect("Attempted to invert zero in secp256k1 field")
            * arr_to_fr(a))
        .into_bigint()
        .0;
        let layout = InlineMemoryLayout::two_inputs(32, 32, 32);

        let mut harness = InlineTestHarness::new(layout);
        harness.setup_registers();
        harness.load_input64(a);
        harness.load_input2_64(b);
        harness.execute_inline(InlineTestHarness::create_default_instruction(
            INLINE_OPCODE,
            SECP256K1_DIVR_FUNCT3,
            SECP256K1_FUNCT7,
        ));
        let result_vec = harness.read_output64(4);
        let mut result = [0u64; 4];
        result.copy_from_slice(&result_vec);
        assert_eq!(result, expected, "secp256k1_divr result mismatch");
    }

    #[test]
    fn test_secp256k1_divr_direct_execution() {
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
        let a = [1u64, 2u64, 3u64, 4u64];
        let b = [5u64, 6u64, 7u64, 8u64];
        assert_divr_trace_equiv(&a, &b);
        let a = [1u64, 1u64, 1u64, 1u64];
        let b = [1u64, 1u64, 1u64, 1u64];
        assert_divr_trace_equiv(&a, &b);
    }

    /// Division with the result buffer aliasing the dividend (`rs3 == rs1`).
    /// The advised quotient must be stored only after every `VirtualAssertEQ`
    /// against the dividend has run, otherwise the checks compare the stored
    /// result against itself and any value would be accepted.
    fn assert_div_trace_equiv_aliased(funct3: u32, a: &[u64; 4], b: &[u64; 4], expected: [u64; 4]) {
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
            SECP256K1_FUNCT7,
        ));
        let result_vec = harness.read_output64(4);
        let mut result = [0u64; 4];
        result.copy_from_slice(&result_vec);
        assert_eq!(result, expected, "aliased div result mismatch");
    }

    #[test]
    fn test_secp256k1_div_aliased_dividend_and_result() {
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

        let expected_q = (Fq::new(BigInt(b)).inverse().unwrap() * Fq::new(BigInt(a)))
            .into_bigint()
            .0;
        assert_div_trace_equiv_aliased(SECP256K1_DIVQ_FUNCT3, &a, &b, expected_q);

        let expected_r = (Fr::new(BigInt(b)).inverse().unwrap() * Fr::new(BigInt(a)))
            .into_bigint()
            .0;
        assert_div_trace_equiv_aliased(SECP256K1_DIVR_FUNCT3, &a, &b, expected_r);
    }

    fn assert_mulr_trace_equiv(a: &[u64; 4], b: &[u64; 4]) {
        let arr_to_fr = |arr: &[u64; 4]| Fr::new(BigInt(*arr));
        let expected = (arr_to_fr(a) * arr_to_fr(b)).into_bigint().0;
        let layout = InlineMemoryLayout::two_inputs(32, 32, 32);

        let mut harness = InlineTestHarness::new(layout);
        harness.setup_registers();
        harness.load_input64(a);
        harness.load_input2_64(b);
        harness.execute_inline(InlineTestHarness::create_default_instruction(
            INLINE_OPCODE,
            SECP256K1_MULR_FUNCT3,
            SECP256K1_FUNCT7,
        ));
        let result_vec = harness.read_output64(4);
        let mut result = [0u64; 4];
        result.copy_from_slice(&result_vec);
        assert_eq!(result, expected, "secp256k1_mulr result mismatch");
    }

    #[test]
    fn test_secp256k1_mulr_direct_execution() {
        let a = [0u64, 0u64, 0u64, 1u64];
        let b = [0u64, 1u64, 0u64, 0u64];
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
        let a = [1u64, 2u64, 3u64, 4u64];
        let b = [5u64, 6u64, 7u64, 8u64];
        assert_mulr_trace_equiv(&a, &b);
        let a = [1u64, 1u64, 1u64, 1u64];
        let b = [1u64, 1u64, 1u64, 1u64];
        assert_mulr_trace_equiv(&a, &b);
    }

    fn assert_squarer_trace_equiv(a: &[u64; 4]) {
        let arr_to_fr = |arr: &[u64; 4]| Fr::new(BigInt(*arr));
        let expected = (arr_to_fr(a) * arr_to_fr(a)).into_bigint().0;
        let layout = InlineMemoryLayout::two_inputs(32, 32, 32);

        let mut harness = InlineTestHarness::new(layout);
        harness.setup_registers();
        harness.load_input64(a);
        harness.execute_inline(InlineTestHarness::create_default_instruction(
            INLINE_OPCODE,
            SECP256K1_SQUARER_FUNCT3,
            SECP256K1_FUNCT7,
        ));
        let result_vec = harness.read_output64(4);
        let mut result = [0u64; 4];
        result.copy_from_slice(&result_vec);
        assert_eq!(result, expected, "secp256k1_squarer result mismatch");
    }

    #[test]
    fn test_secp256k1_squarer_direct_execution() {
        let a = [0u64, 0u64, 0u64, 1u64];
        assert_squarer_trace_equiv(&a);
        let a = [
            0x123456789ABCDEF0,
            0x0FEDCBA987654321,
            0x1111111111111111,
            0x2222222222222222,
        ];
        assert_squarer_trace_equiv(&a);
        let a = [1u64, 2u64, 3u64, 4u64];
        assert_squarer_trace_equiv(&a);
        let a = [1u64, 1u64, 1u64, 1u64];
        assert_squarer_trace_equiv(&a);
    }

    /// `double_and_add` divides by `x_self - x_{self+other}`, which is zero when
    /// `other == -2*self`. That case must return infinity instead of dividing by
    /// zero (which panics host advice generation and is unprovable in-guest).
    #[test]
    fn test_secp256k1_double_and_add_infinity() {
        let g = Secp256k1Point::generator();
        let neg_two_g = g.double().neg();

        let result = g.double_and_add(&neg_two_g);
        assert!(result.is_infinity(), "2G + (-2G) should be infinity");

        let naive = g.double().add(&neg_two_g);
        assert!(naive.is_infinity(), "naive 2G + (-2G) should be infinity");
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
        let k = scalar.e();
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
        assert_eq!(endo_q.y(), q.y());
        let beta = Secp256k1Fq::from_u64_arr(&[
            0xc1396c28719501ee,
            0x9cf0497512f58995,
            0x6e64479eac3434e9,
            0x7ae96a2b657c0710,
        ])
        .unwrap();
        let expected_x = beta.mul(&q.x());
        assert_eq!(endo_q.x(), expected_x);
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
        let k = Secp256k1Fr::from_u64_arr(&[
            0x1234567890ABCDEF,
            0x0FEDCBA987654321,
            0x1111111111111111,
            0x2222222222222222,
        ])
        .unwrap();
        let expected = fr_point_mul(&k, &q);
        let decomp = k.glv_decompose();
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
    fn test_decode_glv_sign_word() {
        assert!(!decode_glv_sign_word(0).unwrap());
        assert!(decode_glv_sign_word(1).unwrap());
        assert!(matches!(
            decode_glv_sign_word(2),
            Err(Secp256k1Error::InvalidGlvSignWord(2))
        ));
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

    #[test]
    fn test_ecdsa_verify_rejects_invalid() {
        use crate::sdk::{ecdsa_verify, Secp256k1Error};

        let g = Secp256k1Point::generator();
        let z = Secp256k1Fr::from_u64_arr(&[1, 0, 0, 0]).unwrap();
        let r = Secp256k1Fr::from_u64_arr(&[1, 0, 0, 0]).unwrap();
        let s = Secp256k1Fr::from_u64_arr(&[1, 0, 0, 0]).unwrap();

        let result = ecdsa_verify(z.clone(), r.clone(), s.clone(), Secp256k1Point::infinity());
        assert!(matches!(result, Err(Secp256k1Error::QAtInfinity)));

        let zero = Secp256k1Fr::from_u64_arr(&[0, 0, 0, 0]).unwrap();
        let result = ecdsa_verify(z.clone(), zero.clone(), s.clone(), g.clone());
        assert!(matches!(result, Err(Secp256k1Error::ROrSZero)));

        let result = ecdsa_verify(z.clone(), r.clone(), zero, g.clone());
        assert!(matches!(result, Err(Secp256k1Error::ROrSZero)));

        let bad_z = Secp256k1Fr::from_u64_arr_unchecked(&[u64::MAX; 4]);
        let result = ecdsa_verify(bad_z, r.clone(), s.clone(), g.clone());
        assert!(matches!(result, Err(Secp256k1Error::InvalidFrElement)));

        let bad_r = Secp256k1Fr::from_u64_arr_unchecked(&[u64::MAX; 4]);
        let result = ecdsa_verify(z.clone(), bad_r, s.clone(), g.clone());
        assert!(matches!(result, Err(Secp256k1Error::InvalidFrElement)));

        let bad_s = Secp256k1Fr::from_u64_arr_unchecked(&[u64::MAX; 4]);
        let result = ecdsa_verify(z.clone(), r.clone(), bad_s, g.clone());
        assert!(matches!(result, Err(Secp256k1Error::InvalidFrElement)));

        let bad_x = Secp256k1Fq::from_u64_arr_unchecked(&[u64::MAX; 4]);
        let bad_q = Secp256k1Point::new_unchecked(bad_x, g.y());
        let result = ecdsa_verify(z.clone(), r.clone(), s.clone(), bad_q);
        assert!(matches!(result, Err(Secp256k1Error::InvalidFqElement)));

        let bad_y = Secp256k1Fq::from_u64_arr_unchecked(&[u64::MAX; 4]);
        let bad_q = Secp256k1Point::new_unchecked(g.x(), bad_y);
        let result = ecdsa_verify(z.clone(), r.clone(), s.clone(), bad_q);
        assert!(matches!(result, Err(Secp256k1Error::InvalidFqElement)));

        let off_curve_y = Secp256k1Fq::from_u64_arr(&[1, 0, 0, 0]).unwrap();
        let bad_q = Secp256k1Point::new_unchecked(g.x(), off_curve_y);
        let result = ecdsa_verify(z.clone(), r.clone(), s.clone(), bad_q);
        assert!(matches!(result, Err(Secp256k1Error::NotOnCurve)));
    }

    #[test]
    fn test_ecdsa_verify_rejects_zero_hash_forgery() {
        use crate::sdk::{ecdsa_verify, Secp256k1Error};

        let q = Secp256k1Point::generator();
        let z = Secp256k1Fr::from_u64_arr(&[0; 4]).unwrap();
        let r = Secp256k1Fr::from_u64_arr(&q.x().e()).unwrap();
        let s = r.clone();

        // With z=0 and r=s=x(Q), the verification equation collapses to R=Q.
        let result = ecdsa_verify(z, r, s, q);
        assert!(matches!(result, Err(Secp256k1Error::ZeroMessageHash)));
    }

    /// Verify assumptions about Fq and Fr modulus limb structure used in
    /// `is_fq_non_canonical` and `is_fr_non_canonical`.
    #[test]
    fn test_modulus_limb_assumptions() {
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

        assert_eq!(
            Fr::MODULUS.0[3],
            u64::MAX,
            "Fr::MODULUS.0[3] should be u64::MAX"
        );
    }
}

#[test]
fn field_equality_and_zero_checks_every_bit() {
    use crate::{Secp256k1Fq, Secp256k1Fr};

    // Every one-bit value is canonical in both fields, including bit 255.
    let zero_q = Secp256k1Fq::from_u64_arr(&[0; 4]).unwrap();
    let zero_r = Secp256k1Fr::from_u64_arr(&[0; 4]).unwrap();
    assert!(zero_q.is_zero());
    assert!(zero_r.is_zero());
    for bit in 0..256 {
        let mut limbs = [0; 4];
        limbs[bit / 64] = 1 << (bit % 64);
        let q = Secp256k1Fq::from_u64_arr(&limbs).unwrap();
        let r = Secp256k1Fr::from_u64_arr(&limbs).unwrap();
        assert_eq!(q, q.clone());
        assert_eq!(r, r.clone());
        assert_ne!(q, zero_q);
        assert_ne!(r, zero_r);
        assert!(!q.is_zero());
        assert!(!r.is_zero());
        for other_bit in 0..bit {
            let mut other = [0; 4];
            other[other_bit / 64] = 1 << (other_bit % 64);
            assert_ne!(q, Secp256k1Fq::from_u64_arr(&other).unwrap());
            assert_ne!(r, Secp256k1Fr::from_u64_arr(&other).unwrap());
        }
    }
}

/// Runs the secp256k1 field inlines with advice that differs from the honest
/// advice. A wrong value must fail one of the sequence's assert rows; an
/// unreduced division result passes the sequence and fails the SDK check.
mod dishonest_advice_tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use ark_ff::{BigInt, BigInteger, PrimeField};
    use ark_secp256k1::{Fq, Fr};
    use jolt_inlines_sdk::host::{InlineAdvice, InlineOp, ModularDivisionAdvice, QuotientAdvice};
    use tracer::emulator::mmu::DRAM_BASE;
    use tracer::instruction::Instruction;
    use tracer::utils::inline_test_harness::{InlineMemoryLayout, InlineTestHarness};

    use crate::sequence_builder::{
        Secp256k1DivQ, Secp256k1DivR, Secp256k1MulQ, Secp256k1MulR, Secp256k1SquareQ,
        Secp256k1SquareR,
    };
    use crate::{Secp256k1Error, Secp256k1Fq, Secp256k1Fr};

    /// A divisor below 2^255, so that adding it to an honest quotient `w`
    /// still fits in four limbs.
    const DIVISOR: [u64; 4] = [
        0x0FED_CBA9_8765_4321,
        0x1234_5678_9ABC_DEF0,
        0x3333_3333_3333_3333,
        0x0444_4444_4444_4444,
    ];
    const FACTOR: [u64; 4] = [
        0x1234_5678_9ABC_DEF0,
        0x0FED_CBA9_8765_4321,
        0x1111_1111_1111_1111,
        0x2222_2222_2222_2222,
    ];

    /// Loads `a` and `b` as the inline's first and second operands. With
    /// `aliased`, the result pointer `rs3` equals `rs1`.
    fn harness(aliased: bool, a: &[u64; 4], b: &[u64; 4]) -> InlineTestHarness {
        let layout = InlineMemoryLayout::two_inputs(32, 32, 32);
        let layout = if aliased {
            InlineMemoryLayout {
                output_base: DRAM_BASE,
                ..layout
            }
        } else {
            layout
        };
        let mut harness = InlineTestHarness::new(layout);
        harness.setup_registers();
        harness.load_input64(a);
        harness.load_input2_64(b);
        harness
    }

    /// Executes `Op`'s rows one at a time, with the honest advice changed by
    /// `tamper` patched into the `VirtualAdvice` rows. Returns the row that
    /// panicked, or `None` if every row ran.
    fn run_with_advice<Op: InlineOp>(
        harness: &mut InlineTestHarness,
        tamper: impl FnOnce(&mut Op::Advice),
    ) -> Option<Instruction> {
        let inline =
            InlineTestHarness::create_default_instruction(Op::OPCODE, Op::FUNCT3, Op::FUNCT7);
        let mut advice =
            Op::build_advice(inline.operands, &mut harness.cpu).expect("honest advice");
        tamper(&mut advice);
        let mut values = advice.into_runtime_advice().expect("inline takes advice");
        for mut row in inline.inline_sequence(&harness.cpu.vr_allocator) {
            if let Instruction::VirtualAdvice(advice_row) = &mut row {
                advice_row.advice = values.pop_front().expect("too few advice values");
            }
            let step = catch_unwind(AssertUnwindSafe(|| {
                harness.execute_sequence(std::slice::from_ref(&row))
            }));
            if step.is_err() {
                return Some(row);
            }
        }
        assert!(values.is_empty(), "too many advice values");
        None
    }

    fn assert_rejected_by_assert_row(failed: Option<Instruction>, case: &str) {
        assert!(
            matches!(
                failed,
                Some(Instruction::VirtualAssertEQ(_) | Instruction::VirtualAssertLTE(_))
            ),
            "{case}: expected an assert row to fail, got {failed:?}"
        );
    }

    fn output(harness: &mut InlineTestHarness) -> [u64; 4] {
        harness
            .read_output64(4)
            .try_into()
            .expect("four output limbs")
    }

    fn add(x: [u64; 4], y: [u64; 4]) -> [u64; 4] {
        let mut sum = BigInt(x);
        assert!(!sum.add_with_carry(&BigInt(y)), "sum exceeds 256 bits");
        sum.0
    }

    fn assert_div_rejects_dishonest_advice<Op, F, E>(
        from_limbs: fn(&[u64; 4]) -> Result<E, Secp256k1Error>,
    ) where
        Op: InlineOp<Advice = ModularDivisionAdvice>,
        F: PrimeField<BigInt = BigInt<4>>,
    {
        let result = F::from(5u64);
        let divisor = F::from_bigint(BigInt(DIVISOR)).expect("divisor is reduced");
        let dividend = (divisor * result).into_bigint().0;
        let wrong_result: fn(&mut ModularDivisionAdvice) = |advice| advice.result[0] ^= 1;
        let wrong_quotient: fn(&mut ModularDivisionAdvice) = |advice| advice.quotient[0] ^= 1;
        for aliased in [false, true] {
            let mut honest = harness(aliased, &dividend, &DIVISOR);
            assert!(run_with_advice::<Op>(&mut honest, |_| {}).is_none());
            assert_eq!(output(&mut honest), result.into_bigint().0);

            for (case, tamper) in [("result", wrong_result), ("quotient", wrong_quotient)] {
                let mut dishonest = harness(aliased, &dividend, &DIVISOR);
                assert_rejected_by_assert_row(
                    run_with_advice::<Op>(&mut dishonest, tamper),
                    &format!("{} wrong {case}, aliased = {aliased}", Op::NAME),
                );
            }

            // `result + modulus` with quotient `w + divisor` satisfies
            // `b * c == w * q + a` over the integers, and the sequence does
            // not range-check `c` (see the comment above `MulqBuilder`). The
            // SDK's canonicity check after the inline is what rejects it.
            let mut unreduced = harness(aliased, &dividend, &DIVISOR);
            let failed = run_with_advice::<Op>(&mut unreduced, |advice| {
                advice.result = add(advice.result, F::MODULUS.0);
                advice.quotient = add(advice.quotient, DIVISOR);
            });
            assert!(failed.is_none(), "{failed:?}");
            let stored = output(&mut unreduced);
            assert_eq!(stored, add(result.into_bigint().0, F::MODULUS.0));
            assert!(from_limbs(&stored).is_err());
        }
    }

    fn assert_quotient_rejects_dishonest_advice<Op: InlineOp<Advice = QuotientAdvice>>() {
        let mut honest = harness(false, &FACTOR, &DIVISOR);
        assert!(run_with_advice::<Op>(&mut honest, |_| {}).is_none());
        let mut dishonest = harness(false, &FACTOR, &DIVISOR);
        assert_rejected_by_assert_row(
            run_with_advice::<Op>(&mut dishonest, |advice| advice.quotient[0] ^= 1),
            &format!("{} wrong quotient", Op::NAME),
        );
    }

    #[test]
    fn divq_rejects_dishonest_advice() {
        assert_div_rejects_dishonest_advice::<Secp256k1DivQ, Fq, _>(Secp256k1Fq::from_u64_arr);
    }

    #[test]
    fn divr_rejects_dishonest_advice() {
        assert_div_rejects_dishonest_advice::<Secp256k1DivR, Fr, _>(Secp256k1Fr::from_u64_arr);
    }

    #[test]
    fn mul_and_square_reject_wrong_quotient_advice() {
        assert_quotient_rejects_dishonest_advice::<Secp256k1MulQ>();
        assert_quotient_rejects_dishonest_advice::<Secp256k1SquareQ>();
        assert_quotient_rejects_dishonest_advice::<Secp256k1MulR>();
        assert_quotient_rejects_dishonest_advice::<Secp256k1SquareR>();
    }
}
