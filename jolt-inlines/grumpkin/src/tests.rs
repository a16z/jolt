mod sequence_tests {
    use crate::sdk::{
        GrumpkinFr, GrumpkinPoint, GRUMPKIN_ENDO_BETA_LIMBS, GRUMPKIN_GLV_LAMBDA_LIMBS,
    };
    use crate::{
        GRUMPKIN_DIVQ_ADV_FUNCT3, GRUMPKIN_DIVR_ADV_FUNCT3, GRUMPKIN_FUNCT7,
        GRUMPKIN_GLVR_ADV_FUNCT3, INLINE_OPCODE,
    };
    use ark_ec::AffineRepr;
    use ark_ff::{BigInt, BigInteger, Field, One, PrimeField, Zero};
    use ark_grumpkin::{Fq, Fr};
    use std::ops::Mul;
    use tracer::emulator::cpu::Xlen;
    use tracer::utils::inline_test_harness::{InlineMemoryLayout, InlineTestHarness};

    fn assert_divq_trace_equiv(a: &[u64; 4], b: &[u64; 4]) {
        // get expected value
        let arr_to_fq = |arr: &[u64; 4]| Fq::new_unchecked(BigInt(*arr));
        let expected = (arr_to_fq(b)
            .inverse()
            .expect("Attempted to invert zero in grumpkin field")
            * arr_to_fq(a))
        .0
         .0;
        // rs1=input1 (32 bytes), rs2=input2 (32 bytes), rs3=output (32 bytes)
        let layout = InlineMemoryLayout::two_inputs(32, 32, 32);
        let mut harness = InlineTestHarness::new(layout, Xlen::Bit64);
        harness.setup_registers();
        harness.load_input64(a);
        harness.load_input2_64(b);
        harness.execute_inline(InlineTestHarness::create_default_instruction(
            INLINE_OPCODE,
            GRUMPKIN_DIVQ_ADV_FUNCT3,
            GRUMPKIN_FUNCT7,
        ));
        let result_vec = harness.read_output64(4);
        let mut result = [0u64; 4];
        result.copy_from_slice(&result_vec);
        assert_eq!(result, expected, "grumpkin_divq_adv result mismatch");
    }

    fn assert_divr_trace_equiv(a: &[u64; 4], b: &[u64; 4]) {
        // get expected value
        let arr_to_fr = |arr: &[u64; 4]| Fr::new_unchecked(BigInt(*arr));
        let expected = (arr_to_fr(b)
            .inverse()
            .expect("Attempted to invert zero in grumpkin scalar field")
            * arr_to_fr(a))
        .0
         .0;
        // rs1=input1 (32 bytes), rs2=input2 (32 bytes), rs3=output (32 bytes)
        let layout = InlineMemoryLayout::two_inputs(32, 32, 32);
        let mut harness = InlineTestHarness::new(layout, Xlen::Bit64);
        harness.setup_registers();
        harness.load_input64(a);
        harness.load_input2_64(b);
        harness.execute_inline(InlineTestHarness::create_default_instruction(
            INLINE_OPCODE,
            GRUMPKIN_DIVR_ADV_FUNCT3,
            GRUMPKIN_FUNCT7,
        ));
        let result_vec = harness.read_output64(4);
        let mut result = [0u64; 4];
        result.copy_from_slice(&result_vec);
        assert_eq!(result, expected, "grumpkin_divr_adv result mismatch");
    }

    #[test]
    fn test_grumpkin_divq_direct_execution() {
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

    #[test]
    fn test_grumpkin_divr_direct_execution() {
        // Use valid (canonical) field elements, then compare their Montgomery limbs.
        let a = Fr::from(123456789u64).0 .0;
        let b = Fr::from(987654321u64).0 .0;
        assert_divr_trace_equiv(&a, &b);

        let a = Fr::from(1u64).0 .0;
        let b = Fr::from(2u64).0 .0;
        assert_divr_trace_equiv(&a, &b);

        let a = Fr::from(42u64).0 .0;
        let b = Fr::from(42u64).0 .0;
        assert_divr_trace_equiv(&a, &b);
    }

    fn assert_glvr_trace_recompose(k: &Fr) {
        let k_limbs = k.0 .0;

        // We want rs1=input, rs3=output. The harness' `single_input` mapping is for
        // hash inlines (rs1=output, rs2=input), so use `two_inputs` and provide a
        // dummy second input region.
        // rs1=input (32 bytes), rs2=dummy (8 bytes), rs3=output (48 bytes = 6 u64)
        let layout = InlineMemoryLayout::two_inputs(32, 8, 48);
        let mut harness = InlineTestHarness::new(layout, Xlen::Bit64);
        harness.setup_registers();
        harness.load_input64(&k_limbs);
        harness.load_input2_64(&[0u64]);
        harness.execute_inline(InlineTestHarness::create_default_instruction(
            INLINE_OPCODE,
            GRUMPKIN_GLVR_ADV_FUNCT3,
            GRUMPKIN_FUNCT7,
        ));

        let out_vec = harness.read_output64(6);
        let out: [u64; 6] = out_vec.try_into().expect("expected 6 u64 outputs");

        // Decode (sign, abs) pairs.
        let sign1 = out[0];
        let k1_u = (out[1] as u128) | ((out[2] as u128) << 64);
        let sign2 = out[3];
        let k2_u = (out[4] as u128) | ((out[5] as u128) << 64);

        // Recompose: k ≡ k1 + k2·lambda (mod r)
        let lambda = Fr::new_unchecked(BigInt(GRUMPKIN_GLV_LAMBDA_LIMBS));
        let mut k1 = Fr::from(k1_u);
        if sign1 == 1 {
            k1 = -k1;
        }
        let mut k2 = Fr::from(k2_u);
        if sign2 == 1 {
            k2 = -k2;
        }
        let recomposed = k1 + k2 * lambda;
        assert_eq!(recomposed, *k, "grumpkin_glvr_adv recomposition mismatch");
    }

    #[test]
    fn test_grumpkin_glvr_direct_execution_recompose() {
        let scalars = [
            Fr::from(0u64),
            Fr::from(1u64),
            Fr::from(2u64),
            Fr::from(3u64),
            Fr::from(5u64),
            Fr::from(7u64),
            Fr::from(0x123456789ABCDEF0u64),
            Fr::from(0x0FEDCBA987654321u64),
        ];
        for k in scalars.iter() {
            assert_glvr_trace_recompose(k);
        }

        // A few deterministic 256-bit samples reduced mod Fr.
        let (a, c) = (6364136223846793005u64, 1442695040888963407u64);
        let mut state = 0xA5A5_A5A5_5A5A_5A5Au64;
        for _ in 0..32 {
            let mut bytes = [0u8; 32];
            for chunk in bytes.chunks_mut(8) {
                state = state.wrapping_mul(a).wrapping_add(c);
                chunk.copy_from_slice(&state.to_le_bytes());
            }
            let k = Fr::from_le_bytes_mod_order(&bytes);
            assert_glvr_trace_recompose(&k);
        }
    }

    fn u64_point_mul(scalar: u64, point: &GrumpkinPoint) -> GrumpkinPoint {
        let mut res = GrumpkinPoint::infinity();
        for i in (0..64).rev() {
            if (scalar >> i) & 1 == 1 {
                res = res.double_and_add(point);
            } else {
                res = res.double();
            }
        }
        res
    }

    fn scalar_mul_consistency_helper(scalar: u64) {
        // generator * scalar in our impl
        let res = u64_point_mul(scalar, &GrumpkinPoint::generator());
        // generator * scalar in arkworks
        let ark_res = ark_grumpkin::Affine::from(
            ark_grumpkin::Affine::generator().mul(ark_grumpkin::Fr::from(scalar)),
        );
        // compare
        assert_eq!(res.x().fq(), ark_res.x);
        assert_eq!(res.y().fq(), ark_res.y);
    }

    #[test]
    fn test_grumpkin_point_mul_consistency() {
        let scalars = [
            0x123456789ABCDEF0,
            0x0FEDCBA987654321,
            0x1111111111111111,
            0x2222222222222222,
            0x0000000000000001,
            0x0000000000000002,
            0x0000000000000003,
            0x0000000000000004,
            0xFFFFFFFFFFFFFFFF,
            0x7FFFFFFFFFFFFFFF,
            0x8000000000000000,
            0x1234567890ABCDEF,
        ];
        for &scalar in scalars.iter() {
            scalar_mul_consistency_helper(scalar);
        }
    }

    /// Regression test: `double_and_add` must not divide by zero for valid inputs.
    ///
    /// In particular, if `other = -2*self`, then `2*self + other = 0` (point at infinity).
    #[test]
    fn test_grumpkin_double_and_add_other_is_neg_two_self() {
        let p = GrumpkinPoint::generator();
        let other = p.double().neg();
        let res = p.double_and_add(&other);
        assert!(res.is_infinity(), "2P + (-2P) must be infinity");
    }

    #[test]
    fn test_grumpkin_endomorphism_generator() {
        let endo_g = GrumpkinPoint::generator().endomorphism();
        let expected = GrumpkinPoint::generator_w_endomorphism();
        assert_eq!(endo_g.to_u64_arr(), expected.to_u64_arr());
    }

    #[test]
    fn test_grumpkin_endo_beta_has_order_3_in_fq() {
        let beta = Fq::new_unchecked(BigInt(GRUMPKIN_ENDO_BETA_LIMBS));
        assert_ne!(beta, Fq::one(), "beta must not be 1");
        assert_eq!(beta.square() * beta, Fq::one(), "beta^3 must equal 1");
    }

    #[test]
    fn test_grumpkin_glv_lambda_satisfies_minpoly() {
        let lambda = Fr::new_unchecked(BigInt(GRUMPKIN_GLV_LAMBDA_LIMBS));
        assert_ne!(lambda, Fr::one(), "lambda must not be 1");
        let lhs = lambda.square() + lambda + Fr::one();
        assert!(lhs.is_zero(), "lambda^2 + lambda + 1 must equal 0");
    }

    #[test]
    fn test_grumpkin_endomorphism_order_3() {
        let g = GrumpkinPoint::generator();
        let g3 = g.endomorphism().endomorphism().endomorphism();
        assert_eq!(
            g3.to_u64_arr(),
            g.to_u64_arr(),
            "endomorphism should have order 3"
        );
    }

    #[test]
    fn test_grumpkin_endomorphism_matches_scalar_mul_lambda_on_generator_multiples() {
        let lambda = Fr::new_unchecked(BigInt(GRUMPKIN_GLV_LAMBDA_LIMBS));
        let scalars = [1u64, 2u64, 3u64, 5u64, 7u64, 0x123456789ABCDEF0u64];
        for &s in scalars.iter() {
            let p = u64_point_mul(s, &GrumpkinPoint::generator());
            let endo_p = p.endomorphism();

            let expected_scalar = Fr::from(s) * lambda;
            let ark_expected =
                ark_grumpkin::Affine::from(ark_grumpkin::Affine::generator().mul(expected_scalar));

            assert_eq!(endo_p.x().fq(), ark_expected.x);
            assert_eq!(endo_p.y().fq(), ark_expected.y);
        }
    }

    #[test]
    fn test_grumpkin_glv_decomposition_recompose() {
        let lambda = Fr::new_unchecked(BigInt(GRUMPKIN_GLV_LAMBDA_LIMBS));
        let scalars = [
            0u64,
            1u64,
            2u64,
            3u64,
            5u64,
            7u64,
            0x123456789ABCDEF0,
            0x0FEDCBA987654321,
        ];
        for scalar in scalars.iter() {
            let k = GrumpkinFr::new(Fr::from(*scalar));
            let decomp = GrumpkinPoint::decompose_scalar(&k);
            let mut k1 = Fr::from(decomp[0].1);
            if decomp[0].0 {
                k1 = -k1;
            }
            let mut k2 = Fr::from(decomp[1].1);
            if decomp[1].0 {
                k2 = -k2;
            }
            let recomposed = k1 + k2 * lambda;
            assert_eq!(recomposed, k.fr());
        }
    }

    #[test]
    fn test_grumpkin_glv_decomposition_recompose_many() {
        let lambda = Fr::new_unchecked(BigInt(GRUMPKIN_GLV_LAMBDA_LIMBS));
        let (a, c) = (6364136223846793005u64, 1442695040888963407u64);
        let mut state = 0xA5A5_A5A5_5A5A_5A5Au64;

        for _ in 0..256 {
            // Deterministic 256-bit input -> reduce mod Fr.
            let mut bytes = [0u8; 32];
            for chunk in bytes.chunks_mut(8) {
                state = state.wrapping_mul(a).wrapping_add(c);
                chunk.copy_from_slice(&state.to_le_bytes());
            }
            let fr = Fr::from_le_bytes_mod_order(&bytes);
            let k = GrumpkinFr::new(fr);

            let decomp = GrumpkinPoint::decompose_scalar(&k);
            let mut k1 = Fr::from(decomp[0].1);
            if decomp[0].0 {
                k1 = -k1;
            }
            let mut k2 = Fr::from(decomp[1].1);
            if decomp[1].0 {
                k2 = -k2;
            }
            let recomposed = k1 + k2 * lambda;
            assert_eq!(recomposed, k.fr());
        }
    }

    #[test]
    fn test_grumpkin_glv_decomposition_edge_cases() {
        let lambda = Fr::new_unchecked(BigInt(GRUMPKIN_GLV_LAMBDA_LIMBS));

        // Helper to verify decomposition
        let verify_decomp = |k: GrumpkinFr, name: &str| {
            let decomp = GrumpkinPoint::decompose_scalar(&k);
            let mut k1 = Fr::from(decomp[0].1);
            if decomp[0].0 {
                k1 = -k1;
            }
            let mut k2 = Fr::from(decomp[1].1);
            if decomp[1].0 {
                k2 = -k2;
            }
            let recomposed = k1 + k2 * lambda;
            assert_eq!(recomposed, k.fr(), "GLV decomposition failed for {name}");
        };

        // k = 0
        verify_decomp(GrumpkinFr::new(Fr::zero()), "k=0");

        // k = 1
        verify_decomp(GrumpkinFr::new(Fr::one()), "k=1");

        // k = n - 1 (largest valid scalar, equivalent to -1)
        verify_decomp(GrumpkinFr::new(-Fr::one()), "k=n-1");

        // k = λ (should decompose to k1=0, k2=1 ideally, but any valid decomposition works)
        verify_decomp(GrumpkinFr::new(lambda), "k=lambda");

        // k = λ - 1
        verify_decomp(GrumpkinFr::new(lambda - Fr::one()), "k=lambda-1");

        // k = λ + 1
        verify_decomp(GrumpkinFr::new(lambda + Fr::one()), "k=lambda+1");

        // k = λ² (which equals -λ - 1 since λ² + λ + 1 = 0)
        let lambda_sq = lambda.square();
        verify_decomp(GrumpkinFr::new(lambda_sq), "k=lambda^2");
    }

    #[test]
    fn test_grumpkin_glv_lattice_determinant() {
        use num_bigint::BigInt as NBigInt;
        use num_bigint::Sign;

        // GLV basis vectors (same as in decompose_scalar)
        let n11 = NBigInt::from(147946756881789319000765030803803410729i128);
        let n12 = NBigInt::from(-9931322734385697762i128);
        let n21 = NBigInt::from(9931322734385697762i128);
        let n22 = NBigInt::from(147946756881789319010696353538189108491i128);

        // Compute determinant: n11 * n22 - n12 * n21
        let det = &n11 * &n22 - &n12 * &n21;

        // The determinant should equal the curve order n
        let n = NBigInt::from_bytes_le(Sign::Plus, &Fr::MODULUS.to_bytes_le());

        assert_eq!(det, n, "GLV lattice determinant must equal curve order");
    }
}
