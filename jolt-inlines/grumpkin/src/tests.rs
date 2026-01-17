mod sequence_tests {
    use crate::sdk::{GrumpkinFr, GrumpkinPoint};
    use crate::{GRUMPKIN_DIVQ_ADV_FUNCT3, GRUMPKIN_FUNCT7, INLINE_OPCODE};
    use ark_ec::AffineRepr;
    use ark_ff::{BigInt, Field};
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

    #[test]
    fn test_grumpkin_endomorphism_generator() {
        let endo_g = GrumpkinPoint::generator().endomorphism();
        let expected = GrumpkinPoint::generator_w_endomorphism();
        assert_eq!(endo_g.to_u64_arr(), expected.to_u64_arr());
    }

    #[test]
    fn test_grumpkin_glv_decomposition_recompose() {
        let lambda = Fr::new_unchecked(BigInt {
            0: [
                3697675806616062876,
                9065277094688085689,
                6918009208039626314,
                2775033306905974752,
            ],
        });
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
}
