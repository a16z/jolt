mod sequence_tests {
    use crate::sdk::Secp256k1Point;
    use crate::{
        Secp256k1Fq, Secp256k1Fr, INLINE_OPCODE, SECP256K1_DIVQ_ADV_FUNCT3, SECP256K1_FUNCT7,
    };
    use ark_ff::{BigInt, Field, PrimeField};
    use ark_secp256k1::Fq;
    use tracer::emulator::cpu::Xlen;
    use tracer::utils::inline_test_harness::{InlineMemoryLayout, InlineTestHarness};

    fn assert_divq_trace_equiv(a: &[u64; 4], b: &[u64; 4]) {
        // get expected value
        let arr_to_fq = |arr: &[u64; 4]| Fq::new_unchecked(BigInt(*arr));
        let expected = (arr_to_fq(b)
            .inverse()
            .expect("Attempted to invert zero in secp256k1 field")
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
            SECP256K1_DIVQ_ADV_FUNCT3,
            SECP256K1_FUNCT7,
        ));
        let result_vec = harness.read_output64(4);
        let mut result = [0u64; 4];
        result.copy_from_slice(&result_vec);
        assert_eq!(result, expected, "secp256k1_divq_adv result mismatch");
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
        assert_eq!(combined.x().fq(), expected.x().fq());
        assert_eq!(combined.y().fq(), expected.y().fq());
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
}
