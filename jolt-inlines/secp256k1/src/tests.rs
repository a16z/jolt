mod sequence_tests {
    use crate::sdk::Secp256k1Point;
    use crate::{INLINE_OPCODE, SECP256K1_DIVQ_ADV_FUNCT3, SECP256K1_FUNCT7};
    use ark_ec::{AffineRepr, CurveGroup};
    use ark_ff::{BigInt, Field};
    use ark_secp256k1::Affine;
    use ark_secp256k1::Fq;
    use tracer::emulator::cpu::Xlen;
    use tracer::utils::inline_test_harness::{InlineMemoryLayout, InlineTestHarness};

    // wrapper to make it easier to create Fq from [u64; 4]
    fn arr_to_fq(a: &[u64; 4]) -> Fq {
        Fq::new_unchecked(BigInt { 0: *a })
    }

    fn assert_divq_trace_equiv(a: &[u64; 4], b: &[u64; 4]) {
        // get expected value
        let expected = (arr_to_fq(&b)
            .inverse()
            .expect("Attempted to invert zero in secp256k1 field")
            * arr_to_fq(&a))
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
        // get 7 as Fq and print
        //let seven = Fq::from(7);
        //println!("{:?}", seven.0 .0);
    }
    #[test]
    fn test_double_and_add() {
        // check that generator matches
        let p = Secp256k1Point::generator();
        let pa = Affine::GENERATOR;
        // print p and pa
        assert_eq!(p.x().fq(), pa.x().unwrap());
        assert_eq!(p.y().fq(), pa.y().unwrap());
        // test point doubling
        let q = Secp256k1Point::generator().add(&Secp256k1Point::generator());
        let qa = (Affine::GENERATOR + Affine::GENERATOR).into_affine();
        // print q and qa
        //println!("{:?} {:?}", q, qa);
        assert_eq!(q.x().fq(), qa.x().unwrap());
        assert_eq!(q.y().fq(), qa.y().unwrap());
        // compute 2q + p using double and add
        let r1 = q.double_and_add(&p);
        let r2 = q.add(&q).add(&p);
        let r3 = q.add(&p).add(&q);
        //println!("{:?} {:?} {:?}", r1, r2, r3);
        let r1a = (qa + qa + pa).into_affine();
        assert_eq!(r1.x().fq(), r1a.x().unwrap());
        assert_eq!(r1.y().fq(), r1a.y().unwrap());
    }
    #[test]
    fn test_scalar_mul() {
        // compute n * g
        let n: u64 = 0xF123456789ABCDEF;
        let g = Secp256k1Point::generator();
        let mut res = Secp256k1Point::infinity();
        // compute expected using arkworks
        let g_a = Affine::GENERATOR;
        let mut res_a = Affine::identity();
        for i in (0..64).rev() {
            //println!("i: {}", i);
            //println!("{:?}\n{:?}", res, res_a);
            if res_a.is_zero() {
                assert!(res.is_infinity());
            } else {
                assert_eq!(res.x().fq(), res_a.x().unwrap());
                assert_eq!(res.y().fq(), res_a.y().unwrap());
            }
            if (n >> i) & 1 == 1 {
                res = res.double_and_add(&g);
                //res = res.double().add(&g);
                res_a = (res_a + res_a + g_a).into_affine();
            } else {
                res = res.double();
                res_a = (res_a + res_a).into_affine();
            }
        }
    }

    fn scalar_mul_native(scalar: u128, point: Secp256k1Point) -> Secp256k1Point {
        let mut res = Secp256k1Point::infinity();
        for i in (0..128).rev() {
            if (scalar >> i) & 1 == 1 {
                res = res.double_and_add(&point);
            } else {
                res = res.double();
            }
        }
        res
    }
    #[test]
    fn get_test_vectors() {
        println!("Test Vectors:");
        let point = Secp256k1Point::generator();
        let scalars = [
            0x1234567890ABCDEF1234567890ABCDEFu128,
            0x0FEDCBA9876543210FEDCBA987654321u128,
            0x11111111111111111111111111111111u128,
            0x22222222222222222222222222222222u128,
        ];
        for &scalar in scalars.iter() {
            let result = scalar_mul_native(scalar, point.clone());
            let arr = result.to_u64_arr();
            println!("Scalar: {:#034x}", scalar);
            println!("Result: [{:#018x}, {:#018x}, {:#018x}, {:#018x}, {:#018x}, {:#018x}, {:#018x}, {:#018x}]",
                arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7]);
        }
    }
}

/*[0xfd7914a271ed2e42, 0x7fb20973e1035805, 0x8c2c7e3c55347a2f, 0xe069d2fb3df133fd, 0x70e6973fb3b3c61e, 0xaed7312cd8530080, 0x390fa40885dbc7f2, 0x3142c3b27c54160e, 0x62cdfbc1358ff2e7, 0x95bce326ef8d07c0, 0x1a0637809a7c16e3, 0x0197263b9b73d8fe, 0x921e6ffa3fe39600, 0xc1b77824c49ecaa6, 0x25b5d035fbbdcd93, 0xd25330b456437bc4,0xbfc6759e3ab1d57a, 0x2e822c47f143f7dc, 0xf8d88465f162255a, 0xac8cbfb4707c3ba1, 0x92b8007c0027e3b6, 0x3d3a2aaa3b129d3c, 0xc71a36833e579582, 0x63fa22b365e65edc,0x84c60f988985bb6d, 0x3771987a8626ed1b, 0x7d2d842df22e3972, 0x68c3e1d401738d23, 0x7ba86c982b250320, 0x845453face9978fb, 0xd480f970fa1501a4, 0xd9ccbc62a5f896f9]*/
