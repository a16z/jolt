mod sequence_tests {
    use crate::sdk::Secp256k1Point;
    use crate::{INLINE_OPCODE, SECP256K1_DIVQ_ADV_FUNCT3, SECP256K1_FUNCT7};
    use ark_ec::{AffineRepr, CurveGroup};
    use ark_ff::{BigInt, Field, PrimeField};
    use ark_secp256k1::Affine;
    use ark_secp256k1::{Fq, Fr};
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

    fn scalar_mul_native(scalar: u128, point: &Secp256k1Point) -> Secp256k1Point {
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
    /*#[test]
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
    }*/
    // check that lambda * Q = (beta*x mod p, y)
    /*#[test]
    fn test_endomorphism() {
        let lambda = [
            0x122e22ea20816678df02967c1b23bd72u128,
            0x5363ad4cc05c30e0a5261c028812645au128,
        ];
        let mut res = Secp256k1Point::infinity();
        for j in (0..2).rev() {
            let scalar = lambda[j];
            for i in (0..128).rev() {
                if (scalar >> i) & 1 == 1 {
                    res = res.double_and_add(&Secp256k1Point::generator());
                } else {
                    res = res.double();
                }
            }
        }
        println!("Generator  : {:?}", Secp256k1Point::generator().y());
        println!("endomorphism: {:?}", res.y());
        // 0x7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee
        let beta = Fq::new(BigInt {
            0: [
                0xc1396c28719501ee,
                0x9cf0497512f58995,
                0x6e64479eac3434e9,
                0x7ae96a2b657c0710,
            ],
        });
        println!("{:?}", beta.0 .0);
        println!(
            "Beta * x     : {:?}",
            Secp256k1Point::generator().x().fq() * beta
        );
        println!("endomorphism x: {:?}", res.x().fq());
        let comp = Secp256k1Point::generator().endomorphism();
        assert_eq!(res.x().fq(), comp.x().fq());
        assert_eq!(res.y().fq(), comp.y().fq());
    }*/
    #[test]
    fn test_decompose_scalar() {
        let scalar = Fr::from_bigint(BigInt {
            0: [
                15694125933356685049,
                15312512996687020452,
                9535338647723276539,
                8910491263567201056,
            ],
        })
        .unwrap();
        // print scalar
        println!("Scalar: {:?}", scalar.into_bigint().0);
        let decomp = Secp256k1Point::decompose_scalar(&scalar);
        println!("{:?}", decomp);
    }

    fn scalar_mul_fr(scalar: &Fr, point: &Secp256k1Point) -> Secp256k1Point {
        let mut res = Secp256k1Point::infinity();
        let k = scalar.into_bigint().0;
        for i in (0..256).rev() {
            if (k[i / 64] >> (i % 64)) & 1 == 1 {
                res = res.double_and_add(point);
            } else {
                res = res.double();
            }
        }
        res
    }
    /*#[test]
    fn print_lambda() {
        // print lambda in montgomery form
        let lambda = Fr::from_bigint(BigInt {
            0: [
                0xdf02967c1b23bd72,
                0x122e22ea20816678,
                0xa5261c028812645a,
                0x5363ad4cc05c30e0,
            ],
        })
        .unwrap();
        println!("Lambda: {:?}", lambda.0 .0);
    }*/

    #[test]
    fn test_endomorphism_consistency() {
        let mut point = Secp256k1Point::generator();
        let mut endo_point = Secp256k1Point::generator_w_endomorphism();
        let k = Fr::NEG_ONE
            * Fr::new(BigInt {
                0: [
                    0x1234567890ABCDEF,
                    0x0FEDCBA987654321,
                    0x1111111111111111,
                    0x2222222222222222,
                ],
            });
        let decomp = Secp256k1Point::decompose_scalar(&k);
        if decomp[0].0 {
            point = point.neg();
        }
        if decomp[1].0 {
            endo_point = endo_point.neg();
        }
        let k1 = decomp[0].1;
        let k2 = decomp[1].1;
        let p1 = scalar_mul_native(k1, &point);
        let p2 = scalar_mul_native(k2, &endo_point);
        let combined = p1.add(&p2);
        let expected = scalar_mul_fr(&k, &Secp256k1Point::generator());
        assert_eq!(combined.x().fq(), expected.x().fq());
        assert_eq!(combined.y().fq(), expected.y().fq());
        // also check that sign1 * k1 + sign2 * k2 * lambda mod r = k
        let lambda = Fr::from_bigint(BigInt {
            0: [
                0xdf02967c1b23bd72,
                0x122e22ea20816678,
                0xa5261c028812645a,
                0x5363ad4cc05c30e0,
            ],
        })
        .unwrap();
        let mut sk1 = Fr::from_bigint(BigInt {
            0: [k1 as u64, (k1 >> 64) as u64, 0, 0],
        })
        .unwrap();
        if decomp[0].0 {
            sk1 = -sk1;
        }
        let mut sk2 = Fr::from_bigint(BigInt {
            0: [k2 as u64, (k2 >> 64) as u64, 0, 0],
        })
        .unwrap();
        if decomp[1].0 {
            sk2 = -sk2;
        }
        let recombined = sk1 + sk2 * lambda;
        assert_eq!(recombined, k);
    }

    #[test]
    fn test_signature_verify_w_glv() {
        // check that u G + v Q = u1 G1 + u2 lambdaG + v1 Q + v2 lambdaQ
        // test vectors
        let g = Secp256k1Point::generator();
        let lg = Secp256k1Point::generator_w_endomorphism();
        let q = Secp256k1Point::from_u64_arr_unchecked(&[
            0x84c60f988985bb6d,
            0x3771987a8626ed1b,
            0x7d2d842df22e3972,
            0x68c3e1d401738d23,
            0x7ba86c982b250320,
            0x845453face9978fb,
            0xd480f970fa1501a4,
            0xd9ccbc62a5f896f9,
        ]);
        let u = Fr::from_bigint(BigInt {
            0: [
                0x1234567890ABCDEF,
                0x0FEDCBA987654321,
                0x1111111111111111,
                0x2222222222222222,
            ],
        })
        .unwrap();
        let v = Fr::from_bigint(BigInt {
            0: [
                0x0FEDCBA987654321,
                0x1234567890ABCDEF,
                0x3333333333333333,
                0x4444444444444444,
            ],
        })
        .unwrap();
        // scalar mul without decomposition
        let u_g = scalar_mul_fr(&u, &g);
        let v_q = scalar_mul_fr(&v, &q);
        let combined = u_g.add(&v_q);
        // scalar mul with decomposition
        let decomp_u = Secp256k1Point::decompose_scalar(&u);
        let decomp_v = Secp256k1Point::decompose_scalar(&v);
        let gs = if decomp_u[0].0 { g.neg() } else { g };
        let gls = if decomp_u[1].0 { lg.neg() } else { lg };
        let qs = if decomp_v[0].0 { q.neg() } else { q.clone() };
        let qls = if decomp_v[1].0 {
            q.endomorphism().neg()
        } else {
            q.endomorphism()
        };
        let u1_g1 = scalar_mul_native(decomp_u[0].1, &gs);
        let u2_lg = scalar_mul_native(decomp_u[1].1, &gls);
        let v1_q1 = scalar_mul_native(decomp_v[0].1, &qs);
        let v2_lq = scalar_mul_native(decomp_v[1].1, &qls);
        let decombined = u1_g1.add(&u2_lg).add(&v1_q1).add(&v2_lq);
        assert_eq!(combined.x().fq(), decombined.x().fq());
        assert_eq!(combined.y().fq(), decombined.y().fq());
    }

    /*#[test]
    fn test_mont() {
        // check internal representation in montgomery form
        let one = Fr::ONE;
        println!("Fq::ONE: {:?}", one.0 .0);
        let one_alt = Fr::new(BigInt {
            0: [
                0x0000000000000001,
                0x0000000000000000,
                0x0000000000000000,
                0x0000000000000000,
            ],
        });
        println!("big::ONE: {:?}", one_alt.0 .0);
    }*/
    /*#[test]
    fn convert() {
        use ark_ff::PrimeField;
        use num_bigint::BigInt as NBigInt;
        use num_bigint::Sign;
        use num_integer::Integer;
        use std::str::FromStr;
        let r = NBigInt::from_str(Fr::MODULUS.to_string().as_str()).unwrap();
        let a1 = NBigInt::from_str("64502973549206556628585045361533709077").unwrap();
        let b1 = NBigInt::from_str("303414439467246543595250775667605759171").unwrap();
        let a2 = NBigInt::from_str("367917413016453100223835821029139468248").unwrap();
        println!("r: {:?}", r.to_bytes_le());
        println!("a1: {:?}", a1.to_bytes_le());
        println!("b1: {:?}", b1.to_bytes_le());
        println!("a2: {:?}", a2.to_bytes_le());
    }*/
}

/*[0xfd7914a271ed2e42, 0x7fb20973e1035805, 0x8c2c7e3c55347a2f, 0xe069d2fb3df133fd, 0x70e6973fb3b3c61e, 0xaed7312cd8530080, 0x390fa40885dbc7f2, 0x3142c3b27c54160e, 0x62cdfbc1358ff2e7, 0x95bce326ef8d07c0, 0x1a0637809a7c16e3, 0x0197263b9b73d8fe, 0x921e6ffa3fe39600, 0xc1b77824c49ecaa6, 0x25b5d035fbbdcd93, 0xd25330b456437bc4,0xbfc6759e3ab1d57a, 0x2e822c47f143f7dc, 0xf8d88465f162255a, 0xac8cbfb4707c3ba1, 0x92b8007c0027e3b6, 0x3d3a2aaa3b129d3c, 0xc71a36833e579582, 0x63fa22b365e65edc,0x84c60f988985bb6d, 0x3771987a8626ed1b, 0x7d2d842df22e3972, 0x68c3e1d401738d23, 0x7ba86c982b250320, 0x845453face9978fb, 0xd480f970fa1501a4, 0xd9ccbc62a5f896f9]*/
