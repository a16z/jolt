#![cfg(feature = "binary")]

use jolt_field::{Accumulator, ExtField, JoltField, WithAccumulator, F128, F192, F64};

struct Fixtures<F: JoltField> {
    products: [(F, F, F); 3],
    addends: [F; 2],
    mixed: F,
    persistence_odd: F,
}

#[derive(Clone, Copy)]
enum Operation<F> {
    Add(F),
    Product(F, F),
}

impl<F: JoltField> Operation<F> {
    fn apply(self, accumulator: &mut F::Accumulator) {
        match self {
            Self::Add(value) => accumulator.add(value),
            Self::Product(a, b) => accumulator.fmadd(a, b),
        }
    }
}

impl<F: JoltField> Fixtures<F> {
    fn sequence(&self) -> [Operation<F>; 5] {
        let [(a0, b0, _), (a1, b1, _), (a2, b2, _)] = self.products;
        [
            Operation::Product(a0, b0),
            Operation::Add(self.addends[0]),
            Operation::Product(a1, b1),
            Operation::Add(self.addends[1]),
            Operation::Product(a2, b2),
        ]
    }

    fn check_products(&self) {
        for (a, b, expected) in self.products {
            let mut accumulator = F::Accumulator::default();
            accumulator.fmadd(a, b);
            assert_eq!(accumulator.reduce(), expected);
        }
    }

    fn check_algebra(&self) {
        assert_eq!(F::Accumulator::default().reduce(), F::zero());
        let [(a0, b0, product0), (a1, b1, _), _] = self.products;
        let mut seeded = F::Accumulator::default();
        seeded.fmadd(a0, b0);
        assert_eq!(seeded.reduce(), product0);

        let mut cancelled = seeded;
        cancelled.fmadd(a1, b1);
        cancelled.fmadd(a1, b1);
        assert_eq!(cancelled.reduce(), product0);
        let mut zero_products = seeded;
        zero_products.fmadd(a1, F::zero());
        zero_products.fmadd(F::zero(), b1);
        assert_eq!(zero_products.reduce(), product0);

        let sequence = self.sequence();
        assert_eq!(accumulate(&sequence).reduce(), self.mixed);
        let first = accumulate(&sequence[..2]);
        let second = accumulate(&sequence[2..4]);
        let third = accumulate(&sequence[4..]);
        let mut left = first;
        left.merge(second);
        left.merge(third);
        assert_eq!(left.reduce(), self.mixed);
        let mut right = second;
        right.merge(third);
        let mut associated = first;
        associated.merge(right);
        assert_eq!(associated.reduce(), self.mixed);
        let mut commuted = second;
        commuted.merge(first);
        commuted.merge(third);
        assert_eq!(commuted.reduce(), self.mixed);

        for split in 0..=sequence.len() {
            let mut prefix = accumulate(&sequence[..split]);
            prefix.merge(accumulate(&sequence[split..]));
            assert_eq!(prefix.reduce(), self.mixed, "split at {split}");
        }
    }

    fn check_persistence(&self) {
        let (a, b, _) = self.products[0];
        let sentinel = self.addends[0];
        let mut accumulator = F::Accumulator::default();
        accumulator.add(sentinel);
        for _ in 0..(1 << 16) {
            accumulator.fmadd(a, b);
        }
        assert_eq!(accumulator.reduce(), sentinel);
        accumulator.fmadd(a, b);
        assert_eq!(accumulator.reduce(), self.persistence_odd);
    }
}

fn accumulate<F: JoltField>(operations: &[Operation<F>]) -> F::Accumulator {
    let mut accumulator = F::Accumulator::default();
    for &operation in operations {
        operation.apply(&mut accumulator);
    }
    accumulator
}

// Products are fixtures 4–6 of binary_vectors.rs; mixed and odd-term results
// are frozen XORs of those product words and the literal addends.
fn f64_fixtures() -> Fixtures<F64> {
    Fixtures {
        products: [
            (
                0x9cc2_a2f3_6303_dc3a,
                0x2711_8a65_14b1_9502,
                0x2c81_ccfb_ec6f_3649,
            ),
            (
                0x3334_1219_2d0d_f6a8,
                0x03ba_1c29_6c75_4745,
                0xe9f3_3b6f_118c_8b46,
            ),
            (
                0xdf36_59ed_3202_dc8b,
                0xc487_8ed3_04a4_9bf2,
                0x7852_2efe_95b6_0f58,
            ),
        ]
        .map(|(a, b, product)| (F64::from_raw(a), F64::from_raw(b), F64::from_raw(product))),
        addends: [0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210].map(F64::from_raw),
        mixed: F64::from_raw(0x42df_2695_97aa_4da8),
        persistence_odd: F64::from_raw(0x2da2_899c_65c4_fba6),
    }
}

fn f128_fixtures() -> Fixtures<F128> {
    Fixtures {
        products: [
            (
                0xe2e6_e912_0c07_3e95_5fa5_51c3_0749_1177,
                0x69ad_31be_cfa6_fbc3_7c3c_bdc7_a552_35a8,
                0xa8db_36a2_4aad_1221_6565_e498_c58d_04a5,
            ),
            (
                0x04da_76f8_cae0_35b9_7530_d464_c917_26f4,
                0x4790_687d_6658_c865_8066_6ee4_f2eb_bb92,
                0x7c2e_1fc5_d605_e11c_e535_0dd2_cfd0_b02e,
            ),
            (
                0xddeb_a2a9_772b_cc74_bc74_dc6a_92c0_c0b5,
                0x4fcf_dac3_f148_fde1_4a44_dc24_9b60_06ec,
                0x9987_6e25_7b4f_db66_dd65_d9b1_9831_a8b2,
            ),
        ]
        .map(|(a, b, product)| {
            (
                F128::from_raw(a),
                F128::from_raw(b),
                F128::from_raw(product),
            )
        }),
        addends: [
            0x0123_4567_89ab_cdef_fedc_ba98_7654_3210,
            0xfedc_ba98_7654_3210_0123_4567_89ab_cdef,
        ]
        .map(F128::from_raw),
        mixed: F128::from_raw(0xb28d_b8bd_1818_d7a4_a2ca_cf04_6d93_e3c6),
        persistence_odd: F128::from_raw(0xa9f8_73c5_c306_dfce_9bb9_5e00_b3d9_36b5),
    }
}

fn from_coefficients(words: [u64; 3]) -> F192 {
    F192::from_base_slice(&words.map(F64::from_raw))
}

fn f192_fixtures() -> Fixtures<F192> {
    Fixtures {
        products: [
            (
                [
                    13_316_095_090_024_323_137,
                    14_751_285_138_429_925_937,
                    7_781_250_287_217_855_685,
                ],
                [
                    15_399_382_051_422_031_900,
                    594_865_723_888_223_676,
                    2_556_603_478_692_854_071,
                ],
                [
                    13_947_920_174_051_519_348,
                    400_695_020_070_036_944,
                    4_979_101_278_107_147_318,
                ],
            ),
            (
                [
                    11_423_748_968_284_640_005,
                    15_529_916_621_196_270_819,
                    1_933_574_444_584_954_138,
                ],
                [
                    8_696_870_316_683_815_752,
                    12_871_165_662_592_704_158,
                    1_176_870_124_150_419_235,
                ],
                [
                    7_969_868_643_475_495_527,
                    15_412_500_157_466_384_471,
                    16_615_051_260_178_339_262,
                ],
            ),
            (
                [
                    9_221_629_071_983_985_981,
                    5_740_200_691_270_525_175,
                    3_481_045_019_550_695_030,
                ],
                [
                    12_044_444_523_982_065_546,
                    11_483_315_135_218_490_901,
                    223_802_242_728_893_033,
                ],
                [
                    3_350_132_397_039_891_550,
                    5_148_492_885_859_911_641,
                    4_548_659_086_283_305_593,
                ],
            ),
        ]
        .map(|(a, b, product)| {
            (
                from_coefficients(a),
                from_coefficients(b),
                from_coefficients(product),
            )
        }),
        addends: [
            [
                0x0123_4567_89ab_cdef,
                0xfedc_ba98_7654_3210,
                0x1020_3040_5060_7080,
            ],
            [
                0xfedc_ba98_7654_3210,
                0x1020_3040_5060_7080,
                0x0123_4567_89ab_cdef,
            ],
        ]
        .map(from_coefficients),
        mixed: from_coefficients([
            0x7e8b_ae64_72d1_3ab2,
            0x79e4_2b1c_c49f_78ce,
            0x8dae_b75d_521c_469e,
        ]),
        persistence_odd: from_coefficients([
            0xc0b3_b218_4873_a69b,
            0xfb53_3767_18de_d7c0,
            0x5539_627b_3625_6cb6,
        ]),
    }
}

#[test]
fn f64_frozen_products() {
    f64_fixtures().check_products();
}

#[test]
fn f128_frozen_products() {
    f128_fixtures().check_products();
}

#[test]
fn f192_frozen_products() {
    f192_fixtures().check_products();
}

#[test]
fn f64_accumulator_laws() {
    f64_fixtures().check_algebra();
}

#[test]
fn f128_accumulator_laws() {
    f128_fixtures().check_algebra();
}

#[test]
fn f192_accumulator_laws() {
    f192_fixtures().check_algebra();
}

#[test]
fn f64_many_terms_preserve_sentinel() {
    f64_fixtures().check_persistence();
}

#[test]
fn f128_many_terms_preserve_sentinel() {
    f128_fixtures().check_persistence();
}

#[test]
fn f192_many_terms_preserve_sentinel() {
    f192_fixtures().check_persistence();
}

#[test]
fn f64_reduction_boundaries() {
    let top = F64::from_raw(1 << 63);
    let mut accumulator = <F64 as WithAccumulator>::Accumulator::default();
    accumulator.fmadd(top, F64::from_raw(2));
    assert_eq!(accumulator.reduce().to_raw(), 0x1b);
    accumulator.add(F64::from_raw(1));
    assert_eq!(accumulator.reduce().to_raw(), 0x1a);
    let mut accumulator = <F64 as WithAccumulator>::Accumulator::default();
    accumulator.fmadd(top, F64::from_raw(1));
    assert_eq!(accumulator.reduce().to_raw(), 1 << 63);
    let mut accumulator = <F64 as WithAccumulator>::Accumulator::default();
    accumulator.fmadd(top, top);
    assert_eq!(accumulator.reduce().to_raw(), 0xc000_0000_0000_005a);
}

#[test]
fn f128_reduction_boundaries() {
    let top = F128::from_raw(1 << 127);
    let mut accumulator = <F128 as WithAccumulator>::Accumulator::default();
    accumulator.fmadd(top, F128::from_raw(2));
    assert_eq!(accumulator.reduce().to_raw(), 0x87);
    accumulator.add(F128::from_raw(1));
    assert_eq!(accumulator.reduce().to_raw(), 0x86);
    let mut accumulator = <F128 as WithAccumulator>::Accumulator::default();
    accumulator.fmadd(top, F128::from_raw(1));
    assert_eq!(accumulator.reduce().to_raw(), 1 << 127);
    let mut accumulator = <F128 as WithAccumulator>::Accumulator::default();
    accumulator.fmadd(top, top);
    assert_eq!(
        accumulator.reduce().to_raw(),
        0xc000_0000_0000_0000_0000_0000_0000_1067
    );
}

#[test]
fn f192_all_coefficient_pairs() {
    // The F64 fixture product is placed using y^3 = y + 1, y^4 = y^2 + y.
    let expected = [
        [
            [0x2c81_ccfb_ec6f_3649, 0, 0],
            [0, 0x2c81_ccfb_ec6f_3649, 0],
            [0, 0, 0x2c81_ccfb_ec6f_3649],
        ],
        [
            [0, 0x2c81_ccfb_ec6f_3649, 0],
            [0, 0, 0x2c81_ccfb_ec6f_3649],
            [0x2c81_ccfb_ec6f_3649, 0x2c81_ccfb_ec6f_3649, 0],
        ],
        [
            [0, 0, 0x2c81_ccfb_ec6f_3649],
            [0x2c81_ccfb_ec6f_3649, 0x2c81_ccfb_ec6f_3649, 0],
            [0, 0x2c81_ccfb_ec6f_3649, 0x2c81_ccfb_ec6f_3649],
        ],
    ];
    for (i, row) in expected.into_iter().enumerate() {
        for (j, product) in row.into_iter().enumerate() {
            let mut a = [0; 3];
            let mut b = [0; 3];
            a[i] = 0x9cc2_a2f3_6303_dc3a;
            b[j] = 0x2711_8a65_14b1_9502;
            let mut accumulator = <F192 as WithAccumulator>::Accumulator::default();
            accumulator.fmadd(from_coefficients(a), from_coefficients(b));
            assert_eq!(
                accumulator.reduce(),
                from_coefficients(product),
                "positions {i}, {j}"
            );
        }
    }
}
