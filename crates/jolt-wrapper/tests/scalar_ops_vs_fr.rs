//! Cross-validation of `scalar_ops` against `jolt_field::Fr`.
//!
//! Every `scalar_ops` function must produce the same result as the corresponding
//! `Fr` operation. Since `Fr` uses Montgomery form internally, we compare via
//! `to_bytes()` (canonical LE serialization) which is representation-agnostic.

use jolt_field::{Field, Fr};
use jolt_wrapper::scalar_ops;

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_core::RngCore;

/// Convert `Fr` to `[u64; 4]` via canonical LE bytes.
fn fr_to_limbs(fr: Fr) -> [u64; 4] {
    let bytes = fr.to_bytes();
    scalar_ops::from_bytes_le(&bytes)
}

/// Convert `[u64; 4]` to `Fr` via canonical LE bytes.
fn limbs_to_fr(limbs: [u64; 4]) -> Fr {
    let bytes = scalar_ops::to_bytes_le(limbs);
    Fr::from_bytes(&bytes)
}

/// Generate a random field element via Fr, return both representations.
fn random_pair(rng: &mut impl RngCore) -> (Fr, [u64; 4]) {
    let fr = Fr::random(rng);
    let limbs = fr_to_limbs(fr);
    (fr, limbs)
}

const N_TRIALS: usize = 200;

#[test]
fn addition_matches_fr() {
    let mut rng = StdRng::seed_from_u64(0xADD);
    for _ in 0..N_TRIALS {
        let (a_fr, a_limbs) = random_pair(&mut rng);
        let (b_fr, b_limbs) = random_pair(&mut rng);

        let sum_fr = a_fr + b_fr;
        let sum_limbs = scalar_ops::add(a_limbs, b_limbs);

        assert_eq!(
            fr_to_limbs(sum_fr),
            sum_limbs,
            "add mismatch: {a_limbs:?} + {b_limbs:?}"
        );
    }
}

#[test]
fn subtraction_matches_fr() {
    let mut rng = StdRng::seed_from_u64(0x50B);
    for _ in 0..N_TRIALS {
        let (a_fr, a_limbs) = random_pair(&mut rng);
        let (b_fr, b_limbs) = random_pair(&mut rng);

        let diff_fr = a_fr - b_fr;
        let diff_limbs = scalar_ops::sub(a_limbs, b_limbs);

        assert_eq!(
            fr_to_limbs(diff_fr),
            diff_limbs,
            "sub mismatch: {a_limbs:?} - {b_limbs:?}"
        );
    }
}

#[test]
fn multiplication_matches_fr() {
    let mut rng = StdRng::seed_from_u64(0x1401);
    for _ in 0..N_TRIALS {
        let (a_fr, a_limbs) = random_pair(&mut rng);
        let (b_fr, b_limbs) = random_pair(&mut rng);

        let prod_fr = a_fr * b_fr;
        let prod_limbs = scalar_ops::mul(a_limbs, b_limbs);

        assert_eq!(
            fr_to_limbs(prod_fr),
            prod_limbs,
            "mul mismatch: {a_limbs:?} * {b_limbs:?}"
        );
    }
}

#[test]
fn negation_matches_fr() {
    let mut rng = StdRng::seed_from_u64(0x4E6);
    for _ in 0..N_TRIALS {
        let (a_fr, a_limbs) = random_pair(&mut rng);

        let neg_fr = -a_fr;
        let neg_limbs = scalar_ops::neg(a_limbs);

        assert_eq!(
            fr_to_limbs(neg_fr),
            neg_limbs,
            "neg mismatch for {a_limbs:?}"
        );
    }
}

#[test]
fn inverse_matches_fr() {
    let mut rng = StdRng::seed_from_u64(0x14F);
    for _ in 0..N_TRIALS {
        let (a_fr, a_limbs) = random_pair(&mut rng);

        let inv_fr = a_fr.inverse();
        let inv_limbs = scalar_ops::inv(a_limbs);

        match (inv_fr, inv_limbs) {
            (Some(fr_val), Some(limbs_val)) => {
                assert_eq!(
                    fr_to_limbs(fr_val),
                    limbs_val,
                    "inv mismatch for {a_limbs:?}"
                );
            }
            (None, None) => {} // both zero — correct
            _ => panic!("inv existence mismatch for {a_limbs:?}"),
        }
    }
}

#[test]
fn division_matches_fr() {
    let mut rng = StdRng::seed_from_u64(0xD17);
    for _ in 0..N_TRIALS {
        let (a_fr, a_limbs) = random_pair(&mut rng);
        let (b_fr, b_limbs) = random_pair(&mut rng);

        if b_fr == Fr::from_u64(0) {
            continue;
        }

        let div_fr = a_fr / b_fr;
        let div_limbs = scalar_ops::div(a_limbs, b_limbs).unwrap();

        assert_eq!(
            fr_to_limbs(div_fr),
            div_limbs,
            "div mismatch: {a_limbs:?} / {b_limbs:?}"
        );
    }
}

#[test]
fn square_matches_fr() {
    let mut rng = StdRng::seed_from_u64(0x5D2);
    for _ in 0..N_TRIALS {
        let (a_fr, a_limbs) = random_pair(&mut rng);

        let sq_fr = a_fr.square();
        let sq_limbs = scalar_ops::square(a_limbs);

        assert_eq!(
            fr_to_limbs(sq_fr),
            sq_limbs,
            "square mismatch for {a_limbs:?}"
        );
    }
}

#[test]
fn from_u64_matches_fr() {
    let test_values: Vec<u64> = vec![
        0,
        1,
        2,
        42,
        255,
        256,
        65535,
        u32::MAX as u64,
        u64::MAX,
        u64::MAX - 1,
        0xDEAD_BEEF,
        0xCAFE_BABE_1234_5678,
    ];
    for n in test_values {
        let fr = Fr::from_u64(n);
        let limbs = scalar_ops::from_u64(n);
        assert_eq!(fr_to_limbs(fr), limbs, "from_u64 mismatch for {n}");
    }
}

#[test]
fn from_i64_matches_fr() {
    let test_values: Vec<i64> = vec![
        0,
        1,
        -1,
        42,
        -42,
        i64::MAX,
        i64::MIN,
        i64::MIN + 1,
        -1000,
        1000,
    ];
    for n in test_values {
        let fr = Fr::from_i64(n);
        let limbs = scalar_ops::from_i64(n);
        assert_eq!(fr_to_limbs(fr), limbs, "from_i64 mismatch for {n}");
    }
}

#[test]
fn from_u128_matches_fr() {
    let test_values: Vec<u128> = vec![
        0,
        1,
        u64::MAX as u128,
        u64::MAX as u128 + 1,
        u128::MAX,
        u128::MAX - 1,
        (1u128 << 127) + 42,
        (1u128 << 64) + 1,
    ];
    for n in test_values {
        let fr = Fr::from_u128(n);
        let limbs = scalar_ops::from_u128(n);
        assert_eq!(fr_to_limbs(fr), limbs, "from_u128 mismatch for {n}");
    }
}

#[test]
fn from_i128_matches_fr() {
    let test_values: Vec<i128> = vec![
        0,
        1,
        -1,
        i128::MAX,
        i128::MIN,
        i128::MIN + 1,
        i64::MAX as i128,
        i64::MIN as i128,
        -(1i128 << 100),
        (1i128 << 100),
    ];
    for n in test_values {
        let fr = Fr::from_i128(n);
        let limbs = scalar_ops::from_i128(n);
        assert_eq!(fr_to_limbs(fr), limbs, "from_i128 mismatch for {n}");
    }
}

#[test]
fn bytes_roundtrip_matches_fr() {
    let mut rng = StdRng::seed_from_u64(0xB17E);
    for _ in 0..N_TRIALS {
        let (fr, limbs) = random_pair(&mut rng);

        // Fr.to_bytes() == scalar_ops::to_bytes_le(limbs)
        let fr_bytes = fr.to_bytes();
        let limbs_bytes = scalar_ops::to_bytes_le(limbs);
        assert_eq!(
            fr_bytes, limbs_bytes,
            "to_bytes mismatch for {limbs:?}"
        );

        // Roundtrip: from_bytes(to_bytes(x)) == x
        let fr_restored = Fr::from_bytes(&fr_bytes);
        let limbs_restored = scalar_ops::from_bytes_le(&limbs_bytes);
        assert_eq!(fr_to_limbs(fr_restored), limbs_restored);
    }
}

#[test]
fn num_bits_matches_fr() {
    let test_values: Vec<u64> = vec![0, 1, 2, 255, 256, u32::MAX as u64, u64::MAX];
    for n in test_values {
        let fr = Fr::from_u64(n);
        let limbs = scalar_ops::from_u64(n);
        assert_eq!(
            fr.num_bits(),
            scalar_ops::num_bits(limbs),
            "num_bits mismatch for {n}"
        );
    }
}

#[test]
fn to_u64_matches_fr() {
    let test_values: Vec<u64> = vec![0, 1, 42, u32::MAX as u64, u64::MAX];
    for n in test_values {
        let fr = Fr::from_u64(n);
        let limbs = scalar_ops::from_u64(n);

        let fr_u64 = fr.to_u64();
        // scalar_ops stores canonical (non-Montgomery) form, so check directly
        let limbs_u64 = if limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0 {
            Some(limbs[0])
        } else {
            None
        };

        assert_eq!(fr_u64, limbs_u64, "to_u64 mismatch for from_u64({n})");
    }
}

#[test]
fn modulus_matches_bn254() {
    // The modulus in scalar_ops must match the BN254 scalar field order.
    // Verify by checking that p-1 + 1 == 0 in both systems.
    let p_minus_1_limbs = scalar_ops::sub(
        scalar_ops::from_bytes_le(&scalar_ops::to_bytes_le(scalar_ops::MODULUS)),
        scalar_ops::ONE,
    );
    let p_minus_1_fr = limbs_to_fr(p_minus_1_limbs);

    // p-1 + 1 should be 0
    let zero_fr = p_minus_1_fr + Fr::from_u64(1);
    assert!(zero_fr == Fr::from_u64(0), "modulus doesn't match BN254");

    // Also check that p == 0 in the field
    let p_as_fr = limbs_to_fr(scalar_ops::MODULUS);
    assert!(
        p_as_fr == Fr::from_u64(0),
        "MODULUS should be zero in the field"
    );
}

#[test]
fn symbolic_field_constant_folding_matches_fr() {
    use jolt_wrapper::arena::ArenaSession;
    use jolt_wrapper::symbolic::SymbolicField;

    let _session = ArenaSession::new();

    let mut rng = StdRng::seed_from_u64(0xF01D);
    for _ in 0..50 {
        let (a_fr, a_limbs) = random_pair(&mut rng);
        let (b_fr, b_limbs) = random_pair(&mut rng);

        let a_sym = SymbolicField::constant(a_limbs);
        let b_sym = SymbolicField::constant(b_limbs);

        // Addition
        let sum_sym = (a_sym + b_sym).as_constant().unwrap();
        let sum_fr = fr_to_limbs(a_fr + b_fr);
        assert_eq!(sum_sym, sum_fr, "SymbolicField add folding mismatch");

        // Subtraction
        let diff_sym = (a_sym - b_sym).as_constant().unwrap();
        let diff_fr = fr_to_limbs(a_fr - b_fr);
        assert_eq!(diff_sym, diff_fr, "SymbolicField sub folding mismatch");

        // Multiplication
        let prod_sym = (a_sym * b_sym).as_constant().unwrap();
        let prod_fr = fr_to_limbs(a_fr * b_fr);
        assert_eq!(prod_sym, prod_fr, "SymbolicField mul folding mismatch");

        // Negation
        let neg_sym = (-a_sym).as_constant().unwrap();
        let neg_fr = fr_to_limbs(-a_fr);
        assert_eq!(neg_sym, neg_fr, "SymbolicField neg folding mismatch");
    }
}
