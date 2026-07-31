//! Differential tests: jolt-field-two's BN254 backend against jolt-field.
//!
//! jolt-field (the crate being rebuilt) is the oracle: every operation,
//! serde byte stream, and transcript byte stream must match exactly.

#![cfg(feature = "bn254")]
#![expect(clippy::unwrap_used, reason = "test code")]

use jolt_field as base;
use jolt_field_two as two;

use base::{
    Accumulator as _, CanonicalBytes, CanonicalRepr, FieldCore, FromPrimitiveInt, RingCore,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use two::{Accumulator as _, CanonicalBytes as _, CanonicalEncoding, Field as _, Ring};

fn rng() -> ChaCha20Rng {
    ChaCha20Rng::seed_from_u64(0xb254_b254)
}

/// Sample a matched (baseline, rebuilt) element pair from the same bytes.
fn sample_pair(rng: &mut ChaCha20Rng) -> (base::Fr, two::Fr) {
    let bytes: [u8; 32] = rng.gen();
    let b = <base::Fr as CanonicalRepr>::from_le_bytes_mod_order(&bytes);
    let t = <two::Fr as CanonicalEncoding>::from_bytes_le_reduced(&bytes);
    assert_matches(t, b);
    (b, t)
}

/// Byte-level equality between the two crates' elements.
fn assert_matches(ours: two::Fr, theirs: base::Fr) {
    assert_eq!(ours.to_bytes_le_vec(), theirs.to_bytes_le_vec());
}

fn assert_matches_fq(ours: two::Fq, theirs: base::Fq) {
    assert_eq!(ours.to_bytes_le_vec(), theirs.to_bytes_le_vec());
}

#[test]
fn arithmetic_matches() {
    let mut rng = rng();
    for _ in 0..500 {
        let (b1, t1) = sample_pair(&mut rng);
        let (b2, t2) = sample_pair(&mut rng);
        assert_matches(t1 + t2, b1 + b2);
        assert_matches(t1 - t2, b1 - b2);
        assert_matches(t1 * t2, b1 * b2);
        assert_matches(-t1, -b1);
        assert_matches(Ring::square(&t1), RingCore::square(&b1));
        let (ti, bi) = (t1.inverse(), b1.inverse());
        assert_eq!(ti.is_some(), bi.is_some(), "inverse disagreement");
        if let (Some(ti), Some(bi)) = (ti, bi) {
            assert_matches(ti, bi);
        }
        if t2 != two::Zero::zero() {
            assert_matches(t1 / t2, b1 / b2);
        }
    }
}

#[test]
fn integer_conversions_match() {
    let mut rng = rng();
    let check = |v_u64: u64, v_i64: i64, v_u128: u128, v_i128: i128| {
        assert_matches(two::Fr::from_u64(v_u64), FromPrimitiveInt::from_u64(v_u64));
        assert_matches(two::Fr::from_i64(v_i64), FromPrimitiveInt::from_i64(v_i64));
        assert_matches(
            two::Fr::from_u128(v_u128),
            FromPrimitiveInt::from_u128(v_u128),
        );
        assert_matches(
            two::Fr::from_i128(v_i128),
            FromPrimitiveInt::from_i128(v_i128),
        );
    };
    // Boundary values, including both sides of the Montgomery precomp table.
    for v in [0u64, 1, 2, 16383, 16384, 16385, u64::MAX] {
        check(v, v as i64, v as u128, v as i128);
    }
    check(0, -1, u128::MAX, i128::MIN);
    check(0, i64::MIN, 1 << 127, -(1 << 100));
    for _ in 0..300 {
        check(rng.gen(), rng.gen(), rng.gen(), rng.gen());
    }
    assert_matches(two::Fr::from_bool(true), FromPrimitiveInt::from_bool(true));
}

#[test]
fn scalar_mul_fast_paths_match() {
    let mut rng = rng();
    for _ in 0..300 {
        let (b, t) = sample_pair(&mut rng);
        let s64: u64 = rng.gen();
        let s128: u128 = rng.gen();
        let si64: i64 = rng.gen();
        let si128: i128 = rng.gen();
        assert_matches(t.mul_u64(s64), b.mul_u64(s64));
        assert_matches(t.mul_i64(si64), b.mul_i64(si64));
        assert_matches(t.mul_u128(s128), b.mul_u128(s128));
        assert_matches(t.mul_i128(si128), b.mul_i128(si128));
        // Low-limb-only u128 exercises the single-round Barrett path.
        assert_matches(t.mul_u128(s64 as u128), b.mul_u128(s64 as u128));
        for edge in [0u64, 1, 2] {
            assert_matches(t.mul_u64(edge), b.mul_u64(edge));
        }
    }
}

#[test]
fn serde_bytes_match() {
    let mut rng = rng();
    let cfg = bincode::config::standard();
    for _ in 0..100 {
        let (b, t) = sample_pair(&mut rng);
        let b_bytes = bincode::serde::encode_to_vec(b, cfg).unwrap();
        let t_bytes = bincode::serde::encode_to_vec(t, cfg).unwrap();
        assert_eq!(b_bytes, t_bytes, "wire bytes diverge");
        // Cross-decode: each crate accepts the other's encoding.
        let (b_back, _): (base::Fr, usize) =
            bincode::serde::decode_from_slice(&t_bytes, cfg).unwrap();
        let (t_back, _): (two::Fr, usize) =
            bincode::serde::decode_from_slice(&b_bytes, cfg).unwrap();
        assert_matches(t_back, b_back);
    }
    // Non-canonical wire bytes rejected by both.
    let bad = bincode::serde::encode_to_vec([0xffu8; 32], cfg).unwrap();
    assert!(bincode::serde::decode_from_slice::<base::Fr, _>(&bad, cfg).is_err());
    assert!(bincode::serde::decode_from_slice::<two::Fr, _>(&bad, cfg).is_err());
}

#[test]
fn transcript_surface_matches() {
    let mut rng = rng();
    for _ in 0..200 {
        let (b, t) = sample_pair(&mut rng);
        assert_eq!(CanonicalEncoding::num_bits(&t), CanonicalRepr::num_bits(&b));
        assert_eq!(t.to_u64_checked(), b.to_canonical_u64_checked());

        let challenge: [u8; 16] = rng.gen();
        assert_matches(
            <two::Fr as CanonicalEncoding>::from_challenge_bytes(&challenge),
            <base::Fr as CanonicalRepr>::from_challenge_bytes(&challenge),
        );
        let digest: [u8; 32] = rng.gen();
        assert_matches(
            <two::Fr as CanonicalEncoding>::from_scalar_challenge_bytes(&digest),
            <base::Fr as CanonicalRepr>::from_scalar_challenge_bytes(&digest),
        );
        let wide: [u8; 48] = std::array::from_fn(|_| rng.gen());
        assert_matches(
            <two::Fr as CanonicalEncoding>::from_bytes_le_reduced(&wide),
            <base::Fr as CanonicalRepr>::from_le_bytes_mod_order(&wide),
        );
    }
    // Small-value integer views agree with construction.
    for v in [0u64, 1, 999, u64::MAX] {
        assert_eq!(two::Fr::from_u64(v).to_u64_checked(), Some(v));
        assert_eq!(two::Fr::from_u64(v).to_u128_checked(), Some(v as u128));
    }
    assert_eq!(
        two::Fr::from_u128(u128::MAX).to_u128_checked(),
        Some(u128::MAX)
    );
    assert_eq!(two::Fr::from_u128(u128::MAX).to_u64_checked(), None);
}

#[test]
fn wide_accumulator_matches() {
    let mut rng = rng();
    let mut base_acc = <base::Fr as base::WithAccumulator>::Accumulator::default();
    let mut two_acc = <two::Fr as two::WithAccumulator>::Accumulator::default();
    for _ in 0..1000 {
        let (b1, t1) = sample_pair(&mut rng);
        let (b2, t2) = sample_pair(&mut rng);
        base_acc.fmadd(b1, b2);
        two_acc.fmadd(t1, t2);
    }
    assert_matches(two_acc.reduce(), base_acc.reduce());

    // add / small-scalar fmadds / merge parity.
    let (b, t) = sample_pair(&mut rng);
    let mut base_acc = <base::Fr as base::WithAccumulator>::Accumulator::default();
    let mut two_acc = <two::Fr as two::WithAccumulator>::Accumulator::default();
    base_acc.add(b);
    two_acc.add(t);
    base_acc.fmadd_u8(b, 200);
    two_acc.fmadd_u8(t, 200);
    base_acc.fmadd_u64(b, u64::MAX);
    two_acc.fmadd_u64(t, u64::MAX);
    base_acc.fmadd_i64(b, -12345);
    two_acc.fmadd_i64(t, -12345);
    base_acc.fmadd_bool(b, true);
    two_acc.fmadd_bool(t, true);

    let mut base_other = <base::Fr as base::WithAccumulator>::Accumulator::default();
    let mut two_other = <two::Fr as two::WithAccumulator>::Accumulator::default();
    base_other.fmadd(b, b);
    two_other.fmadd(t, t);
    base_acc.merge(base_other);
    two_acc.merge(two_other);
    assert_matches(two_acc.reduce(), base_acc.reduce());

    // Empty accumulators reduce to zero.
    let empty = <two::Fr as two::WithAccumulator>::Accumulator::default();
    assert_eq!(empty.reduce(), two::Fr::from_u64(0));
}

#[test]
fn fq_matches() {
    let mut rng = rng();
    for _ in 0..300 {
        let bytes: [u8; 32] = rng.gen();
        let b1 = <base::Fq as CanonicalRepr>::from_le_bytes_mod_order(&bytes);
        let t1 = <two::Fq as CanonicalEncoding>::from_bytes_le_reduced(&bytes);
        assert_matches_fq(t1, b1);
        let bytes2: [u8; 32] = rng.gen();
        let b2 = <base::Fq as CanonicalRepr>::from_le_bytes_mod_order(&bytes2);
        let t2 = <two::Fq as CanonicalEncoding>::from_bytes_le_reduced(&bytes2);

        assert_matches_fq(t1 + t2, b1 + b2);
        assert_matches_fq(t1 * t2, b1 * b2);
        assert_matches_fq(-t1, -b1);
        if let (Some(ti), Some(bi)) = (t1.inverse(), b1.inverse()) {
            assert_matches_fq(ti, bi);
        }

        let v: u64 = rng.gen();
        assert_matches_fq(two::Fq::from_u64(v), FromPrimitiveInt::from_u64(v));

        let challenge: [u8; 16] = rng.gen();
        assert_matches_fq(
            <two::Fq as CanonicalEncoding>::from_challenge_bytes(&challenge),
            <base::Fq as CanonicalRepr>::from_challenge_bytes(&challenge),
        );
        assert_matches_fq(
            <two::Fq as CanonicalEncoding>::from_scalar_challenge_bytes(&challenge),
            <base::Fq as CanonicalRepr>::from_scalar_challenge_bytes(&challenge),
        );

        let cfg = bincode::config::standard();
        assert_eq!(
            bincode::serde::encode_to_vec(t1, cfg).unwrap(),
            bincode::serde::encode_to_vec(b1, cfg).unwrap(),
        );
    }
}

fn inner_product<F: two::JoltField>(xs: &[F], ys: &[F]) -> F {
    let mut acc = F::Accumulator::default();
    for (&x, &y) in xs.iter().zip(ys) {
        acc.fmadd(x, y);
    }
    acc.reduce()
}

#[test]
fn jolt_field_blanket_covers_bn254() {
    let xs = [two::Fr::from_u64(2), two::Fr::from_u64(3)];
    let ys = [two::Fr::from_u64(5), two::Fr::from_u64(7)];
    assert_eq!(inner_product(&xs, &ys), two::Fr::from_u64(31));
    let xq = [two::Fq::from_u64(2)];
    let yq = [two::Fq::from_u64(5)];
    assert_eq!(inner_product(&xq, &yq), two::Fq::from_u64(10));
}
