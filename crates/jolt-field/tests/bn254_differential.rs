//! Differential tests: jolt-field's BN254 backend against exact
//! num-bigint modular arithmetic. The canonical value of an element is read
//! through `to_bytes_le`, whose faithfulness is pinned by the golden
//! fixtures in golden_bytes.rs.

#![cfg(feature = "bn254")]
#![expect(clippy::unwrap_used, reason = "test code")]

use jolt_field as two;

use num_bigint::{BigInt, BigUint, Sign};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use two::{Accumulator as _, CanonicalBytes as _, CanonicalEncoding, Field as _, Ring};

fn rng() -> ChaCha20Rng {
    ChaCha20Rng::seed_from_u64(0xb254_b254)
}

/// BN254 scalar-field modulus r.
fn p_fr() -> BigUint {
    BigUint::parse_bytes(
        b"30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001",
        16,
    )
    .unwrap()
}

/// BN254 base-field modulus q.
fn p_fq() -> BigUint {
    BigUint::parse_bytes(
        b"30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47",
        16,
    )
    .unwrap()
}

/// Canonical value of an element via its (fixture-pinned) LE encoding.
fn val<F: CanonicalEncoding>(x: &F) -> BigUint {
    BigUint::from_bytes_le(&x.to_bytes_le_vec())
}

fn assert_val<F: CanonicalEncoding>(x: &F, expected: &BigUint) {
    assert_eq!(val(x), *expected);
}

/// `v mod p` for a possibly negative BigInt.
fn imod(v: &BigInt, p: &BigUint) -> BigUint {
    let p_int = BigInt::from_biguint(Sign::Plus, p.clone());
    let r = ((v % &p_int) + &p_int) % &p_int;
    r.to_biguint().unwrap()
}

/// Sample an element together with its oracle value from the same bytes.
fn sample_fr(rng: &mut ChaCha20Rng, p: &BigUint) -> (two::Fr, BigUint) {
    let bytes: [u8; 32] = rng.gen();
    let t = <two::Fr as CanonicalEncoding>::from_bytes_le_reduced(&bytes);
    let v = BigUint::from_bytes_le(&bytes) % p;
    assert_val(&t, &v);
    (t, v)
}

#[test]
fn moduli_are_consistent() {
    // The hardcoded moduli agree with the crate: -1 encodes p − 1, and the
    // reducing decode sends p to zero.
    for (minus_one_bytes, p) in [
        (two::Fr::from_i64(-1).to_bytes_le_vec(), p_fr()),
        (two::Fq::from_i64(-1).to_bytes_le_vec(), p_fq()),
    ] {
        assert_eq!(BigUint::from_bytes_le(&minus_one_bytes) + 1u32, p);
    }
}

#[test]
fn arithmetic_matches() {
    let p = p_fr();
    let mut rng = rng();
    for _ in 0..500 {
        let (t1, v1) = sample_fr(&mut rng, &p);
        let (t2, v2) = sample_fr(&mut rng, &p);
        assert_val(&(t1 + t2), &((&v1 + &v2) % &p));
        assert_val(&(t1 - t2), &((&v1 + &p - &v2) % &p));
        assert_val(&(t1 * t2), &(&v1 * &v2 % &p));
        assert_val(&(-t1), &((&p - &v1) % &p));
        assert_val(&Ring::square(&t1), &(&v1 * &v1 % &p));
        match t1.inverse() {
            Some(ti) => {
                assert_val(&ti, &v1.modpow(&(&p - 2u32), &p));
                assert_eq!(ti * t1, two::Fr::from_u64(1));
            }
            None => assert_eq!(v1, BigUint::ZERO, "inverse must exist for nonzero"),
        }
        if v2 != BigUint::ZERO {
            let inv2 = v2.modpow(&(&p - 2u32), &p);
            assert_val(&(t1 / t2), &(&v1 * inv2 % &p));
        }
    }
}

#[test]
fn integer_conversions_match() {
    let p = p_fr();
    let mut rng = rng();
    let check = |v_u64: u64, v_i64: i64, v_u128: u128, v_i128: i128| {
        assert_val(&two::Fr::from_u64(v_u64), &(BigUint::from(v_u64) % &p));
        assert_val(&two::Fr::from_i64(v_i64), &imod(&BigInt::from(v_i64), &p));
        assert_val(&two::Fr::from_u128(v_u128), &(BigUint::from(v_u128) % &p));
        assert_val(
            &two::Fr::from_i128(v_i128),
            &imod(&BigInt::from(v_i128), &p),
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
    assert_val(&two::Fr::from_bool(true), &BigUint::from(1u32));
}

#[test]
fn scalar_mul_fast_paths_match() {
    let p = p_fr();
    let mut rng = rng();
    for _ in 0..300 {
        let (t, v) = sample_fr(&mut rng, &p);
        let s64: u64 = rng.gen();
        let s128: u128 = rng.gen();
        let si64: i64 = rng.gen();
        let si128: i128 = rng.gen();
        assert_val(&t.mul_u64(s64), &(&v * s64 % &p));
        assert_val(
            &t.mul_i64(si64),
            &imod(&(BigInt::from_biguint(Sign::Plus, v.clone()) * si64), &p),
        );
        assert_val(&t.mul_u128(s128), &(&v * s128 % &p));
        assert_val(
            &t.mul_i128(si128),
            &imod(&(BigInt::from_biguint(Sign::Plus, v.clone()) * si128), &p),
        );
        // Low-limb-only u128 exercises the single-round Barrett path.
        assert_val(&t.mul_u128(s64 as u128), &(&v * s64 % &p));
        for edge in [0u64, 1, 2] {
            assert_val(&t.mul_u64(edge), &(&v * edge % &p));
        }
    }
}

#[test]
fn serde_bytes_match() {
    let p = p_fr();
    let mut rng = rng();
    let cfg = bincode::config::standard();
    for _ in 0..100 {
        let (t, v) = sample_fr(&mut rng, &p);
        let t_bytes = bincode::serde::encode_to_vec(t, cfg).unwrap();
        // Wire format is the canonical 32-byte LE encoding (absolute bytes
        // pinned by the golden fixtures).
        assert_eq!(t_bytes, t.to_bytes_le_vec(), "wire = transcript bytes");
        let (t_back, read): (two::Fr, usize) =
            bincode::serde::decode_from_slice(&t_bytes, cfg).unwrap();
        assert_eq!(read, 32);
        assert_val(&t_back, &v);
    }
    // Non-canonical wire bytes rejected.
    let bad = bincode::serde::encode_to_vec([0xffu8; 32], cfg).unwrap();
    assert!(bincode::serde::decode_from_slice::<two::Fr, _>(&bad, cfg).is_err());
}

/// The legacy 125-bit shifted challenge for Fr: the masked value is placed
/// in the two HIGH limbs of a raw Montgomery representation, so the field
/// value is `(low·2^128 + high·2^192) · R⁻¹ mod r` with `R = 2^256 mod r`.
fn fr_challenge_model(bytes: &[u8], p: &BigUint) -> BigUint {
    let mut buf = [0u8; 16];
    let len = bytes.len().min(16);
    buf[..len].copy_from_slice(&bytes[..len]);
    let value = u128::from_le_bytes(buf);
    let low = BigUint::from(value as u64);
    let high = BigUint::from(((value >> 64) as u64) & (u64::MAX >> 3));
    let integer = (low << 128u32) + (high << 192u32);
    let r_pow = (BigUint::from(1u32) << 256u32) % p;
    let r_inv = r_pow.modpow(&(p - 2u32), p);
    integer * r_inv % p
}

#[test]
fn transcript_surface_matches() {
    let p = p_fr();
    let mut rng = rng();
    for _ in 0..200 {
        let (t, v) = sample_fr(&mut rng, &p);
        assert_eq!(CanonicalEncoding::num_bits(&t) as u64, v.bits());
        let expected_u64 =
            (v.bits() <= 64).then(|| v.to_u64_digits().first().copied().unwrap_or(0));
        assert_eq!(t.to_u64_checked(), expected_u64);

        let challenge: [u8; 16] = rng.gen();
        assert_val(
            &<two::Fr as CanonicalEncoding>::from_challenge_bytes(&challenge),
            &fr_challenge_model(&challenge, &p),
        );
        let digest: [u8; 32] = rng.gen();
        assert_val(
            &<two::Fr as CanonicalEncoding>::from_scalar_challenge_bytes(&digest),
            &(BigUint::from_bytes_be(&digest) % &p),
        );
        let wide: [u8; 48] = std::array::from_fn(|_| rng.gen());
        assert_val(
            &<two::Fr as CanonicalEncoding>::from_bytes_le_reduced(&wide),
            &(BigUint::from_bytes_le(&wide) % &p),
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
    let p = p_fr();
    let mut rng = rng();
    let mut acc = <two::Fr as two::WithAccumulator>::Accumulator::default();
    let mut expect = BigUint::ZERO;
    for _ in 0..1000 {
        let (t1, v1) = sample_fr(&mut rng, &p);
        let (t2, v2) = sample_fr(&mut rng, &p);
        acc.fmadd(t1, t2);
        expect = (expect + v1 * v2) % &p;
    }
    assert_val(&acc.reduce(), &expect);

    // add / small-scalar fmadds / merge, mirrored in exact integers.
    let (t, v) = sample_fr(&mut rng, &p);
    let vi = BigInt::from_biguint(Sign::Plus, v.clone());
    let mut acc = <two::Fr as two::WithAccumulator>::Accumulator::default();
    let mut expect = BigInt::ZERO;
    acc.add(t);
    expect += &vi;
    acc.fmadd_u8(t, 200);
    expect += &vi * 200;
    acc.fmadd_u64(t, u64::MAX);
    expect += &vi * u64::MAX;
    acc.fmadd_i64(t, -12345);
    expect += &vi * -12345i64;
    acc.fmadd_bool(t, true);
    expect += &vi;

    let mut other = <two::Fr as two::WithAccumulator>::Accumulator::default();
    other.fmadd(t, t);
    expect += &vi * &vi;
    acc.merge(other);
    assert_val(&acc.reduce(), &imod(&expect, &p));

    // Empty accumulators reduce to zero.
    let empty = <two::Fr as two::WithAccumulator>::Accumulator::default();
    assert_eq!(empty.reduce(), two::Fr::from_u64(0));
}

/// The legacy challenge for Fq places the masked value in the high limbs of
/// a checked (non-Montgomery) bigint, so the field value is the integer
/// itself: `low·2^128 + high·2^192 < q`.
fn fq_challenge_model(bytes: &[u8]) -> BigUint {
    let mut buf = [0u8; 16];
    let len = bytes.len().min(16);
    buf[..len].copy_from_slice(&bytes[..len]);
    let value = u128::from_le_bytes(buf);
    let low = BigUint::from(value as u64);
    let high = BigUint::from(((value >> 64) as u64) & (u64::MAX >> 3));
    (low << 128u32) + (high << 192u32)
}

#[test]
fn fq_matches() {
    let p = p_fq();
    let mut rng = rng();
    let cfg = bincode::config::standard();
    for _ in 0..300 {
        let bytes: [u8; 32] = rng.gen();
        let t1 = <two::Fq as CanonicalEncoding>::from_bytes_le_reduced(&bytes);
        let v1 = BigUint::from_bytes_le(&bytes) % &p;
        assert_val(&t1, &v1);
        let bytes2: [u8; 32] = rng.gen();
        let t2 = <two::Fq as CanonicalEncoding>::from_bytes_le_reduced(&bytes2);
        let v2 = BigUint::from_bytes_le(&bytes2) % &p;

        assert_val(&(t1 + t2), &((&v1 + &v2) % &p));
        assert_val(&(t1 * t2), &(&v1 * &v2 % &p));
        assert_val(&(-t1), &((&p - &v1) % &p));
        match t1.inverse() {
            Some(ti) => assert_val(&ti, &v1.modpow(&(&p - 2u32), &p)),
            None => assert_eq!(v1, BigUint::ZERO),
        }

        let v: u64 = rng.gen();
        assert_val(&two::Fq::from_u64(v), &BigUint::from(v));

        let challenge: [u8; 16] = rng.gen();
        assert_val(
            &<two::Fq as CanonicalEncoding>::from_challenge_bytes(&challenge),
            &fq_challenge_model(&challenge),
        );
        assert_val(
            &<two::Fq as CanonicalEncoding>::from_scalar_challenge_bytes(&challenge),
            &(BigUint::from_bytes_be(&challenge) % &p),
        );

        assert_eq!(
            bincode::serde::encode_to_vec(t1, cfg).unwrap(),
            t1.to_bytes_le_vec(),
            "wire = transcript bytes"
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
